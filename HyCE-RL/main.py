import chex
from craftax.craftax_env import make_craftax_env_from_name
import distrax
from flax import struct
import functools
import jax
import jax.numpy as jnp
import optax
from typing import Any, List, Tuple

from lib.networks import SkillSelector, SkillPolicy, SkillCritic, Embedder
from lib.states import TrainState, AgentState, SkillTrainState
from lib.env_wrappers import BatchEnvWrapper, GymnaxWrapper


class Config:
    population_size: int = 100
    num_skills: int = 16
    env_name: str = "Craftax-Symbolic-v1"
    seed: int = 42

    # Embedder Hyperparameters
    embedder_hidden_dim: int = 512
    embedder_embedding_dim: int = 256
    embedder_lr: float = 1e-3

    # Skill Hyperparameters
    skill_policy_hidden_dim: int = 512
    skill_policy_lr: float = 1e-3
    skill_critic_hidden_dim: int = 512
    skill_critic_lr: float = 1e-3

    # Selector Hyperparameters
    selector_hidden_dim: int = 512
    min_skills_per_agent: int = 4

    # EA Hyperparameters
    num_elites: int = 10
    selector_param_mutation_std: float = 0.01
    skill_mask_mutation_rate: float = 0.05


@struct.dataclass
class TrajectoryData:
    """Stores trajectory data from one evaluation phase."""
    observations: chex.Array # Shape [P, T, obs_shape...]
    actions: chex.Array # Shape [P, T]
    log_probs: chex.Array # Shape [P, T] (Log prob of action taken)
    rewards: chex.Array # Shape [P, T]
    dones: chex.Array # Shape [P, T]
    values: chex.Array # Shape [P, T] (Value estimates from critic V(s))
    skill_indices: chex.Array # Shape [P, T]
    agent_indices: chex.Array # Shape [P, T]
    next_values: chex.Array # Shape [P, T] (Value estimates from critic V(s'))


def mutate_selector_params(params: Any, key: chex.PRNGKey, stddev: float) -> Any:
    """Adds Gaussian noise to the selector parameters."""
    mutated_params = jax.tree_util.tree_map(
        lambda p: p + jax.random.normal(key, p.shape, p.dtype) * stddev,
        params
    )
    # TODO(wmaxlees): This simple noise addition might not be ideal for all layers/params.
    # More sophisticated mutation might be needed for complex networks.
    return mutated_params

def mutate_skill_mask(mask: chex.Array, key: chex.PRNGKey, rate: float, min_skills: int) -> chex.Array:
    """Flips bits in the skill mask with a given probability, ensuring min_skills."""
    mutation_noise = jax.random.uniform(key, mask.shape)

    flipped_mask = jnp.where(mutation_noise < rate, ~mask, mask)

    num_available = jnp.sum(flipped_mask, axis=-1, keepdims=True)

    needs_more = num_available < min_skills

    final_mask = jnp.where(needs_more, mask, flipped_mask)

    # TODO: Implement a better way to add skills back if needed_more is True.

    return final_mask


def init_shared_skills(config: Config, key: chex.PRNGKey, action_dim: int) -> chex.Array:
    key, *skill_keys = jax.random.split(key, config.num_skills + 1)
    skill_policy_net = SkillPolicy(hidden_dim=config.skill_policy_hidden_dim, action_dim=action_dim)
    skill_critic_net = SkillCritic(hidden_dim=config.skill_critic_hidden_dim)
    policy_optimizer = optax.adam(learning_rate=config.skill_policy_lr)
    critic_optimizer = optax.adam(learning_rate=config.skill_critic_lr)

    shared_skill_states: List[SkillTrainState] = []
    dummy_embedding = jnp.zeros((1, config.embedder_embedding_dim))

    for i in range(config.num_skills):
        skill_key_i = skill_keys[i]
        key_policy, key_critic = jax.random.split(skill_key_i)

        # Initialize policy network and optimizer state
        policy_params = skill_policy_net.init(key_policy, dummy_embedding)['params']
        policy_opt_state = policy_optimizer.init(policy_params)

        # Initialize critic network and optimizer state
        critic_params = skill_critic_net.init(key_critic, dummy_embedding)['params']
        critic_opt_state = critic_optimizer.init(critic_params)

        # Create the state object for this skill
        skill_state = SkillTrainState(
            policy_params=policy_params,
            critic_params=critic_params,
            policy_opt_state=policy_opt_state,
            critic_opt_state=critic_opt_state
        )
        shared_skill_states.append(skill_state)

    return shared_skill_states


def init_selectors(config: Config, key: chex.PRNGKey) -> chex.Array:
    selector_net = SkillSelector(
        num_available_skills=config.num_skills,
        hidden_dim=config.selector_hidden_dim
    )

    dummy_embedding = jnp.zeros((1, config.embedder_embedding_dim))
    dummy_mask = jnp.ones((config.num_skills,), dtype=bool)

    pop_keys = jax.random.split(key, config.population_size)

    batched_params = jax.vmap(
        lambda k: selector_net.init(k, dummy_embedding, dummy_mask)['params']
    )(pop_keys)

    return batched_params


def init_skill_masks(config: Config, key: chex.PRNGKey) -> chex.Array:
    """Initializes the skill availability masks for the population."""
    # TODO(wmaxlees): This may select duplicates for a single agent. That's not good.
    chosen = jax.random.choice(key, jnp.arange(config.num_skills),
                               (config.population_size, config.min_skills_per_agent,), replace=True)
    
    one_hot_masks = jax.nn.one_hot(chosen, config.num_skills, axis=-1)
    int_masks = jnp.sum(one_hot_masks, axis=1)
    masks = int_masks > 0

    return masks



def init_train_state(config: Config, env: GymnaxWrapper) -> TrainState:
    """Initializes all components and returns the initial TrainState."""
    key = jax.random.PRNGKey(config.seed)

    # Init Environments
    key, env_key = jax.random.split(key)
    env_params = env.default_params
    obs, env_state = env.reset(env_key, env_params)

    obs_shape = obs.shape[1:]
    action_dim = env.action_space(env_params).n

    # Init the Embedder
    key, embed_key = jax.random.split(key)
    embedder_net = Embedder(hidden_dim=config.embedder_hidden_dim, embedding_dim=config.embedder_embedding_dim) # Add dims to Config
    dummy_obs_single = jnp.zeros((1, *obs_shape), dtype=obs.dtype)
    embedder_params = embedder_net.init(embed_key, dummy_obs_single)['params']
    embedder_optimizer = optax.adam(learning_rate=config.embedder_lr)
    embedder_opt_state = embedder_optimizer.init(embedder_params)

    # Init Shared Skills (Networks + Optimizers)
    shared_skill_states: List[SkillTrainState] = init_shared_skills(config, key, action_dim)

    # Init Agent States
    key, pop_key = jax.random.split(key)
    population_agent_states = AgentState(
        selector_params=init_selectors(config, pop_key),
        skill_subset_mask=init_skill_masks(config, pop_key)
    )

    return TrainState(
        embedder_params=embedder_params,
        embedder_opt_state=embedder_opt_state,
        population_agent_states=population_agent_states,
        shared_skill_states=shared_skill_states,
        env_state=env_state,
        obs=obs,
        population_fitness=jnp.zeros(config.population_size), # Initial fitness
        prng_key=key,
        timestep=0
    )


@functools.partial(jax.jit, static_argnames=("config", "env", "num_steps"))
def run_evaluation_phase(train_state: TrainState, config: Config, env: GymnaxWrapper, env_params: Any, num_steps: int) -> Tuple[TrainState, Any]:
    """
    Runs agents in the environment, collects data, calculates fitness.
    """
    print("Evaluating population and collecting trajectories...")

    action_dim = env.action_space(env_params).n
    embedder_net = Embedder(hidden_dim=config.embedder_hidden_dim, embedding_dim=config.embedder_embedding_dim)
    selector_net = SkillSelector(num_available_skills=config.num_skills, hidden_dim=config.selector_hidden_dim)

    skill_policy_template = SkillPolicy(hidden_dim=config.skill_policy_hidden_dim, action_dim=action_dim)
    skill_critic_template = SkillCritic(hidden_dim=config.skill_critic_hidden_dim)

    initial_carry = {
        "key": train_state.prng_key,
        "obs": train_state.obs,
        "env_state": train_state.env_state,
        "accumulated_reward": jnp.zeros(config.population_size, dtype=jnp.float32),
    }

    def step_env_scan_body(carry, _unused_step_idx):
        key, current_obs, current_env_state, accumulated_reward = \
            carry["key"], carry["obs"], carry["env_state"], carry["accumulated_reward"]

        agent_indices_step = jnp.arange(config.population_size)

        key, select_key, policy_key, step_key = jax.random.split(key, 4)

        embeddings = embedder_net.apply({'params': train_state.embedder_params}, current_obs) # Shape [P, embed_dim]

        agent_states = train_state.population_agent_states # Pytree: params[P,...], mask[P, N_skill]
        selector_logits = jax.vmap(
        selector_net.apply, in_axes=({'params': 0}, 0, 0)
            )(
                {'params': agent_states.selector_params}, embeddings, agent_states.skill_subset_mask
            ) # Shape [P, N_skill]
        skill_indices_k = jax.random.categorical(select_key, selector_logits) # Shape [P]

        stacked_policy_params = jax.tree_util.tree_map(
            lambda *x: jnp.stack(x), *[s.policy_params for s in train_state.shared_skill_states]
        )
        # Stack critic params: List[Pytree[Array]] -> Pytree[Array[N_skill, ...]]
        stacked_critic_params = jax.tree_util.tree_map(
            lambda *x: jnp.stack(x), *[s.critic_params for s in train_state.shared_skill_states]
        )

        def apply_policy_stacked(stacked_params, embeddings_batch):
            # Vmap the apply function over the N_skill dimension of params
            # in_axes: params pytree (0), embeddings (None - broadcast)
            return jax.vmap(
                skill_policy_template.apply, in_axes=({'params': 0}, None)
            )(
                {'params': stacked_params}, embeddings_batch
            ) # Output shape [N_skill, P, action_dim]

        def apply_critic_stacked(stacked_params, embeddings_batch):
            # Vmap the apply function over the N_skill dimension of params
            return jax.vmap(
                skill_critic_template.apply, in_axes=({'params': 0}, None)
            )(
                {'params': stacked_params}, embeddings_batch
            ) # Output shape [N_skill, P]

        all_policy_logits = apply_policy_stacked(stacked_policy_params, embeddings)
        all_values = apply_critic_stacked(stacked_critic_params, embeddings)

        batch_indices = jnp.arange(config.population_size)
        chosen_skill_policy_logits = all_policy_logits[skill_indices_k, batch_indices, :] # Shape [P, action_dim]
        chosen_skill_values = all_values[skill_indices_k, batch_indices] # Shape [P]

        # Create action distribution
        action_dist = distrax.Categorical(logits=chosen_skill_policy_logits)
        actions = action_dist.sample(seed=policy_key) # Shape [P]
        log_probs = action_dist.log_prob(actions) # Shape [P]

        # 4. Step environment
        next_obs, next_env_state, rewards, dones, _ = env.step(
            step_key, current_env_state, actions, env_params
        )

        # 5. Get value estimate for the *next* state (for GAE) using chosen skill's critic
        next_embeddings = embedder_net.apply({'params': train_state.embedder_params}, next_obs)

        # Apply all critics to next embeddings (Can reuse vmapped function)
        all_next_values = apply_critic_stacked(stacked_critic_params, next_embeddings) # Shape [N_skill, P]

        # Select based on the *same* skill k chosen for the action
        chosen_skill_next_values = all_next_values[skill_indices_k, batch_indices] # Shape [P]
        # Handle terminal states: value should be 0 if done
        chosen_skill_next_values = jnp.where(dones, 0.0, chosen_skill_next_values)


        # --- Store transition data for this step ---
        transition = {
            "observations": current_obs,
            "actions": actions,
            "log_probs": log_probs,
            "rewards": rewards,
            "dones": dones,
            "values": chosen_skill_values, # V(s_t)
            "skill_indices": skill_indices_k,
            "agent_indices": agent_indices_step,
            "next_values": chosen_skill_next_values # V(s_{t+1})
        }

        new_accumulated_reward = accumulated_reward + rewards
        next_carry = {
            "key": key,
            "obs": next_obs,
            "env_state": next_env_state,
            "accumulated_reward": new_accumulated_reward
        }
        return next_carry, transition

    # --- Run the scan over num_steps ---
    final_carry, collected_transitions_stacked = jax.lax.scan(
         step_env_scan_body, initial_carry, None, length=num_steps # Scan over T steps
    )
    # collected_transitions_stacked is a pytree where leaves have shape [T, P, ...]

    # Extract final state parts and accumulated rewards
    final_key = final_carry["key"]
    final_obs = final_carry["obs"]
    final_env_state = final_carry["env_state"]
    # Use accumulated reward over the trajectory as fitness
    final_fitness = final_carry["accumulated_reward"]

    # Transpose collected data to [P, T, ...]
    trajectory_data_transposed = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), collected_transitions_stacked)

    # Pack into TrajectoryData structure
    trajectory_data = TrajectoryData(**trajectory_data_transposed)

    # Update TrainState
    new_train_state = train_state._replace(
        prng_key=final_key,
        obs=final_obs, # Store the very last obs
        env_state=final_env_state, # Store the last env state
        population_fitness=final_fitness,
        timestep=train_state.timestep + num_steps * config.population_size
    )

    return new_train_state, trajectory_data


@functools.partial(jax.jit, static_argnames=('config'))
def run_evolution_phase(train_state: TrainState, config: Config) -> TrainState:
    """Applies EA selection and variation (elitism + mutation) to the population."""
    print("Evolving population...")
    key, key_select, key_mutate_selector, key_mutate_mask = jax.random.split(train_state.prng_key, 4)

    # --- Selection ---
    elite_indices = jnp.argsort(train_state.population_fitness)[-config.num_elites:]

    num_offspring = config.population_size - config.num_elites
    offspring_source_indices = jax.random.choice(key_select, elite_indices, shape=(num_offspring,))

    new_pop_source_indices = jnp.concatenate([elite_indices, offspring_source_indices])

    base_population_states = jax.tree_util.tree_map(
        lambda x: x[new_pop_source_indices],
        train_state.population_agent_states
    )

    # --- Variation (Mutation) ---
    selector_mutation_keys = jax.random.split(key_mutate_selector, num_offspring)
    mask_mutation_keys = jax.random.split(key_mutate_mask, num_offspring)

    offspring_base_params = jax.tree_util.tree_map(lambda x: x[config.num_elites:], base_population_states.selector_params)
    offspring_base_masks = base_population_states.skill_subset_mask[config.num_elites:]

    mutated_offspring_params = jax.vmap(
        mutate_selector_params, in_axes=(0, 0, None) # Map over params-pytree and keys
    )(offspring_base_params, selector_mutation_keys, config.selector_param_mutation_std)

    mutated_offspring_masks = jax.vmap(
        mutate_skill_mask, in_axes=(0, 0, None, None) # Map over mask and keys
    )(offspring_base_masks, mask_mutation_keys, config.skill_mask_mutation_rate, config.min_skills_per_agent)

    # --- Combine Elites and Mutated Offspring ---
    elite_params = jax.tree_util.tree_map(lambda x: x[:config.num_elites], base_population_states.selector_params)
    elite_masks = base_population_states.skill_subset_mask[:config.num_elites]

    new_selector_params = jax.tree_util.tree_map(
        lambda elite, offspring: jnp.concatenate([elite, offspring], axis=0),
        elite_params,
        mutated_offspring_params
    )
    new_skill_subset_mask = jnp.concatenate([elite_masks, mutated_offspring_masks], axis=0)

    new_population_agent_states = AgentState(
        selector_params=new_selector_params,
        skill_subset_mask=new_skill_subset_mask
    )

    return train_state._replace(
        population_agent_states=new_population_agent_states,
        prng_key=key
    )


@functools.partial(jax.jit, static_argnames=("config", "gbl_update_fn"))
def run_skill_refinement_phase(train_state: TrainState, config: Config, gbl_update_fn: callable) -> TrainState:
    """Filters data, samples batches, and updates shared skill modules via GBL."""
    # 1. Filter replay buffer data based on successful agents (using train_state.population_fitness).
    # 2. For each skill `k` (or in parallel using vmap if possible):
    #    a. Sample a batch of relevant filtered data where skill `k` was used.
    #    b. Call the GBL update function `gbl_update_fn` for skill `k`.
    #       This function takes the skill's current state (SkillTrainState[k]) and the batch,
    #       computes gradients using jax.grad, and uses optax to update params/opt_state.
    #    c. Update the shared_skill_states[k] with the new state returned by `gbl_update_fn`.
    # 3. Return train_state with updated shared_skill_states and split prng_key.

    print("Refining skills...") # Replace with actual implementation
    key, _ = jax.random.split(train_state.prng_key)

    # Placeholder: Just pass state through
    # new_shared_skill_states = update_skills(train_state, config, gbl_update_fn) # Placeholder

    return train_state._replace(
        # shared_skill_states=new_shared_skill_states, # Uncomment when implemented
        prng_key=key
    )


def main():
    config = Config() # Load config

    env = make_craftax_env_from_name(config.env_name, True)
    env = BatchEnvWrapper(env, num_envs=config.population_size)

    # Define GBL update function (e.g., PPO, SAC step)
    # gbl_update_fn = create_gbl_update_function(config, SkillPolicy(), SkillCritic(), action_dim, obs_shape) # Placeholder

    # Initialize training state
    train_state = init_train_state(config, env)

    num_generations = 100 # Example
    eval_steps_per_generation = 1000 # Example

    for generation in range(num_generations):
        print(f"\n--- Generation {generation} ---")

        # 1. Evaluation Phase
        train_state, trajectory_data = run_evaluation_phase(train_state, config, env, env.default_params, eval_steps_per_generation)
        # (Log fitness, maybe other metrics from trajectories)
        print(f"Max Fitness: {jnp.max(train_state.population_fitness):.2f}")

        # 2. Evolution Phase
        train_state = run_evolution_phase(train_state, config)

        # 3. Skill Refinement Phase
        # train_state = run_skill_refinement_phase(train_state, config, gbl_update_fn) # Uncomment when implemented

        # (Add logging, saving checkpoints, etc.)

if __name__ == "__main__":
    main()
