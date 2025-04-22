import chex
from craftax.craftax_env import make_craftax_env_from_name
import distrax
from flax import struct
import functools
import jax
import jax.numpy as jnp
import optax
import time
from typing import Any, List, Tuple, Optional, Dict
import wandb

from lib.networks import SkillSelector, SkillPolicy, SkillCritic, Embedder
from lib.states import TrainState, AgentState, SkillTrainState
from lib.env_wrappers import BatchEnvWrapper, GymnaxWrapper


class Config:
    population_size: int = 10 # 100
    num_skills: int = 4 # 16
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

    # PPO Hyperparameters
    ppo_epochs: int = 4
    ppo_num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    target_kl: Optional[float] = None
    gradient_accumulation_steps: int = 16

    # WandB Config
    wandb_project: str = "HyCE-RL"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None


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
    advantages: Optional[chex.Array] = None # Shape [N] after GAE
    targets: Optional[chex.Array] = None    # Shape [N] after GAE



def calculate_gae(
    rewards: chex.Array, # Shape [N_steps]
    values: chex.Array, # Shape [N_steps] - V(s_t)
    next_values: chex.Array, # Shape [N_steps] - V(s_{t+1})
    dones: chex.Array, # Shape [N_steps]
    gamma: float,
    gae_lambda: float
) -> Tuple[chex.Array, chex.Array]:
    """Calculates GAE advantages and value targets."""
    next_values_corrected = jnp.where(dones, 0.0, next_values)

    deltas = rewards + gamma * next_values_corrected - values
    advantages = jnp.zeros_like(rewards)
    last_gae_lam = 0.0

    # Scan function remains the same, operates over the single dimension
    def _adv_scan_step(carry, delta_and_done):
        gae_lam = carry
        delta, done = delta_and_done
        gae_lam = delta + gamma * gae_lambda * (1.0 - done) * gae_lam
        return gae_lam, gae_lam

    # Scan backwards through the single (time) dimension
    _, advantages_reversed = jax.lax.scan(
        _adv_scan_step, last_gae_lam, (deltas, dones), reverse=True
    )
    advantages = jnp.flip(advantages_reversed, axis=0) # Flip the single dimension back

    targets = advantages + values
    # Returns 1D arrays: advantages [N_steps], targets [N_steps]
    return advantages, targets

def ppo_update_step(
params_to_update: Tuple[Any, Any, Any], # (embedder_params, policy_params_k, critic_params_k)
    # Opt states/optimizers removed, handled by caller
    full_minibatch: Dict, # The full minibatch from shuffled_filtered_data
    skill_k_index: int, # Index of the skill being updated
    config: Config,
    action_dim: int
) -> Tuple[Tuple[Any, Any, Any], Dict]: # Returns (embedder_grads, policy_grads_k, critic_grads_k), metrics
    """Calculates PPO loss and gradients for one skill's data within a minibatch."""

    embedder_params, policy_params_k, critic_params_k = params_to_update

    # Instantiate networks
    embedder_net = Embedder(hidden_dim=config.embedder_hidden_dim, embedding_dim=config.embedder_embedding_dim)
    skill_policy_net = SkillPolicy(hidden_dim=config.skill_policy_hidden_dim, action_dim=action_dim)
    skill_critic_net = SkillCritic(hidden_dim=config.skill_critic_hidden_dim)

    # Define PPO loss function for skill k, operating on the full minibatch
    def ppo_loss_fn(embed_p, policy_p, critic_p, mb):
        # Extract data from the full minibatch
        obs = mb.observations
        actions = mb.actions
        log_probs_old = mb.log_probs
        skill_indices_mb = mb.skill_indices
        adv = mb.advantages
        targets = mb.targets

        # --- Create mask for skill k within the minibatch ---
        skill_k_mask = (skill_indices_mb == skill_k_index) # Boolean mask [minibatch_size]
        # ---

        # Forward pass for ALL data points in minibatch
        embeddings = embedder_net.apply({'params': embed_p}, obs)
        new_logits = skill_policy_net.apply({'params': policy_p}, embeddings)
        new_values = skill_critic_net.apply({'params': critic_p}, embeddings)

        # --- Policy Loss (Compute per element, then mask and average) ---
        new_action_dist = distrax.Categorical(logits=new_logits)
        new_log_probs = new_action_dist.log_prob(actions)
        ratio = jnp.exp(new_log_probs - log_probs_old)
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8) # Normalize over the whole minibatch advantages

        pg_loss1 = adv_norm * ratio
        pg_loss2 = adv_norm * jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
        # Per-element policy loss (before mean)
        per_element_pg_loss = -jnp.minimum(pg_loss1, pg_loss2)
        # Masked mean
        policy_loss = jnp.sum(per_element_pg_loss * skill_k_mask) / (jnp.sum(skill_k_mask) + 1e-8)

        # --- Value Loss (Compute per element, then mask and average) ---
        per_element_value_loss = 0.5 * jnp.square(new_values - targets)
        # Masked mean
        value_loss = jnp.sum(per_element_value_loss * skill_k_mask) / (jnp.sum(skill_k_mask) + 1e-8)

        # --- Entropy Bonus (Compute per element, then mask and average) ---
        entropy = new_action_dist.entropy()
        # Masked mean
        entropy_bonus = jnp.sum(entropy * skill_k_mask) / (jnp.sum(skill_k_mask) + 1e-8)

        # --- Total Loss ---
        # Note: Coefficients multiply the *average* masked losses
        total_loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy_bonus

        metrics = {
            "total_loss": total_loss, # Already masked and averaged correctly
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_bonus, # Store the averaged entropy bonus
            "skill_k_active_ratio": jnp.mean(skill_k_mask.astype(jnp.float32)), # Track how much data was used
        }
        return total_loss, metrics

    # Calculate gradients w.r.t. (embedder, policy_k, critic_k) params
    grad_calc_fn = jax.value_and_grad(ppo_loss_fn, argnums=(0, 1, 2), has_aux=True)

    (loss, metrics), grads_tuple = grad_calc_fn(
        embedder_params, policy_params_k, critic_params_k,
        full_minibatch # Pass the TrajectoryData object directly
    )

    return grads_tuple, metrics

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
        "total_skills_selected": jnp.zeros((config.population_size, config.num_skills), dtype=jnp.int32),
    }

    def step_env_scan_body(carry, _unused_step_idx):
        key, current_obs, current_env_state, accumulated_reward = \
            carry["key"], carry["obs"], carry["env_state"], carry["accumulated_reward"]
        accumulated_reward = carry["accumulated_reward"]
        total_skills_selected = carry["total_skills_selected"]

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

        skill_one_hot = jax.nn.one_hot(skill_indices_k, num_classes=config.num_skills, dtype=jnp.int32)
        new_total_skills_selected = total_skills_selected + skill_one_hot

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
            "accumulated_reward": new_accumulated_reward,
            "total_skills_selected": new_total_skills_selected
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
    final_total_skills_selected = final_carry["total_skills_selected"] # Shape [P, N_skill]

    # Transpose collected data to [P, T, ...]
    trajectory_data_transposed = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), collected_transitions_stacked)

    # Pack into TrajectoryData structure
    trajectory_data = TrajectoryData(**trajectory_data_transposed)

    eval_metrics = {
        "eval/fitness_mean": jnp.mean(final_fitness),
        "eval/fitness_median": jnp.median(final_fitness),
        "eval/fitness_max": jnp.max(final_fitness),
        "eval/fitness_min": jnp.min(final_fitness),
        "eval/fitness_std": jnp.std(final_fitness),
        "eval/avg_skill_usage_fraction": jnp.mean(final_total_skills_selected / num_steps, axis=0), # Average fraction usage per skill across pop [N_skill]
        # Add more metrics as needed
    }

    # Update TrainState
    new_train_state = train_state._replace(
        prng_key=final_key,
        obs=final_obs, # Store the very last obs
        env_state=final_env_state, # Store the last env state
        population_fitness=final_fitness,
        timestep=train_state.timestep + num_steps * config.population_size
    )

    return new_train_state, trajectory_data, eval_metrics


@functools.partial(jax.jit, static_argnames=('config'))
def run_evolution_phase(train_state: TrainState, config: Config) -> TrainState:
    """Applies EA selection and variation (elitism + mutation) to the population."""
    print("Evolving population...")
    key, key_select, key_mutate_selector, key_mutate_mask = jax.random.split(train_state.prng_key, 4)

    # --- Selection ---
    elite_indices = jnp.argsort(train_state.population_fitness)[-config.num_elites:]
    elite_fitness = train_state.population_fitness[elite_indices]

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

    avg_skills_per_agent = jnp.mean(jnp.sum(new_skill_subset_mask, axis=-1))

    evo_metrics = {
        "evo/elite_fitness_mean": jnp.mean(elite_fitness),
        "evo/elite_fitness_min": jnp.min(elite_fitness),
        "evo/elite_fitness_max": jnp.max(elite_fitness),
        "evo/avg_skills_per_agent": avg_skills_per_agent,
    }

    new_train_state = train_state._replace(
        population_agent_states=new_population_agent_states,
        prng_key=key
    )
    return new_train_state, evo_metrics


# @functools.partial(jax.jit, static_argnames=("config","action_dim")) # Jitting this whole thing might be hard
def run_skill_refinement_phase(
    train_state: TrainState,
    config: Config,
    trajectory_data: TrajectoryData,
    action_dim: int
    ) -> TrainState:
    """Filters data, runs PPO updates for skills and shared embedder."""
    print("Refining skills using PPO...")
    key = train_state.prng_key
    ppo_metrics_aggregated = {} # Initialize dict for metrics

    num_top_agents_float = config.population_size * 4 / 10
    num_top_agents = int(num_top_agents_float)
    num_top_agents = max(1, num_top_agents)

    _, top_agent_indices = jax.lax.top_k(train_state.population_fitness, k=num_top_agents)
    
    def gather_top_agents(leaf_data):
        # Check if it has the population dimension
        if leaf_data.shape[0] == config.population_size:
            return leaf_data[top_agent_indices]
        # Otherwise, assume it doesn't have the pop dimension (shouldn't happen here)
        return leaf_data
    top_agent_data = jax.tree_util.tree_map(gather_top_agents, trajectory_data)
    pop_size_filtered, traj_len = top_agent_data.observations.shape[:2]
    num_filtered_steps = pop_size_filtered * traj_len
    def flatten_fn(leaf_data):
         if leaf_data.shape[:2] == (pop_size_filtered, traj_len):
             return leaf_data.reshape(num_filtered_steps, *leaf_data.shape[2:])
         return leaf_data
    filtered_data = jax.tree_util.tree_map(flatten_fn, top_agent_data)
    # ---

    print(f"  Filtered steps from top agents: {num_filtered_steps}")
    ppo_metrics_aggregated["ppo/num_agents_filtered"] = float(num_top_agents) # Convert JAX scalar
    ppo_metrics_aggregated["ppo/num_steps_filtered"] = float(num_filtered_steps)

    min_req_steps = config.ppo_num_minibatches
    if num_filtered_steps < min_req_steps:
        print(f"  Skipping PPO updates: Not enough data ({num_filtered_steps}) for {config.ppo_num_minibatches} minibatches.")
        key, _ = jax.random.split(key)
        ppo_metrics_aggregated["ppo/skipped_updates"] = 1.0
        return train_state._replace(prng_key=key), ppo_metrics_aggregated

    # --- 2. Calculate GAE for ALL filtered data ---
    # Calculate GAE once for the whole filtered dataset
    advantages, targets = calculate_gae(
        rewards=filtered_data.rewards,
        values=filtered_data.values,
        next_values=filtered_data.next_values,
        dones=filtered_data.dones,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda
    )
    # Add advantages/targets to the filtered_data pytree
    # Ensure no key collisions if TrajectoryData already had these fields
    filtered_data_with_adv = filtered_data.replace(
        advantages=advantages,
        targets=targets
    )
    ppo_metrics_aggregated["ppo/advantages_mean"] = float(jnp.mean(advantages))
    ppo_metrics_aggregated["ppo/targets_mean"] = float(jnp.mean(targets))
    # ---

    # --- 3. Initialize Loop Variables ---
    current_embedder_params = train_state.embedder_params
    current_embedder_opt_state = train_state.embedder_opt_state
    current_shared_skill_states = list(train_state.shared_skill_states)
    embedder_tx = optax.adam(learning_rate=config.embedder_lr)
    policy_tx = optax.adam(learning_rate=config.skill_policy_lr)
    critic_tx = optax.adam(learning_rate=config.skill_critic_lr)
    optimizers = (embedder_tx, policy_tx, critic_tx)

    epoch_metrics = []

    # --- 4. PPO Epoch Loop ---
    for epoch in range(config.ppo_epochs):
        key, shuffle_key = jax.random.split(key)
        permuted_indices = jax.random.permutation(shuffle_key, num_filtered_steps)
        shuffled_filtered_data = jax.tree_util.tree_map(lambda x: x[permuted_indices], filtered_data_with_adv)

        accumulated_embedder_grads = jax.tree_util.tree_map(jnp.zeros_like, current_embedder_params)
        updates_counted_for_embedder = 0

        minibatch_metrics = []

        # --- 5. Minibatch Loop (Iterate over the *whole* filtered dataset) ---
        minibatch_size = num_filtered_steps // config.ppo_num_minibatches
        if minibatch_size == 0:
            minibatch_size = num_filtered_steps # Handle small data case
            actual_num_minibatches = 1
        else:
            actual_num_minibatches = config.ppo_num_minibatches

        num_processed_steps = actual_num_minibatches * minibatch_size
        processed_data = jax.tree_util.tree_map(
            lambda x: x[:num_processed_steps], shuffled_filtered_data
        )

        minibatched_data = jax.tree_util.tree_map(
            lambda x: x.reshape((actual_num_minibatches, minibatch_size) + x.shape[1:]),
            processed_data
        )

        for i in range(actual_num_minibatches):
            minibatch = jax.tree_util.tree_map(lambda x: x[i], minibatched_data)

            step_metrics_for_minibatch = []

            # --- 6. Skill Loop (Apply updates for each skill based on this minibatch) ---
            for k in range(config.num_skills):
                # a. Get current states for skill k
                skill_state_k = current_shared_skill_states[k]
                params_tuple = (current_embedder_params, skill_state_k.policy_params, skill_state_k.critic_params)

                # b. Perform PPO update step (calculates grads based on masked loss)
                key, step_key = jax.random.split(key)
                (embedder_grads, policy_grads_k, critic_grads_k), metrics = ppo_update_step(
                    params_tuple,
                    minibatch, # Pass full minibatch
                    k,         # Pass skill index k
                    config,
                    action_dim
                )

                metrics['skill_index'] = k
                step_metrics_for_minibatch.append(metrics)

                # Check if any updates happened for this skill (mask was not all False)
                # If skill_k_active_ratio is 0, grads might be zero/NaN - skip update?
                # Optax handles zero gradients fine, NaNs would be an issue.
                # if metrics["skill_k_active_ratio"] > 0:
                if True:

                    # c. Update Policy and Critic for skill k immediately
                    policy_updates, new_policy_opt_state = policy_tx.update(policy_grads_k, skill_state_k.policy_opt_state, skill_state_k.policy_params)
                    new_policy_params_k = optax.apply_updates(skill_state_k.policy_params, policy_updates)

                    critic_updates, new_critic_opt_state = critic_tx.update(critic_grads_k, skill_state_k.critic_opt_state, skill_state_k.critic_params)
                    new_critic_params_k = optax.apply_updates(skill_state_k.critic_params, critic_updates)

                    # Store the updated state for skill k back into the list
                    current_shared_skill_states[k] = skill_state_k._replace(
                        policy_params=new_policy_params_k,
                        critic_params=new_critic_params_k,
                        policy_opt_state=new_policy_opt_state,
                        critic_opt_state=new_critic_opt_state
                    )

                    # d. Accumulate gradients for the shared embedder
                    # Only accumulate if skill k was active in this minibatch
                    accumulated_embedder_grads = jax.tree_util.tree_map(
                        lambda acc, new: acc + new, accumulated_embedder_grads, embedder_grads
                    )
                    updates_counted_for_embedder += 1 # Count how many valid grad contributions we got

                    # e. Apply embedder update periodically
                    if updates_counted_for_embedder > 0 and updates_counted_for_embedder % config.gradient_accumulation_steps == 0:
                         # Average over number of *accumulated* updates, not config steps
                         num_accum = config.gradient_accumulation_steps
                         avg_embedder_grads = jax.tree_util.tree_map(
                             lambda g: g / num_accum, accumulated_embedder_grads
                         )
                         embedder_updates, new_embedder_opt_state = embedder_tx.update(
                             avg_embedder_grads, current_embedder_opt_state, current_embedder_params
                         )
                         current_embedder_params = optax.apply_updates(current_embedder_params, embedder_updates)
                         current_embedder_opt_state = new_embedder_opt_state

                         # Reset accumulator
                         accumulated_embedder_grads = jax.tree_util.tree_map(
                             jnp.zeros_like, current_embedder_params
                         )
                         # Reset counter (implicitly handled by modulo next time, or explicitly reset counter here)
                         # updates_counted_for_embedder = 0 # If resetting here

            # End skill loop
            if step_metrics_for_minibatch:
                agg_mb_metrics = {}
                keys_to_avg = ["total_loss", "policy_loss", "value_loss", "entropy", "approx_kl",
                               "grad_norm/embedder", "grad_norm/policy", "grad_norm/critic",
                               "skill_k_active_ratio"]
                num_skills_active_in_mb = sum(m['skill_k_active_ratio'] > 1e-6 for m in step_metrics_for_minibatch)

                for metric_name in keys_to_avg:
                    # Average only over skills that were active in this minibatch
                    valid_vals = [m[metric_name] for m in step_metrics_for_minibatch if metric_name in m and m['skill_k_active_ratio'] > 1e-6]
                    if valid_vals:
                        # Ensure values are numeric before mean
                        numeric_vals = [v for v in valid_vals if isinstance(v, (int, float, jax.Array))]
                        if numeric_vals:
                           # Also update the f-string key below
                           agg_mb_metrics[f"mb_avg/{metric_name}"] = jnp.mean(jnp.array(numeric_vals)).item()
                        else:
                           agg_mb_metrics[f"mb_avg/{metric_name}"] = float('nan')
                    else:
                         # Also update the f-string key below
                        agg_mb_metrics[f"mb_avg/{metric_name}"] = 0.0

                minibatch_metrics.append(agg_mb_metrics)
        # End minibatch loop

        # Apply final embedder update for any remaining accumulated gradients at end of epoch
        remaining_updates = updates_counted_for_embedder % config.gradient_accumulation_steps
        if remaining_updates > 0:
             avg_embedder_grads = jax.tree_util.tree_map(
                 lambda g: g / remaining_updates, accumulated_embedder_grads
             )
             embedder_updates, new_embedder_opt_state = embedder_tx.update(
                 avg_embedder_grads, current_embedder_opt_state, current_embedder_params
             )
             current_embedder_params = optax.apply_updates(current_embedder_params, embedder_updates)
             current_embedder_opt_state = new_embedder_opt_state
        
        if minibatch_metrics:
          epoch_agg = {}
          # Get all the keys from the first minibatch's aggregated metrics
          # (Assumes all minibatch dicts have the same keys)
          all_mb_keys = minibatch_metrics[0].keys()

          # --- Use a different loop variable name ---
          for mb_metric_key in all_mb_keys:
              # Calculate the mean for this metric across all minibatches in the epoch
              # Add .item() to store a Python float in epoch_agg
              epoch_agg[f"epoch_avg/{mb_metric_key}"] = jnp.mean(jnp.array([m[mb_metric_key] for m in minibatch_metrics])).item()

          epoch_metrics.append(epoch_agg)
        # End epoch final embedder update

    # End epoch loop

    if epoch_metrics:
        final_keys = epoch_metrics[0].keys()
        for final_key in final_keys:
             # Average the epoch averages
             ppo_metrics_aggregated[f"ppo/{final_key.replace('epoch_avg/mb_avg/', '')}"] = float(jnp.mean(jnp.array([e[final_key] for e in epoch_metrics])))
             # Example key becomes: ppo/total_loss

    # Update the main train state
    final_train_state = train_state._replace(
        embedder_params=current_embedder_params,
        embedder_opt_state=current_embedder_opt_state,
        shared_skill_states=current_shared_skill_states,
        prng_key=key
    )

    return final_train_state, ppo_metrics_aggregated


def main():
    config = Config() # Load config

    wandb.init(project=config.wandb_project,
        config=vars(config),
        name=config.wandb_run_name,
        save_code=False,
    )
    config = Config(**wandb.config) # Update config with sweep params if any

    env = make_craftax_env_from_name(config.env_name, True)
    env = BatchEnvWrapper(env, num_envs=config.population_size)
    env_params = env.default_params
    obs_shape = env.observation_space(env_params).shape
    action_dim = env.action_space(env_params).n # Get action_dim

    # Initialize training state
    train_state = init_train_state(config, env)

    num_generations = 100 # Example
    eval_steps_per_generation = 100 # Example

    start_time = time.time()

    for generation in range(num_generations):
        gen_start_time = time.time()
        logs = {"generation": generation, "timesteps": train_state.timestep}
        print(f"\n--- Generation {generation} ---")

        # 1. Evaluation Phase
        eval_start_time = time.time()
        train_state, trajectory_data, eval_metrics = run_evaluation_phase(train_state, config, env, env.default_params, eval_steps_per_generation)
        eval_time = time.time() - eval_start_time
        logs.update({k: float(v) for k, v in eval_metrics.items() if 'avg_skill_usage_fraction' not in k})
        for i, usage in enumerate(eval_metrics["eval/avg_skill_usage_fraction"]):
            logs[f"eval/skill_{i}_usage_fraction"] = float(usage)
        logs["timing/eval_phase_sec"] = eval_time
        print(f"Max Fitness: {jnp.max(train_state.population_fitness):.2f}")

        # 2. Evolution Phase
        evo_start_time = time.time()
        train_state, evo_metrics = run_evolution_phase(train_state, config)
        evo_time = time.time() - evo_start_time
        logs.update({k: float(v) for k, v in evo_metrics.items()}) # Convert JAX scalars
        logs["timing/evo_phase_sec"] = evo_time

        # 3. Skill Refinement Phase
        refine_start_time = time.time()
        train_state, ppo_metrics = run_skill_refinement_phase(
             train_state, config, trajectory_data, action_dim
        )
        refine_time = time.time() - refine_start_time
        logs.update(ppo_metrics) # Already floats from the function
        logs["timing/refine_phase_sec"] = refine_time

        gen_time = time.time() - gen_start_time
        logs["timing/generation_sec"] = gen_time
        total_elapsed_time = time.time() - start_time
        logs["timing/total_elapsed_sec"] = total_elapsed_time
        logs["timing/sps"] = (eval_steps_per_generation * config.population_size) / gen_time # Steps per second

        wandb.log(logs)

        # (Add saving checkpoints, etc.)

    wandb.finish()

if __name__ == "__main__":
    main()
