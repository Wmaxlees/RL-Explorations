from typing import NamedTuple, Any, List, Optional
import chex
import optax

class AgentState(NamedTuple):
    selector_params: Any
    skill_subset_mask: chex.Array

# Use TrainState from flax.training or build a custom one if needed
# This holds parameters + optimizer state for things trained with Optax
class SkillTrainState(NamedTuple):
    policy_params: Any
    critic_params: Any
    policy_opt_state: optax.OptState
    critic_opt_state: Optional[optax.OptState]

# The main state holding *everything* that changes
class TrainState(NamedTuple):
    # Embedder data
    embedder_params: Any
    embedder_opt_state: optax.OptState

    # EA Population State (P agents)
    # We often "vmap" the AgentState structure itself
    population_agent_states: AgentState # AgentState pytree, but leaves have shape [P, ...]

    # Shared Skill State (n skills)
    # A list/tuple or pytree holding the state for each skill
    shared_skill_states: List[SkillTrainState] # Or a pytree for easier manipulation

    # Environment State (Potentially batched for the population)
    env_state: Any # Specific to the JAX environment
    obs: chex.Array # Current observation, shape [P, obs_shape...]

    # Fitness (Calculated during evaluation)
    population_fitness: chex.Array # Shape [P]

    # Global PRNG Key
    prng_key: chex.PRNGKey

    # Timestep counter
    timestep: int
