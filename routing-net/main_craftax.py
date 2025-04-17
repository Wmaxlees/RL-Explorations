import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax
import distrax
import numpy as np
from typing import Sequence, Optional, Any, Dict, Tuple
import functools
import time
import tqdm
import dataclasses
import jax.tree_util

# Try importing craftax environment and utilities
try:
    import craftax.craftax_env as craftax_env
    # Use the actual state class if needed, otherwise Observation is often sufficient
    # import craftax.craftax.craftax_state as EnvState
    import craftax.craftax.constants as constants
    from craftax.craftax_env import make_craftax_env_from_name
    # Craftax might require gymnax or provide its own JAX interface
except ImportError:
    print("*" * 80)
    print("Craftax environment not found.")
    print("Please install it: pip install craftax")
    print("*" * 80)
    exit()

try:
    from ott.tools import soft_sort
    from ott.tools.soft_sort import ranks as soft_rank
    OTT_AVAILABLE = True
except ImportError:
    OTT_AVAILABLE = False
    print("Warning: ott-jax not found. TopKSelectorOTT will not be available.")

# --- Configuration ---
@dataclasses.dataclass
class TrainConfig:
    """Hyperparameters for training"""
    # Env Args
    env_name: str = "Craftax-Symbolic-v1"
    seed: int = 42
    num_envs: int = 16 # Number of parallel environments

    # PPO Args
    total_timesteps: int = 2e7 # Total training steps
    num_steps: int = 512      # Steps per rollout per environment
    learning_rate: float = 5e-5
    anneal_lr: bool = True    # Linearly anneal learning rate
    gamma: float = 0.99       # Discount factor
    gae_lambda: float = 0.95  # GAE lambda parameter
    num_minibatches: int = 4  # Number of mini-batches per PPO epoch
    update_epochs: int = 4    # Number of PPO epochs per rollout
    clip_coef: float = 0.2    # PPO clipping coefficient
    ent_coef: float = 0.001   # Entropy coefficient
    vf_coef: float = 0.5      # Value function coefficient
    max_grad_norm: float = 0.5 # Max gradient norm for clipping

    # Network Args
    use_routing_network: bool = True
    num_subroutines: int = 8  # Number of parallel subroutines
    selector_hidden_dims: Sequence[int] = (64,) # Hidden dims in the selector MLP
    subroutine_hidden_dims: Sequence[int] = (64, 32, 64,) # Hidden dims in each subroutine MLP
    final_head_input_dim: int = 128 # Hidden dim in the final combined MLP head
    selector_type: str = "sigmoid" # 'sigmoid' or 'topk_ott' (if available)
    topk_k: int = 4 # Value of K if selector_type is 'topk_ott'
    routing_activation: str = "relu"
    # Using standard AC network structure for clarity, will remove unused modular parts
    ac_hidden_dim: int = 512 # Hidden dim for AC network layers
    ac_activation: str = "tanh" # Activation for AC network

    # ICM Args
    train_icm: bool = True  # Enable ICM
    icm_reward_coeff: float = 0.001 # Coefficient for intrinsic reward
    icm_lr: float = 1e-4           # Separate LR for ICM networks (often same as AC)
    icm_forward_loss_coef: float = 0.2 # Weight for forward dynamics loss (Burda et al.)
    icm_inverse_loss_coef: float = 0.8 # Weight for inverse dynamics loss (Burda et al.)
    icm_latent_dim: int = 256       # Dimension of ICM state embeddings (Burda et al.)
    icm_hidden_dim: int = 256       # Hidden layer size in ICM models (Burda et al.)
    icm_activation: str = "relu"    # Activation for ICM network

    # Logging
    log_frequency: int = 10 # Log progress every N updates

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self):
        assert self.batch_size % self.num_minibatches == 0
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self):
        return int(self.total_timesteps // self.batch_size)


# --- Network Definition ---

def get_activation(name: str):
    if name == "relu":
        return nn.relu
    elif name == "tanh":
        return nn.tanh
    else:
        raise ValueError(f"Unknown activation: {name}")

class Encoder(nn.Module):
    """Encodes observations into a feature vector."""
    embed_dim: int
    hidden_dim: int
    activation: callable

    @nn.compact
    def __call__(self, obs: Dict[str, jnp.ndarray]): # Craftax obs is a dict
        # Flatten and concatenate observation dictionary elements
        # Assuming obs contains numerical arrays. Adjust if needed.
        flat_obs_list = jax.tree_util.tree_leaves(obs)
        # Filter out any non-array elements if necessary
        flat_obs_list = [o.reshape(o.shape[:-1] + (-1,)) if o.ndim > 1 else o for o in flat_obs_list if isinstance(o, jnp.ndarray)]

        if not flat_obs_list:
             raise ValueError("Observation dictionary contains no JAX arrays.")

        # Ensure all arrays have compatible batch dimensions if input is batched
        batch_dims = [o.shape[:-1] for o in flat_obs_list]
        if len(set(batch_dims)) > 1:
             print(f"Warning: Inconsistent batch dims in obs: {batch_dims}")
             # Attempt to broadcast - assumes leading dims are batch dims
             common_batch_shape = jnp.broadcast_shapes(*[o.shape[:-1] for o in flat_obs_list])
             flat_obs_list = [jnp.broadcast_to(o, common_batch_shape + o.shape[-1:]) for o in flat_obs_list]


        x = jnp.concatenate(flat_obs_list, axis=-1)
        x = x.astype(jnp.float32)

        # Normalize the flattened observation vector
        x = nn.LayerNorm(epsilon=1e-5)(x)

        x = nn.Dense(features=self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        encoded_features = nn.Dense(features=self.embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        encoded_features = self.activation(encoded_features) # Use activation on output too? Common in some models.
        return encoded_features
    
class SubroutineSelector(nn.Module):
    num_subroutines: int
    hidden_dims: Sequence[int]
    activation_fn: callable

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x
        for i, hidden_dim in enumerate(self.hidden_dims):
            h = nn.Dense(features=hidden_dim, name=f"SelectorHidden_{i}",
                         kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
            h = self.activation_fn(h)
        routing_logits = nn.Dense(features=self.num_subroutines, name="SelectorOutputDense",
                                  kernel_init=orthogonal(0.01), bias_init=constant(0.0))(h)
        selection_weights = nn.sigmoid(routing_logits)
        return selection_weights

class TopKSelectorOTT(nn.Module):
    num_subroutines: int
    k: int
    hidden_dims: Sequence[int]
    activation_fn: callable
    soft_sort_kwargs: dict = dataclasses.field(default_factory=lambda: {"temperature": 0.1})

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if not OTT_AVAILABLE:
            raise ImportError("OTT-JAX is required for TopKSelectorOTT but not installed.")
        h = x
        for i, hidden_dim in enumerate(self.hidden_dims):
            h = nn.Dense(features=hidden_dim, name=f"SelectorHidden_{i}",
                         kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
            h = self.activation_fn(h)
        scores = nn.Dense(features=self.num_subroutines, name="SelectorOutputDense",
                          kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h)
        ranks = soft_rank(scores, axis=-1, **self.soft_sort_kwargs)
        inverted_ranks = (self.num_subroutines - 1) - ranks
        threshold_rank_approx = self.num_subroutines - self.k
        steepness = 10.0
        soft_mask = nn.sigmoid((inverted_ranks - threshold_rank_approx) * steepness)
        # print("Warning: TopKSelectorOTT outputs a heuristic soft mask based on soft ranks.") # Remove excessive print
        return soft_mask

class Subroutine(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    activation_fn: callable

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x
        for i, hidden_dim in enumerate(self.hidden_dims):
            h = nn.Dense(features=hidden_dim, name=f"SubroutineHidden_{i}",
                         kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
            h = self.activation_fn(h)
        output = nn.Dense(features=self.output_dim, name="SubroutineOutput",
                          kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h)
        return output

class ActorCriticNetwork(nn.Module):
    """Standard Actor-Critic Network."""
    num_actions: int
    config: TrainConfig

    @nn.compact
    def __call__(self, obs: Dict[str, jnp.ndarray]):
        activation = get_activation(self.config.ac_activation)
        hidden_dim = self.config.ac_hidden_dim
        routing_activation = get_activation(self.config.routing_activation)

        # --- Encoder ---
        # Using a simple 2-layer MLP encoder as in the original PPO baseline
        flat_obs_list = jax.tree_util.tree_leaves(obs)
        flat_obs_list = [o.reshape(o.shape[:-1] + (-1,)) if o.ndim > 1 else o for o in flat_obs_list if isinstance(o, jnp.ndarray)]
        if not flat_obs_list:
             raise ValueError("Observation dictionary contains no JAX arrays.")
        # Ensure consistent batch shapes before concatenation
        batch_dims = [o.shape[:-1] for o in flat_obs_list]
        if len(set(batch_dims)) > 1:
             common_batch_shape = jnp.broadcast_shapes(*[o.shape[:-1] for o in flat_obs_list])
             flat_obs_list = [jnp.broadcast_to(o, common_batch_shape + o.shape[-1:]) for o in flat_obs_list]

        x = jnp.concatenate(flat_obs_list, axis=-1)
        x = x.astype(jnp.float32)
        x = nn.LayerNorm(epsilon=1e-5)(x)

        x = nn.Dense(features=hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        encoded_state = nn.Dense(features=hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        encoded_state = activation(encoded_state)

        if self.config.use_routing_network:
            # --- 2a. Select Subroutines ---
            if self.config.selector_type == "sigmoid":
                selector = SubroutineSelector(
                    num_subroutines=self.config.num_subroutines,
                    hidden_dims=self.config.selector_hidden_dims,
                    activation_fn=routing_activation,
                    name="Selector"
                )
                selection_weights = selector(encoded_state) # Shape: (batch, num_subroutines)
            elif self.config.selector_type == "topk_ott":
                selector = TopKSelectorOTT(
                    num_subroutines=self.config.num_subroutines,
                    k=self.config.topk_k,
                    hidden_dims=self.config.selector_hidden_dims,
                    activation_fn=routing_activation,
                    name="Selector"
                )
                selection_weights = selector(encoded_state) # Shape: (batch, num_subroutines) - soft mask
            else:
                raise ValueError(f"Unknown selector_type: {self.config.selector_type}")

            # --- 2b. Run Subroutines ---
            subroutine_outputs = []
            subroutine_feature_dim = self.config.final_head_input_dim // self.config.num_subroutines
            if subroutine_feature_dim == 0: subroutine_feature_dim = 1 # Ensure positive dim

            for i in range(self.config.num_subroutines):
                sub = Subroutine(
                    hidden_dims=self.config.subroutine_hidden_dims,
                    output_dim=subroutine_feature_dim,
                    activation_fn=routing_activation,
                    name=f"Subroutine_{i}"
                )
                # Each subroutine processes the original encoded state
                sub_output = sub(encoded_state)
                subroutine_outputs.append(sub_output)

            # Stack outputs: Shape (batch, num_subroutines, sub_feature_dim)
            all_sub_outputs = jnp.stack(subroutine_outputs, axis=1)

            # --- 2c. Merge Subroutine Outputs (Weighted Sum) ---
            # Expand weights for broadcasting: Shape (batch, num_subroutines, 1)
            expanded_weights = jnp.expand_dims(selection_weights, axis=-1)
            weighted_outputs = all_sub_outputs * expanded_weights
            # Sum over subroutines: Shape (batch, sub_feature_dim)
            merged_output = jnp.sum(weighted_outputs, axis=1)

            # --- 2d. Skip Connection & Final Feature Prep ---
            # Project encoded state if needed to match merged_output dim for combining
            # OR project merged_output to match encoded_state dim
            # OR concatenate and project down to final_head_input_dim
            # Let's concatenate and project, similar to some residual MoE patterns:

            # combined_skip_merge = jnp.concatenate([encoded_state, merged_output], axis=-1)
            # combined_features = nn.Dense(features=self.config.final_head_input_dim, name="CombineProj",
            #                              kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(combined_skip_merge)
            # combined_features = routing_activation(combined_features)

            # Alternative: Additive skip (like original) - requires projection if dims mismatch
            if encoded_state.shape[-1] != merged_output.shape[-1]:
                 # Project merged output to match encoder dim
                 merged_output_proj = nn.Dense(features=encoded_state.shape[-1], name="MergeProjSkip",
                                              kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(merged_output)
                 combined_features = encoded_state + merged_output_proj # Additive skip
            else:
                 combined_features = encoded_state + merged_output # Additive skip
            combined_features = routing_activation(combined_features) # Activation after skip


        else:
            # --- No Routing: Use encoded state directly ---
            combined_features = encoded_state # Use output of the main encoder

        # --- Actor Head ---
        actor_h = nn.Dense(features=hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(combined_features)
        actor_h = activation(actor_h)
        actor_logits = nn.Dense(features=self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_h)

        # --- Critic Head ---
        critic_h = nn.Dense(features=hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(combined_features)
        critic_h = activation(critic_h)
        critic_value = nn.Dense(features=1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic_h)

        return actor_logits, jnp.squeeze(critic_value, axis=-1) # Return logits and scalar value


# --- ICM Network Definition ---

class ICMEncoder(nn.Module):
    """Encodes observations for the ICM module."""
    latent_dim: int
    hidden_dim: int
    activation: callable

    @nn.compact
    def __call__(self, obs: Dict[str, jnp.ndarray]):
        # Reusing the observation flattening logic from AC encoder
        flat_obs_list = jax.tree_util.tree_leaves(obs)
        flat_obs_list = [o.reshape(o.shape[:-1] + (-1,)) if o.ndim > 1 else o for o in flat_obs_list if isinstance(o, jnp.ndarray)]
        if not flat_obs_list:
             raise ValueError("Observation dictionary contains no JAX arrays.")
        # Ensure consistent batch shapes before concatenation
        batch_dims = [o.shape[:-1] for o in flat_obs_list]
        if len(set(batch_dims)) > 1:
             common_batch_shape = jnp.broadcast_shapes(*[o.shape[:-1] for o in flat_obs_list])
             flat_obs_list = [jnp.broadcast_to(o, common_batch_shape + o.shape[-1:]) for o in flat_obs_list]

        x = jnp.concatenate(flat_obs_list, axis=-1)
        x = x.astype(jnp.float32)
        # Note: No LayerNorm here, as per some ICM implementations (features should be dynamic)
        # x = nn.LayerNorm(epsilon=1e-5)(x) # Optional: experiment with normalization

        x = nn.Dense(features=self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        latent_features = nn.Dense(features=self.latent_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        # No final activation on latent features, common practice
        return latent_features

class InverseModel(nn.Module):
    """Predicts action taken between two encoded states."""
    num_actions: int
    hidden_dim: int
    activation: callable

    @nn.compact
    def __call__(self, phi_s: jnp.ndarray, phi_s_next: jnp.ndarray) -> jnp.ndarray:
        # Concatenate the features from state s and s_next
        x = jnp.concatenate([phi_s, phi_s_next], axis=-1)
        x = nn.Dense(features=self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        action_logits = nn.Dense(features=self.num_actions, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return action_logits

class ForwardModel(nn.Module):
    """Predicts the next encoded state given current encoded state and action."""
    latent_dim: int
    hidden_dim: int
    activation: callable

    @nn.compact
    def __call__(self, phi_s: jnp.ndarray, action_one_hot: jnp.ndarray) -> jnp.ndarray:
        # Concatenate the state feature and the one-hot encoded action
        x = jnp.concatenate([phi_s, action_one_hot], axis=-1)
        x = nn.Dense(features=self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation(x)
        predicted_phi_s_next = nn.Dense(features=self.latent_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return predicted_phi_s_next

class ICM(nn.Module):
    """Intrinsic Curiosity Module."""
    num_actions: int
    config: TrainConfig

    def setup(self):
        icm_activation_fn = get_activation(self.config.icm_activation)
        self.encoder = ICMEncoder(latent_dim=self.config.icm_latent_dim,
                                  hidden_dim=self.config.icm_hidden_dim,
                                  activation=icm_activation_fn,
                                  name="ICMEncoder")
        self.inverse_model = InverseModel(num_actions=self.num_actions,
                                          hidden_dim=self.config.icm_hidden_dim,
                                          activation=icm_activation_fn,
                                          name="InverseModel")
        self.forward_model = ForwardModel(latent_dim=self.config.icm_latent_dim,
                                          hidden_dim=self.config.icm_hidden_dim,
                                          activation=icm_activation_fn,
                                          name="ForwardModel")

    def get_features(self, obs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Encode observation."""
        return self.encoder(obs)

    def predict_action(self, phi_s: jnp.ndarray, phi_s_next: jnp.ndarray) -> jnp.ndarray:
        """Predict action logits from state features."""
        return self.inverse_model(phi_s, phi_s_next)

    def predict_next_features(self, phi_s: jnp.ndarray, action_one_hot: jnp.ndarray) -> jnp.ndarray:
        """Predict next state features."""
        return self.forward_model(phi_s, action_one_hot)

    def __call__(self, obs: Dict[str, jnp.ndarray], next_obs: Dict[str, jnp.ndarray], action_one_hot: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Full ICM pass for loss calculation."""
        phi_s = self.get_features(obs)
        phi_s_next = self.get_features(next_obs)

        # Stop gradient for target features in forward loss calculation later
        phi_s_next_target = jax.lax.stop_gradient(phi_s_next)

        action_logits_pred = self.predict_action(phi_s, phi_s_next)
        phi_s_next_pred = self.predict_next_features(phi_s, action_one_hot)

        return phi_s, phi_s_next_target, action_logits_pred, phi_s_next_pred


# --- PPO Agent Logic ---

@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Transition:
    """Stores data for one environment step."""
    obs: Dict[str, jnp.ndarray]
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    next_obs: Dict[str, jnp.ndarray] # Added next_obs

    def tree_flatten(self):
        """Specifies how to flatten the Transition object."""
        # Children are the dynamic JAX types or other PyTrees (like obs dict)
        children = (
            self.obs,
            self.action,
            self.reward,
            self.done,
            self.log_prob,
            self.value,
            self.next_obs,
        )
        # Auxiliary data is static information (none needed here)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies how to reconstruct the Transition object."""
        # Unpack children in the same order as defined in tree_flatten
        (obs, action, reward, done, log_prob, value, next_obs) = children
        return cls(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
            next_obs=next_obs,
        )

@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class TrainState:
    """Consolidated training state."""
    ac_params: Any # Actor-Critic parameters
    ac_opt_state: Any # AC optimizer state
    icm_params: Optional[Any] # ICM parameters (optional)
    icm_opt_state: Optional[Any] # ICM optimizer state (optional)
    obs: Dict[str, jnp.ndarray]
    env_state: Any # Craftax EnvState is already a PyTree
    rng: jax.random.PRNGKey
    update_count: int # Can be treated as a leaf by JAX

    def tree_flatten(self):
        """Specifies how to flatten the TrainState object."""
        # Children are the dynamic JAX types or other PyTrees
        children = (
            self.ac_params,
            self.ac_opt_state,
            self.icm_params,
            self.icm_opt_state,
            self.obs,
            self.env_state,
            self.rng,
            self.update_count # Include update_count as a child
        )
        # Auxiliary data is static information (none needed here)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Specifies how to reconstruct the TrainState object."""
        # Unpack children in the same order as defined in tree_flatten
        (ac_params, ac_opt_state, icm_params, icm_opt_state,
         obs, env_state, rng, update_count) = children
        return cls(
            ac_params=ac_params,
            ac_opt_state=ac_opt_state,
            icm_params=icm_params,
            icm_opt_state=icm_opt_state,
            obs=obs,
            env_state=env_state,
            rng=rng,
            update_count=update_count
        )


def make_train(config: TrainConfig):
    """Creates the training function."""
    env = make_craftax_env_from_name(config.env_name, auto_reset=True)
    env_params = env.default_params
    num_actions = env.action_space(env_params).n

    def linear_schedule(count):
        frac = 1.0 - count / config.num_updates
        return config.learning_rate * frac

    def train(rng):
        # INIT NETWORK (Actor-Critic)
        network = ActorCriticNetwork(num_actions=num_actions, config=config)
        rng, _rng = jax.random.split(rng)

        # Get a dummy observation for initialization
        key_reset = jax.random.PRNGKey(0) # Use fixed key for dummy init
        obs_dummy, _ = env.reset(key_reset, env_params)
        # Craftax obs is a dict, need to create a dummy batch dim
        init_x = jax.tree_util.tree_map(lambda t: jnp.expand_dims(t, axis=0), obs_dummy)
        ac_params = network.init(_rng, init_x)['params']

        # INIT OPTIMIZER (Actor-Critic)
        if config.anneal_lr:
            ac_tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            ac_tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.learning_rate, eps=1e-5),
            )
        ac_opt_state = ac_tx.init(ac_params)

        # INIT ICM NETWORK & OPTIMIZER (if enabled)
        icm_network = None
        icm_params = None
        icm_tx = None
        icm_opt_state = None
        if config.train_icm:
            icm_network = ICM(num_actions=num_actions, config=config)
            rng, _rng_icm = jax.random.split(rng)
            # Dummy inputs for ICM init
            dummy_action_one_hot = jax.nn.one_hot(jnp.zeros((1,), dtype=jnp.int32), num_actions)
            icm_params = icm_network.init(_rng_icm, init_x, init_x, dummy_action_one_hot)['params']

            # ICM uses a separate optimizer (potentially different LR)
            # No LR annealing for ICM optimizer usually
            icm_tx = optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm), # Use same clipping norm? Optional.
                optax.adam(config.icm_lr, eps=1e-5),
            )
            icm_opt_state = icm_tx.init(icm_params)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # INITIAL TRAIN STATE
        initial_update_count = 0
        runner_state = TrainState(
            ac_params=ac_params,
            ac_opt_state=ac_opt_state,
            icm_params=icm_params,
            icm_opt_state=icm_opt_state,
            obs=obsv,
            env_state=env_state,
            rng=rng,
            update_count=initial_update_count
        )


        # TRAIN LOOP
        def _update_step(runner_state: TrainState, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state: TrainState, unused):
                # Unpack runner state
                ac_params = runner_state.ac_params
                obs = runner_state.obs
                env_state = runner_state.env_state
                rng = runner_state.rng

                # SELECT ACTION (Using AC network)
                rng, _rng = jax.random.split(rng)
                logits, value = network.apply({'params': ac_params}, obs)
                pi = distrax.Categorical(logits=logits)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                step_rng = jax.random.split(_rng, config.num_envs)
                obsv_next, env_state_next, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    step_rng, env_state, action, env_params
                )

                # Store transition (including next_obs)
                transition = Transition(obs, action, reward, done, log_prob, value, obsv_next)

                # Update runner state for next step
                runner_state = dataclasses.replace(runner_state,
                    obs=obsv_next, env_state=env_state_next, rng=rng
                )
                return runner_state, transition

            # Run environment steps to collect data
            runner_state_start_rollout = runner_state # Keep track of state before rollout
            runner_state_after_rollout, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # Extract final state info from after rollout
            last_obs = runner_state_after_rollout.obs
            ac_params = runner_state_after_rollout.ac_params # Use potentially updated params if scan modifies them (it shouldn't here)
            icm_params = runner_state_after_rollout.icm_params
            rng = runner_state_after_rollout.rng
            update_count = runner_state_after_rollout.update_count
            ac_opt_state = runner_state_after_rollout.ac_opt_state
            icm_opt_state = runner_state_after_rollout.icm_opt_state

            # CALCULATE INTRINSIC REWARD (if enabled)
            intrinsic_reward = jnp.zeros((config.num_steps, config.num_envs)) # Default zero
            icm_metrics = {"icm_forward_loss": 0.0, "icm_inverse_loss": 0.0, "icm_loss": 0.0, "intrinsic_reward_mean": 0.0}

            if config.train_icm and icm_network is not None:
                # Prepare data for ICM: flatten time and env dims
                def flatten_batch(x):
                    # Tree map for obs dictionaries
                    if isinstance(x, dict):
                        return jax.tree_map(lambda leaf: leaf.reshape((config.batch_size,) + leaf.shape[2:]), x)
                    else:
                        return x.reshape((config.batch_size,) + x.shape[2:])

                obs_flat = flatten_batch(traj_batch.obs)
                next_obs_flat = flatten_batch(traj_batch.next_obs)
                actions_flat = flatten_batch(traj_batch.action)
                action_one_hot_flat = jax.nn.one_hot(actions_flat, num_actions)

                # Apply ICM forward model to get prediction error (intrinsic reward)
                # Note: We only need the forward prediction error for the reward signal
                phi_s = icm_network.apply({'params': icm_params}, obs_flat, method=icm_network.get_features)
                phi_s_next = icm_network.apply({'params': icm_params}, next_obs_flat, method=icm_network.get_features)
                phi_s_next_pred = icm_network.apply({'params': icm_params}, phi_s, action_one_hot_flat, method=icm_network.predict_next_features)

                # Calculate Forward Loss (MSE) - this is the base for intrinsic reward
                # Use stop_gradient on the actual next state features
                forward_error = 0.5 * jnp.sum((phi_s_next_pred - jax.lax.stop_gradient(phi_s_next))**2, axis=-1)

                # Reshape reward back to (num_steps, num_envs)
                intrinsic_reward = forward_error.reshape((config.num_steps, config.num_envs))
                icm_metrics["intrinsic_reward_mean"] = intrinsic_reward.mean() # Log mean before scaling

                # Scale intrinsic reward by coefficient
                intrinsic_reward = intrinsic_reward * config.icm_reward_coeff


            # COMBINE REWARDS
            combined_reward = traj_batch.reward + intrinsic_reward

            # CALCULATE ADVANTAGE (using combined rewards)
            _, last_val = network.apply({'params': ac_params}, last_obs) # Value of final state

            def _calculate_gae(rewards, values, dones, last_val):
                # Reverse order for GAE calculation
                rewards = jnp.flip(rewards, axis=0)
                dones = jnp.flip(dones, axis=0)
                values = jnp.flip(values, axis=0)

                def _gae_step(carry, xs):
                    gae, next_value, next_done = carry
                    reward, done, value = xs

                    done_float = done.astype(jnp.float32)
                    delta = reward + config.gamma * next_value * (1 - done_float) - value
                    gae = delta + config.gamma * config.gae_lambda * (1 - done_float) * gae
                    return (gae, value, done), gae # Carry next value and done mask

                initial_gae = jnp.zeros_like(last_val)
                initial_next_value = last_val
                initial_next_done = jnp.zeros_like(dones[0])
                # GAE calculation scan
                # Initial carry: gae=0, next_value=last_val, next_done=last_done (assumed 0 if auto-reset)
                # For simplicity with auto-reset, we often assume last_done is 0,
                # or use the actual done from the last step of the trajectory if needed.
                # Here, using 0 like stable-baselines3 for simplicity.
                _, advantages = jax.lax.scan(
                    _gae_step,
                    (initial_gae, initial_next_value, initial_next_done), # Initial (gae, next_val, next_done)
                    (rewards, dones, values), # Inputs per step
                    length=config.num_steps
                )
                advantages = jnp.flip(advantages, axis=0) # Flip back to normal order
                targets = advantages + jnp.flip(values, axis=0) # Calculate returns (value targets)
                return advantages, targets

            advantages, targets = _calculate_gae(combined_reward, traj_batch.value, traj_batch.done, last_val)


            # UPDATE NETWORK (PPO + ICM)
            def _update_epoch(update_state, unused):
                ac_params, ac_opt_state, icm_params, icm_opt_state, traj_batch, advantages, targets, rng, update_count = update_state

                # Flatten and Shuffle Data
                batch_size = config.batch_size
                mb_size = config.minibatch_size
                assert batch_size % mb_size == 0
                num_minibatches = config.num_minibatches

                def flatten_permute(rng_key_in, arr):
                    if isinstance(arr, dict):
                        # Flatten leaves, permute each, then reconstruct dict
                        leaves, treedef = jax.tree_util.tree_flatten(arr)
                        flat_leaves = [leaf.reshape((batch_size,) + leaf.shape[2:]) for leaf in leaves]
                        perm_key, next_rng_key = jax.random.split(rng_key_in)
                        permutation = jax.random.permutation(perm_key, batch_size)
                        shuffled_leaves = [leaf[permutation] for leaf in flat_leaves]
                        shuffled_arr = jax.tree_util.tree_unflatten(treedef, shuffled_leaves)
                        return shuffled_arr, next_rng_key
                    else:
                        # Standard array case
                        arr = arr.reshape((batch_size,) + arr.shape[2:])
                        perm_key, next_rng_key = jax.random.split(rng_key_in)
                        permutation = jax.random.permutation(perm_key, batch_size)
                        shuffled_arr = arr[permutation]
                        return shuffled_arr, next_rng_key

                rng, _rng = jax.random.split(rng)
                flat_obs, rng = flatten_permute(_rng, traj_batch.obs)
                flat_next_obs, rng = flatten_permute(rng, traj_batch.next_obs) # Need next_obs for ICM
                flat_act, rng = flatten_permute(rng, traj_batch.action)
                flat_log_prob, rng = flatten_permute(rng, traj_batch.log_prob)
                flat_adv, rng = flatten_permute(rng, advantages)
                flat_targets, rng = flatten_permute(rng, targets)

                # Minibatch Update Loop (using lax.scan over minibatches)
                def _update_minibatch(train_states, batch_indices):
                    ac_params, ac_opt_state, icm_params, icm_opt_state, update_count = train_states

                    # Extract minibatch data using indices
                    def get_minibatch(data, indices):
                        if isinstance(data, dict):
                             return jax.tree_map(lambda leaf: leaf[indices], data)
                        else:
                             return data[indices]

                    start_idx = batch_indices * mb_size
                    idx = jax.lax.dynamic_slice_in_dim(jnp.arange(batch_size), start_idx, mb_size)

                    obs_mb = get_minibatch(flat_obs, idx)
                    next_obs_mb = get_minibatch(flat_next_obs, idx) # ICM needs this
                    act_mb = get_minibatch(flat_act, idx)
                    log_prob_old_mb = get_minibatch(flat_log_prob, idx)
                    adv_mb = get_minibatch(flat_adv, idx)
                    targets_mb = get_minibatch(flat_targets, idx)

                    # --- PPO Loss Calculation ---
                    def _ppo_loss_fn(params, obs_mb, act_mb, log_prob_old_mb, adv_mb, targets_mb):
                        logits, value_pred = network.apply({'params': params}, obs_mb)
                        pi = distrax.Categorical(logits=logits)
                        log_prob_new = pi.log_prob(act_mb)
                        ratio = jnp.exp(log_prob_new - log_prob_old_mb)

                        # Advantage normalization (per minibatch)
                        adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                        # Clipped Surrogate Objective
                        pg_loss1 = adv_mb * ratio
                        pg_loss2 = adv_mb * jnp.clip(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
                        pg_loss = -jnp.minimum(pg_loss1, pg_loss2).mean()

                        # Value Loss
                        v_loss = 0.5 * ((value_pred - targets_mb)**2).mean()

                        # Entropy Loss
                        entropy = pi.entropy().mean()

                        total_loss = (pg_loss
                                    + config.vf_coef * v_loss
                                    - config.ent_coef * entropy)
                        return total_loss, (pg_loss, v_loss, entropy)

                    ppo_grad_fn = jax.value_and_grad(_ppo_loss_fn, has_aux=True)
                    (ppo_loss, ppo_aux_losses), ppo_grads = ppo_grad_fn(
                        ac_params, obs_mb, act_mb, log_prob_old_mb, adv_mb, targets_mb
                    )
                    # Apply PPO updates
                    ppo_updates, new_ac_opt_state = ac_tx.update(ppo_grads, ac_opt_state, ac_params)
                    new_ac_params = optax.apply_updates(ac_params, ppo_updates)


                    # --- ICM Loss Calculation & Update (if enabled) ---
                    new_icm_params = icm_params # Pass through if not training ICM
                    new_icm_opt_state = icm_opt_state
                    icm_loss = 0.0
                    icm_forward_loss = 0.0
                    icm_inverse_loss = 0.0

                    if config.train_icm and icm_network is not None and icm_params is not None:
                        def _icm_loss_fn(params_icm, obs_mb, next_obs_mb, act_mb):
                            action_one_hot_mb = jax.nn.one_hot(act_mb, num_actions)

                            # Apply full ICM pass
                            phi_s, phi_s_next_target, action_logits_pred, phi_s_next_pred = icm_network.apply(
                                {'params': params_icm}, obs_mb, next_obs_mb, action_one_hot_mb
                            )

                            # Inverse Dynamics Loss (predict action)
                            inv_loss = optax.softmax_cross_entropy_with_integer_labels(
                                action_logits_pred, act_mb
                            ).mean()

                            # Forward Dynamics Loss (predict next state feature)
                            # Target features phi_s_next_target already have stop_gradient applied inside __call__
                            fwd_loss = 0.5 * jnp.mean(jnp.sum((phi_s_next_pred - phi_s_next_target)**2, axis=-1))


                            # Total ICM Loss
                            total_icm_loss = (config.icm_inverse_loss_coef * inv_loss
                                            + config.icm_forward_loss_coef * fwd_loss)

                            return total_icm_loss, (fwd_loss, inv_loss)

                        icm_grad_fn = jax.value_and_grad(_icm_loss_fn, has_aux=True)
                        (icm_loss, (icm_forward_loss, icm_inverse_loss)), icm_grads = icm_grad_fn(
                            icm_params, obs_mb, next_obs_mb, act_mb
                        )

                        # Apply ICM updates
                        icm_updates, new_icm_opt_state = icm_tx.update(icm_grads, icm_opt_state, icm_params)
                        new_icm_params = optax.apply_updates(icm_params, icm_updates)

                    # Combine losses for logging
                    losses = (ppo_loss, ppo_aux_losses[0], ppo_aux_losses[1], ppo_aux_losses[2], icm_loss, icm_forward_loss, icm_inverse_loss)
                    # Return updated states and losses
                    train_states = (new_ac_params, new_ac_opt_state, new_icm_params, new_icm_opt_state, update_count)
                    return train_states, losses

                # Scan over minibatches
                initial_train_states = (ac_params, ac_opt_state, icm_params, icm_opt_state, update_count)
                (final_ac_params, final_ac_opt_state, final_icm_params, final_icm_opt_state, _), mb_losses = jax.lax.scan(
                    _update_minibatch, initial_train_states, jnp.arange(num_minibatches)
                )

                update_state = (final_ac_params, final_ac_opt_state, final_icm_params, final_icm_opt_state, traj_batch, advantages, targets, rng, update_count)
                return update_state, mb_losses # Return losses per minibatch

            # Scan over update epochs
            update_state_start_epochs = (ac_params, ac_opt_state, icm_params, icm_opt_state, traj_batch, advantages, targets, rng, update_count)
            update_state_after_epochs, epoch_losses = jax.lax.scan(
                _update_epoch, update_state_start_epochs, None, config.update_epochs
            )

            # Extract final state after epochs
            ac_params, ac_opt_state, icm_params, icm_opt_state, _, _, _, rng, update_count = update_state_after_epochs
            update_count += 1 # Increment update counter

            # --- Logging ---
            # Average losses over epochs and minibatches
            # epoch_losses shape: (update_epochs, num_minibatches, num_losses)
            avg_losses = jax.tree_util.tree_map(lambda x: x.mean(), epoch_losses)

            # Put back into runner state
            # Use the updated state after rollouts and updates
            runner_state = TrainState(
                 ac_params=ac_params,
                 ac_opt_state=ac_opt_state,
                 icm_params=icm_params,
                 icm_opt_state=icm_opt_state,
                 obs=runner_state_after_rollout.obs, # Use the obs from after the rollout
                 env_state=runner_state_after_rollout.env_state, # Use env_state from after rollout
                 rng=rng,
                 update_count=update_count
            )

            # Collect metrics
            metrics = {
                "total_ppo_loss": avg_losses[0],
                "pg_loss": avg_losses[1],
                "value_loss": avg_losses[2],
                "entropy_loss": avg_losses[3],
                "advantages_mean": advantages.mean(),
                "targets_mean": targets.mean(),
                "reward_extrinsic_mean": traj_batch.reward.mean(), # Log mean extrinsic reward
                **icm_metrics # Add ICM specific metrics computed earlier or during update
            }
            # Update ICM metrics from the averaged minibatch losses
            if config.train_icm:
                metrics["icm_loss"] = avg_losses[4]
                metrics["icm_forward_loss"] = avg_losses[5]
                metrics["icm_inverse_loss"] = avg_losses[6]


            # Log progress periodically using jax.debug.callback for JIT compatibility
            def log_fn(metrics, update_count):
                print(f"Update: {update_count}/{config.num_updates}, Timesteps: {update_count*config.batch_size}")
                print(f"  PPO Loss: {metrics['total_ppo_loss']:.4f} (PG: {metrics['pg_loss']:.4f}, V: {metrics['value_loss']:.4f}, E: {metrics['entropy_loss']:.4f})")
                if config.train_icm:
                     print(f"  ICM Loss: {metrics['icm_loss']:.4f} (Fwd: {metrics['icm_forward_loss']:.4f}, Inv: {metrics['icm_inverse_loss']:.4f})")
                     print(f"  Reward Mean (Ext): {metrics['reward_extrinsic_mean']:.4f}, Reward Mean (Int): {metrics['intrinsic_reward_mean']:.6f}")
                else:
                     print(f"  Reward Mean (Ext): {metrics['reward_extrinsic_mean']:.4f}")
                # print(f"  Adv Mean: {metrics['advantages_mean']:.2f}, Target Mean: {metrics['targets_mean']:.2f}")

            jax.lax.cond(
                (update_count-1) % config.log_frequency == 0, # Log at the *end* of the update step
                lambda: jax.debug.callback(log_fn, metrics, update_count),
                lambda: None,
            )

            return runner_state, metrics

        # Main training loop compiled with JIT
        # Scan over the number of updates
        runner_state_final, metrics_all = jax.lax.scan(
            _update_step, runner_state, None, config.num_updates
        )

        return {"runner_state": runner_state_final, "metrics": metrics_all}

    return train

# --- Main Execution ---
if __name__ == "__main__":
    config = TrainConfig()

    # If using ICM, ensure config settings are appropriate
    if not config.train_icm:
        print("ICM training is DISABLED.")
    else:
        print("ICM training is ENABLED.")
        print(f"  ICM Reward Coeff: {config.icm_reward_coeff}")
        print(f"  ICM LR: {config.icm_lr}")
        print(f"  ICM Latent Dim: {config.icm_latent_dim}")

    print("\nStarting training...")
    print(f"Config: {config}")
    print(f"Total Timesteps: {config.total_timesteps}")
    print(f"Num Updates: {config.num_updates}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Minibatch Size: {config.minibatch_size}")
    print("-" * 30)

    rng = jax.random.PRNGKey(config.seed)
    train_fn = make_train(config)

    # JIT compile the training function
    print("JIT Compiling training function...")
    start_compile_time = time.time()
    # Wrap train_fn before JIT if there are complex static arguments or closures
    # In this case, config is captured, which is fine for JIT.
    train_fn_jit = jax.jit(train_fn)
    # Optional: Run a single step to force compilation (can take time)
    # print("Running one step for compilation...")
    # output_compile = train_fn_jit.lower(rng).compile() # Lower and compile explicitly
    # output_example = train_fn_jit(rng) # Or just run it once
    print(f"JIT Compilation setup initiated. Actual compilation happens on first run.")
    # print(f"JIT Compilation attempt complete. Time: {time.time() - start_compile_time:.2f}s")


    print("Starting actual training run...")
    start_train_time = time.time()
    output = train_fn_jit(rng) # Run the compiled function
    # Ensure computations complete before timing end for async JAX backends
    jax.block_until_ready(output)
    total_train_time = time.time() - start_train_time
    print("-" * 30)
    print(f"Training finished.")
    print(f"Total Training Time: {total_train_time:.2f}s")

    # Can save the final runner state (params, etc.) here if needed
    final_ac_params = output['runner_state'].ac_params
    if config.train_icm:
        final_icm_params = output['runner_state'].icm_params
        print("Final Actor-Critic and ICM parameters obtained.")
    else:
        print("Final Actor-Critic parameters obtained.")
    