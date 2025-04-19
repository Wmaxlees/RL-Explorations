# models/actor_critic.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Tuple, Callable

import distrax

try:
    from .jepa import JEPA, JEPAVector
    from .jepa import ForwardDynamicsAuxiliary
except ImportError:
    print("Warning: Could not import JEPA/JEPAVector/ForwardDynamicsAuxiliary models.")
    JEPA = None
    JEPAVector = None
    ForwardDynamicsAuxiliary = None


# --- Original ActorCriticConvSymbolicCraftax ---
class ActorCriticConvSymbolicCraftax(nn.Module):
    action_dim: Sequence[int]
    map_obs_shape: Sequence[int]
    layer_width: int

    @nn.compact
    def __call__(self, obs):
        # Split into map and flat obs
        flat_map_obs_shape = (
            self.map_obs_shape[0] * self.map_obs_shape[1] * self.map_obs_shape[2]
        )
        image_obs = obs[:, :flat_map_obs_shape]
        image_dim = self.map_obs_shape
        image_obs = image_obs.reshape((image_obs.shape[0], *image_dim))

        flat_obs = obs[:, flat_map_obs_shape:]

        # Convolutions on map
        image_embedding = nn.Conv(features=32, kernel_size=(2, 2))(image_obs)
        image_embedding = nn.relu(image_embedding)
        image_embedding = nn.max_pool(
            image_embedding, window_shape=(2, 2), strides=(1, 1)
        )
        image_embedding = nn.Conv(features=32, kernel_size=(2, 2))(image_embedding)
        image_embedding = nn.relu(image_embedding)
        image_embedding = nn.max_pool(
            image_embedding, window_shape=(2, 2), strides=(1, 1)
        )
        image_embedding = image_embedding.reshape(image_embedding.shape[0], -1)

        # Combine embeddings
        embedding = jnp.concatenate([image_embedding, flat_obs], axis=-1)
        embedding = nn.Dense(
            self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding) # Corrected orthogonal init scale
        embedding = nn.relu(embedding)

        # Actor Head
        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        # Removed extra dense layers + relu from actor head for simplicity, following common patterns
        pi = distrax.Categorical(logits=actor_mean)

        # Critic Head
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        # Removed extra dense layers + relu from critic head for simplicity
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


# --- Original ActorCriticConv ---
class ActorCriticConv(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation_fn: Callable = nn.relu # Use relu consistent with JEPAEncoder

    @nn.compact
    def __call__(self, obs):
        if obs.ndim != 4:
             raise ValueError(f"ActorCriticConv expects image input shape (B, H, W, C), got {obs.shape}")

        x = nn.Conv(features=32, kernel_size=(5, 5), name="conv1")(obs)
        x = self.activation_fn(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3), name="pool1")
        x = nn.Conv(features=64, kernel_size=(5, 5), name="conv2")(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3), name="pool2")
        x = nn.Conv(features=64, kernel_size=(3, 3), name="conv3")(x)
        x = self.activation_fn(x)
        embedding = x.reshape(x.shape[0], -1) # Flatten

        # Actor Head
        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_fc1"
        )(embedding)
        actor_mean = self.activation_fn(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor_logits"
        )(actor_mean)
        # Removed extra dense layers for consistency
        pi = distrax.Categorical(logits=actor_mean)

        # Critic Head
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_fc1"
        )(embedding)
        critic = self.activation_fn(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_value")(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

# --- Original ActorCritic (for flat observations) ---
class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh" # Keep original option here

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "tanh":
            activation_fn = nn.tanh
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        # Actor Head (3 hidden layers)
        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_fc1"
        )(x)
        actor_mean = activation_fn(actor_mean)
        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_fc2"
        )(actor_mean)
        actor_mean = activation_fn(actor_mean)
        actor_mean = nn.Dense( # Added third hidden layer to match original structure
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor_fc3"
        )(actor_mean)
        actor_mean = activation_fn(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor_logits"
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic Head (3 hidden layers)
        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_fc1"
        )(x)
        critic = activation_fn(critic)
        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_fc2"
        )(critic)
        critic = activation_fn(critic)
        critic = nn.Dense( # Added third hidden layer to match original structure
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic_fc3"
        )(critic)
        critic = activation_fn(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_value")(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


# --- Original ActorCriticWithEmbedding (Unused in main.py but kept for completeness) ---
class ActorCriticWithEmbedding(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "tanh":
             activation_fn = nn.tanh
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")


        actor_emb = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_emb = activation_fn(actor_emb)
        actor_emb = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_emb)
        actor_emb = activation_fn(actor_emb)
        actor_emb = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0) # Fixed embedding size
        )(actor_emb)
        actor_emb = activation_fn(actor_emb)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_emb)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic uses original input x
        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation_fn(critic)
        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation_fn(critic)
        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation_fn(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), actor_emb


# --- NEW Actor Critic using JEPA Embedding ---
class ActorCriticJEPA(nn.Module):
    """
    Actor-Critic network using JEPA encoder for feature extraction from images.
    Includes JEPA loss calculation as part of its forward pass.
    """
    action_dim: Sequence[int]
    layer_width: int # Width of the dense layers in actor/critic heads
    # JEPA specific parameters needed to instantiate the JEPA module
    jepa_encoder_output_dim: int
    jepa_predictor_hidden_dim: int
    jepa_predictor_layers: int
    jepa_image_shape: Tuple[int, int, int] # (H, W, C)

    def setup(self):
        if JEPA is None:
            raise ImportError("JEPA model class not available. Please ensure models/jepa.py exists and is importable.")

        # Instantiate the JEPA model internally
        self.jepa = JEPA(encoder_output_dim=self.jepa_encoder_output_dim,
                         predictor_hidden_dim=self.jepa_predictor_hidden_dim,
                         predictor_layers=self.jepa_predictor_layers,
                         image_shape=self.jepa_image_shape,
                         name="jepa_module") # Give the submodule a name

        # Define Actor layers (using JEPA embedding as input)
        # Using relu consistent with JEPA encoder's default activation
        self.actor_fc1 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_fc1")
        self.actor_fc2 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_fc2")
        self.actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor_logits")

        # Define Critic layers (using JEPA embedding as input)
        self.critic_fc1 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_fc1")
        self.critic_fc2 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_fc2")
        self.critic_value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_value")

    def __call__(self, obs: jnp.ndarray, target_encoder_params: dict, rng: jax.random.PRNGKey):
        """
        Forward pass for ActorCriticJEPA. Calculates policy, value, and JEPA loss.

        Args:
            obs: Observations, expected shape (B, H, W, C).
            target_encoder_params: Flax parameter PyTree for the JEPA target encoder.
            rng: JAX PRNGKey for JEPA masking.

        Returns:
            Tuple: (policy_distribution, value_prediction, jepa_loss)
        """
        if obs.ndim != 4:
             raise ValueError(f"ActorCriticJEPA expects image input shape (B, H, W, C), got {obs.shape}")

        rng, jepa_rng = jax.random.split(rng)

        # 1. Get JEPA embedding (from online encoder) for policy/value calculation
        # We use the full observation for the policy/value heads
        embedding = self.jepa.get_embedding(obs)

        # 2. Calculate JEPA loss using the JEPA module's __call__ method
        # This requires the target encoder parameters and an RNG key
        jepa_loss, _ = self.jepa(obs, target_encoder_params, jepa_rng)

        # --- Actor head ---
        actor_x = self.actor_fc1(embedding)
        actor_x = nn.relu(actor_x)
        actor_x = self.actor_fc2(actor_x)
        actor_x = nn.relu(actor_x)
        actor_logits = self.actor_logits(actor_x)
        pi = distrax.Categorical(logits=actor_logits)

        # --- Critic head ---
        critic_x = self.critic_fc1(embedding)
        critic_x = nn.relu(critic_x)
        critic_x = self.critic_fc2(critic_x)
        critic_x = nn.relu(critic_x)
        critic_output = self.critic_value(critic_x)

        return pi, jnp.squeeze(critic_output, axis=-1), jepa_loss
    
class ActorCriticJEPAVector(nn.Module):
    """
    Actor-Critic network using JEPAVector encoder for feature extraction from flat vectors.
    Includes JEPAVector loss calculation as part of its forward pass.
    """
    action_dim: int
    layer_width: int # Width of the dense layers in actor/critic heads
    # JEPAVector specific parameters needed to instantiate the JEPAVector module
    jepa_input_dim: int
    jepa_encoder_output_dim: int
    jepa_encoder_hidden_dim: int
    jepa_encoder_layers: int
    jepa_predictor_hidden_dim: int
    jepa_predictor_layers: int
    jepa_mask_prob: float
    activation_fn: Callable = nn.relu # Activation for AC heads

    def setup(self):
        if JEPAVector is None:
            raise ImportError("JEPAVector model class not available. Please ensure models/jepa.py exists and is importable.")

        # Instantiate the JEPAVector model internally
        self.jepa_vector = JEPAVector(input_dim=self.jepa_input_dim,
                                      encoder_output_dim=self.jepa_encoder_output_dim,
                                      encoder_hidden_dim=self.jepa_encoder_hidden_dim,
                                      encoder_layers=self.jepa_encoder_layers,
                                      predictor_hidden_dim=self.jepa_predictor_hidden_dim,
                                      predictor_layers=self.jepa_predictor_layers,
                                      mask_prob=self.jepa_mask_prob,
                                      name="jepa_vector_module") # Give the submodule a name

        # Define Actor layers (using JEPA embedding as input) - standard MLP head
        self.actor_fc1 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_fc1")
        self.actor_fc2 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_fc2")
        self.actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor_logits")

        # Define Critic layers (using JEPA embedding as input) - standard MLP head
        self.critic_fc1 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_fc1")
        self.critic_fc2 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_fc2")
        self.critic_value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_value")

    def __call__(self, obs: jnp.ndarray, target_encoder_params: dict, rng: jax.random.PRNGKey):
        """
        Forward pass for ActorCriticJEPAVector. Calculates policy, value, and JEPA loss.

        Args:
            obs: Observations, expected shape (B, input_dim).
            target_encoder_params: Flax parameter PyTree for the JEPA target encoder.
            rng: JAX PRNGKey for JEPA masking.

        Returns:
            Tuple: (policy_distribution, value_prediction, jepa_loss)
        """
        if obs.ndim != 2:
             raise ValueError(f"ActorCriticJEPAVector expects flat vector input shape (B, input_dim), got {obs.shape}")

        rng, jepa_rng = jax.random.split(rng)

        # 1. Get JEPA embedding (from online encoder) for policy/value calculation
        embedding = self.jepa_vector.get_embedding(obs)

        # 2. Calculate JEPA loss using the JEPAVector module's __call__ method
        jepa_loss, _ = self.jepa_vector(obs, target_encoder_params, jepa_rng)

        # --- Actor head ---
        actor_x = self.actor_fc1(embedding)
        actor_x = self.activation_fn(actor_x)
        actor_x = self.actor_fc2(actor_x)
        actor_x = self.activation_fn(actor_x)
        actor_logits = self.actor_logits(actor_x)
        pi = distrax.Categorical(logits=actor_logits)

        # --- Critic head ---
        critic_x = self.critic_fc1(embedding)
        critic_x = self.activation_fn(critic_x)
        critic_x = self.critic_fc2(critic_x)
        critic_x = self.activation_fn(critic_x)
        critic_output = self.critic_value(critic_x)

        return pi, jnp.squeeze(critic_output, axis=-1), jepa_loss


class ActorCriticForwardDynamics(nn.Module):
    """
    Actor-Critic network using an encoder trained with a forward dynamics auxiliary loss.
    Works with flat vector observations.
    """
    action_dim: int # Env action dim
    layer_width: int # Width of AC MLP heads

    # Forward Dynamics specific parameters
    fd_input_dim: int
    fd_encoder_output_dim: int
    fd_encoder_hidden_dim: int
    fd_encoder_layers: int
    fd_predictor_hidden_dim: int
    fd_predictor_layers: int

    activation_fn: Callable = nn.relu


    def setup(self):
        if ForwardDynamicsAuxiliary is None:
            raise ImportError("ForwardDynamicsAuxiliary model class not available.")

        # Instantiate the ForwardDynamicsAuxiliary module
        self.forward_dynamics = ForwardDynamicsAuxiliary(
            input_dim=self.fd_input_dim,
            action_dim=self.action_dim, # Pass action_dim here
            encoder_output_dim=self.fd_encoder_output_dim,
            encoder_hidden_dim=self.fd_encoder_hidden_dim,
            encoder_layers=self.fd_encoder_layers,
            predictor_hidden_dim=self.fd_predictor_hidden_dim,
            predictor_layers=self.fd_predictor_layers,
            name="forward_dynamics_module")

        # Define Actor/Critic MLP heads (same as ActorCritic / ActorCriticJEPAVector)
        self.actor_fc1 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_fc1")
        self.actor_fc2 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_fc2")
        self.actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor_logits")
        self.critic_fc1 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_fc1")
        self.critic_fc2 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_fc2")
        self.critic_value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_value")

    def __call__(self, obs_t: jnp.ndarray, action_t: jnp.ndarray, obs_t_plus_1: jnp.ndarray, target_encoder_params: dict):
        """
        Forward pass. Calculates policy, value, and forward dynamics loss.

        Args:
            obs_t: Current observations (B, input_dim).
            action_t: Actions taken in obs_t (B,).
            obs_t_plus_1: Next observations (B, input_dim).
            target_encoder_params: Parameters for the target encoder.

        Returns:
            Tuple: (policy_distribution, value_prediction, forward_dynamics_loss)
        """
        if obs_t.ndim != 2:
             raise ValueError(f"{type(self).__name__} expects flat vector obs_t (B, input_dim), got {obs_t.shape}")

        # 1. Get embedding of current state for policy/value heads
        embedding_t = self.forward_dynamics.get_embedding(obs_t)

        # 2. Calculate Forward Dynamics loss using the auxiliary module
        # This requires obs_t, action_t, obs_t_plus_1, and target params
        forward_loss = self.forward_dynamics(obs_t, action_t, obs_t_plus_1, target_encoder_params)

        # --- Actor head (uses current embedding_t) ---
        actor_x = self.actor_fc1(embedding_t)
        actor_x = self.activation_fn(actor_x)
        actor_x = self.actor_fc2(actor_x)
        actor_x = self.activation_fn(actor_x)
        actor_logits = self.actor_logits(actor_x)
        pi = distrax.Categorical(logits=actor_logits)

        # --- Critic head (uses current embedding_t) ---
        critic_x = self.critic_fc1(embedding_t)
        critic_x = self.activation_fn(critic_x)
        critic_x = self.critic_fc2(critic_x)
        critic_x = self.activation_fn(critic_x)
        critic_output = self.critic_value(critic_x)

        return pi, jnp.squeeze(critic_output, axis=-1), forward_loss
