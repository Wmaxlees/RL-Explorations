# models/actor_critic.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Tuple, Callable, Optional

import distrax

try:
    from .jepa import JEPA, JEPAVector
    from .jepa import ForwardDynamicsAuxiliary
    from .jepa import InversePredictorMLP
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


class ActorCriticICMIntegrated(nn.Module): # Renamed for clarity
    """
    Actor-Critic network using an encoder trained with integrated
    Forward Dynamics and Inverse Dynamics auxiliary losses (ICM style).
    Works with flat vector observations.
    """
    action_dim: int
    layer_width: int

    # Encoder/Forward Dynamics parameters
    fd_input_dim: int
    fd_encoder_output_dim: int
    fd_encoder_hidden_dim: int
    fd_encoder_layers: int
    fd_predictor_hidden_dim: int
    fd_predictor_layers: int

    # Inverse Dynamics parameters (can share hidden dims or be different)
    id_hidden_dim: int = 512
    id_layers: int = 2

    activation_fn: Callable = nn.relu


    def setup(self):
        if ForwardDynamicsAuxiliary is None or InversePredictorMLP is None:
            raise ImportError("Required auxiliary model classes not available.")

        # Instantiate Forward Dynamics module (provides Encoder + Forward Predictor)
        self.forward_module = ForwardDynamicsAuxiliary(
            input_dim=self.fd_input_dim,
            action_dim=self.action_dim,
            encoder_output_dim=self.fd_encoder_output_dim,
            encoder_hidden_dim=self.fd_encoder_hidden_dim,
            encoder_layers=self.fd_encoder_layers,
            predictor_hidden_dim=self.fd_predictor_hidden_dim,
            predictor_layers=self.fd_predictor_layers,
            name="forward_dynamics_module")

        # Instantiate Inverse Dynamics predictor
        self.inverse_predictor = InversePredictorMLP(
            action_dim=self.action_dim,
            embedding_dim=self.fd_encoder_output_dim,
            hidden_dim=self.id_hidden_dim,
            num_layers=self.id_layers,
            name="inverse_predictor_module"
        )

        # --- Actor/Critic Heads ---
        self.actor_fc1 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_fc1")
        self.actor_fc2 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="actor_fc2")
        self.actor_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor_logits")
        self.critic_fc1 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_fc1")
        self.critic_fc2 = nn.Dense(self.layer_width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="critic_fc2")
        self.critic_value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_value")

    # --- Helper Methods ---
    def get_embedding(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Applies the encoder."""
        return self.forward_module.get_embedding(obs)

    def get_policy_and_value(self, obs: jnp.ndarray):
         """Gets policy and value from observation."""
         embedding_t = self.get_embedding(obs)
         # Actor head
         actor_x = self.actor_fc1(embedding_t); actor_x = self.activation_fn(actor_x)
         actor_x = self.actor_fc2(actor_x); actor_x = self.activation_fn(actor_x)
         actor_logits = self.actor_logits(actor_x)
         pi = distrax.Categorical(logits=actor_logits)
         # Critic head
         critic_x = self.critic_fc1(embedding_t); critic_x = self.activation_fn(critic_x)
         critic_x = self.critic_fc2(critic_x); critic_x = self.activation_fn(critic_x)
         critic_output = self.critic_value(critic_x)
         value = jnp.squeeze(critic_output, axis=-1)
         return pi, value

    def get_forward_loss_and_error(self, obs_t: jnp.ndarray, action_t: jnp.ndarray, obs_t_plus_1: jnp.ndarray, target_encoder_params: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
         """Calls the forward dynamics module."""
         return self.forward_module(obs_t, action_t, obs_t_plus_1, target_encoder_params)

    def get_inverse_logits(self, obs_t: jnp.ndarray, obs_t_plus_1: jnp.ndarray) -> jnp.ndarray:
         """Calculates inverse dynamics logits."""
         # Use ONLINE encoder for both embeddings
         embedding_t = self.get_embedding(obs_t)
         embedding_t_plus_1 = self.get_embedding(obs_t_plus_1)
         return self.inverse_predictor(embedding_t, embedding_t_plus_1)
    
    def get_auxiliary_outputs(self, obs_t: jnp.ndarray, action_t: jnp.ndarray, obs_t_plus_1: jnp.ndarray, target_encoder_params: dict):
        """Calculates forward loss, raw error, and inverse logits."""
        # Get forward loss and raw error
        forward_loss, raw_forward_error = self.forward_module(obs_t, action_t, obs_t_plus_1, target_encoder_params)

        # Get embeddings needed for inverse model (using ONLINE encoder)
        embedding_t = self.forward_module.get_embedding(obs_t)
        embedding_t_plus_1 = self.forward_module.get_embedding(obs_t_plus_1) # ONLINE encoder

        # Get inverse model prediction
        inverse_logits = self.inverse_predictor(embedding_t, embedding_t_plus_1)

        return forward_loss, raw_forward_error, inverse_logits

    def __call__(self, obs_t: jnp.ndarray, # Always required
                       action_t: Optional[jnp.ndarray] = None,
                       obs_t_plus_1: Optional[jnp.ndarray] = None,
                       target_encoder_params: Optional[dict] = None,
                       calculate_aux: bool = False): # Flag to compute aux losses
        """
        Main apply function. Returns policy and value.
        If calculate_aux is True and other inputs are provided, also computes aux losses/logits.
        """
        pi, value = self.get_policy_and_value(obs_t)

        # --- Always compute Inverse Logits if possible (ensures init) ---
        # Initialize placeholder
        inverse_logits = jnp.zeros((obs_t.shape[0], self.action_dim))
        if obs_t_plus_1 is not None:
             # This ensures self.inverse_predictor is traced during init if dummy obs_t_plus_1 is passed
             inverse_logits = self.get_inverse_logits(obs_t, obs_t_plus_1)

        # --- Conditionally compute Forward Loss ---
        forward_loss = 0.0 # Placeholder
        if calculate_aux:
            if action_t is None or obs_t_plus_1 is None or target_encoder_params is None:
                raise ValueError("Missing inputs required for auxiliary loss calculation.")
            # Calculate Forward Loss (only loss value needed for return signature)
            forward_loss, _ = self.get_forward_loss_and_error(obs_t, action_t, obs_t_plus_1, target_encoder_params)
            # Note: Inverse logits are already computed above if obs_t_plus_1 was provided
            # If we only run inverse when calculate_aux is True, uncomment the line below
            # and comment out the block above that calculates inverse_logits
            # inverse_logits = self.get_inverse_logits(obs_t, obs_t_plus_1)

        # Always return the same structure
        return pi, value, forward_loss, inverse_logits
