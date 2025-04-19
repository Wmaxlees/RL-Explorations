# models/jepa.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Tuple, Callable
import random # Using standard random for simple masking logic

class JEPAEncoder(nn.Module):
    """Simple CNN Encoder for JEPA."""
    output_dim: int
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        # Assuming input x is image shaped (B, H, W, C)
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = self.activation(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=64, kernel_size=(5, 5))(x)
        x = self.activation(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x) # Adjusted kernel size
        x = self.activation(x)
        # Removed final max pool to retain more spatial info if needed, or flatten
        x = x.reshape(x.shape[0], -1) # Flatten
        x = nn.Dense(features=self.output_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        return x

class Predictor(nn.Module):
    """Predicts target embedding from context embedding."""
    output_dim: int
    hidden_dim: int = 512
    num_layers: int = 3
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, context_embedding):
        x = context_embedding
        for _ in range(self.num_layers - 1):
            x = nn.Dense(features=self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = self.activation(x)
        x = nn.Dense(features=self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return x

class JEPA(nn.Module):
    """JEPA Model."""
    encoder_output_dim: int
    predictor_hidden_dim: int
    predictor_layers: int
    image_shape: Tuple[int, int, int] # (H, W, C)
    mask_scale: Tuple[float, float] = (0.85, 1.0) # Range for context mask size
    mask_ratio: Tuple[float, float] = (0.15, 0.2) # Range for target mask aspect ratio

    def setup(self):
        self.encoder = JEPAEncoder(output_dim=self.encoder_output_dim)
        # Target encoder uses the same architecture but separate parameters (handled via EMA)
        self.predictor = Predictor(output_dim=self.encoder_output_dim,
                                   hidden_dim=self.predictor_hidden_dim,
                                   num_layers=self.predictor_layers)

    def _generate_masks(self, batch_size: int, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generates random context and target masks for a batch."""
        # Note: This is a simplified masking strategy. More sophisticated ones exist.
        # This implementation uses numpy random *during definition* for simplicity,
        # which isn't ideal for pure JAX. A pure JAX implementation would be more complex.
        # For this example, we'll assume a simple block masking.
        # We'll generate ONE mask pair and apply it across the batch for simplicity in JAX.
        # A VMAP approach would be needed for per-sample masks within JAX.

        H, W, C = self.image_shape
        context_mask = jnp.ones((H, W, 1), dtype=jnp.float32)
        target_mask = jnp.zeros((H, W, 1), dtype=jnp.float32)

        # --- Simple Block Masking (using standard random, not ideal for JAX purity) ---
        # This part won't be traced by JAX correctly if random changes per call.
        # A better way involves passing RNG keys and using jax.random for indexing.
        # For this example's scope, we keep it simple. A real implementation
        # would need a JAX-native masking function.

        # Target block
        log_aspect_ratio = (np.log(self.mask_ratio[0]), np.log(self.mask_ratio[1]))
        aspect_ratio = np.exp(random.uniform(*log_aspect_ratio))
        target_h = int(round(np.sqrt(self.mask_scale[0] * H * W * aspect_ratio)))
        target_w = int(round(np.sqrt(self.mask_scale[0] * H * W / aspect_ratio)))
        target_h = min(max(target_h, 1), H -1)
        target_w = min(max(target_w, 1), W - 1)
        top = random.randint(0, H - target_h)
        left = random.randint(0, W - target_w)

        # For simplicity, let's make context the inverse, though often it's a separate larger block
        # Target mask: zeros everywhere, ones in the target block
        target_mask = target_mask.at[top:top+target_h, left:left+target_w, :].set(1.0)
        # Context mask: ones everywhere, zeros in the target block
        context_mask = context_mask.at[top:top+target_h, left:left+target_w, :].set(0.0)
        # --- End Simple Block Masking ---

        # Expand for batch
        context_mask = jnp.expand_dims(context_mask, axis=0) # (1, H, W, 1)
        target_mask = jnp.expand_dims(target_mask, axis=0)  # (1, H, W, 1)

        # We return the same mask for the whole batch here
        # To make per-sample masks, use vmap or generate within a jitted function using rng keys.
        # context_mask = jnp.tile(context_mask, (batch_size, 1, 1, 1))
        # target_mask = jnp.tile(target_mask, (batch_size, 1, 1, 1))

        # Placeholder for RNG usage if implemented in pure JAX
        # rng, _ = jax.random.split(rng)

        return context_mask, target_mask # Shape (1, H, W, 1)


    def __call__(self, x: jnp.ndarray, target_encoder_params: dict, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculates JEPA loss.
        Args:
            x: Input observations (B, H, W, C).
            target_encoder_params: Parameters for the target encoder.
            rng: JAX PRNGKey.

        Returns:
            Tuple of (JEPA loss, predicted target embeddings).
        """
        batch_size = x.shape[0]
        rng, mask_rng = jax.random.split(rng)

        # Generate masks (using the simplified approach for now)
        context_mask, target_mask = self._generate_masks(batch_size, mask_rng)

        # Apply masks
        x_context = x * context_mask # Broadcasts (1,H,W,1) mask to (B,H,W,C)
        x_target = x * target_mask   # Broadcasts (1,H,W,1) mask to (B,H,W,C)

        # Encode context path with the online encoder
        z_context = self.encoder(x_context)

        # Predict target embedding from context embedding
        z_pred_target = self.predictor(z_context)

        # Encode target path with the target encoder (using provided parameters)
        # Use stop_gradient as the target encoder is not updated via this loss's gradient
        z_actual_target = jax.lax.stop_gradient(
            self.encoder.apply({'params': target_encoder_params}, x_target)
        )

        # Calculate Loss (Mean Squared Error between normalized embeddings)
        # Normalization helps prevent representation collapse
        def normalize(v):
            return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)

        loss = jnp.mean(jnp.sum((normalize(z_pred_target) - normalize(z_actual_target))**2, axis=-1))

        return loss, z_pred_target # Return loss and predictions

    def get_embedding(self, x: jnp.ndarray) -> jnp.ndarray:
        """Get embedding from the online encoder for the full image."""
        # Used by the Actor-Critic policy/value heads
        return self.encoder(x)


class JEPAEncoderMLP(nn.Module):
    """MLP Encoder for JEPA working on flat vectors."""
    output_dim: int
    hidden_dim: int = 512 # Or match ActorCritic layer_width
    num_layers: int = 3    # Number of hidden layers
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        if x.ndim != 2:
            raise ValueError(f"JEPAEncoderMLP expects input shape (B, feature_dim), got {x.shape}")

        for i in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim,
                         kernel_init=orthogonal(np.sqrt(2)), # Or lecun_normal()
                         bias_init=constant(0.0),
                         name=f"hidden_{i}")(x)
            x = self.activation(x)
        x = nn.Dense(features=self.output_dim,
                     kernel_init=orthogonal(1.0), # Or default glorot/lecun
                     bias_init=constant(0.0),
                     name="output")(x)
        return x
    
class ForwardPredictorMLP(nn.Module):
    """
    Predicts next state embedding from current state embedding and action.
    """
    output_dim: int         # Dimension of the predicted embedding
    action_dim: int         # Number of discrete actions
    embedding_dim: int      # Dimension of the state embedding input
    hidden_dim: int = 512
    num_layers: int = 3
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, current_embedding, action):
        # action shape: (B,) integers
        # current_embedding shape: (B, embedding_dim)

        # One-hot encode the discrete action
        action_one_hot = jax.nn.one_hot(action, num_classes=self.action_dim) # Shape: (B, action_dim)

        # Concatenate embedding and action representation
        x = jnp.concatenate([current_embedding, action_one_hot], axis=-1)

        # Pass through MLP
        for i in range(self.num_layers - 1):
            x = nn.Dense(features=self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name=f"dense_{i}")(x)
            x = self.activation(x)
        x = nn.Dense(features=self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="dense_out")(x)
        # Predicted next state embedding
        return x


class InversePredictorMLP(nn.Module):
    """
    Predicts action logits from two consecutive state embeddings.
    """
    action_dim: int         # Number of discrete actions (output logits)
    embedding_dim: int      # Dimension of the state embedding input
    hidden_dim: int = 512
    num_layers: int = 2     # Often simpler than forward model
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, embedding_t, embedding_t_plus_1):
        # Concatenate consecutive state embeddings
        # Ensure inputs have shape (B, embedding_dim)
        if embedding_t.shape != embedding_t_plus_1.shape or embedding_t.ndim != 2:
             raise ValueError("InversePredictorMLP inputs must have matching shape (B, embedding_dim)")

        x = jnp.concatenate([embedding_t, embedding_t_plus_1], axis=-1)
        # Pass through MLP
        for i in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name=f"dense_{i}")(x)
            x = self.activation(x)
        # Output logits for each action
        action_logits = nn.Dense(features=self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="action_logits")(x)
        return action_logits    


class ForwardDynamicsAuxiliary(nn.Module):
    """
    Auxiliary module calculating forward prediction loss and raw error.
    Encoder is trained by this loss AND the separate inverse model loss.
    """
    input_dim: int
    action_dim: int # Needed for the predictor
    encoder_output_dim: int
    encoder_hidden_dim: int
    encoder_layers: int
    predictor_hidden_dim: int
    predictor_layers: int

    def setup(self):
        self.encoder = JEPAEncoderMLP(output_dim=self.encoder_output_dim,
                                      hidden_dim=self.encoder_hidden_dim,
                                      num_layers=self.encoder_layers,
                                      name="online_encoder_mlp")
        self.forward_predictor = ForwardPredictorMLP(output_dim=self.encoder_output_dim,
                                                 action_dim=self.action_dim,
                                                 embedding_dim=self.encoder_output_dim,
                                                 hidden_dim=self.predictor_hidden_dim,
                                                 num_layers=self.predictor_layers,
                                                 name="forward_predictor_mlp")
        # Target encoder uses the same architecture (JEPAEncoderMLP)

    def __call__(self, obs_t: jnp.ndarray, action_t: jnp.ndarray, obs_t_plus_1: jnp.ndarray, target_encoder_params: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculates Forward Dynamics prediction loss and raw error.

        Returns:
            Tuple[forward_loss, raw_forward_error_per_sample]:
                - forward_loss: Scalar loss (potentially normalized) for gradients.
                - raw_forward_error_per_sample: Raw prediction error (e.g., MSE) per sample (B,) for intrinsic reward.
        """
        # ... (Input shape checks as before) ...
        if obs_t.ndim != 2 or obs_t.shape[-1] != self.input_dim:
            raise ValueError(f"ForwardDynamicsAuxiliary expects obs_t shape (B, {self.input_dim}), got {obs_t.shape}")
        if obs_t_plus_1.ndim != 2 or obs_t_plus_1.shape[-1] != self.input_dim:
             raise ValueError(f"ForwardDynamicsAuxiliary expects obs_t_plus_1 shape (B, {self.input_dim}), got {obs_t_plus_1.shape}")
        if action_t.ndim != 1:
            raise ValueError(f"ForwardDynamicsAuxiliary expects action_t shape (B,), got {action_t.shape}")

        phi_t = self.encoder(obs_t)
        phi_hat_t_plus_1 = self.forward_predictor(phi_t, action_t)

        phi_target_t_plus_1 = jax.lax.stop_gradient(
            self.encoder.apply({'params': target_encoder_params}, obs_t_plus_1)
        )

        # --- Calculate Raw Error (MSE) for Intrinsic Reward ---
        # Sum squared errors across embedding dimension, shape (B,)
        raw_error_per_sample = jnp.sum(jnp.square(phi_hat_t_plus_1 - phi_target_t_plus_1), axis=-1)

        # --- Calculate Loss (potentially normalized MSE) for gradients ---
        def normalize(v):
            return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)
        # Using normalized loss for gradients as per JEPA influence
        normalized_loss = jnp.mean(jnp.sum((normalize(phi_hat_t_plus_1) - normalize(phi_target_t_plus_1))**2, axis=-1))

        return normalized_loss, raw_error_per_sample

    def get_embedding(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 2:
             raise ValueError(f"ForwardDynamicsAuxiliary.get_embedding expects input shape (B, {self.input_dim}), got {x.shape}")
        return self.encoder(x)

class PredictorMLP(nn.Module):
    """MLP Predictor for JEPA working on embeddings (can be same as JEPAEncoderMLP or simpler)."""
    output_dim: int
    hidden_dim: int = 1024
    num_layers: int = 3
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, context_embedding):
        x = context_embedding
        for i in range(self.num_layers - 1):
            x = nn.Dense(features=self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name=f"dense_{i}")(x)
            x = self.activation(x)
        x = nn.Dense(features=self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="dense_out")(x)
        return x

class JEPAVector(nn.Module):
    """
    JEPA Model adapted for flat vector observations. Uses feature masking.

    Attributes:
        input_dim: Dimensionality of the flat input vector.
        encoder_output_dim: Dimensionality of the encoder's output embedding.
        encoder_hidden_dim: Hidden dimensionality of the MLP encoder.
        encoder_layers: Number of hidden layers in the MLP encoder.
        predictor_hidden_dim: Hidden dimensionality of the MLP predictor.
        predictor_layers: Number of layers in the MLP predictor.
        mask_prob: Probability of masking out (zeroing) a feature for context/target.
    """
    input_dim: int
    encoder_output_dim: int
    encoder_hidden_dim: int
    encoder_layers: int
    predictor_hidden_dim: int
    predictor_layers: int
    mask_prob: float = 0.3 # Probability of zeroing out a feature

    def setup(self):
        # Define online encoder and predictor using MLPs
        self.encoder = JEPAEncoderMLP(output_dim=self.encoder_output_dim,
                                      hidden_dim=self.encoder_hidden_dim,
                                      num_layers=self.encoder_layers,
                                      name="online_encoder_mlp")
        self.predictor = PredictorMLP(output_dim=self.encoder_output_dim,
                                      hidden_dim=self.predictor_hidden_dim,
                                      num_layers=self.predictor_layers,
                                      name="predictor_mlp")
        # Target encoder uses the same architecture (JEPAEncoderMLP) but separate parameters (handled via EMA)

    def _generate_feature_masks(self, rng: jax.random.PRNGKey, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generates random feature masks using JAX."""
        rng_context, rng_target = jax.random.split(rng)

        # Create context mask: Keep features with probability (1 - mask_prob)
        context_mask = jax.random.bernoulli(rng_context, p=(1.0 - self.mask_prob), shape=(batch_size, self.input_dim))
        # Create target mask: Keep features with probability (1 - mask_prob)
        target_mask = jax.random.bernoulli(rng_target, p=(1.0 - self.mask_prob), shape=(batch_size, self.input_dim))

        # Ensure masks are float for multiplication
        return context_mask.astype(jnp.float32), target_mask.astype(jnp.float32)

    def __call__(self, x: jnp.ndarray, target_encoder_params: dict, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculates JEPA loss for vector inputs.
        Args:
            x: Input observations (B, input_dim).
            target_encoder_params: Parameters for the target encoder (Flax PyTree).
            rng: JAX PRNGKey.

        Returns:
            Tuple of (JEPA loss, predicted target embeddings).
        """
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"JEPAVector expects input shape (B, {self.input_dim}), got {x.shape}")

        batch_size = x.shape[0]
        rng, mask_rng = jax.random.split(rng)

        # Generate feature masks using JAX random functions
        context_mask, target_mask = self._generate_feature_masks(mask_rng, batch_size)

        # Apply masks (element-wise multiplication)
        x_context = x * context_mask
        x_target = x * target_mask # Target encoder sees a masked view too

        # Encode context path with the online encoder
        z_context = self.encoder(x_context)

        # Predict target embedding from context embedding
        z_pred_target = self.predictor(z_context)

        # Encode target path with the target encoder (using provided parameters)
        # Use stop_gradient as the target encoder is not updated via this loss's gradient
        target_encoder_instance = JEPAEncoderMLP(output_dim=self.encoder_output_dim,
                                                 hidden_dim=self.encoder_hidden_dim,
                                                 num_layers=self.encoder_layers,
                                                 name="target_encoder_apply_mlp")
        z_actual_target = jax.lax.stop_gradient(
            target_encoder_instance.apply({'params': target_encoder_params}, x_target)
        )

        # --- Calculate Loss (MSE on normalized embeddings) ---
        def normalize(v):
            return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)

        loss = jnp.mean(jnp.sum((normalize(z_pred_target) - normalize(z_actual_target))**2, axis=-1))

        return loss, z_pred_target # Return loss and predictions

    def get_embedding(self, x: jnp.ndarray) -> jnp.ndarray:
        """Get embedding from the online encoder for the full (unmasked) vector."""
        if x.ndim != 2:
            raise ValueError(f"JEPAVector.get_embedding expects input shape (B, {self.input_dim}), got {x.shape}")
        return self.encoder(x)
