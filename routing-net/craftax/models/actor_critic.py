import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, List
from models.routing import (
    SubroutineSelector,
    TopKSelectorOTT,
    SwitchTopKSelector,
    Subroutine,
)

import distrax


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
        # image_embedding = jnp.concatenate([image_embedding, obs[:, : CraftaxEnv.get_flat_map_obs_shape()]], axis=-1)

        # Combine embeddings
        embedding = jnp.concatenate([image_embedding, flat_obs], axis=-1)
        embedding = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticConv(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs):
        x = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        embedding = x.reshape(x.shape[0], -1)

        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    routing_type: str
    num_subroutines: int
    keep_count: int
    num_moe_passes: int = 1
    activation: str = "tanh"

    def setup(self):
        # Define activation function
        if self.activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

        # Define the Gating Network (Selector)
        if self.routing_type == "sigmoid":
            self.selector = SubroutineSelector(
                num_subroutines=self.num_subroutines,
                layer_width=self.layer_width,
                name="SigmoidSelector" # Naming modules is good practice
            )
        elif self.routing_type == "ott":
             self.selector = TopKSelectorOTT(
                num_subroutines=self.num_subroutines,
                k=self.keep_count,
                layer_width=self.layer_width,
                name="OTTSelector"
            )
        elif self.routing_type == "switch":
             self.selector = SwitchTopKSelector(
                num_subroutines=self.num_subroutines,
                k=self.keep_count,
                layer_width=self.layer_width,
                noisy_gating=True, # Assuming noisy gating is desired
            )
        else:
            raise ValueError(f"Unknown routing_type: {self.routing_type}")

        # Define the Experts (Subroutines)
        subroutine_feature_dim = self.layer_width
        self.subroutines = [
            Subroutine(
                layer_width=subroutine_feature_dim,
                bottleneck_width=subroutine_feature_dim // 2,
                activation=self.activation_fn,
            ) for i in range(self.num_subroutines)
        ]

        # Define Actor MLP Layers (excluding final head)
        self.actor_layers = [
            nn.Dense(
                self.layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name=f"actor_dense_{i}"
            ) for i in range(3)
        ]

        # Define Critic MLP Layers (excluding final head)
        self.critic_layers = [
             nn.Dense(
                self.layer_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
                name=f"critic_dense_{i}"
            ) for i in range(3)
        ]

        # Define Final Actor Head
        self.actor_head = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor_head"
        )

        # Define Final Critic Head
        self.critic_head = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic_head"
        )

    @nn.compact
    def __call__(self, x):
        current_embedding = x
        aux_loss_total = 0.0

        current_embedding = nn.Dense(self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0))(current_embedding)
        current_embedding = nn.relu(current_embedding)

        for pass_num in range(self.num_moe_passes):
            # 1. Calculate Gating Weights using the current embedding
            if self.routing_type == "switch":
                # Switch router might need noise epsilon and returns aux_loss
                # Consider making noise_epsilon configurable or decaying
                selection_weights, aux_loss = self.selector(current_embedding)
                aux_loss_total += aux_loss # Accumulate aux loss per pass
            else:
                # Sigmoid and OTT just return weights
                selection_weights = self.selector(current_embedding)

            # 2. Apply all Experts to the current embedding
            subroutine_outputs = []
            for i in range(self.num_subroutines):
                # Use the subroutines defined in setup
                sub_output = self.subroutines[i](current_embedding)
                subroutine_outputs.append(sub_output)

            # Stack outputs: shape becomes (batch_size, num_subroutines, feature_dim)
            all_sub_outputs = jnp.stack(subroutine_outputs, axis=1)

            # 3. Weight and Combine Expert Outputs
            # Expand weights: shape becomes (batch_size, num_subroutines, 1)
            expanded_weights = jnp.expand_dims(selection_weights, axis=-1)

            # Weighted sum: shape becomes (batch_size, feature_dim)
            weighted_outputs = all_sub_outputs * expanded_weights
            current_embedding = jnp.sum(weighted_outputs, axis=1) + current_embedding
            current_embedding = nn.LayerNorm()(current_embedding)

        moe_output = current_embedding

        # --- Actor Head ---
        actor_features = moe_output
        for layer in self.actor_layers:
             actor_features = layer(actor_features)
             actor_features = self.activation_fn(actor_features) # Apply activation
        actor_mean = self.actor_head(actor_features)
        pi = distrax.Categorical(logits=actor_mean)

        # --- Critic Head ---
        critic_features = moe_output
        for layer in self.critic_layers:
            critic_features = layer(critic_features)
            critic_features = self.activation_fn(critic_features) # Apply activation
        critic_value = self.critic_head(critic_features)

        return pi, jnp.squeeze(critic_value, axis=-1), aux_loss_total


class ActorCriticWithEmbedding(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_emb = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_emb = activation(actor_emb)

        actor_emb = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_emb)
        actor_emb = activation(actor_emb)

        actor_emb = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_emb)
        actor_emb = activation(actor_emb)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_emb)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), actor_emb