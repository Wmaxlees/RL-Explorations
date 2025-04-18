import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, Optional
from ott.tools.soft_sort import ranks as soft_rank
import jax

class SubroutineSelector(nn.Module):
    num_subroutines: int
    layer_width: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_subroutines, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        x = nn.sigmoid(x)
        
        return x

class TopKSelectorOTT(nn.Module):
    num_subroutines: int
    k: int
    layer_width: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)

        scores = nn.Dense(features=self.num_subroutines,
                          kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        ranks = soft_rank(scores, axis=-1)
        inverted_ranks = (self.num_subroutines - 1) - ranks
        threshold_rank_approx = self.num_subroutines - self.k
        steepness = 10.0
        soft_mask = nn.sigmoid((inverted_ranks - threshold_rank_approx) * steepness)
        
        return soft_mask
    
class SwitchTopKSelector(nn.Module):
    num_subroutines: int
    k: int
    layer_width: int
    noisy_gating: bool = True
    noise_epsilon: float = 1e-2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gate_logits = nn.Dense(
            features=self.num_subroutines,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0)
        )(x)

        if self.noisy_gating:
            dropout_rng = self.make_rng("gating_noise")

            noise_weights = nn.Dense(
                features=self.num_subroutines,
                kernel_init=orthogonal(2),
                bias_init=constant(0.0)
            )(x)
            noise = jax.random.normal(dropout_rng, gate_logits.shape) * nn.softplus(noise_weights)

            gate_logits += noise

            num_experts = self.num_subroutines
            gating_probs = jax.nn.softmax(gate_logits, axis=-1) # Probs over all experts
            fraction_routed = jnp.mean(gating_probs, axis=0) # Mean prob per expert over batch
            prob_per_expert_mean = jnp.mean(gating_probs, axis=0) # Same as fraction_routed
            
            aux_loss = jnp.sum(fraction_routed * prob_per_expert_mean) * num_experts

        topk_results = jax.lax.top_k(gate_logits, k=self.k)
        topk_indices = topk_results[1]

        one_hot_masks = jax.nn.one_hot(topk_indices, num_classes=self.num_subroutines, dtype=gate_logits.dtype)
        mask = jnp.sum(one_hot_masks, axis=1)

        masked_gate_logits = jnp.where(mask == 1, gate_logits, -jnp.inf)

        selection_weights = nn.softmax(masked_gate_logits, axis=-1)

        return selection_weights, aux_loss



class Subroutine(nn.Module):
    layer_width: int
    bottleneck_width: int
    activation: nn.Module = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(x)
        x = self.activation(x)
        x = nn.Dense(
            self.bottleneck_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(x)
        x = self.activation(x)
        x = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(x)
        x = self.activation(x)
        
        return x