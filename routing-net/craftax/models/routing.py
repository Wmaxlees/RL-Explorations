import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
from ott.tools.soft_sort import ranks as soft_rank

class SubroutineSelector(nn.Module):
    num_subroutines: int
    layer_width: int
    activation_fn: callable

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
    hidden_dims: Sequence[int]
    
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

class Subroutine(nn.Module):
    layer_width: int
    inf_bottleneck_width: int
    output_dim: int
    activation_fn: callable

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.inf_bottleneck_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        
        return x