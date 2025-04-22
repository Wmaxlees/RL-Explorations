import flax.linen as nn
import chex
import jax.numpy as jnp

class SkillSelector(nn.Module):
    hidden_dim: int
    num_available_skills: int

    @nn.compact
    def __call__(self, x: chex.Array, available_mask: chex.Array) -> chex.Array:
        '''
        x: The environment Observation
        available_mask: A mask of the skills that are available for this particular agent.
        '''
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        logits = nn.Dense(features=self.num_available_skills)(x)
        large_negative = jnp.finfo(logits.dtype).min
        masked_logits = jnp.where(available_mask, logits, large_negative)
        return masked_logits


class SkillPolicy(nn.Module):
    hidden_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.action_dim)(x)
        return x


class Embedder(nn.Module):
    hidden_dim: int
    embedding_dim: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.embedding_dim)(x)
        x = nn.LayerNorm()(x)
        return x
    
class SkillCritic(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x.squeeze(axis=-1)
