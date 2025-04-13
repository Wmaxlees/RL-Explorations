import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal, lecun_normal
from typing import Sequence, NamedTuple, Any, Callable
from flax.training.train_state import TrainState
# import distrax # No longer needed for JEPA prediction
import gymnax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import matplotlib.pyplot as plt
import functools # For partial application

# --- JEPA Network Architecture ---
class Encoder(nn.Module):
    embed_dim: int
    activation_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128, kernel_init=lecun_normal(), bias_init=constant(0.0))(x)
        x = self.activation_fn(x)
        x = nn.Dense(128, kernel_init=lecun_normal(), bias_init=constant(0.0))(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.embed_dim, kernel_init=lecun_normal(), bias_init=constant(0.0))(x)
        # Optional: Add LayerNorm or other normalization here
        # x = nn.LayerNorm(epsilon=1e-6)(x)
        return x

class Predictor(nn.Module):
    embed_dim: int
    hidden_dim: int
    activation_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, z): # Takes embedding z as input
        x = nn.Dense(self.hidden_dim, kernel_init=lecun_normal(), bias_init=constant(0.0))(z)
        x = self.activation_fn(x)
        x = nn.Dense(self.embed_dim, kernel_init=lecun_normal(), bias_init=constant(0.0))(x) # Predicts embedding
        return x

class JEPA(nn.Module):
    embed_dim: int
    predictor_hidden_dim: int
    activation: str = "relu"

    def setup(self):
        if self.activation == "relu":
            activation_fn = nn.relu
        elif self.activation == "tanh":
             activation_fn = nn.tanh
        else:
            activation_fn = nn.relu # Default

        self.encoder = Encoder(embed_dim=self.embed_dim, activation_fn=activation_fn)
        self.predictor = Predictor(embed_dim=self.embed_dim,
                                   hidden_dim=self.predictor_hidden_dim,
                                   activation_fn=activation_fn)

    def __call__(self, x):
        # Online path
        z = self.encoder(x)
        z_pred = self.predictor(z)
        return z_pred

    def encode(self, x):
        # Helper to just get the encoding
        return self.encoder(x)

# --- Data Structures ---
class Transition(NamedTuple):
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray # Keep done to potentially mask loss on terminal transitions

# --- Train State for JEPA ---
# We need separate states for online and target networks
class OnlineTrainState(TrainState):
    pass # Standard TrainState holds online encoder + predictor params

class TargetTrainState(NamedTuple):
    # Only store encoder parameters for the target
    params: Any

# --- Utility Functions ---
def l2_normalize(x, axis=-1, epsilon=1e-12):
    """L2 Normalize an array."""
    norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / (norm + epsilon)

@functools.partial(jax.jit, static_argnums=(0,))
def update_target_network(ema_decay: float, online_params: Any, target_params: Any) -> Any:
    """Exponential Moving Average update for target network parameters."""
    new_target_params = jax.tree_map(
        lambda online, target: ema_decay * target + (1.0 - ema_decay) * online,
        online_params,
        target_params
    )
    return new_target_params

# --- Training Function ---
def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    # LogWrapper is less critical here, but keep if useful for env stats
    env = LogWrapper(env)

    # Use a simpler schedule or constant LR for JEPA initially
    if config["ANNEAL_LR"]:
        def linear_schedule(count):
            # Adjust schedule based on total updates * epochs * minibatches
            total_updates = config["NUM_UPDATES"] * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
            frac = 1.0 - (count / total_updates)
            return config["LR"] * frac
    else:
        # Keep learning rate constant if not annealing
         def constant_schedule(count):
             return config["LR"]

    def train(rng):
        # INIT NETWORK
        network = JEPA(embed_dim=config["EMBED_DIM"],
                       predictor_hidden_dim=config["PREDICTOR_HIDDEN_DIM"],
                       activation=config["ACTIVATION"])
        rng, _rng_init = jax.random.split(rng)
        init_x = jnp.zeros((1,) + env.observation_space(env_params).shape) # Add batch dim
        online_params = network.init(_rng_init, init_x)

        # Separate online encoder params for EMA update
        online_encoder_params = online_params['params']['encoder']
        # Target network starts with the same weights as online encoder
        target_encoder_params = online_encoder_params

        # Setup Optimizer
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
             tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=config["LR"], eps=1e-5),
            )

        # TrainState holds online encoder and predictor params
        online_train_state = OnlineTrainState.create(
            apply_fn=network.apply, # Applies encoder and predictor
            params=online_params['params'], # Contains both 'encoder' and 'predictor'
            tx=tx,
        )
        # Target state just holds the target encoder parameters
        target_state = TargetTrainState(params=target_encoder_params)

        # INIT ENV
        rng, _rng_reset = jax.random.split(rng)
        reset_rng = jax.random.split(_rng_reset, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # --- Jitted auxiliary functions ---
        @jax.jit
        def get_embeddings(online_params, target_encoder_params, obs, next_obs):
            # Get online predictions
            z_next_pred = network.apply({'params': online_params}, obs) # Use combined online state

            # Get target embeddings (with stop_gradient)
            # Apply only the encoder part using the target parameters
            z_next_target = network.apply(
                {'params': {'encoder': target_encoder_params}}, # Apply using ONLY target encoder params
                next_obs,
                method=network.encode # Call the encode method explicitly
            )
            z_next_target = jax.lax.stop_gradient(z_next_target) # Crucial for JEPA/BYOL style

            # Normalize embeddings (common practice)
            z_next_pred_norm = l2_normalize(z_next_pred)
            z_next_target_norm = l2_normalize(z_next_target)

            return z_next_pred_norm, z_next_target_norm

        @jax.jit
        def compute_loss(z_pred_norm, z_target_norm):
             # Calculate MSE loss between normalized embeddings
             loss = jnp.mean(jnp.sum((z_pred_norm - z_target_norm)**2, axis=-1))
             # Alternative: Cosine Similarity Loss: 2.0 - jnp.mean(jnp.sum(z_pred_norm * z_target_norm, axis=-1))
             return loss

        # --- TRAINING LOOP ---
        def _update_step(runner_state, unused):
            online_train_state, target_state, env_state, last_obs, rng = runner_state

            # COLLECT TRANSITIONS (obs, next_obs)
            def _env_step(carry, unused):
                online_train_state, target_state, env_state, last_obs, rng = carry

                # SAMPLE RANDOM ACTION (no policy learning)
                rng, _rng_action = jax.random.split(rng)
                # Assuming discrete action space, sample uniformly
                # If continuous, sample from a standard normal or uniform distribution
                # Make sure action shape matches env.action_space
                action_keys = jax.random.split(_rng_action, config["NUM_ENVS"])
                # Sample random action per environment
                actions = jax.vmap(env.action_space(env_params).sample)(action_keys)

                # STEP ENV
                rng, _rng_step = jax.random.split(rng)
                rng_step = jax.random.split(_rng_step, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, actions, env_params
                )
                # Store current obs and the resulting next obs
                transition = Transition(obs=last_obs, next_obs=obsv, done=done)
                carry = (online_train_state, target_state, env_state, obsv, rng)
                return carry, transition

            # Scan over env steps
            runner_state_new, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            online_train_state, target_state, env_state, last_obs, rng = runner_state_new

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_states, batch_info):
                    online_train_state, target_state = train_states
                    obs_batch, next_obs_batch, _ = batch_info # Unpack minibatch data

                    def _loss_fn(online_params, target_encoder_params, obs, next_obs):
                        # RERUN NETWORK FOR PREDICTION AND TARGET
                        z_next_pred_norm, z_next_target_norm = get_embeddings(
                            online_params, target_encoder_params, obs, next_obs
                        )
                        # CALCULATE JEPA LOSS (e.g., MSE on normalized embeddings)
                        loss = compute_loss(z_next_pred_norm, z_next_target_norm)
                        return loss

                    # Compute gradients only w.r.t. online parameters
                    grad_fn = jax.value_and_grad(_loss_fn)
                    loss, grads = grad_fn(
                        online_train_state.params, target_state.params, obs_batch, next_obs_batch
                    )
                    # Apply gradients to update online encoder and predictor
                    online_train_state = online_train_state.apply_gradients(grads=grads)

                    return (online_train_state, target_state), loss

                online_train_state, target_state, traj_batch, rng = update_state
                rng, _rng_perm = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng_perm, batch_size)

                # Flatten transitions first
                batch = jax.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
                )
                # Shuffle data
                shuffled_batch = jax.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Create minibatches
                minibatches = jax.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                # Scan over minibatches
                (online_train_state, target_state), total_loss = jax.lax.scan(
                    _update_minibatch, (online_train_state, target_state), minibatches
                )

                # UPDATE TARGET NETWORK using EMA after each epoch
                new_target_encoder_params = update_target_network(
                    config["EMA_DECAY"],
                    online_train_state.params['encoder'], # Use updated online encoder params
                    target_state.params
                )
                target_state = target_state._replace(params=new_target_encoder_params)

                update_state = (online_train_state, target_state, traj_batch, rng)
                # Return average loss over minibatches in the epoch
                return update_state, total_loss.mean()

            # Scan over update epochs
            update_state = (online_train_state, target_state, traj_batch, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            # Extract final states after epochs
            online_train_state, target_state, _, rng = update_state
            # metric = traj_batch.info # Original RL metric, replace with loss
            metric = loss_info # Store the loss from each epoch

            runner_state = (online_train_state, target_state, env_state, last_obs, rng)
            # Return average loss across epochs for this update step
            return runner_state, metric.mean()

        rng, _rng_run = jax.random.split(rng)
        runner_state_initial = (online_train_state, target_state, env_state, obsv, _rng_run)
        runner_state_final, metrics = jax.lax.scan(
            _update_step, runner_state_initial, None, config["NUM_UPDATES"]
        )
        # metrics will contain the average loss per update step
        return {"runner_state": runner_state_final, "metrics": metrics}

    return train


if __name__ == "__main__":
    config = {
        # Environment/Run settings
        "NUM_ENVS": 16,         # Increase parallelism
        "NUM_STEPS": 128,       # Steps per environment before update
        "TOTAL_TIMESTEPS": 1e6, # Total interactions
        "ENV_NAME": "CartPole-v1",

        # JEPA specific Hyperparameters
        "EMBED_DIM": 64,       # Dimension of the latent embedding
        "PREDICTOR_HIDDEN_DIM": 128, # Hidden layer size in the predictor MLP
        "EMA_DECAY": 0.99,     # Decay rate for target network updates (0.99 - 0.999 typical)

        # Optimizer/Training settings
        "LR": 1e-3,            # Learning rate (may need tuning)
        "UPDATE_EPOCHS": 4,    # Number of passes over collected data per update
        "NUM_MINIBATCHES": 8,  # Number of minibatches per epoch
        "MAX_GRAD_NORM": 1.0,  # Gradient clipping norm
        "ACTIVATION": "relu",  # Activation function (relu or tanh)
        "ANNEAL_LR": False,    # Whether to linearly anneal LR (False might be simpler to start)

        # Removed RL Hyperparameters
        # "GAMMA": 0.99,
        # "GAE_LAMBDA": 0.95,
        # "CLIP_EPS": 0.2,
        # "ENT_COEF": 0.01,
        # "VF_COEF": 0.5,
    }

    rng = jax.random.PRNGKey(42)
    train_fn = make_train(config)
    # JIT the training function (can take a while on first run)
    # Consider disabling jit for debugging: train_fn = make_train(config)
    train_jit = jax.jit(train_fn)
    print("Starting training...")
    out = train_jit(rng)
    print("Training finished.")

    # Plot the JEPA loss over training updates
    plt.figure(figsize=(10, 6))
    plt.plot(out["metrics"])
    plt.xlabel("Update Step")
    plt.ylabel("Average JEPA Loss")
    plt.title(f"JEPA Training Loss ({config['ENV_NAME']})")
    plt.grid(True)
    plt.show()

    # Example: Get embeddings for a sample observation after training
    # final_online_state = out["runner_state"][0]
    # final_target_state = out["runner_state"][1]
    # sample_obs = jnp.zeros((1,) + env.observation_space(env_params).shape) # Example obs

    # network_trained = JEPA(embed_dim=config["EMBED_DIM"],
    #                        predictor_hidden_dim=config["PREDICTOR_HIDDEN_DIM"],
    #                        activation=config["ACTIVATION"])

    # # Get embedding using the trained online encoder
    # embedding = network_trained.apply(
    #     {'params': final_online_state.params},
    #     sample_obs,
    #     method=network_trained.encode
    # )
    # print("Sample embedding shape:", embedding.shape)