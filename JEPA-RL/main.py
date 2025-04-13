import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal, lecun_normal
from typing import Sequence, NamedTuple, Any, Callable
from flax.training.train_state import TrainState
import distrax # Needed again for the Actor
import gymnax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import matplotlib.pyplot as plt
import functools

# --- Combined Network Architecture ---

class Encoder(nn.Module):
    embed_dim: int
    activation_fn: Callable = nn.relu
    name: str = "encoder" # Explicit name

    @nn.compact
    def __call__(self, x):
        # Simple MLP encoder
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="fc1")(x)
        x = self.activation_fn(x)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="fc2")(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="fc_out")(x)
        # Optional: LayerNorm
        # x = nn.LayerNorm(epsilon=1e-6, name="ln_out")(x)
        return x

class Predictor(nn.Module):
    embed_dim: int
    hidden_dim: int
    activation_fn: Callable = nn.relu
    name: str = "predictor" # Explicit name

    @nn.compact
    def __call__(self, z): # Takes embedding z
        x = nn.Dense(self.hidden_dim, kernel_init=lecun_normal(), bias_init=constant(0.0), name="fc1")(z)
        x = self.activation_fn(x)
        x = nn.Dense(self.embed_dim, kernel_init=lecun_normal(), bias_init=constant(0.0), name="fc_out")(z)
        return x

class ActorHead(nn.Module):
    action_dim: int
    activation_fn: Callable = nn.tanh
    name: str = "actor" # Explicit name

    @nn.compact
    def __call__(self, z): # Takes embedding z
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="fc1")(z)
        x = self.activation_fn(x)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="fc2")(x)
        x = self.activation_fn(x)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="fc_out")(x)
        pi = distrax.Categorical(logits=actor_mean)
        return pi

class CriticHead(nn.Module):
    activation_fn: Callable = nn.tanh
    name: str = "critic" # Explicit name

    @nn.compact
    def __call__(self, z): # Takes embedding z
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="fc1")(z)
        x = self.activation_fn(x)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="fc2")(x)
        x = self.activation_fn(x)
        critic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="fc_out")(x)
        return jnp.squeeze(critic_out, axis=-1)


class JEPA_ActorCritic(nn.Module):
    action_dim: int
    embed_dim: int
    predictor_hidden_dim: int
    activation: str = "tanh"

    def setup(self):
        if self.activation == "relu":
            activation_fn = nn.relu
        else:
            activation_fn = nn.tanh

        self.encoder = Encoder(embed_dim=self.embed_dim, activation_fn=activation_fn)
        self.predictor = Predictor(embed_dim=self.embed_dim, hidden_dim=self.predictor_hidden_dim, activation_fn=activation_fn)
        self.actor = ActorHead(action_dim=self.action_dim, activation_fn=activation_fn)
        self.critic = CriticHead(activation_fn=activation_fn)

    def __call__(self, x):
        """ Used for combined Actor-Critic forward pass during rollouts/updates """
        z = self.encoder(x)
        pi = self.actor(z)
        value = self.critic(z)
        _ = self.predictor(jax.lax.stop_gradient(z)) # Include for initialization
        return z, pi, value # Return embedding z as well

    def encode(self, x):
        """ Explicit encode method """
        return self.encoder(x)

    def predict_embedding(self, z):
        """ Explicit prediction method """
        return self.predictor(z)

# --- Data Structures ---
class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray # Value estimate from rollout time
    next_obs: jnp.ndarray # Added for JEPA
    info: jnp.ndarray

# --- Train States ---
# One state for all online parameters
class OnlineTrainState(TrainState):
    pass

# Separate state for target encoder parameters
class TargetEncoderState(NamedTuple):
    params: Any # Holds only the encoder parameters

# --- Utility Functions (l2_normalize, update_target_network - same as before) ---
def l2_normalize(x, axis=-1, epsilon=1e-12):
    """L2 Normalize an array."""
    norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / (norm + epsilon)

@functools.partial(jax.jit, static_argnums=(0,))
def update_target_network(ema_decay: float, online_encoder_params: Any, target_encoder_params: Any) -> Any:
    """Exponential Moving Average update for target network parameters."""
    new_target_params = jax.tree_map(
        lambda online, target: ema_decay * target + (1.0 - ema_decay) * online,
        online_encoder_params,
        target_encoder_params
    )
    return new_target_params

# --- Training Function ---
def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    # Minibatch size applies to both RL and JEPA phases
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env) # Keep for RL metrics

    # Learning rate schedule (can be used by the single optimizer)
    if config["ANNEAL_LR"]:
        # Adjust schedule based on total updates * epochs * minibatches
        # Consider if RL/JEPA epochs differ, maybe use total steps
        total_gradient_steps = config["NUM_UPDATES"] * (config["UPDATE_EPOCHS_RL"] + config["UPDATE_EPOCHS_JEPA"]) * config["NUM_MINIBATCHES"]

        def lr_schedule(count):
            # Linear decay based on total gradient steps applied
            frac = 1.0 - (count / total_gradient_steps)
            # Clip frac to avoid negative LR if schedule exceeds total steps (due to rounding etc.)
            frac = jnp.maximum(0.0, frac)
            lr = config["LR"] * frac
            # print(f"Step: {count}, Frac: {frac:.4f}, LR: {lr:.6f}") # Debugging LR schedule
            return lr
    else:
        def lr_schedule(count):
            return config["LR"]

    def train(rng):

        # INIT NETWORK & STATES
        network = JEPA_ActorCritic(
            action_dim=env.action_space(env_params).n,
            embed_dim=config["EMBED_DIM"],
            predictor_hidden_dim=config["PREDICTOR_HIDDEN_DIM"],
            activation=config["ACTIVATION"]
        )
        rng, _rng_init = jax.random.split(rng)
        init_x = jnp.zeros((1,) + env.observation_space(env_params).shape)
        online_params = network.init(_rng_init, init_x)['params'] # Get all online params

        # Target network starts with the same encoder weights
        target_encoder_params = online_params['encoder']
        target_state = TargetEncoderState(params=target_encoder_params)

        # Setup ONE Optimizer for ALL online parameters
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            # Pass the schedule function to adam
            optax.adam(learning_rate=lr_schedule if config["ANNEAL_LR"] else config["LR"], eps=1e-5),
        )

        # Create the single TrainState
        online_train_state = OnlineTrainState.create(
            apply_fn=network.apply, # Applies encoder->actor/critic
            params=online_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng_reset = jax.random.split(rng)
        reset_rng = jax.random.split(_rng_reset, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # --- Jitted auxiliary functions ---
        # (l2_normalize, update_target_network defined above)

        @jax.jit
        def get_jepa_embeddings(online_params, target_encoder_params, obs, next_obs):
            """ Calculates embeddings for JEPA loss """
            # Online path: encode current obs, then predict next embedding
            z_online = network.apply({'params': online_params}, obs, method=network.encode)
            z_next_pred = network.apply({'params': online_params}, z_online, method=network.predict_embedding)

            # Target path: encode next_obs using target encoder
            z_next_target = network.apply(
                {'params': {'encoder': target_encoder_params}}, # ONLY target encoder params
                next_obs,
                method=network.encode
            )
            z_next_target = jax.lax.stop_gradient(z_next_target) # Stop gradient!

            # Normalize
            z_next_pred_norm = l2_normalize(z_next_pred)
            z_next_target_norm = l2_normalize(z_next_target)

            return z_next_pred_norm, z_next_target_norm

        @jax.jit
        def compute_jepa_loss(z_pred_norm, z_target_norm):
             """ MSE Loss for JEPA """
             loss = jnp.mean(jnp.sum((z_pred_norm - z_target_norm)**2, axis=-1))
             return loss

        # --- TRAINING LOOP ---
        def _update_step(runner_state, unused):
            online_train_state, target_state, env_state, last_obs, rng = runner_state
            step_counter_offset = online_train_state.step # For LR schedule

            # === COLLECT TRAJECTORIES ===
            def _env_step(carry, unused):
                online_train_state, env_state, last_obs, rng = carry

                # Encode observation
                # Use apply_fn directly from state to ensure correct method resolution
                z, pi, value = online_train_state.apply_fn({'params': online_train_state.params}, last_obs)

                # Select action
                rng, _rng_action = jax.random.split(rng)
                action = pi.sample(seed=_rng_action)
                log_prob = pi.log_prob(action)

                # Step environment
                rng, _rng_step = jax.random.split(rng)
                rng_step = jax.random.split(_rng_step, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0,0,0,None))(
                    rng_step, env_state, action, env_params
                )
                transition = Transition(
                    obs=last_obs, action=action, reward=reward, done=done,
                    log_prob=log_prob, value=value, next_obs=obsv, info=info
                )
                carry = (online_train_state, env_state, obsv, rng)
                return carry, transition

            # Scan over env steps
            rollout_carry_init = (online_train_state, env_state, last_obs, rng)
            (online_train_state_post_rollout, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, rollout_carry_init, None, config["NUM_STEPS"]
            )
            # Note: online_train_state is NOT updated during rollout scan, use original state for GAE/updates

            # === CALCULATE ADVANTAGES (using original online_train_state) ===
            # Get value of the last observation
            _, _, last_val = online_train_state.apply_fn({'params': online_train_state.params}, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae
                _, advantages = jax.lax.scan(
                    _get_advantages, (jnp.zeros_like(last_val), last_val),
                    traj_batch, reverse=True, unroll=16
                )
                return advantages, advantages + traj_batch.value # targets = advantages + values
            advantages, targets = _calculate_gae(traj_batch, last_val)

            # === PREPARE BATCHES ===
            def prepare_batches(traj_batch, advantages, targets):
                # Flatten data
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                # traj_batch.(obs, action, log_prob, value, next_obs), advantages, targets
                return batch

            full_batch = prepare_batches(traj_batch, advantages, targets)

            # === RL UPDATE PHASE ===
            def _rl_update_epoch(update_state, unused):
                online_train_state, target_state, rng = update_state # Target state passed through but not used here
                rng, _rng_perm = jax.random.split(rng)
                permutation = jax.random.permutation(_rng_perm, config["NUM_STEPS"] * config["NUM_ENVS"])

                def _rl_update_minibatch(train_state, minibatch_indices):
                    # Extract minibatch data using permuted indices
                    traj_batch_mb, adv_mb, targets_mb = jax.tree_map(
                         lambda x: jnp.take(x, minibatch_indices, axis=0), full_batch
                    )

                    def _rl_loss_fn(params, obs, actions, old_log_probs, advantages, targets):
                        # Rerun network with current params
                        z, pi, value = network.apply({'params': params}, obs) # Use network directly here
                        log_prob = pi.log_prob(actions)

                        # Value loss (PPO clipped)
                        value_pred_clipped = traj_batch_mb.value + (value - traj_batch_mb.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Actor loss (PPO clipped)
                        ratio = jnp.exp(log_prob - old_log_probs)
                        gae = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalize advantages
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        # Entropy loss
                        entropy = pi.entropy().mean()

                        # Total RL Loss
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        # Return loss and metrics for logging
                        return total_loss, (value_loss, loss_actor, entropy)

                    # Calculate gradients for RL loss
                    # Gradients will be calculated for all params involved: encoder, actor, critic
                    grad_fn_rl = jax.value_and_grad(_rl_loss_fn, has_aux=True)
                    (total_rl_loss, rl_metrics), grads_rl = grad_fn_rl(
                        train_state.params,
                        traj_batch_mb.obs,
                        traj_batch_mb.action,
                        traj_batch_mb.log_prob,
                        adv_mb,
                        targets_mb
                    )
                    # Apply gradients
                    train_state = train_state.apply_gradients(grads=grads_rl)
                    return train_state, (total_rl_loss, rl_metrics)

                # Reshape permutation for minibatches and scan
                minibatch_indices = permutation.reshape((config["NUM_MINIBATCHES"], -1))
                train_state, rl_losses_and_metrics = jax.lax.scan(
                    _rl_update_minibatch, online_train_state, minibatch_indices
                )
                update_state = (train_state, target_state, rng)
                # Return average loss/metrics over minibatches
                avg_rl_loss = jax.tree_map(jnp.mean, rl_losses_and_metrics[0])
                avg_rl_metrics = jax.tree_map(jnp.mean, rl_losses_and_metrics[1])
                return update_state, (avg_rl_loss, avg_rl_metrics)

            # Run RL update epochs
            rl_update_state_init = (online_train_state, target_state, rng)
            (online_train_state, target_state, rng), rl_epoch_metrics = jax.lax.scan(
                _rl_update_epoch, rl_update_state_init, None, config["UPDATE_EPOCHS_RL"]
            )
            # Average metrics over RL epochs
            avg_rl_step_metrics = jax.tree_map(jnp.mean, rl_epoch_metrics)


            # === JEPA UPDATE PHASE ===
            def _jepa_update_epoch(update_state, unused):
                online_train_state, target_state, rng = update_state
                rng, _rng_perm = jax.random.split(rng)
                permutation = jax.random.permutation(_rng_perm, config["NUM_STEPS"] * config["NUM_ENVS"])

                def _jepa_update_minibatch(train_state, minibatch_indices):
                    # --- CORRECTED MINIBATCH EXTRACTION ---
                    # Directly access the flattened obs and next_obs arrays from full_batch
                    full_obs_batch = full_batch[0].obs
                    full_next_obs_batch = full_batch[0].next_obs

                    # Take the minibatch subset using the indices
                    obs_mb = jnp.take(full_obs_batch, minibatch_indices, axis=0)
                    next_obs_mb = jnp.take(full_next_obs_batch, minibatch_indices, axis=0)
                    # --- END CORRECTION ---

                    def _jepa_loss_fn(online_params, target_encoder_params, obs, next_obs):
                        # Get embeddings
                        z_next_pred_norm, z_next_target_norm = get_jepa_embeddings(
                            online_params, target_encoder_params, obs, next_obs
                        )
                        # Calculate loss
                        loss = compute_jepa_loss(z_next_pred_norm, z_next_target_norm)
                        return loss

                    # Calculate gradients for JEPA loss
                    # Gradients will be calculated only for encoder and predictor params
                    grad_fn_jepa = jax.value_and_grad(_jepa_loss_fn)
                    jepa_loss, grads_jepa = grad_fn_jepa(
                        train_state.params, # Current online params
                        target_state.params, # Current target encoder params
                        obs_mb,              # Use the correctly extracted obs minibatch
                        next_obs_mb          # Use the correctly extracted next_obs minibatch
                    )
                    # Apply gradients (will only affect encoder and predictor)
                    train_state = train_state.apply_gradients(grads=grads_jepa)
                    return train_state, jepa_loss

                # Reshape permutation and scan over minibatches
                minibatch_indices = permutation.reshape((config["NUM_MINIBATCHES"], -1))
                train_state, jepa_losses = jax.lax.scan(
                     _jepa_update_minibatch, online_train_state, minibatch_indices
                )
                update_state = (train_state, target_state, rng)
                # Return average loss over minibatches
                return update_state, jepa_losses.mean()

            # Run JEPA update epochs
            jepa_update_state_init = (online_train_state, target_state, rng)
            (online_train_state, target_state, rng), jepa_epoch_losses = jax.lax.scan(
                _jepa_update_epoch, jepa_update_state_init, None, config["UPDATE_EPOCHS_JEPA"]
            )
            # Average JEPA loss over JEPA epochs
            avg_jepa_step_loss = jepa_epoch_losses.mean()


            # === UPDATE TARGET ENCODER (after all online updates) ===
            new_target_encoder_params = update_target_network(
                config["EMA_DECAY"],
                online_train_state.params['encoder'], # Use the final updated online encoder params
                target_state.params
            )
            target_state = target_state._replace(params=new_target_encoder_params)


            # === PREPARE OUTPUTS ===
            # Collect metrics: RL loss, value loss, actor loss, entropy, JEPA loss, episode returns
            # Combine RL and JEPA metrics
            metrics = {
                "total_rl_loss": avg_rl_step_metrics[0],
                "value_loss": avg_rl_step_metrics[1][0],
                "actor_loss": avg_rl_step_metrics[1][1],
                "entropy": avg_rl_step_metrics[1][2],
                "jepa_loss": avg_jepa_step_loss,
                "returned_episode_returns": traj_batch.info["returned_episode_returns"] # From LogWrapper
            }

            # Reset runner state for next iteration
            runner_state = (online_train_state, target_state, env_state, last_obs, rng)
            return runner_state, metrics


        # === RUN TRAINING LOOP ===
        rng, _rng_run = jax.random.split(rng)
        runner_state_initial = (online_train_state, target_state, env_state, obsv, _rng_run)
        runner_state_final, metrics = jax.lax.scan(
            _update_step, runner_state_initial, None, config["NUM_UPDATES"]
        )
        # metrics dict contains history of all metrics per update step
        return {"runner_state": runner_state_final, "metrics": metrics}

    return train


if __name__ == "__main__":
    config = {
        # Env/Run settings
        "NUM_ENVS": 4,          # Number of parallel environments
        "NUM_STEPS": 128,       # Steps per environment per update
        "TOTAL_TIMESTEPS": 5e5, # Total env interactions
        "ENV_NAME": "Breakout-MinAtar",

        # JEPA specific Hyperparameters
        "EMBED_DIM": 8,        # Latent embedding dimension
        "PREDICTOR_HIDDEN_DIM": 16,# Predictor MLP hidden size
        "EMA_DECAY": 0.99,      # Target encoder EMA decay

        # PPO specific Hyperparameters
        "GAMMA": 0.99,          # Discount factor
        "GAE_LAMBDA": 0.95,     # GAE lambda
        "CLIP_EPS": 0.2,        # PPO clipping epsilon
        "ENT_COEF": 0.01,       # Entropy coefficient
        "VF_COEF": 0.5,         # Value function coefficient

        # Optimizer/Training settings
        "LR": 3e-4,             # Learning rate for Adam
        "UPDATE_EPOCHS_RL": 4,  # PPO update epochs per data batch
        "UPDATE_EPOCHS_JEPA": 4,# JEPA update epochs per data batch
        "NUM_MINIBATCHES": 4,   # Minibatches per epoch (for both RL and JEPA)
        "MAX_GRAD_NORM": 0.5,   # Global gradient clipping norm
        "ACTIVATION": "tanh",   # Activation for hidden layers
        "ANNEAL_LR": True,      # Linearly anneal learning rate
    }

    rng = jax.random.PRNGKey(42)
    train_fn = make_train(config)

    # Jitting the combined training loop
    # Can take a while, disable for easier debugging: train_fn = make_train(config)
    print("Jitting the training function...")
    train_jit = jax.jit(train_fn)
    print("Jitting complete. Starting training...")
    out = train_jit(rng)
    print("Training finished.")

    # --- Plotting ---
    metrics = out["metrics"]

    # Plot episode returns (sanity check RL progress)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    # Ensure slicing is correct based on LogWrapper output shape
    # Shape is likely (num_updates, num_steps, num_envs) - average over steps/envs?
    # Or maybe (num_updates, num_envs) if LogWrapper aggregates? Let's assume it aggregates per update step.
    # Check shape: print(metrics["returned_episode_returns"].shape)
    # If shape is (num_updates, num_steps, num_envs), average:
    avg_returns = jnp.mean(metrics["returned_episode_returns"], axis=(1, 2))
    # If shape is (num_updates, num_envs)
    # avg_returns = jnp.mean(metrics["returned_episode_returns"], axis=1)
    # Assuming shape is (num_updates, num_steps, num_envs), taking mean over last 2 dims:
    try:
        # LogWrapper usually gives stats per step, filter for actual episode ends
        avg_returns = jnp.nanmean(jnp.where(metrics["returned_episode_returns"] > -jnp.inf, metrics["returned_episode_returns"], jnp.nan), axis=(1,2))
        plt.plot(avg_returns)
    except Exception as e:
        print(f"Could not plot returns automatically (shape might be different): {e}")
        # Try plotting the raw mean if the above fails
        plt.plot(jnp.mean(metrics["returned_episode_returns"], axis=1))


    plt.title("Episode Returns")
    plt.xlabel("Update Step")
    plt.ylabel("Average Return")
    plt.grid(True)

    # Plot JEPA loss
    plt.subplot(2, 2, 2)
    plt.plot(metrics["jepa_loss"])
    plt.title("JEPA Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.grid(True)

    # Plot Value loss
    plt.subplot(2, 2, 3)
    plt.plot(metrics["value_loss"])
    plt.title("Value Function Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.grid(True)

     # Plot Actor loss
    plt.subplot(2, 2, 4)
    plt.plot(metrics["actor_loss"])
    plt.title("Actor Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()