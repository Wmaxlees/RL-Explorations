# main.py

import argparse
import os
import sys
import time
from functools import partial # For cleaner vmap/jit usage

import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name
from gymnasium.spaces import Box, Discrete # Import spaces for checks

from gymnasium.spaces import Box, Discrete # Keep this for general checks
try:
    # Also import the Discrete space specifically from gymnax
    from gymnax.environments.spaces import Discrete as GymnaxDiscrete
except ImportError:
    # Handle case where gymnax might not be installed or spaces moved
    print("Warning: Could not import Discrete from gymnax.environments.spaces. Type checking might be limited.")
    GymnaxDiscrete = None # Set to None if unavailable

import wandb
from typing import NamedTuple, Any, Dict, Optional

from flax.training import orbax_utils
from flax.training.train_state import TrainState # Use standard TrainState
import flax.struct # Use flax.struct for RunnerState to work better with JIT

from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

# Assume logz is a local module/package for logging helpers
try:
    from logz.batch_logging import batch_log, create_log_dict
except ImportError:
    print("Warning: logz module not found. WandB logging might be affected.")
    # Define dummy functions if logz is not available
    def create_log_dict(metric, config): return metric
    def batch_log(step, data, config): print(f"Step {step}: {data}")


# Import Actor-Critic models from the models module
# Ensure models directory is in the Python path or use relative imports if structured as a package
try:
    from models.actor_critic import (
        ActorCritic,
        ActorCriticConv,
        ActorCriticJEPA,
        ActorCriticForwardDynamics
    )
except ImportError as e:
     print(f"Error importing actor_critic models: {e}")
     print("Ensure 'models/actor_critic.py' and 'models/jepa.py' exist and are accessible.")
     sys.exit(1)

# Import environment wrappers
# Ensure wrappers.py is accessible
try:
    from wrappers import (
        LogWrapper,
        OptimisticResetVecEnvWrapper,
        BatchEnvWrapper,
        AutoResetEnvWrapper,
    )
except ImportError as e:
     print(f"Error importing wrappers: {e}")
     print("Ensure 'wrappers.py' exists and is accessible.")
     sys.exit(1)


# Transition tuple remains the same
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray # Combined reward (extrinsic + potential intrinsic)
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: Dict # Env info


# Use flax.struct for RunnerState to handle JIT compatibility better, especially with None types
@flax.struct.dataclass
class RunnerState:
    train_state: TrainState # Combined PPO + JEPA online params
    env_state: Any # Environment internal state
    last_obs: jnp.ndarray
    aux_target_params: Optional[Dict]
    rng: jax.random.PRNGKey
    update_step: int


def make_train(config: Dict):
    """Creates the training function."""

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # --- Environment Setup ---
    # Use partial to pass params easily
    env_factory = partial(make_craftax_env_from_name, config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"])
    env = env_factory() # Create one instance to get specs
    env_params = env.default_params
    env_action_space = env.action_space(env_params)
    env_obs_space = env.observation_space(env_params)

    # Validate action space
    valid_discrete_types = tuple(filter(None, [Discrete, GymnaxDiscrete])) # Filter out None if import failed
    if not isinstance(env_action_space, valid_discrete_types):
        raise ValueError(f"Action space must be an instance of {valid_discrete_types}, got {type(env_action_space)}")
    action_dim = env_action_space.n

    # --- Environment and JEPA Compatibility Check ---
    is_pixel_env = isinstance(env_obs_space, Box) and len(env_obs_space.shape) == 3
    is_symbolic_env = not is_pixel_env # Assuming anything not Box/3D is symbolic-like

    config["AUX_LOSS_MODE"] = None
    if config["USE_AUX_LOSS"]:
         # Simple example: Use Forward Dynamics for vector envs, Image JEPA for pixels
         # Could be made more flexible with another argument like --aux_loss_type
         if is_symbolic_env:
             config["AUX_LOSS_MODE"] = "forward_dynamics_vector"
             config["INPUT_DIM"] = env_obs_space.shape[0] # Assuming Box
             print(f"Auxiliary Loss Enabled: Forward Dynamics (Vector) - Input Dim {config['INPUT_DIM']}")
         elif is_pixel_env:
             config["AUX_LOSS_MODE"] = "jepa_image"
             config["IMAGE_SHAPE"] = env_obs_space.shape
             print(f"Auxiliary Loss Enabled: JEPA (Image) - Image Shape {config['IMAGE_SHAPE']}")
         else:
             raise ValueError("Auxiliary loss enabled, but env is neither recognized vector nor pixel type.")
    else:
         print("Auxiliary Loss Disabled.")

    # --- Vectorized Environment ---
    # Re-create the env using the factory for wrapping
    env = env_factory()
    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        # Check ratio validity
        reset_ratio = min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"])
        if reset_ratio <= 0:
             print("Warning: optimistic_reset_ratio <= 0, disabling optimistic resets.")
             env = AutoResetEnvWrapper(env)
             env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])
        else:
             env = OptimisticResetVecEnvWrapper(
                 env,
                 num_envs=config["NUM_ENVS"],
                 reset_ratio=reset_ratio,
             )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])


    # --- Learning Rate Schedule ---
    def linear_schedule(count):
        # Annealing based on the number of PPO+JEPA updates
        updates_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
        # Frac decreases over total PPO updates
        frac = 1.0 - (count // updates_per_epoch) / config["NUM_UPDATES"]
        return config["LR"] * frac


    # === Main Training Function Definition ===
    def train(rng: jax.random.PRNGKey):
        """JIT-compilable training loop."""
        rng, _rng_net_init = jax.random.split(rng)
        init_x = jnp.zeros((1, *env_obs_space.shape), dtype=env_obs_space.dtype)
        aux_target_params = None # Renamed
        network = None

        # === NETWORK INITIALIZATION ===
        if config["AUX_LOSS_MODE"] == "forward_dynamics_vector":
            print("Initializing ActorCriticForwardDynamics Network...")
            network = ActorCriticForwardDynamics(
                action_dim=action_dim, layer_width=config["LAYER_SIZE"],
                fd_input_dim=config["INPUT_DIM"],
                fd_encoder_output_dim=config["AUX_ENCODER_DIM"], # Use generic aux args
                fd_encoder_hidden_dim=config["AUX_ENCODER_HIDDEN"],
                fd_encoder_layers=config["AUX_ENCODER_LAYERS"],
                fd_predictor_hidden_dim=config["AUX_PREDICTOR_HIDDEN"],
                fd_predictor_layers=config["AUX_PREDICTOR_LAYERS"]
            )
            # Init requires dummy target params and potentially dummy action/next_obs if apply needs them
            # For simplicity, assume init only needs obs shape (Flax usually handles this)
            rng, _rng_aux_enc = jax.random.split(rng, 2)
            # Dummy init call - might need adjustment based on exact ActorCriticForwardDynamics apply signature if it uses more args during init
            dummy_encoder_params = network.forward_dynamics.encoder.init(_rng_aux_enc, init_x)['params']
            network_params = network.init(_rng_net_init, init_x, jnp.zeros(1, dtype=jnp.int32), init_x, dummy_encoder_params)["params"] # Provide dummy args for init call
            aux_target_params = jax.tree.map(lambda x: x, network_params['forward_dynamics_module']['online_encoder_mlp'])

        elif config["AUX_LOSS_MODE"] == "jepa_image":
            # ... (Initialization for ActorCriticJEPA as before) ...
             print("Initializing ActorCriticJEPA Network...")
             network = ActorCriticJEPA( # Assuming ActorCriticJEPA still exists
                 action_dim=action_dim, layer_width=config["LAYER_SIZE"],
                 jepa_encoder_output_dim=config["AUX_ENCODER_DIM"],
                 jepa_predictor_hidden_dim=config["AUX_PREDICTOR_HIDDEN"],
                 jepa_predictor_layers=config["AUX_PREDICTOR_LAYERS"],
                 jepa_image_shape=config["IMAGE_SHAPE"]
             )
             rng, _rng_jepa_enc, _rng_jepa_mask = jax.random.split(rng, 3)
             dummy_encoder_params = network.jepa.encoder.init(_rng_jepa_enc, init_x)['params']
             network_params = network.init(_rng_net_init, init_x, dummy_encoder_params, _rng_jepa_mask)["params"]
             aux_target_params = jax.tree.map(lambda x: x, network_params['jepa_module']['online_encoder'])


        elif is_pixel_env: # No Aux Loss, Pixel Env
            print("Initializing ActorCriticConv Network...")
            network = ActorCriticConv(action_dim=action_dim, layer_width=config["LAYER_SIZE"])
            network_params = network.init(_rng_net_init, init_x)["params"]
        else: # No Aux Loss, Vector Env
            print("Initializing ActorCritic (MLP) Network...")
            network = ActorCritic(action_dim=action_dim, layer_width=config["LAYER_SIZE"], activation=config["ACTIVATION"])
            network_params = network.init(_rng_net_init, init_x)["params"]

        if network is None: raise RuntimeError("Network not initialized.")


        # === OPTIMIZER ===
        # Optimizer handles all parameters in network_params (PPO heads, JEPA online/predictor if used)
        if config["ANNEAL_LR"]:
            print("Using linearly annealed learning rate.")
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            print(f"Using constant learning rate: {config['LR']}")
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        # === TRAIN STATE ===
        # Contains network parameters and optimizer state
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        print("TrainState created.")


        # === ENVIRONMENT RESET ===
        rng, _rng_reset = jax.random.split(rng)
        obsv, env_state = env.reset(_rng_reset, env_params)


        # === UPDATE STEP FUNCTION (to be scanned over) ===
        def _update_step(runner_state: RunnerState, unused):
            """Performs one PPO+JEPA update step."""

            # --- COLLECT TRAJECTORIES ---
            def _env_step(runner_state_in_scan: RunnerState, unused_step_arg):
                """Runs one step in the environment."""
                # Unpack runner state
                train_state_scan = runner_state_in_scan.train_state
                env_state_scan = runner_state_in_scan.env_state
                last_obs_scan = runner_state_in_scan.last_obs
                aux_target_params_scan = runner_state_in_scan.aux_target_params
                rng_scan = runner_state_in_scan.rng

                # SELECT ACTION
                rng_scan, _rng_action, _rng_net_call = jax.random.split(rng_scan, 3)
                if config["AUX_LOSS_MODE"] == "forward_dynamics_vector":
                    # Apply method needs obs_t, action_t, obs_t_plus_1, target_params
                    # For ACTION SELECTION based on obs_t, we only need pi & value.
                    # Pass dummy values for args not relevant to pi/value calculation from obs_t.
                    # IMPORTANT: This assumes the internal network structure allows pi/value
                    # calculation without depending on action_t/obs_t_plus_1 inputs.
                    dummy_action = jnp.zeros_like(action) # Use action shape from previous step or default
                    dummy_next_obs = last_obs_scan
                    pi, value, _ = network.apply(
                        {'params': train_state_scan.params},
                        aux_target_params_scan,
                        last_obs_scan,    # obs_t
                        dummy_action,     # dummy action_t
                        dummy_next_obs    # dummy obs_t_plus_1
                    )
                elif config["AUX_LOSS_MODE"] == "jepa_image":
                    # Apply method needs obs, target_params, rng
                    pi, value, _ = network.apply(
                        {'params': train_state_scan.params},
                        aux_target_params_scan,
                        last_obs_scan,
                        _rng_net_call # Pass RNG
                    )
                else: # No aux loss or unknown mode (treat as no aux loss for action selection)
                    # Standard ActorCritic or ActorCriticConv call
                    pi, value = network.apply({'params': train_state_scan.params}, last_obs_scan)

                action = pi.sample(seed=_rng_action)
                log_prob = pi.log_prob(action)

                # STEP ENVIRONMENT
                rng_scan, _rng_env_step = jax.random.split(rng_scan)
                obsv_scan, env_state_scan, reward_e, done, info = env.step(
                    _rng_env_step, env_state_scan, action, env_params
                )

                # REWARD (Currently just extrinsic reward)
                # Placeholder: Add intrinsic reward calculation here if desired
                # reward_i = calculate_intrinsic_reward(...)
                # reward = reward_e + reward_i * config["INTRINSIC_REWARD_COEFF"]
                reward = reward_e # Using only extrinsic reward

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    log_prob=log_prob,
                    obs=last_obs_scan, # Store the observation *before* the step
                    next_obs=obsv_scan, # Store the observation *after* the step
                    info=info,
                )
                # Update runner state for next env step
                runner_state_out_scan = runner_state_in_scan.replace(
                    env_state=env_state_scan,
                    last_obs=obsv_scan,
                    rng=rng_scan
                 )
                return runner_state_out_scan, transition

            # Scan over NUM_STEPS to collect a batch of trajectories
            runner_state_after_scan, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )


            # --- CALCULATE ADVANTAGES ---
            # Need value prediction for the very last observation collected
            rng = runner_state_after_scan.rng
            rng, _rng_last_val_net = jax.random.split(rng)
            if config["AUX_LOSS_MODE"] == "forward_dynamics_vector":
                 # ActorCriticForwardDynamics apply needs obs_t, action_t, obs_t_plus_1, target_params
                 # For *just* getting value, we only need obs_t (for policy/value heads).
                 # We might need a separate method or adapt the apply call.
                 # Let's assume get_embedding works and we can apply heads separately (cleaner)
                 # OR adapt the apply call to handle value-only prediction.
                 # Simple approach: Pass dummy values for unused args if apply must run fully
                 dummy_action = jnp.zeros_like(traj_batch.action[0]) # Shape (B,)
                 dummy_next_obs = runner_state_after_scan.last_obs
                 pi_last, last_val, _ = network.apply(
                     {'params': runner_state_after_scan.train_state.params},
                     runner_state_after_scan.aux_target_params, # Pass target params
                     runner_state_after_scan.last_obs, # obs_t
                     dummy_action,                     # dummy action_t
                     dummy_next_obs,                   # dummy obs_t_plus_1
                 )
            elif config["AUX_LOSS_MODE"] == "jepa_image":
                 # ActorCriticJEPA needs obs, target_params, rng
                  _, last_val, _ = network.apply(
                      {'params': runner_state_after_scan.train_state.params},
                      runner_state_after_scan.aux_target_params,
                      runner_state_after_scan.last_obs,
                      _rng_last_val_net
                   )
            else:
                 # Standard ActorCritic or ActorCriticConv just need obs
                  _, last_val = network.apply(
                      {'params': runner_state_after_scan.train_state.params},
                      runner_state_after_scan.last_obs
                  )

            def _calculate_gae(traj_batch_gae: Transition, last_val_gae: jnp.ndarray):
                """Calculates GAE."""
                # Inner function for scan
                def _get_advantages(gae_and_next_value, transition: Transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    # Calculate delta and GAE for this step
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae # Return new state and GAE for this step

                # Scan backward through transitions
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val_gae), last_val_gae), # Initial state (gae=0, V(s_T))
                    traj_batch_gae, # Transitions (T-1, ..., 0)
                    reverse=True,
                    unroll=16, # Optimization hint
                )
                # Advantages are computed; calculate targets (V(s) + A(s,a))
                targets = advantages + traj_batch_gae.value
                return advantages, targets

            advantages, targets = _calculate_gae(traj_batch, last_val)


            # === UPDATE NETWORK (PPO + optional JEPA) ===
            def _update_epoch(update_state_epoch, unused_epoch_arg):
                """Scans over minibatches for one epoch."""
                train_state_epoch, aux_target_params_epoch, rng_epoch = update_state_epoch
                
                # --- Define Loss Function (incorporating JEPA if enabled) ---
                def _loss_fn(params, aux_target_params_loss, traj_batch_loss: Transition, gae, targets_loss, rng_loss):
                    """Calculates the combined PPO and JEPA loss."""
                    if config["AUX_LOSS_MODE"] == "forward_dynamics_vector":
                        aux_loss = 0.0
                        # Requires obs_t, action_t, obs_t_plus_1, target_params
                        pi, value, aux_loss = network.apply(
                            {'params': params}, aux_target_params_loss,
                            traj_batch_loss.obs, traj_batch_loss.action, traj_batch_loss.next_obs
                        )
                    elif config["AUX_LOSS_MODE"] == "jepa_image":
                         # Requires obs, target_params, rng
                         rng_loss, _rng_net = jax.random.split(rng_loss)
                         pi, value, aux_loss = network.apply(
                             {'params': params}, aux_target_params_loss,
                             traj_batch_loss.obs, _rng_net
                         )
                    else: # No aux loss
                        pi, value = network.apply({'params': params}, traj_batch_loss.obs)

                    log_prob = pi.log_prob(traj_batch_loss.action)

                    # --- PPO Value Loss (Clipped) ---
                    value_pred_clipped = traj_batch_loss.value + (
                        value - traj_batch_loss.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets_loss)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets_loss)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    # --- PPO Actor Loss (Clipped Surrogate Objective) ---
                    ratio = jnp.exp(log_prob - traj_batch_loss.log_prob)
                    # Normalize advantages per minibatch
                    gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae_norm
                    loss_actor2 = jnp.clip(
                        ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]
                    ) * gae_norm
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                    # --- PPO Entropy Bonus ---
                    entropy = pi.entropy().mean()

                    # --- Total Joint Loss ---
                    total_loss = (
                        loss_actor
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    # Add JEPA loss component if JEPA is enabled
                    total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                    loss_key = "aux_loss" # Generic key
                    if config["USE_AUX_LOSS"]:
                        total_loss += config["AUX_LOSS_COEF"] * aux_loss # Use generic coeff

                    aux_losses_dict = {"ppo_value_loss": value_loss, "ppo_actor_loss": loss_actor, "ppo_entropy": entropy, "total_joint_loss": total_loss}
                    if config["USE_AUX_LOSS"]: aux_losses_dict[loss_key] = aux_loss
                    return total_loss, aux_losses_dict

                # --- Minibatch Update Function ---
                def _update_minibatch(train_state_and_rng_mb, batch_info_mb):
                    """Performs update for a single minibatch."""
                    train_state_mb, rng_mb = train_state_and_rng_mb
                    traj_mb, advantages_mb, targets_mb = batch_info_mb
                    rng_mb, _rng_grad = jax.random.split(rng_mb)

                    # Calculate gradients using the combined loss function
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss_mb, aux_losses_mb), grads = grad_fn(
                        train_state_mb.params,
                        aux_target_params_epoch, # Pass non-optimized target params
                        traj_mb,
                        advantages_mb,
                        targets_mb,
                        _rng_grad # Pass RNG for loss calculation
                    )
                    # Apply gradients -> updates PPO heads + JEPA online encoder/predictor
                    train_state_mb = train_state_mb.apply_gradients(grads=grads)

                    # Return updated state and losses for this minibatch
                    return (train_state_mb, rng_mb), aux_losses_mb

                # --- Shuffle and Iterate Minibatches ---
                rng_epoch, _rng_perm = jax.random.split(rng_epoch)
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                permutation = jax.random.permutation(_rng_perm, batch_size)

                # Flatten trajectory data for shuffling
                batch_data = (traj_batch, advantages, targets)
                batch_flat = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch_data
                )
                # Shuffle data
                shuffled_flat = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch_flat
                )
                # Reshape into minibatches
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_flat,
                )

                # Scan over minibatches to perform updates
                (train_state_after_mb_scan, rng_after_mb_scan), loss_info_batch = jax.lax.scan(
                    _update_minibatch, (train_state_epoch, rng_epoch), minibatches
                )

                # --- Update JEPA Target Encoder using EMA (if JEPA is enabled) ---
                # This happens *after* all minibatch gradient updates for the epoch are done.
                aux_target_params_after_ema = aux_target_params_epoch # Default if not using JEPA
                if config["USE_AUX_LOSS"]:
                    def _ema_update(target_p, online_p):
                        return target_p * config["AUX_EMA_DECAY"] + online_p * (1.0 - config["AUX_EMA_DECAY"]) # Use generic EMA decay arg

                    # Determine the correct path to the online encoder parameters
                    online_encoder_params = None
                    if config["AUX_LOSS_MODE"] == "forward_dynamics_vector":
                         online_encoder_params = train_state_after_mb_scan.params['forward_dynamics_module']['online_encoder_mlp']
                    elif config["AUX_LOSS_MODE"] == "jepa_image":
                         online_encoder_params = train_state_after_mb_scan.params['jepa_module']['online_encoder']

                    if online_encoder_params is not None:
                         updated_target_params = jax.tree.map(_ema_update, aux_target_params_epoch, online_encoder_params)
                         aux_target_params_after_ema = updated_target_params
                    # else: Error already handled during init or mode selection

                update_state_out_epoch = (train_state_after_mb_scan, aux_target_params_after_ema, rng_after_mb_scan)
                return update_state_out_epoch, loss_info_batch

            # --- Scan over Update Epochs ---
            # Initial state for the epoch scan
            update_state_init_epoch = (
                runner_state_after_scan.train_state, # Use train_state after trajectory collection
                runner_state_after_scan.aux_target_params, # Pass current target params
                rng, # Pass current RNG
            )
            # Run the epoch scan
            update_state_final_epoch, loss_info_epochs = jax.lax.scan(
                _update_epoch, update_state_init_epoch, None, config["UPDATE_EPOCHS"]
            )

            # --- Extract final states after all epochs ---
            train_state_final = update_state_final_epoch[0]
            aux_target_params_final = update_state_final_epoch[1]
            rng_final = update_state_final_epoch[2]
            current_update_step = runner_state_after_scan.update_step # Get current update step number


            # === LOGGING ===
            # Calculate average environment metrics over the collected batch
            # Mask metrics by where episodes actually ended during the rollout
            ep_ended_mask = traj_batch.info["returned_episode"]
            # Add epsilon to prevent division by zero if no episodes ended
            valid_ep_count = ep_ended_mask.sum() + 1e-8
            env_metrics = jax.tree.map(
                lambda x: (x * ep_ended_mask).sum() / valid_ep_count,
                traj_batch.info # Use the info dict collected in traj_batch
            )

            # Aggregate PPO/JEPA losses (mean over epochs and minibatches)
            # loss_info_epochs has shape (UPDATE_EPOCHS, NUM_MINIBATCHES, {loss_dict})
            # We want the mean of each loss type across all epochs/minibatches
            agg_losses = jax.tree.map(lambda x: x.mean(), loss_info_epochs)

            # Combine metrics for logging
            metrics_to_log = env_metrics.copy() # Start with env metrics
            metrics_to_log.update(agg_losses) # Add aggregated losses
            metrics_to_log["update_step"] = current_update_step # Add step number
            metrics_to_log["total_episodes"] = ep_ended_mask.sum() # Log how many episodes ended

            # --- WANDB Logging Callback ---
            if config["DEBUG"] and config["USE_WANDB"]:
                # Define the callback function to log metrics
                def wandb_callback(metrics, step):
                    # Use logz helper or simple wandb.log
                    log_payload = create_log_dict(metrics, config) # Prepare log dict if needed
                    # Log directly using wandb.log, ensuring step is passed
                    wandb.log(log_payload, step=step)

                # Use jax.debug.callback to trigger logging from JITted code
                jax.debug.callback(
                    wandb_callback,
                    metrics_to_log,
                    current_update_step # Pass the step number for correct logging
                )

            # --- Prepare state for the next PPO update step ---
            runner_state_out = RunnerState(
                train_state=train_state_final,
                env_state=runner_state_after_scan.env_state, # Use env state after trajectory collection
                last_obs=runner_state_after_scan.last_obs,   # Use last observation after trajectory collection
                aux_target_params=aux_target_params_final, # Pass updated target params
                rng=rng_final, # Pass the final RNG state
                update_step=current_update_step + 1, # Increment update step counter
            )
            # Return the runner state and the metrics collected during this update
            return runner_state_out, metrics_to_log

        # === INITIALIZE RUNNER STATE ===
        rng, _rng_init_runner = jax.random.split(rng)
        runner_state_initial = RunnerState(
            train_state=train_state,
            env_state=env_state,
            last_obs=obsv,
            aux_target_params=aux_target_params, # Initial target params (can be None)
            rng=_rng_init_runner,
            update_step=0,
        )
        print("Initial Runner State created.")

        # === MAIN TRAINING LOOP (scan over update steps) ===
        print(f"Starting training loop for {config['NUM_UPDATES']} updates...")
        runner_state_final, metric_history = jax.lax.scan(
            _update_step, runner_state_initial, None, config["NUM_UPDATES"]
        )
        print("Training loop finished.")

        # Return the final runner state and the history of metrics
        return {"runner_state": runner_state_final, "metrics": metric_history}

    # Return the train function, potentially JITted later
    return train


# === Run Experiment Function ===
def run_ppo(config_obj: argparse.Namespace):
    """Sets up and runs the PPO training experiment."""
    config = {k.upper(): v for k, v in vars(config_obj).items()}

    # --- WandB Initialization ---
    if config["USE_WANDB"]:
        run_name = config["ENV_NAME"] + f"-{int(config['TOTAL_TIMESTEPS'] // 1e6)}M-{config['SEED']}"
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config, # Log the experiment configuration
            name=run_name,
        )
        print("WandB initialized (Basic).")
    else:
        print("WandB Disabled.")

    # --- RNG Setup ---
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"]) # One RNG key per repeat

    # --- Create and Compile Training Function ---
    # Get the train function builder
    train_fn_builder = make_train(config)

    # Apply JIT compilation if enabled
    if config["JIT"]:
        # Partial prevents re-passing config dict during vmap/jit
        train_fn_compiled = jax.jit(train_fn_builder)
        print("JIT compiling the training function...")
    else:
        train_fn_compiled = train_fn_builder # No JIT
        print("Running without JIT compilation...")

    # Apply vmap for parallel repeats
    train_vmap = jax.vmap(train_fn_compiled)
    print(f"Prepared training function for {config['NUM_REPEATS']} repeats.")

    # --- Execute Training ---
    t0 = time.time()
    print("Starting training execution...")
    # Use block_until_ready to ensure computation finishes for accurate timing and error catching
    output = jax.block_until_ready(train_vmap(rngs))
    t1 = time.time()
    duration = t1 - t0
    print(f"Training execution finished. Time taken: {duration:.2f} seconds")

    # Calculate and print Steps Per Second (SPS)
    total_steps_processed = config["TOTAL_TIMESTEPS"] * config["NUM_REPEATS"]
    sps = total_steps_processed / duration if duration > 0 else 0
    print(f"Total Timesteps: {total_steps_processed:.2e}")
    print(f"Steps Per Second (SPS): {sps:.2f}")
    if config["USE_WANDB"]:
         wandb.summary["experiment_duration_seconds"] = duration
         wandb.summary["steps_per_second"] = sps

    # --- Save Policy ---
    if config["SAVE_POLICY"] and config["USE_WANDB"]:
        print("Saving final policy...")
        # Saving the state from the first repeat (index 0)
        try:
            # Extract the runner_state structure for the first repeat
            final_runner_state_first_repeat = jax.tree.map(lambda x: x[0], output["runner_state"])
            final_train_state = final_runner_state_first_repeat.train_state
            # Optional: Save JEPA target params if they exist
            final_target_params = final_runner_state_first_repeat.aux_target_params

            orbax_checkpointer = PyTreeCheckpointer()
            # Use wandb.run.dir which is automatically synced if WandB is enabled
            save_dir = os.path.join(wandb.run.dir, "final_policy")
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            checkpoint_manager = CheckpointManager(save_dir, orbax_checkpointer, options)

            # Save the TrainState (contains policy, value, JEPA online params)
            save_args = orbax_utils.save_args_from_target(final_train_state)
            save_step = config["TOTAL_TIMESTEPS"] # Use total timesteps as the save step identifier
            checkpoint_manager.save(
                save_step,
                final_train_state,
                save_kwargs={"save_args": save_args},
            )
            print(f"Saved final TrainState to {save_dir}/<{save_step}>")

            # Optionally save target params separately
            if final_target_params is not None:
                 target_save_path = os.path.join(save_dir, f"{save_step}_jepa_target")
                 orbax_checkpointer.save(target_save_path, final_target_params)
                 print(f"Saved final JEPA target params to {target_save_path}")

            checkpoint_manager.wait_until_finished() # Ensure saving completes
            print("Policy saving finished.")
        except Exception as e:
            print(f"Error saving policy: {e}")

    # --- Finalize WandB ---
    if config["USE_WANDB"]:
        wandb.finish()
        print("WandB run finished.")


# === Argument Parsing and Main Execution ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO+JEPA Training Script for Craftax")

    # --- Environment Args ---
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1", help="Environment name (e.g., Craftax-Pixels-v1, Craftax-Symbolic-v1)")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")

    # --- PPO Algorithm Args ---
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e8, help="Total training timesteps (e.g., 1e8)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--num_steps", type=int, default=128, help="Number of steps per environment for PPO rollout")
    parser.add_argument("--update_epochs", type=int, default=4, help="Number of epochs to update policy/value per PPO batch")
    parser.add_argument("--num_minibatches", type=int, default=4, help="Number of minibatches to split PPO batch into")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh"], help="Activation function for MLP networks (if not using Conv/JEPA)")
    parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=True, help="Anneal learning rate linearly")

    # --- JEPA Specific Args ---
    parser.add_argument("--use_aux_loss", action="store_true", help="Enable auxiliary loss (JEPA-Image or ForwardDynamics-Vector based on env)")
    parser.add_argument("--aux_encoder_dim", type=int, default=256, help="Output dim of auxiliary task encoder (MLP/CNN)")
    parser.add_argument("--aux_predictor_hidden", type=int, default=512, help="Hidden dim for aux task predictor MLP")
    parser.add_argument("--aux_predictor_layers", type=int, default=3, help="Num layers in aux task predictor MLP")
    parser.add_argument("--aux_ema_decay", type=float, default=0.996, help="EMA decay for aux task target encoder")
    parser.add_argument("--aux_loss_coef", type=float, default=1.0, help="Auxiliary loss coefficient")
    parser.add_argument("--aux_input_dim", type=int, default=None, help="Input dim for Vector aux task (optional, detected if possible)")
    parser.add_argument("--aux_encoder_hidden", type=int, default=512, help="Hidden dim for Vector aux task MLP encoder")
    parser.add_argument("--aux_encoder_layers", type=int, default=3, help="Num hidden layers in Vector aux task MLP encoder")

    # --- Execution & Logging Args ---
    parser.add_argument("--seed", type=int, default=None, help="Random seed (if None, chosen randomly)")
    parser.add_argument("--num_repeats", type=int, default=1, help="Number of parallel experiment repeats")
    parser.add_argument("--layer_size", type=int, default=512, help="Hidden layer size for Actor/Critic MLP heads")
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=True, help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, # Default is None
                    help="WandB entity (username/team). Default: Determined by login.")
    parser.add_argument("--save_policy", action="store_true", help="Save final policy TrainState to WandB run dir")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True, help="Enable debug features (e.g., WandB callback)")
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True, help="Enable JIT compilation")

    # --- Misc Args ---
    parser.add_argument("--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=False, help="Use Optimistic Reset wrapper")
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16, help="Ratio of envs to reset optimistically")


    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        print(f"Warning: Unknown arguments received: {unknown_args}")

    # --- Seed Handling ---
    if args.seed is None:
        args.seed = np.random.randint(0, 2**31 - 1)
        print(f"Generated random seed: {args.seed}")

    # --- Run Experiment ---
    print("Starting experiment with configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    run_ppo(args)

    print("Experiment finished.")