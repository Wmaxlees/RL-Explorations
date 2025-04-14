import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gymnasium as gym
from gymnasium import wrappers
import collections
import tqdm
import matplotlib.pyplot as plt
import ale_py
import random # Needed if using minibatches

# --- Hyperparameters ---
ENV_NAME = "ALE/Breakout-v5"

# Architecture Dimensions
EMBEDDING_DIM = 256
PREDICTOR_HIDDEN_DIM = 512
ENCODER_HIDDEN_DIM = 512
HIDDEN_DIM_AC = 512

# JEPA Specific
TARGET_EMA_TAU = tf.constant(0.995, dtype=tf.float32)
JEPA_LR = 1e-4

# --- PPO Specific ---
PPO_LR = 1e-4
GAMMA = 0.99
N_STEPS = 1024
NUM_EPOCHS_PER_UPDATE = 4
MINIBATCH_SIZE = 256
PPO_CLIP_EPSILON = 0.1
GAE_LAMBDA = 0.95
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01

# --- Training Control ---
TOTAL_STEPS = 1_000_000
MAX_STEPS_PER_EPISODE = 18000
REWARD_THRESHOLD = 30
PRINT_FREQ_STEPS = 100
LOG_FREQ_STEPS = 1000

# --- Model Definitions ---
def create_recurrent_encoder(state_shape, embedding_dim, cnn_filters, dense_hidden_dim, name="RecurrentEncoder"):
    state_input = keras.Input(shape=state_shape, name="state_input", dtype=tf.float32)
    prev_embedding_input = keras.Input(shape=(embedding_dim,), name="prev_embedding_input", dtype=tf.float32)
    normalized_state = state_input / 255.0
    x = layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu')(normalized_state)
    x = layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(x)
    cnn_output = layers.Flatten()(x)
    # processed_prev_embedding = layers.Dense(embedding_dim // 2, activation='relu')(prev_embedding_input)
    #concatenated = layers.Concatenate()([cnn_output, processed_prev_embedding])
    # x = layers.Dense(dense_hidden_dim, activation='relu')(concatenated)
    x = layers.Dense(dense_hidden_dim, activation='relu')(cnn_output)
    outputs = layers.Dense(embedding_dim, activation=None)(x)
    outputs = layers.LayerNormalization()(outputs)
    return keras.Model(inputs=[state_input, prev_embedding_input], outputs=outputs, name=name)

def create_predictor(embedding_dim, hidden_dim, name="Predictor"):
    inputs = keras.Input(shape=(embedding_dim,), dtype=tf.float32)
    x = layers.Dense(hidden_dim, activation='relu')(inputs)
    outputs = layers.Dense(embedding_dim, activation=None)(x)
    return keras.Model(inputs, outputs, name=name)

def create_actor_critic_heads(embedding_dim, hidden_dim, num_actions, name="ActorCritic"):
    inputs = keras.Input(shape=(embedding_dim,), dtype=tf.float32)
    shared = layers.Dense(hidden_dim, activation='relu', name="ac_hidden")(inputs)
    action_logits = layers.Dense(num_actions, activation=None, name="actor_output")(shared)
    state_value = layers.Dense(1, activation=None, name="critic_output")(shared)
    return keras.Model(inputs, [action_logits, state_value], name=name)

# --- Utility Functions ---
def update_target_network_tf(online_network, target_network, tau):
    tau = tf.cast(tau, dtype=tf.float32)
    for online_var, target_var in zip(online_network.weights, target_network.weights):
        new_target_var_value = tau * target_var + (1.0 - tau) * online_var
        target_var.assign(new_target_var_value)
mse_loss = tf.keras.losses.MeanSquaredError()
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

# --- GAE Calculation ---
def calculate_gae(rewards, values, next_values, dones, gamma, lambda_):
    """Calculates Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0
    last_value = next_values # Value estimate after the last step
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t] # 0 if done, 1 otherwise
        # TD Error (delta)
        delta = rewards[t] + gamma * last_value * mask - values[t]
        # GAE Advantage
        advantages[t] = last_advantage = delta + gamma * lambda_ * mask * last_advantage
        last_value = values[t] # Update last_value for next iteration backwards
    # Returns are advantages + old values
    returns = advantages + values
    return advantages, returns

# --- Environment Setup ---
try: gym.register_envs(ale_py)
except Exception as e: print(f"ALE env registration failed or already done: {e}")
# env = gym.make(ENV_NAME, render_mode="human")
env = gym.make(ENV_NAME)
env = wrappers.ResizeObservation(env, (84, 84))
env = wrappers.GrayscaleObservation(env, keep_dim=False)
env = wrappers.FrameStackObservation(env, 4)
state_shape = (84, 84, 4,)
num_actions = env.action_space.n
print(f"Env: {ENV_NAME}, Wrapped State Shape: {state_shape}, Num Actions: {num_actions}")

# --- Model Initialization ---
# Ensure correct dimensions passed
encoder = create_recurrent_encoder(state_shape, EMBEDDING_DIM, None, ENCODER_HIDDEN_DIM)
target_encoder = create_recurrent_encoder(state_shape, EMBEDDING_DIM, None, ENCODER_HIDDEN_DIM, name="TargetRecurrentEncoder")
predictor = create_predictor(EMBEDDING_DIM, PREDICTOR_HIDDEN_DIM)
target_encoder.set_weights(encoder.get_weights())
actor_critic_heads = create_actor_critic_heads(EMBEDDING_DIM, HIDDEN_DIM_AC, num_actions)
# Separate Optimizers
jepa_optimizer = keras.optimizers.Adam(learning_rate=JEPA_LR)
ppo_optimizer = keras.optimizers.Adam(learning_rate=PPO_LR)

# --- Logging Data Structures ---
episode_rewards = collections.deque(maxlen=100)
logged_steps = []
logged_avg_rewards = []
logged_jepa_losses = []
logged_policy_losses = []
logged_value_losses = []
logged_entropy_losses = []
logged_embedding_stds = []

latest_jepa_loss_val = np.nan
latest_policy_loss_val = np.nan
latest_value_loss_val = np.nan
latest_entropy_loss_val = np.nan
latest_embedding_std_val = np.nan

# --- Training Step Functions ---

# JEPA update step remains similar, takes batch data
@tf.function(reduce_retracing=True)
def train_step_jepa(states, prev_embeddings, next_states, current_embeddings):
    """JEPA update step, takes batch data from rollout."""
    with tf.GradientTape() as tape:
        z = encoder([states, prev_embeddings], training=True)
        z_next_target = target_encoder([next_states, tf.stop_gradient(current_embeddings)], training=False)
        z_pred = predictor(z, training=True)
        jepa_loss = mse_loss(tf.stop_gradient(z_next_target), z_pred)
    jepa_trainable_vars = encoder.trainable_variables + predictor.trainable_variables
    gradients = tape.gradient(jepa_loss, jepa_trainable_vars)
    jepa_optimizer.apply_gradients(zip(gradients, jepa_trainable_vars))
    update_target_network_tf(encoder, target_encoder, TARGET_EMA_TAU)
    return jepa_loss

# << PPO Update Step >>
@tf.function(reduce_retracing=True)
def train_step_ppo(
    states, prev_embeddings, actions, old_log_probs, advantages, returns, # Note: 'returns' are GAE-based value targets
    clip_epsilon, value_loss_coef, entropy_coef, num_actions):
    """Performs one PPO update step on a minibatch."""
    with tf.GradientTape() as tape:
        # --- Forward pass for current policy/value ---
        # Get current embedding z_t
        state_embeddings = encoder([states, prev_embeddings], training=True)
        # Get current action logits and value estimates V(s_t)
        action_logits, current_values = actor_critic_heads(state_embeddings, training=True)
        current_values = tf.squeeze(current_values)

        # --- Calculate current log probs and entropy ---
        current_probs_dist = tf.nn.softmax(action_logits)
        # Log prob of the action *taken* under the *current* policy
        current_action_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=action_logits, labels=actions
        )
        # Need negative sign because cross_entropy gives neg log prob
        current_action_log_probs = -current_action_log_probs
        entropy = -tf.reduce_sum(current_probs_dist * tf.math.log(current_probs_dist + 1e-8), axis=-1)

        # --- PPO Policy Loss (Clipped Surrogate Objective) ---
        # Ratio r_t = exp(log_prob_current - log_prob_old)
        log_ratio = current_action_log_probs - old_log_probs
        ratio = tf.exp(log_ratio)

        # Ensure advantages have the same shape as ratio for broadcasting
        # advantages = tf.reshape(advantages, tf.shape(ratio)) # Should already match if calculated correctly

        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) # Maximize objective -> Minimize negative objective

        # --- Value Function Loss ---
        # MSE or Huber loss between current value estimate and GAE returns
        value_loss = huber_loss(returns, current_values) # Huber loss per element
        value_loss = tf.reduce_mean(value_loss) # Average over batch

        # --- Entropy Bonus ---
        entropy_loss = -tf.reduce_mean(entropy) # Maximize entropy -> Minimize negative entropy

        # --- Total Loss ---
        total_loss = policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss

    # --- Calculate and Apply Gradients ---
    ppo_trainable_vars = encoder.trainable_variables + actor_critic_heads.trainable_variables
    gradients = tape.gradient(total_loss, ppo_trainable_vars)
    gradients, _ = tf.clip_by_global_norm(gradients, 0.5)

    checked_gradients = []
    for i, grad in enumerate(gradients):
        if grad is not None:
            checked_gradients.append(tf.debugging.check_numerics(grad, f"Gradient {i}"))
        else:
            checked_gradients.append(None) # Keep None gradients as None
    gradients = checked_gradients

    ppo_optimizer.apply_gradients(zip(gradients, ppo_trainable_vars))

    return policy_loss, value_loss, entropy_loss

# --- Main Training Loop ---

print(f"\nStarting training for {TOTAL_STEPS} steps...")
state, _ = env.reset()
state = np.transpose(state, (1, 2, 0))
episode_reward = 0
episode_steps = 0
total_episodes = 0
previous_embedding_tensor = tf.zeros((1, EMBEDDING_DIM), dtype=tf.float32)

# Storage for rollout data
rollout_states = []
rollout_prev_embeddings = []
rollout_actions = []
rollout_log_probs = []
rollout_rewards = []
rollout_next_states = []
rollout_current_embeddings = []
rollout_values = []
rollout_dones = []

progress_bar = tqdm.tqdm(range(TOTAL_STEPS))

for step in progress_bar:
    # --- Collect One Step ---
    state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

    # Get embedding, value, action, log_prob from *current* policy (no training=True here)
    current_embedding_tensor = encoder([state_tensor, previous_embedding_tensor], training=False)
    action_logits, current_value_tensor = actor_critic_heads(current_embedding_tensor, training=False)

    # Sample action and get its log probability
    action_probs_tensor = tf.nn.softmax(action_logits)
    action = tf.random.categorical(action_logits, 1)[0, 0].numpy() # Sample action
    log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits, labels=[action]) # Get log prob of sampled action

    # Step environment
    next_state, reward, terminated, truncated, _ = env.step(action)
    next_state = np.transpose(next_state, (1, 2, 0))
    done = terminated or truncated

    # --- Store Transition Data ---
    # Store numpy versions
    rollout_states.append(state)
    rollout_prev_embeddings.append(previous_embedding_tensor[0].numpy())
    rollout_actions.append(action)
    rollout_log_probs.append(log_prob[0].numpy())
    rollout_rewards.append(reward)
    rollout_next_states.append(next_state)
    rollout_current_embeddings.append(current_embedding_tensor[0].numpy())
    rollout_values.append(current_value_tensor[0, 0].numpy())
    rollout_dones.append(float(done))

    # Update state and recurrent embedding
    state = next_state
    previous_embedding_tensor = current_embedding_tensor
    episode_reward += reward
    episode_steps += 1

    # Handle episode end within loop
    if done or episode_steps >= MAX_STEPS_PER_EPISODE:
        episode_rewards.append(episode_reward)
        total_episodes += 1
        state, _ = env.reset()
        state = np.transpose(state, (1, 2, 0))
        episode_reward = 0
        episode_steps = 0
        previous_embedding_tensor = tf.zeros((1, EMBEDDING_DIM), dtype=tf.float32) # Reset hidden state

    # --- Perform Update Phase ---
    if len(rollout_states) == N_STEPS:
        # --- Prepare Data Batch from Rollout ---
        states_np = np.array(rollout_states, dtype=np.float32)
        prev_embeddings_np = np.array(rollout_prev_embeddings, dtype=np.float32)
        actions_np = np.array(rollout_actions, dtype=np.int32)
        log_probs_np = np.array(rollout_log_probs, dtype=np.float32)
        rewards_np = np.array(rollout_rewards, dtype=np.float32)
        next_states_np = np.array(rollout_next_states, dtype=np.float32)
        current_embeddings_np = np.array(rollout_current_embeddings, dtype=np.float32)
        values_np = np.array(rollout_values, dtype=np.float32)
        dones_np = np.array(rollout_dones, dtype=np.float32)

        # --- Calculate Advantages and Returns (GAE) ---
        # Need value estimate for the state *after* the last step in rollout
        last_state_tensor = tf.convert_to_tensor([state], dtype=tf.float32) # Use current 'state' which is s_{N+1}
        last_embedding_tensor = previous_embedding_tensor # Use current 'previous_embedding_tensor' which is z_N
        _, last_value_tensor = actor_critic_heads(encoder([last_state_tensor, last_embedding_tensor], training=False), training=False)
        last_value_np = last_value_tensor[0, 0].numpy()

        advantages_np, returns_np = calculate_gae(
            rewards_np, values_np, last_value_np, dones_np, GAMMA, GAE_LAMBDA
        )

        # Normalize advantages (optional but common)
        advantages_np = (advantages_np - np.mean(advantages_np)) / (np.std(advantages_np) + 1e-8)

        assert not np.isnan(advantages_np).any(), "NaN in advantages after GAE"
        assert not np.isnan(returns_np).any(), "NaN in returns after GAE"

        # Convert rollout data to Tensors for training steps
        states_t = tf.convert_to_tensor(states_np)
        prev_embeddings_t = tf.convert_to_tensor(prev_embeddings_np)
        actions_t = tf.convert_to_tensor(actions_np)
        log_probs_t = tf.convert_to_tensor(log_probs_np)
        advantages_t = tf.convert_to_tensor(advantages_np)
        returns_t = tf.convert_to_tensor(returns_np) # These are value targets
        next_states_t = tf.convert_to_tensor(next_states_np)
        current_embeddings_t = tf.convert_to_tensor(current_embeddings_np)

        # --- Calculate Embedding Std Dev for Monitoring ---
        # Compute embeddings for the full rollout batch.
        # Using training=False is consistent with embeddings generated during data collection.
        # Using training=True would reflect embeddings during the update pass (might differ slightly due to dropout/batchnorm if used). Let's use False for monitoring consistency.
        monitor_embeddings = encoder([states_t, prev_embeddings_t], training=False)
        # Calculate std dev across the batch dim (axis=0) for each embedding dim
        embedding_stds = tf.math.reduce_std(monitor_embeddings, axis=0)
        # Calculate the mean std dev across all embedding dimensions
        mean_embedding_std = tf.reduce_mean(embedding_stds)
        latest_embedding_std_val = mean_embedding_std.numpy()
        # Handle potential NaN if std is zero (e.g., batch size 1 or perfect collapse)
        if np.isnan(latest_embedding_std_val):
            latest_embedding_std_val = 0.0

        # --- Train JEPA (Once per rollout) ---
        latest_jepa_loss = train_step_jepa(
            states_t, prev_embeddings_t, next_states_t, current_embeddings_t
        )
        latest_jepa_loss_val = latest_jepa_loss.numpy()


        # --- Train PPO (Multiple Epochs and Minibatches) ---
        # Combine all data for easier batching
        data_size = N_STEPS
        indices = np.arange(data_size)
        for epoch in range(NUM_EPOCHS_PER_UPDATE):
            np.random.shuffle(indices) # Shuffle data each epoch
            for start in range(0, data_size, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                batch_indices = indices[start:end]

                # Sample minibatch data
                mb_states = tf.gather(states_t, batch_indices)
                mb_prev_embeddings = tf.gather(prev_embeddings_t, batch_indices)
                mb_actions = tf.gather(actions_t, batch_indices)
                mb_log_probs_old = tf.gather(log_probs_t, batch_indices)
                mb_advantages = tf.gather(advantages_t, batch_indices)
                mb_returns = tf.gather(returns_t, batch_indices) # Value targets

                # Perform PPO update step on minibatch
                pl, vl, el = train_step_ppo(
                    mb_states, mb_prev_embeddings, mb_actions, mb_log_probs_old,
                    mb_advantages, mb_returns, PPO_CLIP_EPSILON,
                    VALUE_LOSS_COEF, ENTROPY_COEF, num_actions
                )
                # Store latest losses from the last minibatch for logging
                latest_policy_loss_val = pl.numpy()
                latest_value_loss_val = vl.numpy()
                latest_entropy_loss_val = el.numpy()


        # --- Clear Rollout Storage ---
        rollout_states.clear()
        rollout_prev_embeddings.clear()
        rollout_actions.clear()
        rollout_log_probs.clear()
        rollout_rewards.clear()
        rollout_next_states.clear()
        rollout_current_embeddings.clear()
        rollout_values.clear()
        rollout_dones.clear()


    # --- Logging and Monitoring ---
    if step % LOG_FREQ_STEPS == 0:
        logged_steps.append(step)
        logged_avg_rewards.append(np.mean(episode_rewards) if episode_rewards else np.nan)
        logged_jepa_losses.append(latest_jepa_loss_val)
        logged_policy_losses.append(latest_policy_loss_val)
        logged_value_losses.append(latest_value_loss_val)
        logged_entropy_losses.append(latest_entropy_loss_val)
        logged_embedding_stds.append(latest_embedding_std_val)

    if step % PRINT_FREQ_STEPS == 0:
        avg_reward = np.mean(episode_rewards) if episode_rewards else np.nan
        progress_bar.set_description(
           f"Step {step}/{TOTAL_STEPS} | Ep {total_episodes} | Avg Rwd={avg_reward:.2f} | "
           f"Emb Std={latest_embedding_std_val:.4f} | "
           f"PPO [P={latest_policy_loss_val:.3f}, V={latest_value_loss_val:.3f}, E={latest_entropy_loss_val:.3f}] | "
           f"JEPA Ls={latest_jepa_loss_val:.4f}"
        )

# --- End of Training ---
env.close()
print("\nTraining finished.")

# --- Plotting Results ---
print("Generating plots...")
plt.style.use('seaborn-v0_8-darkgrid')
fig, axs = plt.subplots(5, 1, figsize=(12, 18), sharex=True)

# Plot 1: Avg Rewards vs Steps
axs[0].plot(logged_steps, logged_avg_rewards, label=f'Avg Reward (Last 100 Ep)', color='red', linewidth=2)
axs[0].set_ylabel('Average Reward')
axs[0].set_title(f'Training Rewards ({ENV_NAME}) vs Steps')
axs[0].legend()

# Plot 2: PPO Losses vs Steps
axs[1].plot(logged_steps, logged_policy_losses, label='Policy Loss', color='blue', alpha=0.8, linewidth=1)
axs[1].plot(logged_steps, logged_value_losses, label='Value Loss', color='cyan', alpha=0.8, linewidth=1)
axs[1].plot(logged_steps, logged_entropy_losses, label='Entropy Loss', color='magenta', alpha=0.8, linewidth=1)
axs[1].set_ylabel('PPO Loss')
axs[1].set_title('PPO Loss Components vs Steps')
axs[1].legend()
axs[1].set_yscale('symlog')

# Plot 3: JEPA Loss vs Steps
axs[2].plot(logged_steps, logged_jepa_losses, label='JEPA Loss (MSE)', color='purple', linewidth=1.5)
axs[2].set_ylabel('JEPA Loss')
axs[2].set_title('JEPA Loss vs Steps')
axs[2].legend()
axs[2].set_yscale('symlog') # Use symlog if losses vary wildly

# Plot 4: Value Loss Separate
axs[3].plot(logged_steps, logged_value_losses, label='Value Loss', color='cyan', alpha=0.8, linewidth=1)
axs[3].set_xlabel('Environment Steps')
axs[3].set_ylabel('Value Loss')
axs[3].set_title('PPO Value Loss Component vs Steps')
axs[3].legend()
axs[3].set_yscale('symlog')

# Plot 4: Embedding Std Dev vs Steps
axs[4].plot(logged_steps, logged_embedding_stds, label='Mean Embedding Std Dev', color='green', linewidth=1.5)
axs[4].set_ylabel('Mean Emb Std Dev')
axs[4].set_title('Embedding Std Dev vs Steps')
axs[4].legend()

# Add xlabel to the last plot shown
axs[2].set_xlabel('Environment Steps') # Or axs[3] if using 4 plots

plt.tight_layout()
plt.show()
print("Plot display complete.")
