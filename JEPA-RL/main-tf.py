import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gymnasium as gym # Use gymnasium
import collections
import tqdm # For progress bar
import matplotlib.pyplot as plt
import ale_py # Needed for registering ALE environments

# --- Hyperparameters ---
ENV_NAME = "ALE/Breakout-v5" # Using Atari Breakout

# JEPA Hyperparameters
EMBEDDING_DIM = 256
PREDICTOR_HIDDEN_DIM = 512
ENCODER_HIDDEN_DIM = 512
TARGET_EMA_TAU = tf.constant(0.99, dtype=tf.float32)
JEPA_LR = 1e-6

# Actor-Critic Hyperparameters
ACTOR_CRITIC_LR = 1e-6
GAMMA = 0.99
ENTROPY_COEF = 0.01
HIDDEN_DIM_AC = 512

# Training Hyperparameters
MAX_EPISODES =  50000
MAX_STEPS_PER_EPISODE = 1000
REWARD_THRESHOLD = 400
UPDATE_FREQ_EPISODES = 1
PRINT_FREQ_EPISODES = 1

# --- Recurrent CNN Encoder ---
def create_recurrent_encoder(state_shape, embedding_dim, cnn_filters, dense_hidden_dim, name="RecurrentEncoder"):
    """Creates a recurrent CNN encoder."""
    state_input = keras.Input(shape=state_shape, name="state_input", dtype=tf.float32) # Ensure dtype
    prev_embedding_input = keras.Input(shape=(embedding_dim,), name="prev_embedding_input", dtype=tf.float32)

    # Standard CNN architecture (similar to DQN nature paper)
    # Normalize pixel values
    normalized_state = state_input / 255.0

    # CNN Layers to process the image state
    x = layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu')(normalized_state)
    x = layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(x)
    cnn_output = layers.Flatten()(x)

    # Combine CNN output with previous embedding
    # Optional: Process prev_embedding through a Dense layer
    processed_prev_embedding = layers.Dense(embedding_dim // 2, activation='relu')(prev_embedding_input) # Example processing

    concatenated = layers.Concatenate()([cnn_output, processed_prev_embedding]) # Use processed embedding

    # Dense layers after concatenation
    x = layers.Dense(dense_hidden_dim, activation='relu')(concatenated)
    outputs = layers.Dense(embedding_dim, activation=None)(x) # Final embedding

    return keras.Model(inputs=[state_input, prev_embedding_input], outputs=outputs, name=name)

# --- Predictor ---
def create_predictor(embedding_dim, hidden_dim, name="Predictor"):
    inputs = keras.Input(shape=(embedding_dim,), dtype=tf.float32)
    x = layers.Dense(hidden_dim, activation='relu')(inputs)
    outputs = layers.Dense(embedding_dim, activation=None)(x)
    return keras.Model(inputs, outputs, name=name)

# --- Actor-Critic Heads ---
def create_actor_critic_heads(embedding_dim, hidden_dim, num_actions, name="ActorCritic"):
    inputs = keras.Input(shape=(embedding_dim,), dtype=tf.float32)
    action_logits = layers.Dense(num_actions, activation=None, name="actor_output")(inputs)
    state_value = layers.Dense(1, activation=None, name="critic_output")(inputs)
    return keras.Model(inputs, [action_logits, state_value], name=name)

# --- Utility Functions ---
def update_target_network_tf(online_network, target_network, tau):
    tau = tf.cast(tau, dtype=tf.float32)
    for online_var, target_var in zip(online_network.weights, target_network.weights):
        new_target_var_value = tau * target_var + (1.0 - tau) * online_var
        target_var.assign(new_target_var_value)

# (Loss functions remain the same)
mse_loss = tf.keras.losses.MeanSquaredError()
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
def compute_actor_critic_loss(action_logits, values, actions, returns, entropy_coef, num_actions):
    advantages = returns - values
    action_log_probs = tf.nn.log_softmax(action_logits)
    # Ensure num_actions is int for one_hot
    actions_one_hot = tf.one_hot(actions, int(num_actions), dtype=tf.float32)
    selected_action_log_probs = tf.reduce_sum(actions_one_hot * action_log_probs, axis=1)
    actor_loss = -tf.reduce_sum(selected_action_log_probs * tf.stop_gradient(advantages))
    critic_loss = huber_loss(returns, values)
    probs = tf.nn.softmax(action_logits)
    entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
    entropy_bonus = -tf.reduce_sum(entropy) * entropy_coef
    total_loss = actor_loss + 0.5 * critic_loss + entropy_bonus
    return total_loss, actor_loss, critic_loss, entropy_bonus

# --- Environment Setup ---
# Register ALE environments if not already done (idempotent)
try:
    gym.register_envs(ale_py)
except Exception as e:
    print(f"ALE env registration failed or already done: {e}")

# Apply wrappers (RECOMMENDED, but skipped for direct modification)
env = gym.make(ENV_NAME, render_mode='human') # or 'human'
# env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.ResizeObservation(env, (84, 84))
# env = gym.wrappers.GrayscaleObservation(env)
# env = gym.wrappers.FrameStack(env, 4)
# env = gym.wrappers.NormalizeReward(env) # Optional

state_shape = env.observation_space.shape
num_actions = env.action_space.n
print(f"Environment: {ENV_NAME}, State Shape: {state_shape}, Num Actions: {num_actions}")

# --- Model Initialization ---
# Using separate hidden dim parameters for CNN filters and dense layers
encoder = create_recurrent_encoder(state_shape, EMBEDDING_DIM, cnn_filters=None, dense_hidden_dim=ENCODER_HIDDEN_DIM) # Pass relevant dims
target_encoder = create_recurrent_encoder(state_shape, EMBEDDING_DIM, cnn_filters=None, dense_hidden_dim=ENCODER_HIDDEN_DIM, name="TargetRecurrentEncoder")
predictor = create_predictor(EMBEDDING_DIM, PREDICTOR_HIDDEN_DIM)
target_encoder.set_weights(encoder.get_weights())
actor_critic_heads = create_actor_critic_heads(EMBEDDING_DIM, HIDDEN_DIM_AC, num_actions)
jepa_optimizer = keras.optimizers.Adam(learning_rate=JEPA_LR)
actor_critic_optimizer = keras.optimizers.Adam(learning_rate=ACTOR_CRITIC_LR)

# --- Training Loop Data Structures ---
episode_rewards = collections.deque(maxlen=100)
all_episode_rewards, all_avg_rewards = [], []
all_jepa_losses, all_ac_total_losses, all_actor_losses, all_critic_losses, all_entropy_losses = [], [], [], [], []

# << ADD prev_embeddings_memory >>
states_memory = []
prev_embeddings_memory = [] # Store z_{t-1} for state s_t
actions_memory = []
rewards_memory = []
next_states_memory = [] # Store s_{t+1}
dones_memory = []

# --- Training Step Functions
@tf.function
def train_step_jepa(states, prev_embeddings, next_states, current_embeddings):
    """Performs JEPA update step with recurrent encoder."""
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

@tf.function
def train_step_actor_critic(states, prev_embeddings, actions, returns, num_actions):
    """Performs Actor-Critic update step with recurrent encoder."""
    with tf.GradientTape() as tape:
        state_embeddings = encoder([states, prev_embeddings], training=True)
        action_logits, values = actor_critic_heads(state_embeddings, training=True)
        values = tf.squeeze(values)
        total_loss, actor_loss, critic_loss, entropy_bonus = compute_actor_critic_loss(
            action_logits, values, actions, returns, ENTROPY_COEF, num_actions
        )

    ac_trainable_vars = actor_critic_heads.trainable_variables + encoder.trainable_variables
    gradients = tape.gradient(total_loss, ac_trainable_vars)
    actor_critic_optimizer.apply_gradients(zip(gradients, ac_trainable_vars))
    return total_loss, actor_loss, critic_loss, entropy_bonus

# --- Main Training Loop --- << MANAGE RECURRENT STATE >>
print(f"\nStarting training for {MAX_EPISODES} episodes...")
progress_bar = tqdm.trange(MAX_EPISODES)
solved = False
latest_jepa_loss = np.nan
latest_ac_total_loss = np.nan

for episode in progress_bar:
    state, _ = env.reset()
    episode_reward = 0
    # << Initialize previous embedding >>
    previous_embedding_tensor = tf.zeros((1, EMBEDDING_DIM), dtype=tf.float32)

    # Clear memory lists
    states_memory.clear(); prev_embeddings_memory.clear(); actions_memory.clear()
    rewards_memory.clear(); next_states_memory.clear(); dones_memory.clear()

    # --- Collect Trajectory ---
    for step in range(MAX_STEPS_PER_EPISODE):
        # Convert state to tensor, ensure dtype is float32 for normalization
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)

        # 1. Get Embedding z_t using s_t and z_{t-1}
        current_embedding_tensor = encoder([state_tensor, previous_embedding_tensor], training=False)

        # 2. Get Action from Actor Head using z_t
        action_logits, _ = actor_critic_heads(current_embedding_tensor, training=False)
        action_probs_tensor = tf.nn.softmax(action_logits)
        action = np.random.choice(num_actions, p=action_probs_tensor[0].numpy())

        # 3. Step Environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 4. Store Experience (including z_{t-1})
        states_memory.append(state) # Store raw state (uint8)
        prev_embeddings_memory.append(previous_embedding_tensor[0].numpy())
        actions_memory.append(action)
        rewards_memory.append(reward)
        next_states_memory.append(next_state)
        dones_memory.append(done)

        # << Update recurrent state for the NEXT step >>
        state = next_state
        previous_embedding_tensor = current_embedding_tensor
        episode_reward += reward

        if done:
            break

    # --- Calculate Returns ---
    returns = []
    discounted_sum = 0
    for reward in reversed(rewards_memory):
        discounted_sum = reward + GAMMA * discounted_sum
        returns.insert(0, discounted_sum)

    # --- Prepare Tensors for Training --- << ADD prev_embeddings >>
    if len(states_memory) > 1:
        # Convert states to float32 for training (normalization happens in encoder)
        states_tensor = tf.convert_to_tensor(states_memory, dtype=tf.float32)
        prev_embeddings_tensor_batch = tf.convert_to_tensor(prev_embeddings_memory, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions_memory, dtype=tf.int32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states_memory, dtype=tf.float32)

        # Recalculate z_t batch needed for JEPA target generation
        current_embeddings_tensor_batch = encoder([states_tensor, prev_embeddings_tensor_batch], training=False)

        # --- Perform Updates ---
        latest_jepa_loss = train_step_jepa(
            states_tensor, prev_embeddings_tensor_batch, next_states_tensor, current_embeddings_tensor_batch
        )
        latest_ac_total_loss, latest_ac_actor_loss, latest_ac_critic_loss, latest_ac_entropy_bonus = train_step_actor_critic(
            states_tensor, prev_embeddings_tensor_batch, actions_tensor, returns_tensor, num_actions
        )
    else:
        latest_jepa_loss = np.nan; latest_ac_total_loss = np.nan
        latest_ac_actor_loss = np.nan; latest_ac_critic_loss = np.nan
        latest_ac_entropy_bonus = np.nan

    # --- Logging and Monitoring ---
    episode_rewards.append(episode_reward)
    avg_reward = np.mean(episode_rewards)
    all_episode_rewards.append(episode_reward); all_avg_rewards.append(avg_reward)
    all_jepa_losses.append(latest_jepa_loss); all_ac_total_losses.append(latest_ac_total_loss)
    all_actor_losses.append(latest_ac_actor_loss); all_critic_losses.append(latest_ac_critic_loss)
    all_entropy_losses.append(latest_ac_entropy_bonus)

    jepa_print = latest_jepa_loss.numpy() if tf.is_tensor(latest_jepa_loss) else latest_jepa_loss
    ac_print = latest_ac_total_loss.numpy() if tf.is_tensor(latest_ac_total_loss) else latest_ac_total_loss
    if episode % PRINT_FREQ_EPISODES == 0 or solved:
         progress_bar.set_description(
            f"Ep {episode}: Avg Rwd={avg_reward:.2f} | Last Rwd={episode_reward:.1f} | "
            f"JEPA Ls={jepa_print:.4f} | AC Ls={ac_print:.2f}" )
    if not solved and avg_reward >= REWARD_THRESHOLD:
        print(f"\nSolved at episode {episode}! Average reward: {avg_reward:.2f}")
        solved = True # Note: Threshold is likely too high for this setup/runtime

# --- End of Training ---
env.close()
print("\nTraining finished.")

# --- Plotting Results ---
print("Generating plots...")
episodes = range(len(all_episode_rewards))
# (Conversion to numpy before plotting)
all_jepa_losses_np = np.array([v.numpy() if tf.is_tensor(v) else v for v in all_jepa_losses], dtype=np.float32)
all_ac_total_losses_np = np.array([v.numpy() if tf.is_tensor(v) else v for v in all_ac_total_losses], dtype=np.float32)
all_actor_losses_np = np.array([v.numpy() if tf.is_tensor(v) else v for v in all_actor_losses], dtype=np.float32)
all_critic_losses_np = np.array([v.numpy() if tf.is_tensor(v) else v for v in all_critic_losses], dtype=np.float32)
all_entropy_losses_np = np.array([v.numpy() if tf.is_tensor(v) else v for v in all_entropy_losses], dtype=np.float32)

# (Plotting code using _np arrays)
plt.style.use('seaborn-v0_8-darkgrid')
fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
axs[0].plot(episodes, all_episode_rewards, label='Episode Reward', alpha=0.6, linewidth=1)
axs[0].plot(episodes, all_avg_rewards, label=f'Avg Reward (Last {len(episode_rewards)})', color='red', linewidth=2)
axs[0].axhline(REWARD_THRESHOLD, color='green', linestyle='--', label=f'Solve Threshold ({REWARD_THRESHOLD})')
axs[0].set_ylabel('Reward'); axs[0].set_title(f'Training Rewards ({ENV_NAME})'); axs[0].legend()
axs[1].plot(episodes, all_jepa_losses_np, label='JEPA Loss (MSE)', color='purple', linewidth=1.5)
axs[1].plot(episodes, all_ac_total_losses_np, label='Total A2C Loss', color='orange', linewidth=1.5)
axs[1].set_ylabel('Loss'); axs[1].set_title('JEPA and Total Actor-Critic Loss'); axs[1].legend()
axs[2].plot(episodes, all_actor_losses_np, label='Actor Loss', color='blue', alpha=0.8, linewidth=1)
axs[2].plot(episodes, all_critic_losses_np, label='Critic Loss', color='cyan', alpha=0.8, linewidth=1)
axs[2].plot(episodes, all_entropy_losses_np, label='Entropy Loss (Neg Bonus)', color='magenta', alpha=0.8, linewidth=1)
axs[2].set_xlabel('Episode'); axs[2].set_ylabel('Loss Component'); axs[2].set_title('Actor-Critic Loss Components'); axs[2].legend()
plt.tight_layout(); plt.show()
print("Plot display complete.")