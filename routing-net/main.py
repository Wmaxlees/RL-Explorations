import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
import os
import gymnasium as gym # Use gymnasium namespace
import tensorflow_probability as tfp

# Suppress TensorFlow INFO/WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set seed for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# --- Configuration ---
# MoE specific config (adjust for RL task)
RL_HIDDEN_DIM = 64
RL_EXPERT_HIDDEN = 64
RL_EMBEDDING_DIM = 16
RL_NUM_EXPERTS = 4
RL_TOP_K = 2
RL_NUM_ROUTING_STEPS = 2
RL_LOAD_BALANCING_COEFF = 0.01 # Coeff for LB loss *during* MoE forward pass

# PPO specific config
ENV_NAME = "CartPole-v1"
LEARNING_RATE = 7e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 10
MINIBATCH_SIZE = 64
CLIP_EPSILON = 0.2
CRITIC_DISCOUNT = 0.5
ENTROPY_BETA = 0.01
PPO_LB_LOSS_COEFF = 0.01 # Coefficient for MoE LB loss in *PPO total loss*

# Training parameters
TOTAL_TIMESTEPS = 50000
STEPS_PER_ITERATION = 2048


# --- MoE Components (Expert, Router, ComposableMoE - same as before) ---
class Expert(layers.Layer):
    """MLP expert outputting the MoE hidden dimension."""
    def __init__(self, hidden_units, output_dim, name='expert', **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_layer1 = layers.Dense(hidden_units, activation='relu')
        self.hidden_layer2 = layers.Dense(hidden_units // 2, activation='relu')
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        x = self.hidden_layer1(inputs)
        x = self.hidden_layer2(x)
        return self.output_layer(x)

class Router(layers.Layer):
    """Outputs a query vector based on the input representation."""
    def __init__(self, embedding_dim, name='router', **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense = layers.Dense(embedding_dim)

    def call(self, inputs):
        return self.dense(inputs)

class ComposableMoE(layers.Layer):
    """Performs one step of MoE routing and expert processing."""
    def __init__(self, num_experts, k, embedding_dim, expert_hidden_units, hidden_dim, load_balancing_coeff, name='composable_moe_step', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_experts = num_experts
        self.k = k
        self.embedding_dim = embedding_dim
        self.load_balancing_coeff = load_balancing_coeff
        self.hidden_dim = hidden_dim

        self.expert_embeddings = self.add_weight(
            name="expert_embeddings", shape=(self.num_experts, self.embedding_dim),
            initializer="glorot_uniform", trainable=True,
        )
        self.router = Router(embedding_dim)
        self.experts = [
            Expert(expert_hidden_units, self.hidden_dim, name=f'expert_{i}')
            for i in range(self.num_experts)
        ]

    def calculate_load_balancing_loss(self, gating_probs_full):
        fraction_inputs_per_expert = tf.reduce_mean(gating_probs_full, axis=0)
        mean_sq_prob_per_expert = tf.reduce_mean(tf.square(gating_probs_full), axis=0)
        loss = self.load_balancing_coeff * tf.reduce_sum(fraction_inputs_per_expert * mean_sq_prob_per_expert) * (self.num_experts ** 2)
        return loss

    def call(self, inputs, training=False):
        query_vector = self.router(inputs)
        query_expanded = tf.expand_dims(query_vector, axis=1)
        embeddings_expanded = tf.expand_dims(self.expert_embeddings, axis=0)
        distances_sq = tf.reduce_sum(tf.square(query_expanded - embeddings_expanded), axis=-1)
        scores = -distances_sq
        gating_probs_full = tf.nn.softmax(scores, axis=-1)

        if training and self.load_balancing_coeff > 0:
            lb_loss = self.calculate_load_balancing_loss(gating_probs_full)
            self.add_loss(lb_loss)

        top_k_scores, top_k_indices = tf.nn.top_k(scores, k=self.k)
        top_k_gates = tf.nn.softmax(top_k_scores, axis=-1)
        all_expert_outputs = [expert(inputs) for expert in self.experts]
        expert_outputs_stacked = tf.stack(all_expert_outputs, axis=0)
        expert_outputs_transposed = tf.transpose(expert_outputs_stacked, [1, 0, 2])
        selected_expert_outputs = tf.gather(expert_outputs_transposed, top_k_indices, batch_dims=1)
        top_k_gates_expanded = tf.expand_dims(top_k_gates, axis=-1)
        combined_output = tf.reduce_sum(selected_expert_outputs * top_k_gates_expanded, axis=1)

        return {'step_output': combined_output} # Only return needed output


# --- Actor-Critic Model Subclass (with Residuals) ---
class StackedMoEActorCritic(keras.Model):
    """Actor-Critic Model using Stacked MoE with Residuals."""
    def __init__(self, input_dim, hidden_dim, num_actions, num_experts, k, embedding_dim,
                 expert_hidden_units, num_routing_steps, load_balancing_coeff,
                 name="stacked_moe_actor_critic_residual", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_routing_steps = num_routing_steps
        self.hidden_dim = hidden_dim

        self.input_proj = layers.Dense(hidden_dim, name="input_projection", activation='relu')
        self.moe_layer_norm = layers.LayerNormalization(epsilon=1e-6, name="moe_step_layernorm")
        self.moe_step_layer = ComposableMoE(
            num_experts=num_experts, k=k, embedding_dim=embedding_dim,
            expert_hidden_units=expert_hidden_units, hidden_dim=hidden_dim,
            load_balancing_coeff=load_balancing_coeff, name='composable_moe_step'
        )
        self.final_layer_norm = layers.LayerNormalization(epsilon=1e-6, name="final_layernorm")
        self.actor_head = layers.Dense(num_actions, name="actor_head")
        self.critic_head = layers.Dense(1, name="critic_head")

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32)
        current_representation = self.input_proj(inputs)
        tf.ensure_shape(current_representation, (None, self.hidden_dim))

        for _ in range(self.num_routing_steps):
            residual_input = current_representation
            normalized_representation = self.moe_layer_norm(current_representation)
            moe_step_output = self.moe_step_layer(normalized_representation, training=training)['step_output']
            tf.ensure_shape(moe_step_output, (None, self.hidden_dim))
            current_representation = residual_input + moe_step_output

        final_representation_norm = self.final_layer_norm(current_representation)
        action_logits = self.actor_head(final_representation_norm)
        value = self.critic_head(final_representation_norm)
        return action_logits, value

    # train_step includes value clipping
    # @tf.function # Can potentially speed up training step
    def train_step(self, data):
        # Unpack data including old values
        x, y_actions, y_old_log_probs, y_advantages, y_returns, y_old_values = data

        with tf.GradientTape() as tape:
            # Get new model outputs (logits and value)
            new_logits, new_values = self(x, training=True)
            new_values = tf.squeeze(new_values, axis=-1) # Shape: (batch_size,)

            # Calculate Actor Loss (Clipped Surrogate Objective)
            new_dist = tfp.distributions.Categorical(logits=new_logits)
            new_log_probs = new_dist.log_prob(y_actions)
            ratio = tf.exp(new_log_probs - y_old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * y_advantages, clipped_ratio * y_advantages))

            # Calculate Entropy Bonus
            entropy_loss = -tf.reduce_mean(new_dist.entropy())

            # --- Calculate Critic Loss with Value Clipping ---
            # Clip the value difference
            values_clipped = y_old_values + tf.clip_by_value(
                new_values - y_old_values, -CLIP_EPSILON, CLIP_EPSILON
            )
            # Loss based on clipped values
            value_loss_clipped = tf.square(values_clipped - y_returns)
            # Loss based on original values
            value_loss_unclipped = tf.square(new_values - y_returns)
            # Take the maximum of the two losses, averaged over batch
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss_clipped, value_loss_unclipped))
            # --- End Value Clipping ---

            # Get MoE Load Balancing Loss (added internally during forward pass)
            lb_loss = tf.add_n(self.losses) if self.losses else tf.constant(0.0, dtype=tf.float32)

            # Calculate Total Loss
            total_loss = (policy_loss
                          + CRITIC_DISCOUNT * value_loss
                          + ENTROPY_BETA * entropy_loss
                          + PPO_LB_LOSS_COEFF * lb_loss) # Add weighted LB loss

        # Compute and apply gradients
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        # Optional: Gradient Clipping
        # grads, _ = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # Return logs (tracking the value loss component specifically)
        results = {}
        results['total_loss'] = total_loss
        results['policy_loss'] = policy_loss
        results['value_loss'] = value_loss # Report the calculated (clipped) value loss
        results['entropy_loss'] = entropy_loss
        results['lb_loss'] = lb_loss
        return results

    # test_step calculates standard value loss for reporting
    # @tf.function
    def test_step(self, data):
         # Unpack data (might not need all parts depending on what you want to eval)
        x, y_actions, y_old_log_probs, y_advantages, y_returns, y_old_values = data

        # Get model outputs
        new_logits, new_values = self(x, training=False)
        new_values = tf.squeeze(new_values, axis=-1)

        # Calculate standard MSE Value Loss for reporting during evaluation
        value_loss = tf.reduce_mean(tf.square(y_returns - new_values))

        # Can optionally calculate policy loss/entropy etc. if needed for eval logs
        # ...

        # Get internal losses (LB loss)
        lb_loss = tf.add_n(self.losses) if self.losses else tf.constant(0.0, dtype=tf.float32)

        # Return logs including standard value loss
        results = {}
        # Add other calculated losses if desired (policy, entropy)
        results['value_loss'] = value_loss
        results['lb_loss'] = lb_loss
        return results

    # No metrics property needed if not passing metrics list to compile


# --- PPO Helper Functions ---
def compute_advantages_and_returns(rewards, values, dones, next_values, gamma, gae_lambda):
    """Computes advantages and returns using GAE."""
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    last_value = next_values[-1]

    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * last_value * mask - values[t]
        advantages[t] = last_advantage = delta + gamma * gae_lambda * mask * last_advantage
        last_value = values[t]
    returns = advantages + values
    return advantages, returns

# --- PPO Training ---
if __name__ == "__main__":
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"PPO Script started at: {current_time_str}")

    # 1. Environment Setup
    print(f"Creating environment: {ENV_NAME}")
    env = gym.make(ENV_NAME)
    observation_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print(f"Observation dim: {observation_dim}, Action dim: {num_actions}")

    # 2. Instantiate Actor-Critic Model
    print("Instantiating StackedMoEActorCritic model...")
    model = StackedMoEActorCritic(
        input_dim=observation_dim, hidden_dim=RL_HIDDEN_DIM, num_actions=num_actions,
        num_experts=RL_NUM_EXPERTS, k=RL_TOP_K, embedding_dim=RL_EMBEDDING_DIM,
        expert_hidden_units=RL_EXPERT_HIDDEN, num_routing_steps=RL_NUM_ROUTING_STEPS,
        load_balancing_coeff=RL_LOAD_BALANCING_COEFF
    )
    model.build(input_shape=(None, observation_dim))
    model.summary(expand_nested=True)

    # 3. Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=0.5)

    # 4. Compile (mainly for setting the optimizer)
    model.compile(optimizer=optimizer)
    print("Model compiled.")


    # --- Training Loop ---
    print("\nStarting PPO Training...")
    total_steps_done = 0
    iteration = 0

    while total_steps_done < TOTAL_TIMESTEPS:
        iteration += 1
        start_time = time.time()

        # --- Collect Trajectories ---
        print(f"\nIteration {iteration}: Collecting trajectories...")
        trajectories = {
            'states': [], 'actions': [], 'rewards': [], 'dones': [],
            'log_probs': [], 'values': [], 'next_states': [] # Store next_states
        }
        steps_in_iter = 0
        episode_rewards = []
        current_episode_reward = 0
        state, _ = env.reset(seed=seed + iteration)

        while steps_in_iter < STEPS_PER_ITERATION:
            state_tensor = tf.expand_dims(tf.cast(state, tf.float32), 0)
            action_logits, value = model(state_tensor, training=False)
            value_np = tf.squeeze(value).numpy() # Store the value estimate for this state

            action_dist = tfp.distributions.Categorical(logits=action_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            action_np = action.numpy()[0]
            log_prob_np = log_prob.numpy()[0]

            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            trajectories['states'].append(state)
            trajectories['actions'].append(action_np)
            trajectories['rewards'].append(reward)
            trajectories['dones'].append(float(done)) # Store as float for mask calculation
            trajectories['log_probs'].append(log_prob_np)
            trajectories['values'].append(value_np) # <<< Store value estimate
            trajectories['next_states'].append(next_state) # Need last next_state

            state = next_state
            current_episode_reward += reward
            steps_in_iter += 1
            total_steps_done += 1

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                state, _ = env.reset()

        # Get value estimate for the final next_state
        last_next_state = trajectories['next_states'][-1]
        _, last_next_value = model(tf.expand_dims(tf.cast(last_next_state, tf.float32), 0), training=False)
        last_next_value_np = tf.squeeze(last_next_value).numpy()

        avg_episode_reward = np.mean(episode_rewards) if episode_rewards else 0
        print(f"Collected {steps_in_iter} steps. Average episode reward: {avg_episode_reward:.2f}")

        # --- Prepare Data for PPO Update ---
        states_np = np.array(trajectories['states'], dtype=np.float32)
        actions_np = np.array(trajectories['actions'], dtype=np.int32)
        rewards_np = np.array(trajectories['rewards'], dtype=np.float32)
        dones_np = np.array(trajectories['dones'], dtype=np.float32) # Kept as float
        old_log_probs_np = np.array(trajectories['log_probs'], dtype=np.float32)
        # 'values' collected are the 'old_values' needed for clipping
        old_values_np = np.array(trajectories['values'], dtype=np.float32)

        # Compute advantages and returns using GAE
        # We need all value estimates + estimate for the state *after* the last action
        values_for_gae = np.append(old_values_np, last_next_value_np)
        advantages_np, returns_np = compute_advantages_and_returns(
            rewards_np, values_for_gae[:-1], dones_np, values_for_gae[1:], GAMMA, GAE_LAMBDA
        )
        # Normalize advantages
        advantages_np = (advantages_np - np.mean(advantages_np)) / (np.std(advantages_np) + 1e-8)
        # Normalized returns
        returns_mean = np.mean(returns_np)
        returns_std = np.std(returns_np)
        returns_normalized_np = (returns_np - returns_mean) / (returns_std + 1e-8)

        # Create dataset including old values
        dataset = tf.data.Dataset.from_tensor_slices((
            states_np, actions_np, old_log_probs_np, advantages_np, returns_normalized_np, old_values_np # <<< Added old_values_np
        ))
        dataset = dataset.shuffle(buffer_size=STEPS_PER_ITERATION).batch(MINIBATCH_SIZE).prefetch(tf.data.AUTOTUNE)


        # --- PPO Update Phase ---
        print(f"Updating policy for {PPO_EPOCHS} epochs...")
        policy_loss_sum, value_loss_sum, entropy_loss_sum, lb_loss_sum, total_loss_sum = 0, 0, 0, 0, 0
        update_count = 0

        for _ in range(PPO_EPOCHS):
            for batch in dataset:
                # Unpack batch including old values
                batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns, batch_old_values = batch # <<< Unpack old_values
                # Use model.train_on_batch or explicit train_step call
                # Using train_step implicitly via compile requires matching args,
                # let's call train_step directly for clarity with the modified data tuple.
                # Or rely on the fact that train_step is automatically called by TF internals if we structure data right.
                # For clarity, let's use train_on_batch, which calls train_step.
                # We need to structure the input data `x` and `y` correctly.
                # `x` is batch_states. `y` needs to contain everything else train_step unpacks.
                # However, train_step expects `data = (x, y_actions, y_old_log_probs, ...)` which doesn't match fit/evaluate.
                # Easiest is to call train_step explicitly here if compile isn't used for loss/metrics.
                # Since compile *only* sets the optimizer, we *can* call train_step directly.

                # Manually call train_step
                step_logs = model.train_step((batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns, batch_old_values))

                # Accumulate losses from returned logs
                policy_loss_sum += step_logs['policy_loss'].numpy()
                value_loss_sum += step_logs['value_loss'].numpy() # This is now the clipped value loss result
                entropy_loss_sum += step_logs['entropy_loss'].numpy()
                lb_loss_sum += step_logs['lb_loss'].numpy()
                total_loss_sum += step_logs['total_loss'].numpy()
                update_count += 1

        # Log results for the iteration
        avg_policy_loss = policy_loss_sum / update_count
        avg_value_loss = value_loss_sum / update_count
        avg_entropy_loss = entropy_loss_sum / update_count
        avg_lb_loss = lb_loss_sum / update_count
        avg_total_loss = total_loss_sum / update_count
        iter_time = time.time() - start_time

        print(f"Iter {iteration} | Timesteps {total_steps_done}/{TOTAL_TIMESTEPS} | Time: {iter_time:.2f}s")
        print(f"  Avg Reward: {avg_episode_reward:.2f}")
        # Note: 'value_loss' reported here is the *result* of the clipped objective calculation from train_step
        print(f"  Losses -> Total: {avg_total_loss:.4f} | Policy: {avg_policy_loss:.4f} | Value: {avg_value_loss:.4f} | Entropy: {avg_entropy_loss:.4f} | LB: {avg_lb_loss:.4f}")


    print("\nTraining finished.")
    env.close()
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"\nPPO Script finished at: {current_time_str}")