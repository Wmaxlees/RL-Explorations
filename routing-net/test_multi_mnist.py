import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
import time
import os

# Suppress TensorFlow INFO/WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
INPUT_DIM = 784
NUM_CLASSES = 10
NUM_EXPERTS = 64
TOP_K = 8
EMBEDDING_DIM = 32
HIDDEN_DIM = 128 # Dimension for iterative processing within MoE steps
EXPERT_HIDDEN_UNITS = 128 # Hidden units within each expert MLP
NUM_ROUTING_STEPS = 3 # Number of times to repeat Router->Experts->Merge
LOAD_BALANCING_COEFF = 0.00001 # 0.01 works well for equal load balancing
BATCH_SIZE = 64
EPOCHS = 5

# --- Expert Model ---
class Expert(layers.Layer):
    """MLP expert outputting the MoE hidden dimension."""
    def __init__(self, hidden_units, output_dim, name='expert', **kwargs): # output_dim is now HIDDEN_DIM
        super().__init__(name=name, **kwargs)
        self.hidden_layer1 = layers.Dense(hidden_units, activation='relu')
        self.hidden_layer2 = layers.Dense(hidden_units // 2, activation='relu')
        # Output layer now outputs HIDDEN_DIM for iterative processing
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        x = self.hidden_layer1(inputs)
        x = self.hidden_layer2(x)
        return self.output_layer(x)

# --- Router Network ---
class Router(layers.Layer):
    """Outputs a query vector based on the input representation."""
    def __init__(self, embedding_dim, name='router', **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense = layers.Dense(embedding_dim)

    def call(self, inputs):
        return self.dense(inputs)

# --- Main Composable Model Layer ---
class ComposableMoE(layers.Layer):
    """
    Performs one step of MoE routing and expert processing.
    Operates on HIDDEN_DIM and outputs HIDDEN_DIM.
    """
    def __init__(self, num_experts, k, embedding_dim, expert_hidden_units, hidden_dim, load_balancing_coeff, name='composable_moe', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_experts = num_experts
        self.k = k
        self.embedding_dim = embedding_dim
        self.load_balancing_coeff = load_balancing_coeff
        self.hidden_dim = hidden_dim # Store hidden_dim

        self.expert_embeddings = self.add_weight(
            name="expert_embeddings",
            shape=(self.num_experts, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.router = Router(embedding_dim)
        # Experts now output hidden_dim
        self.experts = [
            Expert(expert_hidden_units, self.hidden_dim, name=f'expert_{i}')
            for i in range(self.num_experts)
        ]

    def calculate_load_balancing_loss(self, gating_probs_full):
        fraction_inputs_per_expert = tf.reduce_mean(gating_probs_full, axis=0)
        mean_sq_prob_per_expert = tf.reduce_mean(tf.square(gating_probs_full), axis=0)
        loss = self.load_balancing_coeff * tf.reduce_sum(fraction_inputs_per_expert * mean_sq_prob_per_expert) * (self.num_experts ** 2)
        return loss

    def call(self, inputs, training=False): # 'inputs' here is the current representation (HIDDEN_DIM)
        batch_size = tf.shape(inputs)[0]
        # Route based on the current representation
        query_vector = self.router(inputs)

        query_expanded = tf.expand_dims(query_vector, axis=1)
        embeddings_expanded = tf.expand_dims(self.expert_embeddings, axis=0)
        distances_sq = tf.reduce_sum(tf.square(query_expanded - embeddings_expanded), axis=-1)
        scores = -distances_sq

        gating_probs_full = tf.nn.softmax(scores, axis=-1)

        if training:
            lb_loss = self.calculate_load_balancing_loss(gating_probs_full)
            self.add_loss(lb_loss) # Add loss for this step

        top_k_scores, top_k_indices = tf.nn.top_k(scores, k=self.k)
        top_k_gates = tf.nn.softmax(top_k_scores, axis=-1)

        # Experts operate on the current representation 'inputs'
        all_expert_outputs = [expert(inputs) for expert in self.experts]
        expert_outputs_stacked = tf.stack(all_expert_outputs, axis=0)
        expert_outputs_transposed = tf.transpose(expert_outputs_stacked, [1, 0, 2])
        selected_expert_outputs = tf.gather(expert_outputs_transposed, top_k_indices, batch_dims=1)

        top_k_gates_expanded = tf.expand_dims(top_k_gates, axis=-1)
        # Combined output is the new representation (HIDDEN_DIM)
        combined_output = tf.reduce_sum(selected_expert_outputs * top_k_gates_expanded, axis=1)

        # Return dictionary for this step
        return {
            'step_output': combined_output, # Output of this MoE step
            'query_vector': query_vector,   # Query used in this step
            'gating_probs': gating_probs_full # Gating probs for this step
        }

# --- Custom Metric Class ---
class QueryVectorStddevMetric(keras.metrics.Metric):
    """Calculates the standard deviation over ALL query vector elements from ALL steps."""
    def __init__(self, name="query_stddev", dtype=tf.float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.sum_sq = self.add_weight(name="sum_sq", initializer="zeros", dtype=dtype)
        self.sum_val = self.add_weight(name="sum_val", initializer="zeros", dtype=dtype)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=dtype)

    # Receives stacked query vectors: (batch, steps, dim)
    def update_state(self, all_query_vectors, sample_weight=None):
        # Calculate stats over all elements in the tensor
        batch_sum_sq = tf.reduce_sum(tf.square(all_query_vectors))
        batch_sum_val = tf.reduce_sum(all_query_vectors)
        batch_count = tf.cast(tf.size(all_query_vectors), self.dtype)
        self.sum_sq.assign_add(batch_sum_sq)
        self.sum_val.assign_add(batch_sum_val)
        self.count.assign_add(batch_count)

    def result(self):
        mean_sq = tf.math.divide_no_nan(self.sum_sq, self.count)
        mean_val = tf.math.divide_no_nan(self.sum_val, self.count)
        variance = mean_sq - tf.square(mean_val)
        return tf.sqrt(tf.maximum(variance, 0.))

    def reset_state(self):
        self.sum_sq.assign(0.0)
        self.sum_val.assign(0.0)
        self.count.assign(0.0)

# --- Model Subclass ---
class StackedMoEModel(keras.Model):
    """
    Model implementing stacked MoE layers with Residual Connections and Pre-LayerNorm.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_experts, k, embedding_dim,
                 expert_hidden_units, num_routing_steps, load_balancing_coeff,
                 name="stacked_moe_model_residual", **kwargs): # Added _residual to name
        super().__init__(name=name, **kwargs)
        self.num_routing_steps = num_routing_steps
        self.hidden_dim = hidden_dim # Store hidden_dim for potential checks

        # Input projection layer
        self.input_proj = layers.Dense(hidden_dim, name="input_projection")

        # --- Add Layer Normalization ---
        # Applied before the MoE step in the loop (Pre-Norm)
        self.moe_layer_norm = layers.LayerNormalization(epsilon=1e-6, name="moe_step_layernorm")
        # --- End Layer Normalization ---

        # Single MoE layer instance reused in the loop (parameter sharing)
        self.moe_step_layer = ComposableMoE(
            num_experts=num_experts,
            k=k,
            embedding_dim=embedding_dim,
            expert_hidden_units=expert_hidden_units,
            hidden_dim=hidden_dim, # Operates on hidden_dim
            load_balancing_coeff=load_balancing_coeff,
            name='composable_moe_step'
        )

        # Final Layer Normalization (optional but common before classifier)
        self.final_layer_norm = layers.LayerNormalization(epsilon=1e-6, name="final_layernorm")
        # Final classifier layer
        self.final_classifier = layers.Dense(num_classes, name="classifier")

        # Define metrics
        self.accuracy_metric = keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        self.stddev_metric = QueryVectorStddevMetric()
        # Define loss function instance
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Defines the forward pass with stacked MoE steps and residuals
    def call(self, inputs, training=False):
        # Initial projection
        current_representation = self.input_proj(inputs)
        # Ensure shape is correct (batch_size, hidden_dim)
        tf.ensure_shape(current_representation, (None, self.hidden_dim))


        all_query_vectors = []
        all_gating_probs = []

        # Loop through MoE steps
        for _ in range(self.num_routing_steps):
            # Store input for residual connection
            residual_input = current_representation

            # --- Apply Pre-LayerNorm ---
            normalized_representation = self.moe_layer_norm(current_representation)
            # --- End Pre-LayerNorm ---

            # Call the shared MoE layer instance on the NORMALIZED input
            moe_step_output_dict = self.moe_step_layer(normalized_representation, training=training)

            # Get the output of the MoE processing step
            moe_step_output = moe_step_output_dict['step_output']
            tf.ensure_shape(moe_step_output, (None, self.hidden_dim))


            # --- Add Residual Connection ---
            current_representation = residual_input + moe_step_output
            # --- End Residual Connection ---

            # Store intermediate results for metrics/inspection (from this step's dict)
            all_query_vectors.append(moe_step_output_dict['query_vector'])
            all_gating_probs.append(moe_step_output_dict['gating_probs'])

        # Apply final normalization before classifier
        final_representation_norm = self.final_layer_norm(current_representation)
        # Final classification layer
        final_logits = self.final_classifier(final_representation_norm)

        # Stack intermediate results
        stacked_query_vectors = tf.stack(all_query_vectors, axis=1)
        stacked_gating_probs = tf.stack(all_gating_probs, axis=1)

        # Return final output and stacked intermediates
        return {
            'final_output': final_logits,
            'all_query_vectors': stacked_query_vectors,
            'all_gating_probs': stacked_gating_probs
        }

    # train_step and test_step methods remain exactly the same as before
    # They already handle the dictionary output from call() correctly.
    def train_step(self, data):
        x, y_dict = data
        y_true = y_dict['final_output']

        with tf.GradientTape() as tape:
            y_pred_dict = self(x, training=True)
            y_pred_main = y_pred_dict['final_output']
            primary_loss = self.loss_fn(y_true, y_pred_main)
            internal_losses = self.losses
            total_loss = primary_loss + tf.add_n(internal_losses) if internal_losses else primary_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.accuracy_metric.update_state(y_true, y_pred_main)
        self.stddev_metric.update_state(y_pred_dict['all_query_vectors'])

        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = primary_loss
        results['total_loss'] = total_loss
        avg_lb_loss = tf.add_n(internal_losses) / self.num_routing_steps if internal_losses else 0.0
        results['avg_lb_loss_per_step'] = avg_lb_loss
        return results

    def test_step(self, data):
        x, y_dict = data
        y_true = y_dict['final_output']

        y_pred_dict = self(x, training=False)
        y_pred_main = y_pred_dict['final_output']

        primary_loss = self.loss_fn(y_true, y_pred_main)
        internal_losses = self.losses
        total_loss = primary_loss + tf.add_n(internal_losses) if internal_losses else primary_loss

        self.accuracy_metric.update_state(y_true, y_pred_main)
        self.stddev_metric.update_state(y_pred_dict['all_query_vectors'])

        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = primary_loss
        results['total_loss'] = total_loss
        avg_lb_loss = tf.add_n(internal_losses) / self.num_routing_steps if internal_losses else 0.0
        results['avg_lb_loss_per_step'] = avg_lb_loss
        return results

    @property
    def metrics(self):
        return [self.accuracy_metric, self.stddev_metric]

# --- Main Execution ---
if __name__ == "__main__":
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"Script started at: {current_time_str}")

    # 1. Load and Preprocess MNIST Data
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, INPUT_DIM).astype("float32") / 255.0
    x_test = x_test.reshape(-1, INPUT_DIM).astype("float32") / 255.0
    val_split = 0.1
    num_val_samples = int(len(x_train) * val_split)
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    print(f"Using {len(x_train)} training samples and {len(x_val)} validation samples.")

    # 2. Instantiate the Subclassed Model
    print("\nInstantiating subclassed StackedMoEModel...")
    model = StackedMoEModel(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM, # Pass hidden_dim
        num_classes=NUM_CLASSES,
        num_experts=NUM_EXPERTS,
        k=TOP_K,
        embedding_dim=EMBEDDING_DIM,
        expert_hidden_units=EXPERT_HIDDEN_UNITS,
        num_routing_steps=NUM_ROUTING_STEPS, # Pass number of steps
        load_balancing_coeff=LOAD_BALANCING_COEFF
    )
    model.build(input_shape=(None, INPUT_DIM))
    print("Model built.")
    model.summary(expand_nested=True)

    # 3. Compile the model
    print("\nCompiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam()
    )
    print("Model compiled.")

    # 4. Train the model
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = model.fit(
        x_train,
        {'final_output': y_train},
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, {'final_output': y_val}),
        verbose=1
    )
    print("Training finished.")

    # 5. Evaluate on Test Set
    print("\nEvaluating on test data...")
    results = model.evaluate(
        x_test,
        {'final_output': y_test},
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # 6. Inspect Load Balancing (Now inspects stacked probabilities)
    print("\nInspecting expert gating probabilities on test data...")
    num_inspect_samples = 1000
    if num_inspect_samples > len(x_test):
        num_inspect_samples = len(x_test)
    inspect_data = x_test[:num_inspect_samples]

    print(f"Running prediction on {num_inspect_samples} samples for inspection...")
    predictions = model.predict(inspect_data, batch_size=BATCH_SIZE)

    # Extract the stacked gating probabilities
    # Shape: (num_inspect_samples, num_routing_steps, num_experts)
    all_gate_probabilities = predictions['all_gating_probs']
    print(f"Shape of extracted stacked gate probabilities: {all_gate_probabilities.shape}")

    # Calculate average probability per expert PER STEP
    avg_gate_probs_per_step = np.mean(all_gate_probabilities, axis=0) # Shape: (num_routing_steps, num_experts)

    # Visualize or print per step
    print(f"\nAverage gating probability per expert PER STEP (over {num_inspect_samples} samples):")
    for step in range(NUM_ROUTING_STEPS):
        print(f"--- Step {step+1} ---")
        step_probs = avg_gate_probs_per_step[step]
        for i, prob in enumerate(step_probs):
            print(f"  Expert {i}: {prob:.4f}")
        print(f"  Sum for step {step+1}: {np.sum(step_probs):.4f}")


    # Optional: Visualize average across all steps or per step
    try:
        import matplotlib.pyplot as plt
        print("\nGenerating load balancing chart (Averaged over Steps)...")
        # Average probabilities across steps for an overall view
        avg_overall_probs = np.mean(avg_gate_probs_per_step, axis=0) # Shape: (num_experts,)

        plt.figure(figsize=(10, 4))
        plt.bar(range(NUM_EXPERTS), avg_overall_probs)
        plt.xlabel("Expert Index")
        plt.ylabel("Average Gating Probability (Across all Steps)")
        plt.title(f"Overall Expert Load Balancing (Avg over {num_inspect_samples} samples & {NUM_ROUTING_STEPS} steps)")
        plt.xticks(range(NUM_EXPERTS))
        plt.ylim(bottom=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        chart_filename = "expert_load_balancing_stacked_avg.png"
        plt.savefig(chart_filename)
        print(f"Saved averaged load balancing chart to {chart_filename}")

        # You could also create separate plots for each step if desired
        # ... (loop through steps and create plots) ...

    except ImportError:
        print("\nInstall matplotlib (`pip install matplotlib`) to visualize the distribution.")
    except Exception as e:
        print(f"\nError generating plot: {e}")


    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"\nScript finished at: {current_time_str}")