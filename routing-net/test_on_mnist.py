import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
import time # Using current time below
import os

# Suppress TensorFlow INFO/WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0=all, 1=no info, 2=no info/warning, 3=no info/warning/error

# --- Configuration ---
INPUT_DIM = 784 # 28x28 images flattened
NUM_CLASSES = 10 # Digits 0-9
NUM_EXPERTS = 8
TOP_K = 2
EMBEDDING_DIM = 32
EXPERT_HIDDEN_UNITS = 128 # Increased hidden units slightly for MNIST
LOAD_BALANCING_COEFF = 0.01
BATCH_SIZE = 64
EPOCHS = 5

# --- Expert Model ---
class Expert(layers.Layer):
    """A simple MLP expert."""
    def __init__(self, hidden_units, output_dim, name='expert', **kwargs):
        super().__init__(name=name, **kwargs)
        # Maybe add another layer for better capacity on MNIST
        self.hidden_layer1 = layers.Dense(hidden_units, activation='relu')
        self.hidden_layer2 = layers.Dense(hidden_units // 2, activation='relu')
        self.output_layer = layers.Dense(output_dim) # Output dim matches final desired dim

    def call(self, inputs):
        x = self.hidden_layer1(inputs)
        x = self.hidden_layer2(x)
        return self.output_layer(x)

# --- Router Network ---
class Router(layers.Layer):
    """Outputs a query vector based on the input."""
    def __init__(self, embedding_dim, name='router', **kwargs):
        super().__init__(name=name, **kwargs)
        # Simple router: one dense layer
        self.dense = layers.Dense(embedding_dim)

    def call(self, inputs):
        return self.dense(inputs) # Output shape: (batch_size, embedding_dim)

# --- Main Composable Model Layer ---
class ComposableMoE(layers.Layer):
    """
    Core MoE layer. Calculates expert scores based on router query and expert embeddings,
    selects top-k experts, computes weighted output, and adds load balancing loss.
    Returns a dictionary containing final output, query vector, and gating probabilities.
    """
    def __init__(self, num_experts, k, embedding_dim, expert_hidden_units, output_dim, load_balancing_coeff, name='composable_moe', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_experts = num_experts
        self.k = k
        self.embedding_dim = embedding_dim
        self.load_balancing_coeff = load_balancing_coeff
        self.output_dim = output_dim

        # Learnable embeddings for each expert
        self.expert_embeddings = self.add_weight(
            name="expert_embeddings",
            shape=(self.num_experts, self.embedding_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        # Router instance
        self.router = Router(embedding_dim)
        # List of Expert instances
        self.experts = [
            Expert(expert_hidden_units, self.output_dim, name=f'expert_{i}')
            for i in range(self.num_experts)
        ]

    def calculate_load_balancing_loss(self, gating_probs_full):
        """Calculates the MoE load balancing loss."""
        # Mean probability for each expert across the batch
        fraction_inputs_per_expert = tf.reduce_mean(gating_probs_full, axis=0) # Shape: (num_experts,)
        # Mean squared probability for each expert across the batch
        mean_sq_prob_per_expert = tf.reduce_mean(tf.square(gating_probs_full), axis=0) # Shape: (num_experts,)
        # Loss encourages uniform assignment probabilities and minimizes variance
        loss = self.load_balancing_coeff * tf.reduce_sum(fraction_inputs_per_expert * mean_sq_prob_per_expert) * (self.num_experts ** 2)
        return loss

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]

        # 1. Get Query Vector from Router
        query_vector = self.router(inputs) # Shape: (batch_size, embedding_dim)

        # 2. Calculate Scores (based on distance to expert embeddings)
        query_expanded = tf.expand_dims(query_vector, axis=1) # (batch_size, 1, embedding_dim)
        embeddings_expanded = tf.expand_dims(self.expert_embeddings, axis=0) # (1, num_experts, embedding_dim)
        # Use negative squared L2 distance: higher score = closer embedding
        distances_sq = tf.reduce_sum(tf.square(query_expanded - embeddings_expanded), axis=-1) # Shape: (batch_size, num_experts)
        scores = -distances_sq # Shape: (batch_size, num_experts)

        # 3. Calculate Full Gating Probabilities (for Load Balancing)
        # Use softmax over all experts *before* selection
        gating_probs_full = tf.nn.softmax(scores, axis=-1) # Shape: (batch_size, num_experts)

        # 4. Calculate and Add Load Balancing Loss (only during training)
        if training:
            lb_loss = self.calculate_load_balancing_loss(gating_probs_full)
            self.add_loss(lb_loss) # Add loss to the layer/model

        # 5. Select Top-K Experts based on scores
        top_k_scores, top_k_indices = tf.nn.top_k(scores, k=self.k)
        # top_k_scores shape: (batch_size, k)
        # top_k_indices shape: (batch_size, k)

        # 6. Normalize Top-K Scores for Output Combination
        # Apply softmax *only* over the selected k experts' scores
        top_k_gates = tf.nn.softmax(top_k_scores, axis=-1) # Shape: (batch_size, k)

        # 7. Execute Selected Experts and Combine Outputs
        # Gather approach: Calculate all outputs, then select based on indices
        all_expert_outputs = [expert(inputs) for expert in self.experts] # List of (batch_size, output_dim)
        expert_outputs_stacked = tf.stack(all_expert_outputs, axis=0) # (num_experts, batch_size, output_dim)
        expert_outputs_transposed = tf.transpose(expert_outputs_stacked, [1, 0, 2]) # (batch_size, num_experts, output_dim)
        # Gather the outputs of the top k experts for each batch item
        selected_expert_outputs = tf.gather(expert_outputs_transposed, top_k_indices, batch_dims=1) # (batch_size, k, output_dim)

        # 8. Weighted Combination using normalized top-k gates
        top_k_gates_expanded = tf.expand_dims(top_k_gates, axis=-1) # (batch_size, k, 1)
        # Weighted sum: sum(gate * output) along the k dimension
        combined_output = tf.reduce_sum(selected_expert_outputs * top_k_gates_expanded, axis=1) # (batch_size, output_dim)

        # Return dictionary including intermediate values needed for metrics/inspection
        return {
            'final_output': combined_output,
            'query_vector': query_vector,
            'gating_probs': gating_probs_full
        }

# --- Custom Metric Class ---
class QueryVectorStddevMetric(keras.metrics.Metric):
    """Calculates the standard deviation of the query vector elements."""
    def __init__(self, name="query_stddev", dtype=tf.float32, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        # State variables for incremental calculation: E[X^2] - (E[X])^2
        self.sum_sq = self.add_weight(name="sum_sq", initializer="zeros", dtype=dtype)
        self.sum_val = self.add_weight(name="sum_val", initializer="zeros", dtype=dtype)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=dtype)

    # NOTE: Modified signature - receives query_vector directly in train/test_step
    def update_state(self, query_vector, sample_weight=None):
        batch_sum_sq = tf.reduce_sum(tf.square(query_vector))
        batch_sum_val = tf.reduce_sum(query_vector)
        batch_count = tf.cast(tf.size(query_vector), self.dtype) # Total number of elements
        self.sum_sq.assign_add(batch_sum_sq)
        self.sum_val.assign_add(batch_sum_val)
        self.count.assign_add(batch_count)

    def result(self):
        # Calculate variance: E[X^2] - E[X]^2
        mean_sq = tf.math.divide_no_nan(self.sum_sq, self.count)
        mean_val = tf.math.divide_no_nan(self.sum_val, self.count)
        variance = mean_sq - tf.square(mean_val)
        # Return standard deviation (sqrt of variance), ensuring non-negativity
        return tf.sqrt(tf.maximum(variance, 0.))

    def reset_state(self):
        # Reset state variables at the beginning of each epoch/evaluation
        self.sum_sq.assign(0.0)
        self.sum_val.assign(0.0)
        self.count.assign(0.0)

# --- Model Subclass ---
class MoEModel(keras.Model):
    """
    Top-level Keras Model subclass integrating the ComposableMoE layer.
    Handles custom training and evaluation steps.
    """
    def __init__(self, input_dim, num_classes, num_experts, k, embedding_dim, expert_hidden_units, load_balancing_coeff, name="moe_model", **kwargs):
        super().__init__(name=name, **kwargs)
        # Define the core MoE layer
        self.moe_layer = ComposableMoE(
            num_experts=num_experts,
            k=k,
            embedding_dim=embedding_dim,
            expert_hidden_units=expert_hidden_units,
            output_dim=num_classes,
            load_balancing_coeff=load_balancing_coeff,
            name='composable_moe'
        )
        # Define metrics tracked (initialized here for state management)
        self.accuracy_metric = keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        self.stddev_metric = QueryVectorStddevMetric()
        # Define the primary loss function used in train/test_step
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Defines the forward pass
    def call(self, inputs, training=False):
        # The MoE layer returns the dictionary {'final_output', 'query_vector', 'gating_probs'}
        return self.moe_layer(inputs, training=training)

    # Defines one step of training (on one batch)
    def train_step(self, data):
        # Unpack data - Keras passes `y` as the dictionary target we provided in fit()
        x, y_dict = data
        y_true = y_dict['final_output'] # Extract true labels

        with tf.GradientTape() as tape:
            # Run forward pass, get dictionary output
            y_pred_dict = self(x, training=True)
            # Extract the primary prediction output for loss
            y_pred_main = y_pred_dict['final_output']

            # Calculate primary loss using the designated loss function
            primary_loss = self.loss_fn(y_true, y_pred_main)
            # Get internal losses added via self.add_loss() in ComposableMoE
            internal_losses = self.losses
            # Combine primary loss and internal (load balancing) loss
            total_loss = primary_loss + tf.add_n(internal_losses) if internal_losses else primary_loss

        # Compute gradients of total loss w.r.t trainable variables
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)

        # Update model weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics manually
        self.accuracy_metric.update_state(y_true, y_pred_main)
        self.stddev_metric.update_state(y_pred_dict['query_vector']) # Pass relevant output

        # Return logs - Keras aggregates these across batches
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = primary_loss # Track primary loss
        results['total_loss'] = total_loss # Track combined loss
        results['lb_loss'] = tf.add_n(internal_losses) if internal_losses else 0.0 # Track LB loss
        return results

    # Defines one step of evaluation (on one batch)
    def test_step(self, data):
        x, y_dict = data
        y_true = y_dict['final_output']

        # Run forward pass (training=False)
        y_pred_dict = self(x, training=False)
        y_pred_main = y_pred_dict['final_output']

        # Calculate primary loss
        primary_loss = self.loss_fn(y_true, y_pred_main)
        # Get internal losses
        internal_losses = self.losses
        total_loss = primary_loss + tf.add_n(internal_losses) if internal_losses else primary_loss

        # Update metrics manually
        self.accuracy_metric.update_state(y_true, y_pred_main)
        self.stddev_metric.update_state(y_pred_dict['query_vector'])

        # Return logs
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = primary_loss
        results['total_loss'] = total_loss
        results['lb_loss'] = tf.add_n(internal_losses) if internal_losses else 0.0
        return results

    # Exposes the metrics instances to Keras for tracking
    @property
    def metrics(self):
        # Note: `loss` is implicitly tracked by Keras based on train/test_step return keys
        return [self.accuracy_metric, self.stddev_metric]

# --- Main Execution ---
if __name__ == "__main__":
    # 0. Add current time info
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"Script started at: {current_time_str}")

    # 1. Load and Preprocess MNIST Data
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, INPUT_DIM).astype("float32") / 255.0
    x_test = x_test.reshape(-1, INPUT_DIM).astype("float32") / 255.0
    # Create validation split
    val_split = 0.1
    num_val_samples = int(len(x_train) * val_split)
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    print(f"Using {len(x_train)} training samples and {len(x_val)} validation samples.")

    # 2. Instantiate the Subclassed Model
    print("\nInstantiating subclassed MoEModel...")
    model = MoEModel(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        num_experts=NUM_EXPERTS,
        k=TOP_K,
        embedding_dim=EMBEDDING_DIM,
        expert_hidden_units=EXPERT_HIDDEN_UNITS,
        load_balancing_coeff=LOAD_BALANCING_COEFF
    )
    # Build the model by calling it once (or use input_shape in init)
    # This is needed for model.summary() with subclassed models
    model.build(input_shape=(None, INPUT_DIM))
    print("Model built.")
    model.summary(expand_nested=True) # Show nested structure

    # 3. Compile the model
    print("\nCompiling model...")
    # Compile primarily sets the optimizer. Loss/metrics are handled in train/test_step.
    model.compile(
        optimizer=keras.optimizers.Adam()
        # Loss/metrics can optionally be passed here for Keras tracking alignment,
        # but the primary mechanism is within train_step/test_step and the metrics property.
    )
    print("Model compiled.")

    # 4. Train the model
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = model.fit(
        x_train,
        {'final_output': y_train}, # Pass labels matching the key used in train_step
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_val, {'final_output': y_val}), # Validation labels also need matching key
        verbose=1 # Show progress bar
    )
    print("Training finished.")
    # Access training history (e.g., history.history['accuracy'], history.history['query_stddev'])

    # 5. Evaluate on Test Set
    print("\nEvaluating on test data...")
    results = model.evaluate(
        x_test,
        {'final_output': y_test},
        batch_size=BATCH_SIZE,
        verbose=0 # Set to 1 to see evaluation progress bar
    )

    # 6. Inspect Load Balancing on a subset of test data
    print("\nInspecting expert gating probabilities on test data...")
    num_inspect_samples = 1000
    if num_inspect_samples > len(x_test):
        num_inspect_samples = len(x_test)

    inspect_data = x_test[:num_inspect_samples]

    print(f"Running prediction on {num_inspect_samples} samples for inspection...")
    # model.predict uses the `call` method, which returns the full dictionary
    predictions = model.predict(inspect_data, batch_size=BATCH_SIZE)

    # Extract the gating probabilities (output shape: [num_inspect_samples, num_experts])
    gate_probabilities = predictions['gating_probs']
    print(f"Shape of extracted gate probabilities: {gate_probabilities.shape}")

    # Calculate average probability assigned to each expert
    avg_gate_probs_per_expert = np.mean(gate_probabilities, axis=0)

    print(f"\nAverage gating probability per expert (over {num_inspect_samples} samples):")
    for i, prob in enumerate(avg_gate_probs_per_expert):
        print(f"  Expert {i}: {prob:.4f}")

    # Check if sum is close to 1.0
    print(f"Sum of average probabilities: {np.sum(avg_gate_probs_per_expert):.4f}")

    # Optional: Visualize the distribution
    try:
        import matplotlib.pyplot as plt
        print("\nGenerating load balancing chart...")
        plt.figure(figsize=(10, 4))
        plt.bar(range(NUM_EXPERTS), avg_gate_probs_per_expert)
        plt.xlabel("Expert Index")
        plt.ylabel("Average Gating Probability")
        plt.title(f"Expert Load Balancing (Avg over {num_inspect_samples} samples from Test Set)")
        plt.xticks(range(NUM_EXPERTS))
        plt.ylim(bottom=0) # Ensure y-axis starts at 0
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout() # Adjust layout
        chart_filename = "expert_load_balancing.png"
        plt.savefig(chart_filename)
        print(f"Saved expert load balancing chart to {chart_filename}")
        # plt.show() # Uncomment to display interactively
    except ImportError:
        print("\nInstall matplotlib (`pip install matplotlib`) to visualize the distribution.")
    except Exception as e:
        print(f"\nError generating plot: {e}")

    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"\nScript finished at: {current_time_str}")
