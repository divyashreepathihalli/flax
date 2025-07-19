#!/usr/bin/env python3
"""
Simple Neuroplasticity in Flax NNX Example

This script demonstrates neuroplasticity concepts in Flax NNX with a simpler approach:
1. Dynamic weight updates during forward pass
2. Adaptive connection strengths
3. Hebbian learning rules
4. Synaptic scaling
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

# Set random seed for reproducibility
jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
np.random.seed(42)

def generate_data(n_samples=1000, input_dim=10, output_dim=1):
    """Generate synthetic data for training."""
    X = np.random.normal(0, 1, (n_samples, input_dim))
    Y = np.sum(X**2, axis=1, keepdims=True) + np.random.normal(0, 0.1, (n_samples, output_dim))
    return X, Y

def dataset(X, Y, batch_size=32):
    """Generate batches of data."""
    n_samples = len(X)
    while True:
        indices = np.random.choice(n_samples, size=batch_size, replace=False)
        yield X[indices], Y[indices]

class PlasticLinear(nnx.Module):
    """
    A linear layer with neuroplasticity features:
    - Dynamic weight updates based on activity
    - Hebbian learning
    - Synaptic scaling
    """
    
    def __init__(self, in_features, out_features, *, rngs: nnx.Rngs, plasticity_rate=0.01):
        super().__init__()
        
        # Standard weights and bias
        self.weight = nnx.Param(jax.random.uniform(rngs.params(), (in_features, out_features)))
        self.bias = nnx.Param(jnp.zeros((out_features,)))
        
        # Plasticity parameters
        self.plasticity_rate = plasticity_rate
        self.activity_history = nnx.Variable(jnp.zeros((in_features, out_features)))
        self.connection_strength = nnx.Variable(jnp.ones((in_features, out_features)))
        self.hebbian_weights = nnx.Variable(jnp.zeros((in_features, out_features)))
        
        # Step counter for plasticity updates
        self.step_counter = nnx.Variable(jnp.array(0))
    
    def hebbian_update(self, input_activity, output_activity):
        """Hebbian learning: neurons that fire together, wire together."""
        # Hebbian rule: Δw = η * input_activity * output_activity
        hebbian_update = self.plasticity_rate * input_activity[:, None] * output_activity[None, :]
        self.hebbian_weights.value += hebbian_update
    
    def synaptic_scaling(self, target_activity=1.0):
        """Synaptic scaling to maintain homeostasis."""
        current_activity = jnp.mean(self.activity_history.value)
        if current_activity > 0:
            scaling_factor = target_activity / current_activity
            self.connection_strength.value *= scaling_factor
            # Clamp scaling to reasonable bounds
            self.connection_strength.value = jnp.clip(self.connection_strength.value, 0.1, 10.0)
    
    def __call__(self, x):
        # Store input for plasticity updates
        input_activity = jnp.mean(jnp.abs(x), axis=0)
        
        # Apply connection strength scaling (simplified structural plasticity)
        plastic_weight = self.weight.value * self.connection_strength.value
        
        # Forward pass
        y = x @ plastic_weight + self.bias.value
        
        # Update activity history
        output_activity = jnp.mean(jnp.abs(y), axis=0)
        self.activity_history.value = 0.9 * self.activity_history.value + 0.1 * (
            input_activity[:, None] * output_activity[None, :]
        )
        
        # Apply neuroplasticity updates
        self.hebbian_update(input_activity, output_activity)
        
        # Increment step counter
        self.step_counter.value += 1
        
        # Periodic synaptic scaling
        if self.step_counter.value % 50 == 0:  # Every 50 steps
            self.synaptic_scaling()
        
        return y

class NeuroplasticModel(nnx.Module):
    """
    A neural network with neuroplasticity features.
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, *, rngs: nnx.Rngs):
        super().__init__()
        
        # Create plastic layers
        self.layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(PlasticLinear(prev_dim, hidden_dim, rngs=rngs))
            prev_dim = hidden_dim
        
        # Output layer (less plastic for stability)
        self.output_layer = nnx.Linear(prev_dim, output_dim, rngs=rngs)
        
        # Plasticity metrics
        self.plasticity_metrics = nnx.Variable(jnp.zeros(4))  # [hebbian, structural, scaling, activity]
    
    def __call__(self, x):
        # Forward pass through plastic layers
        for layer in self.layers:
            x = layer(x)
            x = jax.nn.relu(x)  # ReLU activation
        
        # Output layer
        x = self.output_layer(x)
        
        # Update plasticity metrics
        hebbian_total = sum(layer.hebbian_weights.value.sum() for layer in self.layers if hasattr(layer, 'hebbian_weights'))
        structural_total = sum(layer.connection_strength.value.mean() for layer in self.layers if hasattr(layer, 'connection_strength'))
        scaling_total = sum(layer.connection_strength.value.mean() for layer in self.layers if hasattr(layer, 'connection_strength'))
        activity_total = sum(layer.activity_history.value.mean() for layer in self.layers if hasattr(layer, 'activity_history'))
        
        self.plasticity_metrics.value = jnp.array([hebbian_total, structural_total, scaling_total, activity_total])
        
        return x

# Training functions
@jax.jit
def train_step(params, variables, batch, optimizer_state):
    """Single training step with plasticity monitoring."""
    x, y = batch
    
    def loss_fn(params):
        model = nnx.merge(graphdef, params, variables)
        y_pred = model(x)
        loss = jnp.mean((y - y_pred) ** 2)
        new_variables = nnx.state(model, nnx.Variable)
        return loss, new_variables
    
    grads, variables = jax.grad(loss_fn, has_aux=True)(params)
    updates, optimizer_state = global_optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    
    return params, variables, optimizer_state

@jax.jit
def eval_step(params, variables, batch):
    """Evaluation step with plasticity metrics."""
    x, y = batch
    
    model = nnx.merge(graphdef, params, variables)
    y_pred = model(x)
    loss = jnp.mean((y - y_pred) ** 2)
    
    # Get plasticity metrics
    plasticity_metrics = model.plasticity_metrics.value
    
    return {
        'loss': loss,
        'hebbian_activity': plasticity_metrics[0],
        'structural_plasticity': plasticity_metrics[1],
        'synaptic_scaling': plasticity_metrics[2],
        'neural_activity': plasticity_metrics[3]
    }

def main():
    """Main training function with neuroplasticity monitoring."""
    print("🧠 Simple Neuroplasticity in Flax NNX")
    print("=" * 50)
    
    # Generate data
    print("📊 Generating synthetic data...")
    X, Y = generate_data(n_samples=1000, input_dim=10, output_dim=1)
    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    
    # Model parameters
    input_dim = X.shape[1]
    hidden_dims = [32, 16]  # Smaller network for plasticity observation
    output_dim = Y.shape[1]
    
    print(f"🤖 Creating neuroplastic model: {input_dim} -> {hidden_dims} -> {output_dim}")
    
    # Create model
    model = NeuroplasticModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        rngs=nnx.Rngs(42)
    )
    
    # Split model into parameters and variables
    global graphdef, global_optimizer
    graphdef, params, variables = nnx.split(model, nnx.Param, nnx.Variable)
    
    # Create optimizer
    global_optimizer = optax.adam(learning_rate=0.001)
    optimizer_state = global_optimizer.init(params)
    
    print("✅ Neuroplastic model initialized successfully")
    print("📈 Starting training with plasticity monitoring...")
    print("-" * 70)
    print(f"{'Step':<6} {'Loss':<12} {'Hebbian':<12} {'Structural':<12} {'Scaling':<12} {'Activity':<12}")
    print("-" * 70)
    
    # Training loop
    data_gen = dataset(X, Y, batch_size=32)
    
    for step in range(10):
        batch = next(data_gen)
        params, variables, optimizer_state = train_step(params, variables, batch, optimizer_state)
        
        # Evaluation with plasticity metrics
        if step % 2 == 0:
            metrics = eval_step(params, variables, (X, Y))
            print(f"{step:2d}    {metrics['loss']:10.6f} {metrics['hebbian_activity']:10.6f} "
                  f"{metrics['structural_plasticity']:10.6f} {metrics['synaptic_scaling']:10.6f} "
                  f"{metrics['neural_activity']:10.6f}")
    
    print("-" * 70)
    print("🎉 Neuroplasticity training completed!")
    
    # Final evaluation
    final_metrics = eval_step(params, variables, (X, Y))
    print(f"📊 Final loss: {final_metrics['loss']:.6f}")
    print(f"🧠 Final plasticity metrics:")
    print(f"   - Hebbian activity: {final_metrics['hebbian_activity']:.6f}")
    print(f"   - Structural plasticity: {final_metrics['structural_plasticity']:.6f}")
    print(f"   - Synaptic scaling: {final_metrics['synaptic_scaling']:.6f}")
    print(f"   - Neural activity: {final_metrics['neural_activity']:.6f}")
    
    # Test prediction
    final_model = nnx.merge(graphdef, params, variables)
    test_input = np.random.normal(0, 1, (5, input_dim))
    test_output = final_model(test_input)
    print(f"🧪 Test prediction shape: {test_output.shape}")
    print(f"🧪 Sample predictions: {test_output.flatten()[:3]}")
    
    print("\n✅ Neuroplasticity in NNX demonstration completed!")

if __name__ == "__main__":
    main() 