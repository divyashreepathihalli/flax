#!/usr/bin/env python3
"""
Dense Model Training Example using Flax NNX

This script demonstrates how to create a dense neural network using Flax NNX
and train it for 10 forward pass training steps.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

# Set random seed for reproducibility
jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
np.random.seed(42)

# Generate synthetic data
def generate_data(n_samples=1000, input_dim=10, output_dim=1):
    """Generate synthetic data for training."""
    X = np.random.normal(0, 1, (n_samples, input_dim))
    # Create a simple non-linear function: y = sum(x^2) + noise
    Y = np.sum(X**2, axis=1, keepdims=True) + np.random.normal(0, 0.1, (n_samples, output_dim))
    return X, Y

# Data generator
def dataset(X, Y, batch_size=32):
    """Generate batches of data."""
    n_samples = len(X)
    while True:
        indices = np.random.choice(n_samples, size=batch_size, replace=False)
        yield X[indices], Y[indices]

# Define the dense model using NNX
class DenseModel(nnx.Module):
    """A simple dense neural network using Flax NNX."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, *, rngs: nnx.Rngs):
        super().__init__()
        
        # Create layers
        self.layers = []
        
        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nnx.Linear(prev_dim, output_dim, rngs=rngs)
    
    def __call__(self, x):
        # Forward pass through hidden layers
        for layer in self.layers:
            x = layer(x)
            x = jax.nn.relu(x)  # ReLU activation
        
        # Output layer
        x = self.output_layer(x)
        return x

# Training step function
@jax.jit
def train_step(params, batch, optimizer_state):
    """Single training step."""
    x, y = batch
    
    def loss_fn(params):
        # Reconstruct model from graph definition and parameters
        model = nnx.merge(graphdef, params)
        y_pred = model(x)
        loss = jnp.mean((y - y_pred) ** 2)
        return loss
    
    # Compute gradients
    grads = jax.grad(loss_fn)(params)
    
    # Apply optimizer
    updates, optimizer_state = global_optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    
    return params, optimizer_state

# Evaluation step function
@jax.jit
def eval_step(params, batch):
    """Single evaluation step."""
    x, y = batch
    
    # Reconstruct model from graph definition and parameters
    model = nnx.merge(graphdef, params)
    y_pred = model(x)
    loss = jnp.mean((y - y_pred) ** 2)
    
    return {'loss': loss}

def main():
    """Main training function."""
    print("🚀 Starting Dense Model Training with Flax NNX")
    print("=" * 50)
    
    # Generate data
    print("📊 Generating synthetic data...")
    X, Y = generate_data(n_samples=1000, input_dim=10, output_dim=1)
    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    
    # Model parameters
    input_dim = X.shape[1]
    hidden_dims = [64, 32, 16]  # Three hidden layers
    output_dim = Y.shape[1]
    
    print(f"🤖 Creating model with architecture: {input_dim} -> {hidden_dims} -> {output_dim}")
    
    # Create model
    model = DenseModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        rngs=nnx.Rngs(42)
    )
    
    # Split model into graph definition and parameters (no Variable state)
    global graphdef, global_optimizer
    graphdef, params = nnx.split(model, nnx.Param)
    
    # Create optimizer
    global_optimizer = optax.adam(learning_rate=0.001)
    optimizer_state = global_optimizer.init(params)
    
    print("✅ Model and training state initialized successfully")
    print(f"📈 Starting training for 10 steps...")
    print("-" * 50)
    
    # Training loop
    data_gen = dataset(X, Y, batch_size=32)
    
    for step in range(10):
        # Get batch
        batch = next(data_gen)
        
        # Training step
        params, optimizer_state = train_step(params, batch, optimizer_state)
        
        # Evaluation on full dataset every few steps
        if step % 2 == 0:
            eval_metrics = eval_step(params, (X, Y))
            print(f"Step {step:2d}: Loss = {eval_metrics['loss']:.6f}")
    
    print("-" * 50)
    print("🎉 Training completed!")
    
    # Final evaluation
    final_loss = eval_step(params, (X, Y))['loss']
    print(f"📊 Final loss: {final_loss:.6f}")
    
    # Reconstruct final model
    final_model = nnx.merge(graphdef, params)
    
    # Test prediction
    test_input = np.random.normal(0, 1, (5, input_dim))
    test_output = final_model(test_input)
    print(f"🧪 Test prediction shape: {test_output.shape}")
    print(f"🧪 Sample predictions: {test_output.flatten()[:3]}")
    
    print("\n✅ Dense model training with NNX completed successfully!")

if __name__ == "__main__":
    main() 