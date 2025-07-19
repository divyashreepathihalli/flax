#!/usr/bin/env python3
"""
Adaptive Activation Functions with Neuroplasticity

This example demonstrates adaptive activation functions that change their behavior
based on input statistics, mimicking homeostatic plasticity in biological neurons.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

class AdaptiveActivation(nnx.Module):
    """
    Adaptive activation function that changes based on input statistics.
    This mimics homeostatic plasticity in biological neurons.
    """
    
    def __init__(self, activation_type='relu'):
        super().__init__()
        self.activation_type = activation_type
        self.input_stats = nnx.Variable(jnp.array([0.0, 1.0]))  # [mean, std]
        self.adaptation_rate = 0.01
    
    def __call__(self, x):
        # Update input statistics
        batch_mean = jnp.mean(x)
        batch_std = jnp.std(x)
        
        # Exponential moving average of statistics
        self.input_stats.value = (
            0.99 * self.input_stats.value + 
            0.01 * jnp.array([batch_mean, batch_std])
        )
        
        # Adaptive normalization
        mean, std = self.input_stats.value
        if std > 0:
            x = (x - mean) / (std + 1e-6)
        
        # Apply activation with adaptive threshold
        if self.activation_type == 'relu':
            # Adaptive ReLU threshold based on input statistics
            threshold = jnp.clip(mean - 0.5 * std, -2.0, 2.0)
            return jax.nn.relu(x - threshold)
        elif self.activation_type == 'sigmoid':
            return jax.nn.sigmoid(x)
        else:
            return x

class AdaptiveNetwork(nnx.Module):
    """
    A simple network with adaptive activation functions.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, *, rngs: nnx.Rngs):
        super().__init__()
        
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.adaptive_relu1 = AdaptiveActivation('relu')
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.adaptive_relu2 = AdaptiveActivation('relu')
        self.output_layer = nnx.Linear(hidden_dim, output_dim, rngs=rngs)
        
        # Track adaptation metrics
        self.adaptation_metrics = nnx.Variable(jnp.zeros(2))  # [mean_shift, std_shift]
    
    def __call__(self, x):
        # Forward pass with adaptive activations
        x = self.linear1(x)
        x = self.adaptive_relu1(x)
        x = self.linear2(x)
        x = self.adaptive_relu2(x)
        x = self.output_layer(x)
        
        # Update adaptation metrics
        mean_shift = jnp.mean(self.adaptive_relu1.input_stats.value[0] + 
                             self.adaptive_relu2.input_stats.value[0])
        std_shift = jnp.mean(self.adaptive_relu1.input_stats.value[1] + 
                            self.adaptive_relu2.input_stats.value[1])
        
        self.adaptation_metrics.value = jnp.array([mean_shift, std_shift])
        
        return x

def main():
    """Demonstrate adaptive activation functions."""
    print("🧠 Adaptive Activation Functions with Neuroplasticity")
    print("=" * 60)
    
    # Create model
    model = AdaptiveNetwork(10, 32, 1, rngs=nnx.Rngs(42))
    
    # Generate test data
    np.random.seed(42)
    test_input = np.random.normal(0, 1, (100, 10))
    
    # Test the model
    output = model(test_input)
    
    print(f"✅ Model created successfully")
    print(f"📊 Input shape: {test_input.shape}")
    print(f"📊 Output shape: {output.shape}")
    print(f"🧬 Adaptation metrics: {model.adaptation_metrics.value}")
    
    # Show adaptive behavior
    print("\n🔄 Testing adaptive behavior...")
    
    # Test with different input distributions
    for i, (mean, std) in enumerate([(0, 1), (2, 0.5), (-1, 2)]):
        test_data = np.random.normal(mean, std, (50, 10))
        _ = model(test_data)
        print(f"   Test {i+1} (mean={mean}, std={std}): "
              f"adaptation={model.adaptation_metrics.value}")
    
    print("\n✅ Adaptive activation demonstration completed!")

if __name__ == "__main__":
    main() 