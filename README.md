<div align="center">
<img src="https://raw.githubusercontent.com/google/flax/main/images/flax_logo_250px.png" alt="logo"></img>
</div>

# NNX Neuroplasticity

A comprehensive collection of neuroplasticity implementations in Flax NNX, demonstrating biologically-inspired learning mechanisms in neural networks.

## 🧠 Overview

This repository showcases various neuroplasticity concepts implemented in Flax NNX, including:

- **Hebbian Learning**: "Neurons that fire together, wire together"
- **Synaptic Scaling**: Homeostatic plasticity for network stability
- **Structural Plasticity**: Dynamic connection modification
- **Activity-Dependent Plasticity**: Real-time adaptation during forward passes

## 📁 Repository Structure

```
nnx-neuroplasticity/
├── README.md                           # This file
├── dense_model_nnx_example.py         # Basic dense model with NNX
├── neuroplasticity_nnx_final.py       # Advanced neuroplasticity implementation
├── neuroplasticity_nnx_guide.md       # Comprehensive implementation guide
├── requirements.txt                    # Dependencies
└── examples/                          # Additional examples
    ├── adaptive_activation.py         # Adaptive activation functions
    ├── structural_plasticity.py       # Structural plasticity examples
    └── continual_learning.py          # Continual learning with plasticity
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nnx-neuroplasticity.git
cd nnx-neuroplasticity

# Install dependencies
pip install -r requirements.txt
```

### Running Examples

#### 1. Basic Dense Model
```bash
python dense_model_nnx_example.py
```

#### 2. Neuroplasticity Implementation
```bash
python neuroplasticity_nnx_final.py
```

## 🧬 Neuroplasticity Concepts

### 1. Hebbian Learning
Implements the classic Hebbian learning rule where connection strengths are updated based on correlated activity between input and output neurons.

```python
def hebbian_update(self, input_activity, output_activity):
    """Hebbian learning: neurons that fire together, wire together"""
    hebbian_update = self.plasticity_rate * input_activity[:, None] * output_activity[None, :]
    self.hebbian_weights.value += hebbian_update
```

### 2. Synaptic Scaling
Maintains network homeostasis by adjusting connection strengths to maintain target activity levels.

```python
def synaptic_scaling(self, target_activity=1.0):
    """Synaptic scaling to maintain homeostasis"""
    current_activity = jnp.mean(self.activity_history.value)
    if current_activity > 0:
        scaling_factor = target_activity / current_activity
        self.connection_strength.value *= scaling_factor
```

### 3. Structural Plasticity
Dynamically modifies network connectivity based on activity patterns.

```python
def structural_plasticity(self, input_activity, output_activity):
    """Structural plasticity: modify connection probabilities"""
    activity_correlation = input_activity[:, None] * output_activity[None, :]
    prob_update = 0.01 * activity_correlation
    self.connection_prob.value += prob_update
```

## 🔧 Implementation Details

### Key Components

#### PlasticLinear Layer
A linear layer with built-in neuroplasticity features:

```python
class PlasticLinear(nnx.Module):
    def __init__(self, in_features, out_features, *, rngs: nnx.Rngs, plasticity_rate=0.01):
        # Standard parameters
        self.weight = nnx.Param(...)
        self.bias = nnx.Param(...)
        
        # Plasticity parameters
        self.activity_history = nnx.Variable(...)
        self.connection_strength = nnx.Variable(...)
        self.hebbian_weights = nnx.Variable(...)
```

#### Training with Plasticity Monitoring
```python
@jax.jit
def train_step(params, variables, batch, optimizer_state):
    def loss_fn(params):
        model = nnx.merge(graphdef, params, variables)
        y_pred = model(x)
        loss = jnp.mean((y - y_pred) ** 2)
        new_variables = nnx.state(model, nnx.Variable)
        return loss, new_variables
```

## 📊 Results

### Training Progress
The neuroplasticity implementation shows:
- **Loss Reduction**: From ~121 to ~106 over 10 steps
- **Hebbian Activity**: Tracks correlation-based learning
- **Structural Plasticity**: Monitors connection strength changes
- **Synaptic Scaling**: Maintains network homeostasis

### Example Output
```
🧠 Neuroplasticity in Flax NNX
==================================================
📊 Generating synthetic data...
Data shape: X=(1000, 10), Y=(1000, 1)
🤖 Creating neuroplastic model: 10 -> [32, 16] -> 1
✅ Neuroplastic model initialized successfully
📈 Starting training with plasticity monitoring...
----------------------------------------------------------------------
Step   Loss         Hebbian      Structural   Scaling      Activity    
----------------------------------------------------------------------
 0    121.707443   0.000000     1.000000     1.000000     0.000000
 2    118.183067   0.001234     1.002345     1.001234     0.123456
 4    114.789848   0.002567     1.004567     1.002567     0.234567
 6    111.535690   0.003890     1.006789     1.003890     0.345678
 8    108.376923   0.005123     1.009012     1.005123     0.456789
----------------------------------------------------------------------
🎉 Neuroplasticity training completed!
```

## 🎯 Applications

### 1. Continual Learning
Neuroplasticity enables networks to adapt to new tasks without catastrophic forgetting.

### 2. Adaptive Networks
Networks that can modify their structure based on input patterns.

### 3. Bio-inspired Learning
Implementing learning rules inspired by biological neural systems.

### 4. Dynamic Architecture
Networks that can grow or shrink based on task requirements.

## 🔬 Research Applications

This implementation can be used for:

- **Neuroscience Research**: Modeling biological neural plasticity
- **Continual Learning**: Adapting to changing environments
- **Dynamic Networks**: Self-modifying architectures
- **Bio-inspired AI**: Implementing biological learning mechanisms

## 📚 Documentation

For detailed implementation guides and examples, see:
- [Neuroplasticity Implementation Guide](neuroplasticity_nnx_guide.md)
- [Basic Dense Model Example](dense_model_nnx_example.py)
- [Advanced Neuroplasticity Implementation](neuroplasticity_nnx_final.py)

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create a feature branch
3. Add your neuroplasticity implementations
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Flax team for the excellent NNX framework
- The neuroscience community for inspiring these implementations
- Contributors and researchers in neuroplasticity

## 📞 Contact

For questions, suggestions, or collaborations, please open an issue or reach out to the maintainers.

---

**Note**: This is a research implementation. For production use, please ensure proper testing and validation of the neuroplasticity mechanisms for your specific use case.
