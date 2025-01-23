# Quantum XAI Tutorial

This tutorial demonstrates how to use the Quantum XAI framework to analyze and understand quantum circuits, states, and their behavior. The tutorial provides interactive visualizations and detailed analysis of quantum system behavior.

## Contents

1. `quantum_xai_tutorial.py` - Main tutorial script with detailed examples
2. `quantum_xai_examples.py` - Additional example scenarios
3. `scenario_examples.py` - Real-world application scenarios

## Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- torch
- numpy
- matplotlib
- seaborn
- cirq
- jupyter (optional, for notebook version)

## Running the Tutorial

### As a Python Script

```bash
python examples/quantum_xai_tutorial.py
```

### As a Jupyter Notebook

To convert the tutorial to a Jupyter notebook:

```bash
jupytext --to notebook quantum_xai_tutorial.py
jupyter notebook quantum_xai_tutorial.ipynb
```

## Tutorial Sections

### 1. Basic XAI Analysis

Learn how to:
- Visualize attention patterns in quantum circuits
- Generate and interpret saliency maps
- Analyze feature importance
- Understand layer activation patterns

### 2. Optimization Trajectory Analysis

Explore:
- Parameter evolution during training
- Loss landscape visualization
- Convergence behavior analysis
- Optimization bottleneck identification

### 3. Noise Analysis

Investigate:
- Impact of different noise levels
- Error correction effectiveness
- Circuit robustness metrics
- Stability analysis

## Output

The tutorial generates visualizations and analysis in the `quantum_xai_output` directory:

```
quantum_xai_output/
├── basic_example/
│   ├── quantum_attention.png
│   ├── quantum_saliency.png
│   ├── feature_importance.png
│   └── layer_activations.png
├── optimization_trajectory/
│   ├── loss_trajectory.png
│   └── state_evolution/
└── noise_analysis/
    ├── noise_0.01/
    ├── noise_0.05/
    └── noise_0.1/
```

## Key Features

1. **Interactive Visualizations**
   - Attention heatmaps
   - Saliency plots
   - Parameter evolution graphs
   - Noise impact analysis

2. **Comprehensive Analysis**
   - Circuit behavior insights
   - Optimization patterns
   - Noise robustness metrics
   - Error correction assessment

3. **Practical Applications**
   - Circuit design optimization
   - Parameter tuning
   - Error mitigation
   - Performance analysis

## Example Usage

```python
# Basic XAI analysis
from examples.quantum_xai_tutorial import plot_attention_patterns, plot_saliency_maps

# Configure XAI
xai_config = QuantumXAIConfig(
    visualization_path="output",
    attention_threshold=0.1,
    saliency_threshold=0.05
)

# Create processor
processor = HybridQuantumProcessor(pqc_config, bridge_config)

# Analyze quantum state
new_state = await processor.process_state(state, loss_function)
xai_results = new_state.classical_state['xai_results']

# Visualize results
plot_attention_patterns(xai_results['attention_analysis'])
plot_saliency_maps(xai_results['saliency_analysis'])
```

## Contributing

Feel free to contribute additional examples, visualizations, or analysis techniques. Please follow the existing code style and add appropriate documentation.

## License

MIT License - See LICENSE file for details 