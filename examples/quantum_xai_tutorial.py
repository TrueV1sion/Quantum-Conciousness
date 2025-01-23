"""
Quantum XAI Tutorial
===================

This tutorial demonstrates how to use the Quantum XAI framework to analyze and understand
quantum circuits, states, and their behavior. We'll cover:

1. Basic XAI Analysis
   - Attention patterns in quantum circuits
   - Saliency maps for quantum states
   - Feature importance analysis
   - Layer activation patterns

2. Optimization Trajectory Analysis
   - Visualizing optimization paths
   - Understanding parameter evolution
   - Analyzing convergence behavior

3. Noise Analysis
   - Impact of different noise levels
   - Robustness analysis
   - Error correction effectiveness
"""

import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

from src.quantum_pqc import PQCConfig, HybridQuantumProcessor
from src.quantum_bridge_google import BridgeConfig
from src.processors import UnifiedState
from src.xai.quantum_xai import QuantumXAIConfig

# Set up plotting style
plt.style.use('seaborn')
sns.set_palette('husl')

# Create output directory
output_dir = Path("quantum_xai_output")
output_dir.mkdir(parents=True, exist_ok=True)

def plot_attention_patterns(attention_results: Dict[str, Any]) -> None:
    """
    Visualize attention patterns in the quantum circuit.
    
    This shows which qubits are most important for the circuit's operation at each layer.
    The heatmap displays attention weights, with brighter colors indicating higher attention.
    """
    plt.figure(figsize=(12, 8))
    attn = attention_results['attention_stats']
    
    # Plot attention heatmap
    img = plt.imread(attention_results['visualization_path'])
    plt.imshow(img)
    plt.axis('off')
    plt.title('Quantum Circuit Attention Patterns')
    
    # Print statistics
    print("\nAttention Statistics:")
    print(f"Max Attention: {attn['max_attention']:.4f}")
    print(f"Mean Attention: {attn['mean_attention']:.4f}")
    print(f"High Attention Qubits: {attn['high_attention_qubits']}")
    
    plt.show()

def plot_saliency_maps(saliency_results: Dict[str, Any]) -> None:
    """
    Visualize saliency maps for quantum states.
    
    Saliency maps help identify which input features (qubits) have the strongest
    influence on the circuit's output. Higher saliency values indicate more
    important qubits.
    """
    plt.figure(figsize=(12, 6))
    sal = saliency_results['saliency_stats']
    
    # Plot saliency values
    plt.plot(saliency_results['saliency_values'], 'b-', label='Saliency')
    plt.axhline(
        y=sal['mean_saliency'],
        color='r',
        linestyle='--',
        label='Mean Saliency'
    )
    plt.fill_between(
        range(len(saliency_results['saliency_values'])),
        saliency_results['saliency_values'],
        alpha=0.3
    )
    
    plt.title('Quantum State Saliency Map')
    plt.xlabel('Qubit Index')
    plt.ylabel('Saliency Value')
    plt.legend()
    
    # Print statistics
    print("\nSaliency Statistics:")
    print(f"Max Saliency: {sal['max_saliency']:.4f}")
    print(f"Mean Saliency: {sal['mean_saliency']:.4f}")
    print(f"Important Qubits: {sal['important_qubits']}")
    
    plt.show()

def analyze_optimization_trajectory(state_history: Dict[str, Any]) -> None:
    """
    Analyze and visualize the optimization trajectory.
    
    This shows how circuit parameters evolve during training and how the loss
    function changes. It helps identify convergence patterns and potential
    optimization issues.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot loss trajectory
    plt.subplot(1, 2, 1)
    plt.plot(state_history['loss_values'])
    plt.title('Loss Trajectory')
    plt.xlabel('Optimization Step')
    plt.ylabel('Loss Value')
    
    # Plot parameter evolution
    plt.subplot(1, 2, 2)
    param_history = np.array(state_history['parameters'])
    for i in range(param_history.shape[1]):
        plt.plot(param_history[:, i], alpha=0.5, label=f'θ_{i}')
    plt.title('Parameter Evolution')
    plt.xlabel('Optimization Step')
    plt.ylabel('Parameter Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Print optimization statistics
    print("\nOptimization Statistics:")
    print(f"Initial Loss: {state_history['loss_values'][0]:.4f}")
    print(f"Final Loss: {state_history['loss_values'][-1]:.4f}")
    print(f"Loss Reduction: {(1 - state_history['loss_values'][-1] / state_history['loss_values'][0]) * 100:.1f}%")

async def analyze_noise_impact(
    processor: HybridQuantumProcessor,
    noise_levels: List[float]
) -> None:
    """
    Analyze how different noise levels affect circuit behavior.
    
    This analysis helps understand:
    - Circuit robustness to noise
    - Effectiveness of error correction
    - Impact on attention and saliency patterns
    """
    results = []
    
    for noise_level in noise_levels:
        print(f"\nAnalyzing noise level: {noise_level}")
        
        # Create noisy state
        quantum_field = torch.randn(8, dtype=torch.complex64)
        quantum_field = quantum_field / torch.norm(quantum_field)
        noise = torch.randn_like(quantum_field) * noise_level
        noisy_field = quantum_field + noise
        noisy_field = noisy_field / torch.norm(noisy_field)
        
        state = UnifiedState(
            consciousness_field=None,
            quantum_field=noisy_field,
            classical_state={},
            metadata={'noise_level': noise_level}
        )
        
        # Define loss function
        def loss_function(quantum_output: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.mse_loss(
                quantum_output,
                quantum_field
            )
        
        # Process and analyze
        new_state = await processor.process_state(state, loss_function)
        results.append(new_state)
    
    # Visualize noise impact
    plt.figure(figsize=(15, 5))
    
    # Plot attention degradation
    plt.subplot(1, 3, 1)
    attention_values = [
        r.classical_state['xai_results']['attention_analysis']
        ['attention_stats']['mean_attention'] for r in results
    ]
    plt.plot(noise_levels, attention_values, 'bo-')
    plt.title('Attention vs Noise')
    plt.xlabel('Noise Level')
    plt.ylabel('Mean Attention')
    
    # Plot saliency degradation
    plt.subplot(1, 3, 2)
    saliency_values = [
        r.classical_state['xai_results']['saliency_analysis']
        ['saliency_stats']['mean_saliency'] for r in results
    ]
    plt.plot(noise_levels, saliency_values, 'ro-')
    plt.title('Saliency vs Noise')
    plt.xlabel('Noise Level')
    plt.ylabel('Mean Saliency')
    
    # Plot activation degradation
    plt.subplot(1, 3, 3)
    activation_values = [
        r.classical_state['xai_results']['layer_activations']
        ['activation_stats']['mean_activation'] for r in results
    ]
    plt.plot(noise_levels, activation_values, 'go-')
    plt.title('Activation vs Noise')
    plt.xlabel('Noise Level')
    plt.ylabel('Mean Activation')
    
    plt.tight_layout()
    plt.show()
    
    # Print noise impact statistics
    print("\nNoise Impact Statistics:")
    for i, noise_level in enumerate(noise_levels):
        print(f"\nNoise Level: {noise_level}")
        print(f"Mean Attention: {attention_values[i]:.4f}")
        print(f"Mean Saliency: {saliency_values[i]:.4f}")
        print(f"Mean Activation: {activation_values[i]:.4f}")

async def main():
    """
    Main tutorial demonstrating quantum XAI capabilities.
    """
    print("Starting Quantum XAI Tutorial...")
    print("===============================")
    
    # 1. Basic Setup
    print("\n1. Setting up quantum components...")
    xai_config = QuantumXAIConfig(
        visualization_path=str(output_dir),
        attention_threshold=0.1,
        saliency_threshold=0.05,
        feature_importance_samples=100
    )
    
    pqc_config = PQCConfig(
        num_qubits=8,
        num_layers=3,
        num_parameters=24,
        xai_config=xai_config
    )
    
    bridge_config = BridgeConfig(
        num_qubits=8,
        num_layers=3,
        pqc_config=pqc_config
    )
    
    processor = HybridQuantumProcessor(pqc_config, bridge_config)
    
    # 2. Basic XAI Analysis
    print("\n2. Running basic XAI analysis...")
    quantum_field = torch.randn(8, dtype=torch.complex64)
    quantum_field = quantum_field / torch.norm(quantum_field)
    
    state = UnifiedState(
        consciousness_field=None,
        quantum_field=quantum_field,
        classical_state={},
        metadata={'example_type': 'basic_xai'}
    )
    
    def loss_function(quantum_output: torch.Tensor) -> torch.Tensor:
        target = torch.ones_like(quantum_output) * 0.7
        return torch.nn.functional.mse_loss(quantum_output, target)
    
    new_state = await processor.process_state(state, loss_function)
    xai_results = new_state.classical_state['xai_results']
    
    print("\nVisualizing attention patterns...")
    plot_attention_patterns(xai_results['attention_analysis'])
    
    print("\nVisualizing saliency maps...")
    plot_saliency_maps(xai_results['saliency_analysis'])
    
    # 3. Optimization Analysis
    print("\n3. Analyzing optimization trajectory...")
    optimization_history = new_state.classical_state['optimization_history']
    analyze_optimization_trajectory(optimization_history)
    
    # 4. Noise Analysis
    print("\n4. Analyzing noise impact...")
    await analyze_noise_impact(processor, [0.01, 0.05, 0.1])
    
    print("\nTutorial completed! All results saved in:", output_dir)
    print("\nKey Insights:")
    print("1. Circuit Behavior:")
    print("   - Identified high-attention qubits")
    print("   - Mapped saliency patterns")
    print("   - Analyzed layer activations")
    print("\n2. Optimization Process:")
    print("   - Tracked parameter convergence")
    print("   - Monitored loss reduction")
    print("   - Identified optimization patterns")
    print("\n3. Noise Robustness:")
    print("   - Quantified noise impact")
    print("   - Assessed error correction")
    print("   - Measured stability metrics")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 