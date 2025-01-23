"""
Example notebook demonstrating quantum XAI capabilities.

This notebook shows how to use the XAI framework to analyze and visualize:
1. Quantum circuit attention patterns
2. State saliency maps
3. Feature importance analysis
4. Layer activation patterns
"""

import torch
import numpy as np
from datetime import datetime
from pathlib import Path

from src.quantum_pqc import PQCConfig, HybridQuantumProcessor
from src.quantum_bridge_google import BridgeConfig
from src.processors import UnifiedState
from src.xai.quantum_xai import QuantumXAIConfig

# Create output directory
output_dir = Path("quantum_xai_output")
output_dir.mkdir(parents=True, exist_ok=True)

async def run_basic_xai_example():
    """Basic example of quantum XAI analysis."""
    print("Running basic XAI example...")
    
    # Configure XAI
    xai_config = QuantumXAIConfig(
        visualization_path=str(output_dir),
        attention_threshold=0.1,
        saliency_threshold=0.05,
        feature_importance_samples=100
    )
    
    # Configure quantum components
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
    
    # Create processor
    processor = HybridQuantumProcessor(pqc_config, bridge_config)
    
    # Create example quantum state
    quantum_field = torch.randn(8, dtype=torch.complex64)
    quantum_field = quantum_field / torch.norm(quantum_field)
    
    state = UnifiedState(
        consciousness_field=None,
        quantum_field=quantum_field,
        classical_state={},
        metadata={'example_type': 'basic_xai'}
    )
    
    # Define loss function
    def loss_function(quantum_output: torch.Tensor) -> torch.Tensor:
        target = torch.ones_like(quantum_output) * 0.7
        return torch.nn.functional.mse_loss(quantum_output, target)
    
    # Process state with XAI analysis
    print("Processing state with XAI analysis...")
    new_state = await processor.process_state(state, loss_function)
    
    # Print XAI report
    if 'xai_results' in new_state.classical_state:
        print("\nXAI Analysis Results:")
        print("====================")
        
        xai_results = new_state.classical_state['xai_results']
        
        # Attention analysis
        attn = xai_results['attention_analysis']['attention_stats']
        print("\nAttention Analysis:")
        print(f"Max Attention: {attn['max_attention']:.4f}")
        print(f"Mean Attention: {attn['mean_attention']:.4f}")
        print(f"High Attention Qubits: {attn['high_attention_qubits']}")
        
        # Saliency analysis
        sal = xai_results['saliency_analysis']['saliency_stats']
        print("\nSaliency Analysis:")
        print(f"Max Saliency: {sal['max_saliency']:.4f}")
        print(f"Mean Saliency: {sal['mean_saliency']:.4f}")
        print(f"Important Qubits: {sal['important_qubits']}")
        
        # Feature importance
        feat = xai_results['feature_importance']
        print("\nFeature Importance:")
        print(f"Top Features: {feat['top_features']}")
        
        # Layer activations
        act = xai_results['layer_activations']['activation_stats']
        print("\nLayer Activation Analysis:")
        print(f"Mean Activation: {act['mean_activation']:.4f}")
        print(f"Max Activation: {act['max_activation']:.4f}")
        
        print("\nVisualization files have been saved to:", output_dir)
    else:
        print("No XAI results found in processed state")

async def run_optimization_trajectory_example():
    """Example showing XAI analysis of optimization trajectory."""
    print("\nRunning optimization trajectory XAI example...")
    
    # Configure with more optimization steps
    xai_config = QuantumXAIConfig(
        visualization_path=str(output_dir / "optimization_trajectory"),
        attention_threshold=0.15,
        saliency_threshold=0.1,
        feature_importance_samples=50
    )
    
    pqc_config = PQCConfig(
        num_qubits=4,
        num_layers=2,
        num_parameters=16,
        optimization_steps=50,
        learning_rate=0.02,
        xai_config=xai_config
    )
    
    bridge_config = BridgeConfig(
        num_qubits=4,
        num_layers=2,
        pqc_config=pqc_config
    )
    
    processor = HybridQuantumProcessor(pqc_config, bridge_config)
    
    # Create initial state
    quantum_field = torch.randn(4, dtype=torch.complex64)
    quantum_field = quantum_field / torch.norm(quantum_field)
    
    state = UnifiedState(
        consciousness_field=None,
        quantum_field=quantum_field,
        classical_state={},
        metadata={'example_type': 'optimization_trajectory'}
    )
    
    # Define loss function with a specific target state
    target_state = torch.ones(4, dtype=torch.complex64) / 2
    def loss_function(quantum_output: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(quantum_output, target_state)
    
    # Process and analyze
    print("Processing state with optimization trajectory analysis...")
    new_state = await processor.process_state(state, loss_function)
    
    # Analyze optimization trajectory
    if 'optimization_history' in new_state.classical_state:
        history = new_state.classical_state['optimization_history']
        
        print("\nOptimization Trajectory Analysis:")
        print("===============================")
        print(f"Number of steps: {len(history['loss_values'])}")
        print(f"Initial loss: {history['loss_values'][0]:.4f}")
        print(f"Final loss: {history['loss_values'][-1]:.4f}")
        
        # Plot optimization trajectory
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss_values'])
        plt.title('Optimization Trajectory')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.savefig(output_dir / "optimization_trajectory" / "loss_trajectory.png")
        plt.close()
        
        print("\nOptimization trajectory plot saved to:", 
              output_dir / "optimization_trajectory")

async def run_noise_analysis_example():
    """Example showing XAI analysis with different noise levels."""
    print("\nRunning noise analysis XAI example...")
    
    # Test different noise levels
    noise_levels = [0.01, 0.05, 0.1]
    
    for noise_level in noise_levels:
        print(f"\nAnalyzing noise level: {noise_level}")
        
        xai_config = QuantumXAIConfig(
            visualization_path=str(output_dir / f"noise_{noise_level}"),
            attention_threshold=0.1,
            saliency_threshold=0.05
        )
        
        pqc_config = PQCConfig(
            num_qubits=6,
            num_layers=2,
            num_parameters=12,
            noise_scale=noise_level,
            xai_config=xai_config
        )
        
        bridge_config = BridgeConfig(
            num_qubits=6,
            num_layers=2,
            noise_level=noise_level,
            pqc_config=pqc_config
        )
        
        processor = HybridQuantumProcessor(pqc_config, bridge_config)
        
        # Create noisy state
        quantum_field = torch.randn(6, dtype=torch.complex64)
        quantum_field = quantum_field / torch.norm(quantum_field)
        noise = torch.randn_like(quantum_field) * noise_level
        noisy_field = quantum_field + noise
        noisy_field = noisy_field / torch.norm(noisy_field)
        
        state = UnifiedState(
            consciousness_field=None,
            quantum_field=noisy_field,
            classical_state={},
            metadata={'example_type': f'noise_analysis_{noise_level}'}
        )
        
        def loss_function(quantum_output: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.mse_loss(
                quantum_output,
                quantum_field
            )
        
        # Process and analyze
        new_state = await processor.process_state(state, loss_function)
        
        if 'xai_results' in new_state.classical_state:
            xai_results = new_state.classical_state['xai_results']
            
            print(f"\nNoise Level {noise_level} Analysis:")
            print("------------------------")
            
            # Attention analysis
            attn = xai_results['attention_analysis']['attention_stats']
            print(f"Mean Attention: {attn['mean_attention']:.4f}")
            
            # Saliency analysis
            sal = xai_results['saliency_analysis']['saliency_stats']
            print(f"Mean Saliency: {sal['mean_saliency']:.4f}")
            
            # Layer activations
            act = xai_results['layer_activations']['activation_stats']
            print(f"Mean Activation: {act['mean_activation']:.4f}")

async def main():
    """Run all examples."""
    print("Starting Quantum XAI Examples...")
    print("===============================")
    
    await run_basic_xai_example()
    await run_optimization_trajectory_example()
    await run_noise_analysis_example()
    
    print("\nAll examples completed. Results saved in:", output_dir)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 