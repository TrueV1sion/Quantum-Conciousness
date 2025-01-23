import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..processors import UnifiedState
from ..quantum_pqc import ParameterizedQuantumCircuit, PQCConfig
from ..quantum_bridge_google import GoogleQuantumBridge


@dataclass
class QuantumXAIConfig:
    """Configuration for quantum XAI components."""
    visualization_path: str = "visualizations"
    attention_threshold: float = 0.1
    saliency_threshold: float = 0.05
    feature_importance_samples: int = 100
    layer_activation_bins: int = 50


class QuantumAttentionVisualizer:
    """Visualizes attention patterns in quantum circuits."""
    
    def __init__(self, config: QuantumXAIConfig):
        self.config = config
        self.viz_path = Path(config.visualization_path)
        self.viz_path.mkdir(parents=True, exist_ok=True)
    
    def visualize_circuit_attention(
        self,
        circuit: ParameterizedQuantumCircuit,
        attention_weights: torch.Tensor,
        timestamp: str
    ) -> Dict[str, Any]:
        """Visualize attention weights in the quantum circuit."""
        # Convert attention weights to numpy
        weights = attention_weights.detach().cpu().numpy()
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            weights,
            cmap='viridis',
            xticklabels=[f'Q{i}' for i in range(weights.shape[1])],
            yticklabels=[f'L{i}' for i in range(weights.shape[0])]
        )
        plt.title('Quantum Circuit Attention Weights')
        plt.xlabel('Qubits')
        plt.ylabel('Circuit Layers')
        
        # Save visualization
        viz_file = self.viz_path / f'quantum_attention_{timestamp}.png'
        plt.savefig(viz_file)
        plt.close()
        
        # Calculate attention statistics
        stats = {
            'max_attention': float(np.max(weights)),
            'mean_attention': float(np.mean(weights)),
            'attention_std': float(np.std(weights)),
            'high_attention_qubits': np.where(
                weights.mean(axis=0) > self.config.attention_threshold
            )[0].tolist()
        }
        
        return {
            'visualization_path': str(viz_file),
            'attention_stats': stats
        }


class QuantumSaliencyMapper:
    """Generates saliency maps for quantum states."""
    
    def __init__(self, config: QuantumXAIConfig):
        self.config = config
        self.viz_path = Path(config.visualization_path)
        self.viz_path.mkdir(parents=True, exist_ok=True)
    
    def compute_saliency(
        self,
        state: UnifiedState[Any],
        pqc: ParameterizedQuantumCircuit,
        timestamp: str
    ) -> Dict[str, Any]:
        """Compute saliency maps for quantum states."""
        # Get quantum field
        quantum_field = state.quantum_field
        quantum_field.requires_grad_(True)
        
        # Forward pass through PQC
        output = torch.tensor(
            pqc.evaluate_circuit(quantum_field),
            requires_grad=True
        )
        
        # Compute gradients
        output.sum().backward()
        saliency = quantum_field.grad.abs()
        
        # Normalize saliency
        saliency = saliency / saliency.max()
        
        # Visualize saliency
        plt.figure(figsize=(10, 6))
        plt.plot(saliency.detach().cpu().numpy())
        plt.title('Quantum State Saliency Map')
        plt.xlabel('Qubit Index')
        plt.ylabel('Saliency Value')
        
        # Save visualization
        viz_file = self.viz_path / f'quantum_saliency_{timestamp}.png'
        plt.savefig(viz_file)
        plt.close()
        
        # Calculate saliency statistics
        stats = {
            'max_saliency': float(saliency.max()),
            'mean_saliency': float(saliency.mean()),
            'saliency_std': float(saliency.std()),
            'important_qubits': torch.where(
                saliency > self.config.saliency_threshold
            )[0].tolist()
        }
        
        return {
            'visualization_path': str(viz_file),
            'saliency_stats': stats,
            'saliency_values': saliency.detach().cpu().numpy().tolist()
        }


class QuantumFeatureAnalyzer:
    """Analyzes importance of quantum features."""
    
    def __init__(self, config: QuantumXAIConfig):
        self.config = config
        self.viz_path = Path(config.visualization_path)
        self.viz_path.mkdir(parents=True, exist_ok=True)
    
    def analyze_feature_importance(
        self,
        state: UnifiedState[Any],
        pqc: ParameterizedQuantumCircuit,
        timestamp: str
    ) -> Dict[str, Any]:
        """Analyze importance of quantum features through perturbation."""
        quantum_field = state.quantum_field.detach()
        base_output = torch.tensor(pqc.evaluate_circuit(quantum_field))
        
        # Initialize importance scores
        importance_scores = torch.zeros_like(quantum_field)
        
        # Perturbation analysis
        for _ in range(self.config.feature_importance_samples):
            # Add random perturbation
            perturbed = quantum_field + 0.1 * torch.randn_like(quantum_field)
            perturbed_output = torch.tensor(pqc.evaluate_circuit(perturbed))
            
            # Calculate impact
            impact = torch.abs(perturbed_output - base_output)
            importance_scores += impact
        
        # Normalize scores
        importance_scores /= self.config.feature_importance_samples
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(
            range(len(importance_scores)),
            importance_scores.cpu().numpy()
        )
        plt.title('Quantum Feature Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance Score')
        
        # Save visualization
        viz_file = self.viz_path / f'feature_importance_{timestamp}.png'
        plt.savefig(viz_file)
        plt.close()
        
        return {
            'visualization_path': str(viz_file),
            'importance_scores': importance_scores.cpu().numpy().tolist(),
            'top_features': torch.topk(
                importance_scores,
                k=min(5, len(importance_scores))
            ).indices.tolist()
        }


class QuantumLayerAnalyzer:
    """Analyzes quantum circuit layer activations."""
    
    def __init__(self, config: QuantumXAIConfig):
        self.config = config
        self.viz_path = Path(config.visualization_path)
        self.viz_path.mkdir(parents=True, exist_ok=True)
    
    def analyze_layer_activations(
        self,
        circuit: ParameterizedQuantumCircuit,
        state: UnifiedState[Any],
        timestamp: str
    ) -> Dict[str, Any]:
        """Analyze activations at different layers of the quantum circuit."""
        quantum_field = state.quantum_field
        layer_activations = []
        
        # Get circuit
        full_circuit = circuit.create_circuit()
        
        # Analyze each layer
        current_state = quantum_field
        for layer_idx, moment in enumerate(full_circuit):
            # Apply operations in this layer
            result = circuit.simulator.simulate(
                cirq.Circuit(moment),
                initial_state=current_state.detach().cpu().numpy()
            )
            current_state = torch.tensor(result.final_state_vector)
            layer_activations.append(current_state)
        
        # Convert to tensor
        layer_activations = torch.stack(layer_activations)
        
        # Visualize layer activations
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            np.abs(layer_activations.cpu().numpy()),
            cmap='viridis',
            xticklabels=[f'Q{i}' for i in range(layer_activations.shape[1])],
            yticklabels=[f'L{i}' for i in range(layer_activations.shape[0])]
        )
        plt.title('Quantum Circuit Layer Activations')
        plt.xlabel('Qubit Index')
        plt.ylabel('Layer Index')
        
        # Save visualization
        viz_file = self.viz_path / f'layer_activations_{timestamp}.png'
        plt.savefig(viz_file)
        plt.close()
        
        # Calculate activation statistics
        activation_stats = {
            'mean_activation': float(torch.abs(layer_activations).mean()),
            'max_activation': float(torch.abs(layer_activations).max()),
            'activation_std': float(torch.abs(layer_activations).std()),
            'layer_complexities': [
                float(torch.abs(layer).mean())
                for layer in layer_activations
            ]
        }
        
        return {
            'visualization_path': str(viz_file),
            'activation_stats': activation_stats,
            'layer_activations': layer_activations.cpu().numpy().tolist()
        }


class QuantumXAIManager:
    """Manages quantum XAI components and analysis."""
    
    def __init__(self, config: QuantumXAIConfig):
        self.config = config
        self.attention_visualizer = QuantumAttentionVisualizer(config)
        self.saliency_mapper = QuantumSaliencyMapper(config)
        self.feature_analyzer = QuantumFeatureAnalyzer(config)
        self.layer_analyzer = QuantumLayerAnalyzer(config)
    
    async def analyze_quantum_state(
        self,
        state: UnifiedState[Any],
        pqc: ParameterizedQuantumCircuit,
        timestamp: str
    ) -> Dict[str, Any]:
        """Perform comprehensive XAI analysis of quantum state."""
        # Attention analysis
        attention_results = self.attention_visualizer.visualize_circuit_attention(
            pqc,
            state.quantum_field,
            timestamp
        )
        
        # Saliency analysis
        saliency_results = self.saliency_mapper.compute_saliency(
            state,
            pqc,
            timestamp
        )
        
        # Feature importance analysis
        feature_results = self.feature_analyzer.analyze_feature_importance(
            state,
            pqc,
            timestamp
        )
        
        # Layer activation analysis
        layer_results = self.layer_analyzer.analyze_layer_activations(
            pqc,
            state,
            timestamp
        )
        
        # Combine results
        return {
            'attention_analysis': attention_results,
            'saliency_analysis': saliency_results,
            'feature_importance': feature_results,
            'layer_activations': layer_results,
            'timestamp': timestamp,
            'quantum_state_id': id(state)
        }
    
    def generate_xai_report(
        self,
        analysis_results: Dict[str, Any]
    ) -> str:
        """Generate a human-readable report of XAI analysis."""
        report = []
        report.append("Quantum XAI Analysis Report")
        report.append("=========================")
        report.append(f"Timestamp: {analysis_results['timestamp']}")
        report.append(f"State ID: {analysis_results['quantum_state_id']}")
        report.append("")
        
        # Attention analysis
        report.append("Attention Analysis")
        report.append("-----------------")
        attn = analysis_results['attention_analysis']['attention_stats']
        report.append(f"Max Attention: {attn['max_attention']:.4f}")
        report.append(f"Mean Attention: {attn['mean_attention']:.4f}")
        report.append(
            f"High Attention Qubits: {', '.join(map(str, attn['high_attention_qubits']))}"
        )
        report.append("")
        
        # Saliency analysis
        report.append("Saliency Analysis")
        report.append("----------------")
        sal = analysis_results['saliency_analysis']['saliency_stats']
        report.append(f"Max Saliency: {sal['max_saliency']:.4f}")
        report.append(f"Mean Saliency: {sal['mean_saliency']:.4f}")
        report.append(
            f"Important Qubits: {', '.join(map(str, sal['important_qubits']))}"
        )
        report.append("")
        
        # Feature importance
        report.append("Feature Importance")
        report.append("-----------------")
        feat = analysis_results['feature_importance']
        report.append(
            f"Top Features: {', '.join(map(str, feat['top_features']))}"
        )
        report.append("")
        
        # Layer activations
        report.append("Layer Activation Analysis")
        report.append("----------------------")
        act = analysis_results['layer_activations']['activation_stats']
        report.append(f"Mean Activation: {act['mean_activation']:.4f}")
        report.append(f"Max Activation: {act['max_activation']:.4f}")
        report.append("")
        
        # Visualization paths
        report.append("Visualization Files")
        report.append("------------------")
        report.append(
            f"Attention: {analysis_results['attention_analysis']['visualization_path']}"
        )
        report.append(
            f"Saliency: {analysis_results['saliency_analysis']['visualization_path']}"
        )
        report.append(
            f"Feature Importance: {analysis_results['feature_importance']['visualization_path']}"
        )
        report.append(
            f"Layer Activations: {analysis_results['layer_activations']['visualization_path']}"
        )
        
        return "\n".join(report) 