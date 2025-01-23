"""
Practical Applications of Quantum-Inspired Optimization.

This module implements real-world applications using quantum-inspired
optimization for NLP, Computer Vision, and Time Series tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from quantum_operators import QuantumCircuit, NoiseModel
from tensor_quantum_optimization import TensorTrain, SimulatedQuantumEmbedding


class QuantumNLPModel(nn.Module):
    """
    Quantum-inspired NLP model for text processing tasks.
    Uses quantum circuits for attention and tensor networks for embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_qubits: int = 4,
        circuit_depth: int = 2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Quantum embedding layer
        self.embedding = SimulatedQuantumEmbedding(
            embedding_dim=embedding_dim,
            num_qubits=num_qubits
        )
        
        # Quantum circuit for processing
        self.circuit = QuantumCircuit(
            num_qubits=num_qubits,
            depth=circuit_depth,
            noise_model=NoiseModel(
                decoherence_rate=0.01,
                depolarization_rate=0.01,
                measurement_error_rate=0.01
            )
        )
        
        # Output projection
        self.output_proj = nn.Linear(2 ** num_qubits, vocab_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process input text through quantum circuit."""
        # Create embeddings
        embeddings = self.embedding.embed(
            F.one_hot(input_ids, self.vocab_size).float()
        )
        
        # Apply quantum circuit
        batch_size, seq_len, _ = embeddings.shape
        quantum_states = []
        
        for i in range(seq_len):
            state = embeddings[:, i]
            if attention_mask is None or attention_mask[:, i].bool().all():
                state = self.circuit(state)
            quantum_states.append(state)
        
        processed = torch.stack(quantum_states, dim=1)
        
        # Project to vocabulary space
        return self.output_proj(processed)


class QuantumVisionModel(nn.Module):
    """
    Quantum-inspired computer vision model.
    Uses tensor networks for feature extraction and quantum circuits.
    """
    
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        feature_dim: int = 64,
        num_qubits: int = 6
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Feature extraction using tensor networks
        self.feature_extractor = nn.Sequential(
            QuantumConvBlock(input_channels, 32),
            QuantumConvBlock(32, 64),
            QuantumConvBlock(64, feature_dim)
        )
        
        # Quantum circuit for processing
        self.circuit = QuantumCircuit(
            num_qubits=num_qubits,
            depth=3,
            noise_model=NoiseModel(
                decoherence_rate=0.005,
                depolarization_rate=0.005,
                measurement_error_rate=0.005
            )
        )
        
        # Classification head
        self.classifier = nn.Linear(2 ** num_qubits, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process image through quantum-inspired model."""
        # Extract features
        features = self.feature_extractor(x)
        
        # Prepare quantum state
        batch_size = features.shape[0]
        features = features.view(batch_size, -1)
        features = features / torch.norm(features, dim=1, keepdim=True)
        
        # Apply quantum circuit
        quantum_state = self.circuit(features)
        
        # Classify
        return self.classifier(quantum_state)


class QuantumTimeSeriesModel(nn.Module):
    """
    Quantum-inspired time series model.
    Uses quantum circuits for temporal processing.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_qubits: int = 4,
        sequence_length: int = 10
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, 2 ** num_qubits)
        
        # Quantum circuit per timestep
        self.circuits = nn.ModuleList([
            QuantumCircuit(
                num_qubits=num_qubits,
                depth=2,
                noise_model=NoiseModel(
                    decoherence_rate=0.01,
                    depolarization_rate=0.01,
                    measurement_error_rate=0.01
                )
            )
            for _ in range(sequence_length)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(2 ** num_qubits, output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process time series through quantum circuits."""
        batch_size, seq_len, _ = x.shape
        
        # Project input to quantum state space
        states = self.input_proj(x)
        states = states / torch.norm(states, dim=2, keepdim=True)
        
        # Process each timestep
        outputs = []
        hidden = None
        
        for t in range(seq_len):
            if mask is None or mask[:, t].bool().all():
                state = states[:, t]
                # Apply quantum circuit with temporal connection
                if hidden is not None:
                    state = state + 0.1 * hidden
                    state = state / torch.norm(state, dim=1, keepdim=True)
                state = self.circuits[t](state)
                hidden = state
                outputs.append(state)
        
        # Combine outputs
        combined = torch.stack(outputs, dim=1)
        
        # Project to output space
        return self.output_proj(combined)


class QuantumConvBlock(nn.Module):
    """Quantum-inspired convolutional block."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Decompose convolution kernel using tensor train
        self.tt_conv = TensorTrain(
            shape=(out_channels, in_channels, 3, 3),
            rank=[4, 4, 4]
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantum-inspired convolution."""
        # Get convolution kernel
        weight = self.tt_conv.to_tensor()
        
        # Apply convolution
        x = F.conv2d(x, weight, padding=1)
        x = self.bn(x)
        x = self.activation(x)
        
        return x


def example_nlp():
    """Example usage of quantum NLP model."""
    model = QuantumNLPModel(
        vocab_size=1000,
        embedding_dim=64
    )
    
    # Create dummy input
    batch_size = 16
    seq_len = 20
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    output = model(input_ids)
    print(f"NLP output shape: {output.shape}")


def example_vision():
    """Example usage of quantum vision model."""
    model = QuantumVisionModel(
        input_channels=3,
        num_classes=10
    )
    
    # Create dummy input
    batch_size = 16
    x = torch.randn(batch_size, 3, 32, 32)
    
    # Forward pass
    output = model(x)
    print(f"Vision output shape: {output.shape}")


def example_time_series():
    """Example usage of quantum time series model."""
    model = QuantumTimeSeriesModel(
        input_dim=10,
        hidden_dim=32,
        output_dim=1
    )
    
    # Create dummy input
    batch_size = 16
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 10)
    
    # Forward pass
    output = model(x)
    print(f"Time series output shape: {output.shape}")


if __name__ == "__main__":
    example_nlp()
    example_vision()
    example_time_series() 