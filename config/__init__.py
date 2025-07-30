from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any


@dataclass
class BridgeConfig:
    """Configuration for quantum-consciousness bridge."""
    consciousness_dim: int = 1024
    quantum_dim: int = 512
    hidden_dim: int = 1024
    buffer_size: int = 1024
    batch_size: int = 32
    history_size: int = 512
    learning_rate: float = 1e-4
    warmup_steps: int = 1000

    # Quantum processing parameters
    num_qubits: int = 32
    entanglement_threshold: float = 0.5
    coherence_threshold: float = 0.7
    resonance_threshold: float = 0.6

    # Model parameters
    max_sequence_length: int = 512
    attention_heads: int = 16
    num_layers: int = 12
    dropout_rate: float = 0.1

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.consciousness_dim > 0
        assert self.quantum_dim > 0
        assert self.hidden_dim > 0
        assert self.num_qubits > 0
        assert 0 <= self.entanglement_threshold <= 1
        assert 0 <= self.coherence_threshold <= 1
        assert 0 <= self.resonance_threshold <= 1


class SystemMode(Enum):
    INFERENCE = auto()
    EXTREME = auto()


@dataclass
class PathwayConfig:
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 512
    dropout: float = 0.1


@dataclass
class UnifiedConfig:
    bridge: BridgeConfig
    pathway: PathwayConfig
    max_sequence_length: int = 512


# Re-export definitions from src.config for convenience
from src.config import SystemConfig, UnifiedState, ProcessingDimension

__all__ = [
    "BridgeConfig",
    "SystemMode",
    "PathwayConfig",
    "UnifiedConfig",
    "SystemConfig",
    "UnifiedState",
    "ProcessingDimension",
]
