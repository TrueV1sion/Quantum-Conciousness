# config.py

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum
import torch

class ProcessingDimension(Enum):
    """Processing dimensions for the system."""
    QUANTUM = "quantum"
    CLASSICAL = "classical"
    HYBRID = "hybrid"

@dataclass
class SystemConfig:
    """System configuration settings."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_qubits: int = 8
    batch_size: int = 32
    learning_rate: float = 0.001
    processing_dimension: ProcessingDimension = ProcessingDimension.HYBRID
    quantum_params: Optional[Dict[str, float]] = None
    classical_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.quantum_params is None:
            self.quantum_params = {
                'entanglement_strength': 0.7,
                'decoherence_rate': 0.1,
                'phase_damping': 0.05,
                'amplitude_damping': 0.05
            }
        
        if self.classical_params is None:
            self.classical_params = {
                'scaling_factor': 1.1,
                'noise_level': 0.1,
                'threshold': 0.5
            }

@dataclass
class UnifiedState:
    """Unified state representation."""
    quantum_field: torch.Tensor
    classical_state: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def validate(self):
        """Validate state components."""
        if not isinstance(self.quantum_field, torch.Tensor):
            raise ValueError("quantum_field must be a torch.Tensor")
        
        if not isinstance(self.classical_state, dict):
            raise ValueError("classical_state must be a dictionary")
        
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary if provided")

@dataclass
class ProcessingConfig:
    """Configuration for state processing."""
    dimension: ProcessingDimension
    batch_size: int = 32
    num_iterations: int = 100
    convergence_threshold: float = 1e-6
    max_time: float = 3600  # seconds
    
    def validate(self):
        """Validate processing configuration."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        
        if self.convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be positive")
        
        if self.max_time <= 0:
            raise ValueError("max_time must be positive")
