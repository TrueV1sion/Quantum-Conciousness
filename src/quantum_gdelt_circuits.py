import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from dataclasses import dataclass

from .config.gdelt_config import QuantumEncodingConfig
from .gdelt_integration import GDELTEvent

@dataclass
class CircuitParameters:
    """Parameters for quantum circuit generation."""
    event_amplitude: Parameter
    tone_phase: Parameter
    impact_rotation: Parameter
    entanglement_strength: Parameter

class GDELTQuantumCircuitGenerator:
    """Generates quantum circuits for GDELT event processing."""
    
    def __init__(self, config: QuantumEncodingConfig):
        self.config = config
        self.parameters = self._initialize_parameters()
    
    def _initialize_parameters(self) -> CircuitParameters:
        """Initialize circuit parameters."""
        return CircuitParameters(
            event_amplitude=Parameter('event_amplitude'),
            tone_phase=Parameter('tone_phase'),
            impact_rotation=Parameter('impact_rotation'),
            entanglement_strength=Parameter('entanglement_strength')
        )
    
    def create_event_encoding_circuit(self, 
                                    events: List[GDELTEvent]) -> Tuple[QuantumCircuit, Dict]:
        """Create quantum circuit for encoding GDELT events."""
        # Initialize quantum registers
        q_reg = QuantumRegister(self.config.qubit_count, 'q')
        c_reg = ClassicalRegister(self.config.qubit_count, 'c')
        circuit = QuantumCircuit(q_reg, c_reg)
        
        # Parameter bindings for the circuit
        param_bindings = {}
        
        # Encode each event into the quantum state
        for idx, event in enumerate(events):
            # Calculate parameter values
            amplitude = self._calculate_amplitude(event)
            phase = self._calculate_phase(event)
            impact = self._calculate_impact_rotation(event)
            
            # Add parameter bindings
            param_bindings.update({
                f'event_{idx}_amplitude': amplitude,
                f'event_{idx}_phase': phase,
                f'event_{idx}_impact': impact
            })
            
            # Add encoding gates
            self._add_event_encoding_gates(circuit, q_reg, idx, event)
        
        # Add entanglement layers
        self._add_entanglement_layers(circuit, q_reg)
        
        # Add error correction if enabled
        if self.config.error_correction:
            self._add_error_correction(circuit, q_reg)
        
        return circuit, param_bindings
    
    def _calculate_amplitude(self, event: GDELTEvent) -> float:
        """Calculate amplitude encoding for an event."""
        # Normalize impact score to [0, 1]
        return min(max(event.impact_score, 0.0), 1.0)
    
    def _calculate_phase(self, event: GDELTEvent) -> float:
        """Calculate phase encoding based on event tone."""
        # Map tone from [-10, 10] to [0, 2π]
        normalized_tone = (event.tone + 10) / 20
        return normalized_tone * 2 * np.pi
    
    def _calculate_impact_rotation(self, event: GDELTEvent) -> float:
        """Calculate rotation angle based on event impact."""
        return event.impact_score * np.pi
    
    def _add_event_encoding_gates(self,
                                circuit: QuantumCircuit,
                                q_reg: QuantumRegister,
                                idx: int,
                                event: GDELTEvent) -> None:
        """Add quantum gates for encoding a single event."""
        # Amplitude encoding
        circuit.ry(
            self.parameters.event_amplitude,
            q_reg[idx % self.config.qubit_count]
        )
        
        # Phase encoding
        circuit.rz(
            self.parameters.tone_phase,
            q_reg[idx % self.config.qubit_count]
        )
        
        # Impact rotation
        circuit.rx(
            self.parameters.impact_rotation,
            q_reg[idx % self.config.qubit_count]
        )
    
    def _add_entanglement_layers(self,
                                circuit: QuantumCircuit,
                                q_reg: QuantumRegister) -> None:
        """Add entanglement layers to the circuit."""
        for layer in range(self.config.entanglement_layers):
            # Add CNOT gates between adjacent qubits
            for i in range(self.config.qubit_count - 1):
                circuit.cx(q_reg[i], q_reg[i + 1])
            
            # Add parametric rotation gates
            for i in range(self.config.qubit_count):
                circuit.rz(self.parameters.entanglement_strength, q_reg[i])
    
    def _add_error_correction(self,
                            circuit: QuantumCircuit,
                            q_reg: QuantumRegister) -> None:
        """Add basic error correction to the circuit."""
        # Implement a simple bit-flip error correction
        for i in range(0, self.config.qubit_count - 2, 3):
            # Add parity check ancilla
            circuit.cx(q_reg[i], q_reg[i + 1])
            circuit.cx(q_reg[i], q_reg[i + 2])
    
    def create_measurement_circuit(self) -> QuantumCircuit:
        """Create circuit for measuring in specified basis."""
        q_reg = QuantumRegister(self.config.qubit_count, 'q')
        c_reg = ClassicalRegister(self.config.qubit_count, 'c')
        circuit = QuantumCircuit(q_reg, c_reg)
        
        if self.config.measurement_basis == "computational":
            # Standard computational basis measurement
            circuit.measure(q_reg, c_reg)
        else:
            # Add basis rotation gates before measurement
            for i in range(self.config.qubit_count):
                circuit.h(q_reg[i])  # Hadamard for X-basis
                circuit.measure(q_reg[i], c_reg[i])
        
        return circuit 