# quantum_simulation.py

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from qiskit.extensions import UnitaryGate
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import logging

class QuantumSimulator:
    """
    Simulate quantum circuits and states using Qiskit.
    Provides integration between quantum simulations and the main system.
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize quantum simulator.
        
        Args:
            num_qubits: Number of qubits to simulate
        """
        self.num_qubits = num_qubits
        self.logger = logging.getLogger(__name__)
        self.backend = Aer.get_backend('statevector_simulator')
        self.current_state = None
        self.logger.info(f"Quantum Simulator initialized with {num_qubits} qubits")
    
    def create_circuit(self, operations: List[Dict[str, Any]] = None) -> QuantumCircuit:
        """
        Create a quantum circuit with specified operations.
        
        Args:
            operations: List of operations to apply, each specified as a dictionary
                       with 'type' and 'params' keys
            
        Returns:
            QuantumCircuit instance
        """
        try:
            circuit = QuantumCircuit(self.num_qubits)
            
            if operations:
                for op in operations:
                    self._apply_operation(circuit, op)
            else:
                # Default: Create superposition state
                for qubit in range(self.num_qubits):
                    circuit.h(qubit)
            
            return circuit
            
        except Exception as e:
            self.logger.error(f"Circuit creation failed: {str(e)}")
            raise
    
    def simulate_circuit(self, circuit: QuantumCircuit) -> Statevector:
        """
        Simulate quantum circuit and return statevector.
        
        Args:
            circuit: QuantumCircuit to simulate
            
        Returns:
            Resulting Statevector
        """
        try:
            job = execute(circuit, self.backend)
            result = job.result()
            statevector = result.get_statevector(circuit)
            self.current_state = Statevector(statevector)
            return self.current_state
            
        except Exception as e:
            self.logger.error(f"Circuit simulation failed: {str(e)}")
            raise
    
    def get_density_matrix(self) -> np.ndarray:
        """
        Get density matrix of current state.
        
        Returns:
            Density matrix as numpy array
        """
        if self.current_state is None:
            raise ValueError("No current state available. Run simulation first.")
        
        try:
            density_matrix = DensityMatrix(self.current_state)
            return density_matrix.data
            
        except Exception as e:
            self.logger.error(f"Density matrix calculation failed: {str(e)}")
            raise
    
    def get_entanglement_map(self) -> Dict[str, float]:
        """
        Calculate entanglement measures between qubits.
        
        Returns:
            Dictionary mapping qubit pairs to their entanglement measures
        """
        if self.current_state is None:
            raise ValueError("No current state available. Run simulation first.")
        
        try:
            density_matrix = self.get_density_matrix()
            entanglement_map = {}
            
            # Calculate pairwise entanglement
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    entanglement = self._calculate_entanglement(density_matrix, i, j)
                    entanglement_map[f"q{i}-q{j}"] = float(entanglement)
            
            return entanglement_map
            
        except Exception as e:
            self.logger.error(f"Entanglement mapping failed: {str(e)}")
            raise
    
    def state_to_tensor(self) -> torch.Tensor:
        """
        Convert current quantum state to PyTorch tensor.
        
        Returns:
            Quantum state as PyTorch tensor
        """
        if self.current_state is None:
            raise ValueError("No current state available. Run simulation first.")
        
        try:
            state_vector = self.current_state.data
            return torch.from_numpy(np.abs(state_vector)).float()
            
        except Exception as e:
            self.logger.error(f"State to tensor conversion failed: {str(e)}")
            raise
    
    def tensor_to_state(self, tensor: torch.Tensor) -> None:
        """
        Convert PyTorch tensor to quantum state.
        
        Args:
            tensor: PyTorch tensor representing quantum state
        """
        try:
            # Normalize tensor
            normalized = tensor / torch.norm(tensor)
            # Convert to complex numpy array
            state_vector = normalized.numpy().astype(np.complex128)
            self.current_state = Statevector(state_vector)
            
        except Exception as e:
            self.logger.error(f"Tensor to state conversion failed: {str(e)}")
            raise
    
    def get_quantum_features(self) -> Dict[str, Any]:
        """
        Extract quantum features from current state.
        
        Returns:
            Dictionary of quantum features
        """
        if self.current_state is None:
            raise ValueError("No current state available. Run simulation first.")
        
        try:
            features = {
                'entanglement_map': self.get_entanglement_map(),
                'state_purity': self._calculate_purity(),
                'state_entropy': self._calculate_entropy(),
                'superposition_measure': self._calculate_superposition(),
                'phase_distribution': self._calculate_phase_distribution()
            }
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise
    
    def _apply_operation(self, circuit: QuantumCircuit, operation: Dict[str, Any]) -> None:
        """Apply quantum operation to circuit."""
        op_type = operation['type']
        params = operation.get('params', {})
        
        if op_type == 'hadamard':
            circuit.h(params['qubit'])
        elif op_type == 'cnot':
            circuit.cx(params['control'], params['target'])
        elif op_type == 'phase':
            circuit.p(params['phi'], params['qubit'])
        elif op_type == 'custom_unitary':
            unitary = UnitaryGate(params['matrix'])
            circuit.append(unitary, [params['qubit']])
    
    def _calculate_entanglement(self, density_matrix: np.ndarray, qubit1: int, qubit2: int) -> float:
        """Calculate entanglement between two qubits using partial trace."""
        # Simplified entanglement measure using linear entropy of reduced density matrix
        reduced_dm = self._partial_trace(density_matrix, [qubit1, qubit2])
        linear_entropy = 1 - np.trace(np.matmul(reduced_dm, reduced_dm))
        return linear_entropy
    
    def _partial_trace(self, density_matrix: np.ndarray, qubits: List[int]) -> np.ndarray:
        """Calculate partial trace over specified qubits."""
        # Simplified implementation for 2-qubit reduced density matrix
        dims = [2] * self.num_qubits
        reduced_dm = np.zeros((4, 4), dtype=np.complex128)
        
        # Perform partial trace (simplified version)
        for i in range(2**self.num_qubits):
            for j in range(2**self.num_qubits):
                if all(i >> k & 1 == j >> k & 1 for k in range(self.num_qubits) if k not in qubits):
                    reduced_dm[i & 3, j & 3] += density_matrix[i, j]
        
        return reduced_dm
    
    def _calculate_purity(self) -> float:
        """Calculate state purity."""
        density_matrix = self.get_density_matrix()
        return float(np.real(np.trace(np.matmul(density_matrix, density_matrix))))
    
    def _calculate_entropy(self) -> float:
        """Calculate von Neumann entropy."""
        density_matrix = self.get_density_matrix()
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        # Remove small imaginary components and zero eigenvalues
        eigenvalues = np.real(eigenvalues[eigenvalues > 1e-10])
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))
    
    def _calculate_superposition(self) -> float:
        """Calculate degree of superposition."""
        if self.current_state is None:
            return 0.0
        state_vector = self.current_state.data
        # Use normalized L1 norm as superposition measure
        return float(np.sum(np.abs(state_vector)) / np.sqrt(len(state_vector)))
    
    def _calculate_phase_distribution(self) -> Dict[str, float]:
        """Calculate phase distribution of state."""
        if self.current_state is None:
            return {}
        state_vector = self.current_state.data
        phases = np.angle(state_vector)
        return {
            'mean_phase': float(np.mean(phases)),
            'phase_std': float(np.std(phases)),
            'phase_entropy': float(-np.sum(phases**2 * np.log2(phases**2 + 1e-10)))
        }
