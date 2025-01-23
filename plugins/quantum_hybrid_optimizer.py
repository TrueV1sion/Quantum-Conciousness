import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from qiskit.algorithms.optimizers import SPSA, ADAM
from qiskit.circuit import ParameterVector

from quantum_processor import QuantumProcessor


class QuantumHybridOptimizer:
    """Quantum-Classical Hybrid Optimization Plugin."""
    
    def __init__(
        self,
        quantum_processor: QuantumProcessor,
        classical_optimizer: str = 'adam',
        learning_rate: float = 0.01,
        max_iterations: int = 100
    ):
        self.quantum_processor = quantum_processor
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
        # Initialize classical optimizer
        if classical_optimizer.lower() == 'adam':
            self.optimizer = ADAM(
                maxiter=max_iterations,
                lr=learning_rate
            )
        elif classical_optimizer.lower() == 'spsa':
            self.optimizer = SPSA(
                maxiter=max_iterations,
                learning_rate=learning_rate
            )
        else:
            raise ValueError(f"Unsupported optimizer: {classical_optimizer}")
        
        # Initialize quantum parameters
        self.params = ParameterVector('θ', length=quantum_processor.n_qubits * 3)
        
    def optimize_quantum_circuit(
        self,
        initial_state: torch.Tensor,
        target_state: torch.Tensor,
        optimization_type: str = 'fidelity'
    ) -> Tuple[Dict[str, float], float]:
        """Optimize quantum circuit parameters to achieve target state."""
        
        def cost_function(params: np.ndarray) -> float:
            # Create parameterized circuit
            circuit = self.quantum_processor.create_variational_circuit_enhanced(
                {f'param_{i}': p for i, p in enumerate(params)},
                entanglement_pattern='full',
                add_zz_entanglement=True
            )
            
            # Get output state
            output_state = self.quantum_processor.get_statevector(circuit)
            output_tensor = torch.tensor(output_state.data)
            
            if optimization_type == 'fidelity':
                # Calculate quantum fidelity
                fidelity = torch.abs(torch.sum(torch.conj(target_state) * output_tensor)) ** 2
                return -float(fidelity.real)  # Negative for minimization
            else:
                # Calculate L2 distance
                distance = torch.norm(target_state - output_tensor)
                return float(distance)
        
        # Run optimization
        initial_params = np.random.randn(len(self.params))
        optimal_params, optimal_value = self.optimizer.optimize(
            num_vars=len(self.params),
            objective_function=cost_function,
            initial_point=initial_params
        )
        
        # Create result dictionary
        result = {
            f'param_{i}': float(p) for i, p in enumerate(optimal_params)
        }
        result['final_cost'] = float(optimal_value)
        result['iterations'] = self.optimizer.maxiter
        
        return result, -optimal_value if optimization_type == 'fidelity' else optimal_value
    
    def optimize_quantum_encoding(
        self,
        data: torch.Tensor,
        encoding_dim: int,
        batch_size: int = 32
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Optimize quantum encoding for classical data."""
        
        def encoding_cost(params: np.ndarray) -> float:
            # Create encoding circuit
            circuit = self.quantum_processor.encode_classical_data(
                data,
                add_entanglement=True
            )
            
            # Add variational layers with parameters
            circuit = self.quantum_processor.create_variational_circuit_enhanced(
                {f'param_{i}': p for i, p in enumerate(params)},
                num_layers=2
            )
            
            # Get encoded state
            encoded_state = self.quantum_processor.get_statevector(circuit)
            encoded_tensor = torch.tensor(encoded_state.data)
            
            # Calculate encoding quality (reconstruction loss)
            projected = self.quantum_processor.quantum_feature_projection(
                encoded_tensor,
                projection_type='hybrid'
            )
            loss = torch.nn.functional.mse_loss(projected[:encoding_dim], data)
            return float(loss)
        
        # Run optimization
        initial_params = np.random.randn(len(self.params))
        optimal_params, optimal_value = self.optimizer.optimize(
            num_vars=len(self.params),
            objective_function=encoding_cost,
            initial_point=initial_params
        )
        
        # Get final encoded state
        final_circuit = self.quantum_processor.encode_classical_data(
            data,
            add_entanglement=True
        )
        final_circuit = self.quantum_processor.create_variational_circuit_enhanced(
            {f'param_{i}': p for i, p in enumerate(optimal_params)},
            num_layers=2
        )
        final_state = self.quantum_processor.get_statevector(final_circuit)
        
        # Create result dictionary
        result = {
            f'param_{i}': float(p) for i, p in enumerate(optimal_params)
        }
        result['final_loss'] = float(optimal_value)
        result['encoding_dim'] = encoding_dim
        
        return torch.tensor(final_state.data), result 