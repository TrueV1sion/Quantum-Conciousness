import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from collections import OrderedDict
import threading
import time

from quantum_processor import QuantumProcessor


class QuantumMemoryManager:
    """Quantum Memory Management Plugin for efficient state handling."""
    
    def __init__(
        self,
        quantum_processor: QuantumProcessor,
        max_memory_size: int = 1000,
        cleanup_threshold: float = 0.8
    ):
        self.quantum_processor = quantum_processor
        self.max_memory_size = max_memory_size
        self.cleanup_threshold = cleanup_threshold
        
        # Initialize memory structures
        self.quantum_memory: OrderedDict[str, Dict] = OrderedDict()
        self.classical_cache: Dict[str, torch.Tensor] = {}
        self.memory_lock = threading.Lock()
        
        # Start background cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._background_cleanup,
            daemon=True
        )
        self.cleanup_thread.start()
    
    def store_quantum_state(
        self,
        state: Union[QuantumCircuit, Statevector, torch.Tensor],
        identifier: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Store quantum state in memory with metadata."""
        with self.memory_lock:
            if len(self.quantum_memory) >= self.max_memory_size:
                self._cleanup_memory()
            
            # Convert state to standard format
            if isinstance(state, QuantumCircuit):
                state_vector = Statevector.from_instruction(state)
                tensor_state = torch.tensor(state_vector.data)
            elif isinstance(state, Statevector):
                tensor_state = torch.tensor(state.data)
            else:  # torch.Tensor
                tensor_state = state
            
            # Store state with metadata
            self.quantum_memory[identifier] = {
                'state': tensor_state,
                'metadata': metadata or {},
                'timestamp': time.time(),
                'access_count': 0
            }
            
            return True
    
    def retrieve_quantum_state(
        self,
        identifier: str,
        as_type: str = 'tensor'
    ) -> Optional[Union[torch.Tensor, QuantumCircuit, Statevector]]:
        """Retrieve quantum state from memory."""
        with self.memory_lock:
            if identifier not in self.quantum_memory:
                return None
            
            # Update access statistics
            self.quantum_memory[identifier]['access_count'] += 1
            self.quantum_memory[identifier]['last_access'] = time.time()
            
            state_data = self.quantum_memory[identifier]['state']
            
            # Convert to requested format
            if as_type == 'tensor':
                return state_data
            elif as_type == 'statevector':
                return Statevector(state_data.numpy())
            elif as_type == 'circuit':
                return self._state_to_circuit(state_data)
            else:
                raise ValueError(f"Unsupported return type: {as_type}")
    
    def _state_to_circuit(self, state: torch.Tensor) -> QuantumCircuit:
        """Convert state vector to quantum circuit."""
        n_qubits = int(np.log2(len(state)))
        qr = QuantumRegister(n_qubits)
        circuit = QuantumCircuit(qr)
        
        # Initialize circuit to represent the state
        circuit.initialize(state.numpy(), qr)
        return circuit
    
    def _cleanup_memory(self) -> None:
        """Clean up memory based on access patterns and age."""
        if len(self.quantum_memory) < self.max_memory_size * self.cleanup_threshold:
            return
        
        # Calculate scores for each state
        current_time = time.time()
        scores = {}
        for id_, data in self.quantum_memory.items():
            age = current_time - data['timestamp']
            access_rate = data['access_count'] / max(age, 1)
            scores[id_] = access_rate
        
        # Remove lowest scoring states
        target_size = int(self.max_memory_size * 0.7)  # Remove 30% of memory
        to_remove = sorted(
            scores.keys(),
            key=lambda x: scores[x]
        )[:len(scores) - target_size]
        
        for id_ in to_remove:
            del self.quantum_memory[id_]
    
    def _background_cleanup(self) -> None:
        """Background thread for periodic memory cleanup."""
        while True:
            time.sleep(60)  # Check every minute
            with self.memory_lock:
                self._cleanup_memory()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        with self.memory_lock:
            total_states = len(self.quantum_memory)
            total_size = sum(
                state['state'].element_size() * state['state'].nelement()
                for state in self.quantum_memory.values()
            )
            
            # Calculate average access patterns
            access_counts = [
                state['access_count'] for state in self.quantum_memory.values()
            ]
            avg_access = np.mean(access_counts) if access_counts else 0
            
            return {
                'total_states': total_states,
                'memory_usage_bytes': total_size,
                'memory_utilization': total_states / self.max_memory_size,
                'average_access_count': avg_access
            }
    
    def optimize_memory_layout(self) -> None:
        """Optimize memory layout based on access patterns."""
        with self.memory_lock:
            # Group frequently accessed states together
            sorted_states = sorted(
                self.quantum_memory.items(),
                key=lambda x: x[1]['access_count'],
                reverse=True
            )
            
            # Rebuild memory with optimized layout
            self.quantum_memory.clear()
            for id_, data in sorted_states:
                self.quantum_memory[id_] = data
    
    def cache_classical_data(
        self,
        data: torch.Tensor,
        identifier: str,
        max_cache_size: int = 100
    ) -> bool:
        """Cache classical data for quantum operations."""
        if len(self.classical_cache) >= max_cache_size:
            # Remove oldest items
            oldest = sorted(
                self.classical_cache.keys(),
                key=lambda x: self.quantum_memory.get(x, {}).get('timestamp', 0)
            )[:len(self.classical_cache) - max_cache_size + 1]
            
            for key in oldest:
                del self.classical_cache[key]
        
        self.classical_cache[identifier] = data
        return True
    
    def get_cached_data(
        self,
        identifier: str
    ) -> Optional[torch.Tensor]:
        """Retrieve cached classical data."""
        return self.classical_cache.get(identifier)
    
    def clear_cache(self) -> None:
        """Clear all cached classical data."""
        self.classical_cache.clear() 