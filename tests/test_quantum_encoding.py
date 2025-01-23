import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import torch
import asyncio

from quantum_simulation import QuantumSimulator
from enhanced_quantum_agent import EnhancedQuantumAgent
from config import SystemConfig, PathwayConfig

@pytest.fixture
def quantum_simulator():
    """Create a quantum simulator instance for testing."""
    return QuantumSimulator(num_qubits=4)

@pytest.fixture
def quantum_agent():
    """Create a quantum agent instance for testing."""
    config = SystemConfig(
        quantum_dim=16,
        consciousness_dim=32,
        unified_dim=64,
        active_dimensions=['quantum', 'consciousness'],
        processing_mode='QUANTUM'
    )
    pathway_config = PathwayConfig()
    return EnhancedQuantumAgent(config=config, pathway_config=pathway_config)

def test_quantum_circuit_creation(quantum_simulator):
    """Test creation of quantum circuits with basic operations."""
    # Test creating a circuit with Hadamard gates
    operations = [
        {'type': 'hadamard', 'params': {'qubit': 0}},
        {'type': 'hadamard', 'params': {'qubit': 1}}
    ]
    circuit = quantum_simulator.create_circuit(operations)
    
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 4
    
    # Simulate and verify superposition
    state = quantum_simulator.simulate_circuit(circuit)
    probabilities = np.abs(state.data) ** 2
    
    # First two qubits should be in superposition
    expected_prob = 0.25  # 1/4 for each basis state
    assert np.allclose(probabilities[:4], expected_prob, atol=1e-7)

def test_entanglement_creation(quantum_simulator):
    """Test creation and verification of entangled states."""
    # Create Bell state
    operations = [
        {'type': 'hadamard', 'params': {'qubit': 0}},
        {'type': 'cnot', 'params': {'control': 0, 'target': 1}}
    ]
    circuit = quantum_simulator.create_circuit(operations)
    state = quantum_simulator.simulate_circuit(circuit)
    
    # Get entanglement map
    entanglement_map = quantum_simulator.get_entanglement_map()
    
    # Check entanglement between qubits 0 and 1
    assert entanglement_map['q0-q1'] > 0.9  # High entanglement
    # Other pairs should have low entanglement
    assert all(entanglement_map[k] < 0.1 for k in entanglement_map if k != 'q0-q1')

def test_quantum_features(quantum_simulator):
    """Test extraction of quantum features."""
    # Create superposition state
    circuit = quantum_simulator.create_circuit()  # Default creates superposition
    quantum_simulator.simulate_circuit(circuit)
    
    features = quantum_simulator.get_quantum_features()
    
    assert 'entanglement_map' in features
    assert 'state_purity' in features
    assert 'state_entropy' in features
    assert 'superposition_measure' in features
    assert 'phase_distribution' in features
    
    # Verify superposition measure
    assert 0.0 <= features['superposition_measure'] <= 1.0
    
    # Verify entropy is non-negative
    assert features['state_entropy'] >= 0.0

@pytest.mark.asyncio
async def test_text_encoding(quantum_agent):
    """Test encoding and processing of text information."""
    test_input = "quantum consciousness"
    
    # Process input through agent
    response = await quantum_agent.forward(test_input)
    
    assert response.confidence > 0.5  # Should have reasonable confidence
    assert response.processing_time > 0  # Should take some time to process
    assert isinstance(response.text, str)  # Should return text response
    
    # Verify quantum coherence in metadata
    assert 'quantum_coherence' in response.metadata
    assert 0.0 <= response.metadata['quantum_coherence'] <= 1.0

def test_density_matrix_properties(quantum_simulator):
    """Test properties of the quantum state density matrix."""
    # Create a test circuit
    circuit = quantum_simulator.create_circuit()
    quantum_simulator.simulate_circuit(circuit)
    
    # Get density matrix
    density_matrix = quantum_simulator.get_density_matrix()
    
    # Test properties of density matrix
    # 1. Hermitian
    assert np.allclose(density_matrix, density_matrix.conj().T)
    
    # 2. Trace = 1
    assert np.allclose(np.trace(density_matrix), 1.0)
    
    # 3. Positive semidefinite (all eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    assert np.all(eigenvalues >= -1e-10)  # Allow for numerical error

def test_state_conversion(quantum_simulator):
    """Test conversion between quantum states and classical tensors."""
    # Create quantum state
    circuit = quantum_simulator.create_circuit()
    state = quantum_simulator.simulate_circuit(circuit)
    
    # Convert to tensor
    tensor = quantum_simulator.state_to_tensor()
    assert isinstance(tensor, torch.Tensor)
    
    # Verify tensor properties
    assert torch.is_floating_point(tensor)
    assert torch.all(tensor >= 0)  # Amplitudes should be non-negative
    assert np.allclose(torch.sum(tensor ** 2).item(), 1.0)  # Normalized 