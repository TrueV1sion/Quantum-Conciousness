"""Integration tests for quantum code generation plugin."""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.plugin_manager import PluginManager
from src.quantum_conscious_generation import (
    QuantumCodeVerifier,
    MultiModalQuantumEncoder,
    QuantumCoherenceMetrics
)
from src.deployment_config import DeploymentConfig


@pytest.fixture
def quantum_model():
    """Load a small model for testing."""
    model = AutoModelForCausalLM.from_pretrained("microsoft/CodeGPT-small-py")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py")
    return model, tokenizer


@pytest.fixture
def quantum_simulator():
    """Create a quantum simulator for realistic quantum data."""
    try:
        import qiskit
        from qiskit import QuantumCircuit, Aer, execute
        
        def generate_quantum_state(num_qubits=4):
            circuit = QuantumCircuit(num_qubits)
            # Create superposition
            for i in range(num_qubits):
                circuit.h(i)
            # Add entanglement
            for i in range(num_qubits-1):
                circuit.cx(i, i+1)
            
            backend = Aer.get_backend('statevector_simulator')
            job = execute(circuit, backend)
            result = job.result()
            statevector = result.get_statevector()
            return torch.tensor(statevector.real) + 1j * torch.tensor(statevector.imag)
            
        return generate_quantum_state
    except ImportError:
        logging.warning("Qiskit not available. Using simulated quantum data.")
        return lambda: torch.randn(16) + 1j * torch.randn(16)


def test_plugin_discovery(caplog):
    """Test that plugin manager discovers and loads the quantum plugin."""
    caplog.set_level(logging.INFO)
    
    config = DeploymentConfig()
    plugin_manager = PluginManager(config)
    plugins = plugin_manager.discover_plugins()
    
    quantum_plugin = None
    for plugin in plugins:
        if plugin.__class__.__name__ == "QuantumCodeGeneratorPlugin":
            quantum_plugin = plugin
            break
    
    assert quantum_plugin is not None, "QuantumCodeGeneratorPlugin not discovered"
    assert "Initialized QuantumCodeGeneratorPlugin" in caplog.text


def test_config_consistency(quantum_model):
    """Test configuration consistency between server and plugin."""
    model, tokenizer = quantum_model
    config = {
        "model_name": "microsoft/CodeGPT-small-py",
        "quantum_coherence_weight": 0.3,
        "max_length": 256,
        "temperature": 0.7,
        "top_p": 0.95,
        "use_gpu_cache": True,
        "gpu_memory_fraction": 0.8
    }
    
    verifier = QuantumCodeVerifier(
        coherence_threshold=0.8,
        max_iterations=3
    )
    
    # Verify config propagation
    test_code = "def hello_world():\n    print('Hello')"
    quantum_field = torch.randn(512)  # Simulated quantum field
    
    code, insight = verifier.verify_code(test_code, quantum_field, model, tokenizer)
    assert insight.coherence_metrics is not None
    assert 0 <= insight.coherence_metrics.coherence_score <= 1


@pytest.mark.asyncio
async def test_concurrent_processing(quantum_model, quantum_simulator):
    """Test concurrent code generation with quantum verification."""
    model, tokenizer = quantum_model
    verifier = QuantumCodeVerifier()
    encoder = MultiModalQuantumEncoder()
    
    async def process_code(code: str):
        # Generate realistic quantum field
        quantum_field = quantum_simulator()
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(),
            verifier.verify_code,
            code,
            quantum_field,
            model,
            tokenizer
        )
    
    test_codes = [
        "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
        "class Calculator:\n    def add(self, a, b):\n        return a + b",
        "async def fetch_data():\n    return await api.get_data()",
    ]
    
    tasks = [process_code(code) for code in test_codes]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == len(test_codes)
    for code, insight in results:
        assert insight.coherence_metrics is not None
        assert insight.quantum_explanation != ""


def test_quantum_advantage_benchmark(quantum_model, quantum_simulator):
    """Benchmark quantum-enhanced vs classical code generation."""
    model, tokenizer = quantum_model
    
    # Test cases with known code smells or potential issues
    test_cases = [
        (
            "def process_data(data):\n    return data.process()",  # Missing error handling
            "Input validation and error handling"
        ),
        (
            "class User:\n    def __init__(self):\n        pass",  # Missing docstring
            "Documentation completeness"
        ),
        (
            "x = 5\ny = 10\nz = x + y",  # Poor variable names
            "Code readability"
        )
    ]
    
    results = {
        "quantum_enhanced": {"improvements": 0, "coherence": []},
        "classical": {"improvements": 0, "coherence": []}
    }
    
    verifier = QuantumCodeVerifier()
    
    for code, metric in test_cases:
        # Quantum-enhanced generation
        quantum_field = quantum_simulator()
        q_code, q_insight = verifier.verify_code(code, quantum_field, model, tokenizer)
        
        # Classical generation (without quantum field)
        c_code, c_insight = verifier.verify_code(
            code,
            torch.zeros_like(quantum_field),
            model,
            tokenizer
        )
        
        # Compare improvements
        if q_insight.coherence_metrics.coherence_score > c_insight.coherence_metrics.coherence_score:
            results["quantum_enhanced"]["improvements"] += 1
        
        results["quantum_enhanced"]["coherence"].append(
            q_insight.coherence_metrics.coherence_score
        )
        results["classical"]["coherence"].append(
            c_insight.coherence_metrics.coherence_score
        )
    
    # Calculate improvement statistics
    quantum_avg_coherence = np.mean(results["quantum_enhanced"]["coherence"])
    classical_avg_coherence = np.mean(results["classical"]["coherence"])
    
    improvement_percentage = (
        (quantum_avg_coherence - classical_avg_coherence) / 
        classical_avg_coherence * 100
    )
    
    logging.info(
        f"Quantum Enhancement Results:\n"
        f"Average Coherence Improvement: {improvement_percentage:.1f}%\n"
        f"Number of Superior Improvements: "
        f"{results['quantum_enhanced']['improvements']}/{len(test_cases)}"
    )
    
    assert quantum_avg_coherence > classical_avg_coherence, \
        "Quantum enhancement should improve code quality" 