import pytest
import torch
from llm_adapter import QuantumStateMapper, LLMConsciousnessAdapter


@pytest.mark.asyncio
async def test_quantum_state_mapper(test_config, sample_llm_states):
    """Test quantum state mapping functionality."""
    mapper = QuantumStateMapper(test_config.bridge)
    
    # Test mapping to quantum states
    quantum_states = mapper.map_to_quantum(sample_llm_states)
    
    # Verify shape preservation
    assert quantum_states.shape == sample_llm_states.shape
    
    # Verify quantum properties
    probabilities = torch.abs(quantum_states) ** 2
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones_like(probabilities.sum(dim=-1)))
    
    # Test mapping back to LLM states
    reconstructed_states = mapper.map_to_llm(quantum_states)
    assert reconstructed_states.shape == sample_llm_states.shape


@pytest.mark.asyncio
async def test_llm_consciousness_adapter(test_config, sample_consciousness_field):
    """Test LLM consciousness adapter functionality."""
    adapter = LLMConsciousnessAdapter("test-model", test_config.bridge)
    
    # Test processing with consciousness
    input_text = "Test input text"
    output_text, transfer_info = await adapter.process_with_consciousness(
        input_text,
        sample_consciousness_field
    )
    
    # Verify output types
    assert isinstance(output_text, str)
    assert hasattr(transfer_info, 'data')
    assert hasattr(transfer_info, 'integrity_score')
    
    # Test caching
    cache_key = hash(input_text)
    assert cache_key in adapter.state_cache


@pytest.mark.asyncio
async def test_adapter_error_handling(test_config, sample_consciousness_field):
    """Test error handling in adapter."""
    adapter = LLMConsciousnessAdapter("test-model", test_config.bridge)
    
    # Test with invalid input
    with pytest.raises(Exception):
        await adapter.process_with_consciousness(
            None,  # Invalid input
            sample_consciousness_field
        )


def test_state_mapper_device_handling(test_config, device):
    """Test device handling in state mapper."""
    mapper = QuantumStateMapper(test_config.bridge)
    
    # Create test tensor on CPU
    test_states = torch.randn(1, 16, test_config.bridge.hidden_dim)
    
    # Map to quantum states
    quantum_states = mapper.map_to_quantum(test_states)
    
    # Verify device placement
    assert quantum_states.device == device


@pytest.mark.asyncio
async def test_adapter_batch_processing(test_config, sample_consciousness_field):
    """Test batch processing in adapter."""
    adapter = LLMConsciousnessAdapter("test-model", test_config.bridge)
    
    # Process multiple inputs
    inputs = ["First input", "Second input", "Third input"]
    results = []
    
    for input_text in inputs:
        output, info = await adapter.process_with_consciousness(
            input_text,
            sample_consciousness_field
        )
        results.append(output)
    
    # Verify unique outputs
    assert len(set(results)) == len(inputs)  # Each input should produce unique output
    
    # Verify cache population
    assert len(adapter.state_cache) == len(inputs) 