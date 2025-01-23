import pytest
import torch
from consciousness_attention import AttentionMapper, ConsciousnessGuidedAttention


@pytest.mark.asyncio
async def test_attention_mapper(test_config, sample_attention_patterns):
    """Test attention mapping functionality."""
    mapper = AttentionMapper(test_config.bridge)
    
    # Create test patterns
    patterns = {
        'correlation': torch.rand(1, device=sample_attention_patterns.device),
        'frequency_patterns': torch.rand_like(sample_attention_patterns),
        'phase_patterns': torch.rand_like(sample_attention_patterns)
    }
    
    # Apply consciousness patterns
    modified_attention = mapper.apply_consciousness_patterns(
        sample_attention_patterns,
        patterns
    )
    
    # Verify output properties
    assert modified_attention.shape == sample_attention_patterns.shape
    assert torch.allclose(modified_attention.sum(dim=-1), torch.ones_like(modified_attention.sum(dim=-1)))
    assert (modified_attention >= 0).all()  # Attention weights should be non-negative


@pytest.mark.asyncio
async def test_consciousness_guided_attention(test_config, sample_llm_states, sample_consciousness_field):
    """Test consciousness-guided attention mechanism."""
    attention = ConsciousnessGuidedAttention(test_config.bridge)
    
    # Forward pass
    enhanced_attention, patterns = await attention.forward(
        sample_llm_states,
        sample_consciousness_field
    )
    
    # Verify output shapes
    assert enhanced_attention.shape == sample_llm_states.shape
    assert isinstance(patterns, dict)
    assert all(key in patterns for key in ['correlation', 'frequency_patterns', 'phase_patterns'])
    
    # Verify attention properties
    attention_weights = torch.softmax(enhanced_attention, dim=-1)
    assert torch.allclose(attention_weights.sum(dim=-1), torch.ones_like(attention_weights.sum(dim=-1)))


def test_attention_initialization(test_config):
    """Test initialization of attention parameters."""
    attention = ConsciousnessGuidedAttention(test_config.bridge)
    
    # Test parameter initialization
    attention._init_parameters()
    
    # Verify parameter shapes
    assert attention.consciousness_gate.shape == (test_config.bridge.hidden_dim,)
    assert attention.attention_proj.weight.shape == (
        test_config.bridge.hidden_dim,
        test_config.bridge.hidden_dim
    )
    
    # Verify initialization values
    gate_mean = attention.consciousness_gate.mean().item()
    assert 0.4 <= gate_mean <= 0.6  # Should be initialized around 0.5


@pytest.mark.asyncio
async def test_attention_error_handling(test_config, sample_llm_states):
    """Test error handling in attention mechanism."""
    attention = ConsciousnessGuidedAttention(test_config.bridge)
    
    # Test with mismatched dimensions
    invalid_consciousness_field = torch.randn(
        1,
        test_config.bridge.consciousness_dim + 1  # Invalid dimension
    )
    
    with pytest.raises(Exception):
        await attention.forward(sample_llm_states, invalid_consciousness_field)


@pytest.mark.asyncio
async def test_attention_device_handling(test_config, device):
    """Test device handling in attention mechanism."""
    attention = ConsciousnessGuidedAttention(test_config.bridge)
    attention = attention.to(device)
    
    # Create test inputs on specific device
    llm_states = torch.randn(
        1,
        16,
        test_config.bridge.hidden_dim,
        device=device
    )
    consciousness_field = torch.randn(
        1,
        test_config.bridge.consciousness_dim,
        device=device
    )
    
    # Forward pass
    enhanced_attention, _ = await attention.forward(llm_states, consciousness_field)
    
    # Verify device placement
    assert enhanced_attention.device == device
    assert attention.consciousness_gate.device == device 