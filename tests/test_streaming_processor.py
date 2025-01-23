import pytest
import torch
import asyncio
from typing import AsyncIterator
from streaming_processor import StreamingConsciousnessProcessor


async def token_generator(tokens: list[str]) -> AsyncIterator[str]:
    """Helper function to generate tokens asynchronously."""
    for token in tokens:
        yield token
        await asyncio.sleep(0.01)  # Simulate processing time


@pytest.mark.asyncio
async def test_streaming_processor_initialization(test_config):
    """Test streaming processor initialization."""
    processor = StreamingConsciousnessProcessor(test_config.bridge)
    
    # Verify initialization
    assert processor.buffer.maxlen == test_config.bridge.buffer_size
    assert processor.current_state is None
    assert len(processor.state_history) == 0


@pytest.mark.asyncio
async def test_stream_processing(test_config, sample_consciousness_field):
    """Test processing of token stream."""
    processor = StreamingConsciousnessProcessor(test_config.bridge)
    
    # Create test tokens
    test_tokens = ["token1", "token2", "token3", "token4"]
    token_stream = token_generator(test_tokens)
    
    # Process stream
    enhanced_tokens = []
    async for token in processor.process_stream(
        token_stream,
        sample_consciousness_field,
        batch_size=2
    ):
        enhanced_tokens.append(token)
    
    # Verify processing
    assert len(enhanced_tokens) == len(test_tokens)
    assert all(isinstance(token, str) for token in enhanced_tokens)


@pytest.mark.asyncio
async def test_batch_processing(test_config, sample_consciousness_field):
    """Test batch processing functionality."""
    processor = StreamingConsciousnessProcessor(test_config.bridge)
    
    # Create large test batch
    test_tokens = [f"token{i}" for i in range(10)]
    token_stream = token_generator(test_tokens)
    
    # Process with different batch sizes
    batch_size = 3
    enhanced_tokens = []
    async for token in processor.process_stream(
        token_stream,
        sample_consciousness_field,
        batch_size=batch_size
    ):
        enhanced_tokens.append(token)
    
    # Verify batch processing
    assert len(enhanced_tokens) == len(test_tokens)
    assert len(processor.buffer) <= test_config.bridge.buffer_size


@pytest.mark.asyncio
async def test_state_history_management(test_config, sample_consciousness_field):
    """Test state history management."""
    processor = StreamingConsciousnessProcessor(test_config.bridge)
    
    # Add states to history
    for i in range(test_config.bridge.history_size + 5):
        state = torch.randn(test_config.bridge.hidden_dim)
        processor._update_state_history(i, state)
    
    # Verify history size management
    assert len(processor.state_history) <= test_config.bridge.history_size
    assert min(processor.state_history.keys()) >= 5  # Oldest states should be removed


@pytest.mark.asyncio
async def test_error_handling(test_config, sample_consciousness_field):
    """Test error handling in streaming processor."""
    processor = StreamingConsciousnessProcessor(test_config.bridge)
    
    # Test with invalid consciousness field
    invalid_field = torch.randn(
        1,
        test_config.bridge.consciousness_dim + 1  # Invalid dimension
    )
    
    with pytest.raises(Exception):
        async for _ in processor.process_stream(
            token_generator(["test"]),
            invalid_field
        ):
            pass


@pytest.mark.asyncio
async def test_device_handling(test_config, device):
    """Test device handling in streaming processor."""
    processor = StreamingConsciousnessProcessor(test_config.bridge)
    
    # Create test inputs on specific device
    consciousness_field = torch.randn(
        1,
        test_config.bridge.consciousness_dim,
        device=device
    )
    
    # Process stream
    test_tokens = ["test1", "test2"]
    enhanced_tokens = []
    async for token in processor.process_stream(
        token_generator(test_tokens),
        consciousness_field
    ):
        enhanced_tokens.append(token)
    
    # Verify processing completed
    assert len(enhanced_tokens) == len(test_tokens)
    
    # Verify internal tensors are on correct device
    if processor.current_state is not None:
        assert processor.current_state.device == device 