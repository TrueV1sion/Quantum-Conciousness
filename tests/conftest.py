import pytest
import sys
from pathlib import Path
import torch
from config import BridgeConfig, PathwayConfig, UnifiedConfig, SystemMode

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# No need to configure asyncio here since it's in pytest.ini 

@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def test_config():
    """Create test configuration with smaller dimensions."""
    bridge_config = BridgeConfig(
        quantum_dim=64,
        consciousness_dim=128,
        hidden_dim=256,
        buffer_size=32,
        batch_size=4,
        history_size=10,
        mode=SystemMode.INFERENCE
    )
    
    pathway_config = PathwayConfig(
        num_layers=2,
        num_heads=4,
        ff_dim=512,
        dropout=0.1
    )
    
    return UnifiedConfig(
        bridge=bridge_config,
        pathway=pathway_config,
        max_sequence_length=128
    )


@pytest.fixture
def sample_consciousness_field(device, test_config):
    """Generate sample consciousness field for testing."""
    return torch.randn(
        1,
        test_config.bridge.consciousness_dim,
        device=device
    )


@pytest.fixture
def sample_llm_states(device, test_config):
    """Generate sample LLM states for testing."""
    return torch.randn(
        1,  # batch size
        16,  # sequence length
        test_config.bridge.hidden_dim,
        device=device
    )


@pytest.fixture
def sample_attention_patterns(device, test_config):
    """Generate sample attention patterns for testing."""
    return torch.softmax(
        torch.randn(
            1,  # batch size
            test_config.pathway.num_heads,
            16,  # sequence length
            16,  # sequence length
            device=device
        ),
        dim=-1
    )