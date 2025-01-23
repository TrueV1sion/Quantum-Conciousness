import asyncio
import torch
import os
from src.plugins.plugin_manager import QuantumPluginManager
from quantum_consciousness_core import QuantumConsciousnessCore

async def demonstrate_plugins():
    """Demonstrate quantum consciousness plugins."""
    print("\nQuantum Consciousness Plugin System Demo")
    print("======================================")
    
    # Initialize plugin manager
    plugins_dir = os.path.join(os.path.dirname(__file__), "../src/plugins")
    plugin_manager = QuantumPluginManager(plugins_dir)
    
    # Discover and load plugins
    plugin_manager.discover_plugins()
    
    # Initialize quantum core
    quantum_core = QuantumConsciousnessCore(
        hidden_dim=1024,
        num_qubits=32
    )
    
    # Create test quantum state
    test_input = torch.randn(1, 512, 1024)  # [batch, seq_len, hidden_dim]
    quantum_state = quantum_core.create_quantum_state(test_input)
    
    # Create consciousness field
    consciousness_field = torch.randn(1, 1024)  # [batch, consciousness_dim]
    
    # Initialize plugins with configuration
    config = {
        "QuantumResonance": {
            "resonance_threshold": 0.6,
            "coherence_threshold": 0.8
        }
    }
    plugin_manager.initialize_plugins(config)
    
    # List available plugins
    print("\nAvailable Plugins:")
    plugin_info = plugin_manager.get_plugin_info()
    for name, info in plugin_info.items():
        print(f"- {name} v{info['version']} ({info['type']})")
    
    # Execute quantum resonance plugin
    print("\nExecuting QuantumResonance plugin...")
    try:
        results = plugin_manager.execute_plugin(
            "QuantumResonance",
            quantum_state.amplitude,
            consciousness_field
        )
        
        # Print results
        print("\nResonance Analysis Results:")
        print(f"- Resonance Score: {results['resonance_score']:.3f}")
        print(f"- Phase Coherence: {results['phase_coherence']:.3f}")
        print(f"- Quantum Coherence: {results['quantum_coherence']:.3f}")
        print(f"- Is Resonant: {results['is_resonant']}")
        print(f"- Is Coherent: {results['is_coherent']}")
        
        if 'consciousness_interaction' in results:
            interaction = results['consciousness_interaction']
            print("\nConsciousness Interaction:")
            print(f"- Interaction Strength: {interaction['interaction_strength']:.3f}")
            print(f"- Phase Alignment: {interaction['phase_alignment']:.3f}")
            print(f"- Is Aligned: {interaction['is_aligned']}")
        
        # Visualize resonance patterns
        print("\nResonance Pattern Visualization:")
        patterns = results['resonance_patterns'][0]
        visualize_patterns(patterns)
        
    except Exception as e:
        print(f"Error executing plugin: {str(e)}")

def visualize_patterns(patterns: torch.Tensor) -> None:
    """Create a simple visualization of resonance patterns."""
    # Normalize patterns for visualization
    normalized = (patterns - patterns.min()) / (patterns.max() - patterns.min())
    
    # Create visualization
    for val in normalized:
        line = ''
        if val < 0.2:
            line += '.'
        elif val < 0.4:
            line += '▄'
        elif val < 0.6:
            line += '█'
        elif val < 0.8:
            line += '▀'
        else:
            line += '◉'
        print(line)

if __name__ == "__main__":
    asyncio.run(demonstrate_plugins()) 