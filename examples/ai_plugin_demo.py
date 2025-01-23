import asyncio
import os
import torch
from src.plugins.plugin_manager import QuantumPluginManager
from quantum_consciousness_core import QuantumConsciousnessCore


async def demonstrate_ai_plugins():
    """Demonstrate AI model plugins integration."""
    print("\nAI Model Plugin System Demo")
    print("==========================\n")
    
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
        "AI_BERT": {
            "model_name": "bert-base-uncased",
            "model_config": {
                "output_hidden_states": True
            },
            "max_length": 512
        },
        "ModelAnalyzer": {
            "analysis_config": {
                "track_time": True,
                "track_memory": True,
                "track_gradients": False,
                "track_activations": True
            }
        }
    }
    plugin_manager.initialize_plugins(config)
    
    # List available plugins
    print("\nAvailable Plugins:")
    plugin_info = plugin_manager.get_plugin_info()
    for name, info in plugin_info.items():
        print(f"- {name} v{info['version']} ({info['type']})")
    
    # Execute BERT model plugin
    print("\nExecuting BERT model plugin...")
    try:
        results = plugin_manager.execute_plugin(
            "AI_BERT",
            quantum_state,
            consciousness_field
        )
        
        # Print results
        print("\nBERT Model Results:")
        print(f"- Model Type: {results['model_type']}")
        print(f"- Hidden Size: {results['hidden_size']}")
        print(f"- Vocab Size: {results['vocab_size']}")
        
        # Analyze model performance
        print("\nAnalyzing model performance...")
        analysis = plugin_manager.execute_plugin(
            "ModelAnalyzer",
            results['output'],
            consciousness_field
        )
        
        print("\nPerformance Metrics:")
        print(f"- Execution Time: {analysis['execution_time']:.3f}s")
        print(f"- Memory Usage: {analysis['memory_usage'] / 1024**2:.2f} MB")
        print(f"- Peak Memory: {analysis['peak_memory'] / 1024**2:.2f} MB")
        print("\nTensor Properties:")
        print(f"- Mean: {analysis['mean']:.3f}")
        print(f"- Std: {analysis['std']:.3f}")
        print(f"- Sparsity: {analysis['sparsity']:.3f}")
        print("\nConsciousness Interaction:")
        print(
            f"- Correlation: {analysis['consciousness_correlation']:.3f}"
        )
        print(
            f"- Cosine Similarity: {analysis['consciousness_cosine_similarity']:.3f}"
        )
        
    except Exception as e:
        print(f"Error executing plugins: {str(e)}")


if __name__ == "__main__":
    asyncio.run(demonstrate_ai_plugins()) 