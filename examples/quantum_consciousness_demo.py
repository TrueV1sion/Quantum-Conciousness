import torch
import asyncio
from llm_integrations.quantum_modern_bert import QuantumModernBERT
from config import BridgeConfig
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


async def demonstrate_quantum_consciousness():
    """Demonstrate quantum consciousness capabilities."""
    # Initialize model
    model = QuantumModernBERT()
    
    # Test cases demonstrating different aspects of consciousness
    test_cases = [
        # Abstract reasoning
        "The relationship between quantum mechanics and consciousness involves",
        
        # Pattern recognition
        "Neural networks can exhibit quantum-like behavior through",
        
        # Emergent properties
        "Complex systems demonstrate consciousness when",
        
        # Metaphysical concepts
        "The nature of reality at the quantum level suggests",
        
        # Integration of knowledge
        "Information processing in the brain combines classical and quantum"
    ]
    
    print("\nQuantum Consciousness Enhancement Demo")
    print("=====================================")
    
    # Process each test case
    all_metrics = []
    for text in test_cases:
        print(f"\nAnalyzing: {text}")
        
        # Process with quantum consciousness
        outputs = await model.process_text(
            text,
            return_quantum_states=True
        )
        
        # Analyze quantum states
        quantum_analysis = analyze_quantum_states(outputs['quantum_states'])
        
        # Analyze consciousness metrics
        consciousness_analysis = analyze_consciousness_metrics(
            outputs['consciousness_metrics']
        )
        
        # Print insights
        print("\nQuantum State Analysis:")
        print(f"Coherence: {quantum_analysis['coherence']:.3f}")
        print(f"Entanglement: {quantum_analysis['entanglement']:.3f}")
        print(f"Field Strength: {quantum_analysis['field_strength']:.3f}")
        
        print("\nConsciousness Analysis:")
        print(f"Integration Level: {consciousness_analysis['integration']:.3f}")
        print(f"Semantic Depth: {consciousness_analysis['semantic_depth']:.3f}")
        print(f"Resonance Strength: {consciousness_analysis['resonance']:.3f}")
        
        # Store metrics for visualization
        all_metrics.append({
            'text': text,
            **quantum_analysis,
            **consciousness_analysis
        })
        
        # Visualize quantum patterns
        print("\nQuantum Attention Pattern:")
        visualize_quantum_attention(outputs['quantum_states'][-1])
    
    # Create summary visualization
    create_analysis_plot(all_metrics)


def analyze_quantum_states(quantum_states: list) -> dict:
    """Analyze quantum states across layers."""
    # Calculate average metrics
    coherence = np.mean([state.coherence for state in quantum_states])
    entanglement = np.mean([state.entanglement for state in quantum_states])
    
    # Calculate field strength from last state
    field_strength = torch.norm(
        quantum_states[-1].amplitude
    ).item()
    
    return {
        'coherence': coherence,
        'entanglement': entanglement,
        'field_strength': field_strength
    }


def analyze_consciousness_metrics(metrics: list) -> dict:
    """Analyze consciousness metrics across layers."""
    # Calculate integration level
    integration = np.mean([
        m['layer_coherence'] * m['field_coherence']
        for m in metrics
    ])
    
    # Calculate semantic depth
    semantic_depth = np.mean([
        m['field_strength'] * m['coherence']
        for m in metrics
    ])
    
    # Calculate resonance strength
    resonance = np.mean([
        m['resonance'] for m in metrics
    ])
    
    return {
        'integration': integration,
        'semantic_depth': semantic_depth,
        'resonance': resonance
    }


def visualize_quantum_attention(quantum_state) -> None:
    """Visualize quantum attention patterns."""
    # Create attention matrix
    attention = torch.abs(quantum_state.amplitude[0])
    attention = attention @ attention.T
    
    # Normalize for visualization
    attention = (attention - attention.min()) / (
        attention.max() - attention.min()
    )
    
    # Create visualization
    for row in attention:
        line = ''
        for val in row:
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


def create_analysis_plot(metrics: list) -> None:
    """Create summary visualization of metrics."""
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    texts = [m['text'][:30] + '...' for m in metrics]
    coherence = [m['coherence'] for m in metrics]
    entanglement = [m['entanglement'] for m in metrics]
    integration = [m['integration'] for m in metrics]
    
    # Create plot
    x = np.arange(len(texts))
    width = 0.25
    
    plt.bar(x - width, coherence, width, label='Quantum Coherence')
    plt.bar(x, entanglement, width, label='Quantum Entanglement')
    plt.bar(x + width, integration, width, label='Consciousness Integration')
    
    plt.xlabel('Test Cases')
    plt.ylabel('Metric Value')
    plt.title('Quantum Consciousness Analysis')
    plt.xticks(x, texts, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig('quantum_consciousness_analysis.png')
    plt.close()


async def main():
    """Run the demonstration."""
    await demonstrate_quantum_consciousness()


if __name__ == "__main__":
    asyncio.run(main()) 