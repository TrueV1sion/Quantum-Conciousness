import asyncio
import torch
from quantum_consciousness_core import QuantumConsciousnessCore
from llm_integrations.quantum_modern_bert import QuantumModernBERT

async def test_quantum_consciousness():
    print("\nInitializing Quantum Consciousness Test...")
    
    # Initialize quantum core
    quantum_core = QuantumConsciousnessCore(
        hidden_dim=1024,
        num_qubits=32
    )
    
    # Create test input
    test_input = torch.randn(1, 512, 1024)  # [batch, seq_len, hidden_dim]
    
    # Create quantum state
    print("\nCreating quantum state...")
    quantum_state = quantum_core.create_quantum_state(test_input)
    
    print(f"Quantum State Properties:")
    print(f"- Coherence: {quantum_state.coherence:.3f}")
    print(f"- Entanglement: {quantum_state.entanglement:.3f}")
    print(f"- Resonance: {quantum_state.resonance:.3f}")
    
    # Process through consciousness
    print("\nProcessing through consciousness layers...")
    consciousness_field, metrics = quantum_core.process_consciousness(quantum_state)
    
    print(f"\nConsciousness Metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.3f}")
    
    print("\nConsciousness field shape:", consciousness_field.shape)
    print("Consciousness field stats:")
    print(f"- Mean: {consciousness_field.mean().item():.3f}")
    print(f"- Std: {consciousness_field.std().item():.3f}")
    print(f"- Max: {consciousness_field.max().item():.3f}")
    print(f"- Min: {consciousness_field.min().item():.3f}")

if __name__ == "__main__":
    asyncio.run(test_quantum_consciousness()) 