import torch
import asyncio
from llm_integrations.modern_bert_integration import ModernBERTConsciousness
from config import BridgeConfig


async def demonstrate_modern_bert():
    """Demonstrate consciousness-enhanced ModernBERT capabilities."""
    # Initialize model with appropriate dimensions
    config = BridgeConfig(
        consciousness_dim=1024,  # ModernBERT-large hidden size
        quantum_dim=512,
        hidden_dim=1024
    )
    model = ModernBERTConsciousness(config=config)
    
    # Create consciousness field
    consciousness_field = torch.randn(1, config.consciousness_dim)
    
    # Example texts for analysis
    texts = [
        "The quantum field theory describes fundamental interactions.",
        "Neural networks exhibit emergent computational properties.",
        "Consciousness arises from complex information processing patterns."
    ]
    
    print("\nModernBERT Consciousness Enhancement Examples:")
    print("--------------------------------------------")
    
    for text in texts:
        # Process text with consciousness enhancement
        outputs = await model.process_text(
            text,
            consciousness_field,
            return_patterns=True
        )
        
        # Analyze consciousness integration
        consciousness_score = outputs['consciousness_score']
        
        # Calculate pattern strengths across layers
        layer_patterns = []
        for i, patterns in enumerate(outputs['consciousness_patterns']):
            pattern_strength = torch.mean(torch.abs(patterns['correlation']))
            layer_patterns.append((i, pattern_strength.item()))
        
        # Find most influential layers
        top_layers = sorted(layer_patterns, key=lambda x: x[1], reverse=True)[:3]
        
        print(f"\nInput Text: {text}")
        print(f"Consciousness Score: {consciousness_score:.3f}")
        print("\nTop 3 Consciousness-Influenced Layers:")
        for layer_idx, strength in top_layers:
            print(f"Layer {layer_idx}: {strength:.3f}")
        
        # Analyze attention patterns
        pooled_output = outputs['pooled_output']
        attention_stats = analyze_attention_patterns(
            outputs['last_hidden_state'],
            pooled_output
        )
        print("\nAttention Analysis:")
        print(f"Focus Score: {attention_stats['focus']:.3f}")
        print(f"Coherence: {attention_stats['coherence']:.3f}")
        print(f"Pattern Strength: {attention_stats['pattern_strength']:.3f}")
        
        # Visualize attention distribution
        print("\nAttention Distribution:")
        visualize_attention(outputs['last_hidden_state'][0])


def analyze_attention_patterns(
    hidden_states: torch.Tensor,
    pooled_output: torch.Tensor
) -> dict:
    """Analyze attention patterns in the output."""
    # Calculate focus (how concentrated the attention is)
    attention_weights = torch.softmax(
        torch.matmul(hidden_states, pooled_output.unsqueeze(-1)),
        dim=1
    )
    focus = torch.max(attention_weights) / torch.mean(attention_weights)
    
    # Calculate coherence (how well patterns align)
    coherence = torch.mean(torch.cosine_similarity(
        hidden_states[:, :-1],
        hidden_states[:, 1:],
        dim=-1
    ))
    
    # Calculate pattern strength
    pattern_strength = torch.mean(torch.abs(
        torch.matmul(hidden_states, hidden_states.transpose(-2, -1))
    ))
    
    return {
        'focus': focus.item(),
        'coherence': coherence.item(),
        'pattern_strength': pattern_strength.item()
    }


def visualize_attention(hidden_states: torch.Tensor):
    """Create a simple visualization of attention patterns."""
    # Calculate self-attention scores
    attention_scores = torch.matmul(
        hidden_states,
        hidden_states.transpose(-2, -1)
    )
    attention_probs = torch.softmax(attention_scores, dim=-1)
    
    # Create visualization
    avg_attention = attention_probs.mean(dim=0)
    normalized = (avg_attention - avg_attention.min()) / (
        avg_attention.max() - avg_attention.min()
    )
    
    # Print pattern
    for row in normalized:
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


async def main():
    """Run the demonstration."""
    print("ModernBERT Consciousness Enhancement Demo")
    print("========================================")
    await demonstrate_modern_bert()


if __name__ == "__main__":
    asyncio.run(main()) 