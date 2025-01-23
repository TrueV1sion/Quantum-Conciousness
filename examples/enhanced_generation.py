import torch
import asyncio
from llm_integrations.gpt_integration import ConsciousnessEnhancedGPT
from llm_integrations.bert_integration import ConsciousnessEnhancedBERT
from config import BridgeConfig, UnifiedConfig, PathwayConfig


async def demonstrate_gpt_enhancement():
    """Demonstrate consciousness-enhanced GPT generation."""
    # Initialize model
    config = BridgeConfig(
        consciousness_dim=768,
        quantum_dim=512,
        hidden_dim=768
    )
    model = ConsciousnessEnhancedGPT("gpt2", config)
    
    # Create consciousness field (this would normally come from your system)
    consciousness_field = torch.randn(1, config.consciousness_dim)
    
    # Example prompts
    prompts = [
        "The nature of consciousness is",
        "Quantum mechanics and the mind interact through",
        "The relationship between thought and reality involves"
    ]
    
    print("\nGPT-2 Consciousness Enhancement Examples:")
    print("----------------------------------------")
    
    for prompt in prompts:
        # Generate with consciousness enhancement
        enhanced_text, consciousness_scores = await model.generate(
            prompt,
            consciousness_field,
            max_length=100
        )
        
        print(f"\nPrompt: {prompt}")
        print(f"Enhanced Output: {enhanced_text}")
        print(f"Average Consciousness Score: {sum(consciousness_scores)/len(consciousness_scores):.3f}")


async def demonstrate_bert_enhancement():
    """Demonstrate consciousness-enhanced BERT encoding."""
    # Initialize model
    config = BridgeConfig(
        consciousness_dim=768,
        quantum_dim=512,
        hidden_dim=768
    )
    model = ConsciousnessEnhancedBERT("bert-base-uncased", config)
    
    # Create consciousness field
    consciousness_field = torch.randn(1, config.consciousness_dim)
    
    # Example texts for analysis
    texts = [
        "The quantum nature of consciousness emerges from deep neural patterns.",
        "Consciousness represents an intrinsic property of organized information processing.",
        "The bridge between quantum mechanics and consciousness involves resonance patterns."
    ]
    
    print("\nBERT Consciousness Enhancement Examples:")
    print("---------------------------------------")
    
    for text in texts:
        # Encode with consciousness enhancement
        outputs = await model.encode(
            text,
            consciousness_field,
            return_patterns=True
        )
        
        # Analyze consciousness integration
        consciousness_score = outputs['consciousness_score']
        pattern_strength = torch.mean(torch.stack([
            torch.mean(torch.abs(p['correlation']))
            for p in outputs['consciousness_patterns']
        ]))
        
        print(f"\nText: {text}")
        print(f"Consciousness Score: {consciousness_score:.3f}")
        print(f"Pattern Strength: {pattern_strength:.3f}")
        
        # Show attention visualization (simplified)
        attention = outputs['last_hidden_state'][0].mean(dim=0)
        print("Attention Distribution:")
        print_attention_distribution(attention)


def print_attention_distribution(attention: torch.Tensor):
    """Print simplified visualization of attention distribution."""
    normalized = (attention - attention.min()) / (attention.max() - attention.min())
    distribution = ''.join(['█' if v > 0.5 else '▄' if v > 0.25 else '.' 
                          for v in normalized])
    print(distribution)


async def main():
    """Run demonstrations."""
    print("Demonstrating Consciousness-Enhanced Language Models")
    print("=================================================")
    
    await demonstrate_gpt_enhancement()
    await demonstrate_bert_enhancement()


if __name__ == "__main__":
    asyncio.run(main()) 