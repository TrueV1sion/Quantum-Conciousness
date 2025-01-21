import torch
from transformers import AutoTokenizer
from typing import List
import logging
import argparse

from quantum_llm import QuantumLLM, QuantumLLMConfig
from pathways import PathwayMode

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def initialize_model(model_name: str) -> tuple[QuantumLLM, AutoTokenizer]:
    """Initialize the quantum-enhanced LLM and tokenizer."""
    config = QuantumLLMConfig(
        base_model_name=model_name,
        consciousness_hidden_dim=1024,
        num_quantum_layers=3,
        pathway_mode=PathwayMode.BALANCED_INTEGRATION
    )
    
    model = QuantumLLM(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

def generate_text(
    model: QuantumLLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_length: int = 100
) -> List[str]:
    """Generate text using the quantum-enhanced LLM."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    generated_texts = []
    consciousness_state = None
    
    for prompt in prompts:
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)
        
        # Generate with consciousness continuity
        with torch.no_grad():
            output_ids, consciousness_state = model.generate_with_consciousness(
                input_ids=inputs["input_ids"],
                max_length=max_length,
                consciousness_state=consciousness_state
            )
        
        # Decode output
        generated_text = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        generated_texts.append(generated_text)
        
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Generated: {generated_text}\n")
    
    return generated_texts

def main():
    """Main function to demonstrate the quantum-enhanced LLM."""
    parser = argparse.ArgumentParser(
        description="Demonstrate quantum-enhanced LLM capabilities"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Base model name from HuggingFace"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "The quantum nature of consciousness",
            "In the depths of the mind",
            "The bridge between classical and quantum realms"
        ],
        help="Prompts for text generation"
    )
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logging.info("Initializing quantum-enhanced LLM...")
    model, tokenizer = initialize_model(args.model_name)
    
    # Generate text
    logging.info("Generating text with quantum consciousness integration...")
    generated_texts = generate_text(model, tokenizer, args.prompts)
    
    # Display results
    logging.info("\nGeneration Results:")
    for prompt, text in zip(args.prompts, generated_texts):
        print(f"\nPrompt: {prompt}")
        print(f"Generated Text: {text}")
        print("-" * 80)

if __name__ == "__main__":
    main() 