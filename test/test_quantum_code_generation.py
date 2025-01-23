import asyncio
import torch

from src.meta_cognitive_pipeline import MetaCognitivePipeline

async def main():
    config = {
        "model_name": "microsoft/CodeGPT-small-py",
        "quantum_coherence_weight": 0.2,
        "max_length": 64,
        "temperature": 0.8,
        "top_p": 0.95
    }
    pipeline = MetaCognitivePipeline(plugin_dir="plugins", config=config)

    initial_state = {
        "code_prompt": "def fibonacci(n):\n    # compute fibonacci sequence up to n",
        "quantum_field": torch.randn(16)  # random quantum field for testing
    }

    results = await pipeline.run_pipeline(initial_state)
    plugin_results = results["plugin_results"].get("QuantumCodeGeneratorPlugin", {})

    # Print result
    print("Generated Code:", plugin_results.get("final_code", "No code generated"))
    print("Quantum Coherence Used:", plugin_results.get("quantum_coherence"))
    print("Adjusted Temperature:", plugin_results.get("adjusted_temperature"))
    print("Adjusted Top_p:", plugin_results.get("adjusted_top_p"))

if __name__ == "__main__":
    asyncio.run(main()) 