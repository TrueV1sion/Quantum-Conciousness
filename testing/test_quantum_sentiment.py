import pytest
import torch
import asyncio
import numpy as np

from src.meta_cognitive_pipeline import MetaCognitivePipeline

@pytest.mark.asyncio
async def test_quantum_sentiment_plugin():
    # 1. Define plugin config
    config = {
        "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
        "quantum_coherence_weight": 0.3
    }

    # 2. Initialize pipeline with the plugin directory
    pipeline = MetaCognitivePipeline(plugin_dir="plugins", config=config)

    # 3. Create a sample input (text + quantum_field)
    #    Here, quantum_field is just random for testing.
    initial_state = {
        "text": "I absolutely love this new product. It's fantastic!",
        "quantum_field": torch.randn(16)
    }

    # 4. Run the pipeline
    results = await pipeline.run_pipeline(initial_state)

    # 5. Extract plugin results
    plugin_results = results["plugin_results"].get("QuantumSentimentPlugin", {})
    
    # 6. Assertions to ensure plugin outcomes are returned
    assert "text" in plugin_results, "The plugin result must include 'text'."
    assert "classical_sentiment_score" in plugin_results, "No classical sentiment score was found."
    assert "quantum_coherence" in plugin_results, "No quantum coherence was found in the result."
    assert "final_sentiment_score" in plugin_results, "No final sentiment score was found."

    # Print out the final results for manual inspection
    print("\nQuantumSentimentPlugin Results:")
    for k, v in plugin_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}") 