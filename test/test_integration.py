import pytest
import torch
import asyncio
from src.meta_cognitive_pipeline import MetaCognitivePipeline

@pytest.mark.asyncio
async def test_integration_pipeline():
    config = {"model_name": "microsoft/CodeGPT-small-py"}
    pipeline = MetaCognitivePipeline(plugin_dir="plugins", config=config)

    # Verify plugin discovery
    loaded_plugin_names = [p.name() for p in pipeline.plugin_manager.plugins]
    assert "QuantumCodeGeneratorPlugin" in loaded_plugin_names, "QuantumCodeGeneratorPlugin not loaded!"

    # Now run the pipeline with a basic initial_state
    initial_state = {"code_prompt": "def test_func(x):"}
    results = await pipeline.run_pipeline(initial_state)
    plugin_results = results["plugin_results"].get("QuantumCodeGeneratorPlugin", {})

    assert "final_code" in plugin_results, "Generated code was not returned by the plugin."
    print("Integration test passed! Plugins are loaded and producing code.")

    # Additional assertions about plugin output or final state 