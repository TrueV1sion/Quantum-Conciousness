from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
import torch
import asyncio

# Import your pipeline and plugin manager classes
from src.meta_cognitive_pipeline import MetaCognitivePipeline

app = FastAPI(
    title="Quantum Code Generation SaaS",
    description="A simple SaaS interface for quantum-enhanced code generation.",
    version="0.1.0"
)

# Example config for the quantum code generator plugin
DEFAULT_CONFIG = {
    "model_name": "microsoft/CodeGPT-small-py",
    "quantum_coherence_weight": 0.2,
    "max_length": 64,
    "temperature": 0.8,
    "top_p": 0.95
}

# Pydantic Models for request/response typing
class CodeGenRequest(BaseModel):
    code_prompt: str
    use_quantum: bool = True

class CodeGenResponse(BaseModel):
    final_code: str
    quantum_coherence: float
    adjusted_temperature: float
    adjusted_top_p: float


@app.on_event("startup")
async def startup_event():
    """
    Initialize the MetaCognitivePipeline at startup.
    """
    global pipeline
    pipeline = MetaCognitivePipeline(plugin_dir="plugins", config=DEFAULT_CONFIG)
    # The plugins (including QuantumCodeGeneratorPlugin) will be loaded automatically.


@app.post("/generate_code", response_model=CodeGenResponse)
async def generate_code(request: CodeGenRequest):
    try:
        quantum_field = torch.randn(16) if request.use_quantum else None
        initial_state = {"code_prompt": request.code_prompt, "quantum_field": quantum_field}
        results = await pipeline.run_pipeline(initial_state)
        plugin_results = results["plugin_results"].get("QuantumCodeGeneratorPlugin", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

    return CodeGenResponse(
        final_code=plugin_results.get("final_code", ""),
        quantum_coherence=plugin_results.get("quantum_coherence", 1.0),
        adjusted_temperature=plugin_results.get("adjusted_temperature", 0.8),
        adjusted_top_p=plugin_results.get("adjusted_top_p", 0.95)
    ) 