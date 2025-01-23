import torch
import logging
from typing import Dict, Any, Optional
import traceback

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.plugin_interface import QuantumConsciousnessPlugin

class QuantumCodeGeneratorPlugin(QuantumConsciousnessPlugin):
    """
    A plugin that uses a quantum-enhanced approach to generate code:
    1. Employs a GPT-style model for code completion.
    2. Integrates quantum-consciousness signals to self-check and refine tokens.
    """

    def __init__(self):
        self._results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def name(self) -> str:
        return "QuantumCodeGeneratorPlugin"

    def initialize(self, config: Dict[str, Any]) -> None:
        try:
            logging.debug(f"QuantumCodeGeneratorPlugin received config: {config}")
            self.model_name = config.get("model_name", "microsoft/CodeGPT-small-py")
            self.quantum_coherence_weight = config.get("quantum_coherence_weight", 0.2)
            self.max_length = config.get("max_length", 128)
            self.temperature = config.get("temperature", 0.8)
            self.top_p = config.get("top_p", 0.95)
            self.fallback_device = config.get("fallback_device", "cpu")

            # Load model and tokenizer with error handling
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                
                # Check available GPU memory before moving to device
                if self.device.type == "cuda":
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
                    if model_size > gpu_memory * 0.9:  # Leave 10% buffer
                        logging.warning("Insufficient GPU memory. Falling back to CPU.")
                        self.device = torch.device(self.fallback_device)
                
                self.model = self.model.to(self.device)
                self.model.eval()
                
            except Exception as e:
                logging.error(f"Failed to load model: {str(e)}")
                raise RuntimeError(f"Model initialization failed: {str(e)}")

            logging.info(f"Initialized QuantumCodeGeneratorPlugin with model: {self.model_name} on device: {self.device}")
            
        except Exception as e:
            logging.error(f"Plugin initialization failed: {str(e)}\n{traceback.format_exc()}")
            raise

    def validate_quantum_field(self, quantum_field: Optional[torch.Tensor]) -> float:
        """Validate and process quantum field input."""
        if quantum_field is None:
            logging.warning("No quantum field provided, using default coherence value.")
            return 1.0
            
        try:
            if not isinstance(quantum_field, torch.Tensor):
                logging.warning(f"Invalid quantum field type: {type(quantum_field)}. Expected torch.Tensor.")
                return 1.0
                
            if quantum_field.dim() > 2:
                logging.warning("Quantum field has more than 2 dimensions. Using mean across all dimensions.")
                return torch.mean(torch.abs(quantum_field)).item()
                
            return torch.mean(torch.abs(quantum_field)).item()
            
        except Exception as e:
            logging.error(f"Error processing quantum field: {str(e)}")
            return 1.0

    def process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = state.get("code_prompt", "")
            if not prompt:
                logging.warning("Empty code prompt received.")
                return state

            quantum_field = state.get("quantum_field")
            coherence = self.validate_quantum_field(quantum_field)

            # Tokenize with error handling
            try:
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            except Exception as e:
                logging.error(f"Tokenization failed: {str(e)}")
                self._results = {"error": f"Tokenization failed: {str(e)}"}
                state["quantum_code_generation"] = self._results
                return state

            # Adjust sampling parameters
            adjusted_temperature = max(0.1, self.temperature - (coherence * self.quantum_coherence_weight))
            adjusted_top_p = min(1.0, self.top_p + (self.quantum_coherence_weight * (1.0 - coherence)))

            # Generate code with memory management and error handling
            try:
                with torch.no_grad():
                    # Check for CUDA out of memory
                    if self.device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                            output_ids = self.model.generate(
                                input_ids,
                                max_length=len(input_ids[0]) + self.max_length,
                                temperature=adjusted_temperature,
                                top_p=adjusted_top_p,
                                do_sample=True,
                                pad_token_id=self.tokenizer.eos_token_id
                            )
                        except torch.cuda.OutOfMemoryError:
                            logging.warning("CUDA out of memory. Falling back to CPU.")
                            self.model = self.model.to("cpu")
                            input_ids = input_ids.to("cpu")
                            output_ids = self.model.generate(
                                input_ids,
                                max_length=len(input_ids[0]) + self.max_length,
                                temperature=adjusted_temperature,
                                top_p=adjusted_top_p,
                                do_sample=True,
                                pad_token_id=self.tokenizer.eos_token_id
                            )
                            self.model = self.model.to(self.device)  # Try moving back to GPU
                    else:
                        output_ids = self.model.generate(
                            input_ids,
                            max_length=len(input_ids[0]) + self.max_length,
                            temperature=adjusted_temperature,
                            top_p=adjusted_top_p,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )

                generated_code = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

                self._results = {
                    "prompt": prompt,
                    "quantum_coherence": coherence,
                    "final_code": generated_code,
                    "adjusted_temperature": adjusted_temperature,
                    "adjusted_top_p": adjusted_top_p,
                    "device": str(self.device)
                }

            except Exception as e:
                error_msg = f"Code generation failed: {str(e)}"
                logging.error(f"{error_msg}\n{traceback.format_exc()}")
                self._results = {
                    "error": error_msg,
                    "prompt": prompt,
                    "quantum_coherence": coherence
                }

            state["quantum_code_generation"] = self._results
            return state

        except Exception as e:
            logging.error(f"Unexpected error in process_state: {str(e)}\n{traceback.format_exc()}")
            state["quantum_code_generation"] = {"error": f"Plugin processing failed: {str(e)}"}
            return state

    def get_results(self) -> Dict[str, Any]:
        return self._results 