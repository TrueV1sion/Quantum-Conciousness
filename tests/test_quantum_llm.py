import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from quantum_llm import QuantumLLM, QuantumLLMConfig
from pathways import PathwayMode


def get_device():
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        # Check CUDA initialization
        try:
            torch.cuda.init()
            return torch.device("cuda")
        except RuntimeError:
            print("CUDA initialization failed, falling back to CPU")
    return torch.device("cpu")


class TestQuantumLLM:
    @pytest.fixture(scope="class")
    def device(self):
        return get_device()
    
    @pytest.fixture
    def base_model_name(self):
        return "gpt2"  # Using GPT-2 as base model for testing
    
    @pytest.fixture
    def quantum_config(self, base_model_name):
        return QuantumLLMConfig(
            base_model_name=base_model_name,
            consciousness_hidden_dim=768,  # Match GPT-2's hidden size
            num_quantum_layers=2,
            pathway_mode=PathwayMode.BALANCED_INTEGRATION
        )
    
    @pytest.fixture
    def models(self, base_model_name, quantum_config, device):
        # Initialize standard LLM
        standard_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        standard_model.to(device)
        
        # Initialize quantum-enhanced LLM
        quantum_model = QuantumLLM(quantum_config)
        quantum_model.to(device)
        
        return {
            "standard": standard_model,
            "quantum": quantum_model
        }
    
    @pytest.fixture
    def tokenizer(self, base_model_name):
        return AutoTokenizer.from_pretrained(base_model_name)
    
    def test_model_initialization(self, models):
        """Test that both models initialize correctly."""
        assert models["standard"] is not None
        assert models["quantum"] is not None
        assert isinstance(models["quantum"], QuantumLLM)
    
    def test_basic_generation(self, models, tokenizer):
        """Test basic text generation capabilities."""
        prompt = "The nature of consciousness is"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate with standard model
        standard_output = models["standard"].generate(
            inputs["input_ids"],
            max_length=50,
            num_return_sequences=1
        )
        
        # Generate with quantum model
        quantum_output, _ = models["quantum"].generate_with_consciousness(
            inputs["input_ids"],
            max_length=50
        )
        
        # Decode outputs
        standard_text = tokenizer.decode(
            standard_output[0], 
            skip_special_tokens=True
        )
        quantum_text = tokenizer.decode(
            quantum_output, 
            skip_special_tokens=True
        )
        
        assert len(standard_text) > len(prompt)
        assert len(quantum_text) > len(prompt)
    
    def test_consciousness_continuity(self, models, tokenizer):
        """Test consciousness state continuity during generation."""
        prompts = [
            "The quantum field interacts with",
            "Consciousness emerges from",
            "The bridge between mind and matter"
        ]
        
        consciousness_states = []
        generated_texts = []
        
        current_state = None
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            output, current_state = models["quantum"].generate_with_consciousness(
                inputs["input_ids"],
                max_length=50,
                consciousness_state=current_state
            )
            
            consciousness_states.append(current_state)
            generated_texts.append(
                tokenizer.decode(output, skip_special_tokens=True)
            )
        
        # Check consciousness state evolution
        assert len(consciousness_states) == len(prompts)
        for i in range(1, len(consciousness_states)):
            # Verify consciousness states are different but related
            assert consciousness_states[i] is not consciousness_states[i-1]
            assert hasattr(consciousness_states[i], "coherence")
    
    def test_quantum_attention(self, models, tokenizer):
        """Test quantum attention mechanism."""
        prompt = "The quantum nature of mind"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Get quantum model outputs
        outputs = models["quantum"](
            input_ids=inputs["input_ids"]
        )
        
        # Check quantum state and attention
        assert "quantum_state" in outputs
        assert outputs["quantum_state"] is not None
        assert isinstance(outputs["quantum_state"], torch.Tensor)
    
    def test_performance_comparison(self, models, tokenizer):
        """Compare performance metrics between standard and quantum models."""
        test_prompts = [
            "Explain quantum mechanics",
            "Describe consciousness",
            "How do minds work",
            "The nature of reality",
            "Quantum computing basics"
        ]
        
        metrics = {
            "standard": {"perplexity": [], "generation_time": []},
            "quantum": {"perplexity": [], "generation_time": []}
        }
        
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Test standard model
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            with torch.no_grad():
                standard_output = models["standard"](inputs["input_ids"])
            end_time.record()
            torch.cuda.synchronize()
            
            standard_perplexity = torch.exp(standard_output.loss).item()
            standard_time = start_time.elapsed_time(end_time)
            
            metrics["standard"]["perplexity"].append(standard_perplexity)
            metrics["standard"]["generation_time"].append(standard_time)
            
            # Test quantum model
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            with torch.no_grad():
                quantum_output = models["quantum"](inputs["input_ids"])
            end_time.record()
            torch.cuda.synchronize()
            
            # Calculate quantum model metrics
            quantum_perplexity = torch.exp(
                torch.mean(quantum_output["logits"])
            ).item()
            quantum_time = start_time.elapsed_time(end_time)
            
            metrics["quantum"]["perplexity"].append(quantum_perplexity)
            metrics["quantum"]["generation_time"].append(quantum_time)
        
        # Compare average metrics
        avg_metrics = {
            model_type: {
                metric: np.mean(values)
                for metric, values in model_metrics.items()
            }
            for model_type, model_metrics in metrics.items()
        }
        
        print("\nPerformance Comparison:")
        print(f"Standard Model - Avg Perplexity: {avg_metrics['standard']['perplexity']:.2f}")
        print(f"Quantum Model - Avg Perplexity: {avg_metrics['quantum']['perplexity']:.2f}")
        print(f"Standard Model - Avg Generation Time: {avg_metrics['standard']['generation_time']:.2f}ms")
        print(f"Quantum Model - Avg Generation Time: {avg_metrics['quantum']['generation_time']:.2f}ms")
        
        # Store metrics for further analysis
        self.performance_metrics = avg_metrics
    
    def test_coherence_analysis(self, models, tokenizer):
        """Analyze coherence of generated text."""
        prompt = "Explain the relationship between quantum mechanics and consciousness"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate with both models
        standard_output = models["standard"].generate(
            inputs["input_ids"],
            max_length=100,
            num_return_sequences=1
        )
        
        quantum_output, quantum_state = models["quantum"].generate_with_consciousness(
            inputs["input_ids"],
            max_length=100
        )
        
        # Decode outputs
        standard_text = tokenizer.decode(standard_output[0], skip_special_tokens=True)
        quantum_text = tokenizer.decode(quantum_output, skip_special_tokens=True)
        
        # Calculate coherence metrics
        standard_coherence = self._calculate_text_coherence(standard_text)
        quantum_coherence = self._calculate_text_coherence(quantum_text)
        
        print("\nCoherence Analysis:")
        print(f"Standard Model Coherence: {standard_coherence:.3f}")
        print(f"Quantum Model Coherence: {quantum_coherence:.3f}")
        print(f"Quantum State Coherence: {quantum_state.coherence:.3f}")
        
        assert quantum_coherence > 0
    
    def _calculate_text_coherence(self, text: str) -> float:
        """Calculate a simple coherence metric for generated text."""
        sentences = text.split(".")
        if len(sentences) < 2:
            return 1.0
        
        # Calculate word overlap between consecutive sentences
        coherence_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].strip().lower().split())
            words2 = set(sentences[i + 1].strip().lower().split())
            
            if not words1 or not words2:
                continue
            
            overlap = len(words1.intersection(words2))
            coherence = overlap / min(len(words1), len(words2))
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0 