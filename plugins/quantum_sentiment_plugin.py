import torch
import logging
import numpy as np

from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.plugin_interface import QuantumConsciousnessPlugin

class QuantumSentimentPlugin(QuantumConsciousnessPlugin):
    """
    A plugin that computes an overall 'Quantum Sentiment' by merging a
    classical sentiment analysis score with quantum-inspired coherence metrics.

    This plugin can be especially appealing to devs looking for a
    cutting-edge twist on standard audience/blog/product feedback:
    1. Classical sentiment (transformer-based).
    2. Quantum coherence/resonance from your pipeline.
    3. Merged for a 'Quantum Sentiment Index'.
    """

    def __init__(self):
        self._results = {}

    def name(self) -> str:
        return "QuantumSentimentPlugin"

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Load or set up the sentiment model and any quantum config.
        Example config keys:
          - model_name: HuggingFace model for sentiment (default: "nlptown/bert-base-multilingual-uncased-sentiment")
          - quantum_coherence_weight: how heavily quantum coherence influences the final score
        """
        self.model_name = config.get("model_name", "nlptown/bert-base-multilingual-uncased-sentiment")
        self.quantum_coherence_weight = config.get("quantum_coherence_weight", 0.5)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()

        # If GPU is available and desired, you could move the model
        # to CUDA here; for simplicity, we keep it on CPU.
        logging.info(f"Initialized QuantumSentimentPlugin with model: {self.model_name}")

    def process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects 'state' to contain:
          - text: A string to analyze
          - quantum_field: (Optional) A tensor or array with quantum properties
            (coherence, resonance, etc.) or a structure from your pipeline.

        We'll compute classical sentiment, then scale/shift it by quantum metrics.
        """
        text = state.get("text", "")
        quantum_field = state.get("quantum_field", None)
        if not text:
            # No text to analyze, just return state unchanged
            return state

        # 1. Classical sentiment analysis -------------------------------------
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Typically, you might apply softmax to the logits to get probabilities
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        # For a 5-star model, classes are often [1 star, 2 stars, 3 stars, 4 stars, 5 stars]
        # We do a weighted sum to get a rough "score" from 1.0 to 5.0
        # If you have a different model, adjust accordingly
        star_values = torch.arange(1, 1 + probs.shape[0], dtype=torch.float)
        classical_sentiment_score = torch.dot(probs, star_values).item()

        # 2. Extract quantum coherence/resonance (if provided) ---------------
        if quantum_field is not None and isinstance(quantum_field, (torch.Tensor, np.ndarray)):
            # For illustration, assume quantum_field is a 1D tensor
            # We can compute a coherence metric as average absolute value
            if isinstance(quantum_field, np.ndarray):
                quantum_field = torch.from_numpy(quantum_field)

            coherence = torch.mean(torch.abs(quantum_field)).item()
        else:
            coherence = 1.0  # default if no quantum_field provided

        # 3. Merge classical sentiment with quantum coherence -----------------
        # Weighted combination:
        # e.g., final_score = classical * (1 - w) + coherence * w
        # This can raise or lower the sentiment score depending on quantum coherence
        final_sentiment_score = (
            classical_sentiment_score * (1.0 - self.quantum_coherence_weight)
            + coherence * self.quantum_coherence_weight * 5.0  # scale coherence ~ 1-5 range
        )

        # Store results for retrieval
        self._results = {
            "text": text,
            "classical_sentiment_score": classical_sentiment_score,
            "quantum_coherence": coherence,
            "final_sentiment_score": final_sentiment_score
        }

        # Optionally add to state so subsequent plugins can see it
        state["quantum_sentiment_response"] = self._results
        return state

    def get_results(self) -> Dict[str, Any]:
        """
        Return the last computed sentiment metrics.
        """
        return self._results 