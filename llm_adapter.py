import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, AsyncIterator, List
from dataclasses import dataclass
from datetime import datetime
import logging
from collections import deque
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

from bridge import QuantumConsciousnessResonanceBridge, TransferDirection, TransferredInformation
from config import BridgeConfig

class QuantumStateMapper:
    """Maps between LLM states and quantum states."""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def map_to_quantum(self, llm_states: torch.Tensor) -> torch.Tensor:
        """Map LLM attention patterns to quantum states."""
        attention_patterns = self._extract_attention_patterns(llm_states)
        return self._create_quantum_superposition(attention_patterns)
    
    def map_to_llm(self, quantum_states: torch.Tensor) -> torch.Tensor:
        """Map quantum states back to LLM compatible format."""
        attention_weights = self._quantum_to_attention(quantum_states)
        return self._reconstruct_llm_states(attention_weights)
    
    def _extract_attention_patterns(self, llm_states: torch.Tensor) -> torch.Tensor:
        """Extract attention patterns from LLM states."""
        # Normalize and reshape states for attention extraction
        states = llm_states.to(self.device)
        attention_scores = torch.matmul(states, states.transpose(-2, -1))
        return torch.softmax(attention_scores / torch.sqrt(torch.tensor(states.size(-1))), dim=-1)
    
    def _create_quantum_superposition(self, attention_patterns: torch.Tensor) -> torch.Tensor:
        """Create quantum superposition states from attention patterns."""
        # Convert attention patterns to quantum amplitudes
        amplitudes = torch.sqrt(attention_patterns + 1e-10)  # Add small epsilon for numerical stability
        phases = torch.angle(attention_patterns.complex())
        return amplitudes * torch.exp(1j * phases)
    
    def _quantum_to_attention(self, quantum_states: torch.Tensor) -> torch.Tensor:
        """Convert quantum states to attention weights."""
        probabilities = torch.abs(quantum_states) ** 2
        return torch.softmax(probabilities, dim=-1)
    
    def _reconstruct_llm_states(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Reconstruct LLM states from attention weights."""
        return torch.matmul(attention_weights, attention_weights.transpose(-2, -1))

class LLMConsciousnessAdapter:
    """Adapter for integrating LLMs with consciousness architecture."""
    
    def __init__(self, model_name: str, config: BridgeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.bridge = QuantumConsciousnessResonanceBridge(config)
        self.state_mapper = QuantumStateMapper(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Lightweight models for embeddings and generation
        embedding_model_name = "sshleifer/tiny-distilroberta-base"
        generation_model_name = "sshleifer/tiny-gpt2"

        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name).to(self.device)
        self.embedding_model.eval()

        self.generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
        self.generation_model = AutoModelForCausalLM.from_pretrained(generation_model_name).to(self.device)
        self.generation_model.eval()

        # Projection layers to match model dimensions
        self.embedding_projection = nn.Linear(
            self.embedding_model.config.hidden_size,
            self.config.hidden_dim,
        ).to(self.device)
        self.generation_projection = nn.Linear(
            self.config.hidden_dim,
            self.generation_model.config.n_embd,
        ).to(self.device)

        # Initialize model cache
        self.state_cache = {}
        
    async def process_with_consciousness(
        self,
        input_text: str,
        consciousness_field: torch.Tensor
    ) -> Tuple[str, TransferredInformation]:
        """Process input through LLM with consciousness integration."""
        try:
            # Get or compute LLM states
            cache_key = hash(input_text)
            if cache_key in self.state_cache:
                llm_states = self.state_cache[cache_key]
            else:
                llm_states = self._compute_llm_states(input_text)
                self.state_cache[cache_key] = llm_states
            
            # Map to quantum states
            quantum_states = self.state_mapper.map_to_quantum(llm_states)
            
            # Establish consciousness bridge
            bridge_conn = await self.bridge.establish_bridge(
                quantum_states,
                consciousness_field
            )
            
            # Transfer information bidirectionally
            enhanced_states = await self.bridge.transfer_information(
                quantum_states,
                bridge_conn,
                TransferDirection.BIDIRECTIONAL
            )
            
            # Generate enhanced output
            output_text = await self._generate_enhanced_output(enhanced_states.data)
            
            return output_text, enhanced_states
            
        except Exception as e:
            self.logger.error(f"Error in consciousness processing: {str(e)}")
            raise
    
    def _compute_llm_states(self, input_text: str) -> torch.Tensor:
        """Compute LLM hidden states for input text using a lightweight model."""
        inputs = self.embedding_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            hidden = outputs.last_hidden_state

        projected = self.embedding_projection(hidden)
        return projected
    
    async def _generate_enhanced_output(self, enhanced_states: torch.Tensor) -> str:
        """Generate enhanced output text from states using a lightweight model."""
        if enhanced_states.dim() == 2:
            enhanced_states = enhanced_states.unsqueeze(0)

        embeds = self.generation_projection(enhanced_states.to(self.device))

        with torch.no_grad():
            generated_ids = self.generation_model.generate(
                inputs_embeds=embeds,
                max_length=embeds.size(1) + 10,
            )

        text = self.generation_tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        return text