import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Dict, Optional, Tuple, List
import logging

from consciousness_attention import ConsciousnessGuidedAttention
from llm_adapter import LLMConsciousnessAdapter
from config import BridgeConfig

class ModernBERTConsciousness:
    """ModernBERT-large with consciousness integration."""
    
    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-large",
        config: Optional[BridgeConfig] = None
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize consciousness components
        self.config = config or BridgeConfig(
            consciousness_dim=1024,  # ModernBERT-large hidden size
            quantum_dim=512,
            hidden_dim=1024
        )
        self.consciousness_attention = ConsciousnessGuidedAttention(self.config)
        self.adapter = LLMConsciousnessAdapter(model_name, self.config)
        
        # Add specialized pooling for ModernBERT
        self.modern_pooler = ModernConsciousPooler(self.config)
        
        # Modify model's attention mechanism
        self._integrate_consciousness()
    
    def _integrate_consciousness(self):
        """Integrate consciousness into ModernBERT architecture."""
        for layer in self.model.bert.encoder.layer:
            # Store original attention
            original_attention = layer.attention
            
            # Create specialized wrapper
            class ModernConsciousnessWrapper(nn.Module):
                def __init__(self, original_attn, consciousness_attn):
                    super().__init__()
                    self.original_attn = original_attn
                    self.consciousness_attn = consciousness_attn
                    
                    # Add specialized scaling for ModernBERT
                    self.consciousness_scale = nn.Parameter(
                        torch.ones(1, requires_grad=True)
                    )
                
                async def forward(
                    self,
                    hidden_states: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    consciousness_field: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                    # Get original attention outputs
                    original_output = self.original_attn(
                        hidden_states,
                        attention_mask
                    )
                    
                    if consciousness_field is not None:
                        # Apply consciousness enhancement
                        enhanced_attention, patterns = await self.consciousness_attn(
                            original_output[0],
                            consciousness_field
                        )
                        
                        # Apply learned scaling
                        scaled_attention = (
                            enhanced_attention * self.consciousness_scale.sigmoid()
                            + original_output[0] * (1 - self.consciousness_scale.sigmoid())
                        )
                        
                        return scaled_attention, patterns
                    
                    return original_output[0], {}
            
            # Replace attention with wrapped version
            layer.attention = ModernConsciousnessWrapper(
                original_attention,
                self.consciousness_attention
            )
    
    async def process_text(
        self,
        text: str,
        consciousness_field: torch.Tensor,
        return_patterns: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Process text through consciousness-enhanced ModernBERT."""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Process through consciousness adapter
        enhanced_states, transfer_info = await self.adapter.process_with_consciousness(
            inputs['input_ids'],
            consciousness_field
        )
        
        # Forward pass through model
        outputs = {}
        consciousness_patterns = []
        
        # Process through each layer
        current_states = enhanced_states
        for layer in self.model.bert.encoder.layer:
            layer_output, patterns = await layer.attention(
                current_states,
                inputs['attention_mask'],
                consciousness_field
            )
            current_states = layer_output
            consciousness_patterns.append(patterns)
        
        # Apply modern conscious pooling
        pooled_output = await self.modern_pooler(
            current_states,
            consciousness_field
        )
        
        # Prepare outputs
        outputs['last_hidden_state'] = current_states
        outputs['pooled_output'] = pooled_output
        outputs['consciousness_score'] = transfer_info.integrity_score
        
        if return_patterns:
            outputs['consciousness_patterns'] = consciousness_patterns
        
        return outputs


class ModernConsciousPooler(nn.Module):
    """Specialized pooling for ModernBERT with consciousness."""
    
    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.activation = nn.GELU()  # ModernBERT uses GELU
        
        # Add consciousness-specific components
        self.consciousness_gate = nn.Parameter(torch.randn(config.hidden_dim))
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
    
    async def forward(
        self,
        hidden_states: torch.Tensor,
        consciousness_field: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with consciousness-aware pooling."""
        # Apply initial pooling
        pooled = self.dense(hidden_states[:, 0])
        
        # Consciousness gating with layer norm
        consciousness_gate = torch.sigmoid(self.consciousness_gate)
        consciousness_weight = torch.matmul(
            consciousness_field,
            consciousness_gate.unsqueeze(-1)
        ).squeeze(-1)
        
        # Apply consciousness-weighted activation with layer norm
        pooled = self.layer_norm(
            self.activation(pooled) * consciousness_weight.unsqueeze(-1)
        )
        
        return pooled 