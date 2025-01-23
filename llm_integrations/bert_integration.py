import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import List, Dict, Optional, Tuple
import numpy as np

from consciousness_attention import ConsciousnessGuidedAttention
from llm_adapter import LLMConsciousnessAdapter
from config import BridgeConfig

class ConsciousnessEnhancedBERT:
    """BERT model enhanced with consciousness integration."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        config: Optional[BridgeConfig] = None
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Initialize consciousness components
        self.config = config or BridgeConfig()
        self.consciousness_attention = ConsciousnessGuidedAttention(self.config)
        self.adapter = LLMConsciousnessAdapter(model_name, self.config)
        
        # Add consciousness-enhanced pooling
        self.conscious_pooler = ConsciousPooler(self.config)
        
        # Modify model's attention mechanism
        self._integrate_consciousness_attention()
    
    def _integrate_consciousness_attention(self):
        """Integrate consciousness-guided attention into model."""
        for layer in self.model.encoder.layer:
            # Store original attention
            original_attention = layer.attention
            
            # Create wrapper that adds consciousness
            class ConsciousnessWrapper(nn.Module):
                def __init__(self, original_attn, consciousness_attn):
                    super().__init__()
                    self.original_attn = original_attn
                    self.consciousness_attn = consciousness_attn
                
                async def forward(
                    self,
                    hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor,
                    consciousness_field: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
                    # Get original attention outputs
                    original_output = self.original_attn(
                        hidden_states,
                        attention_mask
                    )
                    
                    # Apply consciousness enhancement
                    enhanced_attention, patterns = await self.consciousness_attn(
                        original_output[0],
                        consciousness_field
                    )
                    
                    return enhanced_attention, patterns
            
            # Replace attention with wrapped version
            layer.attention = ConsciousnessWrapper(
                original_attention,
                self.consciousness_attention
            )
    
    async def encode(
        self,
        text: str,
        consciousness_field: torch.Tensor,
        return_patterns: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Encode text with consciousness enhancement."""
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
        
        # Get attention mask
        attention_mask = inputs['attention_mask']
        
        # Forward pass through model
        outputs = {}
        consciousness_patterns = []
        
        for layer in self.model.encoder.layer:
            # Apply consciousness-enhanced attention
            layer_output, patterns = await layer.attention(
                enhanced_states,
                attention_mask,
                consciousness_field
            )
            enhanced_states = layer_output
            consciousness_patterns.append(patterns)
        
        # Apply consciousness-enhanced pooling
        pooled_output = await self.conscious_pooler(
            enhanced_states,
            consciousness_field
        )
        
        # Prepare outputs
        outputs['last_hidden_state'] = enhanced_states
        outputs['pooler_output'] = pooled_output
        outputs['consciousness_score'] = transfer_info.integrity_score
        
        if return_patterns:
            outputs['consciousness_patterns'] = consciousness_patterns
        
        return outputs


class ConsciousPooler(nn.Module):
    """Consciousness-enhanced pooling layer."""
    
    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.activation = nn.Tanh()
        self.consciousness_gate = nn.Parameter(torch.randn(config.hidden_dim))
    
    async def forward(
        self,
        hidden_states: torch.Tensor,
        consciousness_field: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with consciousness gating."""
        # Regular pooling
        pooled = self.dense(hidden_states[:, 0])
        
        # Consciousness gating
        consciousness_gate = torch.sigmoid(self.consciousness_gate)
        consciousness_weight = torch.matmul(
            consciousness_field,
            consciousness_gate.unsqueeze(-1)
        ).squeeze(-1)
        
        # Apply consciousness-weighted activation
        pooled = self.activation(pooled) * consciousness_weight.unsqueeze(-1)
        
        return pooled 