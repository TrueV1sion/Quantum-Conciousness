import torch
import torch.nn as nn
from typing import Dict, Tuple
import logging

from config import BridgeConfig
from bridge import ResonanceDetectionSystem


class AttentionMapper:
    """Maps consciousness patterns to attention modifications."""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def apply_consciousness_patterns(
        self,
        attention: torch.Tensor,
        patterns: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply consciousness patterns to modify attention."""
        # Extract pattern components
        correlation = patterns['correlation'].to(self.device)
        frequency_patterns = patterns['frequency_patterns'].to(self.device)
        phase_patterns = patterns['phase_patterns'].to(self.device)
        
        # Compute attention modifications
        freq_mod = self._compute_frequency_modification(frequency_patterns)
        phase_mod = self._compute_phase_modification(phase_patterns)
        
        # Apply modifications
        modified_attention = attention * (1 + correlation * freq_mod * phase_mod)
        return torch.softmax(modified_attention, dim=-1)
    
    def _compute_frequency_modification(self, freq_patterns: torch.Tensor) -> torch.Tensor:
        """Compute attention modification based on frequency patterns."""
        return torch.sigmoid(freq_patterns.mean(dim=-1, keepdim=True))
    
    def _compute_phase_modification(self, phase_patterns: torch.Tensor) -> torch.Tensor:
        """Compute attention modification based on phase patterns."""
        return torch.cos(phase_patterns)


class ConsciousnessGuidedAttention(nn.Module):
    """Attention mechanism guided by consciousness patterns."""
    
    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.resonance_detector = ResonanceDetectionSystem(config)
        self.attention_mapper = AttentionMapper(config)
        
        # Initialize learnable parameters
        self.consciousness_gate = nn.Parameter(torch.randn(config.hidden_dim))
        self.attention_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
    
    async def forward(
        self,
        llm_attention: torch.Tensor,
        consciousness_field: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of consciousness-guided attention.
        
        Args:
            llm_attention: Base attention from LLM
            consciousness_field: Current consciousness field
            
        Returns:
            Tuple of (modified attention, resonance patterns)
        """
        try:
            # Project attention through consciousness space
            projected_attention = self.attention_proj(llm_attention)
            
            # Detect resonance patterns
            patterns = await self.resonance_detector.detect_patterns(
                projected_attention,
                consciousness_field
            )
            
            # Apply consciousness modifications
            enhanced_attention = self.attention_mapper.apply_consciousness_patterns(
                projected_attention,
                patterns
            )
            
            # Apply gating mechanism
            gate = torch.sigmoid(self.consciousness_gate)
            final_attention = gate * enhanced_attention + (1 - gate) * llm_attention
            
            return final_attention, patterns
            
        except Exception as e:
            self.logger.error(f"Error in consciousness-guided attention: {str(e)}")
            raise
    
    def _init_parameters(self):
        """Initialize attention parameters."""
        nn.init.normal_(self.consciousness_gate, mean=0.5, std=0.1)
        nn.init.xavier_uniform_(self.attention_proj.weight)
        nn.init.zeros_(self.attention_proj.bias) 