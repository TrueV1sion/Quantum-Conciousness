import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Tuple, Optional

from consciousness_attention import ConsciousnessGuidedAttention
from llm_adapter import LLMConsciousnessAdapter
from config import BridgeConfig

class ConsciousnessEnhancedGPT:
    """GPT model enhanced with consciousness integration."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        config: Optional[BridgeConfig] = None
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Initialize consciousness components
        self.config = config or BridgeConfig()
        self.consciousness_attention = ConsciousnessGuidedAttention(self.config)
        self.adapter = LLMConsciousnessAdapter(model_name, self.config)
        
        # Modify model's attention mechanism
        self._integrate_consciousness_attention()
    
    def _integrate_consciousness_attention(self):
        """Integrate consciousness-guided attention into model."""
        for block in self.model.transformer.h:
            # Store original attention
            original_attention = block.attn
            
            # Create wrapper that adds consciousness
            class ConsciousnessWrapper(nn.Module):
                def __init__(self, original_attn, consciousness_attn):
                    super().__init__()
                    self.original_attn = original_attn
                    self.consciousness_attn = consciousness_attn
                
                async def forward(self, x, consciousness_field, layer_past=None):
                    # Get original attention
                    original_output = self.original_attn(x, layer_past=layer_past)
                    
                    # Apply consciousness enhancement
                    enhanced_attention, _ = await self.consciousness_attn(
                        original_output[0],
                        consciousness_field
                    )
                    
                    return (enhanced_attention,) + original_output[1:]
            
            # Replace attention with wrapped version
            block.attn = ConsciousnessWrapper(
                original_attention,
                self.consciousness_attention
            )
    
    async def generate(
        self,
        prompt: str,
        consciousness_field: torch.Tensor,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Tuple[str, List[float]]:
        """Generate text with consciousness enhancement."""
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Initialize metrics tracking
        consciousness_scores = []
        
        # Generate tokens
        for _ in range(max_length):
            # Get model outputs with consciousness integration
            outputs = await self.adapter.process_with_consciousness(
                input_ids,
                consciousness_field
            )
            
            # Apply temperature and top-p sampling
            logits = outputs[0] / temperature
            filtered_logits = top_p_filtering(logits[:, -1, :], top_p)
            
            # Sample next token
            next_token = torch.multinomial(
                torch.softmax(filtered_logits, dim=-1),
                num_samples=1
            )
            
            # Track consciousness influence
            consciousness_scores.append(outputs[1].integrity_score)
            
            # Add token to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Check for end of sequence
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode output
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        return generated_text, consciousness_scores

def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Filter logits using nucleus (top-p) sampling."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')
    
    return logits 