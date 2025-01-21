import torch
from typing import AsyncIterator, Optional, Dict
from collections import deque
import asyncio
import logging

from config import BridgeConfig
from bridge import QuantumConsciousnessResonanceBridge, TransferDirection
from llm_adapter import QuantumStateMapper


class StreamingConsciousnessProcessor:
    """Process streaming tokens with consciousness integration."""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.buffer = deque(maxlen=config.buffer_size)
        self.bridge = QuantumConsciousnessResonanceBridge(config)
        self.state_mapper = QuantumStateMapper(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize state tracking
        self.current_state: Optional[torch.Tensor] = None
        self.state_history: Dict[int, torch.Tensor] = {}
    
    async def process_stream(
        self,
        token_stream: AsyncIterator[str],
        consciousness_field: torch.Tensor,
        batch_size: int = 32
    ) -> AsyncIterator[str]:
        """
        Process streaming tokens through consciousness bridge.
        
        Args:
            token_stream: Async iterator of input tokens
            consciousness_field: Current consciousness field
            batch_size: Number of tokens to process in parallel
            
        Yields:
            Enhanced tokens with consciousness integration
        """
        try:
            batch = []
            async for token in token_stream:
                # Add token to batch
                batch.append(token)
                self.buffer.append(token)
                
                if len(batch) >= batch_size:
                    # Process batch
                    enhanced_tokens = await self._process_batch(
                        batch,
                        consciousness_field
                    )
                    
                    # Yield enhanced tokens
                    for enhanced_token in enhanced_tokens:
                        yield enhanced_token
                    
                    # Reset batch
                    batch = []
            
            # Process remaining tokens
            if batch:
                enhanced_tokens = await self._process_batch(
                    batch,
                    consciousness_field
                )
                for enhanced_token in enhanced_tokens:
                    yield enhanced_token
                    
        except Exception as e:
            self.logger.error(f"Error in stream processing: {str(e)}")
            raise
    
    async def _process_batch(
        self,
        tokens: list[str],
        consciousness_field: torch.Tensor
    ) -> list[str]:
        """Process a batch of tokens with consciousness integration."""
        # Convert tokens to quantum states
        quantum_states = self._tokens_to_quantum(tokens)
        
        # Establish bridge connection
        bridge_conn = await self.bridge.establish_bridge(
            quantum_states,
            consciousness_field
        )
        
        # Transfer information
        enhanced_states = await self.bridge.transfer_information(
            quantum_states,
            bridge_conn,
            TransferDirection.BIDIRECTIONAL
        )
        
        # Convert enhanced states back to tokens
        return await self._enhanced_states_to_tokens(enhanced_states.data)
    
    def _tokens_to_quantum(self, tokens: list[str]) -> torch.Tensor:
        """Convert tokens to quantum states."""
        # Create embeddings (placeholder - implement actual embedding)
        embeddings = torch.randn(
            len(tokens),
            self.config.hidden_dim,
            device=self.device
        )
        return self.state_mapper.map_to_quantum(embeddings)
    
    async def _enhanced_states_to_tokens(
        self,
        enhanced_states: torch.Tensor
    ) -> list[str]:
        """Convert enhanced quantum states back to tokens."""
        # Convert states back to embeddings
        embeddings = self.state_mapper.map_to_llm(enhanced_states)
        
        # Convert embeddings to tokens (placeholder - implement actual conversion)
        return ["enhanced_" + str(i) for i in range(embeddings.size(0))]
    
    def _update_state_history(self, token_id: int, state: torch.Tensor):
        """Update state history with new token state."""
        self.state_history[token_id] = state
        
        # Cleanup old states
        if len(self.state_history) > self.config.history_size:
            oldest_id = min(self.state_history.keys())
            del self.state_history[oldest_id] 