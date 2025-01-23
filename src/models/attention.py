import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from ..utils.quantum_ops import quantum_state_preparation

class QuantumAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        quantum_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.quantum_dim = quantum_dim
        
        # Quantum projection matrices
        self.q_quantum = nn.Linear(hidden_dim, quantum_dim * num_heads)
        self.k_quantum = nn.Linear(hidden_dim, quantum_dim * num_heads)
        self.v_quantum = nn.Linear(hidden_dim, hidden_dim)
        
        self.quantum_gate = nn.Parameter(torch.randn(quantum_dim, quantum_dim))
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to quantum space
        q = self.q_quantum(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.quantum_dim
        )
        k = self.k_quantum(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.quantum_dim
        )
        v = self.v_quantum(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        
        # Prepare quantum states
        q = quantum_state_preparation(q)
        k = quantum_state_preparation(k)
        
        # Compute quantum attention scores
        attention_scores = torch.einsum(
            "bshd,bthd->bhst",
            q, k
        ) / torch.sqrt(torch.tensor(self.quantum_dim, dtype=torch.float))
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask[:, None, None, :] == 0, float("-inf")
            )
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.einsum("bhst,bthd->bshd", attention_probs, v)
        
        # Reshape and project output
        context = context.reshape(batch_size, seq_length, self.hidden_dim)
        output = self.output_projection(context)
        
        return output 