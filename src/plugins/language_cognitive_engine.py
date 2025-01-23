import torch
from typing import Any, Dict, Optional, List
from transformers import AutoModel, AutoTokenizer, AutoConfig
from .base_cognitive_engine import BaseCognitiveEngine, ContextNode


class LanguageCognitiveEngine(BaseCognitiveEngine):
    """
    Cognitive engine specialized for language processing.
    Uses transformer models for text understanding and generation.
    """
    
    def __init__(self):
        super().__init__()
        self.modality = "language"
        self.model = None
        self.tokenizer = None
        self.max_length = 512
        self.embedding_dim = 768  # Default for BERT-base

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the language model and tokenizer."""
        super().initialize(config)
        
        model_name = config.get("model_name", "bert-base-uncased")
        model_config = config.get("model_config", {})
        
        # Load model configuration
        model_config = AutoConfig.from_pretrained(model_name, **model_config)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            config=model_config
        )
        self.model.to(self.device)
        
        # Set additional configuration
        self.max_length = config.get("max_length", 512)
        self.embedding_dim = self.model.config.hidden_size
        
        # Set model to evaluation mode
        self.model.eval()

    def execute(
        self,
        quantum_state: torch.Tensor,
        consciousness_field: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Process language input and update context lattice."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized")

        try:
            # Convert quantum state to text embeddings
            embeddings = self.pre_process(quantum_state)
            
            # Run model inference
            with torch.no_grad():
                outputs = self.model(inputs_embeds=embeddings)
            
            # Extract hidden states and create context nodes
            hidden_states = outputs.last_hidden_state
            
            # Create context nodes for each significant hidden state
            nodes = []
            for i, hidden in enumerate(hidden_states[0]):
                if torch.norm(hidden) > 0.1:  # Filter out weak signals
                    node = self.create_context_node(
                        content=hidden.cpu(),
                        confidence=torch.norm(hidden).item() / \
                            torch.norm(hidden_states[0]).item(),
                        metadata={'position': i}
                    )
                    nodes.append(node)
            
            # Connect related nodes based on attention patterns
            if len(nodes) > 1:
                attention_weights = outputs.attentions[-1][0] if \
                    hasattr(outputs, 'attentions') else None
                if attention_weights is not None:
                    self._connect_nodes_by_attention(nodes, attention_weights)
            
            # Create ephemeral context
            context_id = f"language_context_{len(self.ephemeral_contexts)}"
            self.create_ephemeral_context(context_id)
            
            # Add nodes to ephemeral context
            for node in nodes:
                self.add_to_ephemeral_context(context_id, node)
            
            return {
                'context_id': context_id,
                'nodes': nodes,
                'hidden_states': hidden_states.cpu()
            }
            
        except Exception as e:
            raise RuntimeError(f"Language processing failed: {str(e)}")

    def get_node_embedding(self, node: ContextNode) -> torch.Tensor:
        """Get embedding representation of a node's content."""
        if isinstance(node.content, torch.Tensor):
            return node.content.to(self.device)
        elif isinstance(node.content, str):
            inputs = self.tokenizer(
                node.content,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            with torch.no_grad():
                outputs = self.model(**inputs.to(self.device))
            return outputs.last_hidden_state.mean(dim=1)
        else:
            raise ValueError(f"Unsupported content type: {type(node.content)}")

    def _are_contradictory(
        self,
        node1: ContextNode,
        node2: ContextNode
    ) -> bool:
        """Detect contradictions between language nodes."""
        # Get embeddings for both nodes
        emb1 = self.get_node_embedding(node1)
        emb2 = self.get_node_embedding(node2)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            emb1.view(1, -1),
            emb2.view(1, -1)
        ).item()
        
        # Nodes with high confidence but opposite meanings
        # (very low similarity) are considered contradictory
        return (
            node1.confidence > 0.7
            and node2.confidence > 0.7
            and similarity < -0.5
        )

    def _connect_nodes_by_attention(
        self,
        nodes: List[ContextNode],
        attention_weights: torch.Tensor
    ) -> None:
        """Connect nodes based on attention patterns."""
        # Average attention weights across heads
        avg_attention = attention_weights.mean(dim=0)
        
        # Connect nodes with strong attention weights
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                weight = avg_attention[
                    nodes[i].metadata['position'],
                    nodes[j].metadata['position']
                ].item()
                if weight > 0.1:  # Attention threshold
                    self.connect_nodes(nodes[i].id, nodes[j].id) 