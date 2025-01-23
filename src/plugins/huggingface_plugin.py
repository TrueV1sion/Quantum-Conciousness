import torch
from typing import Any, Dict, Optional
from transformers import AutoModel, AutoTokenizer, AutoConfig
from .base_ai_plugin import BaseAIPlugin


class HuggingFacePlugin(BaseAIPlugin):
    """
    Plugin for integrating Hugging Face transformer models.
    """
    
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.max_length = 512
        self.supported_tasks = [
            'text-generation',
            'text-classification',
            'token-classification',
            'question-answering'
        ]

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the Hugging Face model and tokenizer."""
        super().initialize(config)
        
        if not self.model_name:
            raise ValueError("model_name must be specified in config")
            
        # Load model configuration
        model_config = AutoConfig.from_pretrained(
            self.model_name,
            **self.model_config
        )
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            config=model_config
        )
        self.model.to(self.device)
        
        # Set additional configuration
        self.max_length = config.get("max_length", 512)
        
        # Set model to evaluation mode
        self.model.eval()

    def pre_process(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Convert quantum state to text embeddings."""
        # Project quantum state to embedding dimension
        batch_size = quantum_state.shape[0]
        projected_state = torch.nn.functional.linear(
            quantum_state.view(batch_size, -1),
            self.model.get_input_embeddings().weight.t()
        )
        return projected_state

    def post_process(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Convert model outputs back to quantum state format."""
        # Project back to quantum state dimension
        batch_size = quantum_state.shape[0]
        projected_state = torch.nn.functional.linear(
            quantum_state.view(batch_size, -1),
            self.model.get_output_embeddings().weight if \
                hasattr(self.model, 'get_output_embeddings') else \
                self.model.get_input_embeddings().weight
        )
        return projected_state

    def execute(
        self,
        quantum_state: torch.Tensor,
        consciousness_field: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Execute the Hugging Face model."""
        if not self._validate_model():
            raise RuntimeError("Model not initialized")

        try:
            # Prepare inputs
            inputs = self._prepare_input(quantum_state, consciousness_field)
            
            # Run model inference
            with torch.no_grad():
                if isinstance(inputs, dict):
                    outputs = self.model(**inputs)
                else:
                    outputs = self.model(inputs)
            
            # Format outputs
            results = self._format_output(outputs)
            
            # Add model-specific metadata
            results.update({
                'model_name': self.model_name,
                'model_type': self.model.config.model_type,
                'hidden_size': self.model.config.hidden_size,
                'vocab_size': self.model.config.vocab_size
            })
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Model execution failed: {str(e)}")

    def _validate_model(self) -> bool:
        """Validate Hugging Face model initialization."""
        return super()._validate_model() and self.tokenizer is not None

    def _prepare_input(
        self,
        quantum_state: torch.Tensor,
        consciousness_field: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for Hugging Face model."""
        # Convert quantum state to attention mask
        attention_mask = torch.ones_like(
            quantum_state[..., 0],
            dtype=torch.long,
            device=self.device
        )
        
        inputs = {
            'inputs_embeds': quantum_state.to(self.device),
            'attention_mask': attention_mask
        }
        
        if consciousness_field is not None:
            # Use consciousness field as token type IDs
            token_type_ids = (consciousness_field > 0).long()
            inputs['token_type_ids'] = token_type_ids.to(self.device)
            
        return inputs 