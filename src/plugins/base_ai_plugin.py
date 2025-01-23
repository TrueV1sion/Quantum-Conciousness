import torch
from typing import Any, Dict, Optional, Union
from .base_plugin import BasePlugin

class BaseAIPlugin(BasePlugin):
    """
    Base class for AI model plugins. Extends the BasePlugin to provide
    common functionality for AI model integration.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = None
        self.model_config = {}

    def name(self) -> str:
        """Return the name of the AI plugin."""
        return f"AI_{self.model_name}" if self.model_name else "BaseAI"

    def version(self) -> str:
        """Return the version of the plugin."""
        return "1.0.0"

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the AI model with configuration."""
        self.model_name = config.get("model_name", "")
        self.model_config = config.get("model_config", {})
        
    def pre_process(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Pre-process quantum state for AI model input."""
        # Default implementation passes through unchanged
        return quantum_state

    def post_process(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Post-process quantum state after AI model processing."""
        # Default implementation passes through unchanged
        return quantum_state

    def execute(
        self,
        quantum_state: torch.Tensor,
        consciousness_field: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Execute AI model inference."""
        raise NotImplementedError("Subclasses must implement execute method")

    def _validate_model(self) -> bool:
        """Validate that the model is properly initialized."""
        return self.model is not None

    def _prepare_input(
        self,
        quantum_state: torch.Tensor,
        consciousness_field: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare input for model inference."""
        if consciousness_field is not None:
            return {
                'quantum_state': quantum_state.to(self.device),
                'consciousness_field': consciousness_field.to(self.device)
            }
        return quantum_state.to(self.device)

    def _format_output(self, model_output: Any) -> Dict[str, Any]:
        """Format model output into standardized structure."""
        if isinstance(model_output, torch.Tensor):
            return {'output': model_output.detach().cpu()}
        elif isinstance(model_output, dict):
            return {
                k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_output.items()
            }
        return {'output': model_output} 