from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch

class BasePlugin(ABC):
    """
    Abstract base class for quantum consciousness plugins.
    """

    @abstractmethod
    def name(self) -> str:
        """Return the name of the plugin."""
        pass

    @abstractmethod
    def version(self) -> str:
        """Return the version of the plugin."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with the given configuration."""
        pass

    @abstractmethod
    def pre_process(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Hook before the main quantum processing."""
        pass

    @abstractmethod
    def post_process(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Hook after the main quantum processing."""
        pass

    @abstractmethod
    def execute(
        self,
        quantum_state: torch.Tensor,
        consciousness_field: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Execute the plugin's quantum consciousness functionality."""
        pass 