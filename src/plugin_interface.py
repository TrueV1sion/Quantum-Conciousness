from abc import ABC, abstractmethod
from typing import Any, Dict

class QuantumConsciousnessPlugin(ABC):
    """
    Abstract base class for quantum consciousness plugins.
    
    Plugins can introduce new quantum gates, consciousness 
    metrics, or advanced analytics that integrate seamlessly 
    into the pipeline.
    """
    
    @abstractmethod
    def name(self) -> str:
        """A unique name for the plugin."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Perform any initialization steps for the plugin.
        E.g., load model files, set up parameters, etc.
        """
        pass

    @abstractmethod
    def process_state(self, state: Any) -> Any:
        """
        Core plugin logic.
        E.g., apply a quantum gate, compute new metrics, or
        transform a consciousness state in some way.
        """
        pass

    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """
        Optionally return any computed results or metrics 
        after process_state is called.
        """
        pass 