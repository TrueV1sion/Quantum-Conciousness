import os
import importlib.util
import logging
from typing import List, Dict, Any
from .plugin_interface import QuantumConsciousnessPlugin

class PluginManager:
    """
    Loads and manages QuantumConsciousnessPlugin instances 
    from a specified directory. 
    """

    def __init__(self, plugin_dir: str, config: Dict[str, Any]):
        self.plugin_dir = plugin_dir
        self.config = config
        self.plugins: List[QuantumConsciousnessPlugin] = []

    def load_plugins(self) -> None:
        """
        Dynamically load and instantiate all plugins 
        in the plugin_dir.
        """
        for filename in os.listdir(self.plugin_dir):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(self.plugin_dir, filename)
            spec = importlib.util.spec_from_file_location(
                filename[:-3], path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for plugin classes in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type) and 
                    issubclass(attr, QuantumConsciousnessPlugin) and 
                    attr is not QuantumConsciousnessPlugin
                ):
                    plugin_instance = attr()
                    plugin_instance.initialize(self.config)
                    self.plugins.append(plugin_instance)
                    logging.info(f"Loaded plugin: {plugin_instance.name()}")

    def process_state_through_plugins(self, state: Any) -> Any:
        """
        Pass the state through each plugin in sequence.
        """
        for plugin in self.plugins:
            state = plugin.process_state(state)
        return state

    def gather_plugin_results(self) -> Dict[str, Any]:
        """
        Collect any final results from all plugins.
        """
        results = {}
        for plugin in self.plugins:
            results[plugin.name()] = plugin.get_results()
        return results 