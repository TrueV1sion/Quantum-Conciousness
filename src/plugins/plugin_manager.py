import os
import importlib.util
import sys
from typing import Dict, Any, Optional
import torch
import logging
from .base_plugin import BasePlugin

class QuantumPluginManager:
    """
    Manages quantum consciousness plugins.
    """

    def __init__(self, plugins_directory: str):
        self.plugins_directory = plugins_directory
        self.plugins: Dict[str, BasePlugin] = {}
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def discover_plugins(self) -> None:
        """Discover all quantum consciousness plugins in the plugins directory."""
        if not os.path.exists(self.plugins_directory):
            os.makedirs(self.plugins_directory)
            self.logger.info(f"Created plugins directory: {self.plugins_directory}")
            return

        for filename in os.listdir(self.plugins_directory):
            if filename.endswith(".py") and not filename.startswith("__"):
                plugin_path = os.path.join(self.plugins_directory, filename)
                plugin_name = filename[:-3]
                self.load_plugin(plugin_name, plugin_path)

    def load_plugin(self, plugin_name: str, plugin_path: str) -> None:
        """Load a single quantum plugin given its name and path."""
        try:
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[plugin_name] = module
                spec.loader.exec_module(module)

                # Identify classes that inherit from BasePlugin
                for attribute_name in dir(module):
                    attribute = getattr(module, attribute_name)
                    if (
                        isinstance(attribute, type)
                        and issubclass(attribute, BasePlugin)
                        and attribute is not BasePlugin
                    ):
                        plugin_instance: BasePlugin = attribute()
                        self.plugins[plugin_instance.name()] = plugin_instance
                        self.logger.info(
                            f"Loaded quantum plugin: {plugin_instance.name()} "
                            f"v{plugin_instance.version()}"
                        )
        except Exception as e:
            self.logger.error(f"Error loading plugin {plugin_name}: {str(e)}")

    def initialize_plugins(self, config: Dict[str, Any]) -> None:
        """Initialize all loaded plugins with the provided configuration."""
        for plugin in self.plugins.values():
            try:
                plugin.initialize(config.get(plugin.name(), {}))
                self.logger.info(f"Initialized plugin: {plugin.name()}")
            except Exception as e:
                self.logger.error(
                    f"Error initializing plugin {plugin.name()}: {str(e)}"
                )

    def execute_plugin(
        self,
        plugin_name: str,
        quantum_state: torch.Tensor,
        consciousness_field: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Execute a specific quantum plugin by name."""
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin '{plugin_name}' not found.")

        try:
            # Move tensors to appropriate device
            quantum_state = quantum_state.to(self.device)
            if consciousness_field is not None:
                consciousness_field = consciousness_field.to(self.device)

            # Execute plugin pipeline
            processed_state = plugin.pre_process(quantum_state)
            results = plugin.execute(processed_state, consciousness_field)
            final_state = plugin.post_process(processed_state)

            # Add processed state to results
            results['processed_state'] = final_state
            return results

        except Exception as e:
            self.logger.error(
                f"Error executing plugin {plugin_name}: {str(e)}"
            )
            raise

    def get_plugin_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about all loaded plugins."""
        return {
            name: {
                'version': plugin.version(),
                'type': plugin.__class__.__name__
            }
            for name, plugin in self.plugins.items()
        } 