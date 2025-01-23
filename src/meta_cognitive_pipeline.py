import os
import asyncio
from typing import Dict, Any, Optional, List
import torch
from .plugins.base_cognitive_engine import BaseCognitiveEngine
from .plugin_manager import PluginManager


class MetaCognitivePipeline:
    """Meta-Cognitive Pipeline for orchestrating cognitive engines."""
    
    def __init__(self, plugin_dir: str, config: Dict[str, Any] = None):
        """Initialize the pipeline."""
        # Load plugin manager
        self.plugin_manager = PluginManager(plugin_dir, config or {})
        self.plugin_manager.load_plugins()
        
        self.cognitive_engines: Dict[str, BaseCognitiveEngine] = {}
        self.echoic_heuristics = []
        self.truth_candidates = []
        self.results = {}

    async def initialize(self, config: Dict[str, Dict[str, Any]]) -> None:
        """Initialize cognitive engines with configuration."""
        # Import cognitive engines
        for filename in os.listdir(self.plugins_dir):
            if (
                filename.endswith('_cognitive_engine.py') and
                not filename.startswith('base_')
            ):
                module_name = filename[:-3]  # Remove .py
                try:
                    # Import the module
                    module = __import__(
                        f'src.plugins.{module_name}',
                        fromlist=['*']
                    )
                    
                    # Get the engine class
                    class_name = ''.join(
                        word.capitalize()
                        for word in module_name.split('_')
                    )
                    engine_class = getattr(module, class_name)
                    
                    # Create engine instance
                    engine = engine_class()
                    
                    # Initialize with config if available
                    if engine.__class__.__name__ in config:
                        await engine.initialize(
                            config[engine.__class__.__name__]
                        )
                    
                    # Add to engines dict
                    self.cognitive_engines[engine.__class__.__name__] = engine
                    print(f"Loaded cognitive engine: {engine.__class__.__name__}")
                    
                except Exception as e:
                    print(f"Error loading plugin {module_name}: {str(e)}")
        
        print(
            f"Initialized {len(self.cognitive_engines)} cognitive engines: "
            f"{list(self.cognitive_engines.keys())}"
        )

    async def process_input(
        self,
        input_data: Dict[str, Any],
        consciousness_field: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Process input through cognitive engines."""
        results = {}
        
        # Process through each engine
        for engine_name, engine in self.cognitive_engines.items():
            try:
                # Convert input to quantum state
                quantum_state = self._prepare_quantum_state(
                    input_data,
                    engine.modality
                )
                
                # Process through engine
                engine_results = await engine.execute(
                    quantum_state,
                    consciousness_field
                )
                results[engine_name] = engine_results
                
            except Exception as e:
                print(f"Error in {engine_name}: {str(e)}")
        
        # Find contradictions
        contradictions = self._find_contradictions()
        if contradictions:
            results['contradictions'] = contradictions
        
        # Update resonances
        self._update_resonances(results)
        
        # Store results
        self.results = results
        
        return results

    def _prepare_quantum_state(
        self,
        input_data: Dict[str, Any],
        modality: str
    ) -> torch.Tensor:
        """Prepare quantum state for specific modality."""
        if modality == 'language' and 'text' in input_data:
            # Convert text to tensor
            return torch.tensor(
                [ord(c) for c in input_data['text']],
                dtype=torch.float32
            ).unsqueeze(0)
            
        elif modality == 'numerical' and 'numerical' in input_data:
            # Convert numerical data to tensor
            data = input_data['numerical']['time_series']
            return torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            
        elif modality == 'visual' and 'visual' in input_data:
            # Use visual tensor directly
            return input_data['visual']
            
        elif modality == 'audio' and 'audio' in input_data:
            # Use audio tensor directly
            return input_data['audio']
            
        else:
            raise ValueError(f"No input data for modality: {modality}")

    def _find_contradictions(self) -> Dict[str, Any]:
        """Find contradictions between nodes."""
        contradictions = {}
        
        # Check each pair of nodes
        for engine1 in self.cognitive_engines.values():
            for context1 in engine1.ephemeral_contexts.values():
                for node1 in context1.nodes:
                    for engine2 in self.cognitive_engines.values():
                        for context2 in engine2.ephemeral_contexts.values():
                            for node2 in context2.nodes:
                                if node1.id != node2.id:
                                    # Check for contradiction
                                    if engine1._are_contradictory(
                                        node1,
                                        node2
                                    ):
                                        contr_id = f"contr_{len(contradictions)}"
                                        contradictions[contr_id] = {
                                            'node1': node1.id,
                                            'node2': node2.id,
                                            'engine1': engine1.__class__.__name__,
                                            'engine2': engine2.__class__.__name__,
                                            'status': 'unresolved',
                                            'resolution_method': None,
                                            'confidence': max(
                                                node1.confidence,
                                                node2.confidence
                                            ),
                                            'reasoning': (
                                                "Contradictory patterns "
                                                "detected in nodes"
                                            )
                                        }
        
        return contradictions

    def _update_resonances(self, results: Dict[str, Any]) -> None:
        """Update resonances between engines."""
        self.echoic_heuristics = []
        self.truth_candidates = []
        
        # Check each pair of engines
        for src_name, src_engine in self.cognitive_engines.items():
            for src_context in src_engine.ephemeral_contexts.values():
                for src_node in src_context.nodes:
                    if src_node.confidence > 0.5:
                        resonances = {}
                        
                        # Check resonance with other engines
                        for tgt_name, tgt_engine in self.cognitive_engines.items():
                            if src_name != tgt_name:
                                for tgt_context in tgt_engine.ephemeral_contexts.values():
                                    for tgt_node in tgt_context.nodes:
                                        # Calculate resonance
                                        similarity = self._calculate_resonance(
                                            src_node,
                                            tgt_node,
                                            src_engine,
                                            tgt_engine
                                        )
                                        if similarity > 0.5:
                                            resonances[tgt_name] = similarity
                        
                        if resonances:
                            # Create echo
                            echo = type(
                                'Echo',
                                (),
                                {
                                    'source_engine': src_name,
                                    'source_id': src_node.id,
                                    'confidence': src_node.confidence,
                                    'resonances': resonances,
                                    'is_truth_candidate': (
                                        len(resonances) > 1 and
                                        all(r > 0.7 for r in resonances.values())
                                    )
                                }
                            )
                            
                            self.echoic_heuristics.append(echo)
                            
                            # Check if truth candidate
                            if echo.is_truth_candidate:
                                self.truth_candidates.append(echo)

    def _calculate_resonance(
        self,
        node1: Any,
        node2: Any,
        engine1: BaseCognitiveEngine,
        engine2: BaseCognitiveEngine
    ) -> float:
        """Calculate resonance between two nodes."""
        try:
            # Get embeddings
            emb1 = engine1.get_node_embedding(node1)
            emb2 = engine2.get_node_embedding(node2)
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                emb1.view(1, -1),
                emb2.view(1, -1)
            ).item()
            
            return max(0.0, similarity)
            
        except Exception as e:
            print(f"Error calculating resonance: {str(e)}")
            return 0.0 

    async def run_pipeline(self, initial_state: dict) -> dict:
        """
        This is a high-level method that orchestrates:
        1. Data ingestion
        2. Processing through quantum-layers
        3. Passing to loaded plugins
        4. Collecting final results
        """
        # Example: initial transformations or quantum ops
        # ...
        
        # Pass the state through plugin manager
        updated_state = self.plugin_manager.process_state_through_plugins(initial_state)
        
        # Gather plugin results
        plugin_results = self.plugin_manager.gather_plugin_results()
        
        # Potentially use these results in further steps
        # ...
        
        # Return the final pipeline outputs
        return {
            "updated_state": updated_state,
            "plugin_results": plugin_results
        } 