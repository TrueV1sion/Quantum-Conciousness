import torch
import numpy as np
from typing import Any, Dict, Optional, List
from .base_cognitive_engine import BaseCognitiveEngine, ContextNode


class NumericalCognitiveEngine(BaseCognitiveEngine):
    """
    Cognitive engine specialized for numerical analysis.
    Detects patterns, correlations, and anomalies in numerical data.
    """
    
    def __init__(self):
        super().__init__()
        self.modality = "numerical"
        self.embedding_dim = 256
        self.pattern_detectors = {
            'linear': self._detect_linear_patterns,
            'periodic': self._detect_periodic_patterns,
            'cluster': self._detect_clusters,
            'anomaly': self._detect_anomalies
        }

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize numerical analysis components."""
        super().initialize(config)
        self.embedding_dim = config.get("embedding_dim", 256)
        
        # Initialize neural network layers for pattern detection
        self.pattern_network = torch.nn.Sequential(
            torch.nn.Linear(1024, self.embedding_dim),
            torch.nn.LayerNorm(self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            torch.nn.LayerNorm(self.embedding_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_dim // 2, self.embedding_dim // 4)
        ).to(self.device)

    def execute(
        self,
        quantum_state: torch.Tensor,
        consciousness_field: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Analyze numerical patterns in quantum state."""
        try:
            # Project quantum state to numerical space
            numerical_repr = self.pre_process(quantum_state)
            
            # Detect patterns using each detector
            nodes = []
            for pattern_type, detector in self.pattern_detectors.items():
                patterns = detector(numerical_repr)
                for i, (pattern, confidence) in enumerate(patterns):
                    node = self.create_context_node(
                        content=pattern,
                        confidence=confidence,
                        metadata={
                            'pattern_type': pattern_type,
                            'index': i
                        }
                    )
                    nodes.append(node)
            
            # Connect related patterns
            if len(nodes) > 1:
                self._connect_related_patterns(nodes)
            
            # Create ephemeral context
            context_id = f"numerical_context_{len(self.ephemeral_contexts)}"
            self.create_ephemeral_context(context_id)
            
            # Add nodes to context
            for node in nodes:
                self.add_to_ephemeral_context(context_id, node)
            
            return {
                'context_id': context_id,
                'nodes': nodes,
                'patterns_found': len(nodes)
            }
            
        except Exception as e:
            raise RuntimeError(f"Numerical analysis failed: {str(e)}")

    def get_node_embedding(self, node: ContextNode) -> torch.Tensor:
        """Get embedding for numerical pattern node."""
        if isinstance(node.content, torch.Tensor):
            return self.pattern_network(node.content.to(self.device))
        elif isinstance(node.content, (list, np.ndarray)):
            tensor = torch.tensor(node.content, device=self.device)
            return self.pattern_network(tensor)
        else:
            raise ValueError(f"Unsupported content type: {type(node.content)}")

    def _are_contradictory(
        self,
        node1: ContextNode,
        node2: ContextNode
    ) -> bool:
        """Detect contradictions in numerical patterns."""
        # Get pattern embeddings
        emb1 = self.get_node_embedding(node1)
        emb2 = self.get_node_embedding(node2)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            emb1.view(1, -1),
            emb2.view(1, -1)
        ).item()
        
        # Check if patterns are mutually exclusive
        same_type = (
            node1.metadata['pattern_type'] == node2.metadata['pattern_type']
        )
        
        # Patterns of same type with opposite characteristics
        return (
            same_type
            and node1.confidence > 0.7
            and node2.confidence > 0.7
            and similarity < -0.5
        )

    def _detect_linear_patterns(
        self,
        data: torch.Tensor
    ) -> List[tuple[torch.Tensor, float]]:
        """Detect linear relationships in data."""
        patterns = []
        
        try:
            # Reshape data for analysis
            flat_data = data.view(-1, data.shape[-1])
            
            # Calculate correlation matrix
            corr_matrix = torch.corrcoef(flat_data.t())
            
            # Find strong linear correlations
            strong_corr = torch.where(
                torch.abs(corr_matrix) > 0.7,
                corr_matrix,
                torch.zeros_like(corr_matrix)
            )
            
            # Extract significant patterns
            values, indices = torch.topk(
                strong_corr.abs().view(-1),
                k=min(5, strong_corr.numel())
            )
            
            for val, idx in zip(values, indices):
                if val > 0:  # Ignore zero correlations
                    row = idx // strong_corr.shape[1]
                    col = idx % strong_corr.shape[1]
                    if row != col:  # Ignore self-correlations
                        pattern = torch.stack([
                            flat_data[:, row],
                            flat_data[:, col]
                        ])
                        patterns.append((pattern, val.item()))
            
        except Exception as e:
            print(f"Error in linear pattern detection: {str(e)}")
        
        return patterns

    def _detect_periodic_patterns(
        self,
        data: torch.Tensor
    ) -> List[tuple[torch.Tensor, float]]:
        """Detect periodic patterns in data."""
        patterns = []
        
        try:
            # Reshape data for analysis
            flat_data = data.view(-1, data.shape[-1])
            
            # For each feature
            for i in range(flat_data.shape[1]):
                signal = flat_data[:, i]
                
                # Compute FFT
                fft = torch.fft.fft(signal)
                freqs = torch.fft.fftfreq(len(signal))
                
                # Find dominant frequencies
                amplitudes = torch.abs(fft)
                peaks = torch.where(
                    amplitudes > 0.7 * amplitudes.max(),
                    amplitudes,
                    torch.zeros_like(amplitudes)
                )
                
                # Extract periodic components
                for j, peak in enumerate(peaks):
                    if peak > 0:
                        # Calculate frequency for metadata
                        freq = freqs[j] if freqs[j] != 0 else float('inf')
                        confidence = peak / amplitudes.max()
                        pattern = torch.stack([signal, freqs])
                        patterns.append((
                            pattern,
                            confidence.item(),
                            {'frequency': freq}
                        ))
            
        except Exception as e:
            print(f"Error in periodic pattern detection: {str(e)}")
        
        return patterns

    def _detect_clusters(
        self,
        data: torch.Tensor
    ) -> List[tuple[torch.Tensor, float]]:
        """Detect clusters in data."""
        patterns = []
        
        try:
            # Reshape data for clustering
            flat_data = data.view(-1, data.shape[-1])
            
            # Simple k-means-like clustering
            n_clusters = min(5, flat_data.shape[0])
            
            # Randomly initialize centroids
            centroids = flat_data[
                torch.randperm(flat_data.shape[0])[:n_clusters]
            ]
            
            # Iterate a few times
            for _ in range(3):
                # Assign points to nearest centroid
                distances = torch.cdist(flat_data, centroids)
                assignments = torch.argmin(distances, dim=1)
                
                # Update centroids
                for k in range(n_clusters):
                    cluster_points = flat_data[assignments == k]
                    if len(cluster_points) > 0:
                        centroids[k] = cluster_points.mean(dim=0)
            
            # Calculate cluster qualities
            for k in range(n_clusters):
                cluster_points = flat_data[assignments == k]
                if len(cluster_points) > 0:
                    # Calculate cluster cohesion
                    distances = torch.cdist(
                        cluster_points,
                        cluster_points.mean(dim=0).unsqueeze(0)
                    )
                    cohesion = 1 / (1 + distances.mean())
                    patterns.append((cluster_points, cohesion.item()))
            
        except Exception as e:
            print(f"Error in cluster detection: {str(e)}")
        
        return patterns

    def _detect_anomalies(
        self,
        data: torch.Tensor
    ) -> List[tuple[torch.Tensor, float]]:
        """Detect anomalies in data."""
        patterns = []
        
        try:
            # Reshape data
            flat_data = data.view(-1, data.shape[-1])
            
            # Calculate z-scores
            mean = flat_data.mean(dim=0)
            std = flat_data.std(dim=0)
            z_scores = (flat_data - mean) / (std + 1e-8)
            
            # Find anomalies (|z| > 2)
            anomalies = torch.abs(z_scores) > 2
            
            # Group consecutive anomalies
            for i in range(flat_data.shape[1]):
                feature_anomalies = anomalies[:, i]
                if feature_anomalies.any():
                    # Extract anomalous segments
                    segments = []
                    start = None
                    
                    for j in range(len(feature_anomalies)):
                        if feature_anomalies[j]:
                            if start is None:
                                start = j
                        elif start is not None:
                            segments.append((start, j))
                            start = None
                    
                    if start is not None:
                        segments.append((start, len(feature_anomalies)))
                    
                    # Create pattern for each anomalous segment
                    for start, end in segments:
                        segment = flat_data[start:end, i]
                        z_score = z_scores[start:end, i].abs().mean()
                        confidence = torch.tanh(z_score).item()
                        patterns.append((segment, confidence))
            
        except Exception as e:
            print(f"Error in anomaly detection: {str(e)}")
        
        return patterns

    def _connect_related_patterns(self, nodes: List[ContextNode]) -> None:
        """Connect related numerical patterns."""
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Get pattern embeddings
                emb1 = self.get_node_embedding(nodes[i])
                emb2 = self.get_node_embedding(nodes[j])
                
                # Calculate similarity
                similarity = torch.nn.functional.cosine_similarity(
                    emb1.view(1, -1),
                    emb2.view(1, -1)
                ).item()
                
                # Connect highly similar patterns
                if similarity > 0.7:
                    self.connect_nodes(nodes[i].id, nodes[j].id) 