import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, List, Tuple, cast
from torchvision import models, transforms
from .base_cognitive_engine import BaseCognitiveEngine, ContextNode


class VisualCognitiveEngine(BaseCognitiveEngine):
    """
    Cognitive engine specialized for visual processing.
    Uses pre-trained vision models for feature extraction.
    """
    
    def __init__(self):
        super().__init__()
        self.modality = "visual"
        self.model = None
        self.feature_extractors = {}
        self.transform = None
        self.feature_dim = 2048  # ResNet feature dimension

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize visual processing components."""
        super().initialize(config)
        
        # Load pre-trained model
        model_name = config.get("model_name", "resnet50")
        self.model = getattr(models, model_name)(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Remove classification layer to get features
        self.feature_extractors = {
            'low_level': self.model.layer1,
            'mid_level': self.model.layer2,
            'high_level': self.model.layer3,
            'semantic': self.model.layer4
        }
        
        # Set up image transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _normalize_features(
        self,
        features: torch.Tensor,
        dims: List[int]
    ) -> torch.Tensor:
        """Normalize features along specified dimensions."""
        return cast(
            torch.Tensor,
            F.normalize(features, dim=dims[0])
        )

    def execute(
        self,
        quantum_state: torch.Tensor,
        consciousness_field: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Process visual information from quantum state."""
        try:
            # Convert quantum state to visual representation
            visual_input = self._quantum_to_visual(quantum_state)
            
            # Extract features at different levels
            nodes = []
            features = {}
            
            with torch.no_grad():
                x = visual_input
                for level, extractor in self.feature_extractors.items():
                    # Extract features
                    x = extractor(x)
                    features[level] = x
                    
                    # Create nodes for significant features
                    significant_features = self._find_significant_features(
                        x,
                        level
                    )
                    nodes.extend(significant_features)
            
            # Connect related features across levels
            if len(nodes) > 1:
                self._connect_visual_features(nodes)
            
            # Create ephemeral context
            context_id = f"visual_context_{len(self.ephemeral_contexts)}"
            self.create_ephemeral_context(context_id)
            
            # Add nodes to context
            for node in nodes:
                self.add_to_ephemeral_context(context_id, node)
            
            return {
                'context_id': context_id,
                'nodes': nodes,
                'features': {
                    k: v.cpu() for k, v in features.items()
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Visual processing failed: {str(e)}")

    def get_node_embedding(self, node: ContextNode) -> torch.Tensor:
        """Get embedding for visual feature node."""
        if isinstance(node.content, torch.Tensor):
            # Average pooling to get fixed-size representation
            return F.adaptive_avg_pool2d(
                node.content.to(self.device),
                (1, 1)
            ).squeeze()
        else:
            raise ValueError(f"Unsupported content type: {type(node.content)}")

    def _are_contradictory(
        self,
        node1: ContextNode,
        node2: ContextNode
    ) -> bool:
        """Detect contradictions in visual features."""
        # Get feature embeddings
        emb1 = self.get_node_embedding(node1)
        emb2 = self.get_node_embedding(node2)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(
            emb1.view(1, -1),
            emb2.view(1, -1)
        ).item()
        
        # Check if features are from same level
        same_level = (
            node1.metadata['level'] == node2.metadata['level']
        )
        
        # Features from same level with very different characteristics
        return (
            same_level
            and node1.confidence > 0.7
            and node2.confidence > 0.7
            and similarity < -0.3
        )

    def _quantum_to_visual(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Convert quantum state to visual representation."""
        batch_size = quantum_state.shape[0]
        
        # Reshape quantum state to image-like format
        # Assuming quantum state can be reshaped to [B, C, H, W]
        visual = quantum_state.view(batch_size, 3, 224, 224)
        
        # Apply normalization
        return self.transform(visual)

    def _find_significant_features(
        self,
        features: torch.Tensor,
        level: str
    ) -> List[ContextNode]:
        """Find significant features at given level."""
        nodes = []
        
        # Calculate feature importance
        importance = torch.norm(features, dim=1, keepdim=True)
        importance = self._normalize_features(importance, dims=[2, 3])
        
        # Find peaks in feature maps
        peaks = self._find_local_peaks(importance)
        
        # Create nodes for significant peaks
        for i, (pos, value) in enumerate(peaks):
            if value > 0.5:  # Significance threshold
                # Extract feature patch around peak
                patch = self._extract_feature_patch(features, pos)
                
                # Create node
                node = self.create_context_node(
                    content=patch,
                    confidence=value.item(),
                    metadata={
                        'level': level,
                        'position': pos,
                        'feature_index': i
                    }
                )
                nodes.append(node)
        
        return nodes

    def _find_local_peaks(
        self,
        feature_map: torch.Tensor
    ) -> List[Tuple[Tuple[int, int], torch.Tensor]]:
        """Find local peaks in feature map."""
        peaks = []
        
        # Get feature map dimensions
        B, C, H, W = feature_map.shape
        
        # Pad feature map for edge detection
        padded = F.pad(feature_map, (1, 1, 1, 1), mode='replicate')
        
        # Find local maxima
        for i in range(1, H + 1):
            for j in range(1, W + 1):
                patch = padded[:, :, i-1:i+2, j-1:j+2]
                center = padded[:, :, i, j]
                
                if (center >= patch).all():
                    peaks.append(((i-1, j-1), center))
        
        return peaks

    def _extract_feature_patch(
        self,
        features: torch.Tensor,
        position: Tuple[int, int],
        size: int = 3
    ) -> torch.Tensor:
        """Extract feature patch around given position."""
        i, j = position
        H, W = features.shape[2:]
        
        # Calculate patch boundaries
        top = max(0, i - size // 2)
        left = max(0, j - size // 2)
        bottom = min(H, i + size // 2 + 1)
        right = min(W, j + size // 2 + 1)
        
        return features[:, :, top:bottom, left:right]

    def _connect_visual_features(self, nodes: List[ContextNode]) -> None:
        """Connect related visual features."""
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Get feature embeddings
                emb1 = self.get_node_embedding(nodes[i])
                emb2 = self.get_node_embedding(nodes[j])
                
                # Calculate similarity
                similarity = F.cosine_similarity(
                    emb1.view(1, -1),
                    emb2.view(1, -1)
                ).item()
                
                # Check spatial relationship
                pos1 = nodes[i].metadata['position']
                pos2 = nodes[j].metadata['position']
                
                # Calculate Euclidean distance between feature positions
                spatial_dist = (
                    (pos1[0] - pos2[0]) ** 2 +
                    (pos1[1] - pos2[1]) ** 2
                ) ** 0.5
                
                # Connect features that are similar and spatially close
                if similarity > 0.7 and spatial_dist < 5:
                    self.connect_nodes(nodes[i].id, nodes[j].id) 