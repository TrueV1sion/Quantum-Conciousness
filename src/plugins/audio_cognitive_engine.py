import torch
import torch.nn.functional as F
import torchaudio
from typing import Any, Dict, Optional, List, Tuple, cast
from .base_cognitive_engine import BaseCognitiveEngine, ContextNode


class AudioCognitiveEngine(BaseCognitiveEngine):
    """
    Cognitive engine specialized for audio processing.
    Uses wavelet transforms and spectral analysis for feature extraction.
    """
    
    def __init__(self):
        super().__init__()
        self.modality = "audio"
        self.model = None
        self.mel_transform = None
        self.wavelet_transform = None
        self.feature_dim = 512

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize audio processing components."""
        super().initialize(config)
        
        # Set up mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.get("sample_rate", 16000),
            n_fft=config.get("n_fft", 400),
            n_mels=config.get("n_mels", 128),
            hop_length=config.get("hop_length", 160)
        ).to(self.device)
        
        # Initialize feature extraction network
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 16, self.feature_dim)
        ).to(self.device)

    def execute(
        self,
        quantum_state: torch.Tensor,
        consciousness_field: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Process audio information from quantum state."""
        try:
            # Convert quantum state to audio representation
            audio_input = self._quantum_to_audio(quantum_state)
            
            # Extract spectral features
            mel_spec = self.mel_transform(audio_input)
            
            # Find significant audio events
            nodes = self._find_audio_events(mel_spec)
            
            # Connect related audio events
            if len(nodes) > 1:
                self._connect_audio_events(nodes)
            
            # Create ephemeral context
            context_id = f"audio_context_{len(self.ephemeral_contexts)}"
            self.create_ephemeral_context(context_id)
            
            # Add nodes to context
            for node in nodes:
                self.add_to_ephemeral_context(context_id, node)
            
            return {
                'context_id': context_id,
                'nodes': nodes,
                'mel_spectrogram': mel_spec.cpu()
            }
            
        except Exception as e:
            raise RuntimeError(f"Audio processing failed: {str(e)}")

    def get_node_embedding(self, node: ContextNode) -> torch.Tensor:
        """Get embedding for audio event node."""
        if isinstance(node.content, torch.Tensor):
            return self.model(node.content.unsqueeze(0)).squeeze()
        else:
            raise ValueError(f"Unsupported content type: {type(node.content)}")

    def _are_contradictory(
        self,
        node1: ContextNode,
        node2: ContextNode
    ) -> bool:
        """Detect contradictions in audio events."""
        # Get event embeddings
        emb1 = self.get_node_embedding(node1)
        emb2 = self.get_node_embedding(node2)
        
        # Calculate cosine similarity
        similarity = F.cosine_similarity(
            emb1.view(1, -1),
            emb2.view(1, -1)
        ).item()
        
        # Check temporal overlap
        time1 = node1.metadata['time_range']
        time2 = node2.metadata['time_range']
        overlapping = (
            time1[0] < time2[1] and
            time2[0] < time1[1]
        )
        
        # Events that overlap in time but have very different characteristics
        return overlapping and similarity < -0.3

    def _quantum_to_audio(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Convert quantum state to audio representation."""
        batch_size = quantum_state.shape[0]
        
        # Reshape quantum state to audio waveform
        # Assuming quantum state can be reshaped to [B, T]
        return quantum_state.view(batch_size, -1)

    def _find_audio_events(
        self,
        mel_spec: torch.Tensor
    ) -> List[ContextNode]:
        """Find significant audio events in mel spectrogram."""
        nodes = []
        
        # Calculate energy in each time-frequency bin
        energy = torch.norm(mel_spec, dim=1, keepdim=True)
        
        # Find peaks in energy
        peaks = self._find_spectral_peaks(energy)
        
        # Create nodes for significant peaks
        for i, (time_idx, freq_idx, value) in enumerate(peaks):
            if value > 0.5:  # Significance threshold
                # Extract spectrogram patch around peak
                patch = self._extract_spec_patch(
                    mel_spec,
                    time_idx,
                    freq_idx
                )
                
                # Calculate time range
                time_range = (
                    time_idx / mel_spec.shape[2],
                    (time_idx + 1) / mel_spec.shape[2]
                )
                
                # Create node
                node = self.create_context_node(
                    content=patch,
                    confidence=value.item(),
                    metadata={
                        'time_range': time_range,
                        'frequency_bin': freq_idx,
                        'event_index': i
                    }
                )
                nodes.append(node)
        
        return nodes

    def _find_spectral_peaks(
        self,
        energy: torch.Tensor
    ) -> List[Tuple[int, int, torch.Tensor]]:
        """Find peaks in spectral energy."""
        peaks = []
        
        # Get dimensions
        B, C, T, F = energy.shape
        
        # Pad energy map
        padded = F.pad(energy, (1, 1, 1, 1), mode='replicate')
        
        # Find local maxima
        for t in range(1, T + 1):
            for f in range(1, F + 1):
                patch = padded[:, :, t-1:t+2, f-1:f+2]
                center = padded[:, :, t, f]
                
                if (center >= patch).all():
                    peaks.append((t-1, f-1, center))
        
        return peaks

    def _extract_spec_patch(
        self,
        spec: torch.Tensor,
        time_idx: int,
        freq_idx: int,
        size: int = 5
    ) -> torch.Tensor:
        """Extract spectrogram patch around peak."""
        T, F = spec.shape[2:]
        
        # Calculate patch boundaries
        t_start = max(0, time_idx - size // 2)
        t_end = min(T, time_idx + size // 2 + 1)
        f_start = max(0, freq_idx - size // 2)
        f_end = min(F, freq_idx + size // 2 + 1)
        
        return spec[:, :, t_start:t_end, f_start:f_end]

    def _connect_audio_events(self, nodes: List[ContextNode]) -> None:
        """Connect related audio events."""
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Get event embeddings
                emb1 = self.get_node_embedding(nodes[i])
                emb2 = self.get_node_embedding(nodes[j])
                
                # Calculate similarity
                similarity = F.cosine_similarity(
                    emb1.view(1, -1),
                    emb2.view(1, -1)
                ).item()
                
                # Check temporal proximity
                time1 = nodes[i].metadata['time_range']
                time2 = nodes[j].metadata['time_range']
                time_dist = min(
                    abs(time1[1] - time2[0]),
                    abs(time2[1] - time1[0])
                )
                
                # Connect events that are similar and temporally close
                if similarity > 0.7 and time_dist < 0.1:
                    self.connect_nodes(nodes[i].id, nodes[j].id) 