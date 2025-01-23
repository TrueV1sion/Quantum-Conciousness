from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

@dataclass
class GDELTAPIConfig:
    """GDELT API configuration."""
    base_url: str = "https://api.gdeltproject.org/api/v2"
    version: str = "2.0"
    max_retries: int = 3
    timeout: int = 30
    rate_limit: int = 60  # requests per minute

@dataclass
class EventProcessingConfig:
    """Event processing configuration."""
    min_tone_threshold: float = -10.0
    max_tone_threshold: float = 10.0
    min_mentions: int = 5
    relevance_threshold: float = 0.1
    impact_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.impact_weights is None:
            self.impact_weights = {
                'tone': 0.4,
                'mentions': 0.3,
                'relevance': 0.3
            }

@dataclass
class QuantumEncodingConfig:
    """Quantum encoding configuration."""
    encoding_scheme: str = "amplitude"  # or "phase"
    qubit_count: int = 8
    entanglement_layers: int = 2
    measurement_basis: str = "computational"
    error_correction: bool = True

@dataclass
class GDELTIntegrationConfig:
    """Main GDELT integration configuration."""
    api: GDELTAPIConfig = GDELTAPIConfig()
    processing: EventProcessingConfig = EventProcessingConfig()
    quantum: QuantumEncodingConfig = QuantumEncodingConfig()
    cache_enabled: bool = True
    cache_duration: int = 3600  # seconds
    async_batch_size: int = 100
    
    # Event type weights for financial impact
    event_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.event_weights is None:
            self.event_weights = {
                'POLITICAL': 0.7,
                'ECONOMIC': 1.0,
                'CONFLICT': 0.8,
                'DIPLOMATIC': 0.6,
                'FINANCIAL': 1.0
            } 