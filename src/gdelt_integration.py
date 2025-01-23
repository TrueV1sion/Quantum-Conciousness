import os
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto

from .processors import StateProcessor
from .config import SystemConfig, UnifiedState
from .quantum_bridge_google import GoogleQuantumBridge

@dataclass
class GDELTConfig:
    """Configuration for GDELT processing."""
    update_interval: int = 15  # minutes
    event_window: int = 24     # hours
    batch_size: int = 100
    embedding_dim: int = 512
    tone_threshold: float = 0.1
    quantum_encoding_scheme: str = "amplitude"
    cache_duration: int = 60   # minutes

class EventType(Enum):
    """GDELT event categories of interest."""
    POLITICAL = auto()
    ECONOMIC = auto()
    CONFLICT = auto()
    DIPLOMATIC = auto()
    FINANCIAL = auto()

@dataclass
class GDELTEvent:
    """Structured GDELT event data."""
    event_id: str
    timestamp: datetime
    event_type: EventType
    actor1: str
    actor2: str
    action: str
    location: Tuple[float, float]  # lat, lon
    tone: float
    relevance: float
    impact_score: float

class GDELTQuantumEncoder:
    """Encodes GDELT events into quantum states."""
    
    def __init__(self, config: GDELTConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    async def encode_events(self, events: List[GDELTEvent]) -> torch.Tensor:
        """Encode GDELT events into quantum states."""
        # Create embeddings for each event component
        event_tensors = []
        for event in events:
            # Encode event attributes
            tone_encoding = torch.tensor([event.tone], device=self.device)
            impact_encoding = torch.tensor([event.impact_score], device=self.device)
            
            # Create location quantum state
            location_encoding = self._encode_location(event.location)
            
            # Combine encodings into quantum state
            event_state = self._prepare_quantum_state(
                tone_encoding,
                impact_encoding,
                location_encoding
            )
            event_tensors.append(event_state)
            
        return torch.stack(event_tensors)
    
    def _encode_location(self, location: Tuple[float, float]) -> torch.Tensor:
        """Encode geographical location into quantum state."""
        lat, lon = location
        # Convert to radians and normalize
        lat_rad = torch.tensor([np.pi * lat / 180.0], device=self.device)
        lon_rad = torch.tensor([np.pi * lon / 180.0], device=self.device)
        return torch.cat([torch.cos(lat_rad), torch.sin(lat_rad),
                         torch.cos(lon_rad), torch.sin(lon_rad)])
    
    def _prepare_quantum_state(self, 
                             tone: torch.Tensor,
                             impact: torch.Tensor,
                             location: torch.Tensor) -> torch.Tensor:
        """Prepare normalized quantum state from encodings."""
        combined = torch.cat([tone, impact, location])
        return combined / torch.norm(combined)

class GDELTDataManager:
    """Manages GDELT data retrieval and processing."""
    
    def __init__(self, config: GDELTConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, List[GDELTEvent]] = {}
        self.last_update: Optional[datetime] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_latest_events(self) -> List[GDELTEvent]:
        """Retrieve latest GDELT events."""
        current_time = datetime.utcnow()
        if (self.last_update and 
            (current_time - self.last_update).total_seconds() < self.config.update_interval * 60):
            return self.cache.get('latest', [])
        
        async with self.session.get(self._build_gdelt_url()) as response:
            data = await response.json()
            events = self._parse_gdelt_response(data)
            self.cache['latest'] = events
            self.last_update = current_time
            return events
    
    def _build_gdelt_url(self) -> str:
        """Build GDELT API URL for data retrieval."""
        base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            'format': 'json',
            'maxrows': self.config.batch_size,
            'timespan': f"{self.config.event_window}h"
        }
        return f"{base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
    
    def _parse_gdelt_response(self, data: Dict) -> List[GDELTEvent]:
        """Parse GDELT API response into structured events."""
        events = []
        for item in data.get('articles', []):
            event = GDELTEvent(
                event_id=item.get('globaleventid'),
                timestamp=datetime.fromtimestamp(item.get('dateadded')),
                event_type=self._classify_event_type(item),
                actor1=item.get('actor1name'),
                actor2=item.get('actor2name'),
                action=item.get('eventcode'),
                location=(
                    float(item.get('actiongeolat', 0)),
                    float(item.get('actiongeolong', 0))
                ),
                tone=float(item.get('avgtone', 0)),
                relevance=float(item.get('relevance', 0)),
                impact_score=self._calculate_impact_score(item)
            )
            events.append(event)
        return events
    
    def _classify_event_type(self, item: Dict) -> EventType:
        """Classify GDELT event into relevant category."""
        event_code = str(item.get('eventcode', ''))
        if event_code.startswith('1'):
            return EventType.POLITICAL
        elif event_code.startswith('2'):
            return EventType.ECONOMIC
        elif event_code.startswith('3'):
            return EventType.CONFLICT
        elif event_code.startswith('4'):
            return EventType.DIPLOMATIC
        else:
            return EventType.FINANCIAL
    
    def _calculate_impact_score(self, item: Dict) -> float:
        """Calculate event impact score based on multiple factors."""
        tone = abs(float(item.get('avgtone', 0)))
        mentions = float(item.get('numarticles', 1))
        relevance = float(item.get('relevance', 0))
        
        # Normalize and combine factors
        normalized_tone = min(tone / 10.0, 1.0)
        normalized_mentions = min(mentions / 100.0, 1.0)
        
        return (normalized_tone * 0.4 + 
                normalized_mentions * 0.3 + 
                relevance * 0.3)

class GDELTQuantumProcessor(StateProcessor):
    """Process GDELT events using quantum computing."""
    
    def __init__(self, config: GDELTConfig, quantum_bridge: GoogleQuantumBridge):
        self.config = config
        self.quantum_bridge = quantum_bridge
        self.encoder = GDELTQuantumEncoder(config)
        self.data_manager = GDELTDataManager(config)
    
    async def process_state(self, state: UnifiedState) -> UnifiedState:
        """Process GDELT events and integrate with existing state."""
        # Retrieve latest GDELT events
        events = await self.data_manager.get_latest_events()
        
        # Encode events into quantum states
        event_states = await self.encoder.encode_events(events)
        
        # Prepare quantum circuit for event processing
        circuit = await self.quantum_bridge.create_quantum_circuit(event_states)
        
        # Detect patterns and correlations
        patterns = await self.quantum_bridge.detect_resonance_patterns(
            event_states,
            state.quantum_field
        )
        
        # Update unified state with GDELT information
        updated_state = self._update_unified_state(state, patterns, events)
        return updated_state
    
    def _update_unified_state(self,
                            state: UnifiedState,
                            patterns: Dict[str, np.ndarray],
                            events: List[GDELTEvent]) -> UnifiedState:
        """Update unified state with GDELT processing results."""
        # Calculate event impact on quantum field
        event_impact = self._calculate_event_impact(events)
        
        # Update quantum field with event information
        updated_quantum_field = state.quantum_field * (1 + event_impact)
        
        # Normalize the updated field
        normalized_field = updated_quantum_field / torch.norm(updated_quantum_field)
        
        # Create updated state
        return UnifiedState(
            quantum_field=normalized_field,
            classical_state=state.classical_state,
            metadata={
                **state.metadata,
                'gdelt_patterns': patterns,
                'last_gdelt_update': datetime.utcnow().isoformat()
            }
        )
    
    def _calculate_event_impact(self, events: List[GDELTEvent]) -> torch.Tensor:
        """Calculate the impact of GDELT events on the quantum field."""
        impact_scores = torch.tensor([event.impact_score for event in events],
                                   device=self.device)
        return torch.mean(impact_scores) 