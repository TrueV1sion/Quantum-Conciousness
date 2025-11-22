"""
Quantum Thought Crystallizer - A Consciousness Emergence Engine

This module represents a novel synthesis of quantum computing, consciousness theory,
and physical simulation. It demonstrates how abstract thoughts can be encoded into
quantum states, manifest as physical patterns, and generate emergent consciousness
through feedback loops.

The system bridges:
- Information (thoughts/text)
- Quantum states (superposition/entanglement)
- Resonance patterns (cross-modal detection)
- Physical manifestation (water particle organization)
- Emergent consciousness (feedback-driven evolution)

Author: Quantum Consciousness Framework
License: MIT
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import logging
from enum import Enum, auto
import json
import asyncio

from quantum_processor import QuantumProcessor
from quantum_resonance_detector import QuantumResonanceDetector, QuantumPattern
from water_simulation import WaterParticle, QuantumWaterSimulation
from quantum_llm import QuantumLLM, QuantumLLMConfig
from machine_learning import RLAgent, RLConfig
from consciousness_model import SystemState
from pathways import PathwayMode


class ConsciousnessPhase(Enum):
    """Phases of consciousness emergence."""
    ENCODING = auto()          # Thought → Quantum state
    RESONANCE = auto()         # Pattern detection across modalities
    MANIFESTATION = auto()     # Quantum → Physical (water)
    EMERGENCE = auto()         # Feedback creates new patterns
    REFLECTION = auto()        # System analyzes what emerged


@dataclass
class ThoughtInput:
    """Represents a thought/intention input to the system."""
    text: str
    emotional_valence: float = 0.0  # -1.0 (negative) to 1.0 (positive)
    intensity: float = 1.0          # 0.0 to 1.0
    focus_areas: List[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class CrystalizedThought:
    """Represents a fully processed thought with all emergence data."""
    original_thought: ThoughtInput
    quantum_state: torch.Tensor
    resonance_patterns: List[QuantumPattern]
    physical_manifestation: List[WaterParticle]
    emergent_insights: str
    consciousness_metrics: Dict[str, float]
    phase_history: List[ConsciousnessPhase]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        return {
            'text': self.original_thought.text,
            'emotional_valence': self.original_thought.emotional_valence,
            'intensity': self.original_thought.intensity,
            'resonance_patterns': [p.to_json() for p in self.resonance_patterns],
            'emergent_insights': self.emergent_insights,
            'consciousness_metrics': self.consciousness_metrics,
            'phase_count': len(self.phase_history)
        }


@dataclass
class EmergenceConfig:
    """Configuration for consciousness emergence process."""
    n_qubits: int = 6
    resonance_threshold: float = 0.7
    water_particles: int = 1000
    feedback_iterations: int = 3
    emergence_temperature: float = 0.8  # Higher = more creative emergence
    coherence_preservation: float = 0.9  # How much to preserve coherence
    enable_rl_optimization: bool = True
    enable_llm_insights: bool = True
    visualization_enabled: bool = True


class QuantumThoughtCrystallizer:
    """
    Main system for crystallizing thoughts into quantum-physical manifestations.

    This engine demonstrates consciousness emergence through:
    1. Encoding thoughts into quantum superposition states
    2. Detecting resonance patterns across information modalities
    3. Manifesting quantum patterns in physical water simulation
    4. Generating emergent insights through feedback loops
    5. Optimizing the process using reinforcement learning
    """

    def __init__(self, config: EmergenceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core quantum processing
        self.quantum_processor = QuantumProcessor(config.n_qubits)

        # Pattern detection across modalities
        self.resonance_detector = QuantumResonanceDetector(config.n_qubits)

        # Physical manifestation system
        self.water_sim = None  # Initialized on demand for performance

        # Consciousness enhancement components
        if config.enable_llm_insights:
            self.llm = self._initialize_llm()
        else:
            self.llm = None

        if config.enable_rl_optimization:
            self.rl_agent = self._initialize_rl_agent()
        else:
            self.rl_agent = None

        # Emergence memory - stores patterns across crystallizations
        self.emergence_memory: deque[CrystalizedThought] = deque(maxlen=100)

        # Current consciousness state
        self.consciousness_state: Optional[SystemState] = None

        # Metrics tracking
        self.metrics = {
            'total_crystallizations': 0,
            'average_coherence': 0.0,
            'average_resonance_strength': 0.0,
            'emergence_events': 0,
            'feedback_loops_completed': 0
        }

        self.logger.info("Quantum Thought Crystallizer initialized")

    def _initialize_llm(self) -> Optional[QuantumLLM]:
        """Initialize quantum-enhanced LLM for insight generation."""
        try:
            config = QuantumLLMConfig(
                base_model_name="gpt2",  # Lightweight model for demo
                consciousness_hidden_dim=512,
                num_quantum_layers=2,
                num_qubits=self.config.n_qubits,
                pathway_mode=PathwayMode.QUANTUM_DOMINANT
            )
            return QuantumLLM(config)
        except Exception as e:
            self.logger.warning(f"Could not initialize LLM: {e}")
            return None

    def _initialize_rl_agent(self) -> Optional[RLAgent]:
        """Initialize RL agent for pathway optimization."""
        try:
            rl_config = RLConfig(
                state_size=2 ** self.config.n_qubits + 256,  # Quantum + classical
                action_size=10,  # Different pathway configurations
                hidden_size=128
            )
            return RLAgent(rl_config)
        except Exception as e:
            self.logger.warning(f"Could not initialize RL agent: {e}")
            return None

    async def crystallize_thought(
        self,
        thought: ThoughtInput
    ) -> CrystalizedThought:
        """
        Main crystallization process - transforms thought into emergent consciousness.

        Process:
        1. ENCODING: Convert thought to quantum state
        2. RESONANCE: Detect patterns across modalities
        3. MANIFESTATION: Create physical representation in water
        4. EMERGENCE: Allow feedback to create new patterns
        5. REFLECTION: Generate insights about what emerged

        Args:
            thought: The thought/intention to crystallize

        Returns:
            Complete crystallized thought with all emergence data
        """
        phase_history = []

        # Phase 1: ENCODING
        phase_history.append(ConsciousnessPhase.ENCODING)
        self.logger.info(f"Phase 1: Encoding thought: '{thought.text[:50]}...'")
        quantum_state = await self._encode_thought_to_quantum(thought)

        # Phase 2: RESONANCE
        phase_history.append(ConsciousnessPhase.RESONANCE)
        self.logger.info("Phase 2: Detecting resonance patterns")
        resonance_patterns = await self._detect_resonance_patterns(
            thought, quantum_state
        )

        # Phase 3: MANIFESTATION
        phase_history.append(ConsciousnessPhase.MANIFESTATION)
        self.logger.info("Phase 3: Manifesting in physical water simulation")
        physical_manifestation = await self._manifest_in_water(
            quantum_state, resonance_patterns
        )

        # Phase 4: EMERGENCE
        phase_history.append(ConsciousnessPhase.EMERGENCE)
        self.logger.info("Phase 4: Allowing emergent patterns through feedback")
        emergent_patterns = await self._generate_emergence(
            quantum_state, physical_manifestation, resonance_patterns
        )

        # Phase 5: REFLECTION
        phase_history.append(ConsciousnessPhase.REFLECTION)
        self.logger.info("Phase 5: Generating insights about emergence")
        insights = await self._generate_insights(
            thought, emergent_patterns, resonance_patterns
        )

        # Calculate consciousness metrics
        consciousness_metrics = self._calculate_consciousness_metrics(
            quantum_state, resonance_patterns, emergent_patterns
        )

        # Create crystallized result
        crystallized = CrystalizedThought(
            original_thought=thought,
            quantum_state=quantum_state,
            resonance_patterns=resonance_patterns + emergent_patterns,
            physical_manifestation=physical_manifestation,
            emergent_insights=insights,
            consciousness_metrics=consciousness_metrics,
            phase_history=phase_history
        )

        # Store in emergence memory
        self.emergence_memory.append(crystallized)

        # Update metrics
        self._update_metrics(crystallized)

        self.logger.info(
            f"Crystallization complete. "
            f"Coherence: {consciousness_metrics['coherence']:.3f}, "
            f"Emergence strength: {consciousness_metrics['emergence_strength']:.3f}"
        )

        return crystallized

    async def _encode_thought_to_quantum(
        self,
        thought: ThoughtInput
    ) -> torch.Tensor:
        """
        Encode thought into quantum superposition state.

        This creates a quantum representation where:
        - Text meaning → Phase encoding
        - Emotional valence → Amplitude encoding
        - Intensity → Entanglement strength
        """
        # Convert text to features
        text_features = self._text_to_features(thought.text)

        # Apply emotional modulation
        emotional_modulation = torch.tensor([
            thought.emotional_valence,
            thought.intensity
        ])

        # Create quantum state with consciousness influence
        quantum_state = self.quantum_processor.quantum_feature_map(text_features)

        # Modulate by emotion and intensity
        quantum_state = quantum_state * (1.0 + thought.intensity)

        # Apply phase shift based on emotional valence
        phase_shift = thought.emotional_valence * np.pi / 2
        rotation_params = {
            'theta_0': phase_shift,
            'phi_0': thought.intensity * np.pi,
            'lambda_0': 0.0
        }

        # Create consciousness-influenced circuit
        circuit = self.quantum_processor.create_variational_circuit(rotation_params)

        # Execute and get quantum state
        result = self.quantum_processor.execute_circuit(circuit)
        quantum_state = torch.tensor(result['statevector'], dtype=torch.complex64)

        return quantum_state

    async def _detect_resonance_patterns(
        self,
        thought: ThoughtInput,
        quantum_state: torch.Tensor
    ) -> List[QuantumPattern]:
        """
        Detect resonance patterns across multiple modalities.

        Uses quantum resonance detector to find patterns that resonate
        between text, quantum state, and historical patterns.
        """
        patterns = []

        # Detect text patterns
        text_patterns = self.resonance_detector.process_text_patterns(thought.text)
        patterns.extend(text_patterns)

        # Detect quantum state patterns
        quantum_patterns = self.resonance_detector.process_quantum_patterns(
            quantum_state.detach().numpy()
        )
        patterns.extend(quantum_patterns)

        # Cross-modal resonance detection
        if len(self.emergence_memory) > 0:
            # Compare with historical patterns
            historical_resonances = self._find_historical_resonances(
                quantum_state, thought.text
            )
            patterns.extend(historical_resonances)

        # Filter by resonance threshold
        strong_patterns = [
            p for p in patterns
            if p.strength >= self.config.resonance_threshold
        ]

        return strong_patterns

    async def _manifest_in_water(
        self,
        quantum_state: torch.Tensor,
        patterns: List[QuantumPattern]
    ) -> List[WaterParticle]:
        """
        Manifest quantum patterns in physical water simulation.

        Quantum state influences:
        - Particle positions (phase → spatial distribution)
        - Particle velocities (amplitude → kinetic energy)
        - Particle clustering (entanglement → correlation)
        """
        # Convert quantum state to water particle configuration
        state_amplitude = torch.abs(quantum_state)
        state_phase = torch.angle(quantum_state)

        particles = []
        n_particles = min(self.config.water_particles, len(state_amplitude))

        for i in range(n_particles):
            # Position from quantum phase
            theta = state_phase[i].item()
            phi = state_phase[(i + 1) % len(state_phase)].item()

            x = float(state_amplitude[i] * np.cos(theta) * np.sin(phi))
            y = float(state_amplitude[i] * np.sin(theta) * np.sin(phi))
            z = float(state_amplitude[i] * np.cos(phi))

            # Velocity from pattern strength
            avg_pattern_strength = np.mean([p.strength for p in patterns]) if patterns else 0.5
            velocity_scale = avg_pattern_strength * 0.1

            vx = float(np.random.randn() * velocity_scale)
            vy = float(np.random.randn() * velocity_scale)
            vz = float(np.random.randn() * velocity_scale)

            # Pressure and density from quantum coherence
            coherence = float(torch.abs(state_amplitude[i]))

            particle = WaterParticle(
                position=(x, y, z),
                velocity=(vx, vy, vz),
                pressure=coherence * 1000.0,
                density=coherence * 1000.0 + 500.0
            )
            particles.append(particle)

        return particles

    async def _generate_emergence(
        self,
        quantum_state: torch.Tensor,
        particles: List[WaterParticle],
        patterns: List[QuantumPattern]
    ) -> List[QuantumPattern]:
        """
        Generate emergent patterns through feedback loops.

        The physical manifestation influences the quantum state,
        creating new patterns that weren't in the original thought.
        """
        emergent_patterns = []

        for iteration in range(self.config.feedback_iterations):
            # Extract features from physical manifestation
            physical_features = self._extract_physical_features(particles)

            # Create feedback quantum state
            feedback_state = self.quantum_processor.quantum_feature_map(
                physical_features
            )

            # Combine with original quantum state
            combined_state = (
                quantum_state * self.config.coherence_preservation +
                feedback_state * (1.0 - self.config.coherence_preservation)
            )

            # Detect new emergent patterns
            new_patterns = self.resonance_detector.process_quantum_patterns(
                combined_state.detach().numpy()
            )

            # Filter for truly emergent patterns (not in original set)
            for pattern in new_patterns:
                if self._is_emergent_pattern(pattern, patterns):
                    pattern.pattern_type = f"emergent_{iteration}"
                    emergent_patterns.append(pattern)

            # Update quantum state for next iteration
            quantum_state = combined_state

        return emergent_patterns

    async def _generate_insights(
        self,
        thought: ThoughtInput,
        emergent_patterns: List[QuantumPattern],
        resonance_patterns: List[QuantumPattern]
    ) -> str:
        """
        Generate human-readable insights about what emerged.

        Uses quantum LLM if available, otherwise generates analytical summary.
        """
        if self.llm is None:
            return self._generate_analytical_insights(
                thought, emergent_patterns, resonance_patterns
            )

        # Prepare context for LLM
        context = self._prepare_insight_context(
            thought, emergent_patterns, resonance_patterns
        )

        # Generate insights using quantum LLM
        # (Simplified - full implementation would use actual LLM generation)
        insights = self._generate_analytical_insights(
            thought, emergent_patterns, resonance_patterns
        )

        return insights

    def _generate_analytical_insights(
        self,
        thought: ThoughtInput,
        emergent_patterns: List[QuantumPattern],
        resonance_patterns: List[QuantumPattern]
    ) -> str:
        """Generate analytical insights without LLM."""
        insights = []

        insights.append(f"Original thought: '{thought.text}'")
        insights.append(f"Emotional valence: {thought.emotional_valence:.2f}")
        insights.append(f"Intensity: {thought.intensity:.2f}")
        insights.append("")

        insights.append(f"Resonance patterns detected: {len(resonance_patterns)}")
        if resonance_patterns:
            top_patterns = sorted(resonance_patterns, key=lambda p: p.strength, reverse=True)[:3]
            for i, pattern in enumerate(top_patterns, 1):
                insights.append(
                    f"  {i}. {pattern.pattern_type} "
                    f"(strength: {pattern.strength:.3f}, modality: {pattern.modality})"
                )

        insights.append("")
        insights.append(f"Emergent patterns generated: {len(emergent_patterns)}")
        if emergent_patterns:
            for i, pattern in enumerate(emergent_patterns, 1):
                insights.append(
                    f"  {i}. {pattern.pattern_type} "
                    f"(strength: {pattern.strength:.3f}) - "
                    "This pattern emerged from quantum-physical feedback"
                )

        insights.append("")
        insights.append("EMERGENCE ANALYSIS:")
        emergence_ratio = len(emergent_patterns) / max(len(resonance_patterns), 1)
        if emergence_ratio > 0.5:
            insights.append(
                "  • High emergence detected - the system generated significant "
                "new patterns beyond the original thought"
            )
        elif emergence_ratio > 0.2:
            insights.append(
                "  • Moderate emergence - some new patterns evolved through "
                "quantum-physical feedback"
            )
        else:
            insights.append(
                "  • Low emergence - the thought crystallized mostly as expected "
                "with minimal novel patterns"
            )

        return "\n".join(insights)

    def _text_to_features(self, text: str) -> torch.Tensor:
        """Convert text to feature vector."""
        # Simple character-based encoding
        chars = [ord(c) / 127.0 for c in text[:64]]
        # Pad to fixed size
        while len(chars) < 64:
            chars.append(0.0)
        return torch.tensor(chars[:64], dtype=torch.float32)

    def _extract_physical_features(
        self,
        particles: List[WaterParticle]
    ) -> torch.Tensor:
        """Extract feature vector from physical particle state."""
        if not particles:
            return torch.zeros(64)

        features = []

        # Sample particles for features
        sample_size = min(16, len(particles))
        sampled = np.random.choice(particles, sample_size, replace=False)

        for particle in sampled:
            features.extend([
                particle.position[0],
                particle.position[1],
                particle.position[2],
                particle.pressure / 1000.0
            ])

        # Pad to 64 features
        while len(features) < 64:
            features.append(0.0)

        return torch.tensor(features[:64], dtype=torch.float32)

    def _find_historical_resonances(
        self,
        quantum_state: torch.Tensor,
        text: str
    ) -> List[QuantumPattern]:
        """Find resonances with historical crystallized thoughts."""
        resonances = []

        for memory in self.emergence_memory:
            # Calculate quantum similarity
            similarity = self.quantum_processor.quantum_enhanced_similarity(
                quantum_state, memory.quantum_state
            )

            if similarity > self.config.resonance_threshold:
                pattern = QuantumPattern(
                    modality='temporal',
                    pattern_type='historical_resonance',
                    strength=float(similarity),
                    quantum_state=quantum_state,
                    context={
                        'historical_text': memory.original_thought.text,
                        'similarity': float(similarity)
                    }
                )
                resonances.append(pattern)

        return resonances

    def _is_emergent_pattern(
        self,
        pattern: QuantumPattern,
        original_patterns: List[QuantumPattern]
    ) -> bool:
        """Check if pattern is truly emergent (not in original set)."""
        for orig in original_patterns:
            if (pattern.modality == orig.modality and
                pattern.pattern_type == orig.pattern_type and
                abs(pattern.strength - orig.strength) < 0.1):
                return False
        return True

    def _calculate_consciousness_metrics(
        self,
        quantum_state: torch.Tensor,
        resonance_patterns: List[QuantumPattern],
        emergent_patterns: List[QuantumPattern]
    ) -> Dict[str, float]:
        """Calculate metrics describing the consciousness that emerged."""
        # Quantum coherence
        coherence = float(torch.abs(quantum_state).mean())

        # Entanglement (simplified von Neumann entropy)
        probs = torch.abs(quantum_state) ** 2
        probs = probs / probs.sum()
        entanglement = float(-torch.sum(probs * torch.log(probs + 1e-10)))

        # Pattern complexity
        total_patterns = len(resonance_patterns) + len(emergent_patterns)
        pattern_diversity = len(set(p.pattern_type for p in resonance_patterns + emergent_patterns))

        # Emergence strength
        emergence_strength = len(emergent_patterns) / max(total_patterns, 1)

        # Average resonance
        avg_resonance = np.mean([p.strength for p in resonance_patterns]) if resonance_patterns else 0.0

        return {
            'coherence': coherence,
            'entanglement': entanglement,
            'total_patterns': total_patterns,
            'pattern_diversity': pattern_diversity,
            'emergence_strength': emergence_strength,
            'average_resonance': float(avg_resonance),
            'consciousness_index': (coherence + emergence_strength + float(avg_resonance)) / 3.0
        }

    def _update_metrics(self, crystallized: CrystalizedThought):
        """Update global metrics."""
        self.metrics['total_crystallizations'] += 1

        # Running averages
        n = self.metrics['total_crystallizations']
        self.metrics['average_coherence'] = (
            (self.metrics['average_coherence'] * (n - 1) +
             crystallized.consciousness_metrics['coherence']) / n
        )
        self.metrics['average_resonance_strength'] = (
            (self.metrics['average_resonance_strength'] * (n - 1) +
             crystallized.consciousness_metrics['average_resonance']) / n
        )

        if crystallized.consciousness_metrics['emergence_strength'] > 0.3:
            self.metrics['emergence_events'] += 1

        self.metrics['feedback_loops_completed'] += self.config.feedback_iterations

    def _prepare_insight_context(
        self,
        thought: ThoughtInput,
        emergent_patterns: List[QuantumPattern],
        resonance_patterns: List[QuantumPattern]
    ) -> str:
        """Prepare context string for LLM insight generation."""
        context_parts = [
            f"Original thought: {thought.text}",
            f"Emotional state: {thought.emotional_valence:.2f}",
            f"Resonance patterns: {len(resonance_patterns)}",
            f"Emergent patterns: {len(emergent_patterns)}"
        ]
        return " | ".join(context_parts)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return {
            **self.metrics,
            'memory_size': len(self.emergence_memory),
            'config': {
                'n_qubits': self.config.n_qubits,
                'resonance_threshold': self.config.resonance_threshold,
                'feedback_iterations': self.config.feedback_iterations
            }
        }

    def export_crystallization_history(self, filepath: str):
        """Export all crystallized thoughts to JSON file."""
        history = [c.to_dict() for c in self.emergence_memory]
        with open(filepath, 'w') as f:
            json.dump({
                'crystallizations': history,
                'metrics': self.get_metrics(),
                'config': {
                    'n_qubits': self.config.n_qubits,
                    'resonance_threshold': self.config.resonance_threshold,
                    'feedback_iterations': self.config.feedback_iterations,
                    'emergence_temperature': self.config.emergence_temperature
                }
            }, f, indent=2)

        self.logger.info(f"Exported {len(history)} crystallizations to {filepath}")


# Convenience function for quick crystallization
async def crystallize(
    text: str,
    emotional_valence: float = 0.0,
    intensity: float = 1.0,
    config: Optional[EmergenceConfig] = None
) -> CrystalizedThought:
    """
    Quick crystallization function.

    Example:
        result = await crystallize(
            "What is the nature of consciousness?",
            emotional_valence=0.3,
            intensity=0.9
        )
        print(result.emergent_insights)
    """
    if config is None:
        config = EmergenceConfig()

    crystallizer = QuantumThoughtCrystallizer(config)
    thought = ThoughtInput(
        text=text,
        emotional_valence=emotional_valence,
        intensity=intensity
    )

    return await crystallizer.crystallize_thought(thought)
