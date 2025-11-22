"""
Integration Tests for Quantum Thought Crystallizer

Tests the complete crystallization pipeline and various components.
"""

import asyncio
import torch
import numpy as np
from typing import List

from quantum_thought_crystallizer import (
    QuantumThoughtCrystallizer,
    ThoughtInput,
    EmergenceConfig,
    CrystalizedThought,
    ConsciousnessPhase,
    crystallize
)


class TestCrystallizer:
    """Test suite for Quantum Thought Crystallizer."""

    def __init__(self):
        self.config = EmergenceConfig(
            n_qubits=4,  # Smaller for faster tests
            resonance_threshold=0.5,
            water_particles=100,
            feedback_iterations=2,
            enable_rl_optimization=False,
            enable_llm_insights=False
        )
        self.crystallizer = QuantumThoughtCrystallizer(self.config)

    async def test_basic_crystallization(self):
        """Test basic crystallization process."""
        print("\n[TEST] Basic Crystallization...")

        thought = ThoughtInput(
            text="Test thought for crystallization",
            emotional_valence=0.5,
            intensity=0.7
        )

        result = await self.crystallizer.crystallize_thought(thought)

        # Assertions
        assert isinstance(result, CrystalizedThought), "Result should be CrystalizedThought"
        assert result.quantum_state is not None, "Quantum state should exist"
        assert len(result.phase_history) == 5, "Should have 5 phases"
        assert result.consciousness_metrics['coherence'] > 0, "Coherence should be positive"

        print("✓ Basic crystallization works correctly")
        return result

    async def test_quantum_encoding(self):
        """Test thought-to-quantum encoding."""
        print("\n[TEST] Quantum Encoding...")

        thoughts = [
            ThoughtInput(text="Short", emotional_valence=0.0, intensity=0.5),
            ThoughtInput(text="A much longer thought with more content", emotional_valence=0.8, intensity=1.0),
        ]

        states = []
        for thought in thoughts:
            result = await self.crystallizer.crystallize_thought(thought)
            states.append(result.quantum_state)

        # Different thoughts should produce different states
        similarity = torch.abs(torch.dot(states[0].flatten(), states[1].flatten().conj()))
        assert similarity < 0.99, "Different thoughts should produce different quantum states"

        print(f"✓ Quantum encoding produces distinct states (similarity: {similarity:.3f})")

    async def test_resonance_detection(self):
        """Test resonance pattern detection."""
        print("\n[TEST] Resonance Detection...")

        # First thought
        thought1 = ThoughtInput(
            text="Quantum consciousness is fundamental",
            emotional_valence=0.3,
            intensity=0.8
        )
        result1 = await self.crystallizer.crystallize_thought(thought1)
        patterns1 = len(result1.resonance_patterns)

        # Related thought - should show historical resonance
        thought2 = ThoughtInput(
            text="Consciousness arises from quantum processes",
            emotional_valence=0.3,
            intensity=0.8
        )
        result2 = await self.crystallizer.crystallize_thought(thought2)

        historical = [p for p in result2.resonance_patterns
                     if p.pattern_type == 'historical_resonance']

        assert len(historical) > 0, "Should detect historical resonance"

        print(f"✓ Resonance detection works (found {len(historical)} historical resonances)")

    async def test_emergence_generation(self):
        """Test emergence of new patterns."""
        print("\n[TEST] Emergence Generation...")

        thought = ThoughtInput(
            text="Complex thought about quantum entanglement and consciousness",
            emotional_valence=0.2,
            intensity=0.9
        )

        result = await self.crystallizer.crystallize_thought(thought)

        emergent_patterns = [p for p in result.resonance_patterns
                           if 'emergent' in p.pattern_type]

        assert len(emergent_patterns) > 0, "Should generate emergent patterns"
        assert result.consciousness_metrics['emergence_strength'] > 0, "Emergence strength should be positive"

        print(f"✓ Emergence generation works ({len(emergent_patterns)} emergent patterns)")

    async def test_emotional_spectrum(self):
        """Test crystallization across emotional spectrum."""
        print("\n[TEST] Emotional Spectrum...")

        emotions = [-1.0, -0.5, 0.0, 0.5, 1.0]
        results = []

        for emotion in emotions:
            thought = ThoughtInput(
                text="Same text different emotion",
                emotional_valence=emotion,
                intensity=0.8
            )
            result = await self.crystallizer.crystallize_thought(thought)
            results.append(result)

        # Check that different emotions produce different coherence patterns
        coherences = [r.consciousness_metrics['coherence'] for r in results]
        assert len(set([round(c, 2) for c in coherences])) > 1, \
            "Different emotions should produce different coherence patterns"

        print(f"✓ Emotional spectrum affects crystallization")
        print(f"  Coherence range: {min(coherences):.3f} - {max(coherences):.3f}")

    async def test_physical_manifestation(self):
        """Test water particle manifestation."""
        print("\n[TEST] Physical Manifestation...")

        thought = ThoughtInput(
            text="Test physical manifestation",
            emotional_valence=0.5,
            intensity=0.8
        )

        result = await self.crystallizer.crystallize_thought(thought)

        assert len(result.physical_manifestation) > 0, "Should create water particles"
        assert len(result.physical_manifestation) <= self.config.water_particles, \
            "Should not exceed max particles"

        # Check particles have valid properties
        for particle in result.physical_manifestation[:5]:  # Check first 5
            assert len(particle.position) == 3, "Position should be 3D"
            assert len(particle.velocity) == 3, "Velocity should be 3D"
            assert particle.pressure >= 0, "Pressure should be non-negative"
            assert particle.density > 0, "Density should be positive"

        print(f"✓ Physical manifestation works ({len(result.physical_manifestation)} particles)")

    async def test_metrics_tracking(self):
        """Test system metrics tracking."""
        print("\n[TEST] Metrics Tracking...")

        initial_metrics = self.crystallizer.get_metrics()

        # Perform several crystallizations
        for i in range(3):
            thought = ThoughtInput(
                text=f"Test thought number {i}",
                emotional_valence=0.5,
                intensity=0.7
            )
            await self.crystallizer.crystallize_thought(thought)

        final_metrics = self.crystallizer.get_metrics()

        assert final_metrics['total_crystallizations'] > initial_metrics['total_crystallizations'], \
            "Should track total crystallizations"
        assert final_metrics['memory_size'] > 0, "Should have memory"

        print(f"✓ Metrics tracking works")
        print(f"  Total crystallizations: {final_metrics['total_crystallizations']}")
        print(f"  Memory size: {final_metrics['memory_size']}")

    async def test_convenience_function(self):
        """Test convenience crystallize function."""
        print("\n[TEST] Convenience Function...")

        result = await crystallize(
            text="Quick crystallization test",
            emotional_valence=0.3,
            intensity=0.8
        )

        assert isinstance(result, CrystalizedThought), "Should return CrystalizedThought"
        assert result.emergent_insights is not None, "Should generate insights"

        print("✓ Convenience function works")

    async def test_consciousness_metrics_validity(self):
        """Test that consciousness metrics are valid."""
        print("\n[TEST] Consciousness Metrics Validity...")

        thought = ThoughtInput(
            text="Testing consciousness metrics",
            emotional_valence=0.5,
            intensity=0.8
        )

        result = await self.crystallizer.crystallize_thought(thought)
        metrics = result.consciousness_metrics

        # Check all metrics are within valid ranges
        assert 0 <= metrics['coherence'] <= 1.5, "Coherence should be in reasonable range"
        assert metrics['entanglement'] >= 0, "Entanglement should be non-negative"
        assert metrics['total_patterns'] >= 0, "Pattern count should be non-negative"
        assert 0 <= metrics['emergence_strength'] <= 1, "Emergence strength should be 0-1"
        assert 0 <= metrics['consciousness_index'] <= 1.5, "Consciousness index should be reasonable"

        print("✓ All consciousness metrics are valid")
        for key, value in metrics.items():
            print(f"  {key}: {value:.3f}")

    async def test_export_functionality(self):
        """Test export of crystallization history."""
        print("\n[TEST] Export Functionality...")

        import os
        import json

        # Create some crystallizations
        for i in range(2):
            thought = ThoughtInput(
                text=f"Export test {i}",
                emotional_valence=0.5,
                intensity=0.7
            )
            await self.crystallizer.crystallize_thought(thought)

        # Export
        export_path = "test_export.json"
        self.crystallizer.export_crystallization_history(export_path)

        assert os.path.exists(export_path), "Export file should exist"

        # Verify JSON structure
        with open(export_path, 'r') as f:
            data = json.load(f)

        assert 'crystallizations' in data, "Should have crystallizations"
        assert 'metrics' in data, "Should have metrics"
        assert 'config' in data, "Should have config"

        # Cleanup
        os.remove(export_path)

        print("✓ Export functionality works")

    async def run_all_tests(self):
        """Run all tests."""
        print("\n" + "=" * 80)
        print("  QUANTUM THOUGHT CRYSTALLIZER - TEST SUITE")
        print("=" * 80)

        tests = [
            self.test_basic_crystallization,
            self.test_quantum_encoding,
            self.test_resonance_detection,
            self.test_emergence_generation,
            self.test_emotional_spectrum,
            self.test_physical_manifestation,
            self.test_metrics_tracking,
            self.test_convenience_function,
            self.test_consciousness_metrics_validity,
            self.test_export_functionality
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                await test()
                passed += 1
            except AssertionError as e:
                print(f"✗ Test failed: {e}")
                failed += 1
            except Exception as e:
                print(f"✗ Test error: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

        print("\n" + "=" * 80)
        print(f"  TEST RESULTS: {passed} passed, {failed} failed")
        print("=" * 80 + "\n")

        return passed, failed


async def main():
    """Run test suite."""
    tester = TestCrystallizer()
    passed, failed = await tester.run_all_tests()

    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
