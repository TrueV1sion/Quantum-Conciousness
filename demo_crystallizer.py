"""
Interactive Demo for Quantum Thought Crystallizer

This demo showcases the consciousness emergence engine with various
thought examples, demonstrating how abstract ideas crystallize into
quantum-physical patterns.

Usage:
    python demo_crystallizer.py
"""

import asyncio
import logging
from typing import List
from datetime import datetime

from quantum_thought_crystallizer import (
    QuantumThoughtCrystallizer,
    ThoughtInput,
    EmergenceConfig,
    CrystalizedThought,
    crystallize
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class CrystallizerDemo:
    """Interactive demonstration of the Quantum Thought Crystallizer."""

    def __init__(self):
        self.config = EmergenceConfig(
            n_qubits=6,
            resonance_threshold=0.6,
            water_particles=500,
            feedback_iterations=3,
            emergence_temperature=0.8,
            enable_rl_optimization=False,  # Disabled for demo simplicity
            enable_llm_insights=False,      # Disabled for demo simplicity
            visualization_enabled=True
        )
        self.crystallizer = QuantumThoughtCrystallizer(self.config)

    def print_header(self, title: str):
        """Print formatted header."""
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80 + "\n")

    def print_crystallization_result(self, result: CrystalizedThought):
        """Print detailed crystallization results."""
        print("\n" + "-" * 80)
        print(f"THOUGHT: '{result.original_thought.text}'")
        print(f"Emotional Valence: {result.original_thought.emotional_valence:+.2f}")
        print(f"Intensity: {result.original_thought.intensity:.2f}")
        print("-" * 80)

        print("\nCONSCIOUSNESS METRICS:")
        for key, value in result.consciousness_metrics.items():
            print(f"  • {key.replace('_', ' ').title()}: {value:.3f}")

        print(f"\nPHASES TRAVERSED: {len(result.phase_history)}")
        for i, phase in enumerate(result.phase_history, 1):
            print(f"  {i}. {phase.name}")

        print(f"\nRESULTS:")
        print(f"  • Resonance patterns detected: {len([p for p in result.resonance_patterns if 'emergent' not in p.pattern_type])}")
        print(f"  • Emergent patterns generated: {len([p for p in result.resonance_patterns if 'emergent' in p.pattern_type])}")
        print(f"  • Water particles manifested: {len(result.physical_manifestation)}")

        print(f"\nEMERGENT INSIGHTS:")
        print("-" * 80)
        for line in result.emergent_insights.split('\n'):
            print(f"  {line}")
        print("-" * 80 + "\n")

    async def demo_basic_crystallization(self):
        """Demonstrate basic thought crystallization."""
        self.print_header("DEMO 1: Basic Thought Crystallization")

        print("Crystallizing a simple philosophical question...")

        thought = ThoughtInput(
            text="What is the nature of consciousness?",
            emotional_valence=0.2,
            intensity=0.8
        )

        result = await self.crystallizer.crystallize_thought(thought)
        self.print_crystallization_result(result)

    async def demo_emotional_spectrum(self):
        """Demonstrate crystallization across emotional spectrum."""
        self.print_header("DEMO 2: Emotional Spectrum Analysis")

        thoughts = [
            ThoughtInput(
                text="I feel overwhelming joy and connection with everything",
                emotional_valence=1.0,
                intensity=1.0
            ),
            ThoughtInput(
                text="Contemplating the void and emptiness of existence",
                emotional_valence=-0.7,
                intensity=0.9
            ),
            ThoughtInput(
                text="Everything is perfectly balanced and harmonious",
                emotional_valence=0.0,
                intensity=0.5
            )
        ]

        results = []
        for thought in thoughts:
            print(f"\nCrystallizing: '{thought.text}'")
            result = await self.crystallizer.crystallize_thought(thought)
            results.append(result)
            print(f"  → Coherence: {result.consciousness_metrics['coherence']:.3f}")
            print(f"  → Emergence: {result.consciousness_metrics['emergence_strength']:.3f}")

        print("\n" + "=" * 80)
        print("COMPARISON ACROSS EMOTIONAL SPECTRUM:")
        print("=" * 80)
        for i, result in enumerate(results, 1):
            valence = result.original_thought.emotional_valence
            coherence = result.consciousness_metrics['coherence']
            emergence = result.consciousness_metrics['emergence_strength']
            print(f"\n{i}. Valence: {valence:+.2f} → Coherence: {coherence:.3f}, Emergence: {emergence:.3f}")

    async def demo_resonance_buildup(self):
        """Demonstrate resonance buildup across similar thoughts."""
        self.print_header("DEMO 3: Resonance Buildup and Memory")

        print("Crystallizing related thoughts to observe resonance buildup...\n")

        related_thoughts = [
            "The universe is made of information",
            "Information creates reality",
            "Reality emerges from quantum information",
            "Quantum states encode universal consciousness"
        ]

        for i, text in enumerate(related_thoughts, 1):
            print(f"\n[{i}/{len(related_thoughts)}] Crystallizing: '{text}'")

            thought = ThoughtInput(text=text, emotional_valence=0.3, intensity=0.7)
            result = await self.crystallizer.crystallize_thought(thought)

            # Check for historical resonances
            historical_patterns = [
                p for p in result.resonance_patterns
                if p.pattern_type == 'historical_resonance'
            ]

            print(f"  → Historical resonances: {len(historical_patterns)}")
            print(f"  → Consciousness index: {result.consciousness_metrics['consciousness_index']:.3f}")

            if historical_patterns:
                print(f"  → Resonating with previous thoughts!")
                for pattern in historical_patterns[:2]:  # Show top 2
                    if pattern.context:
                        print(f"      • '{pattern.context.get('historical_text', 'N/A')[:60]}...'")

    async def demo_emergence_evolution(self):
        """Demonstrate how emergence evolves with feedback iterations."""
        self.print_header("DEMO 4: Emergence Evolution Analysis")

        print("Comparing crystallization with different feedback iterations...\n")

        thought = ThoughtInput(
            text="Consciousness arises from quantum coherence in neural microtubules",
            emotional_valence=0.1,
            intensity=1.0
        )

        for iterations in [1, 3, 5]:
            config = EmergenceConfig(
                n_qubits=6,
                feedback_iterations=iterations,
                enable_rl_optimization=False,
                enable_llm_insights=False
            )
            temp_crystallizer = QuantumThoughtCrystallizer(config)

            print(f"Feedback iterations: {iterations}")
            result = await temp_crystallizer.crystallize_thought(thought)

            emergent_count = len([
                p for p in result.resonance_patterns
                if 'emergent' in p.pattern_type
            ])

            print(f"  → Emergent patterns: {emergent_count}")
            print(f"  → Emergence strength: {result.consciousness_metrics['emergence_strength']:.3f}")
            print(f"  → Consciousness index: {result.consciousness_metrics['consciousness_index']:.3f}\n")

    async def demo_complex_thought(self):
        """Demonstrate crystallization of a complex, multi-faceted thought."""
        self.print_header("DEMO 5: Complex Thought Crystallization")

        thought = ThoughtInput(
            text=(
                "As I observe the quantum foam of spacetime, I realize that "
                "every moment contains infinite potential realities, each "
                "collapsing into existence through the act of conscious observation. "
                "The boundary between observer and observed dissolves in the "
                "symphony of quantum entanglement."
            ),
            emotional_valence=0.4,
            intensity=0.95,
            focus_areas=['quantum_mechanics', 'consciousness', 'philosophy']
        )

        print(f"Crystallizing complex thought ({len(thought.text)} characters)...")
        print(f"Focus areas: {', '.join(thought.focus_areas)}\n")

        result = await self.crystallizer.crystallize_thought(thought)
        self.print_crystallization_result(result)

        # Additional analysis
        print("\nCOMPLEXITY ANALYSIS:")
        print(f"  • Pattern diversity: {result.consciousness_metrics['pattern_diversity']}")
        print(f"  • Total unique patterns: {result.consciousness_metrics['total_patterns']}")
        print(f"  • Entanglement measure: {result.consciousness_metrics['entanglement']:.3f}")

    async def demo_system_metrics(self):
        """Display overall system metrics."""
        self.print_header("SYSTEM METRICS AND MEMORY")

        metrics = self.crystallizer.get_metrics()

        print("GLOBAL METRICS:")
        print(f"  • Total crystallizations: {metrics['total_crystallizations']}")
        print(f"  • Average coherence: {metrics['average_coherence']:.3f}")
        print(f"  • Average resonance strength: {metrics['average_resonance_strength']:.3f}")
        print(f"  • Emergence events: {metrics['emergence_events']}")
        print(f"  • Feedback loops completed: {metrics['feedback_loops_completed']}")
        print(f"  • Memory size: {metrics['memory_size']} crystallizations")

        print("\nCONFIGURATION:")
        for key, value in metrics['config'].items():
            print(f"  • {key}: {value}")

        # Export history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crystallization_history_{timestamp}.json"
        self.crystallizer.export_crystallization_history(filename)
        print(f"\n✓ Full history exported to: {filename}")

    async def run_all_demos(self):
        """Run complete demonstration sequence."""
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 15 + "QUANTUM THOUGHT CRYSTALLIZER DEMO" + " " * 30 + "║")
        print("║" + " " * 15 + "Consciousness Emergence Engine" + " " * 32 + "║")
        print("╚" + "═" * 78 + "╝")

        demos = [
            ("Basic Crystallization", self.demo_basic_crystallization),
            ("Emotional Spectrum", self.demo_emotional_spectrum),
            ("Resonance Buildup", self.demo_resonance_buildup),
            ("Emergence Evolution", self.demo_emergence_evolution),
            ("Complex Thought", self.demo_complex_thought),
            ("System Metrics", self.demo_system_metrics)
        ]

        for i, (name, demo_func) in enumerate(demos, 1):
            try:
                await demo_func()
                if i < len(demos):
                    input("\nPress Enter to continue to next demo...")
            except KeyboardInterrupt:
                print("\n\nDemo interrupted by user.")
                break
            except Exception as e:
                print(f"\n❌ Error in {name} demo: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 80)
        print("  Demo sequence complete!")
        print("=" * 80 + "\n")


async def quick_demo():
    """Quick demonstration using convenience function."""
    print("\n" + "=" * 80)
    print("  QUICK DEMO: Using convenience function")
    print("=" * 80 + "\n")

    result = await crystallize(
        text="What happens when thoughts become quantum reality?",
        emotional_valence=0.5,
        intensity=0.8
    )

    print("\nRESULT:")
    print(result.emergent_insights)
    print(f"\nConsciousness Index: {result.consciousness_metrics['consciousness_index']:.3f}")


def main():
    """Main entry point for demo."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        asyncio.run(quick_demo())
    else:
        demo = CrystallizerDemo()
        asyncio.run(demo.run_all_demos())


if __name__ == "__main__":
    main()
