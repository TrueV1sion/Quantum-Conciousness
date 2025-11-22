"""
Visualization Module for Quantum Thought Crystallizer

Provides various visualization methods for crystallized thoughts:
- Quantum state visualization (Bloch sphere, state vector)
- Resonance pattern networks
- Consciousness metrics over time
- Physical manifestation in 3D space
- Emergence evolution graphs

Author: Quantum Consciousness Framework
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch

from quantum_thought_crystallizer import CrystalizedThought, QuantumPattern


class CrystallizerVisualizer:
    """Visualization suite for crystallized thoughts."""

    def __init__(self, style: str = 'dark_background'):
        """Initialize visualizer with style."""
        self.style = style
        plt.style.use(style)
        sns.set_palette("husl")

    def visualize_crystallization(
        self,
        crystallized: CrystalizedThought,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Create comprehensive visualization of a crystallization.

        Produces a multi-panel figure showing:
        1. Consciousness metrics
        2. Pattern distribution
        3. Quantum state
        4. Physical manifestation
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Panel 1: Consciousness Metrics
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_consciousness_metrics(ax1, crystallized)

        # Panel 2: Pattern Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_pattern_distribution(ax2, crystallized)

        # Panel 3: Quantum State (Real part)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_quantum_state_bars(ax3, crystallized.quantum_state, 'real')

        # Panel 4: Quantum State (Imaginary part)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_quantum_state_bars(ax4, crystallized.quantum_state, 'imag')

        # Panel 5: Quantum State (Phase)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_quantum_phase(ax5, crystallized.quantum_state)

        # Panel 6: Physical Manifestation (3D)
        ax6 = fig.add_subplot(gs[2, :], projection='3d')
        self._plot_physical_manifestation_3d(ax6, crystallized)

        # Add title
        thought_preview = crystallized.original_thought.text[:60] + "..."
        fig.suptitle(
            f'Crystallization Analysis: "{thought_preview}"\n'
            f'Consciousness Index: {crystallized.consciousness_metrics["consciousness_index"]:.3f}',
            fontsize=14,
            fontweight='bold'
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")

        if show:
            plt.show()

        return fig

    def _plot_consciousness_metrics(self, ax, crystallized: CrystalizedThought):
        """Plot consciousness metrics as bars."""
        metrics = crystallized.consciousness_metrics

        # Select key metrics for visualization
        metric_names = [
            'coherence',
            'entanglement',
            'emergence_strength',
            'average_resonance',
            'consciousness_index'
        ]

        values = [metrics.get(name, 0.0) for name in metric_names]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metric_names)))

        bars = ax.barh(metric_names, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(
                value + 0.02,
                i,
                f'{value:.3f}',
                va='center',
                fontweight='bold',
                fontsize=10
            )

        ax.set_xlabel('Value', fontweight='bold')
        ax.set_title('Consciousness Metrics', fontweight='bold', fontsize=12)
        ax.set_xlim(0, max(values) * 1.15)
        ax.grid(axis='x', alpha=0.3)

    def _plot_pattern_distribution(self, ax, crystallized: CrystalizedThought):
        """Plot pattern type distribution as pie chart."""
        pattern_types = {}
        for pattern in crystallized.resonance_patterns:
            ptype = pattern.pattern_type
            if 'emergent' in ptype:
                ptype = 'emergent'
            pattern_types[ptype] = pattern_types.get(ptype, 0) + 1

        if pattern_types:
            colors = plt.cm.Set3(np.linspace(0, 1, len(pattern_types)))
            wedges, texts, autotexts = ax.pie(
                pattern_types.values(),
                labels=pattern_types.keys(),
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                textprops={'fontsize': 9, 'fontweight': 'bold'}
            )

            for autotext in autotexts:
                autotext.set_color('black')
        else:
            ax.text(0.5, 0.5, 'No patterns', ha='center', va='center', fontsize=12)

        ax.set_title('Pattern Distribution', fontweight='bold', fontsize=12)

    def _plot_quantum_state_bars(
        self,
        ax,
        quantum_state: torch.Tensor,
        component: str = 'real'
    ):
        """Plot quantum state amplitudes as bars."""
        if component == 'real':
            values = quantum_state.real.numpy()
            title = 'Quantum State (Real)'
            color = 'skyblue'
        elif component == 'imag':
            values = quantum_state.imag.numpy()
            title = 'Quantum State (Imaginary)'
            color = 'lightcoral'
        else:
            values = torch.abs(quantum_state).numpy()
            title = 'Quantum State (Magnitude)'
            color = 'lightgreen'

        x = np.arange(len(values))
        bars = ax.bar(x, values, color=color, alpha=0.7, edgecolor='black', linewidth=1)

        # Highlight significant amplitudes
        threshold = np.max(np.abs(values)) * 0.3
        for i, (bar, val) in enumerate(zip(bars, values)):
            if abs(val) > threshold:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)

        ax.set_xlabel('Basis State', fontweight='bold')
        ax.set_ylabel('Amplitude', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

    def _plot_quantum_phase(self, ax, quantum_state: torch.Tensor):
        """Plot quantum state phase distribution."""
        phases = torch.angle(quantum_state).numpy()
        amplitudes = torch.abs(quantum_state).numpy()

        scatter = ax.scatter(
            np.arange(len(phases)),
            phases,
            s=amplitudes * 500,  # Size proportional to amplitude
            c=phases,
            cmap='twilight',
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5
        )

        ax.set_xlabel('Basis State', fontweight='bold')
        ax.set_ylabel('Phase (radians)', fontweight='bold')
        ax.set_title('Quantum Phase Distribution', fontweight='bold', fontsize=11)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax.set_ylim(-np.pi - 0.5, np.pi + 0.5)
        ax.grid(alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Phase', fontweight='bold')

    def _plot_physical_manifestation_3d(
        self,
        ax,
        crystallized: CrystalizedThought
    ):
        """Plot 3D visualization of water particles."""
        particles = crystallized.physical_manifestation

        if not particles:
            ax.text(0.5, 0.5, 0.5, 'No physical manifestation',
                   ha='center', va='center', fontsize=12)
            return

        # Extract positions and properties
        positions = np.array([p.position for p in particles])
        pressures = np.array([p.pressure for p in particles])

        # Normalize pressure for coloring
        norm_pressure = (pressures - pressures.min()) / (pressures.max() - pressures.min() + 1e-10)

        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=norm_pressure,
            cmap='plasma',
            s=30,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )

        ax.set_xlabel('X', fontweight='bold')
        ax.set_ylabel('Y', fontweight='bold')
        ax.set_zlabel('Z', fontweight='bold')
        ax.set_title('Physical Manifestation (Water Particles)', fontweight='bold', fontsize=11)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label('Normalized Pressure', fontweight='bold')

        # Set equal aspect ratio
        max_range = np.array([
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ]).max() / 2.0

        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def plot_consciousness_evolution(
        self,
        crystallizations: List[CrystalizedThought],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot evolution of consciousness metrics over multiple crystallizations.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Consciousness Evolution Across Crystallizations', fontsize=14, fontweight='bold')

        metrics_to_plot = [
            ('coherence', 'Coherence'),
            ('emergence_strength', 'Emergence Strength'),
            ('average_resonance', 'Average Resonance'),
            ('consciousness_index', 'Consciousness Index')
        ]

        for ax, (metric_key, metric_name) in zip(axes.flat, metrics_to_plot):
            values = [c.consciousness_metrics[metric_key] for c in crystallizations]
            x = np.arange(1, len(values) + 1)

            # Plot line
            ax.plot(x, values, marker='o', linewidth=2, markersize=8, alpha=0.8, label=metric_name)

            # Add trend line
            if len(values) > 1:
                z = np.polyfit(x, values, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.5, label=f'Trend (slope: {z[0]:.3f})')

            ax.set_xlabel('Crystallization Number', fontweight='bold')
            ax.set_ylabel(metric_name, fontweight='bold')
            ax.set_title(metric_name, fontweight='bold', fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Evolution plot saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_pattern_network(
        self,
        patterns: List[QuantumPattern],
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Visualize patterns as a network graph.
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Group patterns by modality
        modality_groups = {}
        for pattern in patterns:
            if pattern.modality not in modality_groups:
                modality_groups[pattern.modality] = []
            modality_groups[pattern.modality].append(pattern)

        # Create circular layout by modality
        n_modalities = len(modality_groups)
        modality_colors = plt.cm.Set2(np.linspace(0, 1, n_modalities))

        angle_step = 2 * np.pi / n_modalities
        radius = 5

        for i, (modality, mod_patterns) in enumerate(modality_groups.items()):
            base_angle = i * angle_step
            n_patterns = len(mod_patterns)

            for j, pattern in enumerate(mod_patterns):
                # Position
                angle = base_angle + (j - n_patterns/2) * 0.3
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)

                # Size based on strength
                size = pattern.strength * 1000

                # Draw pattern node
                ax.scatter(x, y, s=size, c=[modality_colors[i]],
                          alpha=0.7, edgecolors='black', linewidth=2, zorder=3)

                # Add label
                label = f"{pattern.pattern_type[:15]}\n{pattern.strength:.2f}"
                ax.text(x, y, label, ha='center', va='center',
                       fontsize=7, fontweight='bold', zorder=4)

            # Add modality label
            label_x = (radius + 1) * np.cos(base_angle)
            label_y = (radius + 1) * np.sin(base_angle)
            ax.text(label_x, label_y, modality.upper(),
                   ha='center', va='center', fontsize=12,
                   fontweight='bold', bbox=dict(boxstyle='round', facecolor=modality_colors[i], alpha=0.8))

        ax.set_xlim(-radius - 2, radius + 2)
        ax.set_ylim(-radius - 2, radius + 2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Pattern Resonance Network', fontsize=14, fontweight='bold', pad=20)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Pattern network saved to {save_path}")

        if show:
            plt.show()

        return fig

    def create_summary_report(
        self,
        crystallized: CrystalizedThought,
        save_path: str
    ):
        """
        Create a comprehensive PDF-style summary report.
        """
        fig = plt.figure(figsize=(11, 14))  # Letter size
        gs = fig.add_gridspec(6, 2, hspace=0.5, wspace=0.3,
                            left=0.1, right=0.9, top=0.95, bottom=0.05)

        # Header
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')
        ax_header.text(
            0.5, 0.7,
            'QUANTUM THOUGHT CRYSTALLIZATION REPORT',
            ha='center', va='center', fontsize=18, fontweight='bold'
        )
        ax_header.text(
            0.5, 0.3,
            f'"{crystallized.original_thought.text}"',
            ha='center', va='center', fontsize=12, style='italic'
        )

        # Metrics table
        ax_metrics = fig.add_subplot(gs[1, :])
        ax_metrics.axis('off')
        metrics_text = "CONSCIOUSNESS METRICS\n" + "─" * 50 + "\n"
        for key, value in crystallized.consciousness_metrics.items():
            metrics_text += f"{key.replace('_', ' ').title():.<40} {value:.3f}\n"
        ax_metrics.text(0.1, 0.9, metrics_text, va='top', fontsize=10, family='monospace')

        # Quantum state visualization
        ax_quantum = fig.add_subplot(gs[2, :])
        self._plot_quantum_state_bars(ax_quantum, crystallized.quantum_state, 'real')

        # Pattern distribution
        ax_patterns = fig.add_subplot(gs[3, 0])
        self._plot_pattern_distribution(ax_patterns, crystallized)

        # Consciousness metrics bars
        ax_consc = fig.add_subplot(gs[3, 1])
        self._plot_consciousness_metrics(ax_consc, crystallized)

        # Physical manifestation
        ax_phys = fig.add_subplot(gs[4:, :], projection='3d')
        self._plot_physical_manifestation_3d(ax_phys, crystallized)

        # Save
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Summary report saved to {save_path}")

        return fig


# Example usage
if __name__ == "__main__":
    import asyncio
    from quantum_thought_crystallizer import crystallize

    async def demo():
        # Create sample crystallization
        result = await crystallize(
            text="Consciousness emerges from quantum coherence in neural networks",
            emotional_valence=0.3,
            intensity=0.9
        )

        # Visualize
        viz = CrystallizerVisualizer()
        viz.visualize_crystallization(result, save_path="crystallization_viz.png")
        viz.create_summary_report(result, save_path="crystallization_report.png")

        print("\n✓ Visualization demo complete!")

    asyncio.run(demo())
