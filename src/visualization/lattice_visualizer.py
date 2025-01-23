import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import numpy as np
from ..meta_cognitive_pipeline import MetaCognitivePipeline


class LatticeVisualizer:
    """Visualization tool for the context lattice."""
    
    def __init__(self):
        self.colors = {
            'language': '#1f77b4',  # blue
            'numerical': '#2ca02c',  # green
            'visual': '#ff7f0e',    # orange
            'audio': '#9467bd'      # purple
        }
        self.edge_colors = {
            'connection': '#cccccc',      # light gray
            'contradiction': '#d62728',    # red
            'resonance': '#17becf'        # cyan
        }

    def visualize_lattice(
        self,
        mcp: MetaCognitivePipeline,
        figsize: Tuple[int, int] = (12, 8),
        show_labels: bool = True,
        show_confidence: bool = True,
        highlight_resonances: bool = True,
        min_confidence: float = 0.0
    ) -> None:
        """Visualize the context lattice."""
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        node_positions = {}
        node_colors = []
        node_sizes = []
        labels = {}
        
        for engine_name, engine in mcp.cognitive_engines.items():
            for context_id, context in engine.ephemeral_contexts.items():
                for node in context.nodes:
                    if node.confidence >= min_confidence:
                        # Add node
                        G.add_node(node.id)
                        
                        # Set node attributes
                        node_colors.append(
                            self.colors.get(node.modality, '#7f7f7f')
                        )
                        node_sizes.append(
                            1000 * node.confidence
                        )
                        
                        if show_labels:
                            label_parts = []
                            label_parts.append(f"Node {node.id}")
                            if show_confidence:
                                label_parts.append(
                                    f"({node.confidence:.2f})"
                                )
                            labels[node.id] = "\n".join(label_parts)
        
        # Add edges
        edge_colors = []
        edge_widths = []
        
        # Add connections
        for node_id, node in G.nodes(data=True):
            for conn_id in node.get('connections', []):
                if conn_id in G:
                    G.add_edge(node_id, conn_id)
                    edge_colors.append(self.edge_colors['connection'])
                    edge_widths.append(1.0)
        
        # Add contradictions
        if 'contradictions' in mcp.results:
            for contr_id, contr in mcp.results['contradictions'].items():
                if (
                    contr['node1'] in G and
                    contr['node2'] in G
                ):
                    G.add_edge(contr['node1'], contr['node2'])
                    edge_colors.append(self.edge_colors['contradiction'])
                    edge_widths.append(2.0)
        
        # Add resonances
        if highlight_resonances:
            for echo in mcp.echoic_heuristics:
                if echo.resonances:
                    for engine, resonance in echo.resonances.items():
                        if resonance > 0.7:
                            G.add_edge(echo.source_id, echo.target_id)
                            edge_colors.append(self.edge_colors['resonance'])
                            edge_widths.append(1.5)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Calculate layout
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_widths
        )
        
        # Draw labels
        if show_labels:
            nx.draw_networkx_labels(G, pos, labels)
        
        # Add legend
        legend_elements = []
        
        # Node types
        for modality, color in self.colors.items():
            legend_elements.append(
                plt.Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    markerfacecolor=color,
                    markersize=10,
                    label=f'{modality} node'
                )
            )
        
        # Edge types
        for edge_type, color in self.edge_colors.items():
            legend_elements.append(
                plt.Line2D(
                    [0], [0],
                    color=color,
                    lw=2,
                    label=f'{edge_type}'
                )
            )
        
        plt.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )
        
        plt.title('Context Lattice Visualization')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_resonance_matrix(
        self,
        mcp: MetaCognitivePipeline,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """Plot resonance matrix between cognitive engines."""
        # Get unique engines
        engines = list(mcp.cognitive_engines.keys())
        n_engines = len(engines)
        
        # Create resonance matrix
        matrix = np.zeros((n_engines, n_engines))
        
        # Fill matrix with resonance values
        for i, src_engine in enumerate(engines):
            for j, tgt_engine in enumerate(engines):
                resonances = []
                for echo in mcp.echoic_heuristics:
                    if (
                        echo.source_engine == src_engine and
                        tgt_engine in echo.resonances
                    ):
                        resonances.append(echo.resonances[tgt_engine])
                if resonances:
                    matrix[i, j] = np.mean(resonances)
        
        # Plot matrix
        plt.figure(figsize=figsize)
        plt.imshow(matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        
        # Add labels
        plt.xticks(
            range(n_engines),
            engines,
            rotation=45,
            ha='right'
        )
        plt.yticks(range(n_engines), engines)
        
        # Add colorbar
        plt.colorbar(label='Average Resonance')
        
        plt.title('Resonance Matrix')
        plt.tight_layout()
        plt.show()

    def plot_confidence_distribution(
        self,
        mcp: MetaCognitivePipeline,
        figsize: Tuple[int, int] = (10, 6),
        bins: int = 20
    ) -> None:
        """Plot confidence distribution for each modality."""
        # Collect confidences by modality
        confidences = {
            modality: [] for modality in self.colors.keys()
        }
        
        for engine in mcp.cognitive_engines.values():
            for context in engine.ephemeral_contexts.values():
                for node in context.nodes:
                    if node.modality in confidences:
                        confidences[node.modality].append(node.confidence)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Plot histogram for each modality
        for modality, conf_values in confidences.items():
            if conf_values:
                plt.hist(
                    conf_values,
                    bins=bins,
                    alpha=0.5,
                    label=modality,
                    color=self.colors[modality]
                )
        
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Node Confidence Distribution by Modality')
        plt.legend()
        plt.tight_layout()
        plt.show() 