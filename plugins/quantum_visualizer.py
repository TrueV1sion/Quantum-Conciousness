import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
from qiskit.quantum_info import Statevector, DensityMatrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

from quantum_processor import QuantumProcessor


class QuantumVisualizer:
    """Real-time Quantum State and Circuit Visualization Plugin."""
    
    def __init__(
        self,
        quantum_processor: QuantumProcessor,
        visualization_backend: str = 'plotly'
    ):
        self.quantum_processor = quantum_processor
        self.visualization_backend = visualization_backend
        self.figure_size = (10, 8)
        
        # Color schemes for different visualization types
        self.color_schemes = {
            'amplitude': px.colors.sequential.Plasma,
            'phase': px.colors.sequential.Viridis,
            'probability': px.colors.sequential.Magma,
            'entanglement': px.colors.sequential.Inferno
        }
    
    def visualize_quantum_state(
        self,
        state: Union[torch.Tensor, Statevector],
        plot_type: str = 'bloch',
        interactive: bool = True
    ) -> go.Figure:
        """Visualize quantum state using different representations."""
        if isinstance(state, torch.Tensor):
            state_vector = Statevector(state.numpy())
        else:
            state_vector = state
        
        if plot_type == 'bloch':
            return self._plot_bloch_sphere(state_vector)
        elif plot_type == 'amplitude':
            return self._plot_amplitude_distribution(state_vector)
        elif plot_type == 'phase':
            return self._plot_phase_distribution(state_vector)
        elif plot_type == 'wigner':
            return self._plot_wigner_function(state_vector)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
    
    def _plot_bloch_sphere(self, state: Statevector) -> go.Figure:
        """Plot quantum state on Bloch sphere."""
        # Calculate Bloch sphere coordinates
        density_matrix = DensityMatrix(state)
        x = np.real(density_matrix.data[0][1] + density_matrix.data[1][0])
        y = np.imag(density_matrix.data[0][1] - density_matrix.data[1][0])
        z = density_matrix.data[0][0] - density_matrix.data[1][1]
        
        # Create Bloch sphere
        fig = go.Figure()
        
        # Add sphere surface
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_surface(
            x=x_sphere,
            y=y_sphere,
            z=z_sphere,
            opacity=0.3,
            showscale=False
        )
        
        # Add state vector
        fig.add_scatter3d(
            x=[0, x],
            y=[0, y],
            z=[0, z],
            mode='lines+markers',
            marker=dict(size=8, color='red'),
            line=dict(color='red', width=3)
        )
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title='Quantum State on Bloch Sphere'
        )
        
        return fig
    
    def _plot_amplitude_distribution(self, state: Statevector) -> go.Figure:
        """Plot amplitude distribution of quantum state."""
        amplitudes = np.abs(state.data)
        basis_states = [f'|{format(i, "b").zfill(int(np.log2(len(state))))}⟩'
                       for i in range(len(state))]
        
        fig = go.Figure(data=[
            go.Bar(
                x=basis_states,
                y=amplitudes,
                marker_color=amplitudes,
                colorscale=self.color_schemes['amplitude']
            )
        ])
        
        fig.update_layout(
            title='Quantum State Amplitude Distribution',
            xaxis_title='Basis States',
            yaxis_title='Amplitude',
            showlegend=False
        )
        
        return fig
    
    def _plot_phase_distribution(self, state: Statevector) -> go.Figure:
        """Plot phase distribution of quantum state."""
        phases = np.angle(state.data)
        basis_states = [f'|{format(i, "b").zfill(int(np.log2(len(state))))}⟩'
                       for i in range(len(state))]
        
        fig = go.Figure(data=[
            go.Bar(
                x=basis_states,
                y=phases,
                marker_color=phases,
                colorscale=self.color_schemes['phase']
            )
        ])
        
        fig.update_layout(
            title='Quantum State Phase Distribution',
            xaxis_title='Basis States',
            yaxis_title='Phase (radians)',
            showlegend=False
        )
        
        return fig
    
    def _plot_wigner_function(self, state: Statevector) -> go.Figure:
        """Plot Wigner quasi-probability distribution."""
        # Calculate Wigner function
        x = np.linspace(-5, 5, 100)
        p = np.linspace(-5, 5, 100)
        X, P = np.meshgrid(x, p)
        
        # Simplified Wigner function calculation
        wigner = np.zeros_like(X)
        for i in range(len(state)):
            wigner += np.real(state.data[i] * np.exp(-X**2 - P**2))
        
        fig = go.Figure(data=[
            go.Surface(
                x=X,
                y=P,
                z=wigner,
                colorscale='Viridis'
            )
        ])
        
        fig.update_layout(
            title='Wigner Quasi-probability Distribution',
            scene=dict(
                xaxis_title='Position',
                yaxis_title='Momentum',
                zaxis_title='Amplitude'
            )
        )
        
        return fig
    
    def visualize_quantum_circuit(
        self,
        circuit: QuantumCircuit,
        style: str = 'mpl'
    ) -> Union[go.Figure, plt.Figure]:
        """Visualize quantum circuit with different styles."""
        if style == 'mpl':
            fig = plt.figure(figsize=self.figure_size)
            circuit_drawer(
                circuit,
                output='mpl',
                style={'backgroundcolor': '#FFFFFF'}
            )
            return fig
        elif style == 'interactive':
            # Convert to plotly figure
            fig = go.Figure()
            
            # Create circuit grid
            n_qubits = circuit.num_qubits
            n_gates = len(circuit.data)
            
            # Add gates as shapes
            for i, instruction in enumerate(circuit.data):
                gate_name = instruction[0].name
                qubits = [q.index for q in instruction[1]]
                
                # Add gate representation
                fig.add_shape(
                    type='rect',
                    x0=i - 0.4,
                    x1=i + 0.4,
                    y0=min(qubits) - 0.4,
                    y1=max(qubits) + 0.4,
                    line=dict(color='blue'),
                    fillcolor='lightblue',
                    opacity=0.7
                )
                
                # Add gate label
                fig.add_annotation(
                    x=i,
                    y=np.mean(qubits),
                    text=gate_name,
                    showarrow=False
                )
            
            # Update layout
            fig.update_layout(
                title='Interactive Quantum Circuit Visualization',
                xaxis_title='Gate Sequence',
                yaxis_title='Qubit Index',
                showlegend=False,
                yaxis=dict(range=[-0.5, n_qubits - 0.5])
            )
            
            return fig
        else:
            raise ValueError(f"Unsupported visualization style: {style}")
    
    def visualize_entanglement(
        self,
        state: Union[torch.Tensor, Statevector]
    ) -> go.Figure:
        """Visualize quantum entanglement structure."""
        if isinstance(state, torch.Tensor):
            state_vector = Statevector(state.numpy())
        else:
            state_vector = state
        
        # Calculate reduced density matrices and correlations
        n_qubits = int(np.log2(len(state_vector)))
        correlations = np.zeros((n_qubits, n_qubits))
        
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                # Calculate two-qubit correlation
                reduced_dm = state_vector.partial_trace([
                    k for k in range(n_qubits) if k not in (i, j)
                ])
                correlations[i, j] = correlations[j, i] = np.abs(
                    np.trace(reduced_dm.data @ reduced_dm.data)
                )
        
        # Create graph visualization
        G = nx.Graph()
        for i in range(n_qubits):
            G.add_node(i)
            for j in range(i + 1, n_qubits):
                if correlations[i, j] > 0.1:  # Threshold for showing edges
                    G.add_edge(i, j, weight=correlations[i, j])
        
        # Get node positions
        pos = nx.spring_layout(G)
        
        # Create plotly figure
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        
        # Add nodes
        node_trace = go.Scatter(
            x=[pos[i][0] for i in G.nodes()],
            y=[pos[i][1] for i in G.nodes()],
            mode='markers+text',
            text=[f'Q{i}' for i in G.nodes()],
            marker=dict(
                size=20,
                color=list(range(n_qubits)),
                colorscale=self.color_schemes['entanglement'],
                line_width=2
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title='Quantum Entanglement Graph',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40)
        )
        
        return fig 