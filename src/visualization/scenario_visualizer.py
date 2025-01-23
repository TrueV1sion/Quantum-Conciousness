import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta

from ..gdelt_scenario_analysis import MarketShock, ShockType

class ScenarioVisualizer:
    """Visualizes scenario analysis results."""
    
    def __init__(self, dark_mode: bool = True):
        self.dark_mode = dark_mode
        self._setup_theme()
    
    def _setup_theme(self):
        """Setup visualization theme."""
        self.colors = {
            'background': '#1f2630' if self.dark_mode else '#ffffff',
            'text': '#ffffff' if self.dark_mode else '#000000',
            'grid': '#374151' if self.dark_mode else '#e5e7eb',
            'primary': '#3b82f6',
            'secondary': '#10b981',
            'accent': '#f59e0b',
            'danger': '#ef4444'
        }
        
        self.layout_template = {
            'plot_bgcolor': self.colors['background'],
            'paper_bgcolor': self.colors['background'],
            'font': {'color': self.colors['text']},
            'xaxis': {'gridcolor': self.colors['grid']},
            'yaxis': {'gridcolor': self.colors['grid']}
        }
    
    def plot_simulation_paths(self,
                            simulation_results: np.ndarray,
                            shock: MarketShock,
                            confidence_intervals: List[float] = [0.95, 0.99]) -> go.Figure:
        """Plot Monte Carlo simulation paths with confidence intervals."""
        fig = go.Figure()
        
        # Calculate time axis
        time_axis = np.arange(simulation_results.shape[1])
        
        # Calculate mean path
        mean_path = np.mean(simulation_results, axis=0)
        
        # Add mean path
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=mean_path,
            name='Mean Impact',
            line={'color': self.colors['primary'], 'width': 2},
            mode='lines'
        ))
        
        # Add confidence intervals
        for ci in confidence_intervals:
            lower = np.percentile(simulation_results, (1 - ci) * 50, axis=0)
            upper = np.percentile(simulation_results, 100 - (1 - ci) * 50, axis=0)
            
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=upper,
                mode='lines',
                line={'width': 0},
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=lower,
                name=f'{ci*100}% CI',
                fill='tonexty',
                mode='lines',
                line={'width': 0},
                fillcolor=f'rgba{tuple(int(self.colors["primary"][i:i+2], 16) for i in (1, 3, 5)) + (0.2,)}'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Impact Simulation: {shock.shock_type.name}',
            xaxis_title='Days',
            yaxis_title='Impact Magnitude',
            **self.layout_template
        )
        
        return fig
    
    def plot_sector_correlations(self,
                               correlations: Dict[str, float],
                               min_correlation: float = 0.3) -> go.Figure:
        """Plot sector correlation heatmap."""
        # Extract unique sectors
        sectors = set()
        for pair in correlations.keys():
            s1, s2 = pair.split('-')
            sectors.add(s1)
            sectors.add(s2)
        sectors = sorted(list(sectors))
        
        # Create correlation matrix
        n = len(sectors)
        matrix = np.zeros((n, n))
        for i, s1 in enumerate(sectors):
            for j, s2 in enumerate(sectors):
                if i == j:
                    matrix[i,j] = 1.0
                else:
                    key = f"{s1}-{s2}" if s1 < s2 else f"{s2}-{s1}"
                    matrix[i,j] = matrix[j,i] = correlations.get(key, 0.0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=sectors,
            y=sectors,
            colorscale='RdBu',
            zmid=0,
            text=np.round(matrix, 2),
            texttemplate='%{text}',
            textfont={'color': 'white'},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title='Sector Correlations',
            **self.layout_template
        )
        
        return fig
    
    def plot_risk_metrics(self,
                         var: float,
                         es: float,
                         simulation_results: np.ndarray) -> go.Figure:
        """Plot risk metrics distribution."""
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=['Return Distribution', 'Risk Metrics'])
        
        # Plot return distribution
        fig.add_trace(
            go.Histogram(
                x=simulation_results[:, -1],
                name='Final Returns',
                nbinsx=50,
                marker_color=self.colors['primary']
            ),
            row=1, col=1
        )
        
        # Add VaR and ES lines
        fig.add_vline(x=var, line_dash="dash", line_color=self.colors['danger'],
                     annotation_text="VaR", row=1, col=1)
        fig.add_vline(x=es, line_dash="dash", line_color=self.colors['accent'],
                     annotation_text="ES", row=1, col=1)
        
        # Plot risk metrics over time
        var_series = np.percentile(simulation_results, 5, axis=0)
        es_series = np.array([
            np.mean(simulation_results[simulation_results[:, t] <= var_series[t], t])
            for t in range(simulation_results.shape[1])
        ])
        
        fig.add_trace(
            go.Scatter(y=var_series, name='VaR (5%)',
                      line={'color': self.colors['danger']}),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=es_series, name='Expected Shortfall',
                      line={'color': self.colors['accent']}),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            **self.layout_template
        )
        
        return fig
    
    def create_dashboard(self,
                        simulation_results: np.ndarray,
                        shock: MarketShock,
                        correlations: Dict[str, float],
                        var: float,
                        es: float) -> go.Figure:
        """Create a comprehensive dashboard of all visualizations."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Impact Simulation',
                'Sector Correlations',
                'Return Distribution',
                'Risk Metrics Evolution'
            ],
            specs=[[{'type': 'scatter'}, {'type': 'heatmap'}],
                  [{'type': 'histogram'}, {'type': 'scatter'}]]
        )
        
        # Add all plots
        paths_fig = self.plot_simulation_paths(simulation_results, shock)
        corr_fig = self.plot_sector_correlations(correlations)
        risk_fig = self.plot_risk_metrics(var, es, simulation_results)
        
        # Combine all traces
        for trace in paths_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        fig.add_trace(corr_fig.data[0], row=1, col=2)
        
        for trace in risk_fig.data[:1]:  # Histogram
            fig.add_trace(trace, row=2, col=1)
        
        for trace in risk_fig.data[1:]:  # Risk metrics
            fig.add_trace(trace, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            showlegend=True,
            title_text=f"Scenario Analysis Dashboard: {shock.shock_type.name}",
            **self.layout_template
        )
        
        return fig 