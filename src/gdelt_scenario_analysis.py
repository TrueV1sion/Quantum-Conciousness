import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum, auto
from datetime import datetime, timedelta

from .gdelt_integration import GDELTEvent, EventType
from .quantum_gdelt_circuits import GDELTQuantumCircuitGenerator
from .config.gdelt_config import GDELTIntegrationConfig

@dataclass
class ScenarioConfig:
    """Configuration for scenario analysis."""
    num_monte_carlo_sims: int = 1000
    confidence_level: float = 0.95
    time_horizon_days: int = 30
    shock_magnitude_range: Tuple[float, float] = (-0.3, 0.3)  # -30% to +30%
    volatility_scaling: float = 1.5
    correlation_threshold: float = 0.7

class ShockType(Enum):
    """Types of market shocks to simulate."""
    COMMODITY_PRICE_SPIKE = auto()
    SUPPLY_CHAIN_DISRUPTION = auto()
    GEOPOLITICAL_CRISIS = auto()
    REGULATORY_CHANGE = auto()
    NATURAL_DISASTER = auto()
    TECHNOLOGY_DISRUPTION = auto()

@dataclass
class MarketShock:
    """Represents a market shock scenario."""
    shock_type: ShockType
    magnitude: float
    duration_days: int
    affected_sectors: List[str]
    propagation_speed: float  # 0 to 1, how quickly shock spreads
    recovery_rate: float      # 0 to 1, how quickly market recovers

class ScenarioAnalyzer:
    """Analyzes market scenarios using GDELT data and quantum computing."""
    
    def __init__(self, 
                config: ScenarioConfig,
                gdelt_config: GDELTIntegrationConfig,
                circuit_generator: GDELTQuantumCircuitGenerator):
        self.config = config
        self.gdelt_config = gdelt_config
        self.circuit_generator = circuit_generator
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    async def analyze_shock_scenario(self,
                                   events: List[GDELTEvent],
                                   shock: MarketShock) -> Dict[str, np.ndarray]:
        """Analyze potential market impact of a shock scenario."""
        # Encode events and shock into quantum states
        event_states = await self._prepare_quantum_states(events, shock)
        
        # Run Monte Carlo simulations
        simulation_results = await self._run_monte_carlo(event_states, shock)
        
        # Calculate risk metrics
        var = self._calculate_var(simulation_results)
        es = self._calculate_expected_shortfall(simulation_results)
        
        # Analyze sector correlations
        correlations = self._analyze_sector_correlations(simulation_results, shock)
        
        return {
            'var': var,
            'expected_shortfall': es,
            'correlations': correlations,
            'simulation_paths': simulation_results
        }
    
    async def _prepare_quantum_states(self,
                                    events: List[GDELTEvent],
                                    shock: MarketShock) -> torch.Tensor:
        """Prepare quantum states for scenario analysis."""
        # Create quantum circuit for shock scenario
        circuit, _ = self.circuit_generator.create_event_encoding_circuit(events)
        
        # Add shock-specific gates
        self._add_shock_gates(circuit, shock)
        
        # Run circuit and get quantum states
        return await self._execute_circuit(circuit)
    
    def _add_shock_gates(self, circuit, shock: MarketShock) -> None:
        """Add quantum gates specific to the shock type."""
        if shock.shock_type == ShockType.COMMODITY_PRICE_SPIKE:
            self._add_commodity_shock_gates(circuit, shock)
        elif shock.shock_type == ShockType.GEOPOLITICAL_CRISIS:
            self._add_geopolitical_shock_gates(circuit, shock)
        # Add other shock types...
    
    async def _run_monte_carlo(self,
                             event_states: torch.Tensor,
                             shock: MarketShock) -> np.ndarray:
        """Run Monte Carlo simulations for scenario analysis."""
        results = []
        for _ in range(self.config.num_monte_carlo_sims):
            # Generate random path
            path = self._generate_shock_path(shock)
            
            # Apply quantum noise and decoherence
            noisy_states = self._apply_quantum_noise(event_states)
            
            # Simulate market response
            market_response = self._simulate_market_response(noisy_states, path)
            
            results.append(market_response)
        
        return np.array(results)
    
    def _generate_shock_path(self, shock: MarketShock) -> np.ndarray:
        """Generate a random path for shock propagation."""
        time_steps = self.config.time_horizon_days
        path = np.zeros(time_steps)
        
        # Initial shock
        path[0] = shock.magnitude
        
        # Propagation phase
        for t in range(1, time_steps):
            # Add stochastic component
            random_factor = np.random.normal(0, 0.1)
            
            # Calculate shock decay
            decay = np.exp(-shock.recovery_rate * t)
            
            # Update path
            path[t] = path[t-1] * decay * (1 + random_factor)
            
            # Apply propagation speed
            path[t] *= (1 - np.exp(-shock.propagation_speed * t))
        
        return path
    
    def _apply_quantum_noise(self, states: torch.Tensor) -> torch.Tensor:
        """Apply quantum noise to simulate decoherence."""
        # Add amplitude damping
        gamma = 0.1  # damping parameter
        noise = torch.randn_like(states) * np.sqrt(gamma)
        noisy_states = states + noise
        
        # Normalize
        return noisy_states / torch.norm(noisy_states, dim=-1, keepdim=True)
    
    def _simulate_market_response(self,
                                states: torch.Tensor,
                                shock_path: np.ndarray) -> np.ndarray:
        """Simulate market response to shock scenario."""
        # Convert quantum states to market factors
        market_factors = states.cpu().numpy()
        
        # Initialize response array
        response = np.zeros_like(shock_path)
        
        # Simulate market dynamics
        for t in range(len(shock_path)):
            # Base response from shock
            response[t] = shock_path[t]
            
            # Add market factor influence
            factor_influence = np.mean(market_factors) * self.config.volatility_scaling
            
            # Add mean reversion
            if t > 0:
                mean_reversion = 0.1 * (0 - response[t-1])
                response[t] += mean_reversion
            
            # Add factor influence
            response[t] += factor_influence
        
        return response
    
    def _calculate_var(self, simulation_results: np.ndarray) -> float:
        """Calculate Value at Risk."""
        return np.percentile(simulation_results, 
                           (1 - self.config.confidence_level) * 100)
    
    def _calculate_expected_shortfall(self, simulation_results: np.ndarray) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        var = self._calculate_var(simulation_results)
        return np.mean(simulation_results[simulation_results <= var])
    
    def _analyze_sector_correlations(self,
                                   simulation_results: np.ndarray,
                                   shock: MarketShock) -> Dict[str, float]:
        """Analyze correlations between affected sectors."""
        correlations = {}
        n_sectors = len(shock.affected_sectors)
        
        # Generate sector-specific responses
        sector_responses = np.random.randn(n_sectors, len(simulation_results))
        
        # Calculate correlations
        for i, sector1 in enumerate(shock.affected_sectors):
            for j, sector2 in enumerate(shock.affected_sectors[i+1:], i+1):
                corr = np.corrcoef(sector_responses[i], sector_responses[j])[0, 1]
                if abs(corr) >= self.config.correlation_threshold:
                    correlations[f"{sector1}-{sector2}"] = corr 