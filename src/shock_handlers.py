from typing import Dict, List, Optional, Tuple
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from dataclasses import dataclass

from .gdelt_scenario_analysis import MarketShock, ShockType

@dataclass
class ShockParameters:
    """Parameters for shock-specific quantum circuits."""
    amplitude: float
    phase: float
    entanglement: float
    duration_factor: float

class BaseShockHandler:
    """Base class for market shock handlers."""
    
    def add_shock_gates(self, 
                       circuit: QuantumCircuit,
                       q_reg: QuantumRegister,
                       shock: MarketShock) -> None:
        """Add shock-specific quantum gates."""
        raise NotImplementedError

class CommodityShockHandler(BaseShockHandler):
    """Handler for commodity price shock scenarios."""
    
    def add_shock_gates(self, 
                       circuit: QuantumCircuit,
                       q_reg: QuantumRegister,
                       shock: MarketShock) -> None:
        """Add quantum gates for commodity price shocks."""
        params = self._calculate_parameters(shock)
        
        # Add price impact gates
        for i in range(len(q_reg)):
            # Amplitude adjustment for price impact
            circuit.ry(params.amplitude * np.pi, q_reg[i])
            
            # Phase rotation for market sentiment
            circuit.rz(params.phase * np.pi, q_reg[i])
        
        # Add supply-demand entanglement
        for i in range(len(q_reg) - 1):
            circuit.cx(q_reg[i], q_reg[i + 1])
            circuit.rz(params.entanglement * np.pi, q_reg[i + 1])
    
    def _calculate_parameters(self, shock: MarketShock) -> ShockParameters:
        """Calculate quantum parameters for commodity shock."""
        return ShockParameters(
            amplitude=abs(shock.magnitude),
            phase=np.sign(shock.magnitude) * 0.5,
            entanglement=shock.propagation_speed,
            duration_factor=shock.duration_days / 30.0  # Normalized to monthly
        )

class GeopoliticalShockHandler(BaseShockHandler):
    """Handler for geopolitical crisis scenarios."""
    
    def add_shock_gates(self, 
                       circuit: QuantumCircuit,
                       q_reg: QuantumRegister,
                       shock: MarketShock) -> None:
        """Add quantum gates for geopolitical shocks."""
        params = self._calculate_parameters(shock)
        
        # Create superposition for multiple impact channels
        circuit.h(q_reg[0])  # Control qubit
        
        # Add crisis impact gates
        for i in range(1, len(q_reg)):
            # Controlled impact gates
            circuit.cry(params.amplitude * np.pi, q_reg[0], q_reg[i])
            
            # Add uncertainty phase
            circuit.rz(params.phase * np.pi, q_reg[i])
        
        # Add cross-border effect entanglement
        for i in range(1, len(q_reg) - 1):
            circuit.cswap(q_reg[0], q_reg[i], q_reg[i + 1])
    
    def _calculate_parameters(self, shock: MarketShock) -> ShockParameters:
        """Calculate quantum parameters for geopolitical shock."""
        return ShockParameters(
            amplitude=abs(shock.magnitude) * 1.5,  # Amplified for geopolitical events
            phase=0.75,  # High uncertainty
            entanglement=max(shock.propagation_speed * 1.2, 1.0),  # Enhanced propagation
            duration_factor=shock.duration_days / 60.0  # Normalized to bi-monthly
        )

class SupplyChainShockHandler(BaseShockHandler):
    """Handler for supply chain disruption scenarios."""
    
    def add_shock_gates(self, 
                       circuit: QuantumCircuit,
                       q_reg: QuantumRegister,
                       shock: MarketShock) -> None:
        """Add quantum gates for supply chain shocks."""
        params = self._calculate_parameters(shock)
        
        # Create propagation pathway
        for i in range(len(q_reg) - 1):
            # Disruption impact
            circuit.ry(params.amplitude * np.pi, q_reg[i])
            
            # Propagate through supply chain
            circuit.cx(q_reg[i], q_reg[i + 1])
            circuit.rz(params.phase * np.pi, q_reg[i + 1])
        
        # Add feedback loops
        for i in range(len(q_reg) - 2, 0, -1):
            circuit.cx(q_reg[i], q_reg[i - 1])
            circuit.ry(params.entanglement * np.pi / 2, q_reg[i - 1])
    
    def _calculate_parameters(self, shock: MarketShock) -> ShockParameters:
        """Calculate quantum parameters for supply chain shock."""
        return ShockParameters(
            amplitude=abs(shock.magnitude) * 0.8,
            phase=0.6,
            entanglement=shock.propagation_speed * 0.9,
            duration_factor=shock.duration_days / 45.0  # Normalized to 1.5 months
        )

class RegulatoryShockHandler(BaseShockHandler):
    """Handler for regulatory change scenarios."""
    
    def add_shock_gates(self, 
                      circuit: QuantumCircuit,
                      q_reg: QuantumRegister,
                      shock: MarketShock) -> None:
        """Add quantum gates for regulatory shocks."""
        params = self._calculate_parameters(shock)
        
        # Add compliance impact gates
        for i in range(len(q_reg)):
            # Regulatory impact amplitude
            circuit.ry(params.amplitude * np.pi / 2, q_reg[i])
            
            # Compliance phase adjustment
            circuit.rz(params.phase * np.pi, q_reg[i])
        
        # Add sector-wide impact entanglement
        for i in range(0, len(q_reg) - 2, 2):
            # Create regulatory impact propagation
            circuit.cx(q_reg[i], q_reg[i + 1])
            circuit.cx(q_reg[i + 1], q_reg[i + 2])
            circuit.rz(params.entanglement * np.pi / 2, q_reg[i + 2])
    
    def _calculate_parameters(self, shock: MarketShock) -> ShockParameters:
        """Calculate quantum parameters for regulatory shock."""
        return ShockParameters(
            amplitude=abs(shock.magnitude) * 1.2,
            phase=0.8,  # High certainty due to regulatory nature
            entanglement=shock.propagation_speed * 0.7,
            duration_factor=shock.duration_days / 90.0  # Normalized to quarterly
        )

class NaturalDisasterShockHandler(BaseShockHandler):
    """Handler for natural disaster scenarios."""
    
    def add_shock_gates(self, 
                      circuit: QuantumCircuit,
                      q_reg: QuantumRegister,
                      shock: MarketShock) -> None:
        """Add quantum gates for natural disaster shocks."""
        params = self._calculate_parameters(shock)
        
        # Create initial impact superposition
        circuit.h(q_reg[0])
        
        # Add cascading impact gates
        for i in range(1, len(q_reg)):
            # Immediate impact
            circuit.cry(params.amplitude * np.pi, q_reg[0], q_reg[i])
            
            # Infrastructure disruption phase
            circuit.rz(params.phase * np.pi, q_reg[i])
            
            # Recovery dynamics
            if i < len(q_reg) - 1:
                circuit.crz(params.entanglement * np.pi, q_reg[i], q_reg[i + 1])
    
    def _calculate_parameters(self, shock: MarketShock) -> ShockParameters:
        """Calculate quantum parameters for natural disaster shock."""
        return ShockParameters(
            amplitude=abs(shock.magnitude) * 2.0,  # High immediate impact
            phase=0.9,  # High uncertainty
            entanglement=min(shock.propagation_speed * 1.5, 1.0),
            duration_factor=shock.duration_days / 180.0  # Normalized to semi-annual
        )

class TechnologyShockHandler(BaseShockHandler):
    """Handler for technology disruption scenarios."""
    
    def add_shock_gates(self, 
                      circuit: QuantumCircuit,
                      q_reg: QuantumRegister,
                      shock: MarketShock) -> None:
        """Add quantum gates for technology disruption shocks."""
        params = self._calculate_parameters(shock)
        
        # Innovation wave propagation
        for i in range(len(q_reg)):
            # Technology adoption amplitude
            angle = params.amplitude * np.pi * (i + 1) / len(q_reg)
            circuit.ry(angle, q_reg[i])
            
            # Market adaptation phase
            circuit.rz(params.phase * np.pi, q_reg[i])
        
        # Add disruption entanglement
        for i in range(len(q_reg) - 1):
            # Network effect propagation
            circuit.cx(q_reg[i], q_reg[i + 1])
            circuit.rz(params.entanglement * np.pi / 2, q_reg[i + 1])
            
            # Innovation feedback
            if i > 0:
                circuit.swap(q_reg[i - 1], q_reg[i])
    
    def _calculate_parameters(self, shock: MarketShock) -> ShockParameters:
        """Calculate quantum parameters for technology shock."""
        return ShockParameters(
            amplitude=abs(shock.magnitude) * 1.3,
            phase=0.6,  # Moderate uncertainty
            entanglement=shock.propagation_speed * 1.1,
            duration_factor=shock.duration_days / 365.0  # Normalized to annual
        )

class ShockHandlerFactory:
    """Factory for creating appropriate shock handlers."""
    
    _handlers: Dict[ShockType, BaseShockHandler] = {
        ShockType.COMMODITY_PRICE_SPIKE: CommodityShockHandler(),
        ShockType.GEOPOLITICAL_CRISIS: GeopoliticalShockHandler(),
        ShockType.SUPPLY_CHAIN_DISRUPTION: SupplyChainShockHandler(),
        ShockType.REGULATORY_CHANGE: RegulatoryShockHandler(),
        ShockType.NATURAL_DISASTER: NaturalDisasterShockHandler(),
        ShockType.TECHNOLOGY_DISRUPTION: TechnologyShockHandler(),
    }
    
    @classmethod
    def get_handler(cls, shock_type: ShockType) -> BaseShockHandler:
        """Get appropriate handler for shock type."""
        handler = cls._handlers.get(shock_type)
        if handler is None:
            raise ValueError(f"No handler available for shock type: {shock_type}")
        return handler 