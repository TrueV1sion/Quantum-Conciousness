import asyncio
from datetime import datetime, timedelta
import numpy as np
import plotly.io as pio

from src.gdelt_scenario_analysis import (
    ScenarioAnalyzer, ScenarioConfig, MarketShock, ShockType
)
from src.config.gdelt_config import GDELTIntegrationConfig
from src.quantum_gdelt_circuits import GDELTQuantumCircuitGenerator
from src.visualization.scenario_visualizer import ScenarioVisualizer
from src.gdelt_integration import GDELTEvent, EventType

async def run_commodity_shock_scenario():
    """Example of oil price shock scenario."""
    # Initialize components
    config = ScenarioConfig(
        num_monte_carlo_sims=1000,
        confidence_level=0.95,
        time_horizon_days=60
    )
    gdelt_config = GDELTIntegrationConfig()
    circuit_generator = GDELTQuantumCircuitGenerator(gdelt_config.quantum)
    analyzer = ScenarioAnalyzer(config, gdelt_config, circuit_generator)
    visualizer = ScenarioVisualizer()
    
    # Create sample GDELT events
    events = [
        GDELTEvent(
            event_id="1",
            timestamp=datetime.now(),
            event_type=EventType.ECONOMIC,
            actor1="OPEC",
            actor2="GLOBAL",
            action="REDUCE_PRODUCTION",
            location=(24.467, 54.367),  # Abu Dhabi coordinates
            tone=-3.5,
            relevance=0.9,
            impact_score=0.8
        ),
        GDELTEvent(
            event_id="2",
            timestamp=datetime.now(),
            event_type=EventType.POLITICAL,
            actor1="USA",
            actor2="IRAN",
            action="SANCTION",
            location=(35.689, 51.389),  # Tehran coordinates
            tone=-2.8,
            relevance=0.85,
            impact_score=0.7
        )
    ]
    
    # Define shock scenario
    shock = MarketShock(
        shock_type=ShockType.COMMODITY_PRICE_SPIKE,
        magnitude=0.25,  # 25% price increase
        duration_days=60,
        affected_sectors=[
            "Energy",
            "Transportation",
            "Manufacturing",
            "Utilities"
        ],
        propagation_speed=0.8,
        recovery_rate=0.1
    )
    
    # Run analysis
    results = await analyzer.analyze_shock_scenario(events, shock)
    
    # Create visualization
    dashboard = visualizer.create_dashboard(
        results['simulation_paths'],
        shock,
        results['correlations'],
        results['var'],
        results['expected_shortfall']
    )
    
    # Save dashboard
    pio.write_html(dashboard, 'commodity_shock_dashboard.html')
    return results

async def run_geopolitical_crisis_scenario():
    """Example of geopolitical crisis scenario."""
    config = ScenarioConfig(
        num_monte_carlo_sims=1000,
        time_horizon_days=90
    )
    gdelt_config = GDELTIntegrationConfig()
    circuit_generator = GDELTQuantumCircuitGenerator(gdelt_config.quantum)
    analyzer = ScenarioAnalyzer(config, gdelt_config, circuit_generator)
    visualizer = ScenarioVisualizer()
    
    # Create sample events
    events = [
        GDELTEvent(
            event_id="3",
            timestamp=datetime.now(),
            event_type=EventType.CONFLICT,
            actor1="COUNTRY_A",
            actor2="COUNTRY_B",
            action="MILITARY_ACTION",
            location=(50.450, 30.523),  # Example coordinates
            tone=-8.5,
            relevance=0.95,
            impact_score=0.9
        ),
        GDELTEvent(
            event_id="4",
            timestamp=datetime.now(),
            event_type=EventType.DIPLOMATIC,
            actor1="UN",
            actor2="COUNTRY_A",
            action="SANCTION",
            location=(40.712, -74.006),  # UN coordinates
            tone=-5.5,
            relevance=0.9,
            impact_score=0.85
        )
    ]
    
    # Define shock
    shock = MarketShock(
        shock_type=ShockType.GEOPOLITICAL_CRISIS,
        magnitude=0.3,
        duration_days=90,
        affected_sectors=[
            "Defense",
            "Energy",
            "Financial",
            "Technology",
            "Materials"
        ],
        propagation_speed=0.9,
        recovery_rate=0.05
    )
    
    # Run analysis
    results = await analyzer.analyze_shock_scenario(events, shock)
    
    # Create visualization
    dashboard = visualizer.create_dashboard(
        results['simulation_paths'],
        shock,
        results['correlations'],
        results['var'],
        results['expected_shortfall']
    )
    
    # Save dashboard
    pio.write_html(dashboard, 'geopolitical_crisis_dashboard.html')
    return results

async def run_technology_disruption_scenario():
    """Example of technology disruption scenario."""
    config = ScenarioConfig(
        num_monte_carlo_sims=1000,
        time_horizon_days=180
    )
    gdelt_config = GDELTIntegrationConfig()
    circuit_generator = GDELTQuantumCircuitGenerator(gdelt_config.quantum)
    analyzer = ScenarioAnalyzer(config, gdelt_config, circuit_generator)
    visualizer = ScenarioVisualizer()
    
    # Create sample events
    events = [
        GDELTEvent(
            event_id="5",
            timestamp=datetime.now(),
            event_type=EventType.TECHNOLOGY_DISRUPTION,
            actor1="TECH_COMPANY",
            actor2="GLOBAL",
            action="LAUNCH_INNOVATION",
            location=(37.774, -122.419),  # Silicon Valley coordinates
            tone=7.5,
            relevance=0.95,
            impact_score=0.9
        )
    ]
    
    # Define shock
    shock = MarketShock(
        shock_type=ShockType.TECHNOLOGY_DISRUPTION,
        magnitude=0.4,
        duration_days=180,
        affected_sectors=[
            "Technology",
            "Communication",
            "Financial",
            "Healthcare",
            "Retail"
        ],
        propagation_speed=0.7,
        recovery_rate=0.15
    )
    
    # Run analysis
    results = await analyzer.analyze_shock_scenario(events, shock)
    
    # Create visualization
    dashboard = visualizer.create_dashboard(
        results['simulation_paths'],
        shock,
        results['correlations'],
        results['var'],
        results['expected_shortfall']
    )
    
    # Save dashboard
    pio.write_html(dashboard, 'technology_disruption_dashboard.html')
    return results

async def main():
    """Run all example scenarios."""
    print("Running commodity shock scenario...")
    commodity_results = await run_commodity_shock_scenario()
    print("\nCommodity shock analysis complete.")
    
    print("\nRunning geopolitical crisis scenario...")
    geopolitical_results = await run_geopolitical_crisis_scenario()
    print("\nGeopolitical crisis analysis complete.")
    
    print("\nRunning technology disruption scenario...")
    tech_results = await run_technology_disruption_scenario()
    print("\nTechnology disruption analysis complete.")
    
    return {
        'commodity': commodity_results,
        'geopolitical': geopolitical_results,
        'technology': tech_results
    }

if __name__ == "__main__":
    asyncio.run(main()) 