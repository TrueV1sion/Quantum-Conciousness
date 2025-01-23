"""
Advanced Selection Mechanisms for Quantum-Inspired Optimization.

This module implements sophisticated selection strategies combining quantum principles
with evolutionary algorithms for better optimization.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from quantum_inspired_optimization import Solution, SelectionMethod

@dataclass
class SelectionMetrics:
    """Metrics used for advanced selection strategies."""
    diversity: float
    fitness: float
    age: int
    novelty: float
    pareto_strength: float = 0.0
    crowding_distance: float = 0.0

class QuantumInspiredSelection:
    """Advanced selection mechanisms using quantum-inspired principles."""
    
    def __init__(
        self,
        population_size: int,
        chromosome_size: int,
        selection_pressure: float = 0.7,
        diversity_weight: float = 0.3
    ):
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.selection_pressure = selection_pressure
        self.diversity_weight = diversity_weight
        self.generation = 0
        self.solution_archive = []
        
    def quantum_tournament(
        self,
        solutions: List[Solution],
        tournament_size: int = 3,
        num_winners: int = 1
    ) -> List[Solution]:
        """
        Quantum-inspired tournament selection using superposition states.
        
        Args:
            solutions: List of candidate solutions
            tournament_size: Number of solutions in each tournament
            num_winners: Number of winners to select
            
        Returns:
            Selected solutions
        """
        winners = []
        metrics = self._calculate_metrics(solutions)
        
        for _ in range(num_winners):
            # Select tournament participants
            tournament = np.random.choice(
                len(solutions),
                size=tournament_size,
                replace=False
            )
            
            # Create quantum superposition of tournament participants
            superposition = self._create_superposition(
                [metrics[i] for i in tournament]
            )
            
            # Measure superposition to select winner
            winner_idx = self._measure_superposition(superposition)
            winners.append(solutions[tournament[winner_idx]])
        
        return winners
    
    def _create_superposition(
        self,
        metrics: List[SelectionMetrics]
    ) -> np.ndarray:
        """Create quantum superposition state from metrics."""
        # Calculate amplitudes based on metrics
        amplitudes = np.array([
            self._calculate_amplitude(m) for m in metrics
        ])
        
        # Normalize amplitudes
        return amplitudes / np.sqrt(np.sum(amplitudes ** 2))
    
    def _calculate_amplitude(self, metrics: SelectionMetrics) -> float:
        """Calculate quantum amplitude based on selection metrics."""
        # Combine multiple objectives with weights
        amplitude = (
            self.selection_pressure * metrics.fitness +
            self.diversity_weight * metrics.diversity +
            0.1 * metrics.novelty +
            0.1 * metrics.pareto_strength +
            0.1 * metrics.crowding_distance
        )
        return np.abs(amplitude)
    
    def _measure_superposition(self, superposition: np.ndarray) -> int:
        """Perform quantum measurement on superposition state."""
        probabilities = np.abs(superposition) ** 2
        return np.random.choice(len(superposition), p=probabilities)
    
    def adaptive_selection(
        self,
        solutions: List[Solution],
        generation: int
    ) -> List[Solution]:
        """
        Adaptive selection mechanism that changes strategy based on search progress.
        
        Args:
            solutions: List of candidate solutions
            generation: Current generation number
            
        Returns:
            Selected solutions
        """
        self.generation = generation
        metrics = self._calculate_metrics(solutions)
        
        # Determine selection strategy based on search state
        diversity = self._calculate_population_diversity(solutions)
        convergence = self._estimate_convergence(solutions)
        
        if diversity < 0.2:  # Low diversity
            return self._diversity_focused_selection(solutions, metrics)
        elif convergence > 0.8:  # Near convergence
            return self._exploitation_focused_selection(solutions, metrics)
        else:  # Balanced exploration/exploitation
            return self._balanced_selection(solutions, metrics)
    
    def _calculate_metrics(self, solutions: List[Solution]) -> List[SelectionMetrics]:
        """Calculate comprehensive metrics for each solution."""
        metrics = []
        for i, solution in enumerate(solutions):
            diversity = self._calculate_diversity(solution, solutions)
            novelty = self._calculate_novelty(solution)
            pareto_strength = self._calculate_pareto_strength(solution, solutions)
            
            metrics.append(SelectionMetrics(
                diversity=diversity,
                fitness=np.mean(solution.objectives),
                age=self.generation,
                novelty=novelty,
                pareto_strength=pareto_strength,
                crowding_distance=solution.crowding_distance
            ))
        return metrics
    
    def _calculate_diversity(
        self,
        solution: Solution,
        population: List[Solution]
    ) -> float:
        """Calculate diversity contribution of a solution."""
        distances = []
        for other in population:
            if other is not solution:
                distance = np.mean([
                    abs(a - b)
                    for a, b in zip(solution.binary, other.binary)
                ])
                distances.append(distance)
        return np.mean(distances) if distances else 0.0
    
    def _calculate_novelty(self, solution: Solution) -> float:
        """Calculate novelty of a solution compared to archive."""
        if not self.solution_archive:
            return 1.0
        
        distances = []
        for archived in self.solution_archive[-100:]:  # Consider last 100 solutions
            distance = np.mean([
                abs(a - b)
                for a, b in zip(solution.binary, archived.binary)
            ])
            distances.append(distance)
        return np.mean(distances)
    
    def _calculate_pareto_strength(
        self,
        solution: Solution,
        population: List[Solution]
    ) -> float:
        """Calculate Pareto strength of a solution."""
        dominated_count = 0
        for other in population:
            if self._dominates(solution.objectives, other.objectives):
                dominated_count += 1
        return dominated_count / len(population)
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2."""
        at_least_one_better = False
        for v1, v2 in zip(obj1, obj2):
            if v1 < v2:
                return False
            if v1 > v2:
                at_least_one_better = True
        return at_least_one_better
    
    def _calculate_population_diversity(self, solutions: List[Solution]) -> float:
        """Calculate overall population diversity."""
        total_diversity = 0
        for i, sol1 in enumerate(solutions):
            for j, sol2 in enumerate(solutions[i+1:], i+1):
                diversity = np.mean([
                    abs(a - b)
                    for a, b in zip(sol1.binary, sol2.binary)
                ])
                total_diversity += diversity
        pairs = (len(solutions) * (len(solutions) - 1)) / 2
        return total_diversity / pairs if pairs > 0 else 0
    
    def _estimate_convergence(self, solutions: List[Solution]) -> float:
        """Estimate convergence of the population."""
        if not self.solution_archive:
            return 0.0
        
        # Compare current best with historical best
        current_best = max(solutions, key=lambda s: np.mean(s.objectives))
        historical_best = max(self.solution_archive, key=lambda s: np.mean(s.objectives))
        
        # Calculate relative improvement
        current_fitness = np.mean(current_best.objectives)
        historical_fitness = np.mean(historical_best.objectives)
        
        if historical_fitness == 0:
            return 0.0
        
        improvement = (current_fitness - historical_fitness) / abs(historical_fitness)
        return 1.0 - min(1.0, abs(improvement))
    
    def _diversity_focused_selection(
        self,
        solutions: List[Solution],
        metrics: List[SelectionMetrics]
    ) -> List[Solution]:
        """Selection focused on maintaining diversity."""
        selected = []
        
        # Sort by diversity
        indices = np.argsort([m.diversity for m in metrics])[::-1]
        
        # Select top diverse solutions
        diverse_count = int(self.population_size * 0.7)
        selected.extend([solutions[i] for i in indices[:diverse_count]])
        
        # Fill remaining slots with quantum tournament selection
        remaining = self.population_size - len(selected)
        if remaining > 0:
            selected.extend(
                self.quantum_tournament(
                    solutions,
                    tournament_size=3,
                    num_winners=remaining
                )
            )
        
        return selected
    
    def _exploitation_focused_selection(
        self,
        solutions: List[Solution],
        metrics: List[SelectionMetrics]
    ) -> List[Solution]:
        """Selection focused on exploitation of good solutions."""
        selected = []
        
        # Sort by fitness
        indices = np.argsort([m.fitness for m in metrics])[::-1]
        
        # Select top solutions
        elite_count = int(self.population_size * 0.3)
        selected.extend([solutions[i] for i in indices[:elite_count]])
        
        # Select remaining using quantum tournament with high pressure
        remaining = self.population_size - len(selected)
        if remaining > 0:
            selected.extend(
                self.quantum_tournament(
                    solutions,
                    tournament_size=5,  # Larger tournament size for more pressure
                    num_winners=remaining
                )
            )
        
        return selected
    
    def _balanced_selection(
        self,
        solutions: List[Solution],
        metrics: List[SelectionMetrics]
    ) -> List[Solution]:
        """Balanced selection between exploration and exploitation."""
        selected = []
        
        # Sort by combined metric
        combined_metrics = [
            m.fitness * (1 - self.diversity_weight) +
            m.diversity * self.diversity_weight
            for m in metrics
        ]
        indices = np.argsort(combined_metrics)[::-1]
        
        # Select top solutions
        elite_count = int(self.population_size * 0.2)
        selected.extend([solutions[i] for i in indices[:elite_count]])
        
        # Select remaining using quantum tournament
        remaining = self.population_size - len(selected)
        if remaining > 0:
            selected.extend(
                self.quantum_tournament(
                    solutions,
                    tournament_size=3,
                    num_winners=remaining
                )
            )
        
        return selected

def example_selection():
    """Example usage of advanced selection mechanisms."""
    # Create dummy solutions
    solutions = [
        Solution(
            binary=[np.random.randint(2) for _ in range(10)],
            objectives=[np.random.random() for _ in range(3)],
            crowding_distance=np.random.random()
        )
        for _ in range(50)
    ]
    
    # Initialize selection mechanism
    selector = QuantumInspiredSelection(
        population_size=20,
        chromosome_size=10
    )
    
    # Try different selection methods
    print("Quantum Tournament Selection:")
    winners = selector.quantum_tournament(solutions, tournament_size=3, num_winners=5)
    print(f"Selected {len(winners)} solutions")
    
    print("\nAdaptive Selection:")
    selected = selector.adaptive_selection(solutions, generation=10)
    print(f"Selected {len(selected)} solutions")
    
    # Show diversity metrics
    diversity = selector._calculate_population_diversity(solutions)
    print(f"\nPopulation diversity: {diversity:.3f}")

if __name__ == "__main__":
    example_selection() 