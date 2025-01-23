"""
Quantum-Inspired Optimization Algorithms (QIOAs)

This module implements various quantum-inspired optimization algorithms that simulate
quantum computing principles on classical hardware for solving optimization problems.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import random
from abc import ABC, abstractmethod
from enum import Enum

class SelectionMethod(Enum):
    """Available selection methods for reference solutions."""
    RANDOM = "random"
    TOURNAMENT = "tournament"
    CROWDING_DISTANCE = "crowding_distance"
    WEIGHTED_RANDOM = "weighted_random"

@dataclass
class QBit:
    """Represents a quantum-inspired bit with rotation angle and probability amplitude."""
    angle: float  # Rotation angle
    prob: float   # Probability amplitude
    
    def observe(self) -> int:
        """Collapse the quantum state to either 0 or 1 based on probability."""
        return 1 if random.random() < np.sin(self.angle) ** 2 else 0

class QuantumChromosome:
    """Represents a quantum chromosome consisting of multiple QBits."""
    def __init__(self, size: int):
        self.size = size
        self.qbits = [QBit(angle=np.pi/4, prob=1/np.sqrt(2)) for _ in range(size)]
    
    def measure(self) -> List[int]:
        """Measure all QBits in the chromosome."""
        return [qbit.observe() for qbit in self.qbits]
    
    def update(self, best_solution: List[int], learning_rate: float = 0.1):
        """Update QBit angles based on the best solution found."""
        for i, (qbit, best_bit) in enumerate(zip(self.qbits, best_solution)):
            if best_bit == 1:
                qbit.angle += learning_rate
            else:
                qbit.angle -= learning_rate
            qbit.angle = np.clip(qbit.angle, 0, np.pi/2)
            qbit.prob = np.sin(qbit.angle)

@dataclass
class Solution:
    """Represents a solution with its binary representation and objective values."""
    binary: List[int]
    objectives: List[float]
    crowding_distance: float = 0.0  # For maintaining diversity

class ParetoFront:
    """Manages the Pareto front for multi-objective optimization."""
    def __init__(self, maximize: List[bool]):
        self.maximize = maximize
        self.solutions: List[Solution] = []
    
    def dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2."""
        at_least_one_better = False
        for v1, v2, max_obj in zip(obj1, obj2, self.maximize):
            if max_obj:
                if v1 < v2:
                    return False
                if v1 > v2:
                    at_least_one_better = True
            else:
                if v1 > v2:
                    return False
                if v1 < v2:
                    at_least_one_better = True
        return at_least_one_better
    
    def calculate_crowding_distances(self):
        """Calculate crowding distances for diversity preservation."""
        num_solutions = len(self.solutions)
        if num_solutions <= 2:
            for sol in self.solutions:
                sol.crowding_distance = float('inf')
            return
        
        num_objectives = len(self.maximize)
        for sol in self.solutions:
            sol.crowding_distance = 0
        
        for obj_idx in range(num_objectives):
            # Sort solutions by current objective
            self.solutions.sort(key=lambda x: x.objectives[obj_idx])
            
            # Set infinite distance to boundary points
            self.solutions[0].crowding_distance = float('inf')
            self.solutions[-1].crowding_distance = float('inf')
            
            # Calculate crowding distances
            obj_range = (
                self.solutions[-1].objectives[obj_idx] -
                self.solutions[0].objectives[obj_idx]
            )
            if obj_range == 0:
                continue
            
            for i in range(1, num_solutions - 1):
                distance = (
                    self.solutions[i + 1].objectives[obj_idx] -
                    self.solutions[i - 1].objectives[obj_idx]
                ) / obj_range
                self.solutions[i].crowding_distance += distance
    
    def update(self, solution: Solution):
        """Update Pareto front with new solution."""
        is_dominated = False
        
        # Remove solutions dominated by the new one
        self.solutions = [
            s for s in self.solutions
            if not self.dominates(solution.objectives, s.objectives)
        ]
        
        # Add new solution if it's not dominated
        for s in self.solutions:
            if self.dominates(s.objectives, solution.objectives):
                is_dominated = True
                break
        
        if not is_dominated:
            self.solutions.append(solution)
            self.calculate_crowding_distances()

class MultiObjectiveQIEA:
    """Quantum-Inspired Evolutionary Algorithm for multi-objective optimization."""
    def __init__(
        self,
        chromosome_size: int,
        population_size: int,
        objective_funcs: List[Callable[[List[int]], float]],
        maximize: List[bool],
        selection_method: SelectionMethod = SelectionMethod.CROWDING_DISTANCE,
        tournament_size: int = 3
    ):
        self.chromosome_size = chromosome_size
        self.population_size = population_size
        self.objective_funcs = objective_funcs
        self.maximize = maximize
        self.population = [QuantumChromosome(chromosome_size) for _ in range(population_size)]
        self.pareto_front = ParetoFront(maximize)
        self.selection_method = selection_method
        self.tournament_size = tournament_size
    
    def evaluate_solution(self, binary: List[int]) -> List[float]:
        """Evaluate all objectives for a solution."""
        return [f(binary) for f in self.objective_funcs]
    
    def tournament_selection(self) -> List[int]:
        """Select solution using tournament selection."""
        if len(self.pareto_front.solutions) < self.tournament_size:
            return self.random_selection()
        
        tournament = random.sample(self.pareto_front.solutions, self.tournament_size)
        winner = max(tournament, key=lambda x: x.crowding_distance)
        return winner.binary
    
    def crowding_distance_selection(self) -> List[int]:
        """Select solution based on crowding distance."""
        if not self.pareto_front.solutions:
            return self.random_selection()
        
        # Use roulette wheel selection based on crowding distances
        total_distance = sum(s.crowding_distance for s in self.pareto_front.solutions)
        if total_distance == 0:
            return random.choice(self.pareto_front.solutions).binary
        
        r = random.uniform(0, total_distance)
        current_sum = 0
        for solution in self.pareto_front.solutions:
            current_sum += solution.crowding_distance
            if current_sum >= r:
                return solution.binary
        
        return self.pareto_front.solutions[-1].binary
    
    def weighted_random_selection(self) -> List[int]:
        """Select solution using weighted random selection based on non-domination rank."""
        if not self.pareto_front.solutions:
            return self.random_selection()
        
        weights = [1 / (i + 1) for i in range(len(self.pareto_front.solutions))]
        return random.choices(self.pareto_front.solutions, weights=weights, k=1)[0].binary
    
    def random_selection(self) -> List[int]:
        """Fallback random selection."""
        return [random.randint(0, 1) for _ in range(self.chromosome_size)]
    
    def select_reference_solution(self) -> List[int]:
        """Select a reference solution using the specified method."""
        if self.selection_method == SelectionMethod.TOURNAMENT:
            return self.tournament_selection()
        elif self.selection_method == SelectionMethod.CROWDING_DISTANCE:
            return self.crowding_distance_selection()
        elif self.selection_method == SelectionMethod.WEIGHTED_RANDOM:
            return self.weighted_random_selection()
        else:  # SelectionMethod.RANDOM
            return random.choice(self.pareto_front.solutions).binary if self.pareto_front.solutions else self.random_selection()
    
    def optimize(
        self,
        generations: int,
        learning_rate: float = 0.1
    ) -> List[Solution]:
        """Run the multi-objective quantum-inspired evolutionary algorithm."""
        for generation in range(generations):
            # Measure all chromosomes and evaluate objectives
            for chromosome in self.population:
                binary = chromosome.measure()
                objectives = self.evaluate_solution(binary)
                self.pareto_front.update(Solution(binary, objectives))
            
            # Update quantum chromosomes using selected reference solutions
            for chromosome in self.population:
                reference_solution = self.select_reference_solution()
                chromosome.update(reference_solution, learning_rate)
        
        return self.pareto_front.solutions

# Original QIEA class remains unchanged for backward compatibility
class QIEA:
    """Quantum-Inspired Evolutionary Algorithm for single-objective optimization."""
    def __init__(
        self,
        chromosome_size: int,
        population_size: int,
        fitness_func: Callable[[List[int]], float],
        maximize: bool = True
    ):
        self.chromosome_size = chromosome_size
        self.population_size = population_size
        self.fitness_func = fitness_func
        self.maximize = maximize
        self.population = [QuantumChromosome(chromosome_size) for _ in range(population_size)]
        self.best_solution = None
        self.best_fitness = float('-inf') if maximize else float('inf')
    
    def evaluate_solution(self, solution: List[int]) -> float:
        """Evaluate the fitness of a solution."""
        return self.fitness_func(solution)
    
    def update_best_solution(self, solution: List[int], fitness: float):
        """Update the best solution if the current one is better."""
        if self.best_solution is None or (
            (self.maximize and fitness > self.best_fitness) or
            (not self.maximize and fitness < self.best_fitness)
        ):
            self.best_solution = solution.copy()
            self.best_fitness = fitness
    
    def optimize(self, generations: int, learning_rate: float = 0.1) -> Tuple[List[int], float]:
        """Run the quantum-inspired evolutionary algorithm."""
        for generation in range(generations):
            # Measure all chromosomes
            solutions = []
            fitnesses = []
            for chromosome in self.population:
                solution = chromosome.measure()
                fitness = self.evaluate_solution(solution)
                solutions.append(solution)
                fitnesses.append(fitness)
                self.update_best_solution(solution, fitness)
            
            # Update quantum chromosomes based on best solution
            for chromosome in self.population:
                chromosome.update(self.best_solution, learning_rate)
        
        return self.best_solution, self.best_fitness

def example_optimization():
    """Example usage of QIEA for solving a simple optimization problem."""
    # Define a simple fitness function (maximize the number of 1s)
    def fitness_func(solution: List[int]) -> float:
        return sum(solution)
    
    # Initialize and run the algorithm
    qiea = QIEA(
        chromosome_size=20,
        population_size=10,
        fitness_func=fitness_func,
        maximize=True
    )
    
    best_solution, best_fitness = qiea.optimize(generations=100)
    return best_solution, best_fitness

if __name__ == "__main__":
    solution, fitness = example_optimization()
    print(f"Best solution: {solution}")
    print(f"Best fitness: {fitness}") 