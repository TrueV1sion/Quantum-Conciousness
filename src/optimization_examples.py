"""
Example optimization problems using Quantum-Inspired Optimization Algorithms.
"""

import numpy as np
from typing import List, Tuple, Dict
from quantum_inspired_optimization import QIEA, MultiObjectiveQIEA, Solution, SelectionMethod
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
from collections import defaultdict

def knapsack_problem(
    items: List[Tuple[float, float]],  # List of (value, weight) tuples
    max_weight: float
) -> Tuple[List[int], float]:
    """
    Solve the 0/1 knapsack problem using QIEA.
    
    Args:
        items: List of (value, weight) tuples for each item
        max_weight: Maximum weight capacity of the knapsack
    
    Returns:
        Tuple of (selected items as binary list, total value)
    """
    def fitness_func(solution: List[int]) -> float:
        total_value = sum(items[i][0] * bit for i, bit in enumerate(solution))
        total_weight = sum(items[i][1] * bit for i, bit in enumerate(solution))
        
        # Penalize solutions that exceed weight constraint
        if total_weight > max_weight:
            return -total_weight  # Strong penalty for invalid solutions
        return total_value

    qiea = QIEA(
        chromosome_size=len(items),
        population_size=50,
        fitness_func=fitness_func,
        maximize=True
    )
    
    best_solution, best_fitness = qiea.optimize(generations=200)
    return best_solution, best_fitness

def traveling_salesman(
    distances: np.ndarray,  # Distance matrix between cities
    num_cities: int
) -> Tuple[List[int], float]:
    """
    Solve the Traveling Salesman Problem using QIEA.
    The solution is encoded as a binary string that can be decoded into a valid tour.
    
    Args:
        distances: Square matrix of distances between cities
        num_cities: Number of cities in the problem
    
    Returns:
        Tuple of (binary encoded solution, tour length)
    """
    # We need log2(num_cities) bits to represent each city index
    bits_per_city = int(np.ceil(np.log2(num_cities)))
    chromosome_size = num_cities * bits_per_city
    
    def decode_tour(binary: List[int]) -> List[int]:
        """Convert binary representation to city indices."""
        tour = []
        for i in range(0, len(binary), bits_per_city):
            # Convert binary chunk to city index
            city_bits = binary[i:i + bits_per_city]
            city_idx = sum(bit * (2 ** j) for j, bit in enumerate(reversed(city_bits)))
            city_idx = city_idx % num_cities  # Ensure valid city index
            tour.append(city_idx)
        return tour
    
    def fitness_func(solution: List[int]) -> float:
        tour = decode_tour(solution)
        
        # Check if all cities are visited
        if len(set(tour)) != num_cities:
            return float('-inf')
        
        # Calculate total distance
        total_distance = sum(
            distances[tour[i]][tour[(i + 1) % num_cities]]
            for i in range(num_cities)
        )
        return -total_distance  # Negative because we want to minimize distance
    
    qiea = QIEA(
        chromosome_size=chromosome_size,
        population_size=100,
        fitness_func=fitness_func,
        maximize=True
    )
    
    best_solution, best_fitness = qiea.optimize(generations=500, learning_rate=0.05)
    return best_solution, -best_fitness  # Return positive distance

def feature_selection_optimization(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int
) -> Tuple[List[int], float]:
    """
    Use QIEA to perform feature selection for machine learning.
    
    Args:
        X: Feature matrix
        y: Target labels
        n_features: Total number of features
    
    Returns:
        Tuple of (selected features as binary list, cross-validation score)
    """
    def fitness_func(solution: List[int]) -> float:
        if sum(solution) == 0:  # No features selected
            return float('-inf')
        
        # Select features based on binary solution
        selected_features = [i for i, bit in enumerate(solution) if bit == 1]
        X_selected = X[:, selected_features]
        
        # Train and evaluate a random forest classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(clf, X_selected, y, cv=5, scoring='accuracy')
        
        # Return mean cross-validation score
        return np.mean(scores)

    qiea = QIEA(
        chromosome_size=n_features,
        population_size=30,
        fitness_func=fitness_func,
        maximize=True
    )
    
    best_solution, best_fitness = qiea.optimize(generations=100, learning_rate=0.05)
    return best_solution, best_fitness

def portfolio_optimization(
    returns: np.ndarray,  # Historical returns for assets
    risks: np.ndarray,    # Risk measures for assets
    max_assets: int       # Maximum number of assets to select
) -> List[Solution]:
    """
    Multi-objective portfolio optimization using QIEA.
    Objectives:
    1. Maximize expected return
    2. Minimize risk
    3. Minimize number of assets (for management simplicity)
    
    Args:
        returns: Mean returns for each asset
        risks: Risk measures for each asset
        max_assets: Maximum number of assets to include
    
    Returns:
        List of Pareto-optimal solutions
    """
    def return_objective(solution: List[int]) -> float:
        """Maximize expected return."""
        return sum(r * bit for r, bit in zip(returns, solution))
    
    def risk_objective(solution: List[int]) -> float:
        """Minimize portfolio risk."""
        if sum(solution) == 0:
            return float('inf')
        selected_risks = [r for r, bit in zip(risks, solution) if bit == 1]
        # Simple risk measure: average of individual risks
        return -sum(selected_risks) / len(selected_risks)
    
    def simplicity_objective(solution: List[int]) -> float:
        """Minimize number of assets (negative because we maximize by default)."""
        num_selected = sum(solution)
        if num_selected > max_assets:
            return float('-inf')  # Penalize solutions with too many assets
        return -num_selected

    qiea = MultiObjectiveQIEA(
        chromosome_size=len(returns),
        population_size=50,
        objective_funcs=[return_objective, risk_objective, simplicity_objective],
        maximize=[True, True, True]  # Note: risk and simplicity are negated in their functions
    )
    
    return qiea.optimize(generations=200, learning_rate=0.05)

def network_design_optimization(
    num_nodes: int,
    distances: np.ndarray,
    bandwidth_demands: np.ndarray,
    max_links: int
) -> List[Solution]:
    """
    Multi-objective network design optimization.
    Objectives:
    1. Minimize total network cost (distance-based)
    2. Maximize network reliability (connectivity)
    3. Minimize maximum link load
    
    Args:
        num_nodes: Number of nodes in the network
        distances: Matrix of distances between nodes
        bandwidth_demands: Matrix of bandwidth demands between nodes
        max_links: Maximum number of links allowed
    
    Returns:
        List of Pareto-optimal solutions
    """
    def encode_network(solution: List[int]) -> np.ndarray:
        """Convert binary solution to adjacency matrix."""
        network = np.zeros((num_nodes, num_nodes))
        idx = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if solution[idx] == 1:
                    network[i, j] = network[j, i] = 1
                idx += 1
        return network
    
    def cost_objective(solution: List[int]) -> float:
        """Minimize total network cost."""
        network = encode_network(solution)
        total_cost = sum(
            distances[i, j] * network[i, j]
            for i in range(num_nodes)
            for j in range(i + 1, num_nodes)
        )
        return -total_cost  # Negative because we want to minimize
    
    def reliability_objective(solution: List[int]) -> float:
        """Maximize network reliability (based on connectivity)."""
        network = encode_network(solution)
        # Use number of distinct paths as a simple reliability measure
        num_paths = 0
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.any(network):  # If there's any connection
                    num_paths += 1
        return num_paths
    
    def load_balance_objective(solution: List[int]) -> float:
        """Minimize maximum link load."""
        network = encode_network(solution)
        if not np.any(network):
            return float('-inf')
        
        # Simple load calculation (could be more sophisticated)
        link_loads = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if bandwidth_demands[i, j] > 0:
                    # Use direct link if available
                    if network[i, j] == 1:
                        link_loads[i, j] += bandwidth_demands[i, j]
        
        max_load = np.max(link_loads) if np.any(link_loads) else float('inf')
        return -max_load  # Negative because we want to minimize

    # Calculate chromosome size (one bit for each possible link)
    chromosome_size = (num_nodes * (num_nodes - 1)) // 2
    
    qiea = MultiObjectiveQIEA(
        chromosome_size=chromosome_size,
        population_size=50,
        objective_funcs=[cost_objective, reliability_objective, load_balance_objective],
        maximize=[True, True, True],  # Note: cost and load are negated in their functions
        selection_method=SelectionMethod.CROWDING_DISTANCE
    )
    
    return qiea.optimize(generations=200, learning_rate=0.05)

def manufacturing_optimization(
    num_machines: int,
    num_products: int,
    setup_times: np.ndarray,
    processing_times: np.ndarray,
    energy_consumption: np.ndarray,
    max_parallel: int
) -> List[Solution]:
    """
    Multi-objective manufacturing process optimization.
    Objectives:
    1. Minimize total production time
    2. Minimize energy consumption
    3. Maximize resource utilization
    
    Args:
        num_machines: Number of available machines
        num_products: Number of products to manufacture
        setup_times: Matrix of setup times for each product on each machine
        processing_times: Matrix of processing times for each product on each machine
        energy_consumption: Matrix of energy consumption rates for each product on each machine
        max_parallel: Maximum number of parallel operations allowed
    """
    def encode_schedule(solution: List[int]) -> np.ndarray:
        """Convert binary solution to schedule matrix."""
        schedule = np.zeros((num_machines, num_products))
        idx = 0
        for i in range(num_machines):
            for j in range(num_products):
                schedule[i, j] = solution[idx]
                idx += 1
        return schedule
    
    def time_objective(solution: List[int]) -> float:
        """Minimize total production time."""
        schedule = encode_schedule(solution)
        if not np.any(schedule):
            return float('-inf')
        
        total_time = 0
        for machine in range(num_machines):
            machine_time = 0
            last_product = None
            for product in range(num_products):
                if schedule[machine, product] == 1:
                    if last_product is not None:
                        machine_time += setup_times[machine, product]
                    machine_time += processing_times[machine, product]
                    last_product = product
            total_time = max(total_time, machine_time)
        
        return -total_time  # Negative because we want to minimize
    
    def energy_objective(solution: List[int]) -> float:
        """Minimize energy consumption."""
        schedule = encode_schedule(solution)
        total_energy = sum(
            schedule[i, j] * energy_consumption[i, j]
            for i in range(num_machines)
            for j in range(num_products)
        )
        return -total_energy  # Negative because we want to minimize
    
    def utilization_objective(solution: List[int]) -> float:
        """Maximize resource utilization."""
        schedule = encode_schedule(solution)
        if not np.any(schedule):
            return float('-inf')
        
        total_possible_time = sum(
            max(processing_times[machine]) for machine in range(num_machines)
        )
        actual_time = sum(
            schedule[i, j] * processing_times[i, j]
            for i in range(num_machines)
            for j in range(num_products)
        )
        
        return actual_time / total_possible_time

    chromosome_size = num_machines * num_products
    
    qiea = MultiObjectiveQIEA(
        chromosome_size=chromosome_size,
        population_size=50,
        objective_funcs=[time_objective, energy_objective, utilization_objective],
        maximize=[True, True, True],  # Note: time and energy are negated in their functions
        selection_method=SelectionMethod.TOURNAMENT
    )
    
    return qiea.optimize(generations=300, learning_rate=0.05)

def semantic_space_optimization(
    concept_embeddings: Dict[str, np.ndarray],
    concept_hierarchy: Dict[str, List[str]],
    target_concepts: List[str],
    abstraction_levels: int
) -> List[Solution]:
    """
    Multi-objective optimization for exploring semantic space relationships and abstraction hierarchies.
    
    Objectives:
    1. Maximize semantic coherence within abstraction levels
    2. Maximize hierarchical consistency across levels
    3. Optimize conceptual coverage of target space
    
    Args:
        concept_embeddings: Dictionary mapping concepts to their vector embeddings
        concept_hierarchy: Dictionary mapping higher-level concepts to their subconcepts
        target_concepts: List of concepts we want to explore/connect
        abstraction_levels: Number of abstraction levels to consider
    
    Returns:
        List of Pareto-optimal solutions representing different semantic space organizations
    """
    # Convert embeddings to numpy array for efficient computation
    concepts = list(concept_embeddings.keys())
    embedding_dim = len(next(iter(concept_embeddings.values())))
    embeddings = np.array([concept_embeddings[c] for c in concepts])
    
    def encode_semantic_structure(solution: List[int]) -> List[List[int]]:
        """Convert binary solution to semantic structure matrix."""
        # Split solution into abstraction levels
        bits_per_level = len(solution) // abstraction_levels
        levels = []
        for i in range(abstraction_levels):
            start = i * bits_per_level
            end = start + bits_per_level
            level_bits = solution[start:end]
            levels.append([
                j for j, bit in enumerate(level_bits) if bit == 1
            ])
        return levels
    
    def semantic_coherence_objective(solution: List[int]) -> float:
        """Maximize semantic coherence within each abstraction level."""
        levels = encode_semantic_structure(solution)
        if not any(levels):  # If no concepts selected
            return float('-inf')
        
        total_coherence = 0
        for level in levels:
            if len(level) < 2:
                continue
            
            # Calculate average cosine similarity between concepts in this level
            level_embeddings = embeddings[level]
            similarities = []
            for i in range(len(level)):
                for j in range(i + 1, len(level)):
                    sim = 1 - cosine(level_embeddings[i], level_embeddings[j])
                    similarities.append(sim)
            
            if similarities:
                total_coherence += np.mean(similarities)
        
        return total_coherence
    
    def hierarchical_consistency_objective(solution: List[int]) -> float:
        """Maximize consistency between abstraction levels."""
        levels = encode_semantic_structure(solution)
        if len(levels) < 2:
            return float('-inf')
        
        consistency = 0
        for i in range(len(levels) - 1):
            higher_level = levels[i]
            lower_level = levels[i + 1]
            
            if not higher_level or not lower_level:
                continue
            
            # Check if higher-level concepts subsume lower-level ones
            for high_idx in higher_level:
                high_concept = concepts[high_idx]
                if high_concept in concept_hierarchy:
                    subconcepts = concept_hierarchy[high_concept]
                    coverage = sum(
                        1 for low_idx in lower_level
                        if concepts[low_idx] in subconcepts
                    )
                    consistency += coverage / len(subconcepts)
        
        return consistency
    
    def conceptual_coverage_objective(solution: List[int]) -> float:
        """Optimize coverage of target conceptual space."""
        levels = encode_semantic_structure(solution)
        if not any(levels):
            return float('-inf')
        
        # Calculate coverage of target concepts
        selected_concepts = {
            concepts[idx]
            for level in levels
            for idx in level
        }
        
        coverage = 0
        for target in target_concepts:
            # Find closest selected concept
            if target in selected_concepts:
                coverage += 1
            else:
                target_embedding = concept_embeddings[target]
                similarities = [
                    1 - cosine(target_embedding, concept_embeddings[c])
                    for c in selected_concepts
                ]
                coverage += max(similarities) if similarities else 0
        
        return coverage / len(target_concepts)

    # Calculate chromosome size (one bit per concept per abstraction level)
    chromosome_size = len(concepts) * abstraction_levels
    
    qiea = MultiObjectiveQIEA(
        chromosome_size=chromosome_size,
        population_size=50,
        objective_funcs=[
            semantic_coherence_objective,
            hierarchical_consistency_objective,
            conceptual_coverage_objective
        ],
        maximize=[True, True, True],
        selection_method=SelectionMethod.CROWDING_DISTANCE
    )
    
    return qiea.optimize(generations=300, learning_rate=0.05)

def example_knapsack():
    """Example usage of QIEA for solving a knapsack problem."""
    # Example items: (value, weight)
    items = [
        (4, 12),
        (2, 1),
        (6, 4),
        (1, 1),
        (2, 2),
        (3, 3),
    ]
    max_weight = 15
    
    solution, value = knapsack_problem(items, max_weight)
    print(f"Knapsack Problem Solution:")
    print(f"Selected items (binary): {solution}")
    print(f"Total value: {value}")
    print(f"Total weight: {sum(items[i][1] * bit for i, bit in enumerate(solution))}")

def example_tsp():
    """Example usage of QIEA for solving a TSP problem."""
    # Example distance matrix for 5 cities
    distances = np.array([
        [0, 2, 9, 10, 6],
        [2, 0, 4, 8, 5],
        [9, 4, 0, 3, 7],
        [10, 8, 3, 0, 4],
        [6, 5, 7, 4, 0]
    ])
    
    solution, tour_length = traveling_salesman(distances, 5)
    print(f"\nTraveling Salesman Problem Solution:")
    print(f"Binary encoding: {solution}")
    print(f"Tour length: {tour_length}")

def example_feature_selection():
    """Example usage of QIEA for feature selection."""
    # Generate synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,  # Only 10 features are actually informative
        n_redundant=5,
        n_repeated=5,
        random_state=42
    )
    
    solution, score = feature_selection_optimization(X, y, n_features=20)
    
    print("\nFeature Selection Results:")
    print(f"Selected features (binary): {solution}")
    print(f"Number of selected features: {sum(solution)}")
    print(f"Cross-validation accuracy: {score:.4f}")
    print("Selected feature indices:", [i for i, bit in enumerate(solution) if bit == 1])

def example_portfolio():
    """Example usage of multi-objective QIEA for portfolio optimization."""
    # Generate example data for 10 assets
    np.random.seed(42)
    num_assets = 10
    returns = np.random.normal(0.10, 0.02, num_assets)  # Mean return 10%
    risks = np.random.normal(0.15, 0.03, num_assets)    # Mean risk 15%
    
    solutions = portfolio_optimization(returns, risks, max_assets=5)
    
    print("\nPortfolio Optimization Results:")
    print(f"Found {len(solutions)} Pareto-optimal solutions")
    
    # Display results
    for i, sol in enumerate(solutions):
        selected = [j for j, bit in enumerate(sol.binary) if bit == 1]
        exp_return = sol.objectives[0]
        risk = -sol.objectives[1]  # Unnegate
        num_assets = -sol.objectives[2]  # Unnegate
        print(f"\nSolution {i + 1}:")
        print(f"Selected assets: {selected}")
        print(f"Expected return: {exp_return:.4f}")
        print(f"Portfolio risk: {risk:.4f}")
        print(f"Number of assets: {num_assets}")
    
    # Visualize the Pareto front (first two objectives)
    plt.figure(figsize=(10, 6))
    returns = [s.objectives[0] for s in solutions]
    risks = [-s.objectives[1] for s in solutions]  # Unnegate
    plt.scatter(risks, returns, c='b', marker='o')
    plt.xlabel('Portfolio Risk')
    plt.ylabel('Expected Return')
    plt.title('Pareto Front of Portfolio Optimization')
    plt.grid(True)
    plt.show()

def example_network_design():
    """Example usage of multi-objective QIEA for network design."""
    num_nodes = 5
    # Generate random distances and bandwidth demands
    np.random.seed(42)
    distances = np.random.uniform(1, 10, (num_nodes, num_nodes))
    distances = (distances + distances.T) / 2  # Make symmetric
    np.fill_diagonal(distances, 0)
    
    bandwidth_demands = np.random.uniform(0, 5, (num_nodes, num_nodes))
    bandwidth_demands = (bandwidth_demands + bandwidth_demands.T) / 2
    np.fill_diagonal(bandwidth_demands, 0)
    
    solutions = network_design_optimization(
        num_nodes=num_nodes,
        distances=distances,
        bandwidth_demands=bandwidth_demands,
        max_links=8
    )
    
    print("\nNetwork Design Optimization Results:")
    print(f"Found {len(solutions)} Pareto-optimal solutions")
    
    # Display results
    for i, sol in enumerate(solutions):
        print(f"\nSolution {i + 1}:")
        print(f"Network cost: {-sol.objectives[0]:.2f}")
        print(f"Reliability (paths): {sol.objectives[1]:.0f}")
        print(f"Max link load: {-sol.objectives[2]:.2f}")

def example_manufacturing():
    """Example usage of multi-objective QIEA for manufacturing optimization."""
    num_machines = 3
    num_products = 4
    
    # Generate random problem data
    np.random.seed(42)
    setup_times = np.random.uniform(1, 5, (num_machines, num_products))
    processing_times = np.random.uniform(5, 15, (num_machines, num_products))
    energy_consumption = np.random.uniform(2, 8, (num_machines, num_products))
    
    solutions = manufacturing_optimization(
        num_machines=num_machines,
        num_products=num_products,
        setup_times=setup_times,
        processing_times=processing_times,
        energy_consumption=energy_consumption,
        max_parallel=2
    )
    
    print("\nManufacturing Optimization Results:")
    print(f"Found {len(solutions)} Pareto-optimal solutions")
    
    # Display results
    for i, sol in enumerate(solutions):
        print(f"\nSolution {i + 1}:")
        print(f"Total time: {-sol.objectives[0]:.2f}")
        print(f"Energy consumption: {-sol.objectives[1]:.2f}")
        print(f"Resource utilization: {sol.objectives[2]:.2%}")
        
        # Display schedule
        schedule = np.array(sol.binary).reshape(num_machines, num_products)
        print("Schedule (Machines × Products):")
        print(schedule)

def example_semantic_space():
    """Example usage of semantic space optimization."""
    # Example concept embeddings (simplified 3D vectors for illustration)
    concept_embeddings = {
        # Abstract concepts
        "cognition": np.array([0.8, 0.7, 0.6]),
        "computation": np.array([0.7, 0.8, 0.5]),
        "intelligence": np.array([0.9, 0.8, 0.7]),
        
        # Mid-level concepts
        "learning": np.array([0.6, 0.7, 0.4]),
        "reasoning": np.array([0.7, 0.6, 0.5]),
        "optimization": np.array([0.5, 0.8, 0.4]),
        "processing": np.array([0.6, 0.8, 0.3]),
        
        # Specific concepts
        "neural_networks": np.array([0.4, 0.7, 0.3]),
        "quantum_computing": np.array([0.5, 0.9, 0.2]),
        "genetic_algorithms": np.array([0.3, 0.8, 0.4]),
        "symbolic_logic": np.array([0.6, 0.5, 0.4])
    }
    
    # Example concept hierarchy
    concept_hierarchy = {
        "cognition": ["learning", "reasoning"],
        "computation": ["processing", "optimization"],
        "intelligence": ["learning", "reasoning", "optimization"],
        "learning": ["neural_networks", "genetic_algorithms"],
        "processing": ["quantum_computing"],
        "reasoning": ["symbolic_logic"]
    }
    
    # Target concepts we want to explore
    target_concepts = [
        "neural_networks",
        "quantum_computing",
        "symbolic_logic",
        "genetic_algorithms"
    ]
    
    solutions = semantic_space_optimization(
        concept_embeddings=concept_embeddings,
        concept_hierarchy=concept_hierarchy,
        target_concepts=target_concepts,
        abstraction_levels=3
    )
    
    print("\nSemantic Space Optimization Results:")
    print(f"Found {len(solutions)} Pareto-optimal organizations")
    
    # Display results
    concepts = list(concept_embeddings.keys())
    for i, sol in enumerate(solutions):
        print(f"\nSolution {i + 1}:")
        levels = []
        bits_per_level = len(sol.binary) // 3
        
        for level in range(3):
            start = level * bits_per_level
            end = start + bits_per_level
            level_concepts = [
                concepts[j] 
                for j, bit in enumerate(sol.binary[start:end])
                if bit == 1
            ]
            levels.append(level_concepts)
            print(f"Level {level + 1}: {level_concepts}")
        
        print(f"Semantic Coherence: {sol.objectives[0]:.3f}")
        print(f"Hierarchical Consistency: {sol.objectives[1]:.3f}")
        print(f"Conceptual Coverage: {sol.objectives[2]:.3f}")
    
    # Visualize semantic space organization
    plt.figure(figsize=(12, 8))
    embeddings = np.array(list(concept_embeddings.values()))
    
    # Use t-SNE to visualize high-dimensional relationships
    tsne = TSNE(n_components=2, random_state=42)
    embedded = tsne.fit_transform(embeddings)
    
    # Plot best solution's organization
    best_solution = max(solutions, key=lambda s: sum(s.objectives))
    levels = []
    bits_per_level = len(best_solution.binary) // 3
    colors = ['r', 'g', 'b']
    
    for level in range(3):
        start = level * bits_per_level
        end = start + bits_per_level
        level_indices = [
            j for j, bit in enumerate(best_solution.binary[start:end])
            if bit == 1
        ]
        plt.scatter(
            embedded[level_indices, 0],
            embedded[level_indices, 1],
            c=colors[level],
            label=f'Level {level + 1}'
        )
    
    for i, concept in enumerate(concepts):
        plt.annotate(concept, (embedded[i, 0], embedded[i, 1]))
    
    plt.title('Semantic Space Organization')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    example_knapsack()
    example_tsp()
    example_feature_selection()
    example_portfolio()
    example_network_design()
    example_manufacturing()
    example_semantic_space() 