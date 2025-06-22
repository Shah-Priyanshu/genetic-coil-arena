"""Advanced evolution techniques and research-focused features."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import random
from collections import defaultdict

from .genome import Genome, create_random_population
from ..config import Config


@dataclass
class NoveltyMetrics:
    """Metrics for novelty search."""
    behavior_characterization: np.ndarray
    novelty_score: float
    nearest_neighbors: List[float]
    archive_added: bool = False


class NoveltySearch:
    """Novelty search implementation for diversity preservation."""
    
    def __init__(self, k_nearest: int = 15, novelty_threshold: float = 0.1, 
                 archive_size_limit: int = 100):
        """Initialize novelty search.
        
        Args:
            k_nearest: Number of nearest neighbors for novelty calculation
            novelty_threshold: Threshold for adding to archive
            archive_size_limit: Maximum size of novelty archive
        """
        self.k_nearest = k_nearest
        self.novelty_threshold = novelty_threshold
        self.archive_size_limit = archive_size_limit
        self.archive: List[np.ndarray] = []
        
    def characterize_behavior(self, genome: Genome, 
                            evaluation_results: Dict[str, Any]) -> np.ndarray:
        """Create behavior characterization for a genome.
        
        Args:
            genome: The genome to characterize
            evaluation_results: Results from fitness evaluation
            
        Returns:
            Behavior characterization vector
        """
        # Simple behavior characterization based on game performance
        # Can be extended with more sophisticated features
        features = [
            evaluation_results.get('avg_score', 0.0),
            evaluation_results.get('avg_length', 0.0),
            evaluation_results.get('success_rate', 0.0),
            evaluation_results.get('avg_steps', 0.0),
            # Add movement patterns, exploration behavior, etc.
        ]
        
        return np.array(features, dtype=np.float32)
        
    def calculate_novelty(self, behavior: np.ndarray, 
                         population_behaviors: List[np.ndarray]) -> float:
        """Calculate novelty score for a behavior.
        
        Args:
            behavior: Behavior characterization to evaluate
            population_behaviors: Behaviors from current population
            
        Returns:
            Novelty score (higher = more novel)
        """
        all_behaviors = self.archive + population_behaviors
        
        if len(all_behaviors) < self.k_nearest:
            return 1.0  # High novelty if not enough neighbors
            
        # Calculate distances to all other behaviors
        distances = []
        for other_behavior in all_behaviors:
            if not np.array_equal(behavior, other_behavior):
                dist = np.linalg.norm(behavior - other_behavior)
                distances.append(dist)
        
        # Get k nearest neighbors
        distances.sort()
        k_nearest_distances = distances[:min(self.k_nearest, len(distances))]
        
        # Novelty is average distance to k nearest neighbors
        novelty_score = np.mean(k_nearest_distances) if k_nearest_distances else 0.0
        
        return novelty_score
        
    def update_archive(self, behavior: np.ndarray, novelty_score: float) -> bool:
        """Update novelty archive if behavior is novel enough.
        
        Args:
            behavior: Behavior characterization
            novelty_score: Calculated novelty score
            
        Returns:
            True if added to archive
        """
        if novelty_score > self.novelty_threshold:
            self.archive.append(behavior.copy())
            
            # Limit archive size
            if len(self.archive) > self.archive_size_limit:
                # Remove random entry to make room
                random_idx = random.randint(0, len(self.archive) - 1)
                self.archive.pop(random_idx)
                
            return True
        return False


class MultiObjectiveEvolution:
    """Multi-objective optimization for balancing fitness and novelty."""
    
    def __init__(self, fitness_weight: float = 0.7, novelty_weight: float = 0.3):
        """Initialize multi-objective evolution.
        
        Args:
            fitness_weight: Weight for fitness objective
            novelty_weight: Weight for novelty objective
        """
        self.fitness_weight = fitness_weight
        self.novelty_weight = novelty_weight
        
    def calculate_multi_objective_score(self, fitness: float, 
                                      novelty: float) -> float:
        """Calculate combined multi-objective score.
        
        Args:
            fitness: Fitness score (normalized)
            novelty: Novelty score (normalized)
            
        Returns:
            Combined score
        """
        return (self.fitness_weight * fitness + 
                self.novelty_weight * novelty)
        
    def pareto_ranking(self, population: List[Genome], 
                      fitness_scores: List[float],
                      novelty_scores: List[float]) -> List[int]:
        """Assign Pareto ranks to population.
        
        Args:
            population: List of genomes
            fitness_scores: Fitness scores for each genome
            novelty_scores: Novelty scores for each genome
            
        Returns:
            Pareto rank for each genome (lower is better)
        """
        n = len(population)
        ranks = [0] * n
        dominated_by = [[] for _ in range(n)]
        dominates_count = [0] * n
        
        # Compare all pairs
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if i dominates j
                    if (fitness_scores[i] >= fitness_scores[j] and 
                        novelty_scores[i] >= novelty_scores[j] and
                        (fitness_scores[i] > fitness_scores[j] or 
                         novelty_scores[i] > novelty_scores[j])):
                        dominated_by[i].append(j)
                    # Check if j dominates i
                    elif (fitness_scores[j] >= fitness_scores[i] and 
                          novelty_scores[j] >= novelty_scores[i] and
                          (fitness_scores[j] > fitness_scores[i] or 
                           novelty_scores[j] > novelty_scores[i])):
                        dominates_count[i] += 1
        
        # Assign ranks
        current_rank = 0
        remaining = list(range(n))
        
        while remaining:
            current_rank += 1
            current_front = []
            
            for i in remaining:
                if dominates_count[i] == 0:
                    ranks[i] = current_rank
                    current_front.append(i)
            
            # Remove current front and update domination counts
            for i in current_front:
                remaining.remove(i)
                for j in dominated_by[i]:
                    dominates_count[j] -= 1
        
        return ranks


class SpeciationEvolution:
    """Species-based evolution for maintaining diversity."""
    
    def __init__(self, compatibility_threshold: float = 0.5, 
                 target_species: int = 10):
        """Initialize speciation evolution.
        
        Args:
            compatibility_threshold: Threshold for species membership
            target_species: Target number of species
        """
        self.compatibility_threshold = compatibility_threshold
        self.target_species = target_species
        self.species: Dict[int, List[Genome]] = {}
        self.species_representatives: Dict[int, Genome] = {}
        self.next_species_id = 0
        
    def calculate_compatibility(self, genome1: Genome, genome2: Genome) -> float:
        """Calculate compatibility distance between genomes.
        
        Args:
            genome1: First genome
            genome2: Second genome
            
        Returns:
            Compatibility distance (lower = more compatible)
        """
        # Simple compatibility based on parameter differences
        diff = np.abs(genome1.parameters - genome2.parameters)
        return np.mean(diff)
        
    def assign_species(self, population: List[Genome]) -> Dict[int, List[Genome]]:
        """Assign genomes to species.
        
        Args:
            population: List of genomes to assign
            
        Returns:
            Species assignments
        """
        # Clear previous assignments
        self.species.clear()
        
        for genome in population:
            assigned = False
            
            # Try to assign to existing species
            for species_id, representative in self.species_representatives.items():
                compatibility = self.calculate_compatibility(genome, representative)
                
                if compatibility <= self.compatibility_threshold:
                    if species_id not in self.species:
                        self.species[species_id] = []
                    self.species[species_id].append(genome)
                    assigned = True
                    break
            
            # Create new species if not assigned
            if not assigned:
                new_species_id = self.next_species_id
                self.next_species_id += 1
                self.species[new_species_id] = [genome]
                self.species_representatives[new_species_id] = genome
        
        return self.species
        
    def calculate_adjusted_fitness(self, population: List[Genome]) -> List[float]:
        """Calculate fitness adjusted for species sharing.
        
        Args:
            population: List of genomes
            
        Returns:
            Adjusted fitness scores
        """
        species_assignments = self.assign_species(population)
        adjusted_fitness = []
        
        for genome in population:
            # Find which species this genome belongs to
            species_size = 1
            for species_id, members in species_assignments.items():
                if genome in members:
                    species_size = len(members)
                    break
            
            # Adjust fitness by species size (fitness sharing)
            adjusted = genome.fitness / species_size
            adjusted_fitness.append(adjusted)
        
        return adjusted_fitness


class HyperparameterOptimization:
    """Hyperparameter optimization for evolution parameters."""
    
    def __init__(self):
        """Initialize hyperparameter optimizer."""
        self.parameter_ranges = {
            'population_size': (50, 500),
            'mutation_sigma': (0.05, 0.5),
            'mutation_prob': (0.01, 0.3),
            'crossover_alpha': (0.1, 0.5),
            'tournament_size': (2, 8),
            'elite_size': (1, 20),
        }
        self.optimization_history: List[Dict[str, Any]] = []
        
    def suggest_parameters(self, method: str = 'random') -> Dict[str, Any]:
        """Suggest new hyperparameters.
        
        Args:
            method: Optimization method ('random', 'grid', etc.)
            
        Returns:
            Suggested parameter values
        """
        if method == 'random':
            return self._random_search()
        else:
            return self._random_search()  # Default to random
    
    def _random_search(self) -> Dict[str, Any]:
        """Random search for hyperparameters."""
        params = {}
        
        for param, (min_val, max_val) in self.parameter_ranges.items():
            if param in ['population_size', 'tournament_size', 'elite_size']:
                # Integer parameters
                params[param] = random.randint(min_val, max_val)
            else:
                # Float parameters
                params[param] = random.uniform(min_val, max_val)
        
        return params
        
    def record_result(self, parameters: Dict[str, Any], 
                     performance: Dict[str, Any]) -> None:
        """Record optimization result.
        
        Args:
            parameters: Hyperparameters used
            performance: Evolution performance metrics
        """
        result = {
            'parameters': parameters.copy(),
            'performance': performance.copy(),
            'timestamp': np.datetime64('now')
        }
        self.optimization_history.append(result)
        
    def get_best_parameters(self, metric: str = 'best_fitness') -> Dict[str, Any]:
        """Get best parameters based on a metric.
        
        Args:
            metric: Performance metric to optimize
            
        Returns:
            Best parameters found
        """
        if not self.optimization_history:
            return {}
            
        best_result = max(self.optimization_history, 
                         key=lambda x: x['performance'].get(metric, 0))
        return best_result['parameters']


class AdaptiveEvolution:
    """Adaptive evolution that adjusts parameters during evolution."""
    
    def __init__(self, adaptation_interval: int = 10):
        """Initialize adaptive evolution.
        
        Args:
            adaptation_interval: How often to adapt parameters
        """
        self.adaptation_interval = adaptation_interval
        self.performance_history: List[float] = []
        self.parameter_history: List[Dict[str, Any]] = []
        
    def should_adapt(self, generation: int) -> bool:
        """Check if parameters should be adapted.
        
        Args:
            generation: Current generation number
            
        Returns:
            True if adaptation should occur
        """
        return generation % self.adaptation_interval == 0 and generation > 0
        
    def adapt_parameters(self, config: Config, 
                        recent_performance: List[float]) -> Config:
        """Adapt evolution parameters based on recent performance.
        
        Args:
            config: Current configuration
            recent_performance: Recent fitness improvements
            
        Returns:
            Adapted configuration
        """
        # Calculate performance trend
        if len(recent_performance) < 2:
            return config
            
        trend = np.mean(np.diff(recent_performance))
        
        # Adapt mutation parameters based on performance
        if trend < 0.001:  # Low improvement
            # Increase exploration
            config.evolution.mutation_sigma = min(0.5, config.evolution.mutation_sigma * 1.1)
            config.evolution.mutation_prob = min(0.3, config.evolution.mutation_prob * 1.05)
        elif trend > 0.01:  # Good improvement
            # Fine-tune exploitation
            config.evolution.mutation_sigma = max(0.05, config.evolution.mutation_sigma * 0.95)
            config.evolution.mutation_prob = max(0.01, config.evolution.mutation_prob * 0.98)
        
        return config
