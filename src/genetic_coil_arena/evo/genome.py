"""Genome representation and operations for neural network evolution."""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import random

from deap import creator, base, tools  # type: ignore


@dataclass
class GenomeStats:
    """Statistics for a genome."""
    fitness: float
    episodes_played: int
    avg_score: float
    avg_length: float
    avg_steps: float
    success_rate: float


# Create DEAP fitness and individual classes (check if they already exist)
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


class Genome:
    """Wrapper for neural network genome with additional metadata."""
    
    def __init__(self, weights: Optional[np.ndarray] = None, genome_size: Optional[int] = None):
        """Initialize genome.
        
        Args:
            weights: Pre-existing weight vector
            genome_size: Size of genome if creating random
        """
        if weights is not None:
            self.weights = weights.copy()
        elif genome_size is not None:
            # Initialize with random weights
            self.weights = np.random.normal(0, 0.5, genome_size).astype(np.float32)
        else:
            raise ValueError("Must provide either weights or genome_size")
        
        self.stats: Optional[GenomeStats] = None
        self.generation = 0
        self.id = np.random.randint(0, 1000000)  # Unique identifier
    
    def copy(self) -> "Genome":
        """Create a copy of this genome."""
        new_genome = Genome(weights=self.weights)
        new_genome.stats = self.stats
        new_genome.generation = self.generation
        new_genome.id = self.id
        return new_genome
    
    def mutate(self, sigma: float = 0.2, mutation_prob: float = 0.1) -> "Genome":
        """Create a mutated copy of this genome.
        
        Args:
            sigma: Standard deviation of Gaussian mutation
            mutation_prob: Probability of mutating each gene
            
        Returns:
            Mutated genome
        """
        new_weights = self.weights.copy()
        
        # Apply Gaussian mutation to each weight with given probability
        for i in range(len(new_weights)):
            if random.random() < mutation_prob:
                new_weights[i] += np.random.normal(0, sigma)
        
        mutated = Genome(weights=new_weights)
        mutated.generation = self.generation + 1
        return mutated
    
    def crossover(self, other: "Genome", alpha: float = 0.3) -> Tuple["Genome", "Genome"]:
        """Create two offspring using BLX-α crossover.
        
        Args:
            other: Other parent genome
            alpha: BLX-α parameter controlling exploration
            
        Returns:
            Tuple of two offspring genomes
        """
        min_vals = np.minimum(self.weights, other.weights)
        max_vals = np.maximum(self.weights, other.weights)
        
        # Calculate range and extend by alpha
        ranges = max_vals - min_vals
        extended_min = min_vals - alpha * ranges
        extended_max = max_vals + alpha * ranges
        
        # Create two offspring
        child1_weights = np.random.uniform(extended_min, extended_max).astype(np.float32)
        child2_weights = np.random.uniform(extended_min, extended_max).astype(np.float32)
        
        child1 = Genome(weights=child1_weights)
        child2 = Genome(weights=child2_weights)
        
        child1.generation = max(self.generation, other.generation) + 1
        child2.generation = max(self.generation, other.generation) + 1
        
        return child1, child2
    
    def to_deap_individual(self) -> creator.Individual:  # type: ignore
        """Convert to DEAP individual for use with DEAP toolbox."""
        individual = creator.Individual(self.weights.tolist())  # type: ignore
        if self.stats:
            individual.fitness.values = (self.stats.fitness,)
        return individual
    
    @classmethod
    def from_deap_individual(cls, individual: creator.Individual) -> "Genome":  # type: ignore
        """Create genome from DEAP individual."""
        genome = cls(weights=np.array(individual, dtype=np.float32))
        if individual.fitness.valid:
            # Create minimal stats from fitness
            genome.stats = GenomeStats(
                fitness=individual.fitness.values[0],
                episodes_played=1,
                avg_score=0.0,
                avg_length=0.0,
                avg_steps=0.0,
                success_rate=0.0
            )
        return genome
    
    @property
    def fitness(self) -> float:
        """Get fitness value."""
        return self.stats.fitness if self.stats else 0.0
    
    def __len__(self) -> int:
        """Get genome size."""
        return len(self.weights)
    
    def __str__(self) -> str:
        """String representation."""
        fitness_str = f"fitness={self.fitness:.3f}" if self.stats else "fitness=None"
        return f"Genome(id={self.id}, gen={self.generation}, {fitness_str})"


def create_random_population(population_size: int, genome_size: int, 
                           seed: Optional[int] = None) -> List[Genome]:
    """Create a population of random genomes.
    
    Args:
        population_size: Number of genomes to create
        genome_size: Size of each genome
        seed: Random seed for reproducibility
        
    Returns:
        List of random genomes
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    population = []
    for i in range(population_size):
        genome = Genome(genome_size=genome_size)
        genome.id = i
        population.append(genome)
    
    return population


def tournament_selection(population: List[Genome], k: int = 3, 
                        tournament_size: int = 1) -> List[Genome]:
    """Select parents using tournament selection.
    
    Args:
        population: Population to select from
        k: Tournament size
        tournament_size: Number of individuals to select
        
    Returns:
        Selected parents
    """
    selected = []
    
    for _ in range(tournament_size):
        # Randomly select k individuals for tournament
        tournament = random.sample(population, k)
        
        # Select the best from tournament
        winner = max(tournament, key=lambda x: x.fitness)
        selected.append(winner)
    
    return selected


def create_next_generation(population: List[Genome], elite_size: int,
                         crossover_alpha: float = 0.3, mutation_sigma: float = 0.2,
                         mutation_prob: float = 0.1, tournament_k: int = 3) -> List[Genome]:
    """Create next generation using genetic operators.
    
    Args:
        population: Current population
        elite_size: Number of elite individuals to keep
        crossover_alpha: BLX-α parameter
        mutation_sigma: Mutation standard deviation  
        mutation_prob: Mutation probability per gene
        tournament_k: Tournament size for selection
        
    Returns:
        Next generation population
    """    # Sort population by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)
    
    next_gen = []
    
    # Elitism: keep best individuals (ensure elite_size doesn't exceed population size)
    actual_elite_size = min(elite_size, len(population))
    for i in range(actual_elite_size):
        elite = population[i].copy()
        elite.generation += 1
        next_gen.append(elite)
    
    # Generate rest through crossover and mutation
    while len(next_gen) < len(population):
        # Select parents
        parent1 = tournament_selection(population, tournament_k, 1)[0]
        parent2 = tournament_selection(population, tournament_k, 1)[0]
        
        # Crossover
        child1, child2 = parent1.crossover(parent2, crossover_alpha)
        
        # Mutation
        child1 = child1.mutate(mutation_sigma, mutation_prob)
        child2 = child2.mutate(mutation_sigma, mutation_prob)
        
        next_gen.extend([child1, child2])
    
    # Trim to exact population size
    return next_gen[:len(population)]
