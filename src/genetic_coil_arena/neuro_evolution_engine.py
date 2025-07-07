"""
Neuro-Evolution Engine for Genetic Coil Arena
Phase 6A: Complete evolutionary algorithm for neural networks
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import random
from collections import defaultdict

from .neural_evolution import (
    NeuralAgent, NeuralAgentConfig, NeuroEvolutionConfig, 
    Species, create_random_neural_agent, evaluate_neural_population
)

logger = logging.getLogger(__name__)


class NeuroEvolutionEngine:
    """Main engine for neural network evolution."""
    
    def __init__(self, config: NeuroEvolutionConfig, agent_config: NeuralAgentConfig):
        self.config = config
        self.agent_config = agent_config
        self.population = []
        self.species = {}
        self.generation = 0
        self.best_agent = None
        self.best_fitness = float('-inf')
        
        # Evolution statistics
        self.stats = {
            'generation_history': [],
            'fitness_history': [],
            'diversity_history': [],
            'species_history': [],
            'complexity_history': []
        }
        
        # Innovation tracking for NEAT-like features
        self.innovation_counter = 0
        self.connection_innovations = {}
        
        logger.info(f"NeuroEvolution engine initialized with {config.population_size} agents")
    
    def initialize_population(self):
        """Initialize the population with random neural agents."""
        logger.info("Initializing random population...")
        
        self.population = []
        for i in range(self.config.population_size):
            agent = create_random_neural_agent(self.agent_config)
            self.population.append(agent)
        
        # Initialize species if speciation is enabled
        if self.config.speciation:
            self._initialize_species()
        
        logger.info(f"Population initialized with {len(self.population)} agents")
    
    def _initialize_species(self):
        """Initialize species for the population."""
        self.species = {}
        
        for agent in self.population:
            placed = False
            
            # Try to place in existing species
            for species_id, species in self.species.items():
                compatibility = agent.calculate_compatibility(species.representative)
                if compatibility < self.config.compatibility_threshold:
                    species.add_member(agent)
                    placed = True
                    break
            
            # Create new species if needed
            if not placed:
                species_id = f"species_{len(self.species)}"
                new_species = Species(species_id, agent)
                self.species[species_id] = new_species
                agent.species_id = species_id
        
        logger.info(f"Population organized into {len(self.species)} species")
    
    def evolve(self, evaluation_function: Callable, environment_config: Dict[str, Any],
               generations: Optional[int] = None) -> Dict[str, Any]:
        """Run the neuro-evolution process."""
        generations = generations or self.config.generations
        
        logger.info(f"Starting neuro-evolution for {generations} generations")
        start_time = datetime.now()
        
        # Initialize population if empty
        if not self.population:
            self.initialize_population()
        
        for gen in range(generations):
            generation_start = datetime.now()
            
            logger.info(f"Generation {self.generation + 1}/{generations}")
            
            # Evaluate population
            self.population = evaluate_neural_population(
                self.population, evaluation_function, environment_config
            )
            
            # Update statistics
            self._update_statistics()
            
            # Selection and reproduction
            if gen < generations - 1:  # Don't reproduce on last generation
                self._selection_and_reproduction()
            
            # Update species if enabled
            if self.config.speciation:
                self._update_species()
            
            self.generation += 1
            
            generation_time = datetime.now() - generation_start
            logger.info(f"Generation {self.generation} completed in {generation_time.total_seconds():.2f}s")
            
            # Early stopping if perfect fitness achieved
            if self.best_fitness >= 1.0:
                logger.info(f"Perfect fitness achieved in generation {self.generation}")
                break
        
        total_time = datetime.now() - start_time
        
        # Final results
        results = {
            'best_agent': self.best_agent.to_dict() if self.best_agent else None,
            'best_fitness': self.best_fitness,
            'final_generation': self.generation,
            'total_time_seconds': total_time.total_seconds(),
            'final_population_size': len(self.population),
            'final_species_count': len(self.species) if self.config.speciation else 0,
            'stats': self.stats,
            'config': {
                'population_size': self.config.population_size,
                'generations': generations,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate,
                'elite_size': self.config.elite_size,
                'speciation': self.config.speciation
            }
        }
        
        logger.info(f"Neuro-evolution completed in {total_time.total_seconds():.2f}s")
        logger.info(f"Best fitness achieved: {self.best_fitness:.6f}")
        
        return results
    
    def _update_statistics(self):
        """Update evolution statistics."""
        fitnesses = [agent.fitness for agent in self.population]
        complexities = [agent.complexity_score for agent in self.population]
        
        # Update best agent
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_agent = self.population[best_idx]
        
        # Calculate diversity (average pairwise distance)
        diversity = self._calculate_diversity()
        
        # Update statistics
        generation_stats = {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'average_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'min_fitness': min(fitnesses),
            'diversity': diversity,
            'species_count': len(self.species) if self.config.speciation else 1,
            'average_complexity': np.mean(complexities),
            'timestamp': datetime.now().isoformat()
        }
        
        self.stats['generation_history'].append(generation_stats)
        self.stats['fitness_history'].append(fitnesses)
        self.stats['diversity_history'].append(diversity)
        self.stats['species_history'].append(len(self.species) if self.config.speciation else 1)
        self.stats['complexity_history'].append(complexities)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity based on weight differences."""
        if len(self.population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        # Sample pairs to avoid O(n²) computation for large populations
        sample_size = min(50, len(self.population))
        sampled_agents = random.sample(self.population, sample_size)
        
        for i in range(len(sampled_agents)):
            for j in range(i + 1, len(sampled_agents)):
                try:
                    distance = sampled_agents[i].calculate_compatibility(sampled_agents[j])
                    if distance != float('inf'):
                        total_distance += distance
                        comparisons += 1
                except:
                    continue
        
        return total_distance / max(comparisons, 1)
    
    def _selection_and_reproduction(self):
        """Perform selection and create next generation."""
        if self.config.speciation:
            self._speciated_reproduction()
        else:
            self._standard_reproduction()
    
    def _standard_reproduction(self):
        """Standard reproduction without speciation."""
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Elite selection
        elite_size = min(self.config.elite_size, len(self.population))
        new_population = self.population[:elite_size].copy()
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            
            if random.random() < self.config.crossover_rate:
                # Crossover
                parent2 = self._tournament_selection()
                child1, child2 = parent1.crossover(parent2, 0.5)
                offspring = [child1, child2]
            else:
                # Asexual reproduction (mutation only)
                offspring = [NeuralAgent(parent1.config)]
                offspring[0].network.load_state_dict(parent1.network.state_dict())
                offspring[0].parent_ids = [parent1.agent_id]
            
            # Mutation
            for child in offspring:
                if random.random() < self.config.mutation_rate:
                    child.mutate(self.config.mutation_rate, self.config.mutation_strength)
                
                new_population.append(child)
                
                if len(new_population) >= self.config.population_size:
                    break
        
        # Trim to exact population size
        self.population = new_population[:self.config.population_size]
    
    def _speciated_reproduction(self):
        """Reproduction with speciation."""
        # Calculate species fitness sharing
        for species in self.species.values():
            if species.members:
                # Fitness sharing: divide fitness by species size
                shared_fitness = []
                for member in species.members:
                    if self.config.fitness_sharing:
                        shared = member.fitness / len(species.members)
                    else:
                        shared = member.fitness
                    shared_fitness.append(shared)
                
                species.average_fitness = np.mean(shared_fitness)
        
        # Calculate offspring allocation
        total_shared_fitness = sum(s.average_fitness for s in self.species.values() if s.members)
        new_population = []
        
        for species in self.species.values():
            if not species.members:
                continue
            
            # Allocate offspring based on species fitness
            if total_shared_fitness > 0:
                species_allocation = int(
                    (species.average_fitness / total_shared_fitness) * self.config.population_size
                )
            else:
                species_allocation = len(species.members)
            
            species_allocation = max(1, species_allocation)  # At least 1 offspring per species
            
            # Sort species members by fitness
            species.members.sort(key=lambda x: x.fitness, reverse=True)
            
            # Elite from species
            elite_count = max(1, species_allocation // 4)
            species_offspring = species.members[:elite_count].copy()
            
            # Generate remaining offspring
            while len(species_offspring) < species_allocation:
                if len(species.members) >= 2 and random.random() < self.config.crossover_rate:
                    # Crossover within species
                    parent1 = self._tournament_selection(species.members)
                    parent2 = self._tournament_selection(species.members)
                    child1, child2 = parent1.crossover(parent2, 0.5)
                    offspring = [child1, child2]
                else:
                    # Mutation only
                    parent = self._tournament_selection(species.members)
                    child = NeuralAgent(parent.config)
                    child.network.load_state_dict(parent.network.state_dict())
                    child.parent_ids = [parent.agent_id]
                    offspring = [child]
                
                # Apply mutations
                for child in offspring:
                    if random.random() < self.config.mutation_rate:
                        child.mutate(self.config.mutation_rate, self.config.mutation_strength)
                    
                    species_offspring.append(child)
                    
                    if len(species_offspring) >= species_allocation:
                        break
            
            new_population.extend(species_offspring[:species_allocation])
        
        # Ensure we have enough individuals
        while len(new_population) < self.config.population_size:
            # Add random agents if needed
            random_agent = create_random_neural_agent(self.agent_config)
            new_population.append(random_agent)
        
        self.population = new_population[:self.config.population_size]
    
    def _tournament_selection(self, population: Optional[List[NeuralAgent]] = None) -> NeuralAgent:
        """Tournament selection."""
        if population is None:
            population = self.population
        
        tournament_size = min(self.config.tournament_size, len(population))
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _update_species(self):
        """Update species membership and remove stagnant species."""
        # Reset species memberships
        for species in self.species.values():
            species.members = []
        
        # Reassign agents to species
        for agent in self.population:
            agent.species_id = None
            placed = False
            
            # Try existing species
            for species_id, species in self.species.items():
                compatibility = agent.calculate_compatibility(species.representative)
                if compatibility < self.config.compatibility_threshold:
                    species.add_member(agent)
                    placed = True
                    break
            
            # Create new species if needed
            if not placed:
                species_id = f"species_{len(self.species)}_{self.generation}"
                new_species = Species(species_id, agent)
                self.species[species_id] = new_species
        
        # Remove empty or stagnant species
        species_to_remove = []
        for species_id, species in self.species.items():
            if not species.members or species.should_be_removed():
                species_to_remove.append(species_id)
        
        for species_id in species_to_remove:
            del self.species[species_id]
        
        # Update species representatives
        for species in self.species.values():
            if species.members:
                # Choose best member as new representative
                best_member = max(species.members, key=lambda x: x.fitness)
                species.representative = best_member
    
    def save_checkpoint(self, checkpoint_dir: Path):
        """Save evolution checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save engine state
        engine_state = {
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'config': {
                'population_size': self.config.population_size,
                'generations': self.config.generations,
                'mutation_rate': self.config.mutation_rate,
                'mutation_strength': self.config.mutation_strength,
                'crossover_rate': self.config.crossover_rate,
                'elite_size': self.config.elite_size,
                'tournament_size': self.config.tournament_size,
                'fitness_sharing': self.config.fitness_sharing,
                'speciation': self.config.speciation,
                'compatibility_threshold': self.config.compatibility_threshold
            },
            'agent_config': {
                'input_size': self.agent_config.input_size,
                'hidden_layers': self.agent_config.hidden_layers,
                'output_size': self.agent_config.output_size,
                'activation': self.agent_config.activation,
                'dropout_rate': self.agent_config.dropout_rate,
                'use_batch_norm': self.agent_config.use_batch_norm,
                'network_type': self.agent_config.network_type
            },
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_dir / 'engine_state.json', 'w') as f:
            json.dump(engine_state, f, indent=2)
        
        # Save population
        population_dir = checkpoint_dir / 'population'
        population_dir.mkdir(exist_ok=True)
        
        for i, agent in enumerate(self.population):
            agent.save(population_dir / f'agent_{i}')
        
        # Save best agent separately
        if self.best_agent:
            self.best_agent.save(checkpoint_dir / 'best_agent')
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: Path):
        """Load evolution checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load engine state
        with open(checkpoint_dir / 'engine_state.json', 'r') as f:
            engine_state = json.load(f)
        
        self.generation = engine_state['generation']
        self.best_fitness = engine_state['best_fitness']
        self.stats = engine_state['stats']
        
        # Update configs
        for key, value in engine_state['config'].items():
            setattr(self.config, key, value)
        
        for key, value in engine_state['agent_config'].items():
            setattr(self.agent_config, key, value)
        
        # Load population
        population_dir = checkpoint_dir / 'population'
        self.population = []
        
        agent_files = sorted(population_dir.glob('agent_*.json'))
        for agent_file in agent_files:
            agent = NeuralAgent.load(agent_file.with_suffix(''))
            self.population.append(agent)
        
        # Load best agent
        best_agent_path = checkpoint_dir / 'best_agent.json'
        if best_agent_path.exists():
            self.best_agent = NeuralAgent.load(checkpoint_dir / 'best_agent')
        
        # Reinitialize species if speciation is enabled
        if self.config.speciation:
            self._initialize_species()
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")
        logger.info(f"Resumed at generation {self.generation} with {len(self.population)} agents")
    
    def get_population_summary(self) -> Dict[str, Any]:
        """Get summary of current population."""
        if not self.population:
            return {'message': 'Population not initialized'}
        
        fitnesses = [agent.fitness for agent in self.population]
        complexities = [agent.complexity_score for agent in self.population]
        ages = [agent.age for agent in self.population]
        
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'fitness_stats': {
                'best': max(fitnesses) if fitnesses else 0,
                'average': np.mean(fitnesses) if fitnesses else 0,
                'std': np.std(fitnesses) if fitnesses else 0,
                'min': min(fitnesses) if fitnesses else 0
            },
            'complexity_stats': {
                'average': np.mean(complexities) if complexities else 0,
                'std': np.std(complexities) if complexities else 0,
                'min': min(complexities) if complexities else 0,
                'max': max(complexities) if complexities else 0
            },
            'age_stats': {
                'average': np.mean(ages) if ages else 0,
                'max': max(ages) if ages else 0
            },
            'species_count': len(self.species) if self.config.speciation else 1,
            'best_agent_id': self.best_agent.agent_id if self.best_agent else None
        }
