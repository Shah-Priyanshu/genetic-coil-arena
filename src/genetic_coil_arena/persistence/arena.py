"""
Evolution arena with comprehensive persistence support.

This module extends the base evolution arena with:
- Automatic checkpointing
- Resumable evolution
- Experiment tracking
- Progress monitoring
- State management
"""

import logging
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import numpy as np

from genetic_coil_arena.config import Config
from genetic_coil_arena.evo.arena import EvolutionArena
from genetic_coil_arena.evo.genome import Genome
from genetic_coil_arena.persistence import (
    ModelPersistence, 
    CheckpointState, 
    ExperimentMetadata
)

logger = logging.getLogger(__name__)


class PersistentEvolutionArena(EvolutionArena):
    """Evolution arena with automatic persistence and state management."""
    
    def __init__(self, config: Config, experiment_id: Optional[str] = None,
                 base_path: str = "experiments", auto_checkpoint: bool = True,
                 checkpoint_frequency: int = 10):
        """Initialize persistent evolution arena.
        
        Args:
            config: Evolution configuration
            experiment_id: Existing experiment ID to resume, or None for new
            base_path: Base directory for experiments
            auto_checkpoint: Enable automatic checkpointing
            checkpoint_frequency: Save checkpoint every N generations
        """
        super().__init__(config)
        
        self.persistence = ModelPersistence(base_path)
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_frequency = checkpoint_frequency
        self.experiment_id = experiment_id
        self.experiment_path: Optional[Path] = None
        self.metadata: Optional[ExperimentMetadata] = None
        
        # Evolution state tracking
        self.generation = 0  # Initialize generation counter
        self.population: List[Genome] = []  # Initialize population
        self.fitness_history: List[List[float]] = []
        self.diversity_history: List[float] = []
        self.generation_callbacks: List[Callable] = []
        
        # Resume existing experiment or create new one
        if experiment_id:
            self._resume_experiment(experiment_id)
        else:
            self._create_new_experiment()
    
    def _create_new_experiment(self, description: str = "", tags: List[str] = None):
        """Create a new experiment."""
        self.experiment_id, self.experiment_path = self.persistence.create_experiment(
            self.config, description, tags
        )
        self.metadata = self.persistence.load_metadata(
            self.experiment_path / "metadata.json"
        )
        logger.info(f"Created new experiment: {self.experiment_id}")
    
    def _resume_experiment(self, experiment_id: str):
        """Resume an existing experiment from the latest checkpoint."""
        self.experiment_path = self.persistence.base_path / experiment_id
        
        if not self.experiment_path.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Load metadata
        metadata_path = self.experiment_path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Metadata not found for experiment {experiment_id}")
        
        self.metadata = self.persistence.load_metadata(metadata_path)
        
        # Find and load latest checkpoint
        checkpoint_path = self.persistence.find_latest_checkpoint(self.experiment_path)
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
            logger.info(f"Resumed experiment {experiment_id} from generation {self.generation}")
        else:
            logger.info(f"No checkpoint found for {experiment_id}, starting fresh")
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """Load evolution state from checkpoint."""
        state = self.persistence.load_checkpoint(checkpoint_path)
        
        # Restore evolution state
        self.generation = state.generation
        self.population = state.population
        self.hall_of_fame = state.hall_of_fame
        self.fitness_history = state.fitness_history
        self.diversity_history = state.diversity_history
        
        # Restore random state
        if 'numpy_state' in state.random_state:
            np.random.set_state(state.random_state['numpy_state'])
        if 'python_state' in state.random_state:
            random.setstate(state.random_state['python_state'])
        
        # Restore any additional evolution algorithm state
        for key, value in state.evolution_state.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        logger.info(f"Loaded checkpoint from generation {self.generation}")
    
    def _save_checkpoint(self):
        """Save current evolution state as checkpoint."""
        if not self.experiment_path:
            return
        
        # Capture random states
        random_state = {
            'numpy_state': np.random.get_state(),
            'python_state': random.getstate()
        }
        
        # Capture additional evolution state
        evolution_state = {}
        for attr in ['elite_size', 'mutation_rate', 'crossover_rate']:
            if hasattr(self, attr):
                evolution_state[attr] = getattr(self, attr)
        
        # Update metadata
        if self.metadata:
            self.metadata.total_generations = self.generation
            self.metadata.total_evaluations = len(self.fitness_history) * len(self.population)
            if self.hall_of_fame:
                self.metadata.best_fitness = max(ind.fitness.values[0] for ind in self.hall_of_fame)
        
        # Create checkpoint state
        state = CheckpointState(
            generation=self.generation,
            population=self.population[:],  # Copy to avoid reference issues
            hall_of_fame=self.hall_of_fame[:] if self.hall_of_fame else [],
            fitness_history=self.fitness_history[:],
            diversity_history=self.diversity_history[:],
            metadata=self.metadata,
            config=self.config,
            random_state=random_state,
            evolution_state=evolution_state
        )
        
        # Save checkpoint
        self.persistence.save_checkpoint(self.experiment_path, state)
        
        # Update metadata file
        if self.metadata:
            self.persistence.save_metadata(self.experiment_path, self.metadata)
    
    def add_generation_callback(self, callback: Callable[['PersistentEvolutionArena'], None]):
        """Add callback function to be called after each generation.
        
        Args:
            callback: Function that takes the arena as argument
        """
        self.generation_callbacks.append(callback)
    
    def _calculate_population_diversity(self) -> float:
        """Calculate genetic diversity of current population."""
        if len(self.population) < 2:
            return 0.0
        
        # Calculate pairwise distances between genomes
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Simple L2 distance between weight vectors
                weights1 = self.population[i].weights
                weights2 = self.population[j].weights
                distance = np.linalg.norm(weights1 - weights2)
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def evolve(self, generations: int, verbose: bool = True) -> Dict[str, Any]:
        """Run evolution with persistence and monitoring.
        
        Args:
            generations: Number of generations to evolve
            verbose: Print progress information
            
        Returns:
            Evolution results dictionary
        """
        start_generation = self.generation
        target_generation = start_generation + generations
        
        if verbose:
            logger.info(f"Starting evolution from generation {start_generation} "
                       f"to {target_generation}")
        
        try:
            for gen in range(generations):
                # Run one generation
                population_fitness = self.evolve_generation()
                
                # Track fitness and diversity
                self.fitness_history.append(population_fitness[:])
                diversity = self._calculate_population_diversity()
                self.diversity_history.append(diversity)
                
                # Log progress
                if verbose and (self.generation % 10 == 0 or self.generation == target_generation):
                    best_fitness = max(population_fitness)
                    avg_fitness = np.mean(population_fitness)
                    logger.info(f"Generation {self.generation}: "
                               f"best={best_fitness:.3f}, avg={avg_fitness:.3f}, "
                               f"diversity={diversity:.3f}")
                
                # Run callbacks
                for callback in self.generation_callbacks:
                    try:
                        callback(self)
                    except Exception as e:
                        logger.warning(f"Generation callback failed: {e}")
                
                # Auto-checkpoint
                if (self.auto_checkpoint and 
                    self.generation % self.checkpoint_frequency == 0):
                    self._save_checkpoint()
                    if verbose:
                        logger.info(f"Saved checkpoint at generation {self.generation}")
        
        except KeyboardInterrupt:
            logger.info("Evolution interrupted by user")
            if self.auto_checkpoint:
                self._save_checkpoint()
                logger.info("Saved final checkpoint")
        
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            if self.auto_checkpoint:
                self._save_checkpoint()
                logger.info("Saved emergency checkpoint")
            raise
        
        # Final checkpoint and results
        if self.auto_checkpoint:
            self._save_checkpoint()
        
        results = self._compile_results()
        self._save_final_results(results)
        
        if verbose:
            logger.info(f"Evolution completed. Total generations: {self.generation}")
        
        return results
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile comprehensive evolution results."""
        results = {
            'experiment_id': self.experiment_id,
            'total_generations': self.generation,
            'total_evaluations': len(self.fitness_history) * len(self.population),
            'config': self.config.__dict__,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history,
        }
        
        if self.fitness_history:
            fitness_array = np.array(self.fitness_history)
            results.update({
                'best_fitness_per_generation': np.max(fitness_array, axis=1).tolist(),
                'average_fitness_per_generation': np.mean(fitness_array, axis=1).tolist(),
                'final_best_fitness': np.max(fitness_array[-1]),
                'final_average_fitness': np.mean(fitness_array[-1]),
            })
        
        if self.hall_of_fame:
            results['hall_of_fame_fitness'] = [ind.fitness.values[0] 
                                              for ind in self.hall_of_fame]
        
        return results
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final experiment results."""
        if self.experiment_path:
            # Save detailed results
            self.persistence.save_results(self.experiment_path, results)
            
            # Save best agents
            if self.hall_of_fame:
                best_genome = self.hall_of_fame[0]
                self.persistence.save_genome(
                    self.experiment_path, best_genome, "best_genome.pkl"
                )                # Create and save best agent
                from genetic_coil_arena.snake_core.agent import SnakeAgent, get_observation_size
                input_size = get_observation_size(
                    self.config.game.board_size, 
                    self.config.game.observation_radius
                )
                best_agent = SnakeAgent(
                    input_size=input_size,
                    hidden_size=self.config.agent.hidden_size,
                    second_hidden_size=self.config.agent.second_hidden_size,
                    activation=self.config.agent.activation
                )
                best_agent.set_genome(best_genome.weights)
                self.persistence.save_agent(
                    self.experiment_path, best_agent, "best_agent.pkl"
                )
            
            # Save hall of fame
            if self.hall_of_fame:
                self.persistence.save_population(
                    self.experiment_path, self.hall_of_fame, "hall_of_fame.pkl"
                )
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary information about the current experiment."""
        summary = {
            'experiment_id': self.experiment_id,
            'current_generation': self.generation,
            'population_size': len(self.population) if self.population else 0,
            'config': self.config.__dict__,
        }
        
        if self.metadata:
            summary.update({
                'created_at': self.metadata.created_at.isoformat(),
                'description': self.metadata.description,
                'tags': self.metadata.tags,
            })
        
        if self.fitness_history:
            latest_fitness = self.fitness_history[-1]
            summary.update({
                'current_best_fitness': max(latest_fitness),
                'current_average_fitness': np.mean(latest_fitness),
                'generations_completed': len(self.fitness_history),
            })
        
        if self.diversity_history:
            summary['current_diversity'] = self.diversity_history[-1]
        
        return summary
    
    def export_experiment(self, output_path: Path, include_checkpoints: bool = False):
        """Export the current experiment to a portable archive.
        
        Args:
            output_path: Output file path for the archive
            include_checkpoints: Whether to include checkpoint files
        """
        if not self.experiment_id:
            raise ValueError("No experiment to export")
        
        self.persistence.export_experiment(
            self.experiment_id, output_path, include_checkpoints
        )


def create_persistent_arena(config: Config, experiment_id: Optional[str] = None,
                           **kwargs) -> PersistentEvolutionArena:
    """Convenience function to create a persistent evolution arena.
    
    Args:
        config: Evolution configuration
        experiment_id: Existing experiment ID to resume, or None for new
        **kwargs: Additional arguments for PersistentEvolutionArena
        
    Returns:
        Configured persistent evolution arena
    """
    return PersistentEvolutionArena(config, experiment_id, **kwargs)


def list_experiments(base_path: str = "experiments") -> List[Dict[str, Any]]:
    """List all available experiments with their metadata.
    
    Args:
        base_path: Base directory for experiments
        
    Returns:
        List of experiment information dictionaries
    """
    persistence = ModelPersistence(base_path)
    experiments = []
    
    for exp_id, metadata in persistence.list_experiments():
        experiments.append({
            'experiment_id': exp_id,
            'created_at': metadata.created_at.isoformat(),
            'description': metadata.description,
            'tags': metadata.tags,
            'total_generations': metadata.total_generations,
            'total_evaluations': metadata.total_evaluations,
            'best_fitness': metadata.best_fitness,
        })
    
    return experiments
