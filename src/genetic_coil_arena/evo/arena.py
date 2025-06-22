"""Main evolution arena that orchestrates the genetic algorithm."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import time
import json
import pickle

from ..config import Config
from ..snake_core.agent import get_observation_size, SnakeAgent
from .genome import Genome, GenomeStats, create_random_population, create_next_generation
from .fitness import FitnessEvaluator, BatchEvaluator


@dataclass
class GenerationStats:
    """Statistics for a generation."""
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    best_score: float
    mean_score: float
    best_length: float
    mean_length: float
    success_rate: float
    diversity: float  # Standard deviation of fitness
    evaluation_time: float
    relative_to_baseline: float


@dataclass  
class EvolutionResult:
    """Results from an evolution run."""
    config: Config
    generation_stats: List[GenerationStats]
    hall_of_fame: List[Genome]  # Best genomes across all generations
    final_population: List[Genome]
    total_time: float
    converged: bool
    convergence_generation: Optional[int]


class EvolutionArena:
    """Main class for running genetic algorithm evolution."""
    
    def __init__(self, config: Config):
        """Initialize evolution arena.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Calculate observation size
        self.observation_size = get_observation_size(
            config.game.board_size,
            config.game.observation_radius
        )
        
        # Calculate genome size for neural network
        agent_config = asdict(config.agent)
        self.genome_size = self._calculate_genome_size(agent_config)
        
        # Setup fitness evaluator
        game_config = asdict(config.game)
        eval_config = asdict(config.evaluation)
        
        self.fitness_evaluator = FitnessEvaluator(
            game_config=game_config,
            agent_config=agent_config,
            episodes_per_agent=eval_config["episodes_per_agent"],
            parallel_processes=eval_config["parallel_processes"],
            deterministic_seed=eval_config["deterministic_seed"]
        )
        
        self.batch_evaluator = BatchEvaluator(
            self.fitness_evaluator,
            batch_size=eval_config["batch_size"]
        )
        
        # Initialize tracking
        self.generation_stats: List[GenerationStats] = []
        self.hall_of_fame: List[Genome] = []
        self.best_fitness_history: List[float] = []
        
        # Setup output directory
        self.output_dir = Path(config.logging.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _calculate_genome_size(self, agent_config: Dict[str, Any]) -> int:
        """Calculate total number of parameters in the neural network."""
        input_size = self.observation_size
        hidden_size = agent_config["hidden_size"]
        second_hidden_size = agent_config["second_hidden_size"]
        output_size = 3  # 3 actions
        
        # Weights and biases for each layer
        layer1_params = input_size * hidden_size + hidden_size
        layer2_params = hidden_size * second_hidden_size + second_hidden_size  
        layer3_params = second_hidden_size * output_size + output_size
        
        return layer1_params + layer2_params + layer3_params
    
    def evolve(self) -> EvolutionResult:
        """Run the genetic algorithm evolution.
        
        Returns:
            Evolution results
        """
        start_time = time.time()
        
        print(f"Starting evolution with {self.config.evolution.population_size} agents")
        print(f"Genome size: {self.genome_size} parameters")
        print(f"Observation size: {self.observation_size}")
        print(f"Target: {self.config.evolution.generations} generations")
        
        # Initialize population
        population = create_random_population(
            self.config.evolution.population_size,
            self.genome_size,
            self.config.evaluation.deterministic_seed
        )
        
        converged = False
        convergence_generation = None
        no_improvement_count = 0
        
        for generation in range(self.config.evolution.generations):
            gen_start_time = time.time()
            
            print(f"\n=== Generation {generation + 1}/{self.config.evolution.generations} ===")
            
            # Evaluate population
            stats_list = self.batch_evaluator.evaluate_population(population)
            
            # Calculate generation statistics
            gen_stats = self._calculate_generation_stats(
                generation, stats_list, time.time() - gen_start_time
            )
            
            self.generation_stats.append(gen_stats)
            self.best_fitness_history.append(gen_stats.best_fitness)
            
            # Update hall of fame
            self._update_hall_of_fame(population)
            
            # Print progress
            self._print_generation_summary(gen_stats)
            
            # Check convergence
            if self._check_convergence(gen_stats):
                no_improvement_count += 1
                if no_improvement_count >= self.config.evolution.early_stop_patience:
                    converged = True
                    convergence_generation = generation
                    print(f"\nConverged at generation {generation + 1}")
                    break
            else:
                no_improvement_count = 0
            
            # Save checkpoint
            if (self.config.logging.save_checkpoints and 
                generation % self.config.logging.checkpoint_interval == 0):
                self._save_checkpoint(generation, population)
            
            # Create next generation (except for last generation)
            if generation < self.config.evolution.generations - 1:
                population = create_next_generation(
                    population,
                    self.config.evolution.elite_size,
                    self.config.evolution.crossover_alpha,
                    self.config.evolution.mutation_sigma,
                    self.config.evolution.mutation_prob,
                    self.config.evolution.tournament_size
                )
        
        total_time = time.time() - start_time
        
        # Create final result
        result = EvolutionResult(
            config=self.config,
            generation_stats=self.generation_stats,
            hall_of_fame=self.hall_of_fame,
            final_population=population,
            total_time=total_time,
            converged=converged,
            convergence_generation=convergence_generation
        )
          # Save final results
        self._save_results(result)
        
        print(f"\nEvolution completed in {total_time:.2f} seconds")
        if self.hall_of_fame:
            print(f"Best fitness achieved: {self.hall_of_fame[0].fitness:.3f}")
            if self.generation_stats:
                last_gen_stats = self.generation_stats[-1]
                print(f"Improvement over baseline: {last_gen_stats.relative_to_baseline:.2f}x")
        else:
            print("No valid genomes found during evolution")
        
        return result
    
    def _calculate_generation_stats(self, generation: int, stats_list: List[GenomeStats], 
                                  evaluation_time: float) -> GenerationStats:
        """Calculate statistics for a generation."""
        fitness_values = [stats.fitness for stats in stats_list]
        scores = [stats.avg_score for stats in stats_list]
        lengths = [stats.avg_length for stats in stats_list]
        success_rates = [stats.success_rate for stats in stats_list]
        
        best_fitness = max(fitness_values)
        relative_to_baseline = self.fitness_evaluator.get_performance_relative_to_baseline(best_fitness)
        
        return GenerationStats(
            generation=generation,
            best_fitness=best_fitness,
            mean_fitness=float(np.mean(fitness_values)),
            std_fitness=float(np.std(fitness_values)),
            best_score=max(scores),
            mean_score=float(np.mean(scores)),
            best_length=max(lengths),
            mean_length=float(np.mean(lengths)),
            success_rate=float(np.mean(success_rates)),
            diversity=float(np.std(fitness_values)),
            evaluation_time=evaluation_time,
            relative_to_baseline=relative_to_baseline
        )
    
    def _update_hall_of_fame(self, population: List[Genome]) -> None:
        """Update hall of fame with best genomes."""
        # Add all genomes to hall of fame
        self.hall_of_fame.extend(population)
        
        # Sort by fitness and keep only the best
        self.hall_of_fame.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep top 10 unique genomes
        unique_genomes = []
        seen_fitness = set()
        
        for genome in self.hall_of_fame:
            fitness_key = round(genome.fitness, 6)  # Round to avoid floating point issues
            if fitness_key not in seen_fitness and len(unique_genomes) < 10:
                unique_genomes.append(genome)
                seen_fitness.add(fitness_key)
        
        self.hall_of_fame = unique_genomes
    
    def _check_convergence(self, gen_stats: GenerationStats) -> bool:
        """Check if algorithm has converged (no improvement)."""
        if len(self.best_fitness_history) < 2:
            return False
        
        # Check if there's been no improvement in best fitness
        current_best = gen_stats.best_fitness
        previous_best = self.best_fitness_history[-2]
        
        improvement = current_best - previous_best
        return improvement < 1e-6  # Threshold for "no improvement"
    
    def _print_generation_summary(self, stats: GenerationStats) -> None:
        """Print summary of generation results."""
        print(f"Best fitness: {stats.best_fitness:.3f} "
              f"(Mean: {stats.mean_fitness:.3f} ± {stats.std_fitness:.3f})")
        print(f"Best score: {stats.best_score:.1f} "
              f"(Mean: {stats.mean_score:.2f})")
        print(f"Success rate: {stats.success_rate:.1%}")
        print(f"Relative to baseline: {stats.relative_to_baseline:.2f}x")
    
    def _save_checkpoint(self, generation: int, population: List[Genome]) -> None:
        """Save checkpoint data."""
        checkpoint_path = self.output_dir / f"checkpoint_gen_{generation:03d}.pkl"
        
        checkpoint_data = {
            "generation": generation,
            "population": population,
            "hall_of_fame": self.hall_of_fame,
            "generation_stats": self.generation_stats,
            "config": self.config
        }
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_results(self, result: EvolutionResult) -> None:
        """Save final evolution results."""
        # Save generation statistics as CSV
        stats_df = pd.DataFrame([asdict(stats) for stats in result.generation_stats])
        stats_path = self.output_dir / "generation_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        
        # Save hall of fame
        if result.hall_of_fame and self.config.logging.save_best_agents:
            hall_of_fame_path = self.output_dir / "hall_of_fame.pkl"
            with open(hall_of_fame_path, "wb") as f:
                pickle.dump(result.hall_of_fame, f)
        
        # Save configuration
        config_path = self.output_dir / "config.yaml"
        self.config.to_yaml(config_path)
        
        # Save summary
        summary = {
            "total_time": result.total_time,
            "converged": result.converged,
            "convergence_generation": result.convergence_generation,
            "best_fitness": result.hall_of_fame[0].fitness if result.hall_of_fame else 0.0,
            "final_generation": len(result.generation_stats),
            "genome_size": self.genome_size,
            "observation_size": self.observation_size
        }
        
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to: {self.output_dir}")
    
    def evaluate_agent(self, genome: Genome, episodes: int = 10) -> GenomeStats:
        """Evaluate a specific agent.
        
        Args:
            genome: Genome to evaluate
            episodes: Number of episodes to run
            
        Returns:
            Evaluation statistics
        """
        # Temporarily override episodes per agent
        original_episodes = self.fitness_evaluator.episodes_per_agent
        self.fitness_evaluator.episodes_per_agent = episodes
        
        try:
            stats = self.fitness_evaluator.evaluate_genome(genome)
            return stats
        finally:
            self.fitness_evaluator.episodes_per_agent = original_episodes
    
    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load evolution state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Generation number loaded
        """
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        
        self.generation_stats = checkpoint_data["generation_stats"]
        self.hall_of_fame = checkpoint_data["hall_of_fame"]
        self.best_fitness_history = [stats.best_fitness for stats in self.generation_stats]
        
        return checkpoint_data["generation"]
