"""Fitness evaluation for snake agents."""

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import multiprocessing as mp
import time

from ..snake_core.game import SnakeGame
from ..snake_core.agent import SnakeAgent, get_observation_size
from .genome import Genome, GenomeStats


@dataclass
class EvaluationResult:
    """Result of evaluating a single episode."""
    score: int
    steps: int
    final_length: int
    success: bool  # Whether agent reached food before timeout


def evaluate_agent_episode(genome_weights: np.ndarray, game_config: Dict[str, Any], 
                          agent_config: Dict[str, Any], episode_seed: int) -> EvaluationResult:
    """Evaluate a single episode for an agent.
    
    This function is designed to be called in a separate process.
    
    Args:
        genome_weights: Flattened weight vector for the agent
        game_config: Game configuration parameters
        agent_config: Agent configuration parameters  
        episode_seed: Random seed for this episode
        
    Returns:
        Evaluation result for this episode
    """
    # Create game and agent
    game = SnakeGame(
        board_size=game_config["board_size"],
        observation_radius=game_config["observation_radius"],
        max_steps=game_config["max_steps"],
        seed=episode_seed
    )
    
    obs_size = get_observation_size(
        game_config["board_size"],
        game_config["observation_radius"]
    )
    
    agent = SnakeAgent(
        input_size=obs_size,
        hidden_size=agent_config["hidden_size"],
        second_hidden_size=agent_config["second_hidden_size"],
        activation=agent_config["activation"]
    )
    
    agent.set_genome(genome_weights)
    
    # Run episode
    state = game.reset()
    total_reward = 0.0
    
    while state.is_alive:
        observation = game.get_observation()
        action = agent.predict(observation)
        
        state, reward, done = game.step(action)
        total_reward += reward
        
        if done:
            break
    
    return EvaluationResult(
        score=state.score,
        steps=state.steps,
        final_length=state.snake_length,
        success=state.score > 0
    )


class FitnessEvaluator:
    """Parallel fitness evaluator for snake agents."""
    
    def __init__(self, game_config: Dict[str, Any], agent_config: Dict[str, Any],
                 episodes_per_agent: int = 3, parallel_processes: int = 8,
                 deterministic_seed: Optional[int] = None):
        """Initialize evaluator.
        
        Args:
            game_config: Game configuration parameters
            agent_config: Agent configuration parameters
            episodes_per_agent: Number of episodes to evaluate per agent
            parallel_processes: Number of parallel processes to use
            deterministic_seed: Base seed for deterministic evaluation
        """
        self.game_config = game_config
        self.agent_config = agent_config
        self.episodes_per_agent = episodes_per_agent
        self.parallel_processes = parallel_processes
        self.deterministic_seed = deterministic_seed
        
        # Calculate expected performance metrics for baseline
        self._calculate_baseline()
    
    def _calculate_baseline(self) -> None:
        """Calculate baseline performance for random agent."""
        # Estimate random agent performance
        # Random agent typically gets ~2-3 average fitness
        self.random_baseline = 2.1
    
    def evaluate_genome(self, genome: Genome) -> GenomeStats:
        """Evaluate a single genome across multiple episodes.
        
        Args:
            genome: Genome to evaluate
            
        Returns:
            Genome statistics
        """
        results = self.evaluate_genomes([genome])
        return results[0]
    
    def evaluate_genomes(self, genomes: List[Genome]) -> List[GenomeStats]:
        """Evaluate multiple genomes in parallel.
        
        Args:
            genomes: List of genomes to evaluate
            
        Returns:
            List of genome statistics
        """
        start_time = time.time()
        
        # Prepare evaluation tasks
        tasks = []
        for genome_idx, genome in enumerate(genomes):
            for episode in range(self.episodes_per_agent):
                if self.deterministic_seed is not None:
                    # Use deterministic seed based on genome ID and episode
                    episode_seed = self.deterministic_seed + genome.id * 1000 + episode
                else:
                    episode_seed = None
                
                tasks.append((genome_idx, genome.weights, episode_seed))
        
        # Execute tasks in parallel
        results = {}
        with ProcessPoolExecutor(max_workers=self.parallel_processes) as executor:
            # Submit all tasks
            future_to_task = {}
            for genome_idx, weights, episode_seed in tasks:
                future = executor.submit(
                    evaluate_agent_episode,
                    weights,
                    self.game_config,
                    self.agent_config,
                    episode_seed
                )
                future_to_task[future] = (genome_idx, episode_seed)
            
            # Collect results
            for future in as_completed(future_to_task):
                genome_idx, episode_seed = future_to_task[future]
                try:
                    result = future.result()
                    if genome_idx not in results:
                        results[genome_idx] = []
                    results[genome_idx].append(result)
                except Exception as e:
                    print(f"Error evaluating genome {genome_idx}: {e}")
                    # Add a failed result
                    if genome_idx not in results:
                        results[genome_idx] = []
                    results[genome_idx].append(EvaluationResult(0, 0, 1, False))
        
        # Calculate statistics for each genome
        stats_list = []
        for genome_idx, genome in enumerate(genomes):
            episode_results = results.get(genome_idx, [])
            
            if not episode_results:
                # No results - create failed stats
                stats = GenomeStats(
                    fitness=0.0,
                    episodes_played=0,
                    avg_score=0.0,
                    avg_length=1.0,
                    avg_steps=0.0,
                    success_rate=0.0
                )
            else:
                # Calculate aggregate statistics
                scores = [r.score for r in episode_results]
                steps = [r.steps for r in episode_results]
                lengths = [r.final_length for r in episode_results]
                successes = [r.success for r in episode_results]
                
                # Calculate fitness: food_eaten + 0.1 * steps_survived
                fitness_values = []
                for result in episode_results:
                    fitness = result.score + 0.1 * result.steps
                    # Apply early death penalty if applicable
                    if result.steps < 50 and result.score == 0:
                        fitness -= 2.0
                    fitness_values.append(fitness)
                
                stats = GenomeStats(
                    fitness=float(np.mean(fitness_values)),
                    episodes_played=len(episode_results),
                    avg_score=float(np.mean(scores)),
                    avg_length=float(np.mean(lengths)),
                    avg_steps=float(np.mean(steps)),
                    success_rate=float(np.mean(successes))
                )
            
            # Update genome with stats
            genome.stats = stats
            stats_list.append(stats)
        
        evaluation_time = time.time() - start_time
        games_per_second = (len(genomes) * self.episodes_per_agent) / evaluation_time
        
        print(f"Evaluated {len(genomes)} genomes ({len(genomes) * self.episodes_per_agent} episodes) "
              f"in {evaluation_time:.2f}s ({games_per_second:.0f} games/s)")
        
        return stats_list
    
    def get_performance_relative_to_baseline(self, fitness: float) -> float:
        """Get performance relative to random baseline.
        
        Args:
            fitness: Fitness value to compare
            
        Returns:
            Multiplier relative to baseline (e.g., 2.0 = 2x better than random)
        """
        return fitness / self.random_baseline if self.random_baseline > 0 else 0.0


class BatchEvaluator:
    """Batch evaluator that groups agents for more efficient evaluation."""
    
    def __init__(self, fitness_evaluator: FitnessEvaluator, batch_size: int = 16):
        """Initialize batch evaluator.
        
        Args:
            fitness_evaluator: Base fitness evaluator
            batch_size: Number of agents to evaluate in each batch
        """
        self.fitness_evaluator = fitness_evaluator
        self.batch_size = batch_size
    
    def evaluate_population(self, population: List[Genome]) -> List[GenomeStats]:
        """Evaluate entire population in batches.
        
        Args:
            population: Population to evaluate
            
        Returns:
            List of genome statistics
        """
        all_stats = []
        
        # Process population in batches
        for i in range(0, len(population), self.batch_size):
            batch = population[i:i + self.batch_size]
            batch_stats = self.fitness_evaluator.evaluate_genomes(batch)
            all_stats.extend(batch_stats)
        
        return all_stats
