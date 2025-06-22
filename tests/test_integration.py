"""Integration tests for the complete genetic coil arena system."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from src.genetic_coil_arena.config import load_config
from src.genetic_coil_arena.snake_core.game import SnakeGame
from src.genetic_coil_arena.snake_core.agent import SnakeAgent, get_observation_size
from src.genetic_coil_arena.evo.genome import Genome
from src.genetic_coil_arena.evo.arena import EvolutionArena
from src.genetic_coil_arena.evo.advanced import NoveltySearch, MultiObjectiveOptimizer, SpeciationManager, AdaptiveEvolution, HyperparameterOptimizer
from src.genetic_coil_arena.analysis.evolution_monitor import EvolutionMonitor
from src.genetic_coil_arena.ui.viewer import GameViewer


class TestSystemIntegration:
    """Test complete system integration."""

    def test_config_loading(self):
        """Test configuration loading and validation."""
        config = load_config()
        
        # Check that all required sections exist
        assert hasattr(config, 'evolution')
        assert hasattr(config, 'neural_net')
        assert hasattr(config, 'game')
        assert hasattr(config, 'analysis')
        
        # Check specific values
        assert config.evolution.population_size > 0
        assert config.neural_net.hidden_size > 0
        assert config.game.board_size > 0

    def test_game_agent_integration(self):
        """Test game and agent working together."""
        config = load_config()
        
        # Create game and agent
        game = SnakeGame(
            board_size=config.game.board_size,
            observation_radius=config.game.observation_radius,
            seed=42        )
          # Get observation size and create agent
        obs_size = get_observation_size(
            board_size=config.game.board_size,
            observation_radius=config.game.observation_radius
        )
        agent = SnakeAgent(
            input_size=obs_size,
            hidden_size=16,
            seed=42
        )
        
        # Test game loop
        state = game.reset()
        for _ in range(10):
            observation = game.get_observation()
            action = agent.predict(observation)
            state, reward, done = game.step(action)
            
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            if done:
                break
        
        # Agent should be able to play
        assert game.steps > 0

    def test_evolution_arena_basic(self):
        """Test basic evolution arena functionality."""
        config = load_config()
        
        # Create small arena for testing
        arena = EvolutionArena(
            population_size=10,
            genome_size=50,
            elite_size=2,
            mutation_rate=0.1,
            game_config={
                'board_size': 16,
                'observation_radius': 3,
                'max_steps': 100
            },
            seed=42
        )
        
        # Run one generation
        results = arena.evolve_generation()
        
        assert len(results['population']) == 10
        assert len(results['fitnesses']) == 10
        assert all(f >= 0 for f in results['fitnesses'])
        assert results['generation'] == 1
        assert 'best_fitness' in results
        assert 'avg_fitness' in results

    def test_novelty_search_integration(self):
        """Test novelty search with evolution arena."""
        config = load_config()
        
        arena = EvolutionArena(
            population_size=8,
            genome_size=30,
            elite_size=2,
            mutation_rate=0.1,
            game_config={
                'board_size': 12,
                'observation_radius': 2,
                'max_steps': 50
            },
            seed=42
        )
        
        # Add novelty search
        novelty_search = NoveltySearch(
            archive_size=20,
            k_nearest=3,
            novelty_threshold=0.5
        )
        
        # Run generation with novelty search
        results = arena.evolve_generation()
        
        # Test novelty computation
        behaviors = []
        for genome in arena.population:
            # Simple behavior descriptor (final position)
            behavior = np.array([np.random.random(), np.random.random()])
            behaviors.append(behavior)
        
        novelty_scores = novelty_search.compute_novelty_scores(behaviors)
        
        assert len(novelty_scores) == len(arena.population)
        assert all(score >= 0 for score in novelty_scores)

    def test_multi_objective_optimization(self):
        """Test multi-objective optimization."""
        config = load_config()
        
        arena = EvolutionArena(
            population_size=8,
            genome_size=30,
            elite_size=2,
            mutation_rate=0.1,
            game_config={
                'board_size': 12,
                'observation_radius': 2,
                'max_steps': 50
            },
            seed=42
        )
        
        optimizer = MultiObjectiveOptimizer()
        
        # Create mock objectives (fitness, efficiency)
        objectives = np.random.random((8, 2))
        
        # Test Pareto front computation
        pareto_indices = optimizer.compute_pareto_front(objectives)
        
        assert len(pareto_indices) > 0
        assert all(0 <= idx < 8 for idx in pareto_indices)

    def test_speciation_manager(self):
        """Test speciation functionality."""
        config = load_config()
        
        # Create some genomes
        genomes = [Genome(genome_size=30) for _ in range(10)]
        
        speciation = SpeciationManager(
            compatibility_threshold=0.5,
            species_target=3
        )
        
        species = speciation.assign_species(genomes)
        
        assert len(species) > 0
        assert all(len(sp) > 0 for sp in species)
        assert sum(len(sp) for sp in species) == 10

    def test_adaptive_evolution(self):
        """Test adaptive evolution parameters."""
        config = load_config()
        
        adaptive = AdaptiveEvolution(
            initial_mutation_rate=0.1,
            initial_crossover_rate=0.7
        )
        
        # Test parameter adaptation
        diversity = 0.3  # Low diversity
        avg_fitness = 5.0
        
        new_mutation_rate = adaptive.adapt_mutation_rate(diversity, avg_fitness)
        new_crossover_rate = adaptive.adapt_crossover_rate(diversity, avg_fitness)
        
        assert 0.0 <= new_mutation_rate <= 1.0
        assert 0.0 <= new_crossover_rate <= 1.0

    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        config = load_config()
        
        optimizer = HyperparameterOptimizer(
            param_space={
                'mutation_rate': (0.01, 0.3),
                'population_size': (10, 50),
                'hidden_size': (8, 32)
            }
        )
        
        # Test parameter sampling
        params = optimizer.sample_parameters()
        
        assert 0.01 <= params['mutation_rate'] <= 0.3
        assert 10 <= params['population_size'] <= 50
        assert 8 <= params['hidden_size'] <= 32

    def test_evolution_monitor(self):
        """Test evolution monitoring and analysis."""
        config = load_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = EvolutionMonitor(
                log_dir=temp_dir,
                save_plots=True,
                plot_interval=1
            )
            
            # Log some generation data
            generation_data = {
                'generation': 1,
                'best_fitness': 10.5,
                'avg_fitness': 7.2,
                'diversity': 0.4,
                'population': [Genome(genome_size=30) for _ in range(5)]
            }
            
            monitor.log_generation(generation_data)
            
            # Check that files were created
            log_file = Path(temp_dir) / 'evolution.log'
            assert log_file.exists()
            
            # Test analysis
            analysis = monitor.analyze_evolution()
            assert 'fitness_trend' in analysis
            assert 'diversity_trend' in analysis

    def test_game_viewer_headless(self):
        """Test game viewer in headless mode."""
        config = load_config()
        
        game = SnakeGame(board_size=16, observation_radius=3, seed=42)
        agent = NeuralAgent(observation_size=game.observation_size, hidden_size=16, seed=42)
        
        # Test viewer creation (should not fail in headless mode)
        try:
            viewer = GameViewer(
                game=game,
                agent=agent,
                fps=60,
                window_size=400
            )
            # Just test that it can be created
            assert viewer is not None
        except Exception as e:
            # If pygame fails (e.g., no display), that's expected in CI
            assert "display" in str(e).lower() or "video" in str(e).lower()

    def test_complete_evolution_pipeline(self):
        """Test complete evolution pipeline with all components."""
        config = load_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create evolution arena
            arena = EvolutionArena(
                population_size=8,
                genome_size=40,
                elite_size=2,
                mutation_rate=0.1,
                game_config={
                    'board_size': 12,
                    'observation_radius': 2,
                    'max_steps': 50
                },
                seed=42
            )
            
            # Add monitoring
            monitor = EvolutionMonitor(
                log_dir=temp_dir,
                save_plots=False,  # Don't create plots in test
                plot_interval=10
            )
            
            # Add advanced features
            novelty_search = NoveltySearch(archive_size=10, k_nearest=3)
            adaptive = AdaptiveEvolution(
                initial_mutation_rate=0.1,
                initial_crossover_rate=0.7
            )
            
            # Run small evolution
            for generation in range(3):
                results = arena.evolve_generation()
                
                # Log results
                monitor.log_generation(results)
                
                # Adapt parameters
                diversity = np.std(results['fitnesses']) / (np.mean(results['fitnesses']) + 1e-8)
                new_mutation_rate = adaptive.adapt_mutation_rate(
                    diversity, results['avg_fitness']
                )
                arena.mutation_rate = new_mutation_rate
                
                # Basic validation
                assert len(results['population']) == 8
                assert len(results['fitnesses']) == 8
                assert results['generation'] == generation + 1
            
            # Test final analysis
            analysis = monitor.analyze_evolution()
            assert len(analysis['fitness_trend']) == 3
            
            # Test that log file was created
            log_file = Path(temp_dir) / 'evolution.log'
            assert log_file.exists()
            
            # Test config save/load
            config_file = Path(temp_dir) / 'test_config.yaml'
            config.save(str(config_file))
            assert config_file.exists()
            
            loaded_config = load_config(str(config_file))
            assert loaded_config.evolution.population_size == config.evolution.population_size

    def test_genome_operations(self):
        """Test genome creation, mutation, and crossover."""
        # Test genome creation
        genome1 = Genome(genome_size=50)
        genome2 = Genome(genome_size=50)
        
        assert len(genome1.weights) == 50
        assert len(genome2.weights) == 50
        assert genome1.id != genome2.id
        
        # Test mutation
        original_weights = genome1.weights.copy()
        mutated_genome = genome1.mutate(mutation_rate=0.5, mutation_strength=0.1)
        
        assert len(mutated_genome.weights) == 50
        # Some weights should have changed
        assert not np.allclose(original_weights, mutated_genome.weights)
        
        # Test crossover
        child1, child2 = genome1.crossover(genome2, crossover_rate=0.7)
        
        assert len(child1.weights) == 50
        assert len(child2.weights) == 50
        assert child1.id != child2.id
        assert child1.id != genome1.id
        assert child1.id != genome2.id

    def test_fitness_evaluation_consistency(self):
        """Test that fitness evaluation is consistent and deterministic."""
        config = load_config()
        
        # Create identical genomes
        genome1 = Genome(weights=np.ones(50) * 0.5)
        genome2 = Genome(weights=np.ones(50) * 0.5)
        
        arena = EvolutionArena(
            population_size=2,
            genome_size=50,
            elite_size=1,
            mutation_rate=0.0,  # No mutation for consistency
            game_config={
                'board_size': 12,
                'observation_radius': 2,
                'max_steps': 50
            },
            seed=42
        )
        
        # Replace population with our test genomes
        arena.population = [genome1, genome2]
        
        # Evaluate fitness multiple times
        fitnesses1 = arena.evaluator.evaluate_population(arena.population, seed=42)
        fitnesses2 = arena.evaluator.evaluate_population(arena.population, seed=42)
        
        # Should be identical with same seed
        assert np.allclose(fitnesses1, fitnesses2)
        
        # Should be different with different seed
        fitnesses3 = arena.evaluator.evaluate_population(arena.population, seed=123)
        assert not np.allclose(fitnesses1, fitnesses3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
