"""Simple integration test for core functionality."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.genetic_coil_arena.config import load_config
from src.genetic_coil_arena.snake_core.game import SnakeGame, Action
from src.genetic_coil_arena.snake_core.agent import SnakeAgent, get_observation_size
from src.genetic_coil_arena.evo.genome import Genome


class TestBasicIntegration:
    """Test basic system integration."""

    def test_config_loading(self):
        """Test configuration loading and validation."""
        config = load_config()
        
        # Check that basic sections exist
        assert hasattr(config, 'evolution')
        assert hasattr(config, 'game')
        
        # Check specific values
        assert config.evolution.population_size > 0
        assert config.game.board_size > 0

    def test_game_basic_functionality(self):
        """Test basic game functionality."""
        game = SnakeGame(board_size=16, observation_radius=3, seed=42)
        
        # Test initialization
        state = game.reset()
        assert state.is_alive
        assert state.score == 0
        assert state.steps == 0
        
        # Test steps
        for i in range(5):
            state, reward, done = game.step(Action.STRAIGHT)
            assert state.steps == i + 1
            if done:
                break

    def test_agent_basic_functionality(self):
        """Test basic agent functionality."""
        obs_size = get_observation_size(board_size=16, observation_radius=3)
        agent = SnakeAgent(input_size=obs_size, hidden_size=16, seed=42)
        
        # Test prediction
        dummy_observation = np.random.random(obs_size).astype(np.float32)
        action = agent.predict(dummy_observation)
        
        assert action in [Action.STRAIGHT, Action.LEFT, Action.RIGHT]
        
        # Test genome operations
        genome = agent.get_genome()
        assert isinstance(genome, np.ndarray)
        assert len(genome) == agent.genome_size
        
        # Test setting genome
        new_genome = np.random.random(agent.genome_size).astype(np.float32)
        agent.set_genome(new_genome)
        assert np.allclose(agent.get_genome(), new_genome)

    def test_game_agent_integration(self):
        """Test game and agent working together."""
        # Create game
        game = SnakeGame(board_size=16, observation_radius=3, seed=42)
        
        # Create agent
        obs_size = get_observation_size(board_size=16, observation_radius=3)
        agent = SnakeAgent(input_size=obs_size, hidden_size=16, seed=42)
        
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
        mutated_genome = genome1.mutate(sigma=0.1, mutation_prob=0.5)
        
        assert len(mutated_genome.weights) == 50
        # Some weights should have changed
        assert not np.allclose(original_weights, mutated_genome.weights)
          # Test crossover
        child1, child2 = genome1.crossover(genome2, alpha=0.3)
        
        assert len(child1.weights) == 50
        assert len(child2.weights) == 50
        assert child1.id != child2.id
        assert child1.id != genome1.id
        assert child1.id != genome2.id

    def test_multiple_agents_different_genomes(self):
        """Test that different agents with different genomes behave differently."""
        obs_size = get_observation_size(board_size=16, observation_radius=3)
        
        # Create two agents with different genomes
        agent1 = SnakeAgent(input_size=obs_size, hidden_size=16, seed=42)
        agent2 = SnakeAgent(input_size=obs_size, hidden_size=16, seed=123)
        
        # They should have different genomes
        genome1 = agent1.get_genome()
        genome2 = agent2.get_genome()
        assert not np.allclose(genome1, genome2)
        
        # Test on same observation
        dummy_observation = np.random.random(obs_size).astype(np.float32)
        
        # Get probabilities (should be different)
        probs1 = agent1.predict_probabilities(dummy_observation)
        probs2 = agent2.predict_probabilities(dummy_observation)
        
        assert len(probs1) == 3  # 3 actions
        assert len(probs2) == 3
        assert np.sum(probs1) == pytest.approx(1.0, abs=1e-6)
        assert np.sum(probs2) == pytest.approx(1.0, abs=1e-6)
        
        # Should be different (with high probability)
        assert not np.allclose(probs1, probs2, atol=0.1)

    def test_agent_consistency(self):
        """Test that agent produces consistent results with same input."""
        obs_size = get_observation_size(board_size=16, observation_radius=3)
        agent = SnakeAgent(input_size=obs_size, hidden_size=16, seed=42)
        
        # Create observation
        observation = np.random.random(obs_size).astype(np.float32)
        
        # Should get consistent results
        action1 = agent.predict(observation)
        action2 = agent.predict(observation)
        assert action1 == action2
        
        probs1 = agent.predict_probabilities(observation)
        probs2 = agent.predict_probabilities(observation)
        assert np.allclose(probs1, probs2)

    def test_deterministic_behavior(self):
        """Test that seeded components produce deterministic results."""
        # Test game determinism
        game1 = SnakeGame(board_size=16, observation_radius=3, seed=42)
        game2 = SnakeGame(board_size=16, observation_radius=3, seed=42)
        
        state1 = game1.reset()
        state2 = game2.reset()
        
        assert state1.food_position == state2.food_position
        
        # Test agent determinism
        obs_size = get_observation_size(board_size=16, observation_radius=3)
        agent1 = SnakeAgent(input_size=obs_size, hidden_size=16, seed=42)
        agent2 = SnakeAgent(input_size=obs_size, hidden_size=16, seed=42)
        
        # Should have identical genomes
        assert np.allclose(agent1.get_genome(), agent2.get_genome())

    def test_game_edge_cases(self):
        """Test game edge cases and error conditions."""
        game = SnakeGame(board_size=8, observation_radius=2, seed=42)
        state = game.reset()
          # Test that game handles many steps
        done = False
        for _ in range(100):
            state, reward, done = game.step(Action.STRAIGHT)
            if done:
                break
        
        # Game should eventually end
        assert done or game.steps >= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
