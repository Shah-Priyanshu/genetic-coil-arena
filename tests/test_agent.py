"""Test the neural network agent."""

import pytest
import numpy as np

from genetic_coil_arena.snake_core.agent import (
    SnakeAgent, create_random_agent, get_observation_size, Action
)


class TestSnakeAgent:
    """Test the SnakeAgent class."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = SnakeAgent(input_size=100, hidden_size=32, seed=42)
        
        assert agent.input_size == 100
        assert agent.hidden_size == 32
        assert agent.second_hidden_size == 16
        assert agent.output_size == 3
        assert agent.genome_size > 0
    
    def test_genome_operations(self):
        """Test genome get/set operations."""
        agent = SnakeAgent(input_size=10, hidden_size=5, seed=42)
        
        # Get initial genome
        genome1 = agent.get_genome()
        assert len(genome1) == agent.genome_size
        assert genome1.dtype == np.float32
        
        # Modify genome and set it back
        genome2 = genome1.copy()
        genome2[0] += 1.0
        agent.set_genome(genome2)
        
        # Verify it was set correctly
        genome3 = agent.get_genome()
        np.testing.assert_array_equal(genome2, genome3)
    
    def test_prediction(self):
        """Test action prediction."""
        agent = SnakeAgent(input_size=10, seed=42)
        observation = np.random.random(10).astype(np.float32)
        
        # Test predict method
        action = agent.predict(observation)
        assert isinstance(action, Action)
        assert action in [Action.STRAIGHT, Action.LEFT, Action.RIGHT]
        
        # Test predict_probabilities method
        probs = agent.predict_probabilities(observation)
        assert len(probs) == 3
        assert np.isclose(np.sum(probs), 1.0)  # Should sum to 1 (softmax)
        assert np.all(probs >= 0)  # All probabilities should be non-negative
    
    def test_deterministic_prediction(self):
        """Test that same observation produces same prediction."""
        agent1 = SnakeAgent(input_size=10, seed=42)
        agent2 = SnakeAgent(input_size=10, seed=42)
        
        observation = np.random.random(10).astype(np.float32)
        
        action1 = agent1.predict(observation)
        action2 = agent2.predict(observation)
        
        assert action1 == action2
    
    def test_copy(self):
        """Test agent copying."""
        agent1 = SnakeAgent(input_size=10, seed=42)
        agent2 = agent1.copy()
        
        # Should have same genome
        np.testing.assert_array_equal(agent1.get_genome(), agent2.get_genome())
        
        # But should be different objects
        assert agent1 is not agent2
        
        # Modifying one shouldn't affect the other
        genome = agent1.get_genome()
        genome[0] += 1.0
        agent1.set_genome(genome)
        
        assert not np.array_equal(agent1.get_genome(), agent2.get_genome())
    
    def test_different_architectures(self):
        """Test different network architectures."""
        # Small network
        agent_small = SnakeAgent(input_size=5, hidden_size=8, second_hidden_size=4)
        
        # Large network  
        agent_large = SnakeAgent(input_size=100, hidden_size=64, second_hidden_size=32)
        
        assert agent_small.genome_size < agent_large.genome_size
        
        # Both should work with appropriate observations
        obs_small = np.random.random(5).astype(np.float32)
        obs_large = np.random.random(100).astype(np.float32)
        
        action_small = agent_small.predict(obs_small)
        action_large = agent_large.predict(obs_large)
        
        assert isinstance(action_small, Action)
        assert isinstance(action_large, Action)
    
    def test_activation_functions(self):
        """Test different activation functions."""
        activations = ["tanh", "relu", "sigmoid"]
        
        for activation in activations:
            agent = SnakeAgent(input_size=10, activation=activation, seed=42)
            observation = np.random.random(10).astype(np.float32)
            
            action = agent.predict(observation)
            assert isinstance(action, Action)
    
    def test_genome_size_calculation(self):
        """Test genome size calculation."""
        input_size = 10
        hidden_size = 8
        second_hidden_size = 4
        output_size = 3
        
        agent = SnakeAgent(
            input_size=input_size,
            hidden_size=hidden_size,
            second_hidden_size=second_hidden_size
        )
        
        # Calculate expected size manually
        layer1_params = input_size * hidden_size + hidden_size
        layer2_params = hidden_size * second_hidden_size + second_hidden_size
        layer3_params = second_hidden_size * output_size + output_size
        expected_size = layer1_params + layer2_params + layer3_params
        
        assert agent.genome_size == expected_size


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_random_agent(self):
        """Test random agent creation."""
        agent = create_random_agent(input_size=20, seed=42)
        
        assert agent.input_size == 20
        assert agent.hidden_size == 32  # Default
        assert isinstance(agent, SnakeAgent)
    
    def test_get_observation_size(self):
        """Test observation size calculation."""
        # Default parameters
        size = get_observation_size()
        expected = 11 * 11 + 1 + 2  # Grid + length + food direction
        assert size == expected
        
        # Custom parameters
        size_custom = get_observation_size(board_size=16, observation_radius=3)
        expected_custom = 7 * 7 + 1 + 2  # 7x7 grid + length + food direction
        assert size_custom == expected_custom
    
    def test_observation_size_consistency(self):
        """Test that observation size calculation is consistent."""
        board_size = 32
        observation_radius = 5
        
        # Calculate size using utility function
        calculated_size = get_observation_size(board_size, observation_radius)
        
        # Calculate size manually
        grid_size = (2 * observation_radius + 1) ** 2
        manual_size = grid_size + 1 + 2
        
        assert calculated_size == manual_size


class TestAgentIntegration:
    """Integration tests with game environment."""
    
    def test_agent_with_real_observation(self):
        """Test agent with real game observation."""
        from genetic_coil_arena.snake_core.game import SnakeGame
        
        # Create game and agent with compatible sizes
        game = SnakeGame(board_size=16, observation_radius=3, seed=42)
        obs_size = get_observation_size(16, 3)
        agent = SnakeAgent(input_size=obs_size, seed=42)
        
        # Get real observation from game
        game.reset()
        observation = game.get_observation()
        
        # Agent should be able to process it
        action = agent.predict(observation)
        assert isinstance(action, Action)
        
        # And we should be able to use the action
        state, reward, done = game.step(action)
        assert isinstance(reward, float)
        assert isinstance(done, bool)


if __name__ == "__main__":
    pytest.main([__file__])
