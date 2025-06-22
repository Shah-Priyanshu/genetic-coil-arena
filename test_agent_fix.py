#!/usr/bin/env python3
"""Quick test to verify agent functionality after Numba fixes."""

import sys
import os
sys.path.insert(0, 'src')

def test_agent():
    """Test if agent prediction works."""
    try:
        from genetic_coil_arena.snake_core.agent import SnakeAgent
        from genetic_coil_arena.snake_core.game import SnakeGame
        import numpy as np
        
        print("🔧 Testing agent functionality...")
        
        # Create agent and game
        agent = SnakeAgent(input_size=124, hidden_size=32, second_hidden_size=16)
        game = SnakeGame(board_size=12, observation_radius=5)
        
        # Get observation
        obs = game.get_observation()
        print(f"📊 Observation shape: {obs.shape}, dtype: {obs.dtype}")
        
        # Test prediction
        action = agent.predict(obs)
        print(f"🎯 Prediction successful: {action}")
        
        # Test multiple predictions for consistency
        actions = []
        for i in range(5):
            action = agent.predict(obs)
            actions.append(action)
        
        print(f"🔄 Multiple predictions: {actions}")
        
        # Test game step
        state, reward, done = game.step(actions[0])
        print(f"🎮 Game step successful: reward={reward}, done={done}")
        
        print("✅ All tests passed! Agent is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agent()
    sys.exit(0 if success else 1)
