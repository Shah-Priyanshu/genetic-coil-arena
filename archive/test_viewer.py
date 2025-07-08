#!/usr/bin/env python3
"""Test script for the game viewer."""

import sys
sys.path.insert(0, 'src')

def test_viewer():
    """Test the game viewer with a simple demo."""
    try:
        from genetic_coil_arena.ui.viewer import GameViewer, ViewMode, VisualizationConfig
        from genetic_coil_arena.snake_core.game import SnakeGame
        from genetic_coil_arena.snake_core.agent import SnakeAgent
        
        print("🎮 Testing Game Viewer...")
        print("This will open a pygame window - close it with ESC or the X button")
        
        # Create a simple game
        game = SnakeGame(board_size=12, observation_radius=3, seed=42)
        obs_size = game.get_observation().shape[0]
        agent = SnakeAgent(input_size=obs_size, seed=42)
        
        # Create viewer with custom config
        vis_config = VisualizationConfig(
            window_width=1000,
            window_height=700,
            fps=5  # Slower for testing
        )
        viewer = GameViewer(vis_config)
        
        print("Running game session with AI agent...")
        print("Controls: ESC to quit early")
        
        # Run a game session
        stats = viewer.run_game_session(
            game=game,
            agent=agent,
            max_steps=200,
            mode=ViewMode.GAME_ONLY
        )
        
        print(f"✅ Game session completed!")
        print(f"   Final score: {stats['final_score']}")
        print(f"   Steps taken: {stats['steps_taken']}")
        print(f"   Total reward: {stats['total_reward']:.2f}")
        print(f"   Reason ended: {stats['reason_ended']}")
        
        viewer.close()
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure pygame is installed: pip install pygame")
        return False
    except Exception as e:
        print(f"❌ Error during viewer test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_viewer()
    if success:
        print("\n🎉 Viewer test completed successfully!")
    else:
        print("\n💥 Viewer test failed.")
    sys.exit(0 if success else 1)
