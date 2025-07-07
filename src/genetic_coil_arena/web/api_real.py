"""
Integrated FastAPI routes for the web dashboard with persistent experiment management.
Phase 3B: Real Experiment Execution
"""

import asyncio
import uuid
import logging
import numpy as np
from threading import Thread
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Import analytics and visualization modules
try:
    from .analytics import analytics_router
    from .visualization import viz_router
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

try:
    from .models import (
        ExperimentConfig, ExperimentStatus, GenerationStats, 
        ExperimentListItem, WebSocketMessage
    )
    from ..persistence.arena import PersistentEvolutionArena, list_experiments as list_stored_experiments
    from ..config import Config
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports not available: {e}")
    IMPORTS_AVAILABLE = False
    
    # Create minimal dummy classes for testing
    class ExperimentConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ExperimentStatus:
        pass
    
    class GenerationStats:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ExperimentListItem:
        pass
    
    class WebSocketMessage:
        pass

logger = logging.getLogger(__name__)
router = APIRouter()

# Global state for managing experiments
websocket_connections: List[WebSocket] = []


class RealExperimentManager:
    """Real experiment manager using PersistentEvolutionArena."""
    
    def __init__(self, base_path: str = "experiments"):
        self.base_path = base_path
        self.active_arenas: Dict[str, Dict[str, Any]] = {}
        self.demo_mode = True  # Start with demo mode, can be switched
        self.generation_callbacks: Dict[str, Any] = {}
        
    def enable_real_mode(self):
        """Enable real experiment execution mode."""
        if IMPORTS_AVAILABLE:
            self.demo_mode = False
            logger.info("Switched to real experiment execution mode")
        else:
            logger.warning("Cannot enable real mode - dependencies not available")
            
    def _convert_web_config_to_arena_config(self, web_config: ExperimentConfig):
        """Convert web experiment config to arena config."""
        if not IMPORTS_AVAILABLE:
            return None
            
        try:
            from ..config import Config
            config = Config()
            
            # Map web config to internal config
            config.evolution.population_size = getattr(web_config, 'population_size', 50)
            config.evolution.generations = getattr(web_config, 'generations', 100)
            config.evolution.mutation_prob = getattr(web_config, 'mutation_rate', 0.1)
            config.evolution.crossover_alpha = getattr(web_config, 'crossover_rate', 0.8)
            config.evolution.elite_size = getattr(web_config, 'elite_size', 5)
            
            # Game settings with defaults
            config.game.board_size = getattr(web_config, 'grid_size', 15)
            config.game.observation_radius = getattr(web_config, 'observation_radius', 3)
            config.game.max_steps = getattr(web_config, 'max_steps', 500)
            
            return config
        except Exception as e:
            logger.error(f"Failed to convert config: {e}")
            return None

    async def start_experiment(self, config: ExperimentConfig) -> str:
        """Start a new experiment."""
        if self.demo_mode:
            return await self._start_demo_experiment(config)
        else:
            return await self._start_real_experiment(config)
    
    async def _start_demo_experiment(self, config: ExperimentConfig) -> str:
        """Start a demo experiment with simulated progression."""
        experiment_id = str(uuid.uuid4())
        experiment_data = {
            'id': experiment_id,
            'config': config,
            'status': 'running',
            'start_time': datetime.now(),
            'generation': 0,
            'target_generations': getattr(config, 'generations', 100),
            'best_fitness': 0.0,
            'avg_fitness': 0.0,
            'diversity': 1.0,
            'mode': 'demo'
        }
        self.active_arenas[experiment_id] = experiment_data
        
        # Start demo progression
        self._start_demo_progression(experiment_id)
        logger.info(f"Started demo experiment {experiment_id}")
        return experiment_id
    
    async def _start_real_experiment(self, config: ExperimentConfig) -> str:
        """Start a real experiment using PersistentEvolutionArena."""
        try:
            arena_config = self._convert_web_config_to_arena_config(config)
            if not arena_config:
                raise HTTPException(status_code=500, detail="Failed to create arena configuration")
            
            # Create persistent arena
            arena = PersistentEvolutionArena(
                config=arena_config,
                base_path=self.base_path,
                auto_checkpoint=True,
                checkpoint_frequency=10
            )
            
            experiment_id = arena.experiment_id
            
            # Set up real-time callbacks
            self.generation_callbacks[experiment_id] = self._create_generation_callback(experiment_id)
            arena.add_generation_callback(self.generation_callbacks[experiment_id])
            
            # Create experiment tracking entry
            experiment_data = {
                'id': experiment_id,
                'arena': arena,
                'config': config,
                'status': 'starting',
                'start_time': datetime.now(),
                'thread': None,
                'target_generations': getattr(config, 'generations', 100),
                'generation': 0,
                'best_fitness': 0.0,
                'avg_fitness': 0.0,
                'diversity': 1.0,
                'mode': 'real'
            }
            
            self.active_arenas[experiment_id] = experiment_data
            
            # Start evolution in background thread
            evolution_thread = Thread(
                target=self._run_evolution,
                args=(experiment_id, getattr(config, 'generations', 100)),
                daemon=True
            )
            experiment_data['thread'] = evolution_thread
            experiment_data['status'] = 'running'
            evolution_thread.start()
            
            logger.info(f"Started real experiment {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to start real experiment: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start experiment: {e}")
    
    def _start_demo_progression(self, experiment_id: str):
        """Start demo experiment progression simulation."""
        def demo_progress():
            import time
            import random
            
            while experiment_id in self.active_arenas:
                exp_data = self.active_arenas[experiment_id]
                if exp_data['status'] != 'running':
                    break
                
                # Simulate generation progression
                current_gen = exp_data['generation']
                target_gen = exp_data['target_generations']
                
                if current_gen >= target_gen:
                    exp_data['status'] = 'completed'
                    break
                
                # Update generation stats with realistic progression
                progress = current_gen / target_gen
                exp_data['generation'] = current_gen + 1
                
                # Simulate fitness improvement over generations
                base_fitness = 10.0 + (progress * 80.0)  # 10-90 range
                exp_data['best_fitness'] = base_fitness + random.uniform(0, 10)
                exp_data['avg_fitness'] = base_fitness * 0.7 + random.uniform(-5, 5)
                exp_data['diversity'] = max(0.1, 1.0 - (progress * 0.7) + random.uniform(-0.1, 0.1))
                
                # Broadcast update
                try:
                    asyncio.run(self._broadcast_generation_update(experiment_id, exp_data))
                except Exception as e:
                    logger.error(f"Failed to broadcast update: {e}")
                
                # Wait before next generation (simulate evolution time)
                time.sleep(1.5)  # Faster for demo
            
            if experiment_id in self.active_arenas:
                self.active_arenas[experiment_id]['status'] = 'completed'
                try:
                    asyncio.run(self._broadcast_experiment_completion(experiment_id))
                except Exception as e:
                    logger.error(f"Failed to broadcast completion: {e}")
        
        Thread(target=demo_progress, daemon=True).start()
    
    def _run_evolution(self, experiment_id: str, generations: int):
        """Run evolution in background thread."""
        try:
            experiment = self.active_arenas[experiment_id]
            arena = experiment['arena']
            
            logger.info(f"Running evolution for experiment {experiment_id}")
            results = arena.evolve(generations=generations, verbose=True)
            
            experiment['status'] = 'completed'
            experiment['results'] = results
            
            # Broadcast completion
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._broadcast_experiment_completion(experiment_id))
            loop.close()
            
        except Exception as e:
            logger.error(f"Evolution failed for experiment {experiment_id}: {e}")
            if experiment_id in self.active_arenas:
                self.active_arenas[experiment_id]['status'] = 'failed'
                self.active_arenas[experiment_id]['error'] = str(e)
                
                # Broadcast failure
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self._broadcast_experiment_failure(experiment_id, str(e)))
                    loop.close()
                except Exception as broadcast_error:
                    logger.error(f"Failed to broadcast failure: {broadcast_error}")
    
    def _create_generation_callback(self, experiment_id: str):
        """Create a generation callback for real-time updates."""
        def on_generation_complete(arena):
            try:
                if experiment_id not in self.active_arenas:
                    return
                
                experiment = self.active_arenas[experiment_id]
                
                # Update experiment data with real arena stats
                experiment['generation'] = arena.generation
                
                if arena.fitness_history:
                    latest_fitness = arena.fitness_history[-1]
                    experiment['best_fitness'] = max(latest_fitness)
                    experiment['avg_fitness'] = sum(latest_fitness) / len(latest_fitness)
                
                if arena.diversity_history:
                    experiment['diversity'] = arena.diversity_history[-1]
                
                # Broadcast update
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._broadcast_generation_update(experiment_id, experiment))
                loop.close()
                
            except Exception as e:
                logger.error(f"Generation callback error for {experiment_id}: {e}")
        
        return on_generation_complete
    
    async def _broadcast_generation_update(self, experiment_id: str, experiment_data: Dict[str, Any]):
        """Broadcast generation update to WebSocket clients."""
        if not websocket_connections:
            return
            
        message = {
            "type": "generation_update",
            "experiment_id": experiment_id,
            "generation": experiment_data.get('generation', 0),
            "target_generations": experiment_data.get('target_generations', 100),
            "best_fitness": round(experiment_data.get('best_fitness', 0.0), 2),
            "avg_fitness": round(experiment_data.get('avg_fitness', 0.0), 2),
            "diversity": round(experiment_data.get('diversity', 1.0), 3),
            "status": experiment_data.get('status', 'running'),
            "mode": experiment_data.get('mode', 'demo'),
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected = []
        for websocket in websocket_connections[:]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            if ws in websocket_connections:
                websocket_connections.remove(ws)
    
    async def _broadcast_experiment_completion(self, experiment_id: str):
        """Broadcast experiment completion."""
        exp_data = self.active_arenas.get(experiment_id, {})
        message = {
            "type": "experiment_complete",
            "experiment_id": experiment_id,
            "message": f"Experiment {experiment_id} completed successfully!",
            "final_generation": exp_data.get('generation', 0),
            "best_fitness": round(exp_data.get('best_fitness', 0.0), 2),
            "mode": exp_data.get('mode', 'demo'),
            "timestamp": datetime.now().isoformat()
        }
        
        for websocket in websocket_connections[:]:
            try:
                await websocket.send_json(message)
            except Exception:
                pass
    
    async def _broadcast_experiment_failure(self, experiment_id: str, error: str):
        """Broadcast experiment failure."""
        message = {
            "type": "experiment_error",
            "experiment_id": experiment_id,
            "error": error,
            "message": f"Experiment {experiment_id} failed: {error}",
            "timestamp": datetime.now().isoformat()
        }
        
        for websocket in websocket_connections[:]:
            try:
                await websocket.send_json(message)
            except Exception:
                pass

    async def get_experiments(self) -> List[Dict[str, Any]]:
        """Get list of all experiments."""
        experiments = []
        for exp_id, exp_data in self.active_arenas.items():
            experiments.append({
                'id': exp_id,
                'name': getattr(exp_data.get('config'), 'name', f'Experiment {exp_id[:8]}'),
                'status': exp_data.get('status', 'unknown'),
                'start_time': exp_data.get('start_time'),
                'generation': exp_data.get('generation', 0),
                'target_generations': exp_data.get('target_generations', 100),
                'best_fitness': round(exp_data.get('best_fitness', 0.0), 2),
                'avg_fitness': round(exp_data.get('avg_fitness', 0.0), 2),
                'diversity': round(exp_data.get('diversity', 1.0), 3),
                'mode': exp_data.get('mode', 'demo')
            })
        return experiments
    
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get specific experiment details."""
        if experiment_id in self.active_arenas:
            return self.active_arenas[experiment_id]
        return None
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        if experiment_id in self.active_arenas:
            self.active_arenas[experiment_id]['status'] = 'stopped'
            # Note: In real implementation, would need to properly stop the evolution thread
            logger.info(f"Stopped experiment {experiment_id}")
            return True
        return False


# Include analytics and visualization routers if available
if ANALYTICS_AVAILABLE:
    try:
        router.include_router(analytics_router)
        router.include_router(viz_router)
        logger.info("Analytics and visualization modules loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load analytics modules: {e}")

# Create global experiment manager
experiment_manager = RealExperimentManager()

# Enable real mode if imports are available
if IMPORTS_AVAILABLE:
    experiment_manager.enable_real_mode()


# REST API Endpoints

@router.get("/")
async def dashboard():
    """Serve the enhanced dashboard HTML."""
    mode_indicator = "REAL" if not experiment_manager.demo_mode else "DEMO"
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Genetic Coil Arena - Dashboard ({mode_indicator} Mode)</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
            .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
            .header {{ text-align: center; margin-bottom: 40px; color: white; }}
            .mode-badge {{ background: {'#28a745' if not experiment_manager.demo_mode else '#ffc107'}; color: {'white' if not experiment_manager.demo_mode else 'black'}; padding: 5px 15px; border-radius: 20px; font-size: 12px; font-weight: bold; margin-left: 10px; }}
            .card {{ background: white; border-radius: 12px; padding: 25px; margin: 20px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); }}
            .card h2 {{ margin-top: 0; color: #333; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px; }}
            .btn {{ padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; margin: 5px; font-weight: 600; transition: all 0.3s; }}
            .btn-primary {{ background: linear-gradient(45deg, #667eea, #764ba2); color: white; }}
            .btn-success {{ background: linear-gradient(45deg, #11998e, #38ef7d); color: white; }}
            .btn-danger {{ background: linear-gradient(45deg, #ff6b6b, #ee5a24); color: white; }}
            .btn:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }}
            .status {{ padding: 6px 12px; border-radius: 20px; font-size: 11px; font-weight: bold; text-transform: uppercase; }}
            .status.running {{ background: #28a745; color: white; animation: pulse 2s infinite; }}
            .status.completed {{ background: #6c757d; color: white; }}
            .status.failed {{ background: #dc3545; color: white; }}
            .status.stopped {{ background: #ffc107; color: black; }}
            @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} }}
            .form-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
            .form-group {{ margin-bottom: 20px; }}
            .form-group label {{ display: block; margin-bottom: 8px; font-weight: 600; color: #555; }}
            .form-group input, .form-group select {{ width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 6px; font-size: 14px; transition: border-color 0.3s; }}
            .form-group input:focus {{ border-color: #667eea; outline: none; }}
            .experiment-item {{ border-left: 4px solid #667eea; padding: 20px; margin: 15px 0; background: #f8f9fa; border-radius: 0 8px 8px 0; }}
            .experiment-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
            .experiment-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin: 15px 0; }}
            .stat-box {{ text-align: center; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .stat-value {{ font-size: 28px; font-weight: bold; color: #667eea; }}
            .stat-label {{ font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px; }}
            .progress {{ background: #e9ecef; border-radius: 20px; overflow: hidden; height: 8px; margin: 15px 0; }}
            .progress-bar {{ background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; transition: width 0.5s ease; }}
            .live-updates {{ max-height: 300px; overflow-y: auto; background: #f8f9fa; border-radius: 8px; padding: 15px; }}
            .update-item {{ border-left: 3px solid #667eea; padding: 10px 15px; margin: 8px 0; background: white; border-radius: 0 6px 6px 0; }}
            .connection-status {{ padding: 10px 15px; border-radius: 6px; margin-bottom: 15px; font-weight: 600; }}
            .connected {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
            .disconnected {{ background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
            .mode-indicator {{ background: {'rgba(40, 167, 69, 0.1)' if not experiment_manager.demo_mode else 'rgba(255, 193, 7, 0.1)'}; border: 1px solid {'#28a745' if not experiment_manager.demo_mode else '#ffc107'}; color: {'#155724' if not experiment_manager.demo_mode else '#856404'}; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧬 Genetic Coil Arena</h1>
                <p>Evolutionary Snake AI Dashboard - Phase 3B: Real Experiment Execution</p>
                <span class="mode-badge">{mode_indicator} MODE</span>
            </div>
            
            <div class="mode-indicator">
                <strong>🎯 Current Mode: {mode_indicator}</strong><br>
                {'✅ Real experiment execution with PersistentEvolutionArena integration' if not experiment_manager.demo_mode else '🔧 Demo mode - simulated evolution for testing (switch to real mode when ready)'}
            </div>
            
            <div class="card">
                <h2>🚀 Start New Experiment</h2>
                <form id="experiment-form">
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="name">Experiment Name:</label>
                            <input type="text" id="name" name="name" required placeholder="Advanced Evolution Test" value="Evolution Run {{new Date().getHours()}}:{{new Date().getMinutes()}}">
                        </div>
                        <div class="form-group">
                            <label for="population_size">Population Size:</label>
                            <input type="number" id="population_size" name="population_size" value="50" min="10" max="500">
                        </div>
                        <div class="form-group">
                            <label for="generations">Generations:</label>
                            <input type="number" id="generations" name="generations" value="100" min="1" max="1000">
                        </div>
                        <div class="form-group">
                            <label for="mutation_rate">Mutation Rate:</label>
                            <input type="number" id="mutation_rate" name="mutation_rate" value="0.1" min="0" max="1" step="0.01">
                        </div>
                        <div class="form-group">
                            <label for="crossover_rate">Crossover Rate:</label>
                            <input type="number" id="crossover_rate" name="crossover_rate" value="0.8" min="0" max="1" step="0.01">
                        </div>
                        <div class="form-group">
                            <label for="elite_size">Elite Size:</label>
                            <input type="number" id="elite_size" name="elite_size" value="5" min="0" max="50">
                        </div>
                    </div>
                    <button type="submit" class="btn btn-primary">🚀 Start Evolution</button>
                </form>
            </div>
            
            <div class="card">
                <h2>📊 Active Experiments</h2>
                <div id="experiments-list">
                    <p>Loading experiments...</p>
                </div>
                <button class="btn btn-success" onclick="loadExperiments()">🔄 Refresh</button>
            </div>
            
            <div class="card">
                <h2>📈 Real-time Updates</h2>
                <div id="connection-status" class="connection-status">Connecting to WebSocket...</div>
                <div class="live-updates" id="live-updates">
                    <div class="update-item">
                        <strong>System:</strong> Dashboard initialized, waiting for experiment updates...
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Enhanced WebSocket and UI management
            let ws = null;
            let experiments = {{}};
            
            function connectWebSocket() {{
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${{protocol}}//${{window.location.host}}/ws`;
                
                try {{
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function() {{
                        document.getElementById('connection-status').innerHTML = 
                            '<strong>✅ Connected to real-time updates</strong>';
                        document.getElementById('connection-status').className = 'connection-status connected';
                    }};
                    
                    ws.onmessage = function(event) {{
                        const data = JSON.parse(event.data);
                        handleWebSocketMessage(data);
                    }};
                    
                    ws.onclose = function() {{
                        document.getElementById('connection-status').innerHTML = 
                            '<strong>❌ Disconnected from real-time updates</strong> - Reconnecting...';
                        document.getElementById('connection-status').className = 'connection-status disconnected';
                        setTimeout(connectWebSocket, 5000);
                    }};
                }} catch (error) {{
                    console.log('WebSocket connection failed:', error);
                    document.getElementById('connection-status').innerHTML = 
                        '<strong>⚠️ WebSocket not available</strong> - Running in offline mode';
                    document.getElementById('connection-status').className = 'connection-status disconnected';
                }}
            }}
            
            function handleWebSocketMessage(data) {{
                const updatesDiv = document.getElementById('live-updates');
                const timestamp = new Date().toLocaleTimeString();
                
                let updateContent = '';
                if (data.type === 'generation_update') {{
                    updateContent = `
                        <strong>Generation ${{data.generation}}/${{data.target_generations}}</strong> - 
                        Experiment ${{data.experiment_id.substring(0, 8)}}...<br>
                        🏆 Best: ${{data.best_fitness}} | 📊 Avg: ${{data.avg_fitness}} | 🔀 Diversity: ${{data.diversity}}
                        <span style="float: right; font-size: 10px;">[${{data.mode.toUpperCase()}}]</span>
                    `;
                    
                    // Update experiment in real-time
                    experiments[data.experiment_id] = data;
                    updateExperimentDisplay(data.experiment_id, data);
                }} else if (data.type === 'experiment_complete') {{
                    updateContent = `
                        <strong style="color: #28a745;">🎉 EXPERIMENT COMPLETED!</strong><br>
                        Experiment ${{data.experiment_id.substring(0, 8)}}... finished at generation ${{data.final_generation}}
                        <br>Final best fitness: ${{data.best_fitness}}
                    `;
                }} else if (data.type === 'experiment_error') {{
                    updateContent = `
                        <strong style="color: #dc3545;">❌ EXPERIMENT FAILED!</strong><br>
                        Experiment ${{data.experiment_id.substring(0, 8)}}...<br>
                        Error: ${{data.error}}
                    `;
                }} else {{
                    updateContent = data.message || JSON.stringify(data);
                }}
                
                const updateElement = document.createElement('div');
                updateElement.className = 'update-item';
                updateElement.innerHTML = `
                    <div style="display: flex; justify-content: space-between;">
                        <div>${{updateContent}}</div>
                        <small>${{timestamp}}</small>
                    </div>
                `;
                
                updatesDiv.insertBefore(updateElement, updatesDiv.firstChild);
                
                // Keep only last 15 updates
                while (updatesDiv.children.length > 15) {{
                    updatesDiv.removeChild(updatesDiv.lastChild);
                }}
                
                // Auto-refresh experiments list
                loadExperiments();
            }}
            
            function updateExperimentDisplay(experimentId, data) {{
                // Update the specific experiment in the display
                const expElement = document.querySelector(`[data-experiment-id="${{experimentId}}"]`);
                if (expElement) {{
                    const progressBar = expElement.querySelector('.progress-bar');
                    const statusElement = expElement.querySelector('.status');
                    const statsElements = expElement.querySelectorAll('.stat-value');
                    
                    if (progressBar) {{
                        const progress = (data.generation / data.target_generations) * 100;
                        progressBar.style.width = `${{progress}}%`;
                    }}
                    
                    if (statusElement) {{
                        statusElement.textContent = data.status.toUpperCase();
                        statusElement.className = `status ${{data.status}}`;
                    }}
                    
                    if (statsElements.length >= 3) {{
                        statsElements[0].textContent = data.generation;
                        statsElements[1].textContent = data.best_fitness;
                        statsElements[2].textContent = data.diversity;
                    }}
                }}
            }}
            
            // Load experiments with enhanced display
            async function loadExperiments() {{
                try {{
                    const response = await fetch('/experiments');
                    const data = await response.json();
                    
                    const experimentsDiv = document.getElementById('experiments-list');
                    if (data.experiments.length === 0) {{
                        experimentsDiv.innerHTML = '<p style="text-align: center; color: #666; font-style: italic;">No experiments found. Start one above! 🚀</p>';
                        return;
                    }}
                    
                    experimentsDiv.innerHTML = data.experiments.map(exp => {{
                        const progress = (exp.generation / exp.target_generations * 100);
                        const timeStr = exp.start_time ? new Date(exp.start_time).toLocaleString() : 'Unknown';
                        
                        return `
                            <div class="experiment-item" data-experiment-id="${{exp.id}}">
                                <div class="experiment-header">
                                    <div>
                                        <h4>${{exp.name}} <span class="status ${{exp.status}}">${{exp.status.toUpperCase()}}</span></h4>
                                        <small>ID: ${{exp.id}} | Started: ${{timeStr}} | Mode: <strong>${{exp.mode.toUpperCase()}}</strong></small>
                                    </div>
                                    <div>
                                        ${{exp.status === 'running' ? 
                                            `<button class="btn btn-danger" onclick="stopExperiment('${{exp.id}}')">⏹️ Stop</button>` :
                                            ''
                                        }}
                                    </div>
                                </div>
                                <div class="experiment-stats">
                                    <div class="stat-box">
                                        <div class="stat-value">${{exp.generation}}</div>
                                        <div class="stat-label">Generation</div>
                                    </div>
                                    <div class="stat-box">
                                        <div class="stat-value">${{exp.best_fitness}}</div>
                                        <div class="stat-label">Best Fitness</div>
                                    </div>
                                    <div class="stat-box">
                                        <div class="stat-value">${{exp.diversity}}</div>
                                        <div class="stat-label">Diversity</div>
                                    </div>
                                    <div class="stat-box">
                                        <div class="stat-value">${{exp.target_generations}}</div>
                                        <div class="stat-label">Target Gen</div>
                                    </div>
                                </div>
                                <div class="progress">
                                    <div class="progress-bar" style="width: ${{progress}}%"></div>
                                </div>
                                <small>Progress: ${{Math.round(progress)}}% | Avg Fitness: ${{exp.avg_fitness}}</small>
                            </div>
                        `;
                    }}).join('');
                    
                }} catch (error) {{
                    console.error('Failed to load experiments:', error);
                    document.getElementById('experiments-list').innerHTML = 
                        '<p style="color: red;">❌ Failed to load experiments. Check console for details.</p>';
                }}
            }}
            
            // Enhanced experiment creation
            document.getElementById('experiment-form').addEventListener('submit', async function(e) {{
                e.preventDefault();
                
                const submitBtn = e.target.querySelector('button[type="submit"]');
                const originalText = submitBtn.textContent;
                submitBtn.textContent = '🔄 Starting...';
                submitBtn.disabled = true;
                
                const formData = new FormData(e.target);
                const config = {{
                    name: formData.get('name'),
                    population_size: parseInt(formData.get('population_size')),
                    generations: parseInt(formData.get('generations')),
                    mutation_rate: parseFloat(formData.get('mutation_rate')),
                    crossover_rate: parseFloat(formData.get('crossover_rate')),
                    elite_size: parseInt(formData.get('elite_size'))
                }};
                
                try {{
                    const response = await fetch('/experiments', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(config)
                    }});
                    
                    if (response.ok) {{
                        const result = await response.json();
                        alert(`🎉 Experiment started successfully!\\n\\nExperiment ID: ${{result.experiment_id}}\\nMode: {{'{mode_indicator}' if not experiment_manager.demo_mode else 'DEMO'}}\\n\\nWatch the real-time updates below!`);
                        loadExperiments();
                        
                        // Reset form with new default name
                        const now = new Date();
                        formData.set('name', `Evolution Run ${{now.getHours()}}:${{now.getMinutes().toString().padStart(2, '0')}}`);
                        e.target.reset();
                        document.getElementById('name').value = `Evolution Run ${{now.getHours()}}:${{now.getMinutes().toString().padStart(2, '0')}}`;
                    }} else {{
                        const error = await response.json();
                        alert(`❌ Failed to start experiment:\\n\\n${{error.detail}}`);
                    }}
                }} catch (error) {{
                    console.error('Failed to start experiment:', error);
                    alert('❌ Failed to start experiment. Check console for details.');
                }} finally {{
                    submitBtn.textContent = originalText;
                    submitBtn.disabled = false;
                }}
            }});
            
            // Enhanced experiment stopping
            async function stopExperiment(experimentId) {{
                if (!confirm('⚠️ Are you sure you want to stop this experiment?\\n\\nThis action cannot be undone.')) {{
                    return;
                }}
                
                try {{
                    const response = await fetch(`/experiments/${{experimentId}}/stop`, {{
                        method: 'POST'
                    }});
                    
                    if (response.ok) {{
                        alert('✅ Experiment stopped successfully!');
                        loadExperiments();
                    }} else {{
                        const error = await response.json();
                        alert(`❌ Failed to stop experiment:\\n\\n${{error.detail}}`);
                    }}
                }} catch (error) {{
                    console.error('Failed to stop experiment:', error);
                    alert('❌ Failed to stop experiment. Check console for details.');
                }}
            }}
            
            // Initialize dashboard
            document.addEventListener('DOMContentLoaded', function() {{
                connectWebSocket();
                loadExperiments();
                
                // Auto-refresh experiments every 15 seconds
                setInterval(loadExperiments, 15000);
                
                // Set dynamic experiment name
                const now = new Date();
                document.getElementById('name').value = `Evolution Run ${{now.getHours()}}:${{now.getMinutes().toString().padStart(2, '0')}}`;
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, media_type="text/html")


@router.post("/experiments")
async def create_experiment(config: dict):
    """Create and start a new experiment."""
    try:
        exp_config = ExperimentConfig(**config)
        experiment_id = await experiment_manager.start_experiment(exp_config)
        
        mode = "real" if not experiment_manager.demo_mode else "demo"
        return {
            "experiment_id": experiment_id, 
            "status": "started",
            "mode": mode,
            "message": f"Experiment started in {mode} mode"
        }
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
async def list_experiments():
    """List all experiments with enhanced data."""
    experiments = await experiment_manager.get_experiments()
    return {"experiments": experiments}


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment details."""
    experiment = await experiment_manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.post("/experiments/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """Stop an experiment."""
    success = await experiment_manager.stop_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=404, detail="Experiment not found or cannot be stopped")
    return {"status": "stopped", "experiment_id": experiment_id}


@router.get("/system/status")
async def system_status():
    """Get system status and configuration."""
    return {
        "mode": "real" if not experiment_manager.demo_mode else "demo",
        "imports_available": IMPORTS_AVAILABLE,
        "active_experiments": len(experiment_manager.active_arenas),
        "websocket_connections": len(websocket_connections),
        "version": "3B.1"
    }


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint for real-time updates."""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        # Send welcome message with system info
        await websocket.send_json({
            "type": "connection",
            "message": f"Connected to Genetic Coil Arena real-time updates",
            "mode": "real" if not experiment_manager.demo_mode else "demo",
            "version": "Phase 3B",
            "active_experiments": len(experiment_manager.active_arenas),
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive with stats
                await websocket.send_json({
                    "type": "keepalive",
                    "active_experiments": len(experiment_manager.active_arenas),
                    "timestamp": datetime.now().isoformat()
                })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)


# Export router
__all__ = ["router"]
