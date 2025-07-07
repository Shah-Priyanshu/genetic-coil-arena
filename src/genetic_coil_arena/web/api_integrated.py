"""
Integrated FastAPI routes for the web dashboard with persistent experiment management.
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

try:
    from .models import (
        ExperimentConfig, ExperimentStatus, GenerationStats, 
        ExperimentListItem, WebSocketMessage
    )
    from ..persistence.arena import PersistentEvolutionArena, list_experiments as list_stored_experiments
    from ..config import Config
except ImportError as e:
    # Fallback for development
    print(f"Warning: Could not import some components: {e}")
    
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


class IntegratedExperimentManager:
    """Integrated experiment manager using PersistentEvolutionArena."""
    
    def __init__(self, base_path: str = "experiments"):
        self.base_path = base_path
        self.active_arenas: Dict[str, Dict[str, Any]] = {}
        self.demo_mode = False  # Switch to real mode for production
        self.generation_callbacks: Dict[str, Any] = {}
          def _convert_web_config_to_arena_config(self, web_config: ExperimentConfig) -> Optional[Any]:
        """Convert web experiment config to arena config."""
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
        except ImportError:
            # Fallback if config not available
            return None

    async def start_experiment(self, config: ExperimentConfig) -> str:
        """Start a new experiment."""
        if self.demo_mode:
            # Demo mode - create a fake experiment with progression
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
                'diversity': 1.0
            }
            self.active_arenas[experiment_id] = experiment_data
            
            # Start demo progression
            self._start_demo_progression(experiment_id)
            logger.info(f"Started demo experiment {experiment_id}")
            return experiment_id
        else:
            # Real experiment using PersistentEvolutionArena
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
                    'diversity': 1.0
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
                
                # Update generation stats
                exp_data['generation'] = current_gen + 1
                exp_data['best_fitness'] = min(100.0, exp_data['best_fitness'] + random.uniform(0.5, 2.0))
                exp_data['avg_fitness'] = min(exp_data['best_fitness'], exp_data['avg_fitness'] + random.uniform(0.1, 1.0))
                exp_data['diversity'] = max(0.1, 1.0 - (current_gen / target_gen) * 0.7 + random.uniform(-0.1, 0.1))
                
                # Broadcast update
                asyncio.run(self._broadcast_generation_update(experiment_id, exp_data))
                
                # Wait before next generation (simulate evolution time)
                time.sleep(2)
            
            if experiment_id in self.active_arenas:
                self.active_arenas[experiment_id]['status'] = 'completed'
                asyncio.run(self._broadcast_experiment_completion(experiment_id))
        
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
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._broadcast_experiment_failure(experiment_id, str(e)))
                loop.close()
    
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
            "best_fitness": experiment_data.get('best_fitness', 0.0),
            "avg_fitness": experiment_data.get('avg_fitness', 0.0),
            "diversity": experiment_data.get('diversity', 1.0),
            "status": experiment_data.get('status', 'running'),
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
        message = {
            "type": "experiment_complete",
            "experiment_id": experiment_id,
            "message": f"Experiment {experiment_id} completed successfully!",
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
                'name': getattr(exp_data.get('config'), 'name', 'Unknown'),
                'status': exp_data.get('status', 'unknown'),
                'start_time': exp_data.get('start_time'),
                'generation': exp_data.get('generation', 0),
                'target_generations': exp_data.get('target_generations', 100)
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
            return True
        return False


# Create global experiment manager
experiment_manager = IntegratedExperimentManager()


# REST API Endpoints

@router.get("/")
async def dashboard():
    """Serve the main dashboard HTML."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Genetic Coil Arena - Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .card { background: white; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .btn { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
            .btn-primary { background-color: #007bff; color: white; }
            .btn-success { background-color: #28a745; color: white; }
            .btn-danger { background-color: #dc3545; color: white; }
            .btn:hover { opacity: 0.9; }
            .status { padding: 5px 10px; border-radius: 4px; font-size: 12px; font-weight: bold; }
            .status.running { background-color: #28a745; color: white; }
            .status.completed { background-color: #6c757d; color: white; }
            .status.failed { background-color: #dc3545; color: white; }
            .status.stopped { background-color: #ffc107; color: black; }
            .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .form-group { margin-bottom: 15px; }
            .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
            .form-group input, .form-group select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            .experiment-item { border-bottom: 1px solid #eee; padding: 15px 0; }
            .experiment-item:last-child { border-bottom: none; }
            .progress { background-color: #e9ecef; border-radius: 4px; overflow: hidden; height: 20px; margin: 10px 0; }
            .progress-bar { background-color: #007bff; height: 100%; transition: width 0.3s; }
            .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 15px 0; }
            .stat-box { text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 4px; }
            .stat-value { font-size: 24px; font-weight: bold; color: #007bff; }
            .stat-label { font-size: 12px; color: #6c757d; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧬 Genetic Coil Arena</h1>
                <p>Evolutionary Snake AI Dashboard - Phase 3 Integration</p>
            </div>
            
            <div class="card">
                <h2>🚀 Start New Experiment</h2>
                <form id="experiment-form">
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="name">Experiment Name:</label>
                            <input type="text" id="name" name="name" required placeholder="My Evolution Experiment">
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
                    <button type="submit" class="btn btn-primary">🚀 Start Experiment</button>
                </form>
            </div>
            
            <div class="card">
                <h2>📊 Experiments</h2>
                <div id="experiments-list">
                    <p>Loading experiments...</p>
                </div>
                <button class="btn btn-success" onclick="loadExperiments()">🔄 Refresh</button>
            </div>
            
            <div class="card">
                <h2>📈 Real-time Updates</h2>
                <div id="connection-status">Connecting to WebSocket...</div>
                <div id="live-updates"></div>
            </div>
        </div>
        
        <script>
            // WebSocket connection for real-time updates
            let ws = null;
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                try {
                    ws = new WebSocket(wsUrl);
                    
                    ws.onopen = function() {
                        document.getElementById('connection-status').innerHTML = 
                            '<span style="color: green;">✅ Connected to real-time updates</span>';
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        handleWebSocketMessage(data);
                    };
                    
                    ws.onclose = function() {
                        document.getElementById('connection-status').innerHTML = 
                            '<span style="color: red;">❌ Disconnected from real-time updates</span>';
                        // Try to reconnect after 5 seconds
                        setTimeout(connectWebSocket, 5000);
                    };
                } catch (error) {
                    console.log('WebSocket connection failed:', error);
                    document.getElementById('connection-status').innerHTML = 
                        '<span style="color: orange;">⚠️ WebSocket not available (demo mode)</span>';
                }
            }
            
            function handleWebSocketMessage(data) {
                const updatesDiv = document.getElementById('live-updates');
                const timestamp = new Date().toLocaleTimeString();
                updatesDiv.innerHTML = `
                    <div style="border-left: 3px solid #007bff; padding-left: 10px; margin: 5px 0;">
                        <strong>${timestamp}:</strong> ${data.message || JSON.stringify(data)}
                    </div>
                ` + updatesDiv.innerHTML;
                
                // Keep only last 10 updates
                const updates = updatesDiv.children;
                while (updates.length > 10) {
                    updatesDiv.removeChild(updates[updates.length - 1]);
                }
            }
            
            // Load experiments
            async function loadExperiments() {
                try {
                    const response = await fetch('/experiments');
                    const data = await response.json();
                    
                    const experimentsDiv = document.getElementById('experiments-list');
                    if (data.experiments.length === 0) {
                        experimentsDiv.innerHTML = '<p>No experiments found. Start one above!</p>';
                        return;
                    }
                    
                    experimentsDiv.innerHTML = data.experiments.map(exp => `
                        <div class="experiment-item">
                            <div style="display: flex; justify-content: between; align-items: center;">
                                <div style="flex: 1;">
                                    <h4>${exp.name || exp.id}</h4>
                                    <p>ID: ${exp.id}</p>
                                    <span class="status ${exp.status}">${exp.status.toUpperCase()}</span>
                                </div>
                                <div style="flex: 1;">
                                    <div class="stats">
                                        <div class="stat-box">
                                            <div class="stat-value">${exp.generation}</div>
                                            <div class="stat-label">Generation</div>
                                        </div>
                                        <div class="stat-box">
                                            <div class="stat-value">${exp.target_generations}</div>
                                            <div class="stat-label">Target</div>
                                        </div>
                                    </div>
                                    <div class="progress">
                                        <div class="progress-bar" style="width: ${(exp.generation / exp.target_generations * 100)}%"></div>
                                    </div>
                                </div>
                                <div>
                                    ${exp.status === 'running' ? 
                                        `<button class="btn btn-danger" onclick="stopExperiment('${exp.id}')">⏹️ Stop</button>` :
                                        ''
                                    }
                                </div>
                            </div>
                        </div>
                    `).join('');
                    
                } catch (error) {
                    console.error('Failed to load experiments:', error);
                    document.getElementById('experiments-list').innerHTML = 
                        '<p style="color: red;">Failed to load experiments. Check console for details.</p>';
                }
            }
            
            // Start experiment
            document.getElementById('experiment-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const config = {
                    name: formData.get('name'),
                    population_size: parseInt(formData.get('population_size')),
                    generations: parseInt(formData.get('generations')),
                    mutation_rate: parseFloat(formData.get('mutation_rate')),
                    crossover_rate: parseFloat(formData.get('crossover_rate')),
                    elite_size: parseInt(formData.get('elite_size'))
                };
                
                try {
                    const response = await fetch('/experiments', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(config)
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        alert(`Experiment started successfully! ID: ${result.experiment_id}`);
                        loadExperiments();
                        e.target.reset();
                    } else {
                        const error = await response.json();
                        alert(`Failed to start experiment: ${error.detail}`);
                    }
                } catch (error) {
                    console.error('Failed to start experiment:', error);
                    alert('Failed to start experiment. Check console for details.');
                }
            });
            
            // Stop experiment
            async function stopExperiment(experimentId) {
                if (!confirm('Are you sure you want to stop this experiment?')) {
                    return;
                }
                
                try {
                    const response = await fetch(`/experiments/${experimentId}/stop`, {
                        method: 'POST'
                    });
                    
                    if (response.ok) {
                        alert('Experiment stopped successfully!');
                        loadExperiments();
                    } else {
                        const error = await response.json();
                        alert(`Failed to stop experiment: ${error.detail}`);
                    }
                } catch (error) {
                    console.error('Failed to stop experiment:', error);
                    alert('Failed to stop experiment. Check console for details.');
                }
            }
            
            // Initialize
            document.addEventListener('DOMContentLoaded', function() {
                connectWebSocket();
                loadExperiments();
                
                // Auto-refresh experiments every 10 seconds
                setInterval(loadExperiments, 10000);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, media_type="text/html")


@router.post("/experiments")
async def create_experiment(config: dict):
    """Create and start a new experiment."""
    try:
        # Convert dict to ExperimentConfig-like object
        exp_config = ExperimentConfig(**config)
        experiment_id = await experiment_manager.start_experiment(exp_config)
        return {"experiment_id": experiment_id, "status": "started"}
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
async def list_experiments():
    """List all experiments."""
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
    return {"status": "stopped"}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to Genetic Coil Arena real-time updates"
        })
        
        while True:
            # Keep connection alive - in real implementation, this would handle incoming messages
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({
                    "type": "keepalive", 
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
