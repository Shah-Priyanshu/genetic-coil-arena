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

from .models import (
    ExperimentConfig, ExperimentStatus, GenerationStats, 
    ExperimentListItem, WebSocketMessage
)
from ..persistence.arena import PersistentEvolutionArena, list_experiments as list_stored_experiments
from ..config import Config

logger = logging.getLogger(__name__)
router = APIRouter()

# Global state for managing experiments
websocket_connections: List[WebSocket] = []


class IntegratedExperimentManager:
    """Integrated experiment manager using PersistentEvolutionArena."""
    
    def __init__(self, base_path: str = "experiments"):
        self.base_path = base_path
        self.active_arenas: Dict[str, Dict[str, Any]] = {}
        
    def _convert_web_config_to_arena_config(self, web_config: ExperimentConfig) -> Config:
        """Convert web experiment config to arena config."""
        # Create a basic config object
        config = Config()
        
        # Map web config to internal config
        config.evolution.population_size = web_config.population_size
        config.evolution.generations = web_config.generations
        
        # Map to existing config fields
        config.evolution.mutation_prob = web_config.mutation_rate  # mutation_rate -> mutation_prob
        config.evolution.crossover_alpha = web_config.crossover_rate  # crossover_rate -> crossover_alpha  
        config.evolution.elite_size = web_config.elite_size
        
        # Game settings
        config.game.board_size = getattr(web_config, 'grid_size', 15)
        config.game.observation_radius = getattr(web_config, 'observation_radius', 3)
        config.game.max_steps = getattr(web_config, 'max_steps', 500)
        
        return config
    
    async def start_experiment(self, config: ExperimentConfig) -> str:
        """Start a new experiment using PersistentEvolutionArena."""
        try:
            # Convert web config to arena config
            arena_config = self._convert_web_config_to_arena_config(config)
            
            # Create persistent arena
            arena = PersistentEvolutionArena(
                config=arena_config,
                base_path=self.base_path,
                auto_checkpoint=True,
                checkpoint_frequency=10
            )
            
            # Set up real-time callbacks for web updates
            arena.add_generation_callback(self._on_generation_complete)
            
            # Create experiment tracking entry
            experiment_entry = {
                'id': arena.experiment_id,
                'arena': arena,
                'config': config,
                'status': 'starting',
                'start_time': datetime.now(),
                'thread': None,
                'target_generations': config.generations
            }
            
            self.active_arenas[arena.experiment_id] = experiment_entry
            
            # Start evolution in background thread
            evolution_thread = Thread(
                target=self._run_evolution,
                args=(arena.experiment_id, config.generations),
                daemon=True
            )
            experiment_entry['thread'] = evolution_thread
            experiment_entry['status'] = 'running'
            evolution_thread.start()
            
            logger.info(f"Started experiment {arena.experiment_id}")
            return arena.experiment_id
            
        except Exception as e:
            logger.error(f"Failed to start experiment: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start experiment: {e}")
    
    def _run_evolution(self, experiment_id: str, generations: int):
        """Run evolution in background thread."""
        try:
            experiment = self.active_arenas[experiment_id]
            arena = experiment['arena']
            
            logger.info(f"Running evolution for experiment {experiment_id}")
            results = arena.evolve(generations=generations, verbose=True)
            
            experiment['status'] = 'completed'
            experiment['results'] = results
            
            # Broadcast completion using asyncio from thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._broadcast_experiment_completion(experiment_id))
            loop.close()
            
        except Exception as e:
            logger.error(f"Evolution failed for experiment {experiment_id}: {e}")
            if experiment_id in self.active_arenas:
                self.active_arenas[experiment_id]['status'] = 'failed'
                self.active_arenas[experiment_id]['error'] = str(e)
    
    def _on_generation_complete(self, arena: PersistentEvolutionArena):
        """Callback when a generation completes - broadcast to WebSocket clients."""
        try:
            # Find the experiment entry
            experiment_id = arena.experiment_id
            if experiment_id not in self.active_arenas:
                return
            
            # Get latest fitness data
            if arena.fitness_history:
                latest_fitness = arena.fitness_history[-1]
                best_fitness = max(latest_fitness)
                avg_fitness = sum(latest_fitness) / len(latest_fitness)
                std_fitness = float(np.std(latest_fitness)) if len(latest_fitness) > 1 else 0.0
            else:
                best_fitness = avg_fitness = std_fitness = 0.0
            
            # Get diversity score
            diversity_score = arena.diversity_history[-1] if arena.diversity_history else 0.0
            
            # Create generation stats
            stats = GenerationStats(
                generation=arena.generation,
                best_fitness=best_fitness,
                average_fitness=avg_fitness,
                std_fitness=std_fitness,
                diversity_score=diversity_score,
                timestamp=datetime.now()
            )
            
            # Schedule WebSocket broadcast using asyncio from thread
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._broadcast_generation_update(experiment_id, stats))
                loop.close()
            except Exception as broadcast_error:
                logger.warning(f"Failed to broadcast from thread: {broadcast_error}")
            
        except Exception as e:
            logger.warning(f"Failed to process generation update: {e}")
    
    async def _broadcast_experiment_completion(self, experiment_id: str):
        """Broadcast experiment completion to WebSocket clients."""
        try:
            experiment = self.active_arenas[experiment_id]
            message = WebSocketMessage(
                type='experiment_completed',
                data={
                    'experiment_id': experiment_id,
                    'status': experiment['status'],
                    'total_generations': experiment['arena'].generation
                },                timestamp=datetime.now()
            )
            await self._broadcast_message(message)
        except Exception as e:
            logger.warning(f"Failed to broadcast experiment completion: {e}")
    
    async def _broadcast_generation_update(self, experiment_id: str, stats: GenerationStats):
        """Broadcast generation update to WebSocket clients."""
        try:
            message = WebSocketMessage(
                type='generation_update',
                data={
                    'experiment_id': experiment_id,
                    'stats': stats.model_dump()
                },
                timestamp=datetime.now()
            )
            await self._broadcast_message(message)
        except Exception as e:
            logger.warning(f"Failed to broadcast generation update: {e}")
    
    async def _broadcast_message(self, message: WebSocketMessage):
        """Broadcast message to all WebSocket connections."""
        if websocket_connections:
            message_str = message.model_dump_json()
            disconnected = []
            for ws in websocket_connections:
                try:
                    await ws.send_text(message_str)
                except:
                    disconnected.append(ws)
            
            # Remove disconnected clients
            for ws in disconnected:
                websocket_connections.remove(ws)
    
    def get_experiment_status(self, experiment_id: str) -> Optional[ExperimentStatus]:
        """Get status of an active experiment."""
        if experiment_id in self.active_arenas:
            exp = self.active_arenas[experiment_id]
            arena = exp['arena']
            
            # Get current fitness data
            best_fitness = 0.0
            avg_fitness = 0.0
            if arena.fitness_history:
                latest_fitness = arena.fitness_history[-1]
                best_fitness = max(latest_fitness)
                avg_fitness = sum(latest_fitness) / len(latest_fitness)
            
            return ExperimentStatus(
                id=experiment_id,
                name=exp['config'].name,
                status=exp['status'],
                current_generation=arena.generation,
                total_generations=exp['target_generations'],
                best_fitness=best_fitness,
                average_fitness=avg_fitness,
                start_time=exp['start_time'],
                last_update=datetime.now(),
                elapsed_time=(datetime.now() - exp['start_time']).total_seconds()
            )
          # If not active, try to get from persistent storage
        try:
            experiments = list_stored_experiments(self.base_path)
            for exp_info in experiments:
                if exp_info['experiment_id'] == experiment_id:
                    return ExperimentStatus(
                        id=experiment_id,
                        name=exp_info.get('description', 'Unknown'),
                        status='completed',
                        current_generation=exp_info['total_generations'],
                        total_generations=exp_info['total_generations'],
                        best_fitness=exp_info.get('best_fitness', 0.0),
                        average_fitness=0.0,  # Not stored in metadata
                        start_time=datetime.fromisoformat(exp_info['created_at']),
                        last_update=datetime.fromisoformat(exp_info['created_at']),
                        elapsed_time=0.0
                    )
        except Exception as e:
            logger.warning(f"Failed to get experiment status from storage: {e}")
        
        return None
    
    def list_experiments(self) -> List[ExperimentListItem]:
        """List all experiments (active and completed)."""
        experiments = []
        
        # Add active experiments
        for exp_id, exp in self.active_arenas.items():
            arena = exp['arena']
            best_fitness = 0.0
            if arena.fitness_history:
                best_fitness = max(arena.fitness_history[-1])
            
            experiments.append(ExperimentListItem(
                id=exp_id,
                name=exp['config'].name,
                status=exp['status'],
                current_generation=arena.generation,
                total_generations=exp['target_generations'],
                best_fitness=best_fitness,
                start_time=exp['start_time']
            ))
          # Add completed experiments from storage
        try:
            stored_experiments = list_stored_experiments(self.base_path)
            for exp_info in stored_experiments:
                # Skip if already in active experiments
                if exp_info['experiment_id'] not in self.active_arenas:
                    experiments.append(ExperimentListItem(
                        id=exp_info['experiment_id'],
                        name=exp_info.get('description', 'Stored Experiment'),
                        status='completed',
                        current_generation=exp_info['total_generations'],
                        total_generations=exp_info['total_generations'],
                        best_fitness=exp_info.get('best_fitness', 0.0),
                        start_time=datetime.fromisoformat(exp_info['created_at'])
                    ))
        except Exception as e:
            logger.warning(f"Failed to list stored experiments: {e}")
        
        return experiments
    
    async def pause_experiment(self, experiment_id: str) -> bool:
        """Pause an experiment (simplified - would need more sophisticated pause/resume)."""
        if experiment_id in self.active_arenas:
            # For now, just mark as paused - real pause/resume would need more work
            self.active_arenas[experiment_id]['status'] = 'paused'
            logger.info(f"Marked experiment {experiment_id} as paused")
            return True
        return False
    
    async def resume_experiment(self, experiment_id: str) -> bool:
        """Resume a paused experiment."""
        if experiment_id in self.active_arenas:
            self.active_arenas[experiment_id]['status'] = 'running'
            logger.info(f"Marked experiment {experiment_id} as resumed")
            return True
        
        # Try to resume from persistent storage
        try:
            stored_experiments = list_stored_experiments(self.base_path)
            for exp_info in stored_experiments:
                if exp_info['experiment_id'] == experiment_id:
                    # Create new arena from existing experiment
                    arena = PersistentEvolutionArena(
                        config=Config(),  # Will be loaded from checkpoint
                        experiment_id=experiment_id,
                        base_path=self.base_path,
                        auto_checkpoint=True
                    )
                    
                    # Create basic experiment config for resumed experiments
                    resumed_config = ExperimentConfig(
                        name=exp_info.get('description', 'Resumed'),
                        population_size=50,  # Default values for resume
                        generations=50,
                        mutation_rate=0.1,
                        crossover_rate=0.8,
                        elite_size=5,
                        grid_size=15,
                        max_steps=500
                    )
                    
                    # Create experiment entry
                    experiment_entry = {
                        'id': experiment_id,
                        'arena': arena,
                        'config': resumed_config,
                        'status': 'running',
                        'start_time': datetime.now(),
                        'thread': None,
                        'target_generations': exp_info['total_generations'] + 50  # Continue for 50 more
                    }
                    
                    self.active_arenas[experiment_id] = experiment_entry
                    
                    # Start evolution thread
                    evolution_thread = Thread(
                        target=self._run_evolution,
                        args=(experiment_id, 50),  # Continue for 50 more generations
                        daemon=True
                    )
                    experiment_entry['thread'] = evolution_thread
                    evolution_thread.start()
                    
                    logger.info(f"Resumed experiment {experiment_id} from persistent storage")
                    return True
        except Exception as e:
            logger.error(f"Failed to resume experiment {experiment_id}: {e}")
        
        return False
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        if experiment_id in self.active_arenas:
            experiment = self.active_arenas[experiment_id]
            experiment['status'] = 'stopped'
            
            # Note: In a real implementation, you'd need to signal the evolution thread to stop
            # For now, we just mark it as stopped
            logger.info(f"Marked experiment {experiment_id} as stopped")
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
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            
            .header {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
            .header h1 { color: #4f46e5; font-size: 2.5rem; margin-bottom: 10px; }
            .header p { color: #6b7280; font-size: 1.1rem; }
            
            .grid { display: grid; grid-template-columns: 1fr 2fr; gap: 30px; }
            
            .card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                padding: 25px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            
            .card h2 { margin-bottom: 20px; color: #374151; font-size: 1.5rem; }
            
            .form-group { margin-bottom: 20px; }
            .form-group label { display: block; margin-bottom: 8px; color: #374151; font-weight: 500; }
            .form-group input, .form-group select {
                width: 100%;
                padding: 12px;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.2s;
            }
            .form-group input:focus, .form-group select:focus {
                outline: none;
                border-color: #4f46e5;
            }
            
            .btn {
                background: linear-gradient(135deg, #4f46e5, #7c3aed);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 500;
                transition: transform 0.2s, box-shadow 0.2s;
                width: 100%;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
            }
            .btn:active { transform: translateY(0); }
            
            .btn-small {
                padding: 8px 16px;
                font-size: 14px;
                width: auto;
                margin: 0 4px;
            }
            
            .experiment-list { max-height: 400px; overflow-y: auto; }
            .experiment-item {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 10px;
                background: #fafafa;
            }
            .experiment-item h3 { margin-bottom: 5px; color: #374151; }
            .experiment-item p { color: #6b7280; font-size: 14px; margin-bottom: 5px; }
            .experiment-controls { margin-top: 10px; }
            
            .status { 
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 500;
                text-transform: uppercase;
            }
            .status.running { background: #dcfce7; color: #166534; }
            .status.completed { background: #dbeafe; color: #1d4ed8; }
            .status.paused { background: #fef3c7; color: #92400e; }
            .status.failed { background: #fee2e2; color: #dc2626; }
            
            .logs {
                background: #1f2937;
                color: #f9fafb;
                padding: 20px;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                height: 300px;
                overflow-y: auto;
                margin-top: 20px;
            }
            
            .connection-status {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 500;
            }
            .connected { background: #dcfce7; color: #166534; }
            .disconnected { background: #fee2e2; color: #dc2626; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧬 Genetic Coil Arena</h1>
                <p>Real-time Evolution Monitoring & Experiment Management</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h2>🚀 New Experiment</h2>
                    <form id="experimentForm">
                        <div class="form-group">
                            <label for="name">Experiment Name</label>
                            <input type="text" id="name" name="name" value="Real Evolution Test" required>
                        </div>
                        <div class="form-group">
                            <label for="population_size">Population Size</label>
                            <input type="number" id="population_size" name="population_size" value="20" min="10" max="1000">
                        </div>
                        <div class="form-group">
                            <label for="generations">Generations</label>
                            <input type="number" id="generations" name="generations" value="50" min="1" max="10000">
                        </div>
                        <div class="form-group">
                            <label for="mutation_rate">Mutation Rate</label>
                            <input type="number" id="mutation_rate" name="mutation_rate" value="0.1" min="0" max="1" step="0.01">
                        </div>
                        <div class="form-group">
                            <label for="crossover_rate">Crossover Rate</label>
                            <input type="number" id="crossover_rate" name="crossover_rate" value="0.8" min="0" max="1" step="0.01">
                        </div>
                        <div class="form-group">
                            <label for="elite_size">Elite Size</label>
                            <input type="number" id="elite_size" name="elite_size" value="3" min="1" max="50">
                        </div>
                        <button type="submit" class="btn">Start Real Evolution 🧬</button>
                    </form>
                </div>
                
                <div class="card">
                    <h2>📊 Active Experiments</h2>
                    <div id="experiments" class="experiment-list">
                        <p style="text-align: center; color: #6b7280; padding: 20px;">
                            No experiments running. Start one to see real-time progress!
                        </p>
                    </div>
                </div>
            </div>
            
            <div class="card" style="margin-top: 30px;">
                <h2>📈 Real-time Updates</h2>
                <div id="logs" class="logs">
                    <div style="color: #10b981;">Connected to Real Evolution Dashboard</div>
                    <div style="color: #6b7280;">Ready to monitor live experiments...</div>
                </div>
            </div>
        </div>
        
        <div id="connectionStatus" class="connection-status disconnected">
            ⚡ Connecting...
        </div>
        
        <script>
            let ws = null;
            let experiments = new Map();
            
            // WebSocket connection
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/api/ws`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {
                    document.getElementById('connectionStatus').textContent = '🟢 Connected';
                    document.getElementById('connectionStatus').className = 'connection-status connected';
                    addLog('WebSocket connected to real evolution system', 'success');
                };
                
                ws.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    handleWebSocketMessage(message);
                };
                
                ws.onclose = function() {
                    document.getElementById('connectionStatus').textContent = '🔴 Disconnected';
                    document.getElementById('connectionStatus').className = 'connection-status disconnected';
                    addLog('WebSocket disconnected, reconnecting...', 'warning');
                    setTimeout(connectWebSocket, 3000);
                };
                
                ws.onerror = function() {
                    addLog('WebSocket error', 'error');
                };
            }
            
            function handleWebSocketMessage(message) {
                if (message.type === 'generation_update') {
                    const data = message.data;
                    const stats = data.stats;
                    addLog(`🧬 Gen ${stats.generation}: best=${stats.best_fitness.toFixed(2)}, avg=${stats.average_fitness.toFixed(2)}, diversity=${stats.diversity_score.toFixed(3)}`, 'info');
                    updateExperimentInList(data.experiment_id, stats);
                } else if (message.type === 'experiment_completed') {
                    const data = message.data;
                    addLog(`✅ Experiment ${data.experiment_id} completed after ${data.total_generations} generations`, 'success');
                    loadExperiments(); // Refresh experiment list
                }
            }
            
            function addLog(message, type = 'info') {
                const logs = document.getElementById('logs');
                const timestamp = new Date().toLocaleTimeString();
                const colors = {
                    info: '#60a5fa',
                    success: '#10b981',
                    warning: '#f59e0b',
                    error: '#ef4444'
                };
                
                const logEntry = document.createElement('div');
                logEntry.innerHTML = `<span style="color: #6b7280;">[${timestamp}]</span> <span style="color: ${colors[type] || colors.info};">${message}</span>`;
                logs.appendChild(logEntry);
                logs.scrollTop = logs.scrollHeight;
            }
            
            function updateExperimentInList(experimentId, stats) {
                const experimentDiv = document.querySelector(`[data-experiment-id="${experimentId}"]`);
                if (experimentDiv) {
                    const progressP = experimentDiv.querySelector('.progress');
                    if (progressP) {
                        progressP.textContent = `Gen ${stats.generation} | Best: ${stats.best_fitness.toFixed(2)} | Avg: ${stats.average_fitness.toFixed(2)}`;
                    }
                }
            }
            
            // Load experiments
            async function loadExperiments() {
                try {
                    const response = await fetch('/api/experiments');
                    const experiments = await response.json();
                    
                    const container = document.getElementById('experiments');
                    if (experiments.length === 0) {
                        container.innerHTML = '<p style="text-align: center; color: #6b7280; padding: 20px;">No experiments found. Start one to see progress!</p>';
                        return;
                    }
                    
                    container.innerHTML = experiments.map(exp => `
                        <div class="experiment-item" data-experiment-id="${exp.id}">
                            <h3>${exp.name}</h3>
                            <p class="progress">Gen ${exp.current_generation}/${exp.total_generations} | Best Fitness: ${exp.best_fitness.toFixed(2)}</p>
                            <p>Status: <span class="status ${exp.status}">${exp.status}</span></p>
                            <p>Started: ${new Date(exp.start_time).toLocaleString()}</p>
                            <div class="experiment-controls">
                                <button class="btn btn-small" onclick="pauseExperiment('${exp.id}')" ${exp.status !== 'running' ? 'disabled' : ''}>⏸️ Pause</button>
                                <button class="btn btn-small" onclick="resumeExperiment('${exp.id}')" ${exp.status !== 'paused' ? 'disabled' : ''}>▶️ Resume</button>
                                <button class="btn btn-small" onclick="stopExperiment('${exp.id}')" ${exp.status === 'completed' || exp.status === 'stopped' ? 'disabled' : ''}>⏹️ Stop</button>
                            </div>
                        </div>
                    `).join('');
                } catch (error) {
                    addLog(`Failed to load experiments: ${error.message}`, 'error');
                }
            }
            
            // Experiment form submission
            document.getElementById('experimentForm').onsubmit = async function(e) {
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
                    addLog(`🚀 Starting real evolution experiment: ${config.name}`, 'info');
                    const response = await fetch('/api/experiments', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(config)
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        addLog(`✅ Started real experiment: ${result.experiment_id}`, 'success');
                        loadExperiments();
                    } else {
                        const error = await response.json();
                        addLog(`❌ Failed to start experiment: ${error.detail}`, 'error');
                    }
                } catch (error) {
                    addLog(`❌ Error starting experiment: ${error.message}`, 'error');
                }
            };
            
            // Experiment controls
            async function pauseExperiment(id) {
                try {
                    await fetch(`/api/experiments/${id}/pause`, { method: 'POST' });
                    addLog(`⏸️ Paused experiment ${id}`, 'info');
                    loadExperiments();
                } catch (error) {
                    addLog(`Failed to pause experiment: ${error.message}`, 'error');
                }
            }
            
            async function resumeExperiment(id) {
                try {
                    await fetch(`/api/experiments/${id}/resume`, { method: 'POST' });
                    addLog(`▶️ Resumed experiment ${id}`, 'info');
                    loadExperiments();
                } catch (error) {
                    addLog(`Failed to resume experiment: ${error.message}`, 'error');
                }
            }
            
            async function stopExperiment(id) {
                try {
                    await fetch(`/api/experiments/${id}/stop`, { method: 'POST' });
                    addLog(`⏹️ Stopped experiment ${id}`, 'info');
                    loadExperiments();
                } catch (error) {
                    addLog(`Failed to stop experiment: ${error.message}`, 'error');
                }
            }
            
            // Initialize
            connectWebSocket();
            loadExperiments();
            setInterval(loadExperiments, 5000); // Refresh every 5 seconds
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.post("/experiments")
async def create_experiment(config: ExperimentConfig):
    """Create and start a new experiment."""
    experiment_id = await experiment_manager.start_experiment(config)
    return {"experiment_id": experiment_id, "status": "started"}


@router.get("/experiments")
async def list_experiments():
    """List all experiments."""
    return experiment_manager.list_experiments()


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get detailed information about a specific experiment."""
    status = experiment_manager.get_experiment_status(experiment_id)
    if not status:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return status


@router.post("/experiments/{experiment_id}/pause")
async def pause_experiment(experiment_id: str):
    """Pause an experiment."""
    success = await experiment_manager.pause_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=404, detail="Experiment not found or cannot be paused")
    return {"status": "paused"}


@router.post("/experiments/{experiment_id}/resume")
async def resume_experiment(experiment_id: str):
    """Resume a paused experiment."""
    success = await experiment_manager.resume_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=404, detail="Experiment not found or cannot be resumed")
    return {"status": "resumed"}


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
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)
