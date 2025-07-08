"""
Neural Evolution API Routes for Genetic Coil Arena
Phase 6A: Deep Learning Integration
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import traceback
import asyncio
from pathlib import Path
import uuid
import numpy as np

from ...neuro_evolution_engine import NeuroEvolutionEngine
from ...neural_evolution import NeuralAgentConfig, NeuroEvolutionConfig, NeuralAgent
from ..core.auth import get_current_user
from ..core.models import Experiment, db

logger = logging.getLogger(__name__)

# Create neural evolution router
neural_router = APIRouter(prefix="/neural", tags=["neural-evolution"])

# Global storage for running experiments
running_experiments = {}


# Request/Response models
class NeuralAgentConfigRequest(BaseModel):
    input_size: int = 10
    hidden_layers: List[int] = [64, 32]
    output_size: int = 4
    activation: str = "relu"
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    network_type: str = "feedforward"


class NeuroEvolutionConfigRequest(BaseModel):
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    mutation_strength: float = 0.02
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    fitness_sharing: bool = True
    speciation: bool = True
    compatibility_threshold: float = 3.0


class NeuralExperimentRequest(BaseModel):
    name: str
    description: Optional[str] = None
    agent_config: NeuralAgentConfigRequest
    evolution_config: NeuroEvolutionConfigRequest
    environment_type: str = "grid_world"
    environment_config: Dict[str, Any] = {}


class NeuralExperimentResponse(BaseModel):
    experiment_id: str
    status: str
    message: str
    estimated_duration: Optional[str] = None


@neural_router.get("/health")
async def neural_health():
    """Check neural evolution service health."""
    try:
        import torch
        torch_available = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
        
        return {
            'status': 'healthy',
            'service': 'Neural Evolution Engine',
            'torch_version': torch.__version__,
            'cuda_available': torch_available,
            'running_experiments': len(running_experiments),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Neural health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@neural_router.post("/experiments", response_model=NeuralExperimentResponse)
async def create_neural_experiment(
    request: NeuralExperimentRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user)
):
    """Create and start a new neural evolution experiment."""
    try:
        # Generate experiment ID
        experiment_id = f"neural_exp_{uuid.uuid4().hex[:8]}"
        
        # Convert request to configs
        agent_config = NeuralAgentConfig(
            input_size=request.agent_config.input_size,
            hidden_layers=request.agent_config.hidden_layers,
            output_size=request.agent_config.output_size,
            activation=request.agent_config.activation,
            dropout_rate=request.agent_config.dropout_rate,
            use_batch_norm=request.agent_config.use_batch_norm,
            network_type=request.agent_config.network_type
        )
        
        evolution_config = NeuroEvolutionConfig(
            population_size=request.evolution_config.population_size,
            generations=request.evolution_config.generations,
            mutation_rate=request.evolution_config.mutation_rate,
            mutation_strength=request.evolution_config.mutation_strength,
            crossover_rate=request.evolution_config.crossover_rate,
            elite_size=request.evolution_config.elite_size,
            tournament_size=request.evolution_config.tournament_size,
            fitness_sharing=request.evolution_config.fitness_sharing,
            speciation=request.evolution_config.speciation,
            compatibility_threshold=request.evolution_config.compatibility_threshold
        )
        
        # Create experiment record
        experiment_data = {
            'experiment_id': experiment_id,
            'name': request.name,
            'description': request.description,
            'agent_config': request.agent_config.dict(),
            'evolution_config': request.evolution_config.dict(),
            'environment_type': request.environment_type,
            'environment_config': request.environment_config,
            'status': 'initializing',
            'created_by': current_user.get('id', 'unknown'),
            'created_at': datetime.now().isoformat()
        }
        
        # Store experiment info
        running_experiments[experiment_id] = {
            'config': experiment_data,
            'engine': None,
            'status': 'initializing',
            'progress': 0,
            'results': None,
            'error': None
        }
        
        # Start experiment in background
        background_tasks.add_task(
            run_neural_experiment,
            experiment_id,
            agent_config,
            evolution_config,
            request.environment_type,
            request.environment_config
        )
        
        # Estimate duration (rough estimate)
        estimated_seconds = evolution_config.population_size * evolution_config.generations * 0.1
        estimated_duration = f"{estimated_seconds:.0f} seconds"
        
        return NeuralExperimentResponse(
            experiment_id=experiment_id,
            status="started",
            message=f"Neural evolution experiment '{request.name}' started successfully",
            estimated_duration=estimated_duration
        )
        
    except Exception as e:
        logger.error(f"Error creating neural experiment: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")


async def run_neural_experiment(
    experiment_id: str,
    agent_config: NeuralAgentConfig,
    evolution_config: NeuroEvolutionConfig,
    environment_type: str,
    environment_config: Dict[str, Any]
):
    """Run neural evolution experiment in background."""
    try:
        logger.info(f"Starting neural experiment {experiment_id}")
        
        # Update status
        running_experiments[experiment_id]['status'] = 'running'
        
        # Create engine
        engine = NeuroEvolutionEngine(evolution_config, agent_config)
        running_experiments[experiment_id]['engine'] = engine
        
        # Define evaluation function based on environment type
        evaluation_function = get_evaluation_function(environment_type)
        
        # Run evolution
        results = engine.evolve(evaluation_function, environment_config)
        
        # Store results
        running_experiments[experiment_id]['results'] = results
        running_experiments[experiment_id]['status'] = 'completed'
        running_experiments[experiment_id]['progress'] = 100
        
        logger.info(f"Neural experiment {experiment_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in neural experiment {experiment_id}: {e}")
        logger.error(traceback.format_exc())
        running_experiments[experiment_id]['error'] = str(e)
        running_experiments[experiment_id]['status'] = 'failed'


def get_evaluation_function(environment_type: str):
    """Get evaluation function for the specified environment."""
    
    def grid_world_evaluation(agent: NeuralAgent, config: Dict[str, Any]) -> float:
        """Evaluate agent in a grid world environment."""
        try:
            grid_size = config.get('grid_size', 10)
            steps = config.get('max_steps', 100)
            
            # Simple grid world: agent tries to reach target
            agent_pos = [0, 0]
            target_pos = [grid_size - 1, grid_size - 1]
            
            total_reward = 0.0
            
            for step in range(steps):
                # Create input: normalized position and target direction
                dx = target_pos[0] - agent_pos[0]
                dy = target_pos[1] - agent_pos[1]
                distance = (dx ** 2 + dy ** 2) ** 0.5
                
                inputs = [
                    agent_pos[0] / grid_size,  # normalized x
                    agent_pos[1] / grid_size,  # normalized y
                    dx / grid_size,           # target direction x
                    dy / grid_size,           # target direction y
                    distance / (grid_size * 1.41),  # normalized distance
                    step / steps,             # time progress
                    1.0,                      # bias
                    0.0, 0.0, 0.0            # padding
                ]
                
                # Get action from neural network
                outputs = agent.predict(np.array(inputs))
                action = np.argmax(outputs[0])  # 0: up, 1: right, 2: down, 3: left
                
                # Move agent
                if action == 0 and agent_pos[1] > 0:  # up
                    agent_pos[1] -= 1
                elif action == 1 and agent_pos[0] < grid_size - 1:  # right
                    agent_pos[0] += 1
                elif action == 2 and agent_pos[1] < grid_size - 1:  # down
                    agent_pos[1] += 1
                elif action == 3 and agent_pos[0] > 0:  # left
                    agent_pos[0] -= 1
                
                # Calculate reward
                new_distance = ((target_pos[0] - agent_pos[0]) ** 2 + 
                               (target_pos[1] - agent_pos[1]) ** 2) ** 0.5
                
                # Reward for getting closer to target
                if new_distance < distance:
                    total_reward += 0.1
                elif new_distance > distance:
                    total_reward -= 0.05
                
                # Big reward for reaching target
                if agent_pos == target_pos:
                    total_reward += 10.0
                    break
                
                # Small penalty for each step (encourage efficiency)
                total_reward -= 0.01
            
            # Normalize fitness to [0, 1]
            max_possible_reward = 10.0  # reaching target
            fitness = max(0.0, min(1.0, total_reward / max_possible_reward))
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error in grid world evaluation: {e}")
            return 0.0
    
    def xor_evaluation(agent: NeuralAgent, config: Dict[str, Any]) -> float:
        """Evaluate agent on XOR problem."""
        try:
            import numpy as np
            
            # XOR truth table
            inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            targets = np.array([[0], [1], [1], [0]])
            
            total_error = 0.0
            
            for i in range(len(inputs)):
                # Pad input to match agent's input size
                padded_input = np.zeros(agent.config.input_size)
                padded_input[:2] = inputs[i]
                
                output = agent.predict(padded_input)
                predicted = output[0][0] if len(output[0]) > 0 else 0
                
                error = abs(predicted - targets[i][0])
                total_error += error
            
            # Convert error to fitness (lower error = higher fitness)
            average_error = total_error / len(inputs)
            fitness = max(0.0, 1.0 - average_error)
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error in XOR evaluation: {e}")
            return 0.0
    
    # Return appropriate evaluation function
    if environment_type == "grid_world":
        return grid_world_evaluation
    elif environment_type == "xor":
        return xor_evaluation
    else:
        logger.warning(f"Unknown environment type: {environment_type}, using grid_world")
        return grid_world_evaluation


@neural_router.get("/experiments/{experiment_id}")
async def get_neural_experiment(experiment_id: str):
    """Get neural evolution experiment status and results."""
    if experiment_id not in running_experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = running_experiments[experiment_id]
    
    response = {
        'experiment_id': experiment_id,
        'config': experiment['config'],
        'status': experiment['status'],
        'progress': experiment.get('progress', 0),
        'error': experiment.get('error')
    }
    
    # Add results if completed
    if experiment['results']:
        response['results'] = experiment['results']
    
    # Add current generation info if running
    if experiment['engine'] and experiment['status'] == 'running':
        summary = experiment['engine'].get_population_summary()
        response['current_generation'] = summary
    
    return response


@neural_router.get("/experiments")
async def list_neural_experiments():
    """List all neural evolution experiments."""
    experiments = []
    
    for exp_id, exp_data in running_experiments.items():
        experiments.append({
            'experiment_id': exp_id,
            'name': exp_data['config'].get('name', 'Unknown'),
            'status': exp_data['status'],
            'created_at': exp_data['config'].get('created_at'),
            'progress': exp_data.get('progress', 0)
        })
    
    return {
        'experiments': experiments,
        'total_count': len(experiments),
        'timestamp': datetime.now().isoformat()
    }


@neural_router.delete("/experiments/{experiment_id}")
async def delete_neural_experiment(experiment_id: str):
    """Delete a neural evolution experiment."""
    if experiment_id not in running_experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Stop if running (this is simplified - in production you'd need proper cancellation)
    experiment = running_experiments[experiment_id]
    if experiment['status'] == 'running':
        experiment['status'] = 'cancelled'
    
    # Remove from storage
    del running_experiments[experiment_id]
    
    return {
        'message': f'Experiment {experiment_id} deleted successfully',
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat()
    }


@neural_router.get("/experiments/{experiment_id}/best-agent")
async def get_best_agent(experiment_id: str):
    """Get the best agent from a neural evolution experiment."""
    if experiment_id not in running_experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = running_experiments[experiment_id]
    
    if not experiment['results']:
        raise HTTPException(status_code=400, detail="Experiment not completed yet")
    
    best_agent = experiment['results'].get('best_agent')
    if not best_agent:
        raise HTTPException(status_code=404, detail="No best agent found")
    
    return {
        'experiment_id': experiment_id,
        'best_agent': best_agent,
        'best_fitness': experiment['results']['best_fitness'],
        'timestamp': datetime.now().isoformat()
    }


@neural_router.post("/experiments/{experiment_id}/test-agent")
async def test_best_agent(
    experiment_id: str,
    test_config: Dict[str, Any] = {}
):
    """Test the best agent from an experiment."""
    if experiment_id not in running_experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = running_experiments[experiment_id]
    
    if not experiment['results'] or not experiment['results'].get('best_agent'):
        raise HTTPException(status_code=400, detail="No best agent available")
    
    try:
        # This is a simplified test - in practice you'd reconstruct the agent
        # and run it in the specified test environment
        
        best_agent_data = experiment['results']['best_agent']
        
        # Simulate testing (in practice, you'd load the agent and run tests)
        test_results = {
            'agent_id': best_agent_data['agent_id'],
            'test_fitness': best_agent_data['fitness'],
            'test_config': test_config,
            'test_timestamp': datetime.now().isoformat(),
            'message': 'Agent testing completed (simulated)'
        }
        
        return test_results
        
    except Exception as e:
        logger.error(f"Error testing agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test agent: {str(e)}")


@neural_router.get("/templates")
async def get_neural_templates():
    """Get predefined neural evolution templates."""
    templates = {
        'simple_feedforward': {
            'name': 'Simple Feedforward Network',
            'description': 'Basic feedforward neural network for simple problems',
            'agent_config': {
                'input_size': 10,
                'hidden_layers': [32, 16],
                'output_size': 4,
                'activation': 'relu',
                'network_type': 'feedforward'
            },
            'evolution_config': {
                'population_size': 30,
                'generations': 50,
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'speciation': False
            },
            'recommended_for': ['grid_world', 'simple_control']
        },
        'neat_like': {
            'name': 'NEAT-style Evolution',
            'description': 'Neuroevolution with speciation and fitness sharing',
            'agent_config': {
                'input_size': 10,
                'hidden_layers': [64, 32],
                'output_size': 4,
                'activation': 'tanh',
                'network_type': 'feedforward'
            },
            'evolution_config': {
                'population_size': 100,
                'generations': 200,
                'mutation_rate': 0.15,
                'crossover_rate': 0.8,
                'speciation': True,
                'fitness_sharing': True
            },
            'recommended_for': ['complex_control', 'game_playing']
        },
        'recurrent_network': {
            'name': 'LSTM-based Evolution',
            'description': 'Recurrent neural networks for sequential problems',
            'agent_config': {
                'input_size': 10,
                'hidden_layers': [64, 32],
                'output_size': 4,
                'activation': 'tanh',
                'network_type': 'lstm'
            },
            'evolution_config': {
                'population_size': 50,
                'generations': 100,
                'mutation_rate': 0.08,
                'crossover_rate': 0.6,
                'speciation': True
            },
            'recommended_for': ['sequence_prediction', 'memory_tasks']
        }
    }
    
    return {
        'templates': templates,
        'timestamp': datetime.now().isoformat()
    }


# Error handlers
@neural_router.get("/status")
async def neural_system_status():
    """Get comprehensive neural evolution system status."""
    try:
        import torch
        
        return {
            'service': 'Neural Evolution System',
            'status': 'operational',
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available() if hasattr(torch, 'cuda') else False,
            'running_experiments': len(running_experiments),
            'experiments_by_status': {
                status: len([e for e in running_experiments.values() if e['status'] == status])
                for status in ['initializing', 'running', 'completed', 'failed', 'cancelled']
            },
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'service': 'Neural Evolution System',
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
