"""
Core Experiments API Routes for Genetic Coil Arena
Basic experiment management functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import uuid

from ..core.auth import get_current_user
from ..core.models import ExperimentConfig, ExperimentStatus

logger = logging.getLogger(__name__)

# Create experiments router
router = APIRouter(prefix="/experiments", tags=["experiments"])

# Global storage for experiments (in production, use proper database)
experiments_store = {}


class ExperimentCreateRequest(BaseModel):
    """Request model for creating a new experiment."""
    name: str
    description: Optional[str] = None
    config: ExperimentConfig


class ExperimentResponse(BaseModel):
    """Response model for experiment operations."""
    experiment_id: str
    status: str
    message: str


@router.get("/health")
async def experiments_health():
    """Check experiments service health."""
    return {
        'status': 'healthy',
        'service': 'Core Experiments',
        'running_experiments': len(experiments_store),
        'timestamp': datetime.now().isoformat()
    }


@router.post("/", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentCreateRequest,
    background_tasks: BackgroundTasks,
    current_user=Depends(get_current_user)
):
    """Create a new experiment."""
    try:
        # Generate experiment ID
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        
        # Create experiment record
        experiment_data = {
            'experiment_id': experiment_id,
            'name': request.name,
            'description': request.description,
            'config': request.config.dict(),
            'status': 'created',
            'created_by': current_user.get('username', 'unknown'),
            'created_at': datetime.now().isoformat(),
            'current_generation': 0,
            'best_fitness': 0.0,
            'results': None
        }
        
        # Store experiment
        experiments_store[experiment_id] = experiment_data
        
        return ExperimentResponse(
            experiment_id=experiment_id,
            status="created",
            message=f"Experiment '{request.name}' created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")


@router.get("/")
async def list_experiments():
    """List all experiments."""
    experiments = []
    
    for exp_id, exp_data in experiments_store.items():
        experiments.append({
            'experiment_id': exp_id,
            'name': exp_data.get('name', 'Unknown'),
            'status': exp_data.get('status', 'unknown'),
            'created_at': exp_data.get('created_at'),
            'current_generation': exp_data.get('current_generation', 0),
            'best_fitness': exp_data.get('best_fitness', 0.0)
        })
    
    return {
        'experiments': experiments,
        'total_count': len(experiments),
        'timestamp': datetime.now().isoformat()
    }


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment details."""
    if experiment_id not in experiments_store:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = experiments_store[experiment_id]
    return experiment


@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete an experiment."""
    if experiment_id not in experiments_store:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Remove experiment
    del experiments_store[experiment_id]
    
    return {
        'message': f'Experiment {experiment_id} deleted successfully',
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat()
    }


@router.get("/{experiment_id}/status")
async def get_experiment_status(experiment_id: str):
    """Get experiment status."""
    if experiment_id not in experiments_store:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = experiments_store[experiment_id]
    
    return {
        'experiment_id': experiment_id,
        'status': experiment.get('status', 'unknown'),
        'current_generation': experiment.get('current_generation', 0),
        'best_fitness': experiment.get('best_fitness', 0.0),
        'last_update': experiment.get('updated_at', experiment.get('created_at')),
        'timestamp': datetime.now().isoformat()
    }


@router.post("/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """Start an experiment."""
    if experiment_id not in experiments_store:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = experiments_store[experiment_id]
    experiment['status'] = 'running'
    experiment['updated_at'] = datetime.now().isoformat()
    
    return {
        'message': f'Experiment {experiment_id} started',
        'experiment_id': experiment_id,
        'status': 'running',
        'timestamp': datetime.now().isoformat()
    }


@router.post("/{experiment_id}/pause")
async def pause_experiment(experiment_id: str):
    """Pause an experiment."""
    if experiment_id not in experiments_store:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = experiments_store[experiment_id]
    experiment['status'] = 'paused'
    experiment['updated_at'] = datetime.now().isoformat()
    
    return {
        'message': f'Experiment {experiment_id} paused',
        'experiment_id': experiment_id,
        'status': 'paused',
        'timestamp': datetime.now().isoformat()
    }


@router.post("/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """Stop an experiment."""
    if experiment_id not in experiments_store:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = experiments_store[experiment_id]
    experiment['status'] = 'stopped'
    experiment['updated_at'] = datetime.now().isoformat()
    
    return {
        'message': f'Experiment {experiment_id} stopped',
        'experiment_id': experiment_id,
        'status': 'stopped',
        'timestamp': datetime.now().isoformat()
    }


@router.get("/{experiment_id}/results")
async def get_experiment_results(experiment_id: str):
    """Get experiment results."""
    if experiment_id not in experiments_store:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiment = experiments_store[experiment_id]
    
    if experiment.get('status') not in ['completed', 'stopped']:
        raise HTTPException(status_code=400, detail="Experiment not completed yet")
    
    results = experiment.get('results', {})
    
    return {
        'experiment_id': experiment_id,
        'results': results,
        'best_fitness': experiment.get('best_fitness', 0.0),
        'final_generation': experiment.get('current_generation', 0),
        'timestamp': datetime.now().isoformat()
    }
