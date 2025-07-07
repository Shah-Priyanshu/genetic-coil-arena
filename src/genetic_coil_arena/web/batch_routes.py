"""
Phase 4A: Batch Processing API Routes
RESTful endpoints for managing batch experiment processing.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

from .batch import batch_manager, BatchJobConfig, BatchExperimentConfig, BatchStatus
from .auth import get_current_user

# Create router
batch_router = APIRouter(prefix="/batch", tags=["batch-processing"])


# Pydantic models for API
class ExperimentConfigRequest(BaseModel):
    """Request model for experiment configuration in a batch."""
    name: str
    template_id: str
    custom_config: Optional[Dict[str, Any]] = None
    priority: int = 1
    dependencies: List[str] = []


class BatchJobRequest(BaseModel):
    """Request model for creating a batch job."""
    name: str
    description: str
    experiments: List[ExperimentConfigRequest]
    max_concurrent: int = 3
    auto_start: bool = False


class BatchResponse(BaseModel):
    """Response model for batch information."""
    batch_id: str
    name: str
    description: str
    created_by: str
    created_at: str
    status: str
    total_experiments: int
    progress_percentage: float
    completed_experiments: int
    failed_experiments: int
    running_experiments: int


class BatchDetailsResponse(BaseModel):
    """Detailed response model for batch information."""
    config: Dict[str, Any]
    progress: Dict[str, Any]


@batch_router.post("/", response_model=Dict[str, str])
async def create_batch(
    request: BatchJobRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new batch of experiments."""
    try:
        # Convert request to internal format
        experiments = [
            BatchExperimentConfig(
                name=exp.name,
                template_id=exp.template_id,
                custom_config=exp.custom_config or {},
                priority=exp.priority,
                dependencies=exp.dependencies
            )
            for exp in request.experiments
        ]
        
        config = BatchJobConfig(
            batch_id="",  # Will be generated
            name=request.name,
            description=request.description,
            experiments=experiments,
            max_concurrent=request.max_concurrent,
            auto_start=request.auto_start,
            created_by=current_user.get("username", "unknown")
        )
        
        batch_id = batch_manager.create_batch(config)
        
        return {
            "message": "Batch created successfully",
            "batch_id": batch_id,
            "status": "created" if not request.auto_start else "starting"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create batch: {str(e)}")


@batch_router.get("/", response_model=List[BatchResponse])
async def list_batches(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get list of all batches for the current user."""
    user_id = current_user.get("username")
    batches = batch_manager.list_batches(user_id)
    
    return [
        BatchResponse(
            batch_id=batch["batch_id"],
            name=batch["name"],
            description=batch["description"],
            created_by=batch["created_by"],
            created_at=batch["created_at"],
            status=batch["status"],
            total_experiments=batch["total_experiments"],
            progress_percentage=batch["progress_percentage"],
            completed_experiments=batch.get("completed_experiments", 0),
            failed_experiments=batch.get("failed_experiments", 0),
            running_experiments=batch.get("running_experiments", 0)
        )
        for batch in batches
    ]


@batch_router.get("/{batch_id}", response_model=BatchDetailsResponse)
async def get_batch_details(
    batch_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get detailed information about a specific batch."""
    details = batch_manager.get_batch_details(batch_id)
    
    if not details:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Check ownership
    if details["config"]["created_by"] != current_user.get("username"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return BatchDetailsResponse(
        config=details["config"],
        progress=details["progress"]
    )


@batch_router.post("/{batch_id}/start")
async def start_batch(
    batch_id: str,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Start processing a batch of experiments."""
    # Check if batch exists and user has access
    details = batch_manager.get_batch_details(batch_id)
    if not details:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    if details["config"]["created_by"] != current_user.get("username"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Start batch processing in background
    background_tasks.add_task(batch_manager.start_batch, batch_id)
    
    return {"message": "Batch started", "batch_id": batch_id}


@batch_router.post("/{batch_id}/pause")
async def pause_batch(
    batch_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Pause a running batch."""
    # Check access
    details = batch_manager.get_batch_details(batch_id)
    if not details:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    if details["config"]["created_by"] != current_user.get("username"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = await batch_manager.pause_batch(batch_id)
    
    if success:
        return {"message": "Batch paused", "batch_id": batch_id}
    else:
        raise HTTPException(status_code=400, detail="Could not pause batch")


@batch_router.post("/{batch_id}/resume")
async def resume_batch(
    batch_id: str,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Resume a paused batch."""
    # Check access
    details = batch_manager.get_batch_details(batch_id)
    if not details:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    if details["config"]["created_by"] != current_user.get("username"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = await batch_manager.resume_batch(batch_id)
    
    if success:
        return {"message": "Batch resumed", "batch_id": batch_id}
    else:
        raise HTTPException(status_code=400, detail="Could not resume batch")


@batch_router.post("/{batch_id}/cancel")
async def cancel_batch(
    batch_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Cancel a batch and stop all its experiments."""
    # Check access
    details = batch_manager.get_batch_details(batch_id)
    if not details:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    if details["config"]["created_by"] != current_user.get("username"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = await batch_manager.cancel_batch(batch_id)
    
    if success:
        return {"message": "Batch cancelled", "batch_id": batch_id}
    else:
        raise HTTPException(status_code=400, detail="Could not cancel batch")


@batch_router.get("/{batch_id}/progress")
async def get_batch_progress(
    batch_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get real-time progress of a batch."""
    progress = batch_manager.get_batch_progress(batch_id)
    
    if not progress:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Check access
    details = batch_manager.get_batch_details(batch_id)
    if details and details["config"]["created_by"] != current_user.get("username"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "batch_id": batch_id,
        "status": progress.status.value,
        "progress_percentage": progress.progress_percentage,
        "total_experiments": progress.total_experiments,
        "completed_experiments": progress.completed_experiments,
        "failed_experiments": progress.failed_experiments,
        "running_experiments": len(progress.running_experiments),
        "running_experiment_ids": list(progress.running_experiments),
        "is_finished": progress.is_finished,
        "started_at": progress.started_at.isoformat() if progress.started_at else None,
        "completed_at": progress.completed_at.isoformat() if progress.completed_at else None
    }


@batch_router.get("/status/overview")
async def get_system_overview(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get system-wide batch processing overview."""
    user_id = current_user.get("username")
    user_batches = batch_manager.list_batches(user_id)
    
    # Calculate statistics
    total_batches = len(user_batches)
    running_batches = len([b for b in user_batches if b["status"] == "running"])
    completed_batches = len([b for b in user_batches if b["status"] == "completed"])
    failed_batches = len([b for b in user_batches if b["status"] == "failed"])
    
    total_experiments = sum(b["total_experiments"] for b in user_batches)
    completed_experiments = sum(b.get("completed_experiments", 0) for b in user_batches)
    
    return {
        "user_stats": {
            "total_batches": total_batches,
            "running_batches": running_batches,
            "completed_batches": completed_batches,
            "failed_batches": failed_batches,
            "total_experiments": total_experiments,
            "completed_experiments": completed_experiments
        },
        "system_stats": {
            "global_running_experiments": len(batch_manager.running_experiments),
            "max_global_concurrent": batch_manager.max_global_concurrent
        },
        "recent_batches": user_batches[:5]  # Last 5 batches
    }
