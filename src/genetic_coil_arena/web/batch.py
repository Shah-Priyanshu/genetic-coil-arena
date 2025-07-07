"""
Phase 4A: Batch Experiment Processing
Advanced experiment management with batch processing capabilities.
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Status of batch experiment processing."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchExperimentConfig:
    """Configuration for a single experiment in a batch."""
    name: str
    template_id: str
    custom_config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=low, 2=normal, 3=high
    dependencies: List[str] = field(default_factory=list)  # List of experiment IDs to wait for


@dataclass
class BatchJobConfig:
    """Configuration for a batch of experiments."""
    batch_id: str
    name: str
    description: str
    experiments: List[BatchExperimentConfig]
    max_concurrent: int = 3
    auto_start: bool = False
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BatchProgress:
    """Progress tracking for batch processing."""
    batch_id: str
    status: BatchStatus
    total_experiments: int
    completed_experiments: int
    failed_experiments: int
    running_experiments: Set[str] = field(default_factory=set)
    completed_experiment_ids: Set[str] = field(default_factory=set)
    failed_experiment_ids: Set[str] = field(default_factory=set)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_experiments == 0:
            return 0.0
        return (self.completed_experiments / self.total_experiments) * 100
    
    @property
    def is_finished(self) -> bool:
        """Check if batch is finished (completed, failed, or cancelled)."""
        return self.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]


class BatchExperimentManager:
    """Manages batch processing of multiple experiments."""
    
    def __init__(self, max_global_concurrent: int = 5):
        self.max_global_concurrent = max_global_concurrent
        self.active_batches: Dict[str, BatchJobConfig] = {}
        self.batch_progress: Dict[str, BatchProgress] = {}
        self.running_experiments: Set[str] = set()
        self.experiment_manager = None  # Will be injected
        self._batch_tasks: Dict[str, asyncio.Task] = {}
    
    def set_experiment_manager(self, experiment_manager):
        """Inject the experiment manager dependency."""
        self.experiment_manager = experiment_manager
    
    def create_batch(self, config: BatchJobConfig) -> str:
        """Create a new batch job."""
        batch_id = config.batch_id or str(uuid.uuid4())
        config.batch_id = batch_id
        
        # Validate experiments
        for exp_config in config.experiments:
            if not exp_config.name:
                raise ValueError("All experiments must have a name")
            if not exp_config.template_id:
                raise ValueError("All experiments must specify a template_id")
        
        # Store batch configuration
        self.active_batches[batch_id] = config
        
        # Initialize progress tracking
        self.batch_progress[batch_id] = BatchProgress(
            batch_id=batch_id,
            status=BatchStatus.CREATED,
            total_experiments=len(config.experiments),
            completed_experiments=0,
            failed_experiments=0
        )
        
        logger.info(f"Created batch {batch_id} with {len(config.experiments)} experiments")
        
        # Auto-start if configured
        if config.auto_start:
            asyncio.create_task(self.start_batch(batch_id))
        
        return batch_id
    
    async def start_batch(self, batch_id: str) -> bool:
        """Start processing a batch of experiments."""
        if batch_id not in self.active_batches:
            logger.error(f"Batch {batch_id} not found")
            return False
        
        progress = self.batch_progress[batch_id]
        if progress.status != BatchStatus.CREATED:
            logger.warning(f"Batch {batch_id} already started or finished")
            return False
        
        # Update status
        progress.status = BatchStatus.RUNNING
        progress.started_at = datetime.now()
        
        # Start batch processing task
        task = asyncio.create_task(self._process_batch(batch_id))
        self._batch_tasks[batch_id] = task
        
        logger.info(f"Started batch {batch_id}")
        return True
    
    async def pause_batch(self, batch_id: str) -> bool:
        """Pause a running batch."""
        if batch_id not in self.batch_progress:
            return False
        
        progress = self.batch_progress[batch_id]
        if progress.status == BatchStatus.RUNNING:
            progress.status = BatchStatus.PAUSED
            logger.info(f"Paused batch {batch_id}")
            return True
        
        return False
    
    async def resume_batch(self, batch_id: str) -> bool:
        """Resume a paused batch."""
        if batch_id not in self.batch_progress:
            return False
        
        progress = self.batch_progress[batch_id]
        if progress.status == BatchStatus.PAUSED:
            progress.status = BatchStatus.RUNNING
            
            # Resume processing if task doesn't exist
            if batch_id not in self._batch_tasks or self._batch_tasks[batch_id].done():
                task = asyncio.create_task(self._process_batch(batch_id))
                self._batch_tasks[batch_id] = task
            
            logger.info(f"Resumed batch {batch_id}")
            return True
        
        return False
    
    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a batch and stop all its experiments."""
        if batch_id not in self.batch_progress:
            return False
        
        progress = self.batch_progress[batch_id]
        progress.status = BatchStatus.CANCELLED
        progress.completed_at = datetime.now()
        
        # Cancel the batch task
        if batch_id in self._batch_tasks:
            self._batch_tasks[batch_id].cancel()
        
        # Stop running experiments from this batch
        config = self.active_batches[batch_id]
        for exp_config in config.experiments:
            exp_id = f"{batch_id}_{exp_config.name}"
            if exp_id in progress.running_experiments:
                # Cancel experiment (implementation depends on experiment manager)
                if self.experiment_manager:
                    await self.experiment_manager.stop_experiment(exp_id)
        
        logger.info(f"Cancelled batch {batch_id}")
        return True
    
    async def _process_batch(self, batch_id: str):
        """Internal method to process experiments in a batch."""
        config = self.active_batches[batch_id]
        progress = self.batch_progress[batch_id]
        
        try:
            # Create queue of experiments ordered by priority and dependencies
            experiment_queue = self._create_execution_queue(config)
            
            # Process experiments
            while experiment_queue and progress.status == BatchStatus.RUNNING:
                # Check how many experiments we can start
                current_running = len(progress.running_experiments)
                global_running = len(self.running_experiments)
                
                can_start = min(
                    config.max_concurrent - current_running,
                    self.max_global_concurrent - global_running
                )
                
                if can_start <= 0:
                    # Wait a bit before checking again
                    await asyncio.sleep(1)
                    continue
                
                # Start experiments
                started_count = 0
                for exp_config in list(experiment_queue):
                    if started_count >= can_start:
                        break
                    
                    # Check dependencies
                    if self._can_start_experiment(exp_config, progress):
                        exp_id = await self._start_experiment(batch_id, exp_config)
                        if exp_id:
                            experiment_queue.remove(exp_config)
                            progress.running_experiments.add(exp_id)
                            self.running_experiments.add(exp_id)
                            started_count += 1
                
                # Wait a bit before next iteration
                await asyncio.sleep(2)
            
            # Wait for all experiments to complete
            while progress.running_experiments and progress.status == BatchStatus.RUNNING:
                await asyncio.sleep(1)
                await self._check_experiment_completion(batch_id)
            
            # Update final status
            if progress.status == BatchStatus.RUNNING:
                if progress.failed_experiments > 0:
                    progress.status = BatchStatus.FAILED
                else:
                    progress.status = BatchStatus.COMPLETED
                progress.completed_at = datetime.now()
            
            logger.info(f"Batch {batch_id} finished with status {progress.status.value}")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")
            progress.status = BatchStatus.FAILED
            progress.completed_at = datetime.now()
    
    def _create_execution_queue(self, config: BatchJobConfig) -> List[BatchExperimentConfig]:
        """Create an ordered queue of experiments based on priority and dependencies."""
        # Sort by priority (higher first), then by name for consistency
        return sorted(
            config.experiments,
            key=lambda x: (-x.priority, x.name)
        )
    
    def _can_start_experiment(self, exp_config: BatchExperimentConfig, progress: BatchProgress) -> bool:
        """Check if an experiment can start based on its dependencies."""
        for dep_name in exp_config.dependencies:
            dep_id = f"{progress.batch_id}_{dep_name}"
            if dep_id not in progress.completed_experiment_ids:
                return False
        return True
    
    async def _start_experiment(self, batch_id: str, exp_config: BatchExperimentConfig) -> Optional[str]:
        """Start a single experiment."""
        try:
            exp_id = f"{batch_id}_{exp_config.name}"
            
            # Create experiment from template
            from .templates import template_manager
            experiment_data = template_manager.create_experiment_from_template(
                template_id=exp_config.template_id,
                experiment_name=exp_config.name,
                custom_config=exp_config.custom_config
            )
            
            # Start experiment using experiment manager
            if self.experiment_manager:
                await self.experiment_manager.create_experiment(exp_id, experiment_data)
                logger.info(f"Started experiment {exp_id} in batch {batch_id}")
                return exp_id
            else:
                logger.warning("No experiment manager available")
                return None
                
        except Exception as e:
            logger.error(f"Failed to start experiment {exp_config.name} in batch {batch_id}: {e}")
            return None
    
    async def _check_experiment_completion(self, batch_id: str):
        """Check for completed experiments and update progress."""
        progress = self.batch_progress[batch_id]
        
        for exp_id in list(progress.running_experiments):
            # Check experiment status (implementation depends on experiment manager)
            if self.experiment_manager:
                status = await self.experiment_manager.get_experiment_status(exp_id)
                
                if status and status.get('status') in ['completed', 'failed']:
                    # Remove from running
                    progress.running_experiments.discard(exp_id)
                    self.running_experiments.discard(exp_id)
                    
                    # Update progress
                    if status['status'] == 'completed':
                        progress.completed_experiments += 1
                        progress.completed_experiment_ids.add(exp_id)
                    else:
                        progress.failed_experiments += 1
                        progress.failed_experiment_ids.add(exp_id)
    
    def get_batch_progress(self, batch_id: str) -> Optional[BatchProgress]:
        """Get progress information for a batch."""
        return self.batch_progress.get(batch_id)
    
    def list_batches(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all batches, optionally filtered by user."""
        batches = []
        
        for batch_id, config in self.active_batches.items():
            if user_id and config.created_by != user_id:
                continue
            
            progress = self.batch_progress.get(batch_id)
            
            batch_info = {
                'batch_id': batch_id,
                'name': config.name,
                'description': config.description,
                'created_by': config.created_by,
                'created_at': config.created_at.isoformat(),
                'status': progress.status.value if progress else 'unknown',
                'total_experiments': len(config.experiments),
                'progress_percentage': progress.progress_percentage if progress else 0
            }
            
            if progress:
                batch_info.update({
                    'completed_experiments': progress.completed_experiments,
                    'failed_experiments': progress.failed_experiments,
                    'running_experiments': len(progress.running_experiments)
                })
            
            batches.append(batch_info)
        
        return sorted(batches, key=lambda x: x['created_at'], reverse=True)
    
    def get_batch_details(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific batch."""
        if batch_id not in self.active_batches:
            return None
        
        config = self.active_batches[batch_id]
        progress = self.batch_progress[batch_id]
        
        return {
            'config': {
                'batch_id': config.batch_id,
                'name': config.name,
                'description': config.description,
                'max_concurrent': config.max_concurrent,
                'auto_start': config.auto_start,
                'created_by': config.created_by,
                'created_at': config.created_at.isoformat(),
                'experiments': [
                    {
                        'name': exp.name,
                        'template_id': exp.template_id,
                        'priority': exp.priority,
                        'dependencies': exp.dependencies,
                        'custom_config': exp.custom_config
                    }
                    for exp in config.experiments
                ]
            },
            'progress': {
                'status': progress.status.value,
                'total_experiments': progress.total_experiments,
                'completed_experiments': progress.completed_experiments,
                'failed_experiments': progress.failed_experiments,
                'running_experiments': list(progress.running_experiments),
                'progress_percentage': progress.progress_percentage,
                'started_at': progress.started_at.isoformat() if progress.started_at else None,
                'completed_at': progress.completed_at.isoformat() if progress.completed_at else None,
                'is_finished': progress.is_finished
            }
        }


# Global batch manager instance
batch_manager = BatchExperimentManager()
