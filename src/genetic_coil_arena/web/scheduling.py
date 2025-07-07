"""
Phase 4A: Experiment Scheduling System
Time-based experiment execution with scheduling capabilities.
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from .models import ExperimentConfig
from .database import DatabaseManager

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of scheduling."""
    ONCE = "once"           # Run once at specified time
    DAILY = "daily"         # Run daily at specified time
    WEEKLY = "weekly"       # Run weekly on specified day/time
    INTERVAL = "interval"   # Run every X minutes/hours


class ScheduleStatus(Enum):
    """Status of scheduled experiments."""
    PENDING = "pending"     # Waiting to run
    RUNNING = "running"     # Currently executing
    COMPLETED = "completed" # Successfully completed
    FAILED = "failed"       # Failed to execute
    CANCELLED = "cancelled" # Manually cancelled
    SKIPPED = "skipped"     # Skipped due to conflict/resource


@dataclass
class ScheduleConfig:
    """Configuration for scheduled experiment."""
    schedule_id: str
    name: str
    description: str
    experiment_config: ExperimentConfig
    schedule_type: ScheduleType
    
    # Time-based scheduling
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None
    
    # Interval-based scheduling
    interval_minutes: Optional[int] = None
    
    # Weekly scheduling
    day_of_week: Optional[int] = None  # 0=Monday, 6=Sunday
    time_of_day: Optional[str] = None  # "HH:MM" format
    
    # Execution settings
    max_concurrent: int = 1
    retry_attempts: int = 3
    timeout_minutes: int = 60
    
    # Metadata
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True


@dataclass
class ScheduledExecution:
    """Record of a scheduled experiment execution."""
    execution_id: str
    schedule_id: str
    experiment_id: Optional[str] = None
    scheduled_time: datetime = field(default_factory=datetime.now)
    actual_start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    status: ScheduleStatus = ScheduleStatus.PENDING
    attempt_number: int = 1
    error_message: Optional[str] = None
    result_data: Dict[str, Any] = field(default_factory=dict)


class ExperimentScheduler:
    """Advanced experiment scheduling system."""
    
    def __init__(self, experiment_manager, db_manager: DatabaseManager):
        self.experiment_manager = experiment_manager
        self.db_manager = db_manager
        self.schedules: Dict[str, ScheduleConfig] = {}
        self.executions: Dict[str, ScheduledExecution] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self._scheduler_task: Optional[asyncio.Task] = None
        self._scheduler_running = False
    
    async def start_scheduler(self):
        """Start the background scheduler."""
        if self._scheduler_running:
            logger.warning("Scheduler already running")
            return
        
        self._scheduler_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("🕒 Experiment scheduler started")
    
    async def stop_scheduler(self):
        """Stop the background scheduler."""
        self._scheduler_running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all running tasks
        for task in self.running_tasks.values():
            task.cancel()
        
        logger.info("🛑 Experiment scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._scheduler_running:
            try:
                await self._check_and_execute_schedules()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_and_execute_schedules(self):
        """Check for schedules that need to be executed."""
        current_time = datetime.now()
        
        for schedule_id, schedule in self.schedules.items():
            if not schedule.enabled:
                continue
            
            try:
                next_execution_time = self._calculate_next_execution(schedule)
                
                # Check if it's time to execute
                if next_execution_time and next_execution_time <= current_time:
                    # Check if already running
                    if schedule_id in self.running_tasks:
                        continue
                    
                    # Check concurrent execution limit
                    active_count = len([e for e in self.executions.values() 
                                      if e.schedule_id == schedule_id and e.status == ScheduleStatus.RUNNING])
                    
                    if active_count >= schedule.max_concurrent:
                        logger.debug(f"Schedule {schedule_id} at max concurrent limit ({active_count})")
                        continue
                    
                    # Execute the schedule
                    await self._execute_schedule(schedule)
                    
            except Exception as e:
                logger.error(f"Error checking schedule {schedule_id}: {e}")
    
    def _calculate_next_execution(self, schedule: ScheduleConfig) -> Optional[datetime]:
        """Calculate the next execution time for a schedule."""
        now = datetime.now()
        
        if schedule.schedule_type == ScheduleType.ONCE:
            if schedule.start_datetime and schedule.start_datetime > now:
                return schedule.start_datetime
            return None
        
        elif schedule.schedule_type == ScheduleType.INTERVAL:
            if not schedule.interval_minutes:
                return None
            
            # Find last execution
            last_execution = None
            for execution in self.executions.values():
                if (execution.schedule_id == schedule.schedule_id and 
                    execution.status in [ScheduleStatus.COMPLETED, ScheduleStatus.FAILED]):
                    if not last_execution or execution.completion_time > last_execution.completion_time:
                        last_execution = execution
            
            if last_execution:
                next_time = last_execution.completion_time + timedelta(minutes=schedule.interval_minutes)
            else:
                next_time = schedule.start_datetime or now
            
            return next_time if next_time <= (schedule.end_datetime or datetime.max) else None
        
        elif schedule.schedule_type == ScheduleType.DAILY:
            if not schedule.time_of_day:
                return None
            
            try:
                hour, minute = map(int, schedule.time_of_day.split(':'))
                today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                if today <= now:
                    today += timedelta(days=1)
                
                return today if today <= (schedule.end_datetime or datetime.max) else None
            except ValueError:
                logger.error(f"Invalid time format for schedule {schedule.schedule_id}: {schedule.time_of_day}")
                return None
        
        elif schedule.schedule_type == ScheduleType.WEEKLY:
            if schedule.day_of_week is None or not schedule.time_of_day:
                return None
            
            try:
                hour, minute = map(int, schedule.time_of_day.split(':'))
                
                # Find next occurrence of the specified day
                days_ahead = schedule.day_of_week - now.weekday()
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                
                target_date = now + timedelta(days=days_ahead)
                target_datetime = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                return target_datetime if target_datetime <= (schedule.end_datetime or datetime.max) else None
            except ValueError:
                logger.error(f"Invalid time format for schedule {schedule.schedule_id}: {schedule.time_of_day}")
                return None
        
        return None
    
    async def _execute_schedule(self, schedule: ScheduleConfig):
        """Execute a scheduled experiment."""
        execution_id = str(uuid.uuid4())
        execution = ScheduledExecution(
            execution_id=execution_id,
            schedule_id=schedule.schedule_id,
            scheduled_time=datetime.now(),
            status=ScheduleStatus.PENDING
        )
        
        self.executions[execution_id] = execution
        
        # Create execution task
        task = asyncio.create_task(self._run_scheduled_experiment(schedule, execution))
        self.running_tasks[schedule.schedule_id] = task
        
        logger.info(f"📅 Started scheduled execution {execution_id} for schedule {schedule.schedule_id}")
    
    async def _run_scheduled_experiment(self, schedule: ScheduleConfig, execution: ScheduledExecution):
        """Run the actual scheduled experiment."""
        try:
            execution.status = ScheduleStatus.RUNNING
            execution.actual_start_time = datetime.now()
            
            # Start the experiment using the experiment manager
            experiment_id = await self.experiment_manager.start_experiment(schedule.experiment_config)
            execution.experiment_id = experiment_id
            
            logger.info(f"🧬 Started experiment {experiment_id} from schedule {schedule.schedule_id}")
            
            # Monitor experiment completion (simplified - in real implementation would use callbacks)
            timeout_time = execution.actual_start_time + timedelta(minutes=schedule.timeout_minutes)
            
            while datetime.now() < timeout_time:
                status = self.experiment_manager.get_experiment_status(experiment_id)
                if not status:
                    break
                
                if status.status in ['completed', 'failed', 'stopped']:
                    execution.status = ScheduleStatus.COMPLETED if status.status == 'completed' else ScheduleStatus.FAILED
                    execution.completion_time = datetime.now()
                    execution.result_data = {
                        'final_status': status.status,
                        'best_fitness': status.best_fitness,
                        'generations': status.current_generation
                    }
                    break
                
                await asyncio.sleep(10)  # Check every 10 seconds
            else:
                # Timeout reached
                await self.experiment_manager.stop_experiment(experiment_id)
                execution.status = ScheduleStatus.FAILED
                execution.error_message = f"Experiment timed out after {schedule.timeout_minutes} minutes"
                execution.completion_time = datetime.now()
            
        except Exception as e:
            execution.status = ScheduleStatus.FAILED
            execution.error_message = str(e)
            execution.completion_time = datetime.now()
            logger.error(f"Scheduled experiment failed: {e}")
        
        finally:
            # Clean up
            if schedule.schedule_id in self.running_tasks:
                del self.running_tasks[schedule.schedule_id]
    
    def create_schedule(self, config: ScheduleConfig) -> str:
        """Create a new experiment schedule."""
        if not config.schedule_id:
            config.schedule_id = str(uuid.uuid4())
        
        # Validate schedule configuration
        self._validate_schedule_config(config)
        
        self.schedules[config.schedule_id] = config
        logger.info(f"📅 Created schedule {config.schedule_id}: {config.name}")
        
        return config.schedule_id
    
    def _validate_schedule_config(self, config: ScheduleConfig):
        """Validate schedule configuration."""
        if config.schedule_type == ScheduleType.ONCE and not config.start_datetime:
            raise ValueError("ONCE schedule requires start_datetime")
        
        if config.schedule_type == ScheduleType.INTERVAL and not config.interval_minutes:
            raise ValueError("INTERVAL schedule requires interval_minutes")
        
        if config.schedule_type == ScheduleType.DAILY and not config.time_of_day:
            raise ValueError("DAILY schedule requires time_of_day")
        
        if config.schedule_type == ScheduleType.WEEKLY:
            if config.day_of_week is None or not config.time_of_day:
                raise ValueError("WEEKLY schedule requires day_of_week and time_of_day")
        
        if config.time_of_day:
            try:
                hour, minute = map(int, config.time_of_day.split(':'))
                if not (0 <= hour <= 23 and 0 <= minute <= 59):
                    raise ValueError("Invalid time format")
            except ValueError:
                raise ValueError("time_of_day must be in HH:MM format")
    
    def update_schedule(self, schedule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing schedule."""
        if schedule_id not in self.schedules:
            return False
        
        schedule = self.schedules[schedule_id]
        
        # Update fields
        for key, value in updates.items():
            if hasattr(schedule, key):
                setattr(schedule, key, value)
        
        # Re-validate
        self._validate_schedule_config(schedule)
        
        logger.info(f"📅 Updated schedule {schedule_id}")
        return True
    
    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule."""
        if schedule_id not in self.schedules:
            return False
        
        # Cancel running task if exists
        if schedule_id in self.running_tasks:
            self.running_tasks[schedule_id].cancel()
            del self.running_tasks[schedule_id]
        
        del self.schedules[schedule_id]
        logger.info(f"🗑️ Deleted schedule {schedule_id}")
        return True
    
    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a schedule."""
        if schedule_id in self.schedules:
            self.schedules[schedule_id].enabled = True
            return True
        return False
    
    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a schedule."""
        if schedule_id in self.schedules:
            self.schedules[schedule_id].enabled = False
            return True
        return False
    
    def get_schedule(self, schedule_id: str) -> Optional[ScheduleConfig]:
        """Get a specific schedule."""
        return self.schedules.get(schedule_id)
    
    def list_schedules(self) -> List[ScheduleConfig]:
        """List all schedules."""
        return list(self.schedules.values())
    
    def get_schedule_executions(self, schedule_id: str) -> List[ScheduledExecution]:
        """Get execution history for a schedule."""
        return [e for e in self.executions.values() if e.schedule_id == schedule_id]
    
    def get_execution(self, execution_id: str) -> Optional[ScheduledExecution]:
        """Get a specific execution."""
        return self.executions.get(execution_id)
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get overall scheduler status."""
        active_schedules = len([s for s in self.schedules.values() if s.enabled])
        running_executions = len([e for e in self.executions.values() if e.status == ScheduleStatus.RUNNING])
        
        return {
            "scheduler_running": self._scheduler_running,
            "total_schedules": len(self.schedules),
            "active_schedules": active_schedules,
            "running_executions": running_executions,
            "total_executions": len(self.executions),
            "next_executions": self._get_next_executions(limit=5)
        }
    
    def _get_next_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get upcoming executions."""
        upcoming = []
        
        for schedule in self.schedules.values():
            if not schedule.enabled:
                continue
            
            next_time = self._calculate_next_execution(schedule)
            if next_time:
                upcoming.append({
                    "schedule_id": schedule.schedule_id,
                    "schedule_name": schedule.name,
                    "next_execution_time": next_time.isoformat(),
                    "schedule_type": schedule.schedule_type.value
                })
        
        # Sort by execution time
        upcoming.sort(key=lambda x: x["next_execution_time"])
        return upcoming[:limit]
