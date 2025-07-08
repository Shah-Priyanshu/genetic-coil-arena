"""
Phase 4A: Experiment Scheduling API Routes
REST API endpoints for experiment scheduling and management.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .scheduling import ExperimentScheduler, ScheduleConfig, ScheduleType, ScheduleStatus
from .models import ExperimentConfig
from .database import DatabaseManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scheduling", tags=["experiment-scheduling"])

# Global scheduler instance
scheduler_instance: Optional[ExperimentScheduler] = None


class ScheduleCreateRequest(BaseModel):
    """Request model for creating a schedule."""
    name: str
    description: str = ""
    experiment_config: ExperimentConfig
    schedule_type: ScheduleType
    
    # Time-based scheduling
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None
    
    # Interval-based scheduling
    interval_minutes: Optional[int] = None
    
    # Weekly scheduling
    day_of_week: Optional[int] = Field(None, ge=0, le=6)  # 0=Monday, 6=Sunday
    time_of_day: Optional[str] = Field(None, pattern=r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$')
    
    # Execution settings
    max_concurrent: int = Field(1, ge=1, le=10)
    retry_attempts: int = Field(3, ge=1, le=10)
    timeout_minutes: int = Field(60, ge=5, le=1440)
    enabled: bool = True


class ScheduleUpdateRequest(BaseModel):
    """Request model for updating a schedule."""
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    max_concurrent: Optional[int] = Field(None, ge=1, le=10)
    timeout_minutes: Optional[int] = Field(None, ge=5, le=1440)


async def get_scheduler():
    """Get or create scheduler instance."""
    global scheduler_instance
    if not scheduler_instance:
        from .api import experiment_manager  # Import here to avoid circular dependency
        db_manager = DatabaseManager()
        scheduler_instance = ExperimentScheduler(experiment_manager, db_manager)
        await scheduler_instance.start_scheduler()
    return scheduler_instance


@router.post("/schedules")
async def create_schedule(request: ScheduleCreateRequest):
    """Create a new experiment schedule."""
    try:
        scheduler = await get_scheduler()
        
        # Create schedule config
        config = ScheduleConfig(
            schedule_id="",  # Will be generated
            name=request.name,
            description=request.description,
            experiment_config=request.experiment_config,
            schedule_type=request.schedule_type,
            start_datetime=request.start_datetime,
            end_datetime=request.end_datetime,
            interval_minutes=request.interval_minutes,
            day_of_week=request.day_of_week,
            time_of_day=request.time_of_day,
            max_concurrent=request.max_concurrent,
            retry_attempts=request.retry_attempts,
            timeout_minutes=request.timeout_minutes,
            enabled=request.enabled
        )
        
        schedule_id = scheduler.create_schedule(config)
        
        logger.info(f"Created schedule {schedule_id}")
        return {"schedule_id": schedule_id, "status": "created"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create schedule: {e}")
        raise HTTPException(status_code=500, detail="Failed to create schedule")


@router.get("/schedules")
async def list_schedules():
    """List all experiment schedules."""
    try:
        scheduler = await get_scheduler()
        schedules = scheduler.list_schedules()
        
        return {
            "schedules": [
                {
                    "schedule_id": s.schedule_id,
                    "name": s.name,
                    "description": s.description,
                    "schedule_type": s.schedule_type.value,
                    "enabled": s.enabled,
                    "created_at": s.created_at.isoformat(),
                    "next_execution": None  # TODO: Calculate next execution
                }
                for s in schedules
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to list schedules: {e}")
        raise HTTPException(status_code=500, detail="Failed to list schedules")


@router.get("/schedules/{schedule_id}")
async def get_schedule(schedule_id: str):
    """Get detailed information about a specific schedule."""
    try:
        scheduler = await get_scheduler()
        schedule = scheduler.get_schedule(schedule_id)
        
        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        # Get execution history
        executions = scheduler.get_schedule_executions(schedule_id)
        
        return {
            "schedule": {
                "schedule_id": schedule.schedule_id,
                "name": schedule.name,
                "description": schedule.description,
                "schedule_type": schedule.schedule_type.value,
                "enabled": schedule.enabled,
                "created_at": schedule.created_at.isoformat(),
                "experiment_config": schedule.experiment_config.model_dump(),
                "execution_settings": {
                    "max_concurrent": schedule.max_concurrent,
                    "retry_attempts": schedule.retry_attempts,
                    "timeout_minutes": schedule.timeout_minutes
                }
            },
            "executions": [
                {
                    "execution_id": e.execution_id,
                    "scheduled_time": e.scheduled_time.isoformat(),
                    "actual_start_time": e.actual_start_time.isoformat() if e.actual_start_time else None,
                    "completion_time": e.completion_time.isoformat() if e.completion_time else None,
                    "status": e.status.value,
                    "experiment_id": e.experiment_id,
                    "attempt_number": e.attempt_number,
                    "error_message": e.error_message
                }
                for e in executions[-10:]  # Last 10 executions
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get schedule: {e}")
        raise HTTPException(status_code=500, detail="Failed to get schedule")


@router.put("/schedules/{schedule_id}")
async def update_schedule(schedule_id: str, request: ScheduleUpdateRequest):
    """Update an existing schedule."""
    try:
        scheduler = await get_scheduler()
        
        # Prepare updates
        updates = {}
        if request.name is not None:
            updates["name"] = request.name
        if request.description is not None:
            updates["description"] = request.description
        if request.enabled is not None:
            updates["enabled"] = request.enabled
        if request.max_concurrent is not None:
            updates["max_concurrent"] = request.max_concurrent
        if request.timeout_minutes is not None:
            updates["timeout_minutes"] = request.timeout_minutes
        
        success = scheduler.update_schedule(schedule_id, updates)
        
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return {"status": "updated"}
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update schedule: {e}")
        raise HTTPException(status_code=500, detail="Failed to update schedule")


@router.delete("/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str):
    """Delete a schedule."""
    try:
        scheduler = await get_scheduler()
        success = scheduler.delete_schedule(schedule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return {"status": "deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete schedule: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete schedule")


@router.post("/schedules/{schedule_id}/enable")
async def enable_schedule(schedule_id: str):
    """Enable a schedule."""
    try:
        scheduler = await get_scheduler()
        success = scheduler.enable_schedule(schedule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return {"status": "enabled"}
        
    except Exception as e:
        logger.error(f"Failed to enable schedule: {e}")
        raise HTTPException(status_code=500, detail="Failed to enable schedule")


@router.post("/schedules/{schedule_id}/disable")
async def disable_schedule(schedule_id: str):
    """Disable a schedule."""
    try:
        scheduler = await get_scheduler()
        success = scheduler.disable_schedule(schedule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")
        
        return {"status": "disabled"}
        
    except Exception as e:
        logger.error(f"Failed to disable schedule: {e}")
        raise HTTPException(status_code=500, detail="Failed to disable schedule")


@router.get("/status")
async def get_scheduler_status():
    """Get overall scheduler status."""
    try:
        scheduler = await get_scheduler()
        status = scheduler.get_scheduler_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get scheduler status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get scheduler status")


@router.get("/executions/{execution_id}")
async def get_execution(execution_id: str):
    """Get detailed information about a specific execution."""
    try:
        scheduler = await get_scheduler()
        execution = scheduler.get_execution(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return {
            "execution_id": execution.execution_id,
            "schedule_id": execution.schedule_id,
            "experiment_id": execution.experiment_id,
            "scheduled_time": execution.scheduled_time.isoformat(),
            "actual_start_time": execution.actual_start_time.isoformat() if execution.actual_start_time else None,
            "completion_time": execution.completion_time.isoformat() if execution.completion_time else None,
            "status": execution.status.value,
            "attempt_number": execution.attempt_number,
            "error_message": execution.error_message,
            "result_data": execution.result_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to get execution")


@router.get("/dashboard")
async def scheduling_dashboard():
    """Serve the experiment scheduling dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experiment Scheduling - Genetic Coil Arena</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            
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
            
            .grid { display: grid; grid-template-columns: 1fr 2fr; gap: 30px; margin-bottom: 30px; }
            .grid-full { grid-column: 1 / -1; }
            
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
            .form-group input, .form-group select, .form-group textarea {
                width: 100%;
                padding: 12px;
                border: 2px solid #e5e7eb;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.2s;
            }
            .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
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
                margin: 5px;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
            }
            .btn.btn-small { padding: 8px 16px; font-size: 14px; }
            .btn.btn-danger { background: linear-gradient(135deg, #ef4444, #dc2626); }
            .btn.btn-success { background: linear-gradient(135deg, #10b981, #059669); }
            
            .schedule-item {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 10px;
                background: #fafafa;
            }
            .schedule-item h3 { margin-bottom: 5px; color: #374151; }
            .schedule-item p { color: #6b7280; font-size: 14px; margin-bottom: 5px; }
            
            .status { 
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 500;
                text-transform: uppercase;
            }
            .status.pending { background: #fef3c7; color: #92400e; }
            .status.running { background: #dcfce7; color: #166534; }
            .status.completed { background: #dbeafe; color: #1d4ed8; }
            .status.failed { background: #fee2e2; color: #dc2626; }
            .status.cancelled { background: #f3f4f6; color: #6b7280; }
            
            .status-overview {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .status-card {
                background: white;
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            }
            
            .status-card h3 {
                font-size: 2rem;
                color: #4f46e5;
                margin-bottom: 5px;
            }
            
            .status-card p {
                color: #6b7280;
                font-size: 14px;
            }
            
            .upcoming-executions {
                background: linear-gradient(135deg, #10b981, #059669);
                color: white;
                border-radius: 12px;
                padding: 20px;
                margin-top: 20px;
            }
            
            .upcoming-executions h3 {
                margin-bottom: 15px;
                font-size: 1.3rem;
            }
            
            .execution-item {
                padding: 10px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            }
            .execution-item:last-child { border-bottom: none; }
            
            .loading {
                text-align: center;
                padding: 40px;
                color: #6b7280;
            }
            
            .error {
                background: #fee2e2;
                color: #dc2626;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
            }
            
            .hidden { display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📅 Experiment Scheduling</h1>
                <p>Automated time-based experiment execution and management</p>
            </div>
            
            <div class="status-overview" id="statusOverview">
                <div class="loading">Loading status...</div>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h2>🆕 Create Schedule</h2>
                    <form id="scheduleForm">
                        <div class="form-group">
                            <label for="name">Schedule Name</label>
                            <input type="text" id="name" name="name" required>
                        </div>
                        <div class="form-group">
                            <label for="description">Description</label>
                            <textarea id="description" name="description" rows="2"></textarea>
                        </div>
                        <div class="form-group">
                            <label for="schedule_type">Schedule Type</label>
                            <select id="schedule_type" name="schedule_type" required>
                                <option value="once">Run Once</option>
                                <option value="daily">Daily</option>
                                <option value="weekly">Weekly</option>
                                <option value="interval">Interval</option>
                            </select>
                        </div>
                        
                        <div id="timeFields" class="hidden">
                            <div class="form-group">
                                <label for="start_datetime">Start Date/Time</label>
                                <input type="datetime-local" id="start_datetime" name="start_datetime">
                            </div>
                            <div class="form-group">
                                <label for="time_of_day">Time of Day (HH:MM)</label>
                                <input type="time" id="time_of_day" name="time_of_day">
                            </div>
                        </div>
                        
                        <div id="intervalFields" class="hidden">
                            <div class="form-group">
                                <label for="interval_minutes">Interval (minutes)</label>
                                <input type="number" id="interval_minutes" name="interval_minutes" min="5" max="10080">
                            </div>
                        </div>
                        
                        <div id="weeklyFields" class="hidden">
                            <div class="form-group">
                                <label for="day_of_week">Day of Week</label>
                                <select id="day_of_week" name="day_of_week">
                                    <option value="0">Monday</option>
                                    <option value="1">Tuesday</option>
                                    <option value="2">Wednesday</option>
                                    <option value="3">Thursday</option>
                                    <option value="4">Friday</option>
                                    <option value="5">Saturday</option>
                                    <option value="6">Sunday</option>
                                </select>
                            </div>
                        </div>
                        
                        <h3 style="margin: 20px 0 10px 0;">Experiment Configuration</h3>
                        <div class="form-group">
                            <label for="exp_name">Experiment Name</label>
                            <input type="text" id="exp_name" name="exp_name" value="Scheduled Experiment" required>
                        </div>
                        <div class="form-group">
                            <label for="population_size">Population Size</label>
                            <input type="number" id="population_size" name="population_size" value="20" min="10" max="1000">
                        </div>
                        <div class="form-group">
                            <label for="generations">Generations</label>
                            <input type="number" id="generations" name="generations" value="50" min="1" max="1000">
                        </div>
                        
                        <button type="submit" class="btn">Create Schedule</button>
                    </form>
                </div>
                
                <div class="card">
                    <h2>📋 Active Schedules</h2>
                    <div id="schedulesList">
                        <div class="loading">Loading schedules...</div>
                    </div>
                </div>
            </div>
            
            <div class="card grid-full">
                <h2>⏰ Upcoming Executions</h2>
                <div id="upcomingExecutions">
                    <div class="loading">Loading upcoming executions...</div>
                </div>
            </div>
        </div>
        
        <script>
            let schedules = [];
            
            // Load scheduler status
            async function loadStatus() {
                try {
                    const response = await fetch('/api/scheduling/status');
                    const status = await response.json();
                    
                    document.getElementById('statusOverview').innerHTML = `
                        <div class="status-card">
                            <h3>${status.total_schedules}</h3>
                            <p>Total Schedules</p>
                        </div>
                        <div class="status-card">
                            <h3>${status.active_schedules}</h3>
                            <p>Active Schedules</p>
                        </div>
                        <div class="status-card">
                            <h3>${status.running_executions}</h3>
                            <p>Running Now</p>
                        </div>
                        <div class="status-card">
                            <h3>${status.total_executions}</h3>
                            <p>Total Executions</p>
                        </div>
                    `;
                    
                    // Update upcoming executions
                    if (status.next_executions && status.next_executions.length > 0) {
                        document.getElementById('upcomingExecutions').innerHTML = `
                            <div class="upcoming-executions">
                                <h3>Next Scheduled Executions</h3>
                                ${status.next_executions.map(exec => `
                                    <div class="execution-item">
                                        <strong>${exec.schedule_name}</strong><br>
                                        <small>${new Date(exec.next_execution_time).toLocaleString()} (${exec.schedule_type})</small>
                                    </div>
                                `).join('')}
                            </div>
                        `;
                    } else {
                        document.getElementById('upcomingExecutions').innerHTML = 
                            '<p style="text-align: center; color: #6b7280; padding: 20px;">No upcoming executions scheduled</p>';
                    }
                    
                } catch (error) {
                    document.getElementById('statusOverview').innerHTML = 
                        '<div class="error">Failed to load status</div>';
                }
            }
            
            // Load schedules
            async function loadSchedules() {
                try {
                    const response = await fetch('/api/scheduling/schedules');
                    const data = await response.json();
                    schedules = data.schedules;
                    
                    const container = document.getElementById('schedulesList');
                    
                    if (schedules.length === 0) {
                        container.innerHTML = '<p style="text-align: center; color: #6b7280;">No schedules created yet</p>';
                        return;
                    }
                    
                    container.innerHTML = schedules.map(schedule => `
                        <div class="schedule-item">
                            <h3>${schedule.name}</h3>
                            <p><strong>Type:</strong> ${schedule.schedule_type}</p>
                            <p><strong>Status:</strong> <span class="status ${schedule.enabled ? 'running' : 'cancelled'}">${schedule.enabled ? 'enabled' : 'disabled'}</span></p>
                            <p><strong>Created:</strong> ${new Date(schedule.created_at).toLocaleString()}</p>
                            <div style="margin-top: 10px;">
                                <button class="btn btn-small" onclick="viewSchedule('${schedule.schedule_id}')">View Details</button>
                                <button class="btn btn-small ${schedule.enabled ? 'btn-danger' : 'btn-success'}" 
                                        onclick="toggleSchedule('${schedule.schedule_id}', ${schedule.enabled})">
                                    ${schedule.enabled ? 'Disable' : 'Enable'}
                                </button>
                                <button class="btn btn-small btn-danger" onclick="deleteSchedule('${schedule.schedule_id}')">Delete</button>
                            </div>
                        </div>
                    `).join('');
                    
                } catch (error) {
                    document.getElementById('schedulesList').innerHTML = 
                        '<div class="error">Failed to load schedules</div>';
                }
            }
            
            // Handle schedule type change
            document.getElementById('schedule_type').onchange = function() {
                const type = this.value;
                const timeFields = document.getElementById('timeFields');
                const intervalFields = document.getElementById('intervalFields');
                const weeklyFields = document.getElementById('weeklyFields');
                
                // Hide all conditional fields
                timeFields.classList.add('hidden');
                intervalFields.classList.add('hidden');
                weeklyFields.classList.add('hidden');
                
                // Show relevant fields
                if (type === 'once') {
                    timeFields.classList.remove('hidden');
                } else if (type === 'daily') {
                    timeFields.classList.remove('hidden');
                } else if (type === 'interval') {
                    intervalFields.classList.remove('hidden');
                } else if (type === 'weekly') {
                    timeFields.classList.remove('hidden');
                    weeklyFields.classList.remove('hidden');
                }
            };
            
            // Create schedule form submission
            document.getElementById('scheduleForm').onsubmit = async function(e) {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const scheduleType = formData.get('schedule_type');
                
                const payload = {
                    name: formData.get('name'),
                    description: formData.get('description'),
                    schedule_type: scheduleType,
                    experiment_config: {
                        name: formData.get('exp_name'),
                        population_size: parseInt(formData.get('population_size')),
                        generations: parseInt(formData.get('generations')),
                        mutation_rate: 0.1,
                        crossover_rate: 0.8,
                        elite_size: 3
                    }
                };
                
                // Add schedule-specific fields
                if (scheduleType === 'once') {
                    payload.start_datetime = formData.get('start_datetime');
                } else if (scheduleType === 'daily') {
                    payload.time_of_day = formData.get('time_of_day');
                } else if (scheduleType === 'interval') {
                    payload.interval_minutes = parseInt(formData.get('interval_minutes'));
                } else if (scheduleType === 'weekly') {
                    payload.day_of_week = parseInt(formData.get('day_of_week'));
                    payload.time_of_day = formData.get('time_of_day');
                }
                
                try {
                    const response = await fetch('/api/scheduling/schedules', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    
                    if (response.ok) {
                        alert('Schedule created successfully!');
                        e.target.reset();
                        loadSchedules();
                        loadStatus();
                    } else {
                        const error = await response.json();
                        alert('Failed to create schedule: ' + error.detail);
                    }
                } catch (error) {
                    alert('Error creating schedule: ' + error.message);
                }
            };
            
            // Toggle schedule enable/disable
            async function toggleSchedule(scheduleId, currentlyEnabled) {
                try {
                    const action = currentlyEnabled ? 'disable' : 'enable';
                    const response = await fetch(`/api/scheduling/schedules/${scheduleId}/${action}`, {
                        method: 'POST'
                    });
                    
                    if (response.ok) {
                        loadSchedules();
                        loadStatus();
                    } else {
                        alert('Failed to ' + action + ' schedule');
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            // Delete schedule
            async function deleteSchedule(scheduleId) {
                if (!confirm('Are you sure you want to delete this schedule?')) {
                    return;
                }
                
                try {
                    const response = await fetch(`/api/scheduling/schedules/${scheduleId}`, {
                        method: 'DELETE'
                    });
                    
                    if (response.ok) {
                        loadSchedules();
                        loadStatus();
                    } else {
                        alert('Failed to delete schedule');
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            // View schedule details
            function viewSchedule(scheduleId) {
                window.open(`/api/scheduling/schedules/${scheduleId}`, '_blank');
            }
            
            // Initialize
            loadStatus();
            loadSchedules();
            
            // Refresh every 30 seconds
            setInterval(() => {
                loadStatus();
            }, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
