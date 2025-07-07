"""
Pydantic models for the web API.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class User(BaseModel):
    """User model for authentication."""
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="Email address")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class ExperimentConfig(BaseModel):
    """Configuration for a new experiment."""
    name: str = Field(..., description="Experiment name")
    population_size: int = Field(100, ge=10, le=1000, description="Population size")
    generations: int = Field(100, ge=1, le=10000, description="Number of generations")
    mutation_rate: float = Field(0.1, ge=0.0, le=1.0, description="Mutation rate")
    crossover_rate: float = Field(0.8, ge=0.0, le=1.0, description="Crossover rate")
    elite_size: int = Field(5, ge=1, description="Elite size")
    grid_size: int = Field(10, ge=5, le=20, description="Snake game grid size")
    max_steps: int = Field(500, ge=100, le=2000, description="Max steps per game")


class ExperimentStatus(BaseModel):
    """Status of a running experiment."""
    id: str
    name: str
    status: str  # 'running', 'paused', 'completed', 'failed'
    current_generation: int
    total_generations: int
    best_fitness: float
    average_fitness: float
    start_time: datetime
    last_update: datetime
    elapsed_time: float  # seconds


class GenerationStats(BaseModel):
    """Statistics for a single generation."""
    generation: int
    best_fitness: float
    average_fitness: float
    std_fitness: float
    diversity_score: float
    timestamp: datetime


class AgentInfo(BaseModel):
    """Information about a single agent."""
    id: str
    fitness: float
    genome_size: int
    game_score: int
    steps_survived: int
    generation: int


class GameState(BaseModel):
    """Current state of a snake game."""
    snake_positions: List[List[int]]
    food_position: List[int]
    score: int
    steps: int
    game_over: bool
    grid_size: int


class ExperimentListItem(BaseModel):
    """Brief experiment info for listing."""
    id: str
    name: str
    status: str
    current_generation: int
    total_generations: int
    best_fitness: float
    start_time: datetime


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str  # 'generation_update', 'experiment_status', 'agent_update'
    data: Dict[str, Any]
    timestamp: datetime


class CommandRequest(BaseModel):
    """Request to execute a command."""
    command: str  # 'start', 'pause', 'resume', 'stop'
    experiment_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class CommandResponse(BaseModel):
    """Response from command execution."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


# Phase 4E: Mobile & PWA Models

class MobileDeviceInfo(BaseModel):
    """Mobile device information for optimization."""
    device_type: str = Field(..., description="Device type: mobile, tablet, desktop")
    screen_width: int = Field(..., description="Screen width in pixels")
    screen_height: int = Field(..., description="Screen height in pixels")
    user_agent: str = Field(..., description="Browser user agent string")
    is_mobile: bool = Field(..., description="Whether device is mobile")
    supports_touch: bool = Field(True, description="Whether device supports touch")
    network_type: Optional[str] = Field(None, description="Network connection type")


class MobileExperimentConfig(BaseModel):
    """Mobile-optimized experiment configuration."""
    name: str = Field(..., description="Experiment name")
    population_size: int = Field(50, ge=5, le=200, description="Smaller population for mobile")
    generations: int = Field(50, ge=1, le=500, description="Fewer generations for mobile")
    mutation_rate: float = Field(0.1, ge=0.0, le=1.0, description="Mutation rate")
    crossover_rate: float = Field(0.8, ge=0.0, le=1.0, description="Crossover rate")
    elite_size: int = Field(3, ge=1, description="Smaller elite size for mobile")
    grid_size: int = Field(8, ge=5, le=12, description="Smaller grid for mobile")
    max_steps: int = Field(200, ge=50, le=500, description="Fewer steps for mobile")
    mobile_optimized: bool = Field(True, description="Mobile optimization enabled")


class PWAManifest(BaseModel):
    """Progressive Web App manifest data."""
    name: str = Field("Genetic Coil Arena", description="App name")
    short_name: str = Field("CoilArena", description="Short app name")
    description: str = Field("Evolutionary computing platform", description="App description")
    theme_color: str = Field("#667eea", description="Theme color")
    background_color: str = Field("#ffffff", description="Background color")
    display: str = Field("standalone", description="Display mode")
    orientation: str = Field("portrait", description="Preferred orientation")
    start_url: str = Field("/", description="Start URL")
    scope: str = Field("/", description="App scope")


class MobileVisualizationConfig(BaseModel):
    """Mobile-optimized visualization settings."""
    reduced_particles: bool = Field(True, description="Reduce particle count for performance")
    simplified_graphics: bool = Field(True, description="Use simplified graphics")
    animation_fps: int = Field(30, ge=15, le=60, description="Target FPS for animations")
    max_agents_display: int = Field(10, ge=5, le=50, description="Max agents to display")
    touch_controls: bool = Field(True, description="Enable touch controls")
    gesture_navigation: bool = Field(True, description="Enable gesture navigation")


class NotificationConfig(BaseModel):
    """Push notification configuration."""
    enabled: bool = Field(True, description="Notifications enabled")
    experiment_complete: bool = Field(True, description="Notify on experiment completion")
    milestone_updates: bool = Field(True, description="Notify on generation milestones")
    error_alerts: bool = Field(True, description="Notify on errors")
    badge_updates: bool = Field(True, description="Update app badge")
    vibration: bool = Field(True, description="Enable vibration on mobile")


class MobileUserPreferences(BaseModel):
    """Mobile user preferences and settings."""
    theme: str = Field("auto", description="Theme: light, dark, auto")
    reduced_motion: bool = Field(False, description="Reduce animations for accessibility")
    data_saver: bool = Field(False, description="Enable data saver mode")
    offline_mode: bool = Field(True, description="Allow offline functionality")
    sync_frequency: str = Field("auto", description="Background sync frequency")
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    visualization: MobileVisualizationConfig = Field(default_factory=MobileVisualizationConfig)
