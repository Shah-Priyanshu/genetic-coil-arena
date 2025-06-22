"""
Pydantic models for the web API.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


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
