"""Genetic Coil Arena - Evolutionary Snake AI Simulation.

A production-grade, research-friendly Snake variant where autonomous agents
evolve over generations using genetic algorithms.
"""

__version__ = "0.0.1"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config import Config
from .evo.arena import EvolutionArena
from .snake_core.game import SnakeGame
from .snake_core.agent import SnakeAgent

__all__ = [
    "Config",
    "EvolutionArena", 
    "SnakeGame",
    "SnakeAgent",
]
