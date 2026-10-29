"""Drone swarm lab (R14): heterogeneous drones, drifting positions and a lossy byte radio.

See ``env.py`` for the model and ``docs/directions/drone_swarm.md`` for the direction.
"""

from .env import MOVES, N_ACTIONS, ROLES, SwarmConfig, SwarmEnv
from .policies import HandCodec, ScriptedSwarm, heading_action, run_swarm

__all__ = [
    "MOVES",
    "N_ACTIONS",
    "ROLES",
    "HandCodec",
    "ScriptedSwarm",
    "SwarmConfig",
    "SwarmEnv",
    "heading_action",
    "run_swarm",
]
