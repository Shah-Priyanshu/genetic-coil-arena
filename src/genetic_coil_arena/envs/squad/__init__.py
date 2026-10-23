"""Cooperative foraging squad: colour-specialist snakes guided by a command centre."""

from .env import N_ACTIONS, WINDOW_CHANNELS, SquadConfig, SquadEnv
from .policies import (
    CentrePolicy,
    GreedyPolicy,
    HandCodedCentre,
    NoCommPolicy,
    OraclePolicy,
    RandomCentre,
    RandomPolicy,
    SilentCentre,
    SpecialistPolicy,
)
from .runner import measure_throughput, run_episodes

__all__ = [
    "N_ACTIONS",
    "WINDOW_CHANNELS",
    "SquadConfig",
    "SquadEnv",
    "SpecialistPolicy",
    "GreedyPolicy",
    "NoCommPolicy",
    "RandomPolicy",
    "OraclePolicy",
    "CentrePolicy",
    "HandCodedCentre",
    "RandomCentre",
    "SilentCentre",
    "run_episodes",
    "measure_throughput",
]
