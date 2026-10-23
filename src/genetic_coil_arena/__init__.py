"""Genetic Coil Arena: a lab for specialist agents that coordinate through a
learned communication protocol.

The Snake-specific names below are imported lazily, so that importing the
domain-agnostic packages (``protocol``, ``runtime``, ``evaluation``) does not load
``snake_core`` or numba.
"""

import importlib
from typing import Any

__version__ = "0.0.1"
__author__ = "Your Name"
__email__ = "your.email@example.com"

_LAZY = {
    "Config": ".config",
    "EvolutionArena": ".evo.arena",
    "SnakeGame": ".snake_core.game",
    "SnakeAgent": ".snake_core.agent",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
        globals()[name] = value  # Cache: later lookups skip __getattr__
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY))
