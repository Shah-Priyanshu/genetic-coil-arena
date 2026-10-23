"""Batched multi-agent environments behind one domain-agnostic API (``base``).

Each domain lives in its own subpackage (``squad`` is the first). Protocol, specialist
and command-centre code should depend on ``envs.base`` only, never on a domain package.
"""

from .base import EnvSpec, MessageSpec, MultiAgentEnv, Observations, StepResult

__all__ = ["EnvSpec", "MessageSpec", "MultiAgentEnv", "Observations", "StepResult"]
