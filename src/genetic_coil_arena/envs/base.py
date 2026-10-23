"""Domain-agnostic batched multi-agent environment API.

An environment runs B episodes in parallel. Each episode has N specialists that act
every step and a command centre that does not act but may send one message per
specialist every ``MessageSpec.interval`` steps.

Array conventions (shared by every package that touches messages):

- Discrete messages: int64, shape (B, R, L), values in [0, V)
- Continuous messages: float32, shape (B, R, D)

B = parallel episodes, R = receivers (the N specialists), L = symbols per message,
V = vocabulary size, D = vector size. Messages passed to ``step`` at time t appear in
the specialists' observations returned by that call (time t + 1).

The message direction is the environment's choice. In the squad the centre sends to the
specialists (centre -> specialists); in the plant floor the specialists uplink to the
centre (specialists -> centre). Either way the message array keeps the same shape and
dtype conventions; ``EnvSpec.message`` describes the channel.

This module has no knowledge of any particular domain and imports nothing from the
protocol, runtime or evaluation layers: it deals in plain numpy arrays only. The channel
description ``MessageSpec`` is the shared one from ``genetic_coil_arena.messages``
(re-exported here).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..messages import MESSAGE_KINDS, MessageSpec

__all__ = [
    "MESSAGE_KINDS",
    "ArraySpec",
    "EnvSpec",
    "MessageSpec",
    "MultiAgentEnv",
    "Observations",
    "StepResult",
]

# Shape of one observation component (excluding the batch and specialist axes) and its dtype
ArraySpec = tuple[tuple[int, ...], Any]


@dataclass(frozen=True)
class EnvSpec:
    """Static description of a batched multi-agent environment.

    Args:
        n_specialists: Specialists per episode (N); they are also the message receivers
        n_actions: Discrete actions per specialist
        specialist_obs: Observation components per specialist, shapes exclude (B, N)
        centre_obs: Observation components for the centre, shapes exclude (B,)
        message: The message channel, centre -> specialists (squad) or specialists ->
            centre (plant uplink)
        max_steps: Episode length cap (T)
    """

    n_specialists: int
    n_actions: int
    specialist_obs: dict[str, ArraySpec]
    centre_obs: dict[str, ArraySpec]
    message: MessageSpec
    max_steps: int


@dataclass
class Observations:
    """One batched observation: specialist arrays are (B, N, ...), centre arrays (B, ...)."""

    specialists: dict[str, np.ndarray]
    centre: dict[str, np.ndarray]

    @property
    def batch_size(self) -> int:
        """Number of parallel episodes."""
        first = next(iter(self.centre.values()), None)
        if first is None:
            first = next(iter(self.specialists.values()))
        return int(first.shape[0])


@dataclass
class StepResult:
    """Return value of ``MultiAgentEnv.step``; unpacks as (obs, rewards, dones, info)."""

    obs: Observations
    rewards: np.ndarray  # (B, N) float32
    dones: np.ndarray  # (B,) bool
    info: dict[str, np.ndarray] = field(default_factory=dict)

    def __iter__(self):
        return iter((self.obs, self.rewards, self.dones, self.info))


class MultiAgentEnv(ABC):
    """Batched environment with N acting specialists and a messaging command centre.

    Episodes in a batch run in lockstep; a finished episode ignores further actions and
    returns zero rewards until the next ``reset``.
    """

    spec: EnvSpec

    @abstractmethod
    def reset(self, seeds: Sequence[int]) -> Observations:
        """Start one episode per seed; the batch size is ``len(seeds)``."""

    @abstractmethod
    def step(
        self, actions: np.ndarray, messages: Optional[np.ndarray] = None
    ) -> StepResult:
        """Advance every unfinished episode by one step.

        Args:
            actions: (B, N) integer actions in [0, n_actions)
            messages: (B, N, ...) centre messages per the shared conventions, or None
                for silence. Ignored on steps where the centre may not send.

        Returns:
            StepResult(obs, rewards (B, N), dones (B,), info)
        """

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Number of parallel episodes since the last reset."""

    def validate_actions(self, actions: np.ndarray) -> np.ndarray:
        """Check an action batch and return it as int64."""
        arr = np.asarray(actions)
        expected = (self.batch_size, self.spec.n_specialists)
        if arr.shape != expected:
            raise ValueError(f"Actions must have shape {expected}, got {arr.shape}")
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"Actions must be integers, got {arr.dtype}")
        if arr.size and (arr.min() < 0 or arr.max() >= self.spec.n_actions):
            raise ValueError(f"Actions must lie in [0, {self.spec.n_actions})")
        return arr.astype(np.int64, copy=False)
