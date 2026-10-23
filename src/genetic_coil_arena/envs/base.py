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

This module has no knowledge of any particular domain and imports nothing from the
protocol, runtime or evaluation layers: it deals in plain numpy arrays only.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

MESSAGE_KINDS = ("discrete", "continuous", "none")

# Shape of one observation component (excluding the batch and specialist axes) and its dtype
ArraySpec = Tuple[Tuple[int, ...], Any]


@dataclass(frozen=True)
class MessageSpec:
    """Shape of the centre-to-specialist channel.

    Args:
        kind: "discrete", "continuous" or "none"
        length: Symbols per discrete message (L)
        vocab_size: Discrete vocabulary size (V)
        dim: Continuous message size (D)
        interval: The centre may send every ``interval`` steps (k)
    """

    kind: str = "discrete"
    length: int = 1
    vocab_size: int = 8
    dim: int = 0
    interval: int = 1

    def __post_init__(self) -> None:
        if self.kind not in MESSAGE_KINDS:
            raise ValueError(f"Unknown message kind {self.kind!r}; expected one of {MESSAGE_KINDS}")
        if self.kind == "discrete" and (self.length < 1 or self.vocab_size < 2):
            raise ValueError("Discrete messages need length >= 1 and vocab_size >= 2")
        if self.kind == "continuous" and self.dim < 1:
            raise ValueError("Continuous messages need dim >= 1")
        if self.interval < 1:
            raise ValueError("Message interval must be >= 1")

    @property
    def bits(self) -> float:
        """Channel capacity of one message in bits (inf for continuous, 0 for none)."""
        if self.kind == "discrete":
            return self.length * math.log2(self.vocab_size)
        if self.kind == "continuous":
            return math.inf
        return 0.0

    @property
    def payload_shape(self) -> Tuple[int, ...]:
        """Per-receiver payload shape: (L,), (D,) or (0,)."""
        if self.kind == "discrete":
            return (self.length,)
        if self.kind == "continuous":
            return (self.dim,)
        return (0,)

    @property
    def dtype(self) -> Any:
        """Payload dtype."""
        return np.float32 if self.kind == "continuous" else np.int64

    def zeros(self, batch_size: int, receivers: int) -> np.ndarray:
        """All-zero message batch of shape (B, R, ...)."""
        return np.zeros((batch_size, receivers) + self.payload_shape, dtype=self.dtype)

    def validate(self, messages: np.ndarray, batch_size: int, receivers: int) -> np.ndarray:
        """Check a message batch against this spec and return it with the canonical dtype.

        Raises:
            ValueError: On a wrong shape, dtype kind or out-of-vocabulary symbol
        """
        if self.kind == "none":
            raise ValueError("This channel carries no messages; pass messages=None")
        arr = np.asarray(messages)
        expected = (batch_size, receivers) + self.payload_shape
        if arr.shape != expected:
            raise ValueError(f"Messages must have shape {expected}, got {arr.shape}")
        if self.kind == "discrete":
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(f"Discrete messages must be integers, got {arr.dtype}")
            if arr.size and (arr.min() < 0 or arr.max() >= self.vocab_size):
                raise ValueError(f"Discrete symbols must lie in [0, {self.vocab_size})")
            return arr.astype(np.int64, copy=False)
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(f"Continuous messages must be floats, got {arr.dtype}")
        return arr.astype(np.float32, copy=False)


@dataclass(frozen=True)
class EnvSpec:
    """Static description of a batched multi-agent environment.

    Args:
        n_specialists: Specialists per episode (N); they are also the message receivers
        n_actions: Discrete actions per specialist
        specialist_obs: Observation components per specialist, shapes exclude (B, N)
        centre_obs: Observation components for the centre, shapes exclude (B,)
        message: Centre-to-specialist channel
        max_steps: Episode length cap (T)
    """

    n_specialists: int
    n_actions: int
    specialist_obs: Dict[str, ArraySpec]
    centre_obs: Dict[str, ArraySpec]
    message: MessageSpec
    max_steps: int


@dataclass
class Observations:
    """One batched observation: specialist arrays are (B, N, ...), centre arrays (B, ...)."""

    specialists: Dict[str, np.ndarray]
    centre: Dict[str, np.ndarray]

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
    info: Dict[str, np.ndarray] = field(default_factory=dict)

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
    def step(self, actions: np.ndarray,
             messages: Optional[np.ndarray] = None) -> StepResult:
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
