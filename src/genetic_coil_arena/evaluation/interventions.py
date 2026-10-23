"""Message interventions for testing whether receivers actually listen.

Array conventions shared by every environment and protocol in the project:

- discrete messages: int64 array of shape (B, R, L), symbols in [0, V)
- continuous messages: float32 array of shape (B, R, D)

B is the number of parallel episodes, R the number of receivers, L the message length
and D the payload dimension. An intervention is a function
``(messages, rng, spec) -> messages`` that returns a new array of the same shape and
dtype and never modifies its input. Environment loops apply one between send and
receive through a ``MessageHook``.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, Tuple

import numpy as np

DISCRETE = "discrete"
CONTINUOUS = "continuous"

# Interventions that work for any number of receivers; used as the default ablation set
ABLATIONS: Tuple[str, ...] = ("zero", "shuffle_batch", "random_matched")


@dataclass(frozen=True)
class MessageSpec:
    """Shape and type of one message channel."""

    kind: str
    n_receivers: int
    length: int  # L for discrete messages, D for continuous ones
    vocab_size: Optional[int] = None  # V; discrete only

    def __post_init__(self) -> None:
        if self.kind not in (DISCRETE, CONTINUOUS):
            raise ValueError(f"kind must be {DISCRETE!r} or {CONTINUOUS!r}, got {self.kind!r}")
        if self.n_receivers < 1 or self.length < 1:
            raise ValueError("n_receivers and length must be positive")
        if self.kind == DISCRETE and (self.vocab_size is None or self.vocab_size < 1):
            raise ValueError("Discrete channels need vocab_size >= 1")
        if self.kind == CONTINUOUS and self.vocab_size is not None:
            raise ValueError("Continuous channels have no vocab_size")

    @classmethod
    def discrete(cls, n_receivers: int, length: int, vocab_size: int) -> "MessageSpec":
        return cls(DISCRETE, n_receivers, length, vocab_size)

    @classmethod
    def continuous(cls, n_receivers: int, dim: int) -> "MessageSpec":
        return cls(CONTINUOUS, n_receivers, dim)

    @property
    def capacity_bits(self) -> float:
        """Bits one receiver can get per message: L * log2(V); infinite if continuous."""
        if self.kind == CONTINUOUS:
            return float("inf")
        return float(self.length * np.log2(self.vocab_size))

    def validate(self, messages: np.ndarray) -> np.ndarray:
        """Check shape, dtype and symbol range; returns the array unchanged."""
        arr = np.asarray(messages)
        if arr.ndim != 3 or arr.shape[1:] != (self.n_receivers, self.length):
            raise ValueError(f"Expected messages of shape (B, {self.n_receivers}, {self.length}), "
                             f"got {arr.shape}")
        if self.kind == DISCRETE:
            if arr.dtype != np.int64:
                raise TypeError(f"Discrete messages must be int64, got {arr.dtype}")
            if arr.size and (arr.min() < 0 or arr.max() >= self.vocab_size):
                raise ValueError(f"Symbols must lie in [0, {self.vocab_size})")
        elif arr.dtype != np.float32:
            raise TypeError(f"Continuous messages must be float32, got {arr.dtype}")
        return arr

    def empty(self, batch: int) -> np.ndarray:
        """All-zero messages for ``batch`` episodes."""
        dtype = np.int64 if self.kind == DISCRETE else np.float32
        return np.zeros((batch, self.n_receivers, self.length), dtype=dtype)


def infer_kind(messages: np.ndarray) -> str:
    """Channel kind implied by the array dtype."""
    dtype = np.asarray(messages).dtype
    if np.issubdtype(dtype, np.integer):
        return DISCRETE
    if np.issubdtype(dtype, np.floating):
        return CONTINUOUS
    raise TypeError(f"Cannot infer message kind from dtype {dtype}")


def step_rng(seed: int, step: int = 0, *streams: int) -> np.random.Generator:
    """Generator determined only by (seed, step, *streams), whatever ran before."""
    keys = (seed, step, *streams)
    if any(key < 0 for key in keys):
        raise ValueError(f"Seeds, steps and streams must be non-negative, got {keys}")
    return np.random.default_rng([int(key) for key in keys])


def _checked(messages: np.ndarray, spec: Optional[MessageSpec]) -> Tuple[np.ndarray, str]:
    arr = np.asarray(messages)
    if spec is not None:
        return spec.validate(arr), spec.kind
    if arr.ndim != 3:
        raise ValueError(f"Messages must have shape (B, R, L|D), got {arr.shape}")
    return arr, infer_kind(arr)


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation with no fixed points (Sattolo's algorithm, a single cycle)."""
    perm = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def no_intervention(messages: np.ndarray, rng: np.random.Generator,
                    spec: Optional[MessageSpec] = None) -> np.ndarray:
    """Pass messages through unchanged (a copy)."""
    arr, _ = _checked(messages, spec)
    return arr.copy()


def zero_messages(messages: np.ndarray, rng: np.random.Generator,
                  spec: Optional[MessageSpec] = None) -> np.ndarray:
    """Replace every message with a constant: symbol 0 (discrete) or the zero vector."""
    arr, _ = _checked(messages, spec)
    return np.zeros_like(arr)


def shuffle_batch(messages: np.ndarray, rng: np.random.Generator,
                  spec: Optional[MessageSpec] = None) -> np.ndarray:
    """Give each episode another episode's messages (no episode keeps its own).

    Keeps the message distribution intact but breaks its link to the receiver's situation.
    """
    arr, _ = _checked(messages, spec)
    if arr.shape[0] < 2:
        raise ValueError("shuffle_batch needs at least 2 episodes")
    return arr[_derangement(arr.shape[0], rng)].copy()


def swap_receivers(messages: np.ndarray, rng: np.random.Generator,
                   spec: Optional[MessageSpec] = None) -> np.ndarray:
    """Deliver each receiver's message to a different receiver (same episode)."""
    arr, _ = _checked(messages, spec)
    if arr.shape[1] < 2:
        raise ValueError("swap_receivers needs at least 2 receivers")
    return arr[:, _derangement(arr.shape[1], rng)].copy()


def random_matched(messages: np.ndarray, rng: np.random.Generator,
                   spec: Optional[MessageSpec] = None) -> np.ndarray:
    """Random messages with the same per-(receiver, position) marginals as the batch.

    Discrete: each symbol is drawn independently from the empirical distribution at its
    (receiver, position), so per-position entropy is preserved while any dependence on
    the situation (and between positions) is destroyed. Continuous: Gaussian noise with
    the batch mean and standard deviation of each (receiver, dimension).
    """
    arr, kind = _checked(messages, spec)
    if kind == CONTINUOUS:
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        return (mean + std * rng.standard_normal(arr.shape)).astype(arr.dtype)

    batch, receivers, length = arr.shape
    out = np.empty_like(arr)
    for r in range(receivers):
        for pos in range(length):
            symbols, counts = np.unique(arr[:, r, pos], return_counts=True)
            out[:, r, pos] = rng.choice(symbols, size=batch, p=counts / counts.sum())
    return out


Intervention = Callable[[np.ndarray, np.random.Generator, Optional[MessageSpec]], np.ndarray]

INTERVENTIONS: Dict[str, Intervention] = {
    "none": no_intervention,
    "zero": zero_messages,
    "shuffle_batch": shuffle_batch,
    "swap_receivers": swap_receivers,
    "random_matched": random_matched,
}


def get_intervention(name: str) -> Intervention:
    """Look up an intervention by name."""
    if name not in INTERVENTIONS:
        raise KeyError(f"Unknown intervention {name!r}; expected one of {sorted(INTERVENTIONS)}")
    return INTERVENTIONS[name]


class MessageHook(Protocol):
    """Called by an environment loop on messages between send and receive."""

    def __call__(self, messages: np.ndarray, step: int = 0) -> np.ndarray:
        ...


class InterventionHook:
    """Applies one named intervention, deterministically for a given seed.

    Each call draws from a generator keyed on (seed, step, stream, call index), so a
    hook replays exactly when the same sequence of calls is made after ``reset()``.
    """

    def __init__(self, name: str, seed: int = 0, spec: Optional[MessageSpec] = None,
                 stream: int = 0):
        self.name = name
        self.seed = seed
        self.spec = spec
        self.stream = stream
        self._fn = get_intervention(name)
        self.calls = 0

    def __call__(self, messages: np.ndarray, step: int = 0) -> np.ndarray:
        rng = step_rng(self.seed, step, self.stream, self.calls)
        self.calls += 1
        return self._fn(messages, rng, self.spec)

    def reset(self) -> None:
        """Restart the call counter so the same calls replay the same interventions."""
        self.calls = 0

    def __repr__(self) -> str:
        return f"InterventionHook({self.name!r}, seed={self.seed}, stream={self.stream})"
