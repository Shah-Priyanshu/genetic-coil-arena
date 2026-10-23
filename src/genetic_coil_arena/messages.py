"""Message-array conventions shared by every layer that touches messages.

This is the one definition of a message channel (``MessageSpec``). It depends on
numpy only, so environments (``envs``), the evaluation harness (``evaluation``), and
later the protocol and runtime layers can all import it without importing each other.

Array conventions:

- discrete messages: int64, shape (B, R, L), symbols in [0, V)
- continuous messages: float32, shape (B, R, D)

B = parallel episodes, R = receivers, L = symbols per message, V = vocabulary size,
D = vector size. A "none" channel carries no messages (payload shape (0,)).
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

DISCRETE = "discrete"
CONTINUOUS = "continuous"
NONE = "none"
MESSAGE_KINDS = (DISCRETE, CONTINUOUS, NONE)


@dataclass(frozen=True)
class MessageSpec:
    """Shape and type of one sender-to-receivers message channel.

    Args:
        kind: "discrete", "continuous" or "none"
        length: Symbols per discrete message (L)
        vocab_size: Discrete vocabulary size (V); ignored for other kinds
        dim: Continuous message size (D)
        interval: The sender may send every ``interval`` steps (k)
        n_receivers: Receivers per episode (R), when fixed; ``None`` leaves R free
    """

    kind: str = DISCRETE
    length: int = 1
    vocab_size: int = 8
    dim: int = 0
    interval: int = 1
    n_receivers: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in MESSAGE_KINDS:
            raise ValueError(
                f"Unknown message kind {self.kind!r}; expected one of {MESSAGE_KINDS}"
            )
        if self.kind == DISCRETE and (self.length < 1 or self.vocab_size < 2):
            raise ValueError("Discrete messages need length >= 1 and vocab_size >= 2")
        if self.kind == CONTINUOUS and self.dim < 1:
            raise ValueError("Continuous messages need dim >= 1")
        if self.interval < 1:
            raise ValueError("Message interval must be >= 1")
        if self.n_receivers is not None and self.n_receivers < 1:
            raise ValueError("n_receivers must be >= 1")

    @classmethod
    def discrete(
        cls, length: int, vocab_size: int, n_receivers: Optional[int] = None
    ) -> "MessageSpec":
        return cls(DISCRETE, length, vocab_size, n_receivers=n_receivers)

    @classmethod
    def continuous(cls, dim: int, n_receivers: Optional[int] = None) -> "MessageSpec":
        return cls(CONTINUOUS, dim=dim, n_receivers=n_receivers)

    @property
    def bits(self) -> float:
        """Capacity of one message in bits: L log2 V (inf if continuous, 0 if none)."""
        if self.kind == DISCRETE:
            return self.length * math.log2(self.vocab_size)
        if self.kind == CONTINUOUS:
            return math.inf
        return 0.0

    capacity_bits = bits

    @property
    def payload_shape(self) -> tuple[int, ...]:
        """Per-receiver payload shape: (L,), (D,) or (0,)."""
        if self.kind == DISCRETE:
            return (self.length,)
        if self.kind == CONTINUOUS:
            return (self.dim,)
        return (0,)

    @property
    def dtype(self) -> Any:
        """Payload dtype."""
        return np.float32 if self.kind == CONTINUOUS else np.int64

    def _receivers(self, receivers: Optional[int]) -> Optional[int]:
        return self.n_receivers if receivers is None else receivers

    def zeros(self, batch_size: int, receivers: Optional[int] = None) -> np.ndarray:
        """All-zero message batch of shape (B, R, ...)."""
        r = self._receivers(receivers)
        if r is None:
            raise ValueError("Pass receivers or set n_receivers")
        return np.zeros((batch_size, r) + self.payload_shape, dtype=self.dtype)

    def validate(
        self,
        messages: Any,
        batch_size: Optional[int] = None,
        receivers: Optional[int] = None,
    ) -> np.ndarray:
        """Check a message batch and return it with the canonical dtype.

        ``batch_size`` and ``receivers`` (default ``n_receivers``) are checked when
        given. Any integer dtype is accepted for discrete channels and any float dtype
        for continuous ones; the result is int64 / float32.

        Raises:
            ValueError: On a "none" channel, a wrong shape or dtype kind, or an
                out-of-vocabulary symbol
        """
        if self.kind == NONE:
            raise ValueError("This channel carries no messages")
        arr = np.asarray(messages)
        r = self._receivers(receivers)
        expected = (
            batch_size if batch_size is not None else -1,
            r if r is not None else -1,
        ) + self.payload_shape
        if arr.ndim != len(expected) or any(
            e >= 0 and a != e for a, e in zip(arr.shape, expected)
        ):
            shown = tuple("B" if e == -1 else e for e in expected[:1]) + tuple(
                "R" if e == -1 else e for e in expected[1:2]
            )
            raise ValueError(
                f"Messages must have shape {shown + self.payload_shape}, "
                f"got {arr.shape}"
            )
        if self.kind == DISCRETE:
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(f"Discrete messages must be integers, got {arr.dtype}")
            if arr.size and (arr.min() < 0 or arr.max() >= self.vocab_size):
                raise ValueError(f"Discrete symbols must lie in [0, {self.vocab_size})")
            return arr.astype(np.int64, copy=False)
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError(f"Continuous messages must be floats, got {arr.dtype}")
        return arr.astype(np.float32, copy=False)


def infer_kind(messages: Any) -> str:
    """Channel kind implied by the array dtype."""
    dtype = np.asarray(messages).dtype
    if np.issubdtype(dtype, np.integer):
        return DISCRETE
    if np.issubdtype(dtype, np.floating):
        return CONTINUOUS
    raise TypeError(f"Cannot infer message kind from dtype {dtype}")
