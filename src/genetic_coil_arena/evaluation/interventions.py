"""Message interventions for testing whether receivers actually listen.

Array conventions and the channel description ``MessageSpec`` live in the
dependency-free ``genetic_coil_arena.messages`` (re-exported here):

- discrete messages: int64 array of shape (B, R, L), symbols in [0, V)
- continuous messages: float32 array of shape (B, R, D)

B is the number of parallel episodes, R the number of receivers, L the message length
and D the payload dimension. An intervention is a function
``(messages, rng, spec=None, mask=None) -> messages`` that returns a new array of the
same shape and dtype and never modifies its input; ``mask`` is an optional (B, R) bool
array of live receivers, and masked-out entries are left untouched. Environment loops
apply one between send and receive through a ``MessageHook``.
"""

from typing import Callable, Optional, Protocol

import numpy as np

from ..messages import CONTINUOUS, DISCRETE, MessageSpec, infer_kind

__all__ = [
    "ABLATIONS",
    "CONTINUOUS",
    "DISCRETE",
    "INTERVENTIONS",
    "Intervention",
    "InterventionHook",
    "MessageHook",
    "MessageSpec",
    "RecordingHook",
    "apply_hook",
    "default_ablations",
    "get_intervention",
    "infer_kind",
    "no_intervention",
    "random_matched",
    "shuffle_batch",
    "step_rng",
    "swap_receivers",
    "zero_messages",
]

# The default ablation set; ``swap_receivers`` needs at least 2 receivers
ABLATIONS: tuple[str, ...] = (
    "zero",
    "shuffle_batch",
    "swap_receivers",
    "random_matched",
)


def default_ablations(n_receivers: Optional[int] = None) -> tuple[str, ...]:
    """``ABLATIONS`` minus ``swap_receivers`` when there is a single receiver."""
    if n_receivers is not None and n_receivers < 2:
        return tuple(name for name in ABLATIONS if name != "swap_receivers")
    return ABLATIONS


def step_rng(seed: int, step: int = 0, *streams: int) -> np.random.Generator:
    """Generator determined only by (seed, step, *streams), whatever ran before."""
    keys = (seed, step, *streams)
    if any(key < 0 for key in keys):
        raise ValueError(f"Seeds, steps and streams must be non-negative, got {keys}")
    return np.random.default_rng([int(key) for key in keys])


def _checked(
    messages: np.ndarray, spec: Optional[MessageSpec]
) -> tuple[np.ndarray, str]:
    arr = np.asarray(messages)
    if spec is not None:
        return spec.validate(arr), spec.kind
    if arr.ndim != 3:
        raise ValueError(f"Messages must have shape (B, R, L|D), got {arr.shape}")
    return arr, infer_kind(arr)


def _checked_mask(mask: Optional[np.ndarray], arr: np.ndarray) -> Optional[np.ndarray]:
    """Validate a (B, R) live-receiver mask; ``None`` means every entry is live."""
    if mask is None:
        return None
    m = np.asarray(mask, dtype=bool)
    if m.shape != arr.shape[:2]:
        raise ValueError(f"mask must have shape {arr.shape[:2]}, got {m.shape}")
    return m


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation with no fixed points (Sattolo's algorithm, a single cycle)."""
    perm = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def no_intervention(
    messages: np.ndarray,
    rng: np.random.Generator,
    spec: Optional[MessageSpec] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pass messages through unchanged (a copy)."""
    arr, _ = _checked(messages, spec)
    return arr.copy()


def zero_messages(
    messages: np.ndarray,
    rng: np.random.Generator,
    spec: Optional[MessageSpec] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Replace every live message with a constant: symbol 0 or the zero vector."""
    arr, _ = _checked(messages, spec)
    m = _checked_mask(mask, arr)
    if m is None:
        return np.zeros_like(arr)
    out = arr.copy()
    out[m] = 0
    return out


def shuffle_batch(
    messages: np.ndarray,
    rng: np.random.Generator,
    spec: Optional[MessageSpec] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Give each episode another episode's messages (no episode keeps its own).

    Keeps the message distribution intact but breaks its link to the receiver's
    situation. Without a mask whole episodes are deranged. With a (B, R) live mask,
    each receiver r gets the message sent to receiver r in another episode where r is
    live; masked entries are left as they are, and a receiver live in only one episode
    keeps its message.
    """
    arr, _ = _checked(messages, spec)
    m = _checked_mask(mask, arr)
    if m is None:
        if arr.shape[0] < 2:
            raise ValueError("shuffle_batch needs at least 2 episodes")
        return arr[_derangement(arr.shape[0], rng)].copy()

    out = arr.copy()
    for r in range(arr.shape[1]):
        live = np.flatnonzero(m[:, r])
        if live.size >= 2:
            out[live, r] = arr[live[_derangement(live.size, rng)], r]
    return out


def swap_receivers(
    messages: np.ndarray,
    rng: np.random.Generator,
    spec: Optional[MessageSpec] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Deliver each receiver's message to a different receiver (same episode).

    Without a mask one derangement of the receivers is applied to every episode. With a
    (B, R) live mask, messages move only among live receivers: episodes with the same
    set of live receivers share one derangement of that set, and an episode with fewer
    than two live receivers is left unchanged.
    """
    arr, _ = _checked(messages, spec)
    if arr.shape[1] < 2:
        raise ValueError(
            "swap_receivers needs at least 2 receivers; give the task a "
            "TaskSpec.message with n_receivers=1 to drop it from the default ablations"
        )
    m = _checked_mask(mask, arr)
    if m is None:
        return arr[:, _derangement(arr.shape[1], rng)].copy()

    out = arr.copy()
    patterns, groups = np.unique(m, axis=0, return_inverse=True)
    groups = np.asarray(groups).reshape(-1)
    for g, pattern in enumerate(patterns):
        live = np.flatnonzero(pattern)
        if live.size < 2:
            continue
        rows = np.flatnonzero(groups == g)
        src = live[_derangement(live.size, rng)]
        out[rows[:, None], live[None, :]] = arr[rows[:, None], src[None, :]]
    return out


def random_matched(
    messages: np.ndarray,
    rng: np.random.Generator,
    spec: Optional[MessageSpec] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Random messages with the same per-(receiver, position) marginals as the batch.

    Discrete: each symbol is drawn independently from the empirical distribution at its
    (receiver, position), so per-position entropy is preserved while any dependence on
    the situation (and between positions) is destroyed. Continuous: Gaussian noise with
    the batch mean and standard deviation of each (receiver, dimension). With a (B, R)
    live mask, marginals come from live entries only and only live entries are replaced.
    """
    arr, kind = _checked(messages, spec)
    m = _checked_mask(mask, arr)
    if m is None:
        if kind == CONTINUOUS:
            mean = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True)
            return (mean + std * rng.standard_normal(arr.shape)).astype(arr.dtype)
        m = np.ones(arr.shape[:2], dtype=bool)

    out = arr.copy()
    for r in range(arr.shape[1]):
        live = np.flatnonzero(m[:, r])
        if live.size == 0:
            continue
        values = arr[live, r]  # (n_live, L|D)
        if kind == CONTINUOUS:
            noise = rng.standard_normal(values.shape)
            out[live, r] = (values.mean(axis=0) + values.std(axis=0) * noise).astype(
                arr.dtype
            )
            continue
        for pos in range(arr.shape[2]):
            symbols, counts = np.unique(values[:, pos], return_counts=True)
            out[live, r, pos] = rng.choice(
                symbols, size=live.size, p=counts / counts.sum()
            )
    return out


Intervention = Callable[..., np.ndarray]
"""``(messages, rng, spec=None, mask=None) -> messages``"""

INTERVENTIONS: dict[str, Intervention] = {
    "none": no_intervention,
    "zero": zero_messages,
    "shuffle_batch": shuffle_batch,
    "swap_receivers": swap_receivers,
    "random_matched": random_matched,
}


def get_intervention(name: str) -> Intervention:
    """Look up an intervention by name."""
    if name not in INTERVENTIONS:
        raise KeyError(
            f"Unknown intervention {name!r}; expected one of {sorted(INTERVENTIONS)}"
        )
    return INTERVENTIONS[name]


class MessageHook(Protocol):
    """Called by an environment loop on messages between send and receive.

    ``mask`` is an optional (B, R) bool array of live receivers (e.g. alive snakes in
    unfinished episodes); hooks leave masked-out entries alone.
    """

    def __call__(
        self, messages: np.ndarray, step: int = 0, mask: Optional[np.ndarray] = None
    ) -> np.ndarray: ...


class InterventionHook:
    """Applies one named intervention, deterministically for a given seed.

    Each call draws from a generator keyed on (seed, step, stream, call index), so a
    hook replays exactly when the same sequence of calls is made after ``reset()``.
    """

    def __init__(
        self,
        name: str,
        seed: int = 0,
        spec: Optional[MessageSpec] = None,
        stream: int = 0,
    ):
        self.name = name
        self.seed = seed
        self.spec = spec
        self.stream = stream
        self._fn = get_intervention(name)
        self.calls = 0

    def __call__(
        self, messages: np.ndarray, step: int = 0, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        rng = step_rng(self.seed, step, self.stream, self.calls)
        self.calls += 1
        return self._fn(messages, rng, self.spec, mask)

    def reset(self) -> None:
        """Restart the call counter so the same calls replay the same interventions."""
        self.calls = 0

    def __repr__(self) -> str:
        return (
            f"InterventionHook({self.name!r}, seed={self.seed}, stream={self.stream})"
        )


class RecordingHook:
    """Records the messages that pass through it, optionally applying another hook.

    The recorded arrays are copies of the messages as sent (before ``inner`` changes
    them), with the step and live mask of each call, so communication metrics can be
    computed from the same run. The runner composes one around its intervention hook
    when asked to record (``evaluate(..., record=True)``).
    """

    def __init__(self, inner: Optional[MessageHook] = None):
        self.inner = inner
        self.records: list[np.ndarray] = []
        self.steps: list[int] = []
        self.masks: list[Optional[np.ndarray]] = []

    def __call__(
        self, messages: np.ndarray, step: int = 0, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        arr = np.asarray(messages)
        self.records.append(arr.copy())
        self.steps.append(step)
        self.masks.append(None if mask is None else np.array(mask, dtype=bool))
        return apply_hook(self.inner, arr, step, mask)

    def stacked(self, live_only: bool = False) -> np.ndarray:
        """All recorded messages concatenated along the episode axis.

        With ``live_only`` the result is (n, L|D): one row per live receiver per call
        (calls made without a mask count every receiver as live).
        """
        if not self.records:
            raise ValueError("No messages recorded")
        if not live_only:
            return np.concatenate(self.records, axis=0)
        return np.concatenate(
            [
                rec[np.ones(rec.shape[:2], dtype=bool) if m is None else m]
                for rec, m in zip(self.records, self.masks)
            ]
        )

    def reset(self) -> None:
        """Forget recorded messages and reset the inner hook, if it can be reset."""
        self.records.clear()
        self.steps.clear()
        self.masks.clear()
        reset = getattr(self.inner, "reset", None)
        if callable(reset):
            reset()


def apply_hook(
    hook: Optional[MessageHook],
    messages: np.ndarray,
    step: int = 0,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """What an environment loop calls between send and receive; ``None`` is a no-op.

    ``mask`` is passed on only when given, so hooks written as ``(messages, step)``
    keep working on loops that have no live mask.
    """
    if hook is None:
        return messages
    return hook(messages, step) if mask is None else hook(messages, step, mask)
