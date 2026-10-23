"""Scripted squad policies: specialists, command centres and a full-view oracle.

Specialists map ``Observations`` to (B, N) actions (``act``, sampled) and to the
exact action distribution they sample from (``probs``, (B, N, 3)). Centres map
observations to a (B, N, L) message batch; the silent centre sends an explicit all-zero
batch rather than nothing, so message hooks see every step. Every policy is seeded per
episode through ``reset`` so a run is reproducible from its seeds.

The hand-coded centre speaks a 3-bit code per receiver, one discrete symbol in [0, 8):
``code = 2 * direction + far``. ``direction`` says where the receiver's nearest
own-colour food lies relative to its heading at send time (0 ahead, 1 right, 2 behind,
3 left, by the dominant axis) and ``far`` is 1 when the Manhattan distance exceeds
``near_distance``. A message sent at step t is read at t + 1, after the receiver's
move at t, so it is one step stale.
"""

from collections.abc import Sequence

import numba  # type: ignore
import numpy as np

from ..base import Observations
from .engine import DIR_DX, DIR_DY, randint, uniform

HAND_CODED_VOCAB = 8

# Egocentric (right, ahead) offsets of the three actions: straight, left, right
_ACTION_RIGHT = np.array([0, -1, 1], dtype=np.int64)
_ACTION_AHEAD = np.array([1, 0, 0], dtype=np.int64)


def _seed_states(seeds: Sequence[int], salt: int) -> np.ndarray:
    """Per-episode SplitMix64 states, decorrelated from the environment's streams."""
    return np.array(
        [(int(s) * 2654435761 + salt) % (1 << 64) for s in seeds], dtype=np.uint64
    )


@numba.njit(cache=True)  # type: ignore
def _choose(
    rng,
    b: int,
    safe,
    target_right: int,
    target_ahead: int,
    has_target: bool,
    explore: float,
) -> int:
    """Safe action closest to the egocentric target, else a mostly-straight wander."""
    best = -1
    best_dist = 1 << 30
    ties = 0
    if has_target:
        for a in range(3):
            if not safe[a]:
                continue
            dist = abs(target_right - _ACTION_RIGHT[a]) + abs(
                target_ahead - _ACTION_AHEAD[a]
            )
            if dist < best_dist:
                best, best_dist, ties = a, dist, 1
            elif dist == best_dist:
                ties += 1
                if randint(rng, b, ties) == 0:
                    best = a
        if best >= 0:
            return best
    n_safe = 0
    for a in range(3):
        if safe[a]:
            n_safe += 1
    if n_safe == 0:
        return 0
    if safe[0] and uniform(rng, b) >= explore:
        return 0
    k = randint(rng, b, n_safe)
    for a in range(3):
        if safe[a]:
            if k == 0:
                return a
            k -= 1
    return 0


@numba.njit(cache=True)  # type: ignore
def _choose_probs(
    out, safe, target_right: int, target_ahead: int, has_target: bool, explore: float
) -> None:
    """Exact distribution of ``_choose`` over the three actions, written into ``out``."""
    for a in range(3):
        out[a] = 0.0
    if has_target:
        best_dist = 1 << 30
        ties = 0
        for a in range(3):
            if not safe[a]:
                continue
            dist = abs(target_right - _ACTION_RIGHT[a]) + abs(
                target_ahead - _ACTION_AHEAD[a]
            )
            if dist < best_dist:
                best_dist, ties = dist, 1
            elif dist == best_dist:
                ties += 1
        if ties > 0:  # Reservoir sampling picks uniformly among the tied best
            for a in range(3):
                if safe[a]:
                    dist = abs(target_right - _ACTION_RIGHT[a]) + abs(
                        target_ahead - _ACTION_AHEAD[a]
                    )
                    if dist == best_dist:
                        out[a] = 1.0 / ties
            return
    n_safe = 0
    for a in range(3):
        if safe[a]:
            n_safe += 1
    if n_safe == 0:
        out[0] = 1.0
        return
    rest = 1.0
    if safe[0]:  # Straight with probability 1 - explore, else uniform over safe
        out[0] = 1.0 - explore
        rest = explore
    for a in range(3):
        if safe[a]:
            out[a] += rest / n_safe


@numba.njit(cache=True)  # type: ignore
def _window_target(windows, messages, use_messages: bool, b: int, i: int, safe):
    """Fill ``safe`` and return the egocentric (right, ahead, has_target) target."""
    size = windows.shape[3]
    r = size // 2
    for a in range(3):
        oy = r - _ACTION_AHEAD[a]
        ox = r + _ACTION_RIGHT[a]
        safe[a] = windows[b, i, 0, oy, ox] == 0 and windows[b, i, 1, oy, ox] == 0
    # nearest visible own-colour food
    best = 1 << 30
    tr = 0
    ta = 0
    for oy in range(size):
        for ox in range(size):
            if windows[b, i, 2, oy, ox] > 0:
                dist = abs(oy - r) + abs(ox - r)
                if dist < best:
                    best, tr, ta = dist, ox - r, r - oy
    has_target = best < (1 << 30)
    if not has_target and use_messages and messages.shape[2] > 0:
        direction = (messages[b, i, 0] // 2) % 4
        far = r + 2
        has_target = True
        if direction == 0:
            tr, ta = 0, far
        elif direction == 1:
            tr, ta = far, 0
        elif direction == 2:
            tr, ta = 0, -far
        else:
            tr, ta = -far, 0
    return tr, ta, has_target


@numba.njit(cache=True)  # type: ignore
def _window_actions(
    out, windows, messages, use_messages: bool, rng, explore: float
) -> None:
    """Greedy own-colour policy on the egocentric window, optionally message-guided."""
    n_batch, n = out.shape
    safe = np.zeros(3, dtype=np.bool_)
    for b in range(n_batch):
        for i in range(n):
            tr, ta, has_target = _window_target(
                windows, messages, use_messages, b, i, safe
            )
            out[b, i] = _choose(rng, b, safe, tr, ta, has_target, explore)


@numba.njit(cache=True)  # type: ignore
def _window_probs(out, windows, messages, use_messages: bool, explore: float) -> None:
    """Exact action distribution of ``_window_actions``, shape (B, N, 3)."""
    n_batch, n = out.shape[0], out.shape[1]
    safe = np.zeros(3, dtype=np.bool_)
    for b in range(n_batch):
        for i in range(n):
            tr, ta, has_target = _window_target(
                windows, messages, use_messages, b, i, safe
            )
            _choose_probs(out[b, i], safe, tr, ta, has_target, explore)


@numba.njit(cache=True)  # type: ignore
def _nearest_food(food, b: int, colour: int, per_colour: int, x0: int, y0: int):
    """Nearest food of ``colour`` to (x0, y0) by Manhattan distance; x = -1 if none."""
    best = 1 << 30
    bx = -1
    by = -1
    for slot in range(colour * per_colour, (colour + 1) * per_colour):
        x = food[b, slot, 0]
        if x < 0:
            continue
        y = food[b, slot, 1]
        dist = abs(x - x0) + abs(y - y0)
        if dist < best:
            best, bx, by = dist, x, y
    return bx, by, best


@numba.njit(cache=True)  # type: ignore
def _oracle_target(board, heads, heading, food, b: int, i: int, safe):
    """Fill ``safe`` and return the (right, ahead, has_target) target on the full board."""
    n = heading.shape[1]
    height, width = board.shape[2], board.shape[3]
    per_colour = food.shape[1] // n
    hx = heads[b, i, 0]
    hy = heads[b, i, 1]
    d = heading[b, i]
    for a in range(3):
        nd = d if a == 0 else ((d + 3) % 4 if a == 1 else (d + 1) % 4)
        x = hx + DIR_DX[nd]
        y = hy + DIR_DY[nd]
        ok = 0 <= x < width and 0 <= y < height
        if ok:
            for j in range(n):
                if board[b, j, y, x] > 0:
                    ok = False
        safe[a] = ok
    fx, fy, dist = _nearest_food(food, b, i, per_colour, hx, hy)
    tr = 0
    ta = 0
    if fx >= 0:
        ox, oy = fx - hx, fy - hy
        ta = ox * DIR_DX[d] + oy * DIR_DY[d]
        tr = ox * DIR_DX[(d + 1) % 4] + oy * DIR_DY[(d + 1) % 4]
    return tr, ta, fx >= 0


@numba.njit(cache=True)  # type: ignore
def _oracle_actions(out, board, heads, heading, food, rng, explore: float) -> None:
    """Greedy on the full board: nearest own-colour food, never into a wall or body."""
    n_batch, n = out.shape
    safe = np.zeros(3, dtype=np.bool_)
    for b in range(n_batch):
        for i in range(n):
            if heads[b, i, 0] < 0:
                out[b, i] = 0
                continue
            tr, ta, has_target = _oracle_target(board, heads, heading, food, b, i, safe)
            out[b, i] = _choose(rng, b, safe, tr, ta, has_target, explore)


@numba.njit(cache=True)  # type: ignore
def _oracle_probs(out, board, heads, heading, food, explore: float) -> None:
    """Exact action distribution of ``_oracle_actions``, shape (B, N, 3)."""
    n_batch, n = out.shape[0], out.shape[1]
    safe = np.zeros(3, dtype=np.bool_)
    for b in range(n_batch):
        for i in range(n):
            if heads[b, i, 0] < 0:
                out[b, i, :] = 0.0
                out[b, i, 0] = 1.0
                continue
            tr, ta, has_target = _oracle_target(board, heads, heading, food, b, i, safe)
            _choose_probs(out[b, i], safe, tr, ta, has_target, explore)


@numba.njit(cache=True)  # type: ignore
def _hand_coded_messages(out, heads, heading, food, near_distance: int) -> None:
    """3-bit codes: 2 * direction-of-nearest-own-food + far (see module docstring)."""
    n_batch, n = heading.shape
    per_colour = food.shape[1] // n
    for b in range(n_batch):
        for i in range(n):
            out[b, i, :] = 0
            hx = heads[b, i, 0]
            hy = heads[b, i, 1]
            if hx < 0:
                continue
            fx, fy, dist = _nearest_food(food, b, i, per_colour, hx, hy)
            if fx < 0:
                continue
            d = heading[b, i]
            ox, oy = fx - hx, fy - hy
            ahead = ox * DIR_DX[d] + oy * DIR_DY[d]
            right = ox * DIR_DX[(d + 1) % 4] + oy * DIR_DY[(d + 1) % 4]
            if abs(ahead) >= abs(right):
                direction = 0 if ahead > 0 else 2
            else:
                direction = 1 if right > 0 else 3
            out[b, i, 0] = 2 * direction + (1 if dist > near_distance else 0)


# ---------------------------------------------------------------- specialists
class SpecialistPolicy:
    """Maps observations to (B, N) int64 actions and (B, N, 3) action probabilities.

    ``act`` samples from exactly the distribution ``probs`` returns; ``probs`` is
    deterministic and needs no ``reset``.
    """

    name = "specialist"
    salt = 0

    def reset(self, seeds: Sequence[int]) -> None:
        """Seed one stream per episode."""
        self.rng = _seed_states(seeds, self.salt)

    def act(self, obs: Observations) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def probs(self, obs: Observations) -> np.ndarray:  # pragma: no cover - interface
        """Action distribution (B, N, 3) float64 that ``act`` samples from."""
        raise NotImplementedError


class GreedyPolicy(SpecialistPolicy):
    """Head for the nearest visible own-colour food; when none is in view, follow the
    hand-coded message (direction bits). Never steps into a visible wall or body.

    Args:
        explore: Chance of a random safe turn while wandering
    """

    name = "greedy"
    salt = 101
    use_messages = True

    def __init__(self, explore: float = 0.1):
        self.explore = explore

    def _inputs(self, obs: Observations) -> tuple[np.ndarray, np.ndarray]:
        windows = obs.specialists["window"]
        messages = obs.specialists["message"]
        if messages.dtype != np.int64 or messages.ndim != 3:
            messages = np.zeros(windows.shape[:2] + (0,), dtype=np.int64)
        return windows, messages

    def act(self, obs: Observations) -> np.ndarray:
        windows, messages = self._inputs(obs)
        out = np.zeros(windows.shape[:2], dtype=np.int64)
        _window_actions(
            out, windows, messages, self.use_messages, self.rng, self.explore
        )
        return out

    def probs(self, obs: Observations) -> np.ndarray:
        windows, messages = self._inputs(obs)
        out = np.zeros(windows.shape[:2] + (3,), dtype=np.float64)
        _window_probs(out, windows, messages, self.use_messages, self.explore)
        return out


class NoCommPolicy(GreedyPolicy):
    """Greedy on the window only; ignores messages and wanders when no food is in view."""

    name = "no-comm"
    salt = 202
    use_messages = False


class RandomPolicy(SpecialistPolicy):
    """Uniformly random relative actions."""

    name = "random"
    salt = 303

    def reset(self, seeds: Sequence[int]) -> None:
        self._gen = np.random.default_rng([int(s) for s in seeds] + [self.salt])

    def act(self, obs: Observations) -> np.ndarray:
        shape = obs.specialists["window"].shape[:2]
        return self._gen.integers(0, 3, size=shape, dtype=np.int64)

    def probs(self, obs: Observations) -> np.ndarray:
        shape = obs.specialists["window"].shape[:2]
        return np.full(shape + (3,), 1.0 / 3.0)


class OraclePolicy(SpecialistPolicy):
    """Specialists that see the full board (the centre's view): greedy on the nearest
    own-colour food anywhere, never into a wall or body. An upper reference for what
    perfect information buys a greedy specialist; it ignores messages.
    """

    name = "oracle"
    salt = 404

    def __init__(self, explore: float = 0.1):
        self.explore = explore

    def act(self, obs: Observations) -> np.ndarray:
        c = obs.centre
        out = np.zeros(c["heading"].shape, dtype=np.int64)
        _oracle_actions(
            out, c["board"], c["heads"], c["heading"], c["food"], self.rng, self.explore
        )
        return out

    def probs(self, obs: Observations) -> np.ndarray:
        c = obs.centre
        out = np.zeros(c["heading"].shape + (3,), dtype=np.float64)
        _oracle_probs(
            out, c["board"], c["heads"], c["heading"], c["food"], self.explore
        )
        return out


# ---------------------------------------------------------------- centres
class CentrePolicy:
    """Maps observations to a (B, N, ...) message batch, sent every step."""

    name = "centre"

    def reset(self, seeds: Sequence[int]) -> None:
        """Seed per episode (no-op for deterministic centres)."""

    def send(self, obs: Observations) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class HandCodedCentre(CentrePolicy):
    """Direction and near/far of each receiver's nearest own-colour food (3 bits).

    Args:
        length: Message length L (the code is symbol 0; the rest are 0)
        near_distance: Manhattan distance up to which food counts as near
    """

    name = "hand-coded"

    def __init__(self, length: int = 1, near_distance: int = 5):
        self.length = length
        self.near_distance = near_distance

    def send(self, obs: Observations) -> np.ndarray:
        heading = obs.centre["heading"]
        out = np.zeros(heading.shape + (self.length,), dtype=np.int64)
        c = obs.centre
        _hand_coded_messages(out, c["heads"], heading, c["food"], self.near_distance)
        return out


class RandomCentre(CentrePolicy):
    """Uniformly random symbols in [0, vocab_size)."""

    name = "random"

    def __init__(self, length: int = 1, vocab_size: int = HAND_CODED_VOCAB):
        self.length = length
        self.vocab_size = vocab_size

    def reset(self, seeds: Sequence[int]) -> None:
        self._gen = np.random.default_rng([int(s) for s in seeds] + [505])

    def send(self, obs: Observations) -> np.ndarray:
        shape = obs.centre["heading"].shape + (self.length,)
        return self._gen.integers(0, self.vocab_size, size=shape, dtype=np.int64)


class SilentCentre(CentrePolicy):
    """Sends the constant all-zero code every step.

    That is the same payload specialists read before any message arrives, so what they
    observe matches never sending; sending it explicitly lets message hooks and
    recorders see the silent channel like any other.
    """

    name = "silent"

    def send(self, obs: Observations) -> np.ndarray:
        return np.zeros_like(obs.specialists["message"])
