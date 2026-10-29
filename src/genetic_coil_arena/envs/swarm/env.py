"""Drone swarm lab (R14): heterogeneous drones find a gas source and confirm it on camera.

A deliberately abstract model of the hardware sketch in ``docs/directions/drone_swarm.md``.
Each drone has a flight controller and a companion processor. The flight controller
(stabilisation from the gyro and accelerometer) is abstracted away: a drone flies the
velocity set-point its companion gives it. The companion runs the policy, reads the
accessories and the positioning sensors, and uses the radio.

- Arena: a square of side ``arena`` metres with no obstacles; drones are clamped to it.
- Motion: 9 discrete actions per step of ``dt`` seconds: hover, or fly at ``speed`` in
  one of 8 compass directions.
- Positioning: the *environment* sensor (optical flow and height, as on a flow deck)
  gives a dead-reckoned position that drifts as a random walk of ``drift`` metres per
  step and axis, starting exact on a known launch pad. The *swarm* sensor (UWB, or
  ESP32 Wi-Fi FTM) gives noisy ranges to every other drone (``range_noise`` metres).
- Accessories: ``roles`` gives each drone one of ``ROLES``. A ``gas`` drone reads the
  plume concentration: ``exp(-d^2 / (2 plume_sigma^2))`` at distance d from the source,
  present only with probability ``hit_rate`` per step (turbulent plumes are
  intermittent), with multiplicative noise and a floor. A ``camera`` drone sees the
  source when its true distance is at most ``camera_range``; that confirms it. With
  ``n_sources`` > 1 there are several sources (at least ``source_gap`` apart; the
  concentration is the maximum over them), and the episode ends when all are confirmed.
- Radio: each step every drone may broadcast one packet of ``MessageSpec.length`` bytes
  (int64 values in [0, 256)). Each other drone receives it with probability
  ``1 - loss``, in its next observation (``inbox`` and ``inbox_ok``).
- Failures (R14 step 3): ``failures`` drones per episode, chosen at random, go down at a
  step drawn uniformly from ``fail_window`` (a flat battery or a crash): from then on a
  downed drone stays where it landed, its radio is silent and its camera confirms
  nothing. Nobody is told; the others only stop hearing from it.

All randomness of an episode is drawn at ``reset`` from its own seed, so a seed always
gives the same episode whatever else is in the batch.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..base import EnvSpec, MessageSpec, MultiAgentEnv, Observations, StepResult

ROLES = ("gas", "camera")
N_ACTIONS = 9  # hover + 8 compass directions
ANGLES = np.arange(8) * np.pi / 4
MOVES = np.concatenate([[[0.0, 0.0]], np.stack([np.cos(ANGLES), np.sin(ANGLES)], 1)])


@dataclass(frozen=True)
class SwarmConfig:
    """Swarm parameters (metres, seconds, steps).

    Args:
        arena: Side of the square arena
        roles: One accessory role per drone (``ROLES``)
        speed, dt: Flight speed and step length
        max_steps: Episode length (T)
        pad: Launch pad centre; drones start within ``pad_radius`` of it
        source_box: Each source is uniform in ``[lo, hi]^2``
        n_sources, source_gap: Number of sources and their minimum separation
        plume_sigma, hit_rate, gas_noise, gas_floor: Plume and gas sensor model
        camera_range: Distance at which a camera confirms the source
        drift: Dead-reckoning random walk per step and axis
        range_noise: Std of the inter-drone range measurements
        loss: Packet loss probability per sender, receiver and step
        message: Packet format; ``length`` = bytes per packet (0 = no radio)
        failures, fail_window: Drones that go down per episode, and the steps
            ``[lo, hi)`` in which each goes down
    """

    arena: float = 30.0
    roles: tuple[str, ...] = ("gas", "gas", "gas", "camera", "camera", "camera")
    speed: float = 1.0
    dt: float = 0.5
    max_steps: int = 240
    pad: tuple[float, float] = (2.0, 2.0)
    pad_radius: float = 1.0
    source_box: tuple[float, float] = (8.0, 28.0)
    n_sources: int = 1
    source_gap: float = 8.0
    plume_sigma: float = 3.0
    hit_rate: float = 0.6
    gas_noise: float = 0.2
    gas_floor: float = 0.02
    camera_range: float = 1.5
    drift: float = 0.03
    range_noise: float = 0.1
    loss: float = 0.0
    message: MessageSpec = field(
        default_factory=lambda: MessageSpec("discrete", 4, 256)
    )
    failures: int = 0
    fail_window: tuple[int, int] = (20, 160)

    def __post_init__(self) -> None:
        if not self.roles or any(r not in ROLES for r in self.roles):
            raise ValueError(f"roles must be a non-empty tuple of {ROLES}")
        if not 0.0 <= self.loss <= 1.0:
            raise ValueError("loss must lie in [0, 1]")
        if not 0 <= self.failures <= len(self.roles):
            raise ValueError("failures must lie in [0, n_drones]")
        if not 1 <= self.fail_window[0] < self.fail_window[1]:
            raise ValueError("fail_window must be [lo, hi) with 1 <= lo < hi")
        if self.message.kind != "discrete" or self.message.vocab_size != 256:
            raise ValueError("the radio carries bytes: a discrete 256-symbol message")

    @property
    def n_drones(self) -> int:
        return len(self.roles)

    @property
    def role_index(self) -> np.ndarray:
        return np.array([ROLES.index(r) for r in self.roles], dtype=np.int64)


class SwarmEnv(MultiAgentEnv):
    """Batched swarm episodes; ``messages`` passed to ``step`` are the drones' broadcasts."""

    def __init__(self, config: Optional[SwarmConfig] = None):
        self.config = config or SwarmConfig()
        c = self.config
        n, length = c.n_drones, c.message.length
        self.spec = EnvSpec(
            n_specialists=n,
            n_actions=N_ACTIONS,
            specialist_obs={
                "pos_est": ((2,), np.float32),
                "gas": ((1,), np.float32),
                "camera": ((1,), np.float32),
                "ranges": ((n,), np.float32),
                "role": ((len(ROLES),), np.float32),
                "inbox": ((n, length), np.int64),
                "inbox_ok": ((n,), bool),
            },
            centre_obs={
                "pos": ((n, 2), np.float32),
                "source": ((c.n_sources, 2), np.float32),
                "found": ((c.n_sources,), bool),
            },
            message=c.message,
            max_steps=c.max_steps,
        )
        self._b = 0

    @property
    def batch_size(self) -> int:
        return self._b

    # ------------------------------------------------------------------ reset
    def reset(self, seeds: Sequence[int]) -> Observations:
        c = self.config
        seeds = [int(s) for s in seeds]
        if not seeds:
            raise ValueError("reset needs at least one seed")
        b, n, t = len(seeds), c.n_drones, c.max_steps
        self._b = b
        self.pos = np.zeros((b, n, 2))
        k = c.n_sources
        self.source = np.zeros((b, k, 2))
        self._drift = np.zeros((b, t + 1, n, 2))
        self._hit = np.zeros((b, t + 1, n), dtype=bool)
        self._gnoise = np.zeros((b, t + 1, n))
        self._rnoise = np.zeros((b, t + 1, n, n))
        self._deliver = np.zeros((b, t + 1, n, n), dtype=bool)
        self._fail_at = np.full((b, n), t + 1, dtype=np.int64)  # step it goes down
        for i, seed in enumerate(seeds):
            g = np.random.default_rng([seed, 1414])
            ang = g.uniform(0, 2 * np.pi, n)
            rad = c.pad_radius * np.sqrt(g.uniform(0, 1, n))
            self.pos[i] = np.asarray(c.pad) + np.stack(
                [rad * np.cos(ang), rad * np.sin(ang)], 1
            )
            src = g.uniform(*c.source_box, size=(k, 2))
            while k > 1 and _min_gap(src) < c.source_gap:
                src = g.uniform(*c.source_box, size=(k, 2))
            self.source[i] = src
            steps = g.normal(0.0, c.drift, size=(t, n, 2))
            self._drift[i, 1:] = np.cumsum(steps, axis=0)
            self._hit[i] = g.random((t + 1, n)) < c.hit_rate
            self._gnoise[i] = g.normal(0.0, 1.0, (t + 1, n))
            noise = g.normal(0.0, c.range_noise, (t + 1, n, n))
            self._rnoise[i] = (noise + noise.transpose(0, 2, 1)) / np.sqrt(2)
            self._deliver[i] = g.random((t + 1, n, n)) >= c.loss
            if c.failures:  # own stream: episodes without failures stay as they were
                f = np.random.default_rng([seed, 1415])
                who = f.choice(n, size=c.failures, replace=False)
                self._fail_at[i, who] = f.integers(*c.fail_window, size=c.failures)
        self.t = np.zeros(b, dtype=np.int64)
        self.done = np.zeros(b, dtype=bool)
        self.confirmed = np.zeros(b, dtype=bool)
        self.found = np.zeros((b, k), dtype=bool)
        self.steps = np.zeros(b, dtype=np.int64)
        empty = np.zeros((b, n, n, c.message.length), dtype=np.int64)
        cam = self._sees().any(axis=2)
        return self._observe(empty, np.zeros((b, n, n), dtype=bool), cam)

    # ------------------------------------------------------------------ dynamics
    def concentration(self, pos: np.ndarray) -> np.ndarray:
        """Mean plume concentration at true positions (B, N, 2) -> (B, N)."""
        d2 = ((pos[:, :, None, :] - self.source[:, None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * self.config.plume_sigma**2)).max(axis=2)

    def _observe(
        self, inbox: np.ndarray, ok: np.ndarray, cam: np.ndarray
    ) -> Observations:
        """``cam`` (B, N): the camera drone saw a source it had not been found before."""
        c = self.config
        b, n = self.pos.shape[:2]
        rows = np.arange(b)
        tt = np.minimum(self.t, c.max_steps)
        est = self.pos + self._drift[rows, tt]
        gas_role = (c.role_index == ROLES.index("gas"))[None, :]
        conc = self.concentration(self.pos) * self._hit[rows, tt]
        gas = conc * (1 + c.gas_noise * self._gnoise[rows, tt]) + c.gas_floor * np.abs(
            self._gnoise[rows, tt]
        )
        up = self.t[:, None] < self._fail_at  # a downed drone's sensors are off
        gas = np.where(gas_role & up, np.clip(gas, 0.0, None), 0.0)
        diff = self.pos[:, :, None, :] - self.pos[:, None, :, :]
        ranges = np.sqrt((diff**2).sum(-1)) + self._rnoise[rows, tt]
        ranges = np.clip(ranges, 0.0, None) * (1 - np.eye(n))[None]
        role = np.zeros((b, n, len(ROLES)), dtype=np.float32)
        role[:, np.arange(n), c.role_index] = 1.0
        return Observations(
            specialists={
                "pos_est": est.astype(np.float32),
                "gas": gas[..., None].astype(np.float32),
                "camera": cam[..., None].astype(np.float32),
                "ranges": ranges.astype(np.float32),
                "role": role,
                "inbox": inbox,
                "inbox_ok": ok,
            },
            centre={
                "pos": self.pos.astype(np.float32),
                "source": self.source.astype(np.float32),
                "found": self.found.copy(),
            },
        )

    def _sees(self) -> np.ndarray:
        """(B, N, K): camera drone n sees not-yet-found source k."""
        c = self.config
        cam_role = (c.role_index == ROLES.index("camera"))[None, :, None]
        up = (self.t[:, None] < self._fail_at)[:, :, None]
        diff = self.pos[:, :, None, :] - self.source[:, None, :, :]
        d = np.sqrt((diff**2).sum(-1))
        return cam_role & up & (d <= c.camera_range) & ~self.found[:, None, :]

    def step(
        self, actions: np.ndarray, messages: Optional[np.ndarray] = None
    ) -> StepResult:
        c = self.config
        acts = self.validate_actions(actions)
        b, n = acts.shape
        live = ~self.done
        move = MOVES[acts] * (c.speed * c.dt)
        move = np.where((self.t[:, None] >= self._fail_at)[..., None], 0.0, move)
        self.pos = np.where(
            live[:, None, None], np.clip(self.pos + move, 0.0, c.arena), self.pos
        )
        self.t = np.where(live, self.t + 1, self.t)
        length = c.message.length
        if messages is None or length == 0:
            sent = np.zeros((b, n, length), dtype=np.int64)
            ok = np.zeros((b, n, n), dtype=bool)
        else:
            sent = np.asarray(messages, dtype=np.int64)
            if sent.shape != (b, n, length):
                raise ValueError(f"messages must have shape {(b, n, length)}")
            if sent.min(initial=0) < 0 or sent.max(initial=0) > 255:
                raise ValueError("message symbols are bytes in [0, 255]")
            rows = np.arange(b)
            ok = self._deliver[rows, self.t] & ~np.eye(n, dtype=bool)[None]
            ok &= (self.t[:, None] < self._fail_at)[:, None, :]  # downed senders
        # inbox[b, i, j] = what receiver i got from sender j
        inbox = np.where(ok[..., None], sent[:, None, :, :], 0)
        sees = self._sees() & live[:, None, None]
        new = sees.any(axis=1)
        self.found |= new
        self.confirmed = self.found.all(axis=1)
        self.steps = np.where(live, self.t, self.steps)
        self.done = self.done | self.confirmed | (self.t >= c.max_steps)
        rewards = new.sum(axis=1, keepdims=True).repeat(n, 1).astype(np.float32)
        obs = self._observe(inbox, ok, sees.any(axis=2))
        info = {
            "confirmed": self.confirmed.copy(),
            "n_found": self.found.sum(axis=1),
            "steps": self.steps.copy(),
        }
        return StepResult(obs, rewards, self.done.copy(), info)


def _min_gap(points: np.ndarray) -> float:
    d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    return float(d[~np.eye(len(points), dtype=bool)].min())
