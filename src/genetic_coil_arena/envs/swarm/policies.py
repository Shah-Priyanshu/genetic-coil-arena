"""Hand-designed swarm behaviour and radio code (R14 step 1 baselines).

``HandCodec`` packs one (position, value) report into the packet's bytes: ``pos_bits``
per axis over the arena and the rest of the bits for the value (square-root companded,
so low concentrations keep resolution). ``ScriptedSwarm`` is a PSO-flavoured search in
the spirit of Sniffy Bug (Duisterhof et al. 2021):

- every drone explores random waypoints until it knows a reading above ``threshold``;
- a gas drone then searches around the best reading it knows (its own, or the swarm's
  when talking) with fresh jitter of radius ``gas_radius``;
- a camera drone, which cannot smell, flies to the best reading the swarm has reported
  and searches around it within ``camera_radius``;
- with a coarse code both radii grow to 0.75 of a position cell, so the search covers
  the cell a report names;
- when talking, every gas drone broadcasts its best reading so far (position estimate
  and value); camera drones send nothing useful (all-zero packets, decoded as value 0).

Silent swarms run the same rules with an empty inbox, so their cameras only explore.
Positions are dead-reckoned estimates throughout; nothing here reads the true state.
"""

from collections.abc import Sequence
from typing import Optional

import numpy as np

from ..base import Observations
from .env import ANGLES, ROLES, SwarmConfig, SwarmEnv


def _to_bits(q: np.ndarray, n: int) -> np.ndarray:
    return ((q[..., None] >> np.arange(n)) & 1).astype(np.int64)


def _from_bits(bits: np.ndarray) -> np.ndarray:
    return (bits * (1 << np.arange(bits.shape[-1]))).sum(-1)


class HandCodec:
    """``pos_bits`` per axis plus ``8 * budget - 2 * pos_bits`` value bits."""

    def __init__(self, budget: int, pos_bits: int, arena: float):
        self.budget, self.pos_bits, self.arena = int(budget), int(pos_bits), arena
        self.value_bits = 8 * self.budget - 2 * self.pos_bits
        if self.pos_bits < 1 or self.value_bits < 1:
            raise ValueError("need at least one bit for each field")
        if max(self.pos_bits, self.value_bits) > 31:
            raise ValueError("at most 31 bits per field")

    def encode(self, pos: np.ndarray, value: np.ndarray) -> np.ndarray:
        """(..., 2) positions and (...) values in [0, 1] -> (..., budget) bytes."""
        cells = 1 << self.pos_bits
        q = np.clip(np.floor(pos / self.arena * cells), 0, cells - 1).astype(np.int64)
        top = (1 << self.value_bits) - 1
        v = np.rint(np.sqrt(np.clip(value, 0.0, 1.0)) * top).astype(np.int64)
        bits = np.concatenate(
            [
                _to_bits(q[..., 0], self.pos_bits),
                _to_bits(q[..., 1], self.pos_bits),
                _to_bits(v, self.value_bits),
            ],
            axis=-1,
        )
        return _from_bits(bits.reshape(*bits.shape[:-1], self.budget, 8))

    @property
    def cell(self) -> float:
        """Side of one position cell in metres."""
        return self.arena / (1 << self.pos_bits)

    def decode(self, packets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(..., budget) bytes -> ((..., 2) cell-centre positions, (...) values)."""
        bits = _to_bits(np.asarray(packets, dtype=np.int64), 8)
        bits = bits.reshape(*bits.shape[:-2], 8 * self.budget)
        p = self.pos_bits
        qx, qy = _from_bits(bits[..., :p]), _from_bits(bits[..., p : 2 * p])
        v = _from_bits(bits[..., 2 * p :]) / ((1 << self.value_bits) - 1)
        pos = np.stack([(qx + 0.5) * self.cell, (qy + 0.5) * self.cell], axis=-1)
        return pos, v**2


def heading_action(delta: np.ndarray, hover_within: float) -> np.ndarray:
    """(..., 2) displacement -> action: 0 (hover) when short, else the nearest direction."""
    ang = np.arctan2(delta[..., 1], delta[..., 0])
    k = np.rint((ang % (2 * np.pi)) / (np.pi / 4)).astype(np.int64) % len(ANGLES)
    far = np.sqrt((delta**2).sum(-1)) > hover_within
    return np.where(far, k + 1, 0)


class ScriptedSwarm:
    """Hand-designed swarm; ``codec`` None means silent (no packets)."""

    def __init__(
        self,
        config: SwarmConfig,
        codec: Optional[HandCodec] = None,
        threshold: float = 0.15,
        gas_radius: float = 1.5,
        camera_radius: float = 1.0,
        explore_every: int = 40,
        jitter_every: int = 8,
    ):
        self.config, self.codec = config, codec
        self.threshold = threshold
        self.gas_radius, self.camera_radius = gas_radius, camera_radius
        self.explore_every, self.jitter_every = explore_every, jitter_every
        self.is_gas = config.role_index == ROLES.index("gas")

    def reset(self, seeds: Sequence[int]) -> None:
        b, n = len(seeds), self.config.n_drones
        self.rng = np.random.default_rng([int(s) for s in seeds] + [2718])
        a = self.config.arena
        self.wp = self.rng.uniform(0, a, (b, n, 2))
        self.wp_age = np.zeros((b, n), dtype=np.int64)
        self.jit = np.zeros((b, n, 2))
        self.jit_age = np.full((b, n), self.jitter_every)
        self.own = np.zeros((b, n, 3))  # x, y, value of the best own reading
        self.best = np.zeros((b, n, 3))  # best reading known (own or heard)

    def act(self, obs: Observations) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """(actions (B, N), packets (B, N, L) or None)."""
        s = obs.specialists
        pos, gas = s["pos_est"].astype(float), s["gas"][..., 0].astype(float)
        b, n = gas.shape
        better = gas > self.own[..., 2]
        self.own[better] = np.concatenate([pos, gas[..., None]], -1)[better]
        take = self.own[..., 2] > self.best[..., 2]
        self.best[take] = self.own[take]
        if self.codec is not None:
            hp, hv = self.codec.decode(s["inbox"])  # (B, N, N, 2), (B, N, N)
            hv = np.where(s["inbox_ok"], hv, -1.0)
            j = hv.argmax(axis=2)
            top = np.take_along_axis(hv, j[..., None], 2)[..., 0]
            tp = np.take_along_axis(hp, j[..., None, None], 2)[:, :, 0]
            heard = top > self.best[..., 2]
            self.best[heard] = np.concatenate([tp, top[..., None]], -1)[heard]

        a = self.config.arena
        reached = np.sqrt(((self.wp - pos) ** 2).sum(-1)) < 0.5
        renew = reached | (self.wp_age >= self.explore_every)
        self.wp[renew] = self.rng.uniform(0, a, (int(renew.sum()), 2))
        self.wp_age = np.where(renew, 0, self.wp_age + 1)

        # a coarse code only says which cell: search the whole cell around its centre
        cell = 0.0 if self.codec is None else 0.75 * self.codec.cell
        radius = np.where(
            self.is_gas[None, :],
            max(self.gas_radius, cell),
            max(self.camera_radius, cell),
        )
        rejit = self.jit_age >= self.jitter_every
        ang = self.rng.uniform(0, 2 * np.pi, (b, n))
        rad = radius * np.sqrt(self.rng.uniform(0, 1, (b, n)))
        fresh = np.stack([rad * np.cos(ang), rad * np.sin(ang)], -1)
        self.jit = np.where(rejit[..., None], fresh, self.jit)
        self.jit_age = np.where(rejit, 0, self.jit_age + 1)

        known = self.best[..., 2] >= self.threshold
        target = np.where(
            known[..., None], np.clip(self.best[..., :2] + self.jit, 0, a), self.wp
        )
        actions = heading_action(target - pos, hover_within=0.25)
        if self.codec is None:
            return actions, None
        value = np.where(self.is_gas[None, :], self.own[..., 2], 0.0)
        where = np.where(self.is_gas[None, :, None], self.own[..., :2], 0.0)
        return actions, self.codec.encode(where, value)


def run_swarm(
    env: SwarmEnv, policy: ScriptedSwarm, seeds: Sequence[int]
) -> dict[str, np.ndarray]:
    """Play one batch to the end; returns ``confirmed`` (B,) and ``steps`` (B,)."""
    obs = env.reset(seeds)
    policy.reset(seeds)
    info: dict[str, np.ndarray] = {}
    while True:
        actions, packets = policy.act(obs)
        obs, _, done, info = env.step(actions, packets)
        if done.all():
            return {"confirmed": info["confirmed"], "steps": info["steps"]}
