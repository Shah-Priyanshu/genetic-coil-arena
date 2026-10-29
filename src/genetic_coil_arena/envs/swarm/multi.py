"""Several gas sources: a hand-designed gossip swarm and an oracle bound (R14 probe).

``MultiCodec`` packs (position, value, flag) into the packet: ``pos_bits`` per axis,
one flag bit (1 = "a camera confirmed a source here") and the rest for the value
(square-root companded). ``MultiTargetSwarm`` is the hand design:

- every drone keeps up to ``2 K`` hotspots (reports at least ``merge`` metres apart) and
  the sources it knows are confirmed;
- reports within ``exclude`` metres of a confirmed source are dropped, and a gas drone
  whose own best lies there starts afresh;
- camera j takes the (j mod h)-th strongest open hotspot, so cameras spread over the
  hotspots (with ``one_each``, only cameras j < h do, and the rest keep exploring); gas
  drones search around their own best; otherwise drones explore;
- packets alternate: even steps carry the drone's best open reading, odd steps one known
  confirmation (cycling), so finds spread despite packet loss.

``OracleSwarm`` is an upper bound, not a design: cameras are told where the unfound
sources are and split them greedily; it needs no sensing and no radio.
"""

from collections.abc import Sequence

import numpy as np

from ..base import Observations
from .env import ROLES, SwarmConfig
from .policies import ScriptedSwarm, _from_bits, _to_bits, heading_action


class MultiCodec:
    """x, y (``pos_bits`` each), flag (1 bit), value (the remaining bits)."""

    def __init__(self, budget: int, pos_bits: int, arena: float):
        self.budget, self.pos_bits, self.arena = int(budget), int(pos_bits), arena
        self.value_bits = 8 * self.budget - 2 * self.pos_bits - 1
        if self.pos_bits < 1 or self.value_bits < 1:
            raise ValueError("need at least one bit for each field")
        if max(self.pos_bits, self.value_bits) > 31:
            raise ValueError("at most 31 bits per field")

    @property
    def cell(self) -> float:
        return self.arena / (1 << self.pos_bits)

    def encode(
        self, pos: np.ndarray, value: np.ndarray, flag: np.ndarray
    ) -> np.ndarray:
        cells = 1 << self.pos_bits
        q = np.clip(np.floor(pos / self.arena * cells), 0, cells - 1).astype(np.int64)
        top = (1 << self.value_bits) - 1
        v = np.rint(np.sqrt(np.clip(value, 0.0, 1.0)) * top).astype(np.int64)
        bits = np.concatenate(
            [
                _to_bits(q[..., 0], self.pos_bits),
                _to_bits(q[..., 1], self.pos_bits),
                _to_bits(np.asarray(flag, dtype=np.int64), 1),
                _to_bits(v, self.value_bits),
            ],
            axis=-1,
        )
        return _from_bits(bits.reshape(*bits.shape[:-1], self.budget, 8))

    def decode(self, packets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bits = _to_bits(np.asarray(packets, dtype=np.int64), 8)
        bits = bits.reshape(*bits.shape[:-2], 8 * self.budget)
        p = self.pos_bits
        qx, qy = _from_bits(bits[..., :p]), _from_bits(bits[..., p : 2 * p])
        flag = bits[..., 2 * p].astype(bool)
        v = _from_bits(bits[..., 2 * p + 1 :]) / ((1 << self.value_bits) - 1)
        pos = np.stack([(qx + 0.5) * self.cell, (qy + 0.5) * self.cell], axis=-1)
        return pos, v**2, flag


class MultiTargetSwarm:
    """Hand-designed gossip search for ``config.n_sources`` sources."""

    def __init__(
        self,
        config: SwarmConfig,
        codec: MultiCodec,
        threshold: float = 0.15,
        gas_radius: float = 1.5,
        camera_radius: float = 4.5,
        merge: float = 5.0,
        exclude: float = 5.0,
        one_each: bool = False,
    ):
        self.config, self.codec = config, codec
        self.one_each = one_each
        self.threshold, self.merge, self.exclude = threshold, merge, exclude
        # exploration waypoints, jitter and the own-best memory come from ScriptedSwarm
        self.sw = ScriptedSwarm(config, None, threshold, gas_radius, camera_radius)
        self.is_gas = config.role_index == ROLES.index("gas")
        self.cam_rank = np.zeros(config.n_drones, dtype=np.int64)
        cams = np.flatnonzero(~self.is_gas)
        self.cam_rank[cams] = np.arange(len(cams))

    def reset(self, seeds: Sequence[int]) -> None:
        self.sw.reset(seeds)
        b, n, k = len(seeds), self.config.n_drones, self.config.n_sources
        self.hot = np.zeros((b, n, 2 * k, 3))
        self.conf = np.full((b, n, k, 2), np.nan)
        self.step_i = 0

    def _near_conf(self, pts: np.ndarray) -> np.ndarray:
        """(B, N, M, 2) -> (B, N, M): within ``exclude`` of a known confirmation."""
        diff = pts[:, :, :, None, :] - self.conf[:, :, None, :, :]
        d = np.nan_to_num(np.sqrt((diff**2).sum(-1)), nan=np.inf)
        return (d < self.exclude).any(axis=3)

    def _add_conf(self, pos: np.ndarray, mask: np.ndarray) -> None:
        add = mask & ~self._near_conf(pos[:, :, None, :])[..., 0]
        empty = np.isnan(self.conf[..., 0])
        add &= empty.any(axis=2)
        bi, ni = np.nonzero(add)
        self.conf[bi, ni, empty.argmax(axis=2)[bi, ni]] = pos[bi, ni]

    def _add_hot(self, pos: np.ndarray, val: np.ndarray, mask: np.ndarray) -> None:
        d = np.sqrt(((self.hot[..., :2] - pos[:, :, None, :]) ** 2).sum(-1))
        d = np.where(self.hot[..., 2] > 0, d, np.inf)
        j_near = d.argmin(axis=2)
        near = np.take_along_axis(d, j_near[..., None], 2)[..., 0] < self.merge
        slot = np.where(near, j_near, self.hot[..., 2].argmin(axis=2))
        cur = np.take_along_axis(self.hot[..., 2], slot[..., None], 2)[..., 0]
        bi, ni = np.nonzero(mask & (val > cur))
        self.hot[bi, ni, slot[bi, ni]] = np.concatenate(
            [pos[bi, ni], val[bi, ni, None]], -1
        )

    def act(self, obs: Observations) -> tuple[np.ndarray, np.ndarray]:
        s, sw = obs.specialists, self.sw
        pos = s["pos_est"].astype(float)
        gas = s["gas"][..., 0].astype(float)
        b, n = gas.shape
        self._add_conf(pos, s["camera"][..., 0] > 0)
        hp, hv, hf = self.codec.decode(s["inbox"])  # (B, N, N, ...)
        ok = s["inbox_ok"]
        for j in range(n):
            self._add_conf(hp[:, :, j], ok[:, :, j] & hf[:, :, j])
            self._add_hot(hp[:, :, j], hv[:, :, j], ok[:, :, j] & ~hf[:, :, j])
        stale = self._near_conf(sw.own[:, :, None, :2])[..., 0]
        sw.own[stale] = 0.0
        fresh = (gas > sw.own[..., 2]) & ~self._near_conf(pos[:, :, None, :])[..., 0]
        sw.own[fresh] = np.concatenate([pos, gas[..., None]], -1)[fresh]
        self._add_hot(sw.own[..., :2], sw.own[..., 2], sw.own[..., 2] > 0)
        self.hot[self._near_conf(self.hot[..., :2])] = 0.0

        a = self.config.arena
        reached = np.sqrt(((sw.wp - pos) ** 2).sum(-1)) < 0.5
        renew = reached | (sw.wp_age >= sw.explore_every)
        sw.wp[renew] = sw.rng.uniform(0, a, (int(renew.sum()), 2))
        sw.wp_age = np.where(renew, 0, sw.wp_age + 1)
        radius = np.where(self.is_gas[None, :], sw.gas_radius, sw.camera_radius)
        rejit = sw.jit_age >= sw.jitter_every
        ang = sw.rng.uniform(0, 2 * np.pi, (b, n))
        rad = radius * np.sqrt(sw.rng.uniform(0, 1, (b, n)))
        jit = np.stack([rad * np.cos(ang), rad * np.sin(ang)], -1)
        sw.jit = np.where(rejit[..., None], jit, sw.jit)
        sw.jit_age = np.where(rejit, 0, sw.jit_age + 1)

        order = np.argsort(-self.hot[..., 2], axis=2, kind="stable")
        ranked = np.take_along_axis(self.hot, order[..., None], 2)
        open_n = (ranked[..., 2] >= self.threshold).sum(axis=2)
        pick = self.cam_rank[None, :] % np.maximum(open_n, 1)
        cam_t = np.take_along_axis(ranked[..., :2], pick[..., None, None], 2)[:, :, 0]
        own_ok = sw.own[..., 2] >= self.threshold
        gas_d = self.is_gas[None, :]
        cam_has = open_n > 0
        if self.one_each:  # camera j covers hotspot j; spare cameras keep exploring
            cam_has = self.cam_rank[None, :] < open_n
        known = np.where(gas_d, own_ok, cam_has)
        target = np.where(gas_d[..., None], sw.own[..., :2], cam_t)
        target = np.where(known[..., None], np.clip(target + sw.jit, 0, a), sw.wp)
        actions = heading_action(target - pos, hover_within=0.25)

        self.step_i += 1
        n_conf = (~np.isnan(self.conf[..., 0])).sum(axis=2)
        send_conf = (self.step_i % 2 == 1) & (n_conf > 0)
        which = (self.step_i // 2) % np.maximum(n_conf, 1)
        conf_pos = np.take_along_axis(self.conf, which[..., None, None], 2)[:, :, 0]
        out_pos = np.where(
            send_conf[..., None], np.nan_to_num(conf_pos), sw.own[..., :2]
        )
        out_val = np.where(send_conf | ~gas_d, 0.0, sw.own[..., 2])
        return actions, self.codec.encode(out_pos, out_val, send_conf)


def assign(
    pos: np.ndarray, src: np.ndarray, open_k: np.ndarray, cams: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy split of cameras over open sources, closest (camera, source) pair first;
    spare cameras go to the nearest open source. Returns (targets (B, N, 2), assigned
    (B, N))."""
    target = pos.copy()
    assigned = np.zeros(pos.shape[:2], dtype=bool)
    for bi in range(pos.shape[0]):
        unfound = np.flatnonzero(open_k[bi])
        left, free = list(unfound), list(cams)
        while left and free:
            d = np.sqrt(((pos[bi, free][:, None] - src[bi, left][None]) ** 2).sum(-1))
            ci, kj = np.unravel_index(d.argmin(), d.shape)
            c = free.pop(ci)
            target[bi, c], assigned[bi, c] = src[bi, left.pop(kj)], True
        for c in free if unfound.size else []:
            d = np.sqrt(((src[bi, unfound] - pos[bi, c]) ** 2).sum(-1))
            target[bi, c], assigned[bi, c] = src[bi, unfound[d.argmin()]], True
    return target, assigned


class OracleSwarm:
    """Upper bound: cameras know every unfound source and split them greedily."""

    def __init__(self, config: SwarmConfig):
        self.config = config
        self.cams = np.flatnonzero(config.role_index == ROLES.index("camera"))

    def reset(self, seeds: Sequence[int]) -> None:
        pass

    def act(self, obs: Observations) -> tuple[np.ndarray, None]:
        pos = obs.specialists["pos_est"].astype(float)
        src = obs.centre["source"].astype(float)
        target, _ = assign(pos, src, ~obs.centre["found"], self.cams)
        return heading_action(target - pos, hover_within=0.1), None


class SensedOracleSwarm:
    """Coordination bound: a source becomes known to every drone, exactly, the moment
    any gas drone smells it (reading >= ``threshold``, attributed to the nearest
    source). Known unfound sources get cameras greedily; gas drones leave known
    sources (within ``retask`` metres) to look for others; all other drones explore as
    the silent swarm does. It bounds what better messages and allocation could add
    to the swarm's own sensing."""

    def __init__(
        self, config: SwarmConfig, threshold: float = 0.15, retask: float = 8.0
    ):
        self.config, self.threshold, self.retask = config, threshold, retask
        self.sw = ScriptedSwarm(config, None, threshold)
        self.cams = np.flatnonzero(config.role_index == ROLES.index("camera"))

    def reset(self, seeds: Sequence[int]) -> None:
        self.sw.reset(seeds)
        self.known = np.zeros((len(seeds), self.config.n_sources), dtype=bool)

    def act(self, obs: Observations) -> tuple[np.ndarray, None]:
        true = obs.centre["pos"].astype(float)
        src = obs.centre["source"].astype(float)
        d2 = ((true[:, :, None, :] - src[:, None, :, :]) ** 2).sum(-1)
        smell = obs.specialists["gas"][..., 0] >= self.threshold
        bi, ni = np.nonzero(smell)
        self.known[bi, d2.argmin(axis=2)[bi, ni]] = True
        # gas drones ignore known sources (forget them, and do not smell near them)
        own = self.sw.own[..., :2]
        k = self.known[:, None, :]
        near_own = ((own[:, :, None, :] - src[:, None, :, :]) ** 2).sum(-1)
        self.sw.own[((near_own < self.retask**2) & k).any(axis=2)] = 0.0
        here = ((d2 < self.retask**2) & k).any(axis=2)
        gas = np.where(here[..., None], 0.0, obs.specialists["gas"])
        masked = Observations({**obs.specialists, "gas": gas}, obs.centre)
        explore, _ = self.sw.act(masked)
        pos = obs.specialists["pos_est"].astype(float)
        target, assigned = assign(
            pos, src, self.known & ~obs.centre["found"], self.cams
        )
        go = heading_action(target - pos, hover_within=0.1)
        return np.where(assigned, go, explore), None
