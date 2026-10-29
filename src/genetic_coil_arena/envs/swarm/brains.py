"""Learned per-role brains for the swarm (R14 step 2): who goes where, and when.

``LearnedSwarm`` keeps everything of the hand design (``MultiTargetSwarm``) except the
destination choice. The radio, the codec, the hotspot and confirmation memory and the
local search all stay. Each step every drone scores its candidate destinations with the
brain of its role (one small MLP for gas drones, one for cameras) and heads for the best:

- candidate 0: explore (its current random waypoint);
- candidates 1..2K: the hotspots in its memory, strongest first (valid if at least
  ``threshold``);
- candidate 2K+1: its own best reading (gas drones, valid if at least ``threshold``).

Features per candidate (``N_FEATURES``): offset and distance from the drone, value,
candidate kind, hotspot rank, distance to the drone's current destination (lets it
learn to stick), time, confirmed count, the drone's index within its role (lets cameras
learn a split) and the number of open hotspots. A whole ES population plays at once:
episode b belongs to member ``b // episodes_per_member``.
"""

from collections.abc import Sequence
from typing import Optional

import numpy as np

from ...learning import MLP
from .env import SwarmConfig
from .multi import MultiCodec, MultiTargetSwarm

N_FEATURES = 15
MAX_ROLE = 3  # role-index one-hot width


def brain(hidden: int = 16) -> MLP:
    return MLP((N_FEATURES, hidden, 1))


class LearnedSwarm(MultiTargetSwarm):
    """``MultiTargetSwarm`` whose destinations come from per-role brains."""

    def __init__(
        self,
        config: SwarmConfig,
        codec: MultiCodec,
        gas_net: MLP,
        cam_net: MLP,
        **knobs: float,
    ):
        super().__init__(config, codec, **knobs)
        self.gas_net, self.cam_net = gas_net, cam_net
        self.gas_idx = np.flatnonzero(self.is_gas)
        self.cam_idx = np.flatnonzero(~self.is_gas)
        self.role_rank = np.zeros(config.n_drones, dtype=np.int64)
        self.role_rank[self.gas_idx] = np.arange(len(self.gas_idx))
        self.role_rank[self.cam_idx] = np.arange(len(self.cam_idx))
        self.gas_p: Optional[np.ndarray] = None
        self.cam_p: Optional[np.ndarray] = None
        self.per_member = 1

    def set_params(
        self, gas_p: np.ndarray, cam_p: np.ndarray, episodes_per_member: int = 1
    ) -> None:
        """(P, n) parameter rows for each brain; episode b uses row b // E."""
        self.gas_p, self.cam_p = np.atleast_2d(gas_p), np.atleast_2d(cam_p)
        self.per_member = int(episodes_per_member)

    def reset(self, seeds: Sequence[int]) -> None:
        super().reset(seeds)
        if self.gas_p is None or self.cam_p is None:
            raise RuntimeError("set_params() before reset()")
        if len(seeds) != len(self.gas_p) * self.per_member:
            raise ValueError("need episodes_per_member episodes for every member")
        self.cur: Optional[np.ndarray] = None

    def features(self, pos: np.ndarray, ranked: np.ndarray) -> tuple[np.ndarray, ...]:
        """(B, N, C, F) features, (B, N, C, 2) destinations, (B, N, C) valid, (B, N, C)
        search flags."""
        sw, c = self.sw, self.config
        b, n = pos.shape[:2]
        m = ranked.shape[2]
        dest = np.concatenate(
            [sw.wp[:, :, None, :], ranked[..., :2], sw.own[:, :, None, :2]], axis=2
        )
        value = np.concatenate(
            [np.zeros((b, n, 1)), ranked[..., 2:3][..., 0], sw.own[:, :, None, 2]],
            axis=2,
        )
        kind = np.zeros((m + 2, 3))
        kind[0, 0], kind[1 : m + 1, 1], kind[m + 1, 2] = 1, 1, 1
        valid = np.ones((b, n, m + 2), dtype=bool)
        valid[..., 1:] = value[..., 1:] >= self.threshold
        valid[..., m + 1] &= self.is_gas[None, :]
        if self.cur is None:
            self.cur = sw.wp.copy()
        a = c.arena
        rel = (dest - pos[:, :, None, :]) / a
        f = np.zeros((b, n, m + 2, N_FEATURES), dtype=np.float32)
        f[..., 0:2] = rel
        f[..., 2] = np.sqrt((rel**2).sum(-1))
        f[..., 3] = np.sqrt(np.clip(value, 0, None))
        f[..., 4:7] = kind[None, None]
        f[..., 1 : m + 1, 7] = np.arange(m) / m
        f[..., 8] = np.sqrt(((dest - self.cur[:, :, None, :]) ** 2).sum(-1)) / a
        f[..., 9] = self.step_i / c.max_steps
        f[..., 10] = (~np.isnan(self.conf[..., 0])).sum(axis=2)[..., None] / c.n_sources
        onehot = np.eye(MAX_ROLE)[np.minimum(self.role_rank, MAX_ROLE - 1)]
        f[..., 11:14] = onehot[None, :, None, :]
        f[..., 14] = (value[..., 1 : m + 1] >= self.threshold).sum(axis=2)[
            ..., None
        ] / m
        search = np.zeros((m + 2,), dtype=bool)
        search[1:] = True
        return f, dest, valid, np.broadcast_to(search, valid.shape)

    def _scores(self, net: MLP, params: np.ndarray, f: np.ndarray) -> np.ndarray:
        """(B, n_role, C, F) -> (B, n_role, C) with each member's own parameters."""
        b, k, cands, nf = f.shape
        p = len(params)
        x = f.reshape(p, (b // p) * k * cands, nf)
        return net.forward(params, x)[..., 0].reshape(b, k, cands)

    def choose(
        self, pos: np.ndarray, ranked: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.gas_p is not None and self.cam_p is not None
        f, dest, valid, search = self.features(pos, ranked)
        score = np.full(valid.shape, -np.inf, dtype=np.float32)
        for idx, net, params in (
            (self.gas_idx, self.gas_net, self.gas_p),
            (self.cam_idx, self.cam_net, self.cam_p),
        ):
            if len(idx):
                score[:, idx] = self._scores(net, params, f[:, idx])
        score = np.where(valid, score, -np.inf)
        pick = score.argmax(axis=2)
        chosen = np.take_along_axis(dest, pick[..., None, None], 2)[:, :, 0]
        self.cur = chosen
        return chosen, np.take_along_axis(search, pick[..., None], 2)[..., 0]
