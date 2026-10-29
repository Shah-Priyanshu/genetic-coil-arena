"""OpenSwarm person search: scouts sweep an area, confirmers check what they see (R17).

One sortie of ``Mission.drones`` drones over a ``width`` x ``height`` m area with one
person somewhere in it (uniform). Home is 5 m south of the middle of the south edge.
Every drone flies one battery (``battery_s``, about 5 min for OpenSwarm's 3S 650 mAh
pack) and must be home, with ``RESERVE_S`` to spare, before it runs out.

**Camera and detector.** The scout's camera (OV2640, about 60 degrees across) sees a
square footprint ``FOOTPRINT * altitude`` m wide. The K210 runs the detector on a 224 x
224 input (``firmware/src/ai/ai_module.h``), so a person (``PERSON_M`` across from above)
is ``PX_M / altitude`` px wide. Each second of flight is one *look*. A person in the
footprint at any point of that second's path is detected with probability ``Phi(slope * ln(px / px50) - logit(thr))`` at
confidence threshold ``thr`` (``Detector``: half the people of ``px50`` px are seen at
0.5). False detections arrive as a Poisson number per look, ``fa50 * exp(-fa_slope *
logit(thr))``, uniform in the footprint. A detection's ground position is off by
``LOC0 + LOC1 * altitude`` m (GPS plus projection; normal, per axis).

**Plan** (what the leader decides; ``Plan``):

- ``drones - confirmers`` sweepers split the area into equal north-south strips and
  fly boustrophedon lanes at ``sweep_alt``, lanes ``(1 - overlap)`` footprints apart.
  They return home when the strip is done or the battery says so.
- With no confirmers, the first detection at ``sweep_thr`` is declared the find (the
  OpenSwarm default: SEARCH_GRID at 5 m with person-follow's min_conf 0.7).
- Otherwise every detection becomes a candidate (merged within ``MERGE_M`` of an open
  one). The nearest free confirmer that can make the trip on its battery takes off
  (or turns back from its way home), flies to the oldest pending candidate, hovers at
  ``confirm_alt`` for up to ``confirm_looks`` looks at ``confirm_thr``, and the first
  detection there is declared the find. A candidate with no detection is dropped.

For robustness tests only, ``Mission`` can also make the person wander
(``person_speed``), worsen the GPS (``loc_error``) or slow the drones (``speed``, a
headwind); the defaults leave every episode unchanged.

**Outcome:** a declaration within ``FOUND_M`` of the person is a find; any other
declaration is a wrong find (the rescue team goes to the wrong place); no declaration
by the end of the sortie is a miss. ``score = found - 2 * wrong``.

Numbers the OpenSwarm repo does not give (speeds, reserve, localisation error, person
width) are ground-station choices and marked as constants here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import astuple, dataclass, field
from typing import Any

import numpy as np
from numba import njit

FOOTPRINT = 1.155  # footprint side / altitude: 2 tan(30 deg), OV2640's ~60 deg view
INPUT_PX = 224  # K210 detector input (OpenSwarm ai_module.h)
PERSON_M = 0.45  # a person's width seen from above (shoulders); choice
PX_M = INPUT_PX * PERSON_M / FOOTPRINT  # person width in px times altitude in m
SPEED, CLIMB = 4.0, 2.0  # m/s cruise and climb/descent; choice
RESERVE_S = 15.0  # battery seconds kept for landing; choice
LOC0, LOC1 = 1.0, 0.05  # detection position error: GPS (m) + per metre of altitude
FOUND_M = 5.0  # a declaration this close to the person is a find
MERGE_M = 5.0  # detections this close to an open candidate join it
MAX_CANDIDATES = 512
SUBSTEPS = 4  # positions per look checked for the person (every 1 m)
WRONG_COST = 2.0


@dataclass(frozen=True)
class Detector:
    """The scout brain's detection curve (see module)."""

    px50: float = 12.0
    slope: float = 2.5
    fa50: float = 0.005
    fa_slope: float = 1.5

    def p_detect(self, altitude: float, thr: float) -> float:
        z = self.slope * math.log(PX_M / altitude / self.px50) - _logit(thr)
        return _phi(z)

    def false_rate(self, thr: float) -> float:
        return self.fa50 * math.exp(-self.fa_slope * _logit(thr))


@dataclass(frozen=True)
class Mission:
    width: float = 120.0
    height: float = 120.0
    drones: int = 6
    battery_s: float = 300.0
    detector: Detector = field(default_factory=Detector)
    person_speed: float = 0.0  # m/s; a moving person wanders (robustness only)
    loc_error: float = LOC0  # GPS part of a detection's position error (m)
    speed: float = SPEED  # cruise speed (m/s); lower for a headwind

    def vector(self) -> np.ndarray:
        return np.array([self.width, self.height, self.drones, self.battery_s,
                         *astuple(self.detector), self.person_speed, self.loc_error,
                         self.speed], dtype=np.float64)  # fmt: skip


@dataclass(frozen=True)
class Plan:
    sweep_alt: float = 5.0
    overlap: float = 0.0
    sweep_thr: float = 0.7
    confirmers: int = 0
    confirm_alt: float = 4.0
    confirm_thr: float = 0.7
    confirm_looks: int = 3

    def vector(self) -> np.ndarray:
        return np.array(astuple(self), dtype=np.float64)


OPENSWARM_DEFAULT = Plan()  # SEARCH_GRID at 5 m; the first detection >= 0.7 is the find


def hand_two_tier(drones: int) -> Plan:
    """A sensible hand plan: sweep a little higher with a lower threshold, keep one or
    two drones on the ground to confirm from 4 m."""
    return Plan(sweep_alt=8.0, overlap=0.1, sweep_thr=0.5,
                confirmers=1 if drones <= 4 else 2, confirm_alt=4.0, confirm_thr=0.7,
                confirm_looks=3)  # fmt: skip


def run_missions(mission: Mission, plan: Plan, seeds: Sequence[int]) -> dict[str, Any]:
    """Play ``plan`` on ``mission`` once per seed. The person's position depends on the
    seed only, so plans compared on the same seeds search for the same people."""
    if not 0 <= plan.confirmers < mission.drones:
        raise ValueError("need at least one sweeper")
    out = _run_batch(np.asarray(seeds, dtype=np.int64), mission.vector(), plan.vector())
    outcome = out[:, 0].astype(np.int64)
    found, wrong = outcome == 1, outcome == 2
    return {"found": found, "wrong": wrong, "t": out[:, 1], "swept": out[:, 2],
            "score": found - WRONG_COST * wrong}  # fmt: skip


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ------------------------------------------------------------------ numba engine
_phi_nb = njit(cache=True)(_phi)
_logit_nb = njit(cache=True)(_logit)


@njit(cache=True)
def _tier(alt, thr, px50, slope, fa50, fa_slope, loc0):
    p = _phi_nb(slope * math.log(PX_M / alt / px50) - _logit_nb(thr))
    lam = fa50 * math.exp(-fa_slope * _logit_nb(thr))
    return FOOTPRINT * alt, p, lam, loc0 + LOC1 * alt


@njit(cache=True)
def _step_to(x, y, gx, gy, budget):
    """Move up to ``budget`` m towards (gx, gy): new x, y, distance moved, arrived."""
    dx, dy = gx - x, gy - y
    d = math.sqrt(dx * dx + dy * dy)
    if d <= budget:
        return gx, gy, d, True
    return x + dx / d * budget, y + dy / d * budget, budget, False


@njit(cache=True)
def _add(kx, ky, ks, nk, x, y):
    for c in range(nk):
        if ks[c] < 2 and math.hypot(kx[c] - x, ky[c] - y) <= MERGE_M:
            return nk
    if nk < MAX_CANDIDATES:
        kx[nk], ky[nk], ks[nk] = x, y, 0
        nk += 1
    return nk


@njit(cache=True)
def _episode(seed, mv, pv, out):
    np.random.seed(seed)
    width, height, n_drones, battery = mv[0], mv[1], int(mv[2]), mv[3]
    px50, slope, fa50, fa_slope = mv[4], mv[5], mv[6], mv[7]
    walk, loc0, spd = mv[8], mv[9], mv[10]
    tx, ty = np.random.random() * width, np.random.random() * height
    heading = np.random.random() * 2 * math.pi if walk > 0 else 0.0
    hx, hy = width / 2.0, -5.0
    sweep_alt, overlap, n_conf = pv[0], pv[1], int(pv[3])
    conf_alt, looks = pv[4], int(pv[6])
    s_s, p_s, lam_s, sig_s = _tier(sweep_alt, pv[2], px50, slope, fa50, fa_slope, loc0)
    s_c, p_c, lam_c, sig_c = _tier(conf_alt, pv[5], px50, slope, fa50, fa_slope, loc0)
    n_sw = n_drones - n_conf

    strip = width / n_sw
    m = max(1, int(math.ceil(strip / (s_s * (1.0 - overlap)))))
    y_lo, y_hi = min(s_s / 2, height / 2), max(height - s_s / 2, height / 2)
    n_wp = 2 * m
    wx, wy = np.empty((n_sw, n_wp)), np.empty((n_sw, n_wp))
    for i in range(n_sw):
        for j in range(m):
            x = i * strip + (j + 0.5) * strip / m
            a, b = (y_lo, y_hi) if j % 2 == 0 else (y_hi, y_lo)
            wx[i, 2 * j], wy[i, 2 * j], wx[i, 2 * j + 1], wy[i, 2 * j + 1] = x, a, x, b
    lanes = max(n_sw * (m * (y_hi - y_lo) + (m - 1) * strip / m), 1e-9)

    sx, sy = np.full(n_sw, hx), np.full(n_sw, hy)
    swp = np.zeros(n_sw, np.int64)
    smode = np.zeros(n_sw, np.int64)  # 0 climb, 1 sweep, 2 return, 3 landed
    sclimb = np.full(n_sw, sweep_alt / CLIMB)
    sused = np.zeros(n_sw)
    cx, cy = np.full(max(n_conf, 1), hx), np.full(max(n_conf, 1), hy)
    cmode = np.zeros(max(n_conf, 1), np.int64)  # 0 landed 1 climb 2 fly 3 look 4 return
    cclimb, cused = np.zeros(max(n_conf, 1)), np.zeros(max(n_conf, 1))
    ctarget, clooks = np.zeros(max(n_conf, 1), np.int64), np.zeros(
        max(n_conf, 1), np.int64
    )
    kx, ky = np.empty(MAX_CANDIDATES), np.empty(MAX_CANDIDATES)
    ks = np.zeros(MAX_CANDIDATES, np.int64)  # 0 pending, 1 checking, 2 dropped
    nk = 0
    swept = 0.0
    out[0], out[1] = 0.0, battery
    for t in range(1, int(battery) + 200):
        active = False
        if walk > 0:  # the person wanders, turning a little each second
            heading += np.random.normal() * 0.3
            tx += walk * math.cos(heading)
            ty += walk * math.sin(heading)
            if tx < 0 or tx > width:
                heading, tx = math.pi - heading, min(max(tx, 0.0), width)
            if ty < 0 or ty > height:
                heading, ty = -heading, min(max(ty, 0.0), height)
        for i in range(n_sw):
            if smode[i] == 3:
                continue
            active = True
            sused[i] += 1.0
            if smode[i] == 0:
                sclimb[i] -= 1.0
                if sclimb[i] <= 0:
                    smode[i] = 1
                continue
            if smode[i] == 1:
                need = math.hypot(sx[i] - hx, sy[i] - hy) / spd + sweep_alt / CLIMB
                if battery - sused[i] <= need + RESERVE_S or swp[i] >= n_wp:
                    smode[i] = 2
                    continue
                half, seen = s_s / 2, False
                for _ in range(SUBSTEPS):  # the camera sees the whole path flown
                    budget = spd / SUBSTEPS
                    while budget > 1e-9 and swp[i] < n_wp:
                        k = swp[i]
                        sx[i], sy[i], d, arrived = _step_to(sx[i], sy[i], wx[i, k],
                                                            wy[i, k], budget)  # fmt: skip
                        budget -= d
                        if k >= 1:
                            swept += d
                        if arrived:
                            swp[i] += 1
                    if abs(tx - sx[i]) <= half and abs(ty - sy[i]) <= half:
                        seen = True
                if seen:
                    if np.random.random() < p_s:
                        rx = tx + np.random.normal() * sig_s
                        ry = ty + np.random.normal() * sig_s
                        if n_conf == 0:
                            out[0] = (
                                1.0 if math.hypot(rx - tx, ry - ty) <= FOUND_M else 2.0
                            )
                            out[1], out[2] = t, swept / lanes
                            return
                        nk = _add(kx, ky, ks, nk, rx, ry)
                for _ in range(np.random.poisson(lam_s)):
                    fx = sx[i] + (np.random.random() - 0.5) * s_s
                    fy = sy[i] + (np.random.random() - 0.5) * s_s
                    if n_conf == 0:
                        out[0] = 1.0 if math.hypot(fx - tx, fy - ty) <= FOUND_M else 2.0
                        out[1], out[2] = t, swept / lanes
                        return
                    nk = _add(kx, ky, ks, nk, fx, fy)
                continue
            sx[i], sy[i], _, arrived = _step_to(sx[i], sy[i], hx, hy, spd)
            if arrived:
                smode[i] = 3
        for c in range(nk):  # oldest pending candidate first
            if ks[c] != 0:
                continue
            best, best_go = -1, 1e18
            back = math.hypot(kx[c] - hx, ky[c] - hy) / spd + conf_alt / CLIMB
            for j in range(n_conf):
                if cmode[j] != 0 and cmode[j] != 4:
                    continue
                go = math.hypot(cx[j] - kx[c], cy[j] - ky[c]) / spd
                if cmode[j] == 0:
                    go += conf_alt / CLIMB
                if cused[j] + go + looks + back + RESERVE_S <= battery and go < best_go:
                    best, best_go = j, go
            if best >= 0:
                ks[c], ctarget[best], clooks[best] = 1, c, 0
                if cmode[best] == 0:
                    cmode[best], cclimb[best] = 1, conf_alt / CLIMB
                else:
                    cmode[best] = 2
        for j in range(n_conf):
            if cmode[j] == 0:
                continue
            active = True
            cused[j] += 1.0
            c = ctarget[j]
            if cmode[j] == 1:
                cclimb[j] -= 1.0
                if cclimb[j] <= 0:
                    cmode[j] = 2
            elif cmode[j] == 2:
                cx[j], cy[j], _, arrived = _step_to(cx[j], cy[j], kx[c], ky[c], spd)
                if arrived:
                    cmode[j] = 3
            elif cmode[j] == 3:
                half = s_c / 2
                if abs(tx - kx[c]) <= half and abs(ty - ky[c]) <= half:
                    if np.random.random() < p_c:
                        rx = tx + np.random.normal() * sig_c
                        ry = ty + np.random.normal() * sig_c
                        out[0] = 1.0 if math.hypot(rx - tx, ry - ty) <= FOUND_M else 2.0
                        out[1], out[2] = t, swept / lanes
                        return
                if np.random.poisson(lam_c) > 0:
                    fx = kx[c] + (np.random.random() - 0.5) * s_c
                    fy = ky[c] + (np.random.random() - 0.5) * s_c
                    out[0] = 1.0 if math.hypot(fx - tx, fy - ty) <= FOUND_M else 2.0
                    out[1], out[2] = t, swept / lanes
                    return
                clooks[j] += 1
                if clooks[j] >= looks:
                    ks[c], cmode[j] = 2, 4
            else:
                cx[j], cy[j], _, arrived = _step_to(cx[j], cy[j], hx, hy, spd)
                if arrived:
                    cmode[j] = 0
                    cused[j] += conf_alt / CLIMB
        if not active:
            out[1] = t
            break
    out[2] = swept / lanes


@njit(cache=True)
def _run_batch(seeds, mv, pv):
    out = np.zeros((seeds.shape[0], 3))
    for e in range(seeds.shape[0]):
        _episode(seeds[e], mv, pv, out[e])
    return out
