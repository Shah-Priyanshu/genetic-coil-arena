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

**Hand-off (R17 step 2, ``Plan.track_s``):** with confirmers, the scout that makes a
detection stops sweeping and holds it in view for up to ``track_s`` s, flying to the
candidate's latest position and moving it with every re-sighting, until a confirmer
starts looking (then it resumes its lane). ``track_s = 0`` is step 1's tactic. A find
radius (``Mission.found_m``) can grow with the GPS error.

**Measured detector and operator (R17 step 4):** with ``Mission.table`` the per-frame
recall and false detections come from a measured table (``TABLE_THR`` x ``TABLE_PX``,
R15b), person size is ``px_m / altitude`` px and a one-second look holds
``frames_per_look`` independent frames. With ``operator_confirms`` the confirmer's view
goes to the operator, who recognises a person in it (``OP_HIT`` per look) or wrongly
accepts an empty one (``OP_FALSE`` per candidate); the detector's own confirmations are
not used then.

**Outcome:** a declaration within ``found_m`` (default ``FOUND_M``) of the person is a find; any other
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
# R17 step 4: a measured detection table (R15b) instead of the parametric curve.
# Recall is given per frame at these thresholds and person sizes (px, the square root
# of the person's box area, geometric bin centres of R15b's 0-8 / 8-16 / 16-32 / 32+),
# then false detections per tile at the same thresholds.
TABLE_THR = (0.3, 0.5, 0.7)
TABLE_PX = (5.7, 11.3, 22.6, 45.0)
PERSON_AREA_M = 0.34  # sqrt of a person's box area seen from above (0.45 x 0.25 m)
# operator confirmation (step 4): the confirmer streams its view and the operator
# decides; per look the operator recognises a person in view with OP_HIT, and per
# candidate wrongly accepts an empty view with OP_FALSE (choices, not measurements)
OP_HIT, OP_FALSE = 0.9, 0.02
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
    found_m: float = FOUND_M  # a declaration this close to the person is a find
    # the operator's hint about the person (+1 may be moving, -1 still, 0 not said);
    # read by version 3 planners only, never by the simulator
    moving_hint: float = 0.0
    # step 4: person size in px x altitude for the camera, independent frames per
    # one-second look, and the measured table (12 recalls, 3 false-detection rates;
    # empty = the parametric ``detector``)
    px_m: float = PX_M
    frames_per_look: float = 1.0
    table: tuple[float, ...] = ()
    operator_confirms: bool = False

    def vector(self) -> np.ndarray:
        if self.table and len(self.table) != len(TABLE_THR) * (len(TABLE_PX) + 1):
            raise ValueError("table needs 12 recalls and 3 false-detection rates")
        table = list(self.table) or [0.0] * (len(TABLE_THR) * (len(TABLE_PX) + 1))
        return np.array([self.width, self.height, self.drones, self.battery_s,
                         *astuple(self.detector), self.person_speed, self.loc_error,
                         self.speed, self.found_m, self.px_m, self.frames_per_look,
                         1.0 if self.table else 0.0, *table,
                         1.0 if self.operator_confirms else 0.0],
                        dtype=np.float64)  # fmt: skip


@dataclass(frozen=True)
class Plan:
    sweep_alt: float = 5.0
    overlap: float = 0.0
    sweep_thr: float = 0.7
    confirmers: int = 0
    confirm_alt: float = 4.0
    confirm_thr: float = 0.7
    confirm_looks: int = 3
    track_s: float = 0.0  # R17 step 2: a scout holds a sighting this long (0 = never)

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


_TABLE_THR = np.array(TABLE_THR)
_TABLE_LOGPX = np.log(np.array(TABLE_PX))


@njit(cache=True)
def _interp(x, xs, ys):
    """Piecewise-linear interpolation, clamped at both ends."""
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            w = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] * (1.0 - w) + ys[i] * w
    return ys[len(xs) - 1]


@njit(cache=True)
def _tier_table(alt, thr, mv, loc0):
    """Footprint, per-look detection probability and false detections, and position
    error from the measured table: recall interpolated in log px (below the first
    bin it falls in proportion to px) and in threshold (clamped to 0.3-0.7); false
    detections interpolated in log space; ``frames_per_look`` independent frames."""
    px_m, n = mv[12], mv[13]
    rec = mv[15:27].reshape(3, 4)
    fa = mv[27:30]
    px = px_m / alt
    lpx = math.log(px)
    per_thr = np.empty(3)
    for k in range(3):
        r = _interp(lpx, _TABLE_LOGPX, rec[k])
        if px < math.exp(_TABLE_LOGPX[0]):
            r *= px / math.exp(_TABLE_LOGPX[0])
        per_thr[k] = r
    p = min(max(_interp(thr, _TABLE_THR, per_thr), 0.0), 1.0)
    lam = math.exp(_interp(thr, _TABLE_THR, np.log(np.maximum(fa, 1e-9))))
    p_look = 1.0 - (1.0 - p) ** n
    return FOOTPRINT * alt, p_look, lam * n, loc0 + LOC1 * alt


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
    """Add a detection as a candidate, or merge it into an open one within MERGE_M:
    the new count and the candidate's index (-1 when the list is full)."""
    for c in range(nk):
        if ks[c] < 2 and math.hypot(kx[c] - x, ky[c] - y) <= MERGE_M:
            return nk, c
    if nk < MAX_CANDIDATES:
        kx[nk], ky[nk], ks[nk] = x, y, 0
        return nk + 1, nk
    return nk, -1


@njit(cache=True)
def _episode(seed, mv, pv, out):
    np.random.seed(seed)
    width, height, n_drones, battery = mv[0], mv[1], int(mv[2]), mv[3]
    px50, slope, fa50, fa_slope = mv[4], mv[5], mv[6], mv[7]
    walk, loc0, spd, found_m = mv[8], mv[9], mv[10], mv[11]
    track_s = pv[7]
    tx, ty = np.random.random() * width, np.random.random() * height
    heading = np.random.random() * 2 * math.pi if walk > 0 else 0.0
    hx, hy = width / 2.0, -5.0
    sweep_alt, overlap, n_conf = pv[0], pv[1], int(pv[3])
    conf_alt, looks = pv[4], int(pv[6])
    if mv[14] > 0:
        s_s, p_s, lam_s, sig_s = _tier_table(sweep_alt, pv[2], mv, loc0)
        s_c, p_c, lam_c, sig_c = _tier_table(conf_alt, pv[5], mv, loc0)
    else:
        s_s, p_s, lam_s, sig_s = _tier(sweep_alt, pv[2], px50, slope, fa50, fa_slope,
                                       loc0)  # fmt: skip
        s_c, p_c, lam_c, sig_c = _tier(conf_alt, pv[5], px50, slope, fa50, fa_slope,
                                       loc0)  # fmt: skip
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
    smode = np.zeros(n_sw, np.int64)  # 0 climb, 1 sweep, 2 return, 3 landed, 4 hold
    strack, stimer = np.full(n_sw, -1, np.int64), np.zeros(n_sw)
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
                                1.0 if math.hypot(rx - tx, ry - ty) <= found_m else 2.0
                            )
                            out[1], out[2] = t, swept / lanes
                            return
                        nk, idx = _add(kx, ky, ks, nk, rx, ry)
                        if track_s > 0 and idx >= 0:
                            smode[i], strack[i], stimer[i] = 4, idx, track_s
                for _ in range(np.random.poisson(lam_s)):
                    fx = sx[i] + (np.random.random() - 0.5) * s_s
                    fy = sy[i] + (np.random.random() - 0.5) * s_s
                    if n_conf == 0:
                        out[0] = 1.0 if math.hypot(fx - tx, fy - ty) <= found_m else 2.0
                        out[1], out[2] = t, swept / lanes
                        return
                    nk, idx = _add(kx, ky, ks, nk, fx, fy)
                    if track_s > 0 and idx >= 0 and smode[i] == 1:
                        smode[i], strack[i], stimer[i] = 4, idx, track_s
                continue
            if smode[i] == 4:  # hold the sighting in view until a confirmer takes it
                c = strack[i]
                need = math.hypot(sx[i] - hx, sy[i] - hy) / spd + sweep_alt / CLIMB
                handed = False
                for j in range(n_conf):
                    if cmode[j] == 3 and ctarget[j] == c:
                        handed = True
                if battery - sused[i] <= need + RESERVE_S:
                    smode[i], strack[i] = 2, -1
                    continue
                if handed or ks[c] == 2 or stimer[i] <= 0:
                    smode[i], strack[i] = 1, -1
                    continue
                stimer[i] -= 1.0
                sx[i], sy[i], _, _ = _step_to(sx[i], sy[i], kx[c], ky[c], spd)
                half = s_s / 2
                if abs(tx - sx[i]) <= half and abs(ty - sy[i]) <= half:
                    if np.random.random() < p_s:  # re-sighting moves the candidate
                        kx[c] = tx + np.random.normal() * sig_s
                        ky[c] = ty + np.random.normal() * sig_s
                for _ in range(np.random.poisson(lam_s)):
                    fx = sx[i] + (np.random.random() - 0.5) * s_s
                    fy = sy[i] + (np.random.random() - 0.5) * s_s
                    nk, _ = _add(kx, ky, ks, nk, fx, fy)
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
            elif cmode[j] == 3 and mv[30] > 0:  # the operator judges the view
                half = s_c / 2
                if abs(tx - kx[c]) <= half and abs(ty - ky[c]) <= half:
                    if np.random.random() < OP_HIT:
                        rx = tx + np.random.normal() * sig_c
                        ry = ty + np.random.normal() * sig_c
                        out[0] = 1.0 if math.hypot(rx - tx, ry - ty) <= found_m else 2.0
                        out[1], out[2] = t, swept / lanes
                        return
                elif clooks[j] == 0 and np.random.random() < OP_FALSE:
                    out[0] = (
                        1.0 if math.hypot(kx[c] - tx, ky[c] - ty) <= found_m else 2.0
                    )
                    out[1], out[2] = t, swept / lanes
                    return
                clooks[j] += 1
                if clooks[j] >= looks:
                    ks[c], cmode[j] = 2, 4
            elif cmode[j] == 3:
                half = s_c / 2
                if abs(tx - kx[c]) <= half and abs(ty - ky[c]) <= half:
                    if np.random.random() < p_c:
                        rx = tx + np.random.normal() * sig_c
                        ry = ty + np.random.normal() * sig_c
                        out[0] = 1.0 if math.hypot(rx - tx, ry - ty) <= found_m else 2.0
                        out[1], out[2] = t, swept / lanes
                        return
                if np.random.poisson(lam_c) > 0:
                    fx = kx[c] + (np.random.random() - 0.5) * s_c
                    fy = ky[c] + (np.random.random() - 0.5) * s_c
                    out[0] = 1.0 if math.hypot(fx - tx, fy - ty) <= found_m else 2.0
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
