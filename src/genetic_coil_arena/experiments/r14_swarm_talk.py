"""R14 step 1: does a drone swarm need to talk, and does its radio bind? (goal G13)

The lab is ``envs.swarm``: 3 gas drones and 3 camera drones search a 30 m arena for a
gas source, which counts as found when a camera drone comes within 1.5 m of it, within
240 steps (2 min). The hand-designed swarm (``ScriptedSwarm``) runs silent or
broadcasts its best gas reading every step (2 Hz) in packets of 1, 2, 4 or 8 bytes
(``HandCodec``). For each budget, the position bits and the search radius are tuned on
a separate tuning block at the primary setting (30% packet loss, flow-deck drift 0.03 m
per step) and then fixed for every cell.

Grid: silent and the 4 budgets x packet loss {0, 0.3} x drift {0.03, 0.1} m per step.
``N_REPS`` replicates of ``PER_REP`` episodes; every cell plays the same seeds, so
differences are paired by replicate.

Good-sign verdict (G13), at drift 0.03:

- talk matters: 8 B lossless minus silent is at least ``TALK_GAIN`` in success rate,
  and its 95% CI excludes 0;
- the radio binds: at 1 B with 30% loss, the hand design keeps at most
  ``RETAIN_MAX`` of that gain, ``(S_1B - S_silent) / (S_8B - S_silent)``.

Both: GOOD SIGN (learned codes have room here). Talk does not matter: the direction's
learned-communication part has nothing to win on this task. Talk matters but the radio
does not bind: hand codes are enough here, and learned codes need a richer task.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from ..envs.swarm import HandCodec, ScriptedSwarm, SwarmConfig, SwarmEnv, run_swarm
from ..evaluation import mean_ci
from ..messages import MessageSpec

BUDGETS = (1, 2, 4, 8)
POS_BITS = {
    1: (2, 3),
    2: (3, 4, 5, 6),
    4: (6, 8, 10, 12),
    8: (17, 20, 24),
}  # value field <= 31 bits
RADII = (1.0, 2.0, 3.0, 4.5)
LOSSES = (0.0, 0.3)
DRIFTS = (0.03, 0.1)
PRIMARY = {"loss": 0.3, "drift": 0.03}
N_REPS, PER_REP, N_TUNE_REPS = 10, 300, 2
TUNE_SEED_BASE, TEST_SEED_BASE, STRIDE = 7_000_000_000, 7_500_000_000, 10_000
TALK_GAIN, RETAIN_MAX = 0.15, 0.67
SILENT = "silent"


def rep_seeds(base: int, rep: int, n: int) -> list[int]:
    return [base + rep * STRIDE + i for i in range(n)]


def config_for(budget: int, loss: float, drift: float) -> SwarmConfig:
    msg = MessageSpec("discrete", max(budget, 1), 256)
    return replace(SwarmConfig(), loss=loss, drift=drift, message=msg)


def play(
    budget: int,
    hand: dict[str, float],
    loss: float,
    drift: float,
    seeds: Sequence[int],
) -> dict[str, np.ndarray]:
    """One batch; ``budget`` 0 is the silent swarm."""
    cfg = config_for(budget, loss, drift)
    codec = None if budget == 0 else HandCodec(budget, int(hand["pos_bits"]), cfg.arena)
    r = float(hand.get("radius", 1.0))
    policy = ScriptedSwarm(cfg, codec, camera_radius=r, gas_radius=max(1.5, r))
    return run_swarm(SwarmEnv(cfg), policy, seeds)


def tune(per_rep: int = PER_REP, reps: int = N_TUNE_REPS) -> dict[int, dict]:
    """Best (pos_bits, radius) per budget on the tuning block at the primary setting."""
    out: dict[int, dict] = {}
    seeds = [s for r in range(reps) for s in rep_seeds(TUNE_SEED_BASE, r, per_rep)]
    for budget in BUDGETS:
        scores = []
        for pb in POS_BITS[budget]:
            for radius in RADII:
                hand = {"pos_bits": pb, "radius": radius}
                got = play(budget, hand, PRIMARY["loss"], PRIMARY["drift"], seeds)
                scores.append((float(got["confirmed"].mean()), -radius, pb, hand))
        best = max(scores, key=lambda x: x[:3])
        out[budget] = {**best[3], "tune_success": best[0]}
    return out


def run_grid(
    hand: dict[int, dict], per_rep: int = PER_REP, reps: int = N_REPS
) -> dict[str, dict[str, Any]]:
    """Per cell: per-replicate success rate and mean steps (censored at T)."""
    cells: dict[str, dict[str, Any]] = {}

    def record(key: str, budget: int, h: dict, loss: float, drift: float) -> None:
        succ, steps = [], []
        for r in range(reps):
            got = play(budget, h, loss, drift, rep_seeds(TEST_SEED_BASE, r, per_rep))
            succ.append(float(got["confirmed"].mean()))
            steps.append(float(got["steps"].mean()))
        cells[key] = {"budget": budget, "loss": loss, "drift": drift,
                      "success": succ, "steps": steps}  # fmt: skip

    for drift in DRIFTS:
        record(cell_key(0, 0.0, drift), 0, {"pos_bits": 0}, 0.0, drift)
        for budget in BUDGETS:
            for loss in LOSSES:
                record(cell_key(budget, loss, drift), budget, hand[budget], loss, drift)
    return cells


def cell_key(budget: int, loss: float, drift: float) -> str:
    name = SILENT if budget == 0 else f"{budget}B"
    return f"{name}|loss={loss:g}|drift={drift:g}"


def g13_verdict(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    d = PRIMARY["drift"]
    silent = np.asarray(cells[cell_key(0, 0.0, d)]["success"])
    full = np.asarray(cells[cell_key(8, 0.0, d)]["success"])
    one = np.asarray(cells[cell_key(1, PRIMARY["loss"], d)]["success"])
    gain, gain_ci = mean_ci(list(full - silent))
    retained = float((one.mean() - silent.mean()) / gain) if gain > 0 else float("nan")
    talk = gain >= TALK_GAIN and gain - gain_ci > 0
    if not talk:
        verdict = "TALK DOES NOT MATTER"
    elif retained <= RETAIN_MAX:
        verdict = "GOOD SIGN"
    else:
        verdict = "RADIO DOES NOT BIND"
    return {
        "verdict": verdict,
        "talk_gain": float(gain),
        "talk_gain_ci": float(gain_ci),
        "retained_1B": retained,
        "silent": float(silent.mean()),
        "full": float(full.mean()),
        "one_byte": float(one.mean()),
    }


def build_report(hand: dict[int, dict], cells: dict[str, dict]) -> dict[str, Any]:
    g = g13_verdict(cells)
    d = PRIMARY["drift"]
    silent = np.asarray(cells[cell_key(0, 0.0, d)]["success"])
    retained = {}
    for budget in BUDGETS:
        for loss in LOSSES:
            s = np.asarray(cells[cell_key(budget, loss, d)]["success"])
            retained[cell_key(budget, loss, d)] = float(
                (s.mean() - silent.mean()) / g["talk_gain"]
            )
    return {
        "g13": g,
        "hand": {str(k): v for k, v in hand.items()},
        "cells": cells,
        "retained": retained,
        "config": json.loads(json.dumps(SwarmConfig().__dict__, default=str)),
        "grid": {"budgets": BUDGETS, "losses": LOSSES, "drifts": DRIFTS,
                 "reps": len(silent), "per_rep": None},  # fmt: skip
    }


def _pm(values: Sequence[float], scale: float = 100.0, digits: int = 1) -> str:
    m, ci = mean_ci(list(values))
    return f"{scale * m:.{digits}f} ± {scale * ci:.{digits}f}"


def render(rep: dict[str, Any]) -> str:
    g = rep["g13"]
    cells = rep["cells"]
    lines = [
        "# R14 step 1: does a drone swarm need to talk, and does its radio bind? (goal G13)",
        "",
        f"**G13: {g['verdict']}.** At flow-deck drift 0.03 m per step, the silent swarm "
        f"confirms the source in {100 * g['silent']:.1f}% of episodes and the 8-byte "
        f"lossless swarm in {100 * g['full']:.1f}%: talking adds "
        f"{100 * g['talk_gain']:+.1f} ± {100 * g['talk_gain_ci']:.1f} points (need ≥ "
        f"{100 * TALK_GAIN:.0f}). One byte per packet at 30% loss reaches "
        f"{100 * g['one_byte']:.1f}% and keeps {100 * g['retained_1B']:.0f}% of that gain "
        f"(the radio binds if ≤ {100 * RETAIN_MAX:.0f}%).",
        "",
        "Lab: `envs/swarm`, 3 gas and 3 camera drones, 30 m arena, source found when a "
        "camera is within 1.5 m, 240 steps of 0.5 s; one packet per drone per step (2 Hz). "
        "Mean ± 95% t-CI over replicates (same seeds in every cell).",
        "",
        "## Success rate (%)",
        "",
        "| Radio | loss 0, drift 0.03 | loss 0.3, drift 0.03 | loss 0, drift 0.1 | loss 0.3, drift 0.1 |",
        "|---|---|---|---|---|",
    ]

    def row(budget: int, field: str, scale: float, digits: int) -> str:
        name = SILENT if budget == 0 else f"{budget} B"
        vals = []
        for drift in DRIFTS:
            for loss in LOSSES:
                key = cell_key(budget, 0.0 if budget == 0 else loss, drift)
                vals.append(_pm(cells[key][field], scale, digits))
        return f"| {name} | " + " | ".join([vals[0], vals[1], vals[2], vals[3]]) + " |"

    for budget in (0, *BUDGETS):
        lines.append(row(budget, "success", 100.0, 1))
    lines += [
        "",
        "## Steps to confirm (censored at 240)",
        "",
        "| Radio | loss 0, drift 0.03 | loss 0.3, drift 0.03 | loss 0, drift 0.1 | loss 0.3, drift 0.1 |",
        "|---|---|---|---|---|",
    ]
    for budget in (0, *BUDGETS):
        lines.append(row(budget, "steps", 1.0, 1))
    lines += [
        "",
        "## Share of the talk gain kept (drift 0.03)",
        "",
        "| Budget | loss 0 | loss 0.3 |",
        "|---|---|---|",
    ]
    for budget in BUDGETS:
        vals = [
            f"{100 * rep['retained'][cell_key(budget, loss, PRIMARY['drift'])]:.0f}%"
            for loss in LOSSES
        ]
        lines.append(f"| {budget} B | {vals[0]} | {vals[1]} |")
    lines += [
        "",
        "Hand codes tuned on a separate seed block at loss 0.3, drift 0.03: "
        + "; ".join(
            f"{b} B: {h['pos_bits']} position bits per axis, search radius "
            f"{h['radius']:g} m (tuning success {100 * h['tune_success']:.1f}%)"
            for b, h in rep["hand"].items()
        )
        + ".",
        "",
        "Reproduce: `python scripts/r14_swarm_talk.py`.",
        "",
    ]
    return "\n".join(lines)
