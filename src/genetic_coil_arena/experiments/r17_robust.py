"""R17 step 1b: does the mission-reading planner keep its gain when the world gets worse?

R17 trained planners on random missions with a still person, 1 m GPS error, 4 m/s
drones and a 300 s battery. Real searches differ. This step plays the trained planners
(16 blind, 16 conditioned) and the hand plans, none of them retrained, on the 40
held-out missions under ``CONDITIONS``: a person who walks, worse GPS, a headwind, a
cold battery, larger areas and more drones. The planners are not told about the first
four; in the last two their inputs (area, drones) leave the training range
(``OUTSIDE_INPUTS``), so their plans change.

Bar (fixed before the run, with a minimum effect size because R14 step 3b showed a
bar without one can pass on a trivial gain): in **every** condition the conditioned
planner beats the blind plan by >= 3 points with the 95% CI over missions above 0.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..envs.swarm.mission import (
    OPENSWARM_DEFAULT,
    Mission,
    Plan,
    hand_two_tier,
    run_missions,
)
from ..evaluation import mean_ci
from . import r17_mission as r17

CONDITIONS: dict[str, dict[str, float]] = {
    "nominal": {},
    "person walks 0.5 m/s": {"person_speed": 0.5},
    "person walks 1.2 m/s": {"person_speed": 1.2},
    "GPS error 3 m": {"loc_error": 3.0},
    "headwind (3 m/s cruise)": {"speed": 3.0},
    "cold battery (240 s)": {"battery_s": 240.0},
    "areas x1.6": {"scale": 1.6},
    "10 drones": {"drones": 10},
}
OUTSIDE_INPUTS = frozenset({"areas x1.6", "10 drones"})
GAIN_BAR = 0.03
EPISODES = 2000


def missions_for(cond: str) -> list[Mission]:
    over: dict[str, Any] = dict(CONDITIONS[cond])
    scale = over.pop("scale", 1.0)
    if "drones" in over:
        over["drones"] = int(over["drones"])
    return [replace(m, width=m.width * scale, height=m.height * scale, **over)
            for m in r17.held_out_missions()]  # fmt: skip


def plans_for(cond: str, blind: list[dict], cond_runs: list[dict]) -> dict[str, list]:
    ms = missions_for(cond)
    out: dict[str, list[list[Plan]]] = {
        "OpenSwarm default": [[OPENSWARM_DEFAULT] * len(ms)],
        "hand two-tier": [[hand_two_tier(m.drones) for m in ms]],
    }
    for name, runs in (("blind", blind), ("conditioned", cond_runs)):
        out[name] = [r17.Planner(r17.PlannerSettings(**r["settings"])).plans(r["theta"], ms)
                     for r in runs]  # fmt: skip
    return out


def tasks(blind: list[dict], cond_runs: list[dict], episodes: int = EPISODES,
          conditions: list[str] | None = None) -> list[tuple]:  # fmt: skip
    out = []
    for cond in conditions or list(CONDITIONS):
        for name, runs in plans_for(cond, blind, cond_runs).items():
            for r, plans in enumerate(runs):
                for i, plan in enumerate(plans):
                    out.append(((cond, name, r, i), cond, i, plan, episodes))
    return out


def play_cell(task: tuple) -> tuple[tuple, dict[str, float]]:
    key, cond, i, plan, episodes = task
    m = missions_for(cond)[i]
    res = run_missions(m, plan, r17.mission_seeds(r17.TEST_BASE, i, episodes))
    return key, {k: float(res[k].mean()) for k in ("score", "found", "wrong")}


def summarise(cells: dict[tuple, dict]) -> dict[str, Any]:
    conds = [c for c in CONDITIONS if any(k[0] == c for k in cells)]
    by: dict[str, Any] = {}
    for cond in conds:
        names = sorted({k[1] for k in cells if k[0] == cond})
        row: dict[str, Any] = {}
        per: dict[str, np.ndarray] = {}
        for name in names:
            runs = sorted({k[2] for k in cells if k[0] == cond and k[1] == name})
            arr = {m: np.array([[cells[(cond, name, r, i)][m] for i in range(r17.N_TEST)]
                                for r in runs]) for m in ("score", "found", "wrong")}  # fmt: skip
            per[name] = arr["score"].mean(0)
            row[name] = {m: float(v.mean()) for m, v in arr.items()}
        gain = mean_ci((per["conditioned"] - per["blind"]).tolist())
        row["gain"] = gain
        row["pass"] = bool(gain[0] >= GAIN_BAR and gain[0] - gain[1] > 0)
        by[cond] = row
    fails = [c for c in conds if not by[c]["pass"]]
    return {"by_condition": by, "conditions": conds, "pass": not fails, "fails": fails}


def _cell(r: dict[str, float]) -> str:
    return f"{100 * r['score']:.1f} ({100 * r['found']:.0f}% / {100 * r['wrong']:.1f}%)"


def render(rep: dict[str, Any]) -> str:
    lines = [
        "# R17 step 1b: does the mission-reading planner survive a worse world?",
        "",
    ]
    lines.append(
        f"**Robust: {'PASS' if rep['pass'] else 'FAIL'}.** The conditioned planner beats "
        f"the blind plan by ≥ +{100 * GAIN_BAR:.0f} points with the CI above 0 in "
        + (
            "every condition."
            if rep["pass"]
            else f"all but: {', '.join(rep['fails'])}."
        )
    )
    lines += ["", f"{r17.N_TEST} held-out missions × {EPISODES} episodes per cell; score = "
              "found − 2 × wrong find, in points; gain = conditioned − blind, paired by "
              "mission (95% CI). Found / wrong find in brackets.", "",
              "| Condition | OpenSwarm default | Hand two-tier | Blind (16) | Conditioned "
              "(16) | Gain |", "|---|---|---|---|---|---|"]  # fmt: skip
    for cond in rep["conditions"]:
        row = rep["by_condition"][cond]
        mark = " *" if cond in OUTSIDE_INPUTS else ""
        cells = [_cell(row[n]) for n in
                 ("OpenSwarm default", "hand two-tier", "blind", "conditioned")]  # fmt: skip
        g = row["gain"]
        lines.append(f"| {cond}{mark} | " + " | ".join(cells)
                     + f" | {100 * g[0]:+.1f} ± {100 * g[1]:.1f} |")  # fmt: skip
    lines += ["", "\\* the planner's inputs (area, drones) leave the training range.", "",
              "Reproduce: `python scripts/r17_robust.py --from runs:r17-train-*`.", ""]  # fmt: skip
    return "\n".join(lines)
