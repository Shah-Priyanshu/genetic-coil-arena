"""R17 step 3: tell the leader whether the person may be moving.

Step 2's planner could not tell a still person from a walking one, so it hedged and gave
up 15 points on still-person missions. An operator usually knows ("injured hiker" vs
"lost child, may be walking"), and the ground station can ask. Version 3 planners read
that hint (``Mission.moving_hint``: +1 may be moving, -1 still), which is right 85% of
the time in training (``r17_mission.HINT_ACCURACY``). 16 conditioned planners train on
version 3 missions with step 2's budget; there is no blind version 3 (a blind plan reads
nothing).

Conditions (40 held-out missions each, 2,000 episodes, paired by seed):

- ``hint 85% right``: the version 2 held-out missions with their own drawn hints;
- ``hint always right``, ``hint always wrong``, ``no hint``: the same missions;
- ``still person, hint still``: step 1's held-out missions, hint -1.

Systems: the step-1 planner, the version 2 planner and the version 3 planner (16 runs
each; the first two ignore the hint).

Bars (fixed before any version 3 training run), CIs over missions:

- **(a) still people:** on ``still person, hint still`` the version 3 planner is at most
  3 points below the step-1 planner (step 2 was 15.3 below);
- **(b) gain:** on ``hint 85% right`` it beats the version 2 planner by >= 3 points,
  CI above 0;
- **(c) walking people kept:** on the walking missions of ``hint 85% right`` it is at
  most 3 points below the version 2 planner.

The other conditions are reported without a bar.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..envs.swarm.mission import Mission, run_missions
from ..evaluation import mean_ci
from . import r17_mission as r17

EPISODES = 2000
SYSTEMS = {"step-1 planner": 1, "planner v2": 2, "planner v3 (hint)": 3}
CONDITIONS = ("hint 85% right", "hint always right", "hint always wrong", "no hint",
              "still person, hint still")  # fmt: skip


def missions_for(cond: str) -> list[Mission]:
    if cond == "still person, hint still":
        return [replace(m, moving_hint=-1.0) for m in r17.held_out_missions(1)]
    if cond == "hint 85% right":
        return r17.held_out_missions(3)
    out = []
    for m in r17.held_out_missions(2):
        truth = 1.0 if m.person_speed > 0 else -1.0
        hint = {"hint always right": truth, "hint always wrong": -truth,
                "no hint": 0.0}[cond]  # fmt: skip
        out.append(replace(m, moving_hint=hint))
    return out


def tasks(runs: dict[str, list[dict]], episodes: int = EPISODES) -> list[tuple]:
    out = []
    for cond in CONDITIONS:
        ms = missions_for(cond)
        for name, rs in runs.items():
            for r, run in enumerate(rs):
                plans = r17.Planner(r17.PlannerSettings(**run["settings"])).plans(
                    run["theta"], ms
                )
                for i, plan in enumerate(plans):
                    out.append(((cond, name, r, i), cond, i, plan, episodes))
    return out


def play_cell(task: tuple) -> tuple[tuple, dict[str, float]]:
    key, cond, i, plan, episodes = task
    m = missions_for(cond)[i]
    res = run_missions(m, plan, r17.mission_seeds(r17.TEST_BASE, i, episodes))
    return key, {k: float(res[k].mean()) for k in ("score", "found", "wrong")}


def _scores(cells: dict, cond: str, name: str) -> np.ndarray:
    runs = sorted({k[2] for k in cells if k[0] == cond and k[1] == name})
    return np.array([[cells[(cond, name, r, i)]["score"] for i in range(r17.N_TEST)]
                     for r in runs]).mean(0)  # fmt: skip


def summarise(cells: dict[tuple, dict]) -> dict[str, Any]:
    walking = np.array([m.person_speed > 0 for m in r17.held_out_missions(2)])
    table: dict[str, dict[str, float]] = {}
    for cond in CONDITIONS:
        table[cond] = {}
        for name in SYSTEMS:
            s = _scores(cells, cond, name)
            table[cond][name] = float(s.mean())
            if cond != "still person, hint still":
                table[cond][name + " (walking)"] = float(s[walking].mean())
                table[cond][name + " (still)"] = float(s[~walking].mean())
    v3, v2, s1 = "planner v3 (hint)", "planner v2", "step-1 planner"
    main = "hint 85% right"
    diffs = {
        "still": mean_ci((_scores(cells, CONDITIONS[-1], v3)
                          - _scores(cells, CONDITIONS[-1], s1)).tolist()),
        "gain": mean_ci((_scores(cells, main, v3) - _scores(cells, main, v2)).tolist()),
        "walking": mean_ci((_scores(cells, main, v3)
                            - _scores(cells, main, v2))[walking].tolist()),
    }  # fmt: skip
    bars = {"still": diffs["still"][0] >= -0.03,
            "gain": diffs["gain"][0] >= 0.03 and diffs["gain"][0] - diffs["gain"][1] > 0,
            "walking": diffs["walking"][0] >= -0.03}  # fmt: skip
    bars = {k: bool(v) for k, v in bars.items()}
    return {"table": table, "diffs": diffs, "bars": bars, "passed": all(bars.values()),
            "walking_missions": int(walking.sum())}  # fmt: skip


LABELS = {
    "still": "(a) still people: v3 − step-1 planner (still person, hint still)",
    "gain": "(b) gain: v3 − v2 (hint 85% right)",
    "walking": "(c) walking kept: v3 − v2 on walking missions (hint 85% right)",
}
NEED = {"still": "≥ −3", "gain": "≥ +3, CI > 0", "walking": "≥ −3"}


def render(rep: dict[str, Any], episodes: int = EPISODES) -> str:
    lines = ["# R17 step 3: tell the leader whether the person may be moving", "",
             f"**{'GOOD SIGN' if rep['passed'] else 'NOT YET'}** (all three bars "
             f"{'met' if rep['passed'] else 'needed'}).", "",
             "| Bar | Difference (points, 95% CI over missions) | Needed | Met |",
             "|---|---|---|---|"]  # fmt: skip
    for key, label in LABELS.items():
        d = rep["diffs"][key]
        lines.append(f"| {label} | {100 * d[0]:+.1f} ± {100 * d[1]:.1f} | {NEED[key]} | "
                     f"{'yes' if rep['bars'][key] else 'no'} |")  # fmt: skip
    lines += ["", f"Score (found − 2 × wrong find, points); {r17.N_TEST} held-out missions "
              f"× {episodes} episodes per condition ({rep['walking_missions']} with a "
              "walking person in the version 2 set).", "",
              "| Condition | " + " | ".join(SYSTEMS) + " |",
              "|---|" + "---|" * len(SYSTEMS)]  # fmt: skip
    for cond, row in rep["table"].items():
        lines.append(f"| {cond} | "
                     + " | ".join(f"{100 * row[n]:.1f}" for n in SYSTEMS) + " |")  # fmt: skip
    lines += ["", "Reproduce: `python scripts/r17c_hint.py --from runs:r17-train-cond-* "
              "runs:r17b-train-cond-* runs:r17c-train-*`.", ""]  # fmt: skip
    return "\n".join(lines)
