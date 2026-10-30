"""R17 step 6: the recommended search system on realistic missions.

Everything R15-R17 found, together. Missions are step 3's held-out ones: half have a
walking person, with 1-4 m GPS error, wind and a 220-320 s battery, and the operator's
moving hint is right 85% of the time. The scout brain is a measured one (R15b's K210
brain at 224 px or R15c's at 448 px, with false detections as measured or cut 10 times by
an assumed persistence filter), one second of video holds ``frames`` independent frames,
and the operator confirms candidates from the confirmer's view. Plans come from
OpenSwarm's default, the step-2 learned planner, or mission-time tuning on the ground
station (``r17f_mission_time.tune_mission``: ~30 s of one core, warm-started from the
step-2 planner, over the person's unknown speed given the hint).

Conditions (``CONDITIONS``): today's K210 brain; the K210 with the filter; the 448
camera without and with the filter (3 frames per look), and the recommended system again
with 1 frame per look.

Bar (fixed before the run): the recommended system (448 camera, false detections cut
10 times, 3 frames per look, operator confirms, mission-time tuning) finds the person
in >= 50% of episodes with <= 5% wrong finds, averaged over the 40 missions.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from ..envs.swarm.mission import (
    FOOTPRINT,
    OPENSWARM_DEFAULT,
    PERSON_AREA_M,
    Mission,
    run_missions,
)
from . import r17_mission as r17
from . import r17e_requirements as rq
from . import r17f_mission_time as rf

CAMERA_REPORTS = {"K210 224": "docs/results/r15b_aerial_scout.json",
                  "448 camera": "docs/results/r15c_aerial448.json"}  # fmt: skip
# (camera, false-detection multiplier, frames per look)
CONDITIONS = (
    ("K210 224", 1.0, 3.0),
    ("K210 224", 0.1, 3.0),
    ("448 camera", 1.0, 3.0),
    ("448 camera", 0.1, 3.0),
    ("448 camera", 0.1, 1.0),
)
RECOMMENDED = ("448 camera", 0.1, 3.0)
FOUND_BAR, WRONG_BAR = 0.5, 0.05
TEST_EPISODES = 2000


@cache
def camera(name: str, root: Path = Path(".")) -> dict[str, Any]:
    return rq.camera_table(root / CAMERA_REPORTS[name])


def missions_for(cond: tuple, root: Path = Path(".")) -> list[Mission]:
    name, mult, frames = cond
    cam = camera(name, root)
    table = tuple(cam["recall"]) + tuple(f * mult for f in cam["false"])
    px_m = cam["size"] * PERSON_AREA_M / FOOTPRINT
    return [replace(m, table=table, px_m=px_m, frames_per_look=frames,
                    operator_confirms=True) for m in r17.held_out_missions(3)]  # fmt: skip


def tune_task(task: tuple) -> dict[str, Any]:
    cond, i, run2, generations = task
    out = rf.tune_mission(missions_for(cond)[i], i, run2, generations)
    out["condition"] = list(cond)
    return out


def eval_task(task: tuple) -> tuple[tuple, dict[str, float]]:
    key, cond, i, plan = task
    m = missions_for(cond)[i]
    res = run_missions(m, plan, r17.mission_seeds(r17.TEST_BASE, i, TEST_EPISODES))
    return key, {k: float(res[k].mean()) for k in ("score", "found", "wrong")}


def eval_tasks(tuned: dict[tuple, dict], step2: list[dict]) -> list[tuple]:
    out = []
    for cond in CONDITIONS:
        ms = missions_for(cond)
        for i, m in enumerate(ms):
            out.append(((cond, "OpenSwarm default", 0, i), cond, i, OPENSWARM_DEFAULT))
            plan = r17.decode_plan(np.array(tuned[(cond, i)]["u"]), m.drones, 2)
            out.append(((cond, "mission-time tuning", 0, i), cond, i, plan))
        for r, run in enumerate(step2):
            plans = r17.Planner(r17.PlannerSettings(**run["settings"])).plans(
                run["theta"], ms)  # fmt: skip
            out += [
                ((cond, "step-2 planner", r, i), cond, i, p)
                for i, p in enumerate(plans)
            ]
    return out


def summarise(cells: dict[tuple, dict]) -> dict[str, Any]:
    rows = []
    for cond in CONDITIONS:
        row: dict[str, Any] = {"camera": cond[0], "fa_mult": cond[1], "frames": cond[2]}
        for system in ("OpenSwarm default", "step-2 planner", "mission-time tuning"):
            vals = [v for k, v in cells.items() if k[0] == cond and k[1] == system]
            row[system] = {m: float(np.mean([v[m] for v in vals]))
                           for m in ("score", "found", "wrong")}  # fmt: skip
        rows.append(row)
    rec = next(
        r for r in rows if (r["camera"], r["fa_mult"], r["frames"]) == RECOMMENDED
    )
    best = rec["mission-time tuning"]
    return {
        "rows": rows,
        "recommended": best,
        "pass": bool(best["found"] >= FOUND_BAR and best["wrong"] <= WRONG_BAR),
    }


def render(rep: dict[str, Any]) -> str:
    b = rep["recommended"]
    lines = ["# R17 step 6: the recommended search system on realistic missions", "",
             f"**{'PASS' if rep['pass'] else 'FAIL'}.** The recommended system (448 camera, "
             "false detections cut 10×, 3 frames per look, operator confirms, mission-time "
             f"tuning) finds the person in {100 * b['found']:.1f}% of episodes with "
             f"{100 * b['wrong']:.1f}% wrong finds (bar: ≥ {100 * FOUND_BAR:.0f}% and ≤ "
             f"{100 * WRONG_BAR:.0f}%).", "",
             "Step 3's 40 held-out missions (half with a walking person; GPS error, wind, "
             f"battery vary; the moving hint is right 85% of the time) × {TEST_EPISODES} "
             "episodes. Found / wrong finds, averaged over missions.", "",
             "| Camera | False detections | Frames per look | OpenSwarm default | Step-2 "
             "planner | Mission-time tuning |", "|---|---|---|---|---|---|"]  # fmt: skip
    for r in rep["rows"]:
        cells = [f"{100 * r[s]['found']:.0f}% / {100 * r[s]['wrong']:.1f}%"
                 for s in ("OpenSwarm default", "step-2 planner", "mission-time tuning")]  # fmt: skip
        lines.append(f"| {r['camera']} | ×{r['fa_mult']:g} | {r['frames']:g} | "
                     + " | ".join(cells) + " |")  # fmt: skip
    lines += ["", "Assumptions: a person is 0.34 m in R15b's size measure; the OV2640's ~60° "
              "lens on both cameras; the operator recognises a person in view 90% of the time "
              "per look and wrongly accepts an empty view 2% of the time; the ×0.1 filter cuts "
              "false detections and leaves recall unchanged.", "",
              "Reproduce: `python scripts/r17g_system.py --from runs:r17b-train-cond-*`.", ""]  # fmt: skip
    return "\n".join(lines)
