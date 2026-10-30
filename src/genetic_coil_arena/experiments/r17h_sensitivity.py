"""R17 step 6b: which assumption behind the recommended search system matters most?

Step 6's recommended system (448-px camera, false alarms cut 10 times, 3 independent frames
per second, operator confirms, mission-time tuning) passed its bar narrowly (52.3% found,
4.8% wrong finds) on assumptions nobody has measured yet. This step moves one assumption
at a time (``VARIANTS``) and replans every mission at mission time, so the ranking shows
which bench or video measurement to make first. No bar: a sensitivity measurement.
Read-out, fixed in advance: the variants that drop the share found below 40% or push wrong
finds above 5%.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..envs.swarm.mission import FOOTPRINT, Mission, run_missions
from . import r17_mission as r17
from . import r17f_mission_time as rf
from . import r17g_system as rg

BASE = {"frames": 3.0, "recall_mult": 1.0, "fa_mult": 0.1, "op_hit": 0.9,
        "op_false": 0.02, "person_m": 0.34}  # fmt: skip
VARIANTS: dict[str, dict[str, float]] = {
    "recommended (as in step 6)": {},
    "1 frame per second of video": {"frames": 1.0},
    "2 frames per second of video": {"frames": 2.0},
    "filter keeps 80% of detections": {"recall_mult": 0.8},
    "filter keeps 60% of detections": {"recall_mult": 0.6},
    "false alarms cut 5x (not 10x)": {"fa_mult": 0.2},
    "false alarms cut 20x": {"fa_mult": 0.05},
    "operator sees a person 70% of looks": {"op_hit": 0.7},
    "operator sees a person 50% of looks": {"op_hit": 0.5},
    "operator accepts 5% of empty views": {"op_false": 0.05},
    "operator accepts 10% of empty views": {"op_false": 0.10},
    "person smaller (0.28 m)": {"person_m": 0.28},
    "person larger (0.40 m)": {"person_m": 0.40},
}
FOUND_FLOOR, WRONG_CEIL = 0.40, 0.05


def missions_for(variant: str) -> list[Mission]:
    v = {**BASE, **VARIANTS[variant]}
    cam = rg.camera("448 camera")
    recall = tuple(min(1.0, r * v["recall_mult"]) for r in cam["recall"])
    table = recall + tuple(f * v["fa_mult"] for f in cam["false"])
    px_m = cam["size"] * v["person_m"] / FOOTPRINT
    return [replace(m, table=table, px_m=px_m, frames_per_look=v["frames"],
                    operator_confirms=True, op_hit=v["op_hit"], op_false=v["op_false"])
            for m in r17.held_out_missions(3)]  # fmt: skip


def tune_task(task: tuple) -> dict[str, Any]:
    variant, i, run2, generations = task
    out = rf.tune_mission(missions_for(variant)[i], i, run2, generations)
    out["variant"] = variant
    return out


def eval_task(task: tuple) -> tuple[tuple, dict[str, float]]:
    variant, i, u = task
    m = missions_for(variant)[i]
    plan = r17.decode_plan(np.array(u), m.drones, 2)
    res = run_missions(m, plan, r17.mission_seeds(r17.TEST_BASE, i, rg.TEST_EPISODES))
    return (variant, i), {k: float(res[k].mean()) for k in ("score", "found", "wrong")}


def summarise(cells: dict[tuple, dict]) -> dict[str, Any]:
    rows = []
    for variant in VARIANTS:
        vals = [v for k, v in cells.items() if k[0] == variant]
        if vals:
            rows.append({"variant": variant, **{m: float(np.mean([v[m] for v in vals]))
                                                for m in ("score", "found", "wrong")}})  # fmt: skip
    base = rows[0]
    for r in rows:
        r["found_change"] = r["found"] - base["found"]
        r["breaks"] = bool(r["found"] < FOUND_FLOOR or r["wrong"] > WRONG_CEIL)
    return {"rows": rows, "breaking": [r["variant"] for r in rows if r["breaks"]]}


def render(rep: dict[str, Any]) -> str:
    lines = ["# R17 step 6b: which assumption matters most?", "",
             f"Read-out (fixed in advance): variants that drop the share found below "
             f"{100 * FOUND_FLOOR:.0f}% or push wrong finds above {100 * WRONG_CEIL:.0f}%: "
             + (", ".join(rep["breaking"]) or "none") + ".", "",
             "Step 3's 40 held-out missions × 2,000 episodes; every mission replanned at "
             "mission time for each variant; one assumption moved at a time from step 6's "
             "recommended system.", "",
             "| Variant | Found | Change | Wrong finds |", "|---|---|---|---|"]  # fmt: skip
    for r in sorted(rep["rows"], key=lambda r: r["found_change"]):
        lines.append(f"| {r['variant']} | {100 * r['found']:.1f}% | "
                     f"{100 * r['found_change']:+.1f} | {100 * r['wrong']:.1f}% |")  # fmt: skip
    lines += ["", "Reproduce: `python scripts/r17h_sensitivity.py --from "
              "runs:r17b-train-cond-*`.", ""]  # fmt: skip
    return "\n".join(lines)
