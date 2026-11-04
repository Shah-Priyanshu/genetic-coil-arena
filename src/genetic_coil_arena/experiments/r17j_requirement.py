"""R17 step 6h: the false-accept rate the evidence chain must reach.

Step 6g showed that planning cannot buy slack under the wrong-find limit: a stiffer
wrong-find cost only makes the search declare less. The margin has to come from a lower
rate at which the chain (scout, confirmer, closer look, operator) accepts an empty
candidate. This step measures the requirement: the search with the closer look (people
kept 92%, as measured), every held-out mission tuned at mission time, over a grid of
false-accept rates per candidate (``OP_FALSE``) at wrong-find costs 2 (the mission score)
and 3 (step 6g's post-hoc setting).

No bar (a requirements measurement, like R17 step 4). Read-out, fixed before the run:
per cost, the largest false-accept rate in the grid at which found >= 40% and wrong finds
<= 5% hold, and the found and wrong finds there; and that rate as a multiple of R19's
joint rate with the operator (4.9%).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..envs.swarm.mission import Mission, run_missions
from ..evaluation import mean_ci
from . import r17_mission as r17
from . import r17f_mission_time as rf
from . import r17g_system as rg
from . import r17h_sensitivity as rh
from . import r17i_wrong_cost as ri

OP_FALSE = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04)
COSTS = (2.0, 3.0)
R19_JOINT = 0.049
FOUND_FLOOR, WRONG_CEIL = rh.FOUND_FLOOR, rh.WRONG_CEIL


def missions_for(op_false: float) -> list[Mission]:
    return [replace(m, op_false=op_false) for m in rh.missions_for(ri.MEASURED)]


def tune_task(task: tuple) -> dict[str, Any]:
    op_false, cost, i, run2, generations = task
    out = rf.tune_mission(missions_for(op_false)[i], i, run2, generations, cost)
    out.update(op_false=op_false, cost=cost)
    return out


def eval_task(task: tuple) -> tuple[tuple, dict[str, float]]:
    op_false, cost, i, u = task
    m = missions_for(op_false)[i]
    res = run_missions(m, r17.decode_plan(np.array(u), m.drones, 2),
                       r17.mission_seeds(r17.TEST_BASE, i, rg.TEST_EPISODES))  # fmt: skip
    return (op_false, cost, i), {k: float(res[k].mean()) for k in ("found", "wrong")}


def summarise(cells: dict[tuple, dict]) -> dict[str, Any]:
    rows = []
    for op_false, cost in sorted({k[:2] for k in cells}, key=lambda k: (k[1], k[0])):
        vals = [v for k, v in cells.items() if k[:2] == (op_false, cost)]
        found, found_hw = mean_ci([v["found"] for v in vals])
        wrong, wrong_hw = mean_ci([v["wrong"] for v in vals])
        rows.append({"op_false": op_false, "cost": cost, "n": len(vals), "found": found,
                     "found_hw": found_hw, "wrong": wrong, "wrong_hw": wrong_hw,
                     "meets": bool(found >= FOUND_FLOOR and wrong <= WRONG_CEIL)})  # fmt: skip
    need = {}
    for cost in sorted({r["cost"] for r in rows}):
        ok = [r for r in rows if r["cost"] == cost and r["meets"]]
        best = max(ok, key=lambda r: r["op_false"]) if ok else None
        need[f"{cost:g}"] = best and {**best, "of_r19": best["op_false"] / R19_JOINT}
    return {"rows": rows, "requirement": need}


def render(rep: dict[str, Any]) -> str:
    lines = ["# R17 step 6h: the false-accept rate the evidence chain must reach", ""]
    for cost, r in rep["requirement"].items():
        lines.append(f"- **Cost {cost}:** " + (
            f"the largest false-accept rate per candidate that meets the limits is "
            f"{100 * r['op_false']:.1f}% ({r['of_r19']:.2f}x R19's 4.9%): "
            f"{100 * r['found']:.1f}% found, {100 * r['wrong']:.1f}% wrong finds."
            if r else "no rate in the grid meets the limits."))  # fmt: skip
    lines += ["", f"Limits: found >= {100 * FOUND_FLOOR:.0f}%, wrong finds <= "
              f"{100 * WRONG_CEIL:.0f}%. Step 3's 40 held-out missions x 2,000 test "
              "episodes; people kept 92% (R19, R20b); every mission tuned at mission "
              "time with found - cost x wrong; 95% CI over missions.", "",
              "| Cost | False accept per candidate | Found | Wrong finds | Meets |",
              "|---|---|---|---|---|"]  # fmt: skip
    for r in rep["rows"]:
        lines.append(f"| {r['cost']:g} | {100 * r['op_false']:.1f}% | "
                     f"{100 * r['found']:.1f}% ± {100 * r['found_hw']:.1f} | "
                     f"{100 * r['wrong']:.1f}% ± {100 * r['wrong_hw']:.1f} | "
                     f"{'yes' if r['meets'] else 'no'} |")  # fmt: skip
    lines += ["", "Reproduce: `python scripts/r17j_requirement.py --from "
              "runs:r17b-train-cond-*`.", ""]  # fmt: skip
    return "\n".join(lines)
