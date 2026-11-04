"""R17 step 6g: plan to the wrong-find limit.

Step 6f put R20b's closer look into the search: 51.9% found with 4.7% wrong finds at the
measured cut, but 5.6% at the cut's upper bound (limit 5%), so no slack. Mission-time
tuning maximises found - 2 x wrong; a stiffer wrong-find cost lets it trade finds for
fewer wrong ones (higher confirm thresholds, more looks, a lower confirming hover). This
step tunes every held-out mission at wrong-find costs ``COSTS`` under R20b's three rates
and R19's, then runs the plans tuned for the upper bound in the measured and replicate
worlds too (the ground station plans for the pessimistic case and meets the real one).

Read-out, fixed before the run:

- W* = the smallest cost at which the upper-bound variant's wrong finds are <= 5%.
- PASS if at W* the upper bound and the measured cut both find >= 40% with wrong finds
  <= 5%, and the upper-bound plans flown at the measured cut also keep wrong <= 5% and
  found >= 40%.
- FAIL if no cost gets there.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..envs.swarm.mission import run_missions
from ..evaluation import mean_ci
from . import r17_mission as r17
from . import r17f_mission_time as rf
from . import r17g_system as rg
from . import r17h_sensitivity as rh

MEASURED = "R20b: confirmer and closer look at the hover, then operator (92% / 2.1%)"
REPLICATE = "R20b: the same at the replicate's cut (92% / 1.3%)"
UPPER = "R20b: the same at the cut's upper bound (92% / 3.2%)"
R19 = "R19: confirmer at the ground, then operator (92% / 5%)"
PLANNED = (MEASURED, REPLICATE, UPPER, R19)
SHORT = {MEASURED: "measured cut (2.1%)", REPLICATE: "replicate's cut (1.3%)",
         UPPER: "cut's upper bound (3.2%)", R19: "R19 alone (4.9%)"}  # fmt: skip
COSTS = (2.0, 4.0, 8.0, 16.0)
FOUND_FLOOR, WRONG_CEIL = rh.FOUND_FLOOR, rh.WRONG_CEIL


def tune_task(task: tuple) -> dict[str, Any]:
    variant, cost, i, run2, generations = task
    out = rf.tune_mission(rh.missions_for(variant)[i], i, run2, generations, cost)
    out.update(variant=variant, cost=cost)
    return out


def eval_tasks(tuned: list[dict]) -> list[tuple]:
    """(plan's variant, world flown, cost, mission, plan): each plan in its own world,
    and the upper bound's plans in the measured and replicate worlds too."""
    out = []
    for r in tuned:
        worlds = [r["variant"]] + ([MEASURED, REPLICATE] if r["variant"] == UPPER else [])
        out += [(r["variant"], w, r["cost"], r["mission"], r["u"]) for w in worlds]
    return out


def eval_task(task: tuple) -> tuple[tuple, dict[str, float]]:
    plan_v, world, cost, i, u = task
    m = rh.missions_for(world)[i]
    res = run_missions(m, r17.decode_plan(np.array(u), m.drones, 2),
                       r17.mission_seeds(r17.TEST_BASE, i, rg.TEST_EPISODES))  # fmt: skip
    return (plan_v, world, cost, i), {k: float(res[k].mean()) for k in ("found", "wrong")}


def summarise(cells: dict[tuple, dict]) -> dict[str, Any]:
    rows = []
    for plan_v, world, cost in sorted({k[:3] for k in cells}, key=lambda k: (
            PLANNED.index(k[0]), PLANNED.index(k[1]), k[2])):  # fmt: skip
        vals = [v for k, v in cells.items() if k[:3] == (plan_v, world, cost)]
        found, found_hw = mean_ci([v["found"] for v in vals])
        wrong, wrong_hw = mean_ci([v["wrong"] for v in vals])
        rows.append({"plan": plan_v, "world": world, "cost": cost, "n": len(vals),
                     "found": found, "found_hw": found_hw, "wrong": wrong,
                     "wrong_hw": wrong_hw,
                     "meets": bool(found >= FOUND_FLOOR and wrong <= WRONG_CEIL)})  # fmt: skip

    def row(plan_v: str, world: str, cost: float) -> dict | None:
        return next((r for r in rows if (r["plan"], r["world"], r["cost"]) ==
                     (plan_v, world, cost)), None)  # fmt: skip

    w_star = next((c for c in COSTS if (r := row(UPPER, UPPER, c)) and
                   r["wrong"] <= WRONG_CEIL), None)  # fmt: skip
    checks = {}
    if w_star is not None:
        checks = {"upper bound planned and flown": row(UPPER, UPPER, w_star)["meets"],
                  "measured cut planned and flown": row(MEASURED, MEASURED, w_star)["meets"],
                  "upper bound's plans flown at the measured cut":
                      row(UPPER, MEASURED, w_star)["meets"]}  # fmt: skip
    verdict = "PASS" if checks and all(checks.values()) else "FAIL"
    return {"rows": rows, "w_star": w_star, "checks": checks, "verdict": verdict}


def render(rep: dict[str, Any]) -> str:
    w = rep["w_star"]
    lines = ["# R17 step 6g: plan to the wrong-find limit", "",
             f"**Verdict: {rep['verdict']}.** W* (smallest wrong-find cost at which the "
             f"upper bound's wrong finds are <= {100 * WRONG_CEIL:.0f}%): "
             + (f"{w:g}" if w is not None else "none") + ".", ""]  # fmt: skip
    for name, ok in rep["checks"].items():
        lines.append(f"- {name}: {'meets' if ok else 'misses'} the limits "
                     f"(found >= {100 * FOUND_FLOOR:.0f}%, wrong <= {100 * WRONG_CEIL:.0f}%)")
    lines += ["", "Step 3's 40 held-out missions x 2,000 test episodes; each mission tuned "
              "at mission time with found - cost x wrong as the objective (the mission "
              "score is found - 2 x wrong); 95% CI over missions.", "",
              "| Planned for | Flown in | Cost | Found | Wrong finds | Meets |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for r in rep["rows"]:
        lines.append(f"| {SHORT[r['plan']]} | {SHORT[r['world']]} | {r['cost']:g} | "
                     f"{100 * r['found']:.1f}% ± {100 * r['found_hw']:.1f} | "
                     f"{100 * r['wrong']:.1f}% ± {100 * r['wrong_hw']:.1f} | "
                     f"{'yes' if r['meets'] else 'no'} |")  # fmt: skip
    lines += ["", "Reproduce: `python scripts/r17i_wrong_cost.py --from "
              "runs:r17b-train-cond-*`.", ""]  # fmt: skip
    return "\n".join(lines)
