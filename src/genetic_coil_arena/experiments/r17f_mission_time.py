"""R17 step 5: plan at mission time instead of with one learned planner.

Steps 2-3b found that one planner trained by evolution over a broad range of missions
stalls well below what tuning each mission alone reaches, whatever its size or training
time. The ground station has a computer and this simulator, so it can tune the plan for
the actual mission just before launch. It knows the area, the drones, the detector, the
GPS quality, the wind, the battery and the operator's moving hint; it does not know
whether the person walks or how fast. So it tunes over ``VARIANTS`` versions of the
mission whose person speed is drawn from what the hint implies (``prior_variants``:
the hint is right 85% of the time; a walker moves at 0.2-1.5 m/s), with a small budget
(``GENERATIONS`` x ``POPULATION`` candidates, a few seconds to a minute of one CPU
core), from two starts: the hand plan with a 30 s hand-off, and the step-2 planner's
own plan for the mission. The start that scores best on separate check seeds wins.

Scored on the true mission's test seeds (2,000 episodes), paired by seed with the
step-1 planner, the step-2 planner (16 runs each) and, for reference, per-mission
tuning that knew the true mission (step 2's ``r17b-oracle``).

Bars (fixed before the run):

- **(a)** on step 3's held-out missions (hint 85% right), mission-time tuning beats the
  step-2 planner by >= 5 points, CI over missions above 0;
- **(b)** on step 1's still-person missions with the hint "still", it is at most 3
  points below the step-1 planner.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np

from ..envs.swarm.mission import Mission, hand_two_tier, run_missions
from ..evaluation import mean_ci
from ..learning import EvolutionStrategy
from . import r17_mission as r17

VARIANTS, EPISODES_PER_VARIANT = 8, 16
GENERATIONS, POPULATION = 100, 32
CHECK_EPISODES = 64
TEST_EPISODES = 2000
TUNE_BASE, CHECK_BASE = 700_000_000, 800_000_000
GAIN_BAR, STILL_BAR = 0.05, -0.03
SETS = ("hint 85% right", "still person, hint still")


def missions_for(which: str) -> list[Mission]:
    if which == "hint 85% right":
        return r17.held_out_missions(3)
    return [replace(m, moving_hint=-1.0) for m in r17.held_out_missions(1)]


def prior_variants(m: Mission, index: int, k: int = VARIANTS) -> list[Mission]:
    """The mission with the person's speed drawn from what the hint implies."""
    rng = np.random.default_rng([index, 1757])
    walk_p = {1.0: r17.HINT_ACCURACY, -1.0: 1 - r17.HINT_ACCURACY}.get(
        m.moving_hint, 0.5
    )
    return [replace(m, person_speed=float(rng.uniform(0.2, 1.5))
                    if rng.random() < walk_p else 0.0) for _ in range(k)]  # fmt: skip


def _fitness(u: np.ndarray, variants: list[Mission], base: int) -> float:
    return float(np.mean([
        run_missions(v, r17.decode_plan(u, v.drones, 2),
                     [base + j * 1000 + e for e in range(EPISODES_PER_VARIANT)])["score"].mean()
        for j, v in enumerate(variants)
    ]))  # fmt: skip


def tune_task(task: tuple) -> dict[str, Any]:
    """(set, mission index, the step-2 planner's run 0) -> the tuned plan."""
    which, i, run2 = task
    start = time.perf_counter()
    m = missions_for(which)[i]
    variants = prior_variants(m, i)
    learned = r17.Planner(r17.PlannerSettings(**run2["settings"])).plans(
        run2["theta"], [m]
    )[0]
    starts = {
        "hand-off": replace(hand_two_tier(m.drones), track_s=30.0),
        "planner": learned,
    }
    best: tuple[float, np.ndarray, str] = (-np.inf, np.zeros(r17.N_PLAN_V2), "")
    for k, (name, plan) in enumerate(starts.items()):
        es = EvolutionStrategy(r17.encode_plan(plan, m.drones, 2), population=POPULATION,
                               sigma=0.15, lr=0.05, weight_decay=0.0,
                               seed=i * 10 + k)  # fmt: skip
        for g in range(GENERATIONS):
            base = TUNE_BASE + i * 1_000_000 + (k * GENERATIONS + g) * 10_000
            es.tell(np.array([_fitness(c, variants, base) for c in es.ask()]))
        check = _fitness(es.theta, variants, CHECK_BASE + i * 100_000)
        if check > best[0]:
            best = (check, es.theta.copy(), name)
    return {"set": which, "mission": i, "u": best[1].tolist(), "start": best[2],
            "check": best[0], "seconds": time.perf_counter() - start}  # fmt: skip


def eval_task(task: tuple) -> tuple[tuple, float]:
    key, which, i, plan = task
    m = missions_for(which)[i]
    res = run_missions(m, plan, r17.mission_seeds(r17.TEST_BASE, i, TEST_EPISODES))
    return key, float(res["score"].mean())


def eval_tasks(tuned: dict[tuple, dict], step1: list[dict], step2: list[dict],
               oracle2: dict[int, dict]) -> list[tuple]:  # fmt: skip
    out = []
    for which in SETS:
        ms = missions_for(which)
        for name, runs in (("step-1 planner", step1), ("step-2 planner", step2)):
            for r, run in enumerate(runs):
                plans = r17.Planner(r17.PlannerSettings(**run["settings"])).plans(
                    run["theta"], ms)  # fmt: skip
                out += [((which, name, r, i), which, i, p) for i, p in enumerate(plans)]
        for i, m in enumerate(ms):
            plan = r17.decode_plan(np.array(tuned[(which, i)]["u"]), m.drones, 2)
            out.append(((which, "mission-time tuning", 0, i), which, i, plan))
            if which == "hint 85% right" and i in oracle2:
                plan = r17.decode_plan(np.array(oracle2[i]["u"]), m.drones, 2)
                out.append(((which, "tuned knowing the person", 0, i), which, i, plan))
    return out


def summarise(cells: dict[tuple, float], tuned: dict[tuple, dict]) -> dict[str, Any]:
    def per(which: str, name: str) -> np.ndarray:
        runs = sorted({k[2] for k in cells if k[0] == which and k[1] == name})
        return np.array([[cells[(which, name, r, i)] for i in range(r17.N_TEST)]
                         for r in runs]).mean(0)  # fmt: skip

    table = {which: {name: float(per(which, name).mean())
                     for name in sorted({k[1] for k in cells if k[0] == which})}
             for which in SETS}  # fmt: skip
    gain = mean_ci((per(SETS[0], "mission-time tuning") - per(SETS[0], "step-2 planner"))
                   .tolist())  # fmt: skip
    still = mean_ci((per(SETS[1], "mission-time tuning") - per(SETS[1], "step-1 planner"))
                    .tolist())  # fmt: skip
    bars = {"gain": bool(gain[0] >= GAIN_BAR and gain[0] - gain[1] > 0),
            "still": bool(still[0] >= STILL_BAR)}  # fmt: skip
    secs = [t["seconds"] for t in tuned.values()]
    starts = [t["start"] for t in tuned.values()]
    return {"table": table, "diffs": {"gain": gain, "still": still}, "bars": bars,
            "passed": all(bars.values()),
            "seconds": {"median": float(np.median(secs)), "max": float(np.max(secs))},
            "planner_start_share": starts.count("planner") / len(starts)}  # fmt: skip


def render(rep: dict[str, Any]) -> str:
    d, b = rep["diffs"], rep["bars"]
    lines = ["# R17 step 5: plan at mission time", "",
             f"**{'GOOD SIGN' if rep['passed'] else 'NOT YET'}.**", "",
             f"- (a) mission-time tuning − step-2 planner, hint 85% right: "
             f"{100 * d['gain'][0]:+.1f} ± {100 * d['gain'][1]:.1f} points (bar ≥ "
             f"+{100 * GAIN_BAR:.0f}, CI > 0): {'met' if b['gain'] else 'missed'}.",
             f"- (b) mission-time tuning − step-1 planner, still person with the hint "
             f"\"still\": {100 * d['still'][0]:+.1f} ± {100 * d['still'][1]:.1f} points "
             f"(bar ≥ {100 * STILL_BAR:.0f}): {'met' if b['still'] else 'missed'}.", "",
             f"Tuning took {rep['seconds']['median']:.0f} s per mission on one CPU core "
             f"(median; max {rep['seconds']['max']:.0f} s); the step-2 planner's own plan "
             f"was the better start in {100 * rep['planner_start_share']:.0f}% of missions.",
             "", "Score (found − 2 × wrong find, points), 40 held-out missions × "
             f"{TEST_EPISODES} episodes:", "", "| System | " + " | ".join(SETS) + " |",
             "|---|" + "---|" * len(SETS)]  # fmt: skip
    names = sorted({n for t in rep["table"].values() for n in t})
    for name in names:
        lines.append(f"| {name} | " + " | ".join(
            f"{100 * rep['table'][w][name]:.1f}" if name in rep["table"][w] else "—"
            for w in SETS) + " |")  # fmt: skip
    lines += ["", "Reproduce: `python scripts/r17f_mission_time.py --from "
              "runs:r17-train-cond-* runs:r17b-train-cond-* runs:r17b-oracle`.", ""]  # fmt: skip
    return "\n".join(lines)
