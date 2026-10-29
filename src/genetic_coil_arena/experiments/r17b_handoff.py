"""R17 step 2: a hand-off tactic for people who walk, and a planner trained for it.

Step 1b found that every plan that confirms collapses when the person walks: the
confirmer hovers over where the person was. Version 2 of the plan adds the hand-off
(``Plan.track_s``: the scout that makes a detection holds it in view and moves the
candidate with each re-sighting until a confirmer takes over). Version 2 missions
(``r17_mission.sample_mission(rng, 2)``) are half walking (0.2-1.5 m/s), with 1-4 m GPS
error, 3-5 m/s cruise (wind), a 220-320 s battery and a find radius that grows with the
GPS error (``find_radius``). The leader reads the GPS quality, wind and battery, never
the person's speed. Planners (16 blind, 16 conditioned) train on version 2 missions
with the step-1 budget.

Systems on the 40 version 2 held-out missions (2,000 episodes each, paired by seed):
OpenSwarm default; hand two-tier; hand two-tier with a 60 s hand-off; the step-1
planner (16 runs, no hand-off); version 2 blind and conditioned. The step-1 planner and
the version 2 planner also play the step-1 held-out missions (a still person).

Bars (fixed before any version 2 training run), CIs over missions:

- **(a) hand-off:** on the walking held-out missions, hand two-tier with the hand-off
  beats hand two-tier by >= 10 points, CI above 0;
- **(b) planner:** the version 2 conditioned planner beats the version 2 blind plan by
  >= 3 points on all 40 missions, CI above 0;
- **(c) fixes step 1b:** on the walking missions the version 2 planner beats the
  step-1 planner by >= 10 points, CI above 0;
- **(d) no regression:** on the step-1 held-out missions it is at most 3 points below
  the step-1 planner.

Disclosure: a pipeline smoke test (2 planners trained 30 generations, 100 episodes per
cell) ran on the held-out missions after these bars were written and before they were
committed. Bar (a) involves no training, so that smoke previewed it (+7.9 ± 5.1 points
against +10); the bars were not changed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..envs.swarm.mission import OPENSWARM_DEFAULT, Plan, hand_two_tier, run_missions
from ..evaluation import mean_ci
from . import r17_mission as r17

HANDOFF_S = 60.0
EPISODES = 2000
BARS = {"handoff": 0.10, "planner": 0.03, "fixes": 0.10, "regression": -0.03}
SYSTEMS_V2 = ("OpenSwarm default", "hand two-tier", "hand two-tier + hand-off",
              "step-1 planner", "blind v2", "conditioned v2")  # fmt: skip


def plans_for(version: int, step1: list[dict], blind2: list[dict],
              cond2: list[dict]) -> dict[str, list[list[Plan]]]:  # fmt: skip
    """Per system, one list of plans (one per held-out mission of ``version``) per run."""
    ms = r17.held_out_missions(version)

    def learned(runs: list[dict]) -> list[list[Plan]]:
        return [r17.Planner(r17.PlannerSettings(**r["settings"])).plans(r["theta"], ms)
                for r in runs]  # fmt: skip

    out: dict[str, list[list[Plan]]] = {"step-1 planner": learned(step1),
                                        "conditioned v2": learned(cond2)}  # fmt: skip
    if version == 2:
        out = {
            "OpenSwarm default": [[OPENSWARM_DEFAULT] * len(ms)],
            "hand two-tier": [[hand_two_tier(m.drones) for m in ms]],
            "hand two-tier + hand-off": [[replace(hand_two_tier(m.drones),
                                                  track_s=HANDOFF_S) for m in ms]],
            **out, "blind v2": learned(blind2),
        }  # fmt: skip
    return out


def tasks(step1: list[dict], blind2: list[dict], cond2: list[dict],
          episodes: int = EPISODES) -> list[tuple]:  # fmt: skip
    out = []
    for version in (2, 1):
        for name, runs in plans_for(version, step1, blind2, cond2).items():
            for r, plans in enumerate(runs):
                for i, plan in enumerate(plans):
                    out.append(((version, name, r, i), version, i, plan, episodes))
    return out


def play_cell(task: tuple) -> tuple[tuple, dict[str, float]]:
    key, version, i, plan, episodes = task
    m = r17.held_out_missions(version)[i]
    res = run_missions(m, plan, r17.mission_seeds(r17.TEST_BASE, i, episodes))
    return key, {k: float(res[k].mean()) for k in ("score", "found", "wrong")}


def _per_mission(
    cells: dict, version: int, name: str, what: str = "score"
) -> np.ndarray:
    runs = sorted({k[2] for k in cells if k[0] == version and k[1] == name})
    return np.array([[cells[(version, name, r, i)][what] for i in range(r17.N_TEST)]
                     for r in runs]).mean(0)  # fmt: skip


def summarise(cells: dict[tuple, dict]) -> dict[str, Any]:
    walking = np.array([m.person_speed > 0 for m in r17.held_out_missions(2)])
    out: dict[str, Any] = {"v2": {}, "v1": {}, "walking_missions": int(walking.sum())}
    for version, names in ((2, SYSTEMS_V2), (1, ("step-1 planner", "conditioned v2"))):
        for name in names:
            row = {w: float(_per_mission(cells, version, name, w).mean())
                   for w in ("score", "found", "wrong")}  # fmt: skip
            if version == 2:
                s = _per_mission(cells, 2, name)
                row["walking"] = float(s[walking].mean())
                row["still"] = float(s[~walking].mean())
            out[f"v{version}"][name] = row
    s2 = {n: _per_mission(cells, 2, n) for n in SYSTEMS_V2}
    s1 = {n: _per_mission(cells, 1, n) for n in ("step-1 planner", "conditioned v2")}
    diffs = {
        "handoff": mean_ci((s2["hand two-tier + hand-off"] - s2["hand two-tier"])[walking]
                           .tolist()),
        "planner": mean_ci((s2["conditioned v2"] - s2["blind v2"]).tolist()),
        "fixes": mean_ci((s2["conditioned v2"] - s2["step-1 planner"])[walking].tolist()),
        "regression": mean_ci((s1["conditioned v2"] - s1["step-1 planner"]).tolist()),
    }  # fmt: skip
    bars = {}
    for key, d in diffs.items():
        if key == "regression":
            bars[key] = bool(d[0] >= BARS[key])
        else:
            bars[key] = bool(d[0] >= BARS[key] and d[0] - d[1] > 0)
    out.update(diffs=diffs, bars=bars, passed=all(bars.values()))
    return out


LABELS = {
    "handoff": "(a) hand-off: hand two-tier + hand-off − hand two-tier, walking missions",
    "planner": "(b) planner: conditioned v2 − blind v2, all missions",
    "fixes": "(c) fixes step 1b: conditioned v2 − step-1 planner, walking missions",
    "regression": "(d) no regression: conditioned v2 − step-1 planner, step-1 missions",
}


def render(rep: dict[str, Any], episodes: int = EPISODES) -> str:
    lines = ["# R17 step 2: hand-off for walking people, and a planner trained for it", "",
             f"**{'GOOD SIGN' if rep['passed'] else 'NOT YET'}** (all four bars "
             f"{'met' if rep['passed'] else 'needed'}).", "",
             "| Bar | Difference (points, 95% CI over missions) | Needed | Met |",
             "|---|---|---|---|"]  # fmt: skip
    for key, label in LABELS.items():
        d = rep["diffs"][key]
        need = f"≥ {100 * BARS[key]:+.0f}" + ("" if key == "regression" else ", CI > 0")
        lines.append(f"| {label} | {100 * d[0]:+.1f} ± {100 * d[1]:.1f} | {need} | "
                     f"{'yes' if rep['bars'][key] else 'no'} |")  # fmt: skip
    lines += ["", f"Version 2 held-out missions ({r17.N_TEST}, {rep['walking_missions']} "
              f"with a walking person) × {episodes} episodes; score = found − 2 × wrong "
              "find, in points.", "",
              "| System | Score | Walking | Still | Found | Wrong find |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for name, r in rep["v2"].items():
        lines.append(f"| {name} | {100 * r['score']:.1f} | {100 * r['walking']:.1f} | "
                     f"{100 * r['still']:.1f} | {100 * r['found']:.1f}% | "
                     f"{100 * r['wrong']:.1f}% |")  # fmt: skip
    lines += ["", "Step-1 held-out missions (a still person, nominal):", "",
              "| System | Score | Found | Wrong find |", "|---|---|---|---|"]  # fmt: skip
    for name, r in rep["v1"].items():
        lines.append(f"| {name} | {100 * r['score']:.1f} | {100 * r['found']:.1f}% | "
                     f"{100 * r['wrong']:.1f}% |")  # fmt: skip
    lines += ["", "Reproduce: `python scripts/r17b_handoff.py --from runs:r17-train-* "
              "runs:r17b-train-*`.", ""]  # fmt: skip
    return "\n".join(lines)
