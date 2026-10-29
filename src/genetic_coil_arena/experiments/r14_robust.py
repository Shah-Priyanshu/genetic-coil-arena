"""R14 step 3: do the learned swarm brains keep their gain when conditions get worse?

Step 2 (``r14_brains``) trained per-role brains on one setting (3 sources, 30% loss,
drift 0.03, 6 drones) and found all three leaks 15.9 points more often than the tuned hand
design. Real flights will not match that setting. This step plays the step-2 brains, the
hand design and the sensed oracle in ``CONDITIONS``: worse radio, more drift, a weaker
gas sensor, other plume widths, drones going down mid-mission (``SwarmConfig.failures``),
fewer drones and a larger arena. Neither the brains nor the hand knobs are retuned.

It also trains a second set of brains (``BrainSettings.randomise``) that play a random
condition every generation (``r14_brains.random_condition``), with the same budget as step
2, and asks whether that buys robustness. The 4-drone and 40 m conditions lie outside
both training ranges; the others lie inside the randomised one (``IN_RANDOM_RANGE``).

Episodes: the probe's test seeds (10 replicates × 300 episodes) at 3 sources. Every system
plays the same episodes, so differences are paired by replicate; CIs are over the 10
replicates (t), with the learned rate averaged over the training runs of a set.

Bars (fixed before any stress run):

- **3a, robust:** in every stress condition the step-2 brains find all leaks more often
  than the hand design, with the replicate CI above 0.
- **3b, randomised training helps:** averaged over the stress conditions, the randomised
  brains beat the step-2 brains (replicate CI above 0), and at the nominal setting they
  stay within 3 points of them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..envs.swarm import SwarmEnv, run_swarm
from ..envs.swarm.multi import MultiCodec, MultiTargetSwarm, SensedOracleSwarm
from ..evaluation import mean_ci
from . import r14_brains as rb
from . import r14_multi_probe as probe

K = 3
CONDITIONS: dict[str, dict[str, Any]] = {
    "nominal": {},
    "loss 50%": {"loss": 0.5},
    "loss 70%": {"loss": 0.7},
    "drift x3": {"drift": 0.09},
    "weak gas sensor": {"hit_rate": 0.4, "gas_noise": 0.4},
    "narrow plume": {"plume_sigma": 2.0},
    "wide plume": {"plume_sigma": 4.5},
    "1 drone down": {"failures": 1},
    "2 drones down": {"failures": 2},
    "4 drones (2+2)": {"roles": ("gas", "gas", "camera", "camera")},
    "40 m arena": {"arena": 40.0, "source_box": (8.0, 38.0)},
}
IN_RANDOM_RANGE = frozenset(CONDITIONS) - {"4 drones (2+2)", "40 m arena"}
REFS = ("hand 2B", "sensed oracle")
NOMINAL_SLACK = 0.03


def seeds(reps: int = probe.N_REPS, per_rep: int = probe.PER_REP) -> list[int]:
    return [probe.TEST_BASE + r * probe.STRIDE + i
            for r in range(reps) for i in range(per_rep)]  # fmt: skip


def play_ref(
    system: str, over: dict[str, Any], eps: list[int]
) -> dict[str, np.ndarray]:
    cfg = rb.make_policy(rb.BrainSettings(), K, over)[0]
    if system == "hand 2B":
        pol: Any = MultiTargetSwarm(
            cfg, MultiCodec(2, probe.POS_BITS[2], cfg.arena), **rb.HAND_KNOBS
        )
    else:
        pol = SensedOracleSwarm(cfg)
    return run_swarm(SwarmEnv(cfg), pol, eps)


def play_brain(run: dict[str, Any], over: dict[str, Any], eps: list[int]) -> dict:
    settings = rb.BrainSettings(**run["settings"])
    cfg, pol = rb.make_policy(settings, K, over)
    gas_p, cam_p = rb.split(run["theta"], pol)
    pol.set_params(gas_p, cam_p, len(eps))
    return run_swarm(SwarmEnv(cfg), pol, eps)


def play_task(task: tuple) -> tuple[tuple, dict[str, list[float]]]:
    """One (condition, system) cell -> per-replicate all-found and found rates."""
    key, cond, what, reps, per_rep = task
    over = CONDITIONS[cond]
    eps = seeds(reps, per_rep)
    got = play_ref(what, over, eps) if isinstance(what, str) else play_brain(
        what, over, eps)  # fmt: skip
    shape = (reps, per_rep)
    return key, {
        "all": got["confirmed"].astype(float).reshape(shape).mean(1).tolist(),
        "found": got["n_found"].astype(float).reshape(shape).mean(1).tolist(),
    }


def tasks(sets: dict[str, list[dict]], reps: int, per_rep: int,
          conditions: list[str] | None = None) -> list[tuple]:  # fmt: skip
    out = []
    for cond in conditions or list(CONDITIONS):
        for ref in REFS:
            out.append(((cond, ref, -1), cond, ref, reps, per_rep))
        for name, runs in sets.items():
            for r in runs:
                out.append(((cond, name, r["run"]), cond, r, reps, per_rep))
    return out


def summarise(cells: dict[tuple, dict], sets: dict[str, list[dict]]) -> dict[str, Any]:
    conds = sorted({c for c, _, _ in cells}, key=list(CONDITIONS).index)
    by: dict[str, Any] = {}
    rep_all: dict[str, dict[str, np.ndarray]] = {}
    for cond in conds:
        row: dict[str, Any] = {}
        alls: dict[str, np.ndarray] = {}
        for ref in REFS:
            alls[ref] = np.array(cells[(cond, ref, -1)]["all"])
            row[ref] = {"all": float(alls[ref].mean()),
                        "found": float(np.mean(cells[(cond, ref, -1)]["found"]))}  # fmt: skip
        for name, runs in sets.items():
            per_run = np.array([cells[(cond, name, r["run"])]["all"] for r in runs])
            found = np.array([cells[(cond, name, r["run"])]["found"] for r in runs])
            alls[name] = per_run.mean(0)
            row[name] = {
                "all": float(per_run.mean()),
                "found": float(found.mean()),
                "runs_all": per_run.mean(1).tolist(),
                "minus_hand": mean_ci((alls[name] - alls["hand 2B"]).tolist()),
            }
        by[cond] = row
        rep_all[cond] = alls
    out: dict[str, Any] = {"by_condition": by, "conditions": conds, "sets": list(sets)}
    stress = [c for c in conds if c != "nominal"]
    if "step2" in sets and stress:
        fails = [c for c in stress if by[c]["step2"]["minus_hand"][0]
                 - by[c]["step2"]["minus_hand"][1] <= 0]  # fmt: skip
        out["robust"] = {"pass": not fails, "fails": fails}
    if {"step2", "random"} <= set(sets) and stress and "nominal" in conds:
        d = np.mean([rep_all[c]["random"] - rep_all[c]["step2"] for c in stress], 0)
        diff = mean_ci(d.tolist())
        nominal = by["nominal"]["random"]["all"] - by["nominal"]["step2"]["all"]
        out["randomised"] = {
            "stress_minus_step2": diff,
            "nominal_minus_step2": float(nominal),
            "pass": bool(diff[0] - diff[1] > 0 and nominal >= -NOMINAL_SLACK),
        }
    return out


def _pm(ci: list[float] | tuple[float, float]) -> str:
    return f"{100 * ci[0]:+.1f} ± {100 * ci[1]:.1f}"


def render(rep: dict[str, Any], n_runs: dict[str, int], episodes: int) -> str:
    sets = rep["sets"]
    lines = ["# R14 step 3: do the learned swarm brains survive worse conditions?", ""]
    if "robust" in rep:
        r = rep["robust"]
        lines.append(
            f"**3a, robust: {'PASS' if r['pass'] else 'FAIL'}.** The step-2 brains beat "
            "the hand design with the replicate CI above 0 in "
            + ("every stress condition." if r["pass"]
               else f"all but: {', '.join(r['fails'])}.")  # fmt: skip
        )
    if "randomised" in rep:
        r = rep["randomised"]
        lines.append(
            f"\n**3b, randomised training helps: {'PASS' if r['pass'] else 'FAIL'}.** "
            f"Averaged over the stress conditions the randomised brains find all leaks "
            f"{_pm(r['stress_minus_step2'])} points more often than the step-2 brains; at "
            f"the nominal setting {100 * r['nominal_minus_step2']:+.1f} points (allowed: "
            f"down to {-100 * NOMINAL_SLACK:.0f})."
        )
    head = "| Condition | Hand design | " + " | ".join(
        f"{s} (gain over hand)" for s in sets) + " | Sensed oracle |"  # fmt: skip
    lines += ["", f"All three leaks found (3 sources, {episodes} episodes per cell; gains "
              "are replicate means ± 95% CI).", "", head,
              "|---|---|" + "---|" * len(sets) + "---|"]  # fmt: skip
    for cond in rep["conditions"]:
        row = rep["by_condition"][cond]
        mark = "" if cond in IN_RANDOM_RANGE else " *"
        cells = [f"{100 * row['hand 2B']['all']:.1f}%"]
        for s in sets:
            cells.append(f"{100 * row[s]['all']:.1f}% ({_pm(row[s]['minus_hand'])})")
        cells.append(f"{100 * row['sensed oracle']['all']:.1f}%")
        lines.append(f"| {cond}{mark} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "\\* outside the randomised training range (and the step-2 setting).",
        "",
        "Sets: "
        + ", ".join(f"`{s}` ({n_runs[s]} training runs)" for s in sets)
        + ". `step2` = trained at the nominal setting only (R14 step 2); `random` = a "
        "random condition every generation, same budget.",
        "",
        "Reproduce: `python scripts/r14_robust.py --set step2=runs:r14b-train-* "
        "--set random=runs:r14c-train-*`.",
        "",
    ]
    return "\n".join(lines)
