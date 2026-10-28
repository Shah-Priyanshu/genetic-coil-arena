"""R7c: is a goal carried only by learned messages reliable to train? (goal G7c)

G7 at 16 runs per condition (10k generations) found that condition (b), whose goal
reaches the specialists only through the 3-bit messages, failed to train in 2 of 16
runs (held-out 6.6 and 8.0 against 18.7-23.7), and that both failures already showed
on the training goals' validation score (5.7 and 8.2 against 18.1-22.4).

G7c trains (b) on 20 fresh runs (16-35) with the G7 settings and applies a restart
rule fixed before those runs: a run whose best validation mission on the training
goals is below ``FLAG_BELOW`` is discarded, and the first ``N_KEEP`` unflagged runs
in run order form the sample. Only training-goal data enters the rule.

References are the 16-run G7 report's hand-coded centre (d) and blind centre (a)
(runs 0-15, ``docs/results/r7b_goal_conditioned_16runs.json``), compared by Welch
t over independent runs.

Verdict: pass if the kept runs' held-out mean reaches ``PASS_FRACTION`` of (d);
kill if it stays below ``KILL_FRACTION`` (no better than (b) without the rule at
16 runs); otherwise not passed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..evaluation import mann_whitney, mean_ci
from ..translator.metrics import wilson
from .learned_protocol import welch_ci
from .r7b_goal_conditioned import CENTRE_GOAL, GoalTrainSettings
from .r7b_report import (
    BLIND,
    CONDITION,
    HAND_GOAL,
    HELDOUT,
    SCALES,
    SEEN,
    evaluate_goal_set,
)

FLAG_BELOW = 12.0
N_KEEP = 16
FRESH_RUNS = (16, 36)
PASS_FRACTION = 0.90
KILL_FRACTION = 0.85
REFERENCE = Path("docs/results/r7b_goal_conditioned_16runs.json")

B, A, D = CONDITION[CENTRE_GOAL], CONDITION[BLIND], CONDITION[HAND_GOAL]


def select_runs(training: Sequence[dict]) -> tuple[list[int], list[int]]:
    """(kept, flagged) run ids: the first ``N_KEEP`` unflagged runs in run order."""
    rows = sorted(training, key=lambda t: int(t["run"]))
    flagged = [int(t["run"]) for t in rows if t["best_val_mission"] < FLAG_BELOW]
    kept = [int(t["run"]) for t in rows if int(t["run"]) not in flagged][:N_KEEP]
    return kept, flagged


def g7c_verdict(kept: Sequence[float], hand: Sequence[float]) -> dict[str, Any]:
    k, d = np.asarray(kept, float), np.asarray(hand, float)
    frac = float(k.mean() / d.mean())
    if frac >= PASS_FRACTION:
        verdict = "PASS"
    elif frac < KILL_FRACTION:
        verdict = "KILL"
    else:
        verdict = "NOT PASSED"
    return {
        "verdict": verdict,
        "fraction_of_d": frac,
        "n_kept": int(k.size),
        "diff_vs_d": float(k.mean() - d.mean()),
        "ci_vs_d": welch_ci(k, d),
        "p_vs_d": float(mann_whitney(k, d)[1]),
    }


def load_reference(path: Path = REFERENCE) -> dict[str, Any]:
    """Held-out/seen per-run scores and training records of the 16-run G7 report."""
    ref = json.loads(path.read_text(encoding="utf-8"))
    per = {
        w: {c["name"]: c["per_seed"] for c in ref["sets"][w]["result"]["conditions"]}
        for w in (HELDOUT, SEEN)
    }
    seeds = ref["sets"][HELDOUT]["result"]["conditions"][0]["seeds"]
    return {"per": per, "seeds": seeds, "training": ref["training"]}


def _pm(values: Sequence[float]) -> str:
    m, ci = mean_ci(list(values))
    return f"{m:.2f} ± {ci:.2f}"


def build_report(
    trained: Sequence[dict],
    settings_d: dict[str, Any],
    reference: dict[str, Any],
    sets: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Evaluate the fresh (b) runs (unless ``sets`` is given) and compute G7c."""
    fresh = [t for t in trained if t["mode"] == CENTRE_GOAL]
    if len(fresh) != len(trained):
        raise ValueError("G7c evaluates centre-goal runs only")
    if sets is None:
        settings = GoalTrainSettings(**settings_d)
        esettings = SCALES["full"]["eval"]
        sets = {
            w: evaluate_goal_set(w, fresh, settings, esettings) for w in (HELDOUT, SEEN)
        }
    held = {
        c.name: dict(zip(c.seeds, c.per_seed))
        for c in sets[HELDOUT]["result"].conditions
    }
    seen = {
        c.name: dict(zip(c.seeds, c.per_seed)) for c in sets[SEEN]["result"].conditions
    }
    kept, flagged = select_runs(fresh)
    runs = sorted(int(t["run"]) for t in fresh)
    val = {int(t["run"]): float(t["best_val_mission"]) for t in fresh}
    ref_held, ref_seen = reference["per"][HELDOUT], reference["per"][SEEN]
    d_mean = float(np.mean(ref_held[D]))
    g7c = g7c_verdict([held[B][r] for r in kept], ref_held[D])
    a_vs = [held[B][r] for r in kept]
    g7c["diff_vs_a"] = float(np.mean(a_vs) - np.mean(ref_held[A]))
    g7c["ci_vs_a"] = welch_ci(np.asarray(a_vs), np.asarray(ref_held[A], float))

    old_flags = [
        t for t in reference["training"]
        if t["mode"] == CENTRE_GOAL and t["best_val_mission"] < FLAG_BELOW
    ]  # fmt: skip
    old_n = sum(t["mode"] == CENTRE_GOAL for t in reference["training"])
    n_flag, n_all = len(flagged) + len(old_flags), len(runs) + old_n
    rows = {
        f"(b) + rule, kept runs (n = {len(kept)})": [held[B][r] for r in kept],
        f"(b), first {N_KEEP} fresh runs, no rule": [held[B][r] for r in runs[:N_KEEP]],
        f"(b), all {len(runs)} fresh runs": [held[B][r] for r in runs],
        "(b), G7 runs 0-15": ref_held[B],
        "(d), G7 runs 0-15": ref_held[D],
        "(a), G7 runs 0-15": ref_held[A],
    }
    return {
        "g7c": g7c,
        "rule": {
            "flag_below": FLAG_BELOW,
            "n_keep": N_KEEP,
            "fresh_runs": list(FRESH_RUNS),
        },
        "kept": kept,
        "flagged": [
            {"run": r, "best_val": val[r], "heldout": held[B][r]} for r in flagged
        ],
        "flag_rate": {
            "flagged": n_flag,
            "runs": n_all,
            "wilson": wilson(n_flag, n_all),
        },
        "heldout_rows": {k: [float(x) for x in v] for k, v in rows.items()},
        "fraction_of_d": {k: float(np.mean(v) / d_mean) for k, v in rows.items()},
        "seen_kept": [float(seen[B][r]) for r in kept],
        "seen_d": [float(x) for x in ref_seen[D]],
        "per_run": [
            {"run": r, "best_val": val[r], "heldout": float(held[B][r]),
             "seen": float(seen[B][r]), "kept": r in kept}
            for r in runs
        ],  # fmt: skip
        "fresh_baselines_heldout": {
            name: float(np.mean(list(v.values())))
            for name, v in held.items()
            if name != B
        },
        "settings": settings_d,
    }


def render(rep: dict[str, Any]) -> str:
    g = rep["g7c"]
    lines = [
        "# R7c: is a goal carried only by learned messages reliable to train? (goal G7c)",
        "",
        f"**G7c verdict: {g['verdict']}.** With the restart rule (discard a run whose best "
        f"training-goal validation mission is below {FLAG_BELOW:g}), the kept (b) runs reach "
        f"{100 * g['fraction_of_d']:.0f}% of the hand-coded centre (d) on held-out goals "
        f"(pass ≥ {100 * PASS_FRACTION:.0f}%, kill < {100 * KILL_FRACTION:.0f}%).",
        "",
        f"- Kept − (d): {g['diff_vs_d']:+.2f} ± {g['ci_vs_d']:.2f} (Welch 95% CI), "
        f"Mann-Whitney p = {g['p_vs_d']:.3g}; n = {g['n_kept']} kept against 16.",
        f"- Kept − (a): {g['diff_vs_a']:+.2f} ± {g['ci_vs_a']:.2f}.",
        f"- Flagged {len(rep['flagged'])} of {len(rep['per_run'])} fresh runs; pooled with G7's "
        f"runs 0-15, {rep['flag_rate']['flagged']} of {rep['flag_rate']['runs']} "
        f"(Wilson 95% CI {100 * rep['flag_rate']['wilson'][1]:.0f}-"
        f"{100 * rep['flag_rate']['wilson'][2]:.0f}%).",
        "",
        "## Held-out goals",
        "",
        "| Runs | Mean ± 95% CI | % of (d) | Min |",
        "|---|---|---|---|",
    ]
    for name, v in rep["heldout_rows"].items():
        lines.append(
            f"| {name} | {_pm(v)} | {100 * rep['fraction_of_d'][name]:.0f}% | {min(v):.1f} |"
        )
    lines += [
        "",
        "(d) and (a) are G7's runs 0-15 (`docs/results/r7b_goal_conditioned_16runs.json`); "
        "fresh runs are evaluated on their own episode seeds, so comparisons are Welch over "
        "independent runs.",
        "",
        f"Seen goals (context): kept (b) {_pm(rep['seen_kept'])}, (d) {_pm(rep['seen_d'])}.",
        "",
        "## Fresh runs",
        "",
        "| Run | Best training-goal validation | Held-out | Seen | Kept |",
        "|---|---|---|---|---|",
    ]
    for r in rep["per_run"]:
        flag = (
            "yes"
            if r["kept"]
            else ("flagged" if r["best_val"] < FLAG_BELOW else "spare")
        )
        lines.append(
            f"| {r['run']} | {r['best_val']:.1f} | {r['heldout']:.1f} | {r['seen']:.1f} | {flag} |"
        )
    lines += [
        "",
        "Baselines on the fresh runs' episode seeds (held-out mean): "
        + ", ".join(f"{k} {v:.2f}" for k, v in rep["fresh_baselines_heldout"].items())
        + ".",
        "",
        f"Training settings: `{json.dumps(rep['settings'])}`.",
        "",
        "Reproduce: `python scripts/r7c_reliability.py --from-json runs:r7c-train-*` "
        "(after the `r7c-train-*` jobs).",
        "",
    ]
    return "\n".join(lines)
