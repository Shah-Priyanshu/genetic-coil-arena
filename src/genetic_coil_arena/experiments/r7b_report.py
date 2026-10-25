"""R7b driver: train the goal-conditioned squad conditions, evaluate, apply goal G7.

    # one training block (a queued job): writes r7b_<mode>_run<k>.{npz,json} into $GCA_RUN_DIR
    python scripts/r7b_goal_conditioned.py --train-only --modes blind --runs 0:8 --workers 8

    # evaluate every trained run and write docs/results/r7b_goal_conditioned.{md,json}
    python scripts/r7b_goal_conditioned.py --from-json runs:r7b-train-* \\
        --out docs/results/r7b_goal_conditioned.md

    python scripts/r7b_goal_conditioned.py --scale smoke       # tiny end to end check

``GCA_WORKERS`` overrides ``--workers``. See ``r7b_goal_conditioned`` for the design.
"""

import argparse
import csv
import glob
import json
import math
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..evaluation import (
    ABLATIONS,
    BaselineRegistry,
    evaluate,
    format_table,
    holm_correction,
    mann_whitney,
    mean_ci,
    paired_difference,
    run_experiment,
)
from . import run_cache
from .learned_protocol import welch_ci
from .r7b_goal_conditioned import (
    BLIND,
    BOTH_GOAL,
    CENTRE_GOAL,
    CONDITION,
    HAND_GOAL,
    HELDOUT,
    MODES,
    NOVEL_COMBO,
    NOVEL_SUBSET,
    SCRIPTED,
    SEEN,
    GoalSystem,
    GoalTrainSettings,
    goal_message_metrics,
    goal_sets,
    make_goal_task,
    policies_for,
    settings_dict,
    split_goals,
    train_run,
)

PREFIX = "r7b"
G7_FRACTION = 0.9
G7_ALPHA = 0.05


# ------------------------------------------------------------------ G7 verdict
def g7_verdict(
    heldout: dict[str, Sequence[float]],
    centre_only: str = CONDITION[CENTRE_GOAL],
    both: str = CONDITION[BOTH_GOAL],
    blind: str = CONDITION[BLIND],
    hand: str = CONDITION[HAND_GOAL],
) -> dict[str, Any]:
    """Goal G7 from per-training-run mean mission scores on the held-out goals.

    Pass: (b) or (c) beats (a) (Welch 95% CI of the difference excludes 0 on the
    positive side and the Holm-corrected Mann-Whitney p < 0.05 over the two
    comparisons) AND its mean reaches at least 90% of (d)'s (needs (d) > 0).
    Kill: neither (b) nor (c) beats (a), i.e. both are within CI of (a) or worse.
    Otherwise ``partial``.
    """
    a = np.asarray(heldout[blind], dtype=float)
    d_mean = float(np.mean(heldout[hand]))
    rows = []
    for name in (centre_only, both):
        x = np.asarray(heldout[name], dtype=float)
        diff = float(x.mean() - a.mean())
        ci = welch_ci(x, a)
        _, p = mann_whitney(x, a)
        rows.append(
            {
                "condition": name,
                "mean": float(x.mean()),
                "diff_vs_blind": diff,
                "ci": ci,
                "p": float(p),
                "fraction_of_hand": (
                    float(x.mean() / d_mean) if d_mean > 0 else math.nan
                ),
            }
        )
    holm = holm_correction([r["p"] for r in rows])
    for r, p_adj in zip(rows, holm):
        r["p_holm"] = float(p_adj)
        r["beats_blind"] = bool(
            np.isfinite(r["ci"])
            and r["diff_vs_blind"] - r["ci"] > 0
            and r["p_holm"] < G7_ALPHA
        )
        r["reaches_90pct_of_hand"] = bool(
            d_mean > 0 and r["mean"] >= G7_FRACTION * d_mean
        )
        r["passes"] = r["beats_blind"] and r["reaches_90pct_of_hand"]
    if any(r["passes"] for r in rows):
        verdict = "pass"
    elif not any(r["beats_blind"] for r in rows):
        verdict = "kill"
    else:
        verdict = "partial"
    if verdict == "pass":
        why = "a goal-conditioned centre beats (a) and reaches 90% of (d) on held-out goals"
    elif verdict == "kill":
        why = "neither (b) nor (c) is better than (a) on held-out goals (CI includes 0 or below)"
    else:
        why = "(b) or (c) beats (a) but none reaches 90% of (d)"
    return {
        "verdict": verdict,
        "why": why,
        "hand_mean": d_mean,
        "blind_mean": float(a.mean()),
        "rows": rows,
    }


# ------------------------------------------------------------------ training
def _train_job(job: tuple[str, int, GoalTrainSettings, Optional[Path]]) -> dict:
    mode, run, settings, cache = job
    key = f"{PREFIX}_{mode}_run{run}"
    if cache is not None:
        hit = run_cache.load(cache, key, settings_dict(settings))
        if hit is not None:
            print(f"[cache] {key}", flush=True)
            return hit
    out = train_run(mode, run, settings)
    if cache is not None:
        arrays = {"theta": out["theta"], "theta_final": out["theta_final"]}
        summary = {k: v for k, v in out.items() if k not in arrays}
        run_cache.save(cache, key, settings_dict(settings), arrays, summary)
    print(
        f"[done] {key}: {out['seconds'] / 60:.1f} min, best gen {out['best_generation']}"
        f", best val mission {out['best_val_mission']:.2f}",
        flush=True,
    )
    return out


def read_trained(sources: Sequence[Path]) -> tuple[list[dict], dict[str, Any]]:
    """Trained runs (``r7b_<mode>_run<k>.json`` with its ``.npz``) under files/folders.

    Returns the runs and the training settings they share (the time limit and progress
    interval are ignored; runs trained with other settings raise ValueError).
    """
    files: list[Path] = []
    for src in sources:
        files += sorted(src.glob(f"{PREFIX}_*_run*.json")) if src.is_dir() else [src]
    runs, settings = [], None
    for path in files:
        if path.suffix != ".json" or ".tmp" in path.name:
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        if "settings" not in meta or not path.with_suffix(".npz").exists():
            continue
        got = {
            k: v
            for k, v in meta["settings"].items()
            if k not in ("time_limit_s", "progress_s")
        }
        if settings is None:
            settings = got
        elif got != settings:
            raise ValueError(f"{path} was trained with different settings")
        with np.load(path.with_suffix(".npz")) as arrays:
            theta = arrays["theta"]
        runs.append({**meta["summary"], "theta": theta})
    if not runs or settings is None:
        raise SystemExit(f"no trained runs found in {[str(s) for s in sources]}")
    seen = {(r["mode"], r["run"]) for r in runs}
    if len(seen) != len(runs):
        raise ValueError("duplicate (mode, run) among the trained runs")
    return runs, settings


# ------------------------------------------------------------------ evaluation
@dataclass(frozen=True)
class GoalEvalSettings:
    """Size of the held-out evaluation (test episodes per goal)."""

    seen_per_goal: int = 8
    heldout_per_goal: int = 20
    record_every: int = 3
    n_shuffles: int = 100
    n_boot: int = 2000


def _group_means(outputs: dict[str, Any], groups: dict[str, list[int]]) -> dict:
    score, idx = outputs["mission"], outputs["goal_index"]
    return {g: float(score[np.isin(idx, m)].mean()) for g, m in groups.items()}


def evaluate_goal_set(
    which: str,
    trained: Sequence[dict],
    settings: GoalTrainSettings,
    esettings: GoalEvalSettings,
) -> dict[str, Any]:
    """All conditions and baselines on the ``seen`` or ``heldout`` goals."""
    goals = goal_sets()[which]
    per_goal = esettings.seen_per_goal if which == SEEN else esettings.heldout_per_goal
    config = settings.squad_config()
    by_mode: dict[str, dict[int, tuple]] = {}
    for t in trained:
        by_mode.setdefault(t["mode"], {})[int(t["run"])] = policies_for(
            t["mode"], settings, t["theta"]
        )
    runs = sorted(next(iter(by_mode.values())))
    if any(sorted(v) != runs for v in by_mode.values()):
        raise ValueError("Every condition needs the same training runs")
    registry = BaselineRegistry()
    name = f"squad-goal-{which}"
    task = make_goal_task(name, goals, runs, per_goal, config, registry)

    def system(mode: str, swap: bool = False) -> GoalSystem:
        return GoalSystem(
            config, goals, by_mode[mode].__getitem__, swap, esettings.record_every
        )

    systems = {CONDITION[m]: system(m) for m in MODES if m in by_mode}
    result = run_experiment(
        task,
        systems,
        seeds=runs,
        ablations=ABLATIONS,
        ablate=list(systems),
        registry=registry,
        n_boot=esettings.n_boot,
        metadata={"goals": [g.label() for g in goals], "per_goal": per_goal},
        keep_outputs=True,
    )
    result.require_complete()

    swaps = {}
    for m in MODES:
        if m not in by_mode:
            continue
        swapped = evaluate(
            task, system(m, swap=True), runs, f"{CONDITION[m]} [goals swapped]",
            n_boot=esettings.n_boot,
        )  # fmt: skip
        diff = paired_difference(
            result.condition(CONDITION[m]).per_seed, swapped.per_seed
        )
        swaps[CONDITION[m]] = {"swapped_mean": swapped.mean, **diff}

    groups: dict[str, list[int]] = {}
    if which == HELDOUT:
        split = split_goals()
        groups = {
            NOVEL_SUBSET: [goals.index(g) for g in split[NOVEL_SUBSET]],
            NOVEL_COMBO: [goals.index(g) for g in split[NOVEL_COMBO]],
        }
    subgroup: dict[str, dict[str, list[float]]] = {}
    per_goal_mean: dict[str, list[float]] = {}
    messages: dict[str, dict[str, Any]] = {}
    for cond in result.conditions:
        outs = cond.outputs or []
        if groups:
            rows = [_group_means(o, groups) for o in outs]
            subgroup[cond.name] = {g: [r[g] for r in rows] for g in groups}
        table = np.stack(
            [
                [o["mission"][o["goal_index"] == i].mean() for i in range(len(goals))]
                for o in outs
            ]
        )
        per_goal_mean[cond.name] = table.mean(axis=0).tolist()
        if cond.name in {CONDITION[m] for m in MODES}:
            per_run: dict[str, list[float]] = {}
            for seed, o in zip(cond.seeds, outs):
                row = goal_message_metrics(o, esettings.n_shuffles, int(seed))
                for k, v in row.items():
                    per_run.setdefault(k, []).append(float(v))
            messages[cond.name] = per_run
    for cond in result.conditions:
        cond.outputs = None
    return {
        "which": which,
        "goals": [g.label() for g in goals],
        "result": result,
        "swaps": swaps,
        "subgroup": subgroup,
        "per_goal_mean": per_goal_mean,
        "messages": messages,
    }


def comparison_rows(per_seed: dict[str, Sequence[float]]) -> list[dict[str, Any]]:
    """Context comparisons between conditions (Welch CI + Holm-corrected Mann-Whitney)."""
    a, b, c, d = (CONDITION[m] for m in (BLIND, CENTRE_GOAL, BOTH_GOAL, HAND_GOAL))
    pairs = [
        (b, a),
        (c, a),
        (d, a),
        (b, d),
        (c, d),
        (c, b),
        (SCRIPTED, a),
        (d, SCRIPTED),
    ]
    pairs = [(x, y) for x, y in pairs if x in per_seed and y in per_seed]
    rows = []
    for x, y in pairs:
        xs, ys = np.asarray(per_seed[x], float), np.asarray(per_seed[y], float)
        rows.append(
            {
                "a": x,
                "b": y,
                "diff": float(xs.mean() - ys.mean()),
                "ci": welch_ci(xs, ys),
                "p": float(mann_whitney(xs, ys)[1]),
            }
        )
    for r, p_adj in zip(rows, holm_correction([r["p"] for r in rows])):
        r["p_holm"] = float(p_adj)
    return rows


# ------------------------------------------------------------------ report
def _pm(values: Sequence[float], digits: int = 2) -> str:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return "n/a"
    m, ci = mean_ci(finite)
    return f"{m:.{digits}f} ± {ci:.{digits}f}" if np.isfinite(ci) else f"{m:.{digits}f}"


def _vs(per: dict[str, list[float]], key: str, ref: str) -> str:
    """``value (reference)`` cell, both as mean ± CI over runs."""
    return f"{_pm(per.get(key, []), 3)} ({_pm(per.get(ref, []), 3)})"


def render(
    sets: dict[str, dict[str, Any]],
    g7: dict[str, Any],
    trained: Sequence[dict],
    settings: dict[str, Any],
    seconds: dict[str, float],
    stem: str,
    scale: str,
) -> str:
    n_runs = len({t["run"] for t in trained})
    lines = [
        "# R7b: a learned command centre that follows a GoalSpec (goal G7)",
        "",
        f"**G7 verdict: {g7['verdict'].upper()}.** {g7['why']}.",
        "",
        f"Scale: {scale}. {n_runs} training runs per condition; every number is a mean over "
        "training runs with a 95% CI (Student t unless stated). Mission score = eater food "
        "- 1.0 x non-eater food - death penalty (0 / 2 / 5 for low / normal / high caution) "
        "x dead eaters (see `experiments/r7b_goal_conditioned.py`). 15 training goals, "
        "6 held-out goals: eater subset {green, blue} at every caution (`novel_subset`) and "
        "three seen-part combinations (`novel_combo`: red/high, red+blue/normal, "
        "red+green+blue/low).",
        "",
        f"Training settings: `{json.dumps(settings)}`.",
        f"Compute: training {seconds.get('train_cpu_h', float('nan')):.1f} CPU-hours "
        f"(sum over runs); evaluation {seconds.get('eval_min', float('nan')):.1f} min.",
        "",
        "## Mission score",
        "",
        "| Condition | Seen goals | Held-out goals | Novel subset | Novel combo |",
        "|---|---|---|---|---|",
    ]
    seen_r, held_r = sets[SEEN]["result"], sets[HELDOUT]["result"]
    names = [c.name for c in held_r.conditions]
    for n in names:
        sub = sets[HELDOUT]["subgroup"].get(n, {})
        lines.append(
            f"| {n} | {_pm(seen_r.condition(n).per_seed)} | "
            f"{_pm(held_r.condition(n).per_seed)} | "
            f"{_pm(sub.get(NOVEL_SUBSET, []))} | {_pm(sub.get(NOVEL_COMBO, []))} |"
        )
    lines += [
        "",
        "## G7",
        "",
        "Pass: (b) or (c) beats (a) on held-out goals (Welch CI of the difference above 0, "
        "Holm-corrected Mann-Whitney p < 0.05) and reaches at least 90% of (d). Kill: neither "
        "(b) nor (c) beats (a).",
        "",
        f"(a) = {g7['blind_mean']:.2f}, (d) = {g7['hand_mean']:.2f} on held-out goals.",
        "",
        "| Condition | Held-out mean | Diff vs (a) | Welch CI | p (Holm) | Beats (a) | % of (d) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in g7["rows"]:
        frac = r["fraction_of_hand"]
        lines.append(
            f"| {r['condition']} | {r['mean']:.2f} | {r['diff_vs_blind']:+.2f} | "
            f"± {r['ci']:.2f} | {r['p_holm']:.3g} | {'yes' if r['beats_blind'] else 'no'} | "
            f"{'n/a' if not np.isfinite(frac) else f'{100 * frac:.0f}%'} |"
        )
    for which in (HELDOUT, SEEN):
        ps = {c.name: c.per_seed for c in sets[which]["result"].conditions}
        lines += [
            "",
            f"### Pairwise differences, {which} goals",
            "",
            "| A | B | A - B | Welch CI | p (Holm) |",
            "|---|---|---|---|---|",
        ]
        for r in comparison_rows(ps):
            lines.append(
                f"| {r['a']} | {r['b']} | {r['diff']:+.2f} | ± {r['ci']:.2f} | "
                f"{r['p_holm']:.3g} |"
            )
    lines += ["", "## Do the messages carry the goal?", ""]
    for which in (HELDOUT, SEEN):
        msg = sets[which]["messages"]
        lines += [
            f"### {which} goals (live receivers; MI in bits, Miller-Madow; null = 95th "
            "percentile of shuffled labels; probe = held-out-episode accuracy vs majority "
            "class)",
            "",
            "| Condition | H(msg) bits | MI own-eater (null) | probe own-eater (chance) | "
            "MI caution (null) | probe caution (chance) | MI goal (null) |",
            "|---|---|---|---|---|---|---|",
        ]
        for n, per in msg.items():
            if not per:
                continue
            cells = [
                _pm(per.get("message_entropy_bits", [])),
                _vs(per, "mi_own_eater_bits", "mi_own_eater_null_q95_bits"),
                _vs(per, "probe_own_eater_acc", "probe_own_eater_chance"),
                _vs(per, "mi_caution_bits", "mi_caution_null_q95_bits"),
                _vs(per, "probe_caution_acc", "probe_caution_chance"),
                _vs(per, "mi_goal_bits", "mi_goal_null_q95_bits"),
            ]
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## Ablations (score drop, intact minus intervened, paired over training runs)",
        "",
        "Message interventions are the harness's; `goals swapped` tells the policies the "
        "goal of another episode (a different eater set) while the score uses the true goal.",
        "",
    ]
    for which in (HELDOUT, SEEN):
        res = sets[which]["result"]
        interventions = sorted({a.intervention for a in res.ablations})
        lines += [
            f"### {which} goals",
            "",
            "| Condition | " + " | ".join(interventions) + " | goals swapped |",
            "|---|" + "---|" * (len(interventions) + 1),
        ]
        for n in [CONDITION[m] for m in MODES]:
            cells = []
            for iv in interventions:
                hit = [
                    a for a in res.ablations if a.system == n and a.intervention == iv
                ]
                cells.append(
                    f"{hit[0].delta_mean:+.2f} ± {hit[0].delta_ci:.2f}"
                    if hit
                    else "n/a"
                )
            sw = sets[which]["swaps"].get(n)
            cells.append(f"{sw['mean']:+.2f} ± {sw['ci']:.2f}" if sw else "n/a")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Per-goal mean mission score (mean over training runs)", ""]
    for which in (SEEN, HELDOUT):
        goals = sets[which]["goals"]
        conds = list(sets[which]["per_goal_mean"])
        lines += [
            f"### {which}",
            "",
            "| Goal | " + " | ".join(conds) + " |",
            "|---|" + "---|" * len(conds),
        ]
        for i, g in enumerate(goals):
            vals = " | ".join(
                f"{sets[which]['per_goal_mean'][c][i]:.1f}" for c in conds
            )
            lines.append(f"| {g} | {vals} |")
        lines.append("")
    for which in (HELDOUT, SEEN):
        lines += [
            f"## Harness table, {which} goals",
            "",
            "```",
            format_table(sets[which]["result"]),
            "```",
            "",
        ]
    lines += [
        "## Notes",
        "",
        "- Checkpoints are selected on validation episodes of the training goals only.",
        "- Snakes not ordered to eat starve after about 60 steps; their deaths cost nothing, "
        "so the score rewards leaving their food alone, not surviving.",
        "- `step_limit` and `food_target` are valid GoalSpec fields but fixed at 300 and 0.",
        "- Reproduce: `python scripts/r7b_goal_conditioned.py --from-json runs:r7b-train-* "
        f"--out docs/results/{stem}.md` (after the `r7b-train-*` jobs).",
        "",
    ]
    return "\n".join(lines)


def write_curves(trained: Sequence[dict], path: Path) -> None:
    keys = ["generation", "episodes", "wall_s", "pop_mean", "pop_max", "val_mission",
            "val_team_food", "val_message_entropy"]  # fmt: skip
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "run", *keys])
        for t in sorted(trained, key=lambda t: (MODES.index(t["mode"]), t["run"])):
            for row in t["curve"]:
                w.writerow([t["mode"], t["run"], *[row.get(k) for k in keys]])


def evaluate_and_report(
    trained: Sequence[dict],
    settings_d: dict[str, Any],
    esettings: GoalEvalSettings,
    out: Optional[Path],
    scale: str,
    workers: int = 1,
) -> dict[str, Any]:
    settings = GoalTrainSettings(**settings_d)
    start = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(evaluate_goal_set, w, trained, settings, esettings)
                for w in (SEEN, HELDOUT)
            ]
            sets = {r["which"]: r for r in (f.result() for f in futures)}
    else:
        sets = {
            w: evaluate_goal_set(w, trained, settings, esettings)
            for w in (SEEN, HELDOUT)
        }
    eval_min = (time.perf_counter() - start) / 60
    held = {c.name: c.per_seed for c in sets[HELDOUT]["result"].conditions}
    g7 = g7_verdict(held)
    seconds = {
        "train_cpu_h": sum(t.get("seconds", 0.0) for t in trained) / 3600,
        "eval_min": eval_min,
    }
    stem = out.stem if out else "r7b_goal_conditioned"
    text = render(sets, g7, trained, settings_d, seconds, stem, scale)
    print(text)
    payload = {
        "g7": g7,
        "scale": scale,
        "settings": settings_d,
        "eval_settings": asdict(esettings),
        "seconds": seconds,
        "sets": {
            w: {
                "goals": s["goals"],
                "result": s["result"].to_dict(),
                "swaps": s["swaps"],
                "subgroup": s["subgroup"],
                "per_goal_mean": s["per_goal_mean"],
                "messages": s["messages"],
                "comparisons": comparison_rows(
                    {c.name: c.per_seed for c in s["result"].conditions}
                ),
            }
            for w, s in sets.items()
        },
        "training": [
            {k: v for k, v in t.items() if k not in ("theta", "theta_final", "curve")}
            for t in trained
        ],
    }
    if out is not None:
        folders = [out.parent]
        if os.environ.get("GCA_RUN_DIR"):
            folders.append(Path(os.environ["GCA_RUN_DIR"]))
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{stem}.md").write_text(text, encoding="utf-8")
            (folder / f"{stem}.json").write_text(
                json.dumps(run_cache.finite(payload), allow_nan=False, default=float),
                encoding="utf-8",
            )
            write_curves(trained, folder / f"{stem}_curves.csv")
        print(f"wrote {out} and {out.with_suffix('.json')}", flush=True)
    return payload


# ------------------------------------------------------------------ CLI
SCALES: dict[str, dict[str, Any]] = {
    "full": {
        "runs": "0:8",
        "generations": 2000,
        "population": 128,
        "eval_every": 20,
        "eval": GoalEvalSettings(),
    },
    "smoke": {
        "runs": "0:2",
        "generations": 6,
        "population": 8,
        "eval_every": 3,
        "eval": GoalEvalSettings(1, 2, 3, 10, 100),
    },
}


def env_workers(default: int) -> int:
    raw = os.environ.get("GCA_WORKERS", "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else int(default)


def resolve_sources(tokens: Sequence[str]) -> list[Path]:
    """Files, folders, globs and ``runs:<pattern>`` (sibling jobs' artifacts folders)."""
    out: list[Path] = []
    for tok in tokens:
        if tok.startswith("runs:"):
            run_dir = os.environ.get("GCA_RUN_DIR")
            if not run_dir:
                raise SystemExit("'runs:' needs GCA_RUN_DIR (it is set by the runner)")
            runs = Path(run_dir).resolve().parents[0]
            for job in sorted(runs.parent.glob(tok[len("runs:") :])):
                out.append(job / "artifacts")
        elif any(c in tok for c in "*?["):
            out += [Path(p) for p in sorted(glob.glob(tok))]
        else:
            out.append(Path(tok))
    return [p for p in out if p.exists()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scale", choices=sorted(SCALES), default="full")
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--runs", help="Training runs a:b (default per scale)")
    p.add_argument("--generations", type=int)
    p.add_argument("--population", type=int)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--spec-hidden", type=int, default=24)
    p.add_argument("--centre-hidden", type=int, default=24)
    p.add_argument("--time-limit", type=float, default=float("inf"))
    p.add_argument("--progress", type=float, default=120.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--from-json", nargs="+", help="Evaluate trained runs; no training")
    p.add_argument("--cache", type=Path, help="Where trained runs are saved")
    p.add_argument(
        "--out", type=Path, default=Path("docs/results/r7b_goal_conditioned.md")
    )
    args = p.parse_args(argv)
    for key, value in SCALES[args.scale].items():
        if key != "eval" and getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    esettings = SCALES[args.scale]["eval"]
    workers = env_workers(args.workers)
    if args.from_json:
        trained, settings_d = read_trained(resolve_sources(args.from_json))
        print(f"[eval] {len(trained)} trained runs", flush=True)
        evaluate_and_report(
            trained, settings_d, esettings, args.out, args.scale, workers
        )
        return
    lo, hi = (int(x) for x in args.runs.split(":"))
    settings = GoalTrainSettings(
        generations=args.generations,
        population=args.population,
        reps=args.reps,
        spec_hidden=args.spec_hidden,
        centre_hidden=args.centre_hidden,
        eval_every=args.eval_every,
        time_limit_s=args.time_limit,
        progress_s=args.progress,
    )
    cache = args.cache
    if cache is None and os.environ.get("GCA_RUN_DIR"):
        cache = Path(os.environ["GCA_RUN_DIR"])
    if cache is None and not args.train_only:
        cache = Path("outputs") / "r7b_runs"
    jobs = [(m, k, settings, cache) for m in args.modes for k in range(lo, hi)]
    print(f"[start] {len(jobs)} training runs on {workers} workers; cache {cache}; "
          f"{settings}", flush=True)  # fmt: skip
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        trained = list(pool.map(_train_job, jobs))
    print(f"[train] done in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    if args.train_only:
        return
    sd = replace(settings, time_limit_s=float("inf"), progress_s=0.0)
    evaluate_and_report(trained, settings_dict(sd), esettings, args.out, args.scale, 1)


if __name__ == "__main__":
    main()
