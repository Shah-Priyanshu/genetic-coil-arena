"""Does a code form? Code-formation aids for the jointly learned centre (R6 follow-up).

With joint ES training of centre and specialists, some runs never form a code: the
centre settles on one symbol, the listeners learn to ignore it, and the squad ends at
the silent control's level (the chicken-and-egg problem of emergent communication).
This script trains the learned-centre condition (c) of ``learned_protocol`` under
several arms that differ only in a code-formation aid, from the same runs (same
initial specialist and centre weights, ES noise and training episodes per run index),
and scores each arm on held-out seeds through the CommArena harness.

Arms (``--arms``):

- ``baseline``: the R6 method (argmax centre, fitness = team return)
- ``entropy``: fitness += w H(M), w annealed from ``--entropy-bonus`` to 0 over the first
  ``--entropy-anneal`` of training (H(M) = entropy of the member's own symbols; for an
  argmax centre this equals the MI between the symbol and the centre's input)
- ``temperature``: the centre samples from softmax(logits / T) in training, T annealed
  from ``--centre-temperature`` to 0 (argmax) over ``--temperature-anneal``

A run *forms a code* when, on its held-out episodes, the centre's symbols carry
H(M) > 0.5 bits and shuffling them across episodes (``shuffle_batch``, which keeps the
symbol marginals) costs team food with a 95% paired t-CI over episodes above 0.

    python scripts/code_formation.py --runs 10 --generations 600 --workers 4 \
        --out docs/results/code_formation.md

Outputs next to ``--out``: ``<stem>.md``, ``<stem>.json`` and ``<stem>_params.npz``
(float32). Every finished run is saved at once to ``--cache`` (``<arm>_run<k>.npy`` +
``.json``); a rerun with the same settings loads those and trains only the rest, so an
interrupted run resumes and a finished one re-renders without training.
"""

import argparse
import datetime
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from genetic_coil_arena.envs.squad import SquadEnv, run_episodes
from genetic_coil_arena.evaluation import (
    InterventionHook,
    format_table,
    holm_correction,
    mean_ci,
    paired_difference,
    wilcoxon_signed_rank,
)
from genetic_coil_arena.experiments.learned_protocol import (
    CONDITION,
    LEARNED,
    SCRIPTED,
    EvalSettings,
    TrainSettings,
    evaluate_radius,
    policies_for,
    test_episode_seeds,
    train_run,
)
from genetic_coil_arena.experiments.learned_protocol_report import (
    _commit,
    _cpu_name,
    _finite,
)

C = CONDITION[LEARNED]
ARMS = ("baseline", "entropy", "temperature")
H_MIN = 0.5  # bits: a centre below this sends (almost) one symbol


def arm_settings(
    arm: str, base: TrainSettings, args: argparse.Namespace
) -> TrainSettings:
    if arm == "baseline":
        return base
    if arm == "entropy":
        return replace(
            base, entropy_bonus=args.entropy_bonus, entropy_anneal=args.entropy_anneal
        )
    if arm == "temperature":
        return replace(
            base,
            centre_temperature=args.centre_temperature,
            temperature_anneal=args.temperature_anneal,
        )
    raise ValueError(f"Unknown arm {arm!r}")


def _cached(cache: Path, arm: str, run: int, settings: TrainSettings) -> Any:
    """A finished run saved by an earlier invocation with the same settings, or None."""
    meta, theta = cache / f"{arm}_run{run}.json", cache / f"{arm}_run{run}.npy"
    if not (meta.exists() and theta.exists()):
        return None
    saved = json.loads(meta.read_text(encoding="utf-8"))
    if saved["settings"] != _finite(asdict(replace(settings, progress_s=0.0))):
        return None
    return {**saved["run"], "theta": np.load(theta), "cached": True}


def _train_job(job: tuple[str, int, TrainSettings, Path]) -> dict[str, Any]:
    """Train one (arm, run) unless it is cached; each finished run is saved at once
    (theta as .npy, then the summary .json, both via rename) so a restart only loses
    the runs in progress."""
    arm, run, settings, cache = job
    hit = _cached(cache, arm, run, settings)
    if hit is not None:
        print(f"[cache] {arm} run {run}", flush=True)
        return hit
    out = train_run(LEARNED, run, settings)
    out = {k: v for k, v in out.items() if k != "theta_final"}
    tmp = cache / f"{arm}_run{run}.tmp.npy"
    np.save(tmp, out["theta"])
    os.replace(tmp, cache / f"{arm}_run{run}.npy")
    summary = {k: v for k, v in out.items() if k != "theta"}
    meta = {
        "settings": _finite(asdict(replace(settings, progress_s=0.0))),
        "run": _finite({**summary, "arm": arm}),
    }
    tmp = cache / f"{arm}_run{run}.tmp.json"
    tmp.write_text(json.dumps(meta, allow_nan=False), encoding="utf-8")
    os.replace(tmp, cache / f"{arm}_run{run}.json")
    last = out["curve"][-1]
    print(
        f"[train] {arm:11s} run {run}: {out['generations_done']} gens in "
        f"{out['seconds']:.0f} s, best val food "
        f"{max(r['val_team_food'] for r in out['curve']):.2f} (gen "
        f"{out['best_generation']}), final val H(M) {last['val_message_entropy']:.2f}",
        flush=True,
    )
    return {**out, "arm": arm}


def formation_check(
    settings: TrainSettings, run: int, theta: np.ndarray, n_episodes: int
) -> dict[str, float]:
    """Per-episode intact vs shuffled / zeroed team food on run's held-out seeds."""
    specialist, centre = policies_for(LEARNED, settings, theta)
    env = SquadEnv(settings.squad_config())
    seeds = test_episode_seeds(run, n_episodes)
    intact = run_episodes(env, specialist, centre, seeds)["team_food"]
    out: dict[str, float] = {"food": float(intact.mean())}
    for name in ("shuffle_batch", "zero"):
        hook = InterventionHook(name, seed=10_000 + run, spec=env.spec.message)
        ablated = run_episodes(env, specialist, centre, seeds, hook)["team_food"]
        d = paired_difference(intact, ablated)
        out[f"{name}_drop"], out[f"{name}_low"] = d["mean"], d["low"]
    return out


def _eval_job(job: tuple[str, list, TrainSettings, EvalSettings]) -> dict[str, Any]:
    arm, trained, settings, esettings = job
    result = evaluate_radius(settings.radius, trained, settings, esettings)
    per = result.metadata["per_run_communication"][C]
    rows = []
    for i, t in enumerate(sorted(trained, key=lambda t: t["run"])):
        check = formation_check(settings, t["run"], t["theta"], esettings.n_episodes)
        h = per["message_entropy_bits"][i]
        rows.append(
            {
                "run": t["run"],
                "food": result.condition(C).per_seed[i],
                "H": h,
                "mi": per["mi_bits"][i],
                "mi_direction": per["mi_direction_bits"][i],
                "cic": per["cic_bits"][i],
                "codes": per["distinct_codes"][i],
                **{k: v for k, v in check.items() if k != "food"},
                "check_food": check["food"],
                "formed": bool(h > H_MIN and check["shuffle_batch_low"] > 0),
                "best_generation": t["best_generation"],
            }
        )
    return {"arm": arm, "result": result, "runs": rows}


# ------------------------------------------------------------------ report
def _pm(mean: float, half: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} ± {half:.{digits}f}"


def _p(value: float) -> str:
    return f"{value:.2g}" if value < 0.01 else f"{value:.3f}"


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    alpha = 1 - confidence
    low = 0.0 if k == 0 else scipy_stats.beta.ppf(alpha / 2, k, n - k + 1)
    high = 1.0 if k == n else scipy_stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(low), float(high)


def summary_rows(evals: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    base = evals.get("baseline")
    rows = []
    for arm, ev in evals.items():
        runs = ev["runs"]
        food = np.array([r["food"] for r in runs])
        formed = [r["formed"] for r in runs]
        k, n = int(sum(formed)), len(formed)
        row: dict[str, Any] = {
            "arm": arm,
            "formed": k,
            "n": n,
            "formed_ci": clopper_pearson(k, n),
            "food": mean_ci(food),
            "food_formed": mean_ci(food[np.array(formed)]) if k >= 2 else None,
            "H": mean_ci([r["H"] for r in runs]),
            "mi": mean_ci([r["mi"] for r in runs]),
            "shuffle_drop": mean_ci([r["shuffle_batch_drop"] for r in runs]),
        }
        if base is not None and arm != "baseline":
            bfood = np.array([r["food"] for r in base["runs"]])
            bformed = [r["formed"] for r in base["runs"]]
            d = paired_difference(food, bfood)
            row["vs_baseline"] = {
                "mean_diff": d["mean"],
                "low": d["low"],
                "high": d["high"],
                "p_wilcoxon": wilcoxon_signed_rank(food, bfood)[1],
                "p_fisher_formed": float(
                    scipy_stats.fisher_exact(
                        [[k, n - k], [sum(bformed), len(bformed) - sum(bformed)]]
                    )[1]
                ),
            }
        rows.append(row)
    extra = [r for r in rows if "vs_baseline" in r]
    for key in ("p_wilcoxon", "p_fisher_formed"):
        adj = holm_correction([r["vs_baseline"][key] for r in extra]) if extra else []
        for r, p in zip(extra, adj):
            r["vs_baseline"][f"{key}_holm"] = float(p)
    return rows


def summary_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| arm | code formed | 95% CI (Clopper-Pearson) | team food (c) | team food, "
        "runs with a code | H(M) bits | MI(M;T) bits | shuffle drop |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ff = r["food_formed"]
        lines.append(
            f"| {r['arm']} | {r['formed']}/{r['n']} | [{r['formed_ci'][0]:.2f}, "
            f"{r['formed_ci'][1]:.2f}] | {_pm(*r['food'])} | "
            f"{_pm(*ff) if ff else 'n/a'} | {_pm(*r['H'])} | {_pm(*r['mi'])} | "
            f"{_pm(*r['shuffle_drop'])} |"
        )
    return lines


def comparison_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| arm vs baseline | team food diff [95% paired t-CI] | p (Wilcoxon, Holm) "
        "| p (Fisher, code formed, Holm) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        v = r.get("vs_baseline")
        if v:
            lines.append(
                f"| {r['arm']} | {v['mean_diff']:+.2f} [{v['low']:+.2f}, "
                f"{v['high']:+.2f}] | {_p(v['p_wilcoxon_holm'])} | "
                f"{_p(v['p_fisher_formed_holm'])} |"
            )
    return lines


def per_run_table(evals: dict[str, dict[str, Any]]) -> list[str]:
    arms = list(evals)
    lines = [
        "| run | "
        + " | ".join(f"{a}: food / H(M) / shuffle drop [low] / formed" for a in arms)
        + " |",
        "|---|" + "---|" * len(arms),
    ]
    by_run = {a: {r["run"]: r for r in evals[a]["runs"]} for a in arms}
    for run in sorted(by_run[arms[0]]):
        cells = []
        for a in arms:
            r = by_run[a][run]
            cells.append(
                f"{r['food']:.1f} / {r['H']:.2f} / {r['shuffle_batch_drop']:.1f} "
                f"[{r['shuffle_batch_low']:.1f}] / {'yes' if r['formed'] else 'no'}"
            )
        lines.append(f"| {run} | " + " | ".join(cells) + " |")
    return lines


def curve_table(trained: list[dict[str, Any]], key: str) -> list[str]:
    arms = list(dict.fromkeys(t["arm"] for t in trained))
    gens = sorted({row["generation"] for t in trained for row in t["curve"]})
    picks = [g for g in gens if g % 100 == 0] or gens
    lines = [
        "| arm | " + " | ".join(f"gen {g}" for g in picks) + " |",
        "|---|" + "---|" * len(picks),
    ]
    for arm in arms:
        cells = []
        for g in picks:
            vals = [
                row[key]
                for t in trained
                if t["arm"] == arm
                for row in t["curve"]
                if row["generation"] == g
            ]
            cells.append(f"{np.mean(vals):.2f}" if vals else "-")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return lines


def render(
    evals: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    trained: list[dict[str, Any]],
    args: argparse.Namespace,
    base: TrainSettings,
    train_seconds: float,
    eval_seconds: float,
) -> str:
    cpu_s = sum(t["seconds"] for t in trained)
    first = next(iter(evals.values()))["result"]
    refs = {c.name: c for c in first.conditions}
    lines = [
        "# Squad: helping a learned code form (R6 follow-up)",
        "",
        f"- Date: {datetime.date.today().isoformat()}",
        f"- Commit: {_commit()} (code; this report is committed on top of it)",
        f"- Compute: {_cpu_name()}, {os.cpu_count()} logical cores, {args.workers} "
        f"worker processes, numpy/numba only. Training wall time "
        f"{train_seconds / 60:.1f} min ({cpu_s / 3600:.2f} CPU-hours), evaluation "
        f"{eval_seconds / 60:.1f} min.",
        f"- Training: condition (c) `{C}` of `learned_protocol` (joint ES of centre "
        f"and specialists, r = {base.radius}), {args.runs} runs (run indices "
        f"0–{args.runs - 1}) per arm, {base.generations} generations × population "
        f"{base.population} × {base.episodes_per_member} episodes per member; all "
        "other settings as in `learned_protocol.md`. Run s of every arm starts from "
        "the same weights and sees the same ES noise and training episodes, so arms "
        "are compared paired by run.",
        f"- Held-out evaluation: run s plays test episodes 3,500,000,000 + s × 10,000 "
        f"+ i, i < {args.eval_episodes} (the same seeds as `learned_protocol.md`).",
        f"- Arms: `baseline` (R6 method); `entropy` (fitness += w·H(M), w from "
        f"{args.entropy_bonus:g} to 0 linearly over the first "
        f"{args.entropy_anneal:.0%} of generations); `temperature` (centre samples "
        f"from softmax(logits / T) in training, T from {args.centre_temperature:g} to "
        f"0 over the first {args.temperature_anneal:.0%}; argmax for validation and "
        "test).",
        f"- **Code formed** (per run): H(M) > {H_MIN} bits on the held-out episodes "
        "and the `shuffle_batch` drop (intact − shuffled team food, paired t-CI over "
        f"the {args.eval_episodes} episodes) has a 95% lower bound > 0. Shuffling, not "
        "zeroing, is the test: for learned listeners a zeroed message is a wrong "
        "message, so it costs food even when the code carries no information (R6's "
        "failed run lost 5.4 food to zeroing but 0.1 to shuffling).",
        "",
        "Reproduce with `python scripts/code_formation.py --runs "
        f"{args.runs} --generations {base.generations} --population {base.population} "
        f"--episodes-per-member {base.episodes_per_member} --entropy-bonus "
        f"{args.entropy_bonus:g} --entropy-anneal {args.entropy_anneal:g} "
        f"--centre-temperature {args.centre_temperature:g} --temperature-anneal "
        f"{args.temperature_anneal:g} --eval-episodes {args.eval_episodes} --workers "
        f"{args.workers} --out {args.out.as_posix()}`.",
        "",
        "## Interpretation",
        "",
        "<!-- discussion -->",
        "",
        "## Code formation and team food (held-out)",
        "",
        "Mean ± 95% t-CI over training runs. References on the same test seeds: "
        f"scripted `{SCRIPTED}` {_pm(refs[SCRIPTED].mean, refs[SCRIPTED].ci)}, "
        f"no-comm {_pm(refs['no-comm'].mean, refs['no-comm'].ci)}, oracle "
        f"{_pm(refs['oracle'].mean, refs['oracle'].ci)}.",
        "",
        *summary_table(rows),
        "",
        "Paired by run; Holm across the arms compared with the baseline. The Fisher "
        "test ignores the pairing (conservative).",
        "",
        *comparison_table(rows),
        "",
        "## Run by run",
        "",
        "team food / H(M) bits / shuffle drop [95% lower bound] / code formed.",
        "",
        *per_run_table(evals),
        "",
        "## Learning curves (validation, ES mean, argmax centre; mean over runs)",
        "",
        "Team food:",
        "",
        *curve_table(trained, "val_team_food"),
        "",
        "H(M) bits of the centre's symbols on the validation episodes:",
        "",
        *curve_table(trained, "val_message_entropy"),
        "",
        "## Harness output",
        "",
        "`evaluation.format_table` per arm (condition (c) with the scripted reference "
        "and the registered baselines, message ablations, communication metrics).",
        "",
    ]
    for arm, ev in evals.items():
        lines += [f"### {arm}", "", "```", format_table(ev["result"]), "```", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--generations", type=int, default=600)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--episodes-per-member", type=int, default=8)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--entropy-bonus", type=float, default=1.0)
    parser.add_argument("--entropy-anneal", type=float, default=0.5)
    parser.add_argument("--centre-temperature", type=float, default=1.0)
    parser.add_argument("--temperature-anneal", type=float, default=0.5)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress", type=float, default=300.0)
    parser.add_argument(
        "--cache",
        type=Path,
        help="Folder for per-run results (default <out stem>_runs next to --out); "
        "finished runs with the same settings are loaded instead of retrained",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/results/code_formation.md")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = TrainSettings(
        radius=args.radius,
        generations=args.generations,
        population=args.population,
        episodes_per_member=args.episodes_per_member,
        eval_every=args.eval_every,
        progress_s=args.progress,
    )
    settings = {arm: arm_settings(arm, base, args) for arm in args.arms}
    stem, folder = args.out.stem, args.out.parent
    folder.mkdir(parents=True, exist_ok=True)
    params_path = folder / f"{stem}_params.npz"
    cache = args.cache or folder / f"{stem}_runs"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(a, run, settings[a], cache) for run in range(args.runs) for a in args.arms]
    print(
        f"[start] {len(jobs)} runs on {args.workers} workers; cache {cache}", flush=True
    )
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        trained = list(pool.map(_train_job, jobs))
    wall_now = time.perf_counter() - start
    # Wall time across restarts: this invocation's plus the CPU time of the runs it
    # loaded from the cache spread over the workers (an estimate)
    cached_cpu = sum(t["seconds"] for t in trained if t.get("cached"))
    train_seconds = wall_now + cached_cpu / args.workers
    np.savez_compressed(
        params_path,
        **{f"{t['arm']}_run{t['run']}": t["theta"].astype(np.float32) for t in trained},
    )
    print(f"[train] done in {train_seconds:.0f} s; evaluating", flush=True)

    esettings = EvalSettings(n_episodes=args.eval_episodes)
    start = time.perf_counter()
    eval_jobs = [
        (a, [t for t in trained if t["arm"] == a], settings[a], esettings)
        for a in args.arms
    ]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(eval_jobs))) as pool:
        evals = {ev["arm"]: ev for ev in pool.map(_eval_job, eval_jobs)}
    eval_seconds = time.perf_counter() - start
    rows = summary_rows(evals)
    text = render(evals, rows, trained, args, base, train_seconds, eval_seconds)
    payload = {
        "date": datetime.date.today().isoformat(),
        "commit": _commit(),
        "cpu": _cpu_name(),
        "workers": args.workers,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "cpu_hours": sum(t["seconds"] for t in trained) / 3600,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "train_settings": {a: asdict(s) for a, s in settings.items()},
        "eval_settings": asdict(esettings),
        "code_formed_rule": f"H(M) > {H_MIN} bits and shuffle_batch drop 95% low > 0",
        "summary": rows,
        "per_run": {a: ev["runs"] for a, ev in evals.items()},
        "results": {a: ev["result"].to_dict() for a, ev in evals.items()},
        "curves": [
            {"arm": t["arm"], "run": t["run"], "curve": t["curve"]} for t in trained
        ],
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(_finite(payload), indent=1, allow_nan=False, default=float),
        encoding="utf-8",
    )
    args.out.write_text(text, encoding="utf-8")
    print("\n".join(summary_table(rows) + [""] + comparison_table(rows)))
    print(f"Wrote {args.out} and {args.out.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    main()
