"""The "add a specialist" test (R7): driver, compute analysis and report.

Trains a newcomer snake into frozen R6 squads (learned language, hand-coded protocol,
silent) and a jointly retrained 4-snake squad as the reference, evaluates everything
on held-out seeds, and writes ``<out>.md`` + ``<out>.json`` (+ ``_curves.csv``).
See ``add_specialist`` for the design. The driver behind ``scripts/add_specialist.py``.

    python scripts/add_specialist.py --scale smoke
    python scripts/add_specialist.py --scale cpu --out docs/results/add_specialist_pilot.md
    python scripts/add_specialist.py --scale gpu      # 16 workers, see SCALES

Source squads: ``--source`` is an R6 ``learned_protocol_params.npz`` path, or
``git:<ref>:<path>`` read with ``git show`` (default: the r6-c-r3 GPU run's artifact
on the ``gpu-results`` branch, so a GPU job needs no extra files).

Interruptions: every finished training run is saved at once to the cache folder
(``--cache``, default ``$GCA_RUN_DIR/<stem>_runs`` or ``outputs/<stem>_runs``); a rerun
with the same settings and source loads those and trains only the rest.
``--resume-job <id>`` (GPU runner only) first copies the cache of an earlier job.

Compute analysis (per run k): reference food ``ref_k`` = the joint squad's per-snake
food on the held-out test episodes (best checkpoint); threshold = ``--target`` (0.9)
× ref_k. Compute to threshold = training episodes (generations × population ×
episodes per member) until the trailing 3-checkpoint mean of the validation curve
first reaches it: the newcomer's own food for newcomer conditions, the per-snake
mean for the joint squad. Kill criterion: if the median over runs of
(newcomer under the frozen learned language) / (joint retraining) compute exceeds
``--kill-ratio`` (0.30), counting a newcomer that never gets there as infinite, the
verdict is "the ecosystem needs a fixed typed core protocol".
"""

import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..evaluation import ExperimentResult, format_table, mean_ci
from . import run_cache
from .add_specialist import (
    COPY,
    HAND_CODED,
    INITS,
    JOINT,
    LEARNED,
    SCRATCH,
    SILENT,
    SOURCE_MODES,
    AddSettings,
    evaluate,
    generations_to,
    old_food_check,
    source_key,
    system_name,
    train_run,
)
from .learned_protocol_report import _commit, _cpu_name

DEFAULT_SOURCE = "git:gpu-results:runs/r6-c-r3/artifacts/learned_protocol_params.npz"
CONDITION_ORDER = (LEARNED, HAND_CODED, JOINT, SILENT)


# ------------------------------------------------------------------ source squads
def read_source(spec: str) -> bytes:
    """Bytes of an R6 params file: a path, or ``git:<ref>:<path>`` (tries the ref,
    then ``origin/<ref>``)."""
    if not spec.startswith("git:"):
        return Path(spec).read_bytes()
    ref, path = spec[4:].split(":", 1)
    errors = []
    for candidate in (ref, f"origin/{ref}"):
        proc = subprocess.run(
            ["git", "show", f"{candidate}:{path}"], capture_output=True, check=False
        )
        if proc.returncode == 0:
            return proc.stdout
        errors.append(proc.stderr.decode(errors="replace").strip())
    raise FileNotFoundError(f"Cannot read {spec}: {'; '.join(errors)}")


def load_sources(
    data: bytes, radius: int, runs: int
) -> dict[tuple[str, int], np.ndarray]:
    """(mode, run) -> float64 parameters for runs 0..runs-1 of every source mode."""
    with np.load(io.BytesIO(data)) as npz:
        out = {}
        for mode in SOURCE_MODES:
            for run in range(runs):
                key = source_key(radius, mode, run)
                if key not in npz.files:
                    raise KeyError(f"Source file has no {key}")
                out[(mode, run)] = npz[key].astype(np.float64)
    return out


# ------------------------------------------------------------------ training jobs
def job_key(condition: str, init: str, run: int) -> str:
    return (
        f"{condition}_run{run}"
        if condition == JOINT
        else f"{condition}_{init}_run{run}"
    )


def _train_job(job: tuple) -> dict[str, Any]:
    condition, init, run, settings, source, source_sha, cache = job
    key = job_key(condition, init, run)
    ident = {
        **asdict(replace(settings, progress_s=0.0)),
        "condition": condition,
        "init": init,
        "run": run,
        "source_sha256": None if condition == JOINT else source_sha,
    }
    if condition == JOINT:
        del ident["fitness"]  # The joint squad always trains on the team return
    if cache is not None:
        hit = run_cache.load(cache, key, ident)
        if hit is not None:
            print(f"[cache] {key}", flush=True)
            return hit
    out = train_run(condition, run, settings, source, init)
    out.pop("theta_final")
    if cache is not None:
        summary = {k: v for k, v in out.items() if k != "theta"}
        run_cache.save(cache, key, ident, {"theta": out["theta"]}, summary)
    last = out["curve"][-1]
    print(
        f"[train] {key}: {out['generations_done']} gens in {out['seconds']:.0f} s "
        f"({out['seconds'] / max(out['generations_done'], 1):.2f} s/gen), final val "
        f"newcomer food {last['val_newcomer_food']:.2f}, team "
        f"{last['val_team_food']:.2f} (best gen {out['best_generation']})",
        flush=True,
    )
    return out


def _eval_job(job: tuple) -> Any:
    kind, settings, trained, sources, n_episodes = job
    if kind == "harness":
        return evaluate(settings, trained, sources, n_episodes)
    newcomers = [t for t in trained if t["condition"] != JOINT]
    return old_food_check(settings, newcomers, sources, n_episodes)


# ------------------------------------------------------------------ analysis
def compute_analysis(
    trained: list[dict[str, Any]],
    result: ExperimentResult,
    target: float = 0.9,
    kill_ratio: float = 0.3,
) -> dict[str, Any]:
    """Compute to reach ``target`` × the joint squad's per-snake food, per run."""
    joint = {int(t["run"]): t for t in trained if t["condition"] == JOINT}
    food = result.metadata["per_run_food"]
    seeds = list(result.seeds)
    ref = {}
    if joint:
        per_snake = food[system_name(JOINT)]["per_snake"]
        ref = {run: per_snake[seeds.index(run)] for run in joint}
    old_ref_food = food.get("newcomer:random", {}).get("old")
    rows = []
    for t in sorted(trained, key=lambda t: (t["condition"], t["init"], t["run"])):
        run = int(t["run"])
        row: dict[str, Any] = {
            "condition": t["condition"],
            "init": t["init"],
            "run": run,
            "generations_done": t["generations_done"],
            "episodes_done": t["curve"][-1]["episodes"],
            "seconds": t["seconds"],
        }
        key = "val_per_snake_food" if t["condition"] == JOINT else "val_newcomer_food"
        if run in ref:
            row["reference_food"] = ref[run]
            row["threshold"] = target * ref[run]
            row["reached"] = generations_to(t["curve"], key, row["threshold"])
        if old_ref_food is not None and t["condition"] != JOINT:
            old_mean = old_ref_food[seeds.index(run)] / 3.0
            row["old_per_snake_food"] = old_mean
            row["reached_old_level"] = generations_to(
                t["curve"], key, target * old_mean
            )
        rows.append(row)
    joint_rows = {r["run"]: r for r in rows if r["condition"] == JOINT}
    for row in rows:
        j = joint_rows.get(row["run"])
        if row["condition"] == JOINT or j is None or "reached" not in row:
            continue
        mine, theirs = row["reached"], j["reached"]
        if theirs is None or theirs["episodes"] <= 0:
            row["ratio"] = float("nan")
        elif mine is None:
            row["ratio"] = float("inf")
        else:
            row["ratio"] = mine["episodes"] / theirs["episodes"]
        full = j["episodes_done"]
        row["ratio_vs_joint_budget"] = (
            float("inf") if mine is None else mine["episodes"] / full
        )
        row["wall_ratio"] = (
            float("inf")
            if mine is None
            else (mine["wall_s"] / theirs["wall_s"] if theirs else float("nan"))
        )
    summary = []
    for cond in CONDITION_ORDER:
        for init in INITS:
            mine = [r for r in rows if r["condition"] == cond and r["init"] == init]
            if not mine:
                continue
            ratios = np.array([r.get("ratio", np.nan) for r in mine], dtype=float)
            valid = ratios[~np.isnan(ratios)]
            reached = [r for r in mine if r.get("reached") is not None]
            entry = {
                "condition": cond,
                "init": init,
                "n_runs": len(mine),
                "n_reached": len(reached),
                "median_ratio": float(np.median(valid)) if valid.size else np.nan,
                "share_within_kill_ratio": (
                    float(np.mean(valid <= kill_ratio)) if valid.size else np.nan
                ),
                "median_generations_to_threshold": (
                    float(
                        np.median(
                            [
                                (
                                    r["reached"]["generation"]
                                    if r.get("reached")
                                    else np.inf
                                )
                                for r in mine
                            ]
                        )
                    )
                    if ref
                    else np.nan
                ),
                "median_generations_to_old_level": float(
                    np.median(
                        [
                            (
                                r["reached_old_level"]["generation"]
                                if r.get("reached_old_level")
                                else np.inf
                            )
                            for r in mine
                        ]
                    )
                    if cond != JOINT and old_ref_food is not None
                    else np.nan
                ),
            }
            summary.append(entry)
    verdict = "n/a (no joint reference or no learned/scratch newcomer)"
    key_rows = [
        s for s in summary if s["condition"] == LEARNED and s["init"] == SCRATCH
    ]
    if key_rows and not np.isnan(key_rows[0]["median_ratio"]):
        med = key_rows[0]["median_ratio"]  # may be inf: most runs never got there
        if med > kill_ratio:
            verdict = (
                f"KILL: a from-scratch newcomer under the frozen learned language needs "
                f"{_ratio(med)} of joint-retraining compute (median over runs, > "
                f"{kill_ratio:.0%}): the ecosystem needs a fixed typed core protocol"
            )
        else:
            verdict = (
                f"PASS: a from-scratch newcomer under the frozen learned language needs "
                f"{_ratio(med)} of joint-retraining compute (median over runs, <= "
                f"{kill_ratio:.0%})"
            )
    return {
        "target": target,
        "kill_ratio": kill_ratio,
        "rows": rows,
        "summary": summary,
        "verdict": verdict,
    }


def _ratio(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if value == float("inf"):
        return "never (∞)"
    return f"{100 * value:.0f}%"


# ------------------------------------------------------------------ tables
def _pm(mean: float, half: float, digits: int = 2) -> str:
    if half is None or not np.isfinite(half):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {half:.{digits}f}"


def _p(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.2g}" if value < 0.01 else f"{value:.3f}"


def performance_table(result: ExperimentResult) -> list[str]:
    food = result.metadata["per_run_food"]
    lines = [
        "| system | role | newcomer food | old snakes' food (sum of 3) | team food |",
        "|---|---|---|---|---|",
    ]
    for c in result.conditions:
        old = mean_ci(food[c.name]["old"])
        team = mean_ci(food[c.name]["team"])
        lines.append(
            f"| {c.name} | {c.role} | {_pm(c.mean, c.ci)} | {_pm(*old)} "
            f"| {_pm(*team)} |"
        )
    return lines


def ablation_table(result: ExperimentResult) -> list[str]:
    lines = [
        "| system | intervention (newcomer's channel only) | intact | ablated "
        "| drop [95% CI] | p (Holm) | listening |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in result.ablations:
        lines.append(
            f"| {a.system} | {a.intervention} | {a.intact_mean:.2f} "
            f"| {a.ablated.mean:.2f} | {a.delta_mean:.2f} "
            f"[{a.delta_low:.2f}, {a.delta_high:.2f}] | {_p(a.p_holm)} "
            f"| {'yes' if a.listening else 'no'} |"
        )
    return lines


def compute_tables(analysis: dict[str, Any]) -> list[str]:
    lines = [
        "| condition | init | runs | reached threshold | median compute vs joint "
        "(to threshold) | runs within kill ratio | median gens to threshold "
        "| median gens to 90% of old snakes' food |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in analysis["summary"]:
        share = s["share_within_kill_ratio"]
        lines.append(
            f"| {s['condition']} | {s['init'] if s['condition'] != JOINT else '-'} "
            f"| {s['n_runs']} | {s['n_reached']}/{s['n_runs']} "
            f"| {_ratio(s['median_ratio']) if s['condition'] != JOINT else '100%'} "
            f"| {'n/a' if not np.isfinite(share) else f'{100 * share:.0f}%'} "
            f"| {_gens(s['median_generations_to_threshold'])} "
            f"| {_gens(s['median_generations_to_old_level'])} |"
        )
    lines += [
        "",
        "Per run:",
        "",
        "| condition | init | run | ref food (joint, per snake) | threshold | gens to "
        "threshold | episodes | compute vs joint | vs joint full budget | wall-time "
        "vs joint |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in analysis["rows"]:
        reached = r.get("reached")
        gens = "never" if reached is None else str(int(reached["generation"]))
        episodes = "-" if reached is None else f"{reached['episodes']:,.0f}"
        lines.append(
            f"| {r['condition']} | {r['init'] if r['condition'] != JOINT else '-'} "
            f"| {r['run']} | {r.get('reference_food', float('nan')):.2f} "
            f"| {r.get('threshold', float('nan')):.2f} | {gens} | {episodes} "
            f"| {_ratio(r.get('ratio', float('nan')))} "
            f"| {_ratio(r.get('ratio_vs_joint_budget', float('nan')))} "
            f"| {_ratio(r.get('wall_ratio', float('nan')))} |"
        )
    return lines


def _gens(value: float) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return "never" if value == np.inf else f"{value:.0f}"


def old_food_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| source squad | newcomer init | 3-snake squad | 4 snakes, random newcomer "
        "| 4 snakes, trained newcomer | trained − 3-snake [95% CI] (p) | trained − "
        "random newcomer [95% CI] (p) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        a, b = r["vs_three_snake"], r["vs_random_newcomer"]
        lines.append(
            f"| {r['mode']} | {r['init']} | {_pm(r['three_snake_mean'], r['three_snake_ci'])} "
            f"| {_pm(r['random_newcomer_mean'], r['random_newcomer_ci'])} "
            f"| {_pm(r['trained_newcomer_mean'], r['trained_newcomer_ci'])} "
            f"| {_pm(a['mean'], a['ci'])} ({_p(a['p'])}) "
            f"| {_pm(b['mean'], b['ci'])} ({_p(b['p'])}) |"
        )
    return lines


def curve_table(trained: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| condition | init | metric | "
        + " | ".join(f"q{int(100 * q)}" for q in (0, 0.1, 0.25, 0.5, 0.75, 1))
        + " |",
        "|---|---|---|" + "---|" * 6,
    ]
    for cond in CONDITION_ORDER:
        for init in INITS:
            mine = [t for t in trained if t["condition"] == cond and t["init"] == init]
            if not mine:
                continue
            key = "val_per_snake_food" if cond == JOINT else "val_newcomer_food"
            cells = []
            for q in (0, 0.1, 0.25, 0.5, 0.75, 1):
                vals, gens = [], []
                for t in mine:
                    row = t["curve"][int(round(q * (len(t["curve"]) - 1)))]
                    vals.append(row[key])
                    gens.append(row["generation"])
                cells.append(f"{np.mean(vals):.2f} (g{int(np.median(gens))})")
            lines.append(
                f"| {cond} | {init if cond != JOINT else '-'} | {key} | "
                + " | ".join(cells)
                + " |"
            )
    return lines


def write_curves(trained: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "condition", "init", "run", "generation", "episodes", "wall_s",
        "pop_mean_return", "val_return", "val_newcomer_return", "val_team_food", "val_newcomer_food",
        "val_old_food", "val_per_snake_food",
    ]  # fmt: skip
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="nan")
        writer.writeheader()
        for t in trained:
            for row in t["curve"]:
                writer.writerow(
                    {
                        "condition": t["condition"],
                        "init": t["init"],
                        "run": t["run"],
                        **{
                            k: (f"{v:.4f}" if isinstance(v, float) else v)
                            for k, v in row.items()
                        },
                    }
                )


def render(
    args: argparse.Namespace,
    settings: dict[str, AddSettings],
    trained: list[dict[str, Any]],
    result: ExperimentResult,
    old_rows: list[dict[str, Any]],
    analysis: dict[str, Any],
    train_seconds: float,
    eval_seconds: float,
    source_sha: str,
) -> str:
    base = settings[SCRATCH]
    cpu_h = sum(t["seconds"] for t in trained) / 3600
    pilot = args.scale != "gpu"
    title = "# Squad: can a new specialist join a learned language? (R7"
    title += ", PILOT)" if pilot else ")"
    gens = {k: s.generations for k, s in settings.items()}
    lines = [
        title,
        "",
        *(
            [
                f"> **Pilot ({args.scale} preset).** Small budgets on {os.cpu_count()} "
                "CPUs to check the pipeline and calibrate the GPU job; the joint "
                "reference is under-trained here, so the compute ratios are not the "
                "result.",
                "",
            ]
            if pilot
            else []
        ),
        f"- Date: {datetime.date.today().isoformat()}",
        f"- Commit: {_commit()} (code; this report is committed on top of it)",
        f"- Compute: {_cpu_name()}, {os.cpu_count()} logical cores, {args.workers} "
        f"worker processes. Training wall time {train_seconds / 60:.1f} min "
        f"({cpu_h:.2f} CPU-hours), evaluation {eval_seconds / 60:.1f} min.",
        f"- Source squads: R6 r = {base.radius} runs 0–{args.runs - 1} "
        f"(`{args.source}`, sha256 {source_sha[:12]}…): 3 snakes, one shared "
        "specialist network per squad, trained 2000 ES generations × population 128 × "
        "8 episodes.",
        f"- Training: ES population {base.population} × {base.episodes_per_member} "
        f"episodes per member; generations: newcomer from scratch {gens[SCRATCH]}, "
        f"newcomer copied {gens[COPY]}, joint retraining {gens[JOINT]}. Validation "
        f"every {base.eval_every} generations on {base.val_episodes} episodes.",
        f"- Held-out evaluation: {args.eval_episodes} test episodes per run (R6's test "
        "seeds, now with 4 snakes). Runs are the independent units for every CI/test.",
        "",
        f"Reproduce with `python scripts/add_specialist.py --scale {args.scale} --runs "
        f"{args.runs} --generations {gens[SCRATCH]} --copy-generations {gens[COPY]} "
        f"--joint-generations {gens[JOINT]} --population {base.population} "
        f"--episodes-per-member {base.episodes_per_member} --fitness {base.fitness} "
        f"--workers {args.workers} "
        f"--eval-episodes {args.eval_episodes} --out {args.out.as_posix()}`.",
        "",
        "## Verdict",
        "",
        f"- **Kill criterion:** {analysis['verdict']}.",
        "",
        "<!-- discussion -->",
        "",
        "## Design",
        "",
        "- **Newcomer:** a 4th snake with a 4th colour joins a trained 3-snake squad "
        "(21×21, 2 food per colour, so the board gains 2 food items and a body). Only "
        "its network trains (`EvolutionStrategy` mask = 1 on its block); the old "
        "specialists' shared network and the centre stay frozen. Newcomer fitness = "
        + (
            "its own return (its food + 0.01 per step alive): with the team return the "
            "frozen snakes' ~30 food per episode varies between ES members once the "
            "newcomer perturbs their trajectories and swamps its first food items. "
            "Harm to the old snakes is measured below. The joint squad trains on the "
            "team return, as in R6."
            if base.fitness == "newcomer"
            else "the team return, as in R6."
        ),
        "- **The frozen centre addresses the newcomer unchanged:** both R6 networks "
        "are receiver-agnostic (the centre sees each receiver's own-colour food "
        "egocentrically; specialists see own/other planes, not colours), so with N = 4 "
        "the centre encodes colour 3's food for snake 3 in the same code (tested: "
        "permuting snakes permutes the codes).",
        "- **Conditions:** (1) frozen learned centre + learned specialists; (2) frozen "
        "`HandCodedCentre` + its learned listeners; (3) joint retraining of all 4 snakes "
        "(one shared network, as in R6) + a learned centre from scratch; (4) silent "
        "squad. Newcomer init: `scratch` (random) or `copy` (the squad's own shared "
        "weights, a zero-shot clone because the network is colour-agnostic).",
        f"- **Compute metric:** reference = (3)'s per-snake food on the test episodes, "
        f"per run; threshold = {analysis['target']:.0%} of it. Compute to threshold = "
        "training episodes until the trailing 3-checkpoint mean of the validation "
        "curve reaches it (newcomer's food; per-snake mean for (3)). Ratio = newcomer / "
        "joint, run k vs run k; a newcomer that never reaches it counts as ∞. Kill if "
        f"the median ratio of (1)-scratch > {analysis['kill_ratio']:.0%}.",
        "- **Listening:** the harness's ablations applied to the newcomer's channel "
        "only (other receivers' messages untouched).",
        "",
        "## Compute to reach the jointly trained newcomer's food",
        "",
        *compute_tables(analysis),
        "",
        "## Newcomer food on held-out episodes",
        "",
        "Mean ± 95% t-CI over runs. Score = the newcomer's (snake 3's) food per "
        "episode. Baselines put a scripted newcomer into run s's frozen learned squad.",
        "",
        *performance_table(result),
        "",
        "Comparisons vs baselines (Mann-Whitney over runs, Holm):",
        "",
        "```",
        format_table(result),
        "```",
        "",
        "## Does the newcomer listen? (ablations of its channel)",
        "",
        "Drop = intact − ablated newcomer food, paired by run; p = Wilcoxon, Holm per "
        "system.",
        "",
        *ablation_table(result),
        "",
        "## Are the old specialists' food preserved?",
        "",
        "The three old snakes' summed food per held-out episode, mean ± 95% CI over "
        "runs; differences paired by run (t-CI, Wilcoxon p).",
        "",
        *old_food_table(old_rows),
        "",
        "## Learning curves",
        "",
        "Validation food (mean over runs) at quantiles of each run's curve (median "
        f"generation in brackets). Full curves in `{args.out.stem}_curves.csv`.",
        "",
        *curve_table(trained),
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ CLI
SCALES: dict[str, dict[str, Any]] = {
    # Tiny end-to-end check (a minute)
    "smoke": {
        "runs": 1,
        "generations": 4,
        "copy_generations": 2,
        "joint_generations": 4,
        "population": 8,
        "episodes_per_member": 2,
        "eval_every": 2,
        "val_episodes": 4,
        "eval_episodes": 4,
        "workers": 2,
    },
    # CPU pilot (3 workers on a shared 4-core box, about 1.5 h): 3 source squads
    "cpu": {
        "runs": 3,
        "generations": 250,
        "copy_generations": 50,
        "joint_generations": 1000,
        "population": 64,
        "episodes_per_member": 8,
        "eval_every": 10,
        "val_episodes": 64,
        "eval_episodes": 100,
        "workers": 3,
    },
    # The 20-core PC, at most 16 workers. 70 runs, 39,000 generations of pop 128 x 8
    # episodes; at 4 s/gen (R6 r=3 on that PC) x 1.45 (4 snakes) ~ 6 s/gen that is
    # ~65 worker-hours, ~4.1 h on 16 workers (joint runs first), + ~0.3 h evaluation
    "gpu": {
        "runs": 10,
        "generations": 800,
        "copy_generations": 100,
        "joint_generations": 1200,
        "population": 128,
        "episodes_per_member": 8,
        "eval_every": 10,
        "val_episodes": 64,
        "eval_episodes": 200,
        "workers": 16,
    },
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", choices=sorted(SCALES), default="cpu")
    parser.add_argument("--runs", type=int, help="Source squads 0..runs-1")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITION_ORDER),
        choices=CONDITION_ORDER,
    )
    parser.add_argument("--inits", nargs="+", default=list(INITS), choices=INITS)
    parser.add_argument("--generations", type=int, help="Newcomer from scratch")
    parser.add_argument("--copy-generations", type=int, help="Newcomer copied")
    parser.add_argument("--joint-generations", type=int, help="Joint retraining")
    parser.add_argument("--population", type=int)
    parser.add_argument("--episodes-per-member", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--val-episodes", type=int)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--fitness",
        choices=("newcomer", "team"),
        default="newcomer",
        help="Newcomer fitness: its own return (default) or the team return",
    )
    parser.add_argument("--target", type=float, default=0.9)
    parser.add_argument("--kill-ratio", type=float, default=0.3)
    parser.add_argument("--progress", type=float, default=180.0)
    parser.add_argument(
        "--time-limit",
        type=float,
        default=float("inf"),
        help="Safety cap in seconds per training run",
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--cache", type=Path, help="Per-run cache folder")
    parser.add_argument(
        "--resume-job",
        help="GPU runner: copy the per-run cache of this earlier job id first",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/results/add_specialist.md")
    )
    args = parser.parse_args(argv)
    for key, value in SCALES[args.scale].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.workers > 16 and args.scale == "gpu":
        raise SystemExit("The GPU PC keeps 4 cores free: use at most 16 workers")
    return args


def make_settings(args: argparse.Namespace) -> dict[str, AddSettings]:
    """Settings per job kind: SCRATCH and COPY newcomers, JOINT."""
    base = AddSettings(
        radius=args.radius,
        generations=args.generations,
        population=args.population,
        episodes_per_member=args.episodes_per_member,
        eval_every=args.eval_every,
        val_episodes=args.val_episodes,
        time_limit_s=args.time_limit,
        progress_s=args.progress,
        fitness=args.fitness,
    )
    return {
        SCRATCH: base,
        COPY: replace(base, generations=args.copy_generations),
        JOINT: replace(base, generations=args.joint_generations),
    }


def _resume(args: argparse.Namespace, cache: Path) -> None:
    run_dir = os.environ.get("GCA_RUN_DIR")
    if not args.resume_job or not run_dir:
        return
    old = Path(run_dir).parent.parent / args.resume_job / "artifacts" / cache.name
    if not old.is_dir():
        print(f"[resume] nothing at {old}", flush=True)
        return
    cache.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in old.iterdir():
        if path.suffix in (".npz", ".json") and ".tmp" not in path.name:
            if not (cache / path.name).exists():
                shutil.copy2(path, cache / path.name)
                n += 1
    print(f"[resume] copied {n} cached files from {old}", flush=True)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    settings = make_settings(args)
    data = read_source(args.source)
    source_sha = hashlib.sha256(data).hexdigest()
    sources = load_sources(data, args.radius, args.runs)
    stem = args.out.stem
    cache = args.cache or run_cache.default_cache_dir(stem)
    _resume(args, cache)

    jobs = []
    for run in range(args.runs):  # joint first: the longest jobs
        if JOINT in args.conditions:
            jobs.append((JOINT, SCRATCH, run, settings[JOINT], None, source_sha, cache))
    for init in sorted(args.inits, key=lambda i: settings[i].generations, reverse=True):
        for run in range(args.runs):
            for cond in CONDITION_ORDER:
                if cond in args.conditions and cond != JOINT:
                    src = sources[(cond, run)]
                    jobs.append(
                        (cond, init, run, settings[init], src, source_sha, cache)
                    )
    print(
        f"[start] scale={args.scale}: {len(jobs)} training runs on {args.workers} "
        f"workers; cache {cache}; {settings[SCRATCH]}",
        flush=True,
    )
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        trained = list(pool.map(_train_job, jobs))
    wall = time.perf_counter() - start
    cached_cpu = sum(t["seconds"] for t in trained if t.get("cached"))
    train_seconds = wall + cached_cpu / args.workers  # estimate across restarts
    print(f"[train] done in {train_seconds:.0f} s; evaluating", flush=True)

    start = time.perf_counter()
    eval_jobs = [
        (kind, settings[SCRATCH], trained, sources, args.eval_episodes)
        for kind in ("harness", "old_food")
    ]
    with ProcessPoolExecutor(max_workers=min(2, args.workers)) as pool:
        result, old_rows = list(pool.map(_eval_job, eval_jobs))
    eval_seconds = time.perf_counter() - start
    analysis = compute_analysis(trained, result, args.target, args.kill_ratio)
    print(f"[verdict] {analysis['verdict']}", flush=True)

    text = render(
        args, settings, trained, result, old_rows, analysis, train_seconds,
        eval_seconds, source_sha,
    )  # fmt: skip
    payload = {
        "scale": args.scale,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "settings": {k: asdict(s) for k, s in settings.items()},
        "source_sha256": source_sha,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "cpu_hours": sum(t["seconds"] for t in trained) / 3600,
        "analysis": analysis,
        "old_food": old_rows,
        "result": result.to_dict(),
        "training": [
            {k: v for k, v in t.items() if k not in ("theta", "curve")} for t in trained
        ],
    }
    targets = [args.out.parent]
    if os.environ.get("GCA_RUN_DIR"):
        targets.append(Path(os.environ["GCA_RUN_DIR"]))
    for folder in targets:
        folder.mkdir(parents=True, exist_ok=True)
        write_curves(trained, folder / f"{stem}_curves.csv")
        (folder / f"{stem}.json").write_text(
            json.dumps(
                run_cache.finite(payload), indent=1, allow_nan=False, default=float
            ),
            encoding="utf-8",
        )
        (folder / f"{stem}.md").write_text(text, encoding="utf-8")
        print(f"Wrote {folder / stem}.{{md,json}} and _curves.csv", flush=True)
    print(f"Done: train {train_seconds:.0f} s, eval {eval_seconds:.0f} s", flush=True)


if __name__ == "__main__":
    main()
