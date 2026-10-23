"""Does a learned protocol match a hand-coded one at the same bit budget? (R6)

Trains three conditions per radius with the same evolution-strategies budget (learned
specialists with a silent centre, with the fixed hand-coded centre, and jointly with a
learned centre), then scores them on held-out seeds through the CommArena harness
against the scripted references, with message ablations and communication metrics.

The driver behind ``scripts/learned_protocol.py`` and ``genetic-coil learned-protocol``
(both take the flags of ``parse_args``). Small CPU run (about an hour on 4 cores):

    python scripts/learned_protocol.py --radii 3 --runs 6 --generations 400 \
        --workers 4 --out docs/results/learned_protocol.md

A bigger run is a flag change, e.g. on 20 cores:

    python scripts/learned_protocol.py --radii 3 5 --runs 10 --generations 2000 \
        --population 256 --episodes-per-member 16 --workers 20

Outputs next to ``--out``: ``<stem>.json`` (every ExperimentResult + training
summaries), ``<stem>_curves.csv`` (learning curves) and ``<stem>_params.npz``
(trained parameters, float32).
"""

import argparse
import csv
import datetime
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..evaluation import ExperimentResult, format_table
from .learned_protocol import (
    CONDITION,
    HAND_CODED,
    LEARNED,
    MODES,
    SCRIPTED,
    SILENT,
    TEST_SEED_BASE,
    EvalSettings,
    TrainSettings,
    evaluate_radius,
    train_run,
)

A, B, C = CONDITION[SILENT], CONDITION[HAND_CODED], CONDITION[LEARNED]


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _pm(mean: float, half: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} ± {half:.{digits}f}"


def _p(value: float) -> str:
    return f"{value:.2g}" if value < 0.01 else f"{value:.3f}"


def _finite(value: Any) -> Any:
    """Copy of a JSON-like structure with non-finite floats (inf, NaN) as None."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def load_trained(
    report: Path, base: TrainSettings
) -> tuple[list[dict[str, Any]], float]:
    """Trained runs saved by an earlier invocation, and its training wall time.

    Raises if that run used different training settings (radius aside).
    """
    data = json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))
    defaults = _finite(asdict(TrainSettings()))  # Keys older runs did not record
    old = {
        k: v for k, v in {**defaults, **data["train_settings"]}.items() if k != "radius"
    }
    new = {k: v for k, v in _finite(asdict(base)).items() if k != "radius"}
    skip = ("time_limit_s", "progress_s")
    diff = {k for k in new if k not in skip and old.get(k) != new[k]}
    if diff:
        raise ValueError(f"--reuse run used different settings: {sorted(diff)}")
    params = np.load(report.with_name(f"{report.stem}_params.npz"))
    curves: dict[tuple[int, str, int], list[dict[str, float]]] = {}
    mode_of = {name: mode for mode, name in CONDITION.items()}
    with report.with_name(f"{report.stem}_curves.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (int(row["radius"]), mode_of[row["condition"]], int(row["run"]))
            curves.setdefault(key, []).append(
                {
                    k: (int(float(row[k])) if k in ("generation", "episodes")
                        else float(row[k]))
                    for k in row
                    if k not in ("radius", "condition", "run")
                }  # fmt: skip
            )
    trained = []
    for t in data["training"]:
        key = (t["radius"], t["mode"], t["run"])
        theta = params[f"r{key[0]}_{key[1]}_run{key[2]}"].astype(np.float64)
        trained.append({**t, "theta": theta, "curve": curves[key]})
    return trained, float(data["train_seconds"])


def _train_job(args: tuple[str, int, TrainSettings]) -> dict[str, Any]:
    mode, run, settings = args
    out = train_run(mode, run, settings)
    print(
        f"[train] r={settings.radius} {mode:10s} run {run}: "
        f"{out['generations_done']} gens in {out['seconds']:.0f} s, best val food "
        f"{max(r['val_team_food'] for r in out['curve']):.2f} "
        f"(gen {out['best_generation']})",
        flush=True,
    )
    return out


def _eval_job(args: tuple[int, list, TrainSettings, EvalSettings]) -> ExperimentResult:
    radius, trained, tsettings, esettings = args
    return evaluate_radius(radius, trained, tsettings, esettings)


# ------------------------------------------------------------------ tables
def performance_table(results: dict[int, ExperimentResult]) -> list[str]:
    names = [c.name for c in next(iter(results.values())).conditions]
    lines = [
        "| condition | role | " + " | ".join(f"r = {r}" for r in results) + " |",
        "|---|---|" + "---|" * len(results),
    ]
    for name in names:
        role = next(iter(results.values())).condition(name).role
        cells = [
            _pm(res.condition(name).mean, res.condition(name).ci)
            for res in results.values()
        ]
        lines.append(f"| {name} | {role} | " + " | ".join(cells) + " |")
    return lines


def headline_table(results: dict[int, ExperimentResult]) -> list[str]:
    lines = [
        "| r | comparison | mean diff [95% Welch CI] | p (Mann-Whitney, Holm) | A12 |",
        "|---|---|---|---|---|",
    ]
    for r, res in results.items():
        for row in res.metadata["headline"]:
            lines.append(
                f"| {r} | {row['a']} vs {row['b']} | {row['mean_diff']:.2f} "
                f"[{row['mean_diff'] - row['welch_ci']:.2f}, "
                f"{row['mean_diff'] + row['welch_ci']:.2f}] | {_p(row['p_holm'])} "
                f"| {row['a12']:.2f} |"
            )
    return lines


def ablation_table(results: dict[int, ExperimentResult]) -> list[str]:
    lines = [
        "| r | system | intervention | intact | ablated | drop [95% CI] "
        "| p (Holm) | A12 | listening |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r, res in results.items():
        for a in res.ablations:
            lines.append(
                f"| {r} | {a.system} | {a.intervention} | {a.intact_mean:.2f} "
                f"| {a.ablated.mean:.2f} | {a.delta_mean:.2f} "
                f"[{a.delta_low:.2f}, {a.delta_high:.2f}] | {_p(a.p_holm)} "
                f"| {a.a12:.2f} | {'yes' if a.listening else 'no'} |"
            )
    return lines


def _m(m: dict[str, float], key: str, digits: int = 3) -> str:
    value = m.get(key, float("nan"))
    if not np.isfinite(value):
        return "n/a"
    ci = m.get(f"{key}_ci", float("nan"))
    if np.isfinite(ci):
        return _pm(value, ci, digits)
    return f"{value:.{digits}f}"


def communication_table(results: dict[int, ExperimentResult]) -> list[str]:
    lines = [
        "| r | squad | H(M) bits | used / 3 bits | codes used | H(T) bits | MI(M;T) bits "
        "| MI null q95 | MI(M;direction) | MI(M;far) | topsim | CIC bits |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r, res in results.items():
        for name, m in res.communication.items():
            lines.append(
                f"| {r} | {name} | {_m(m, 'message_entropy_bits')} "
                f"| {100 * m['utilization']:.0f}% | {_m(m, 'distinct_codes', 1)} "
                f"| {_m(m, 'target_entropy_bits')} | {_m(m, 'mi_bits')} "
                f"| {m['mi_null_q95_bits']:.3f} | {_m(m, 'mi_direction_bits')} "
                f"| {_m(m, 'mi_far_bits')} | {_m(m, 'topsim')} | {_m(m, 'cic_bits')} |"
            )
    return lines


def influence_table(results: dict[int, ExperimentResult]) -> list[str]:
    lines = [
        "| r | squad | JS(shuffle_batch) bits | action changed (shuffle_batch) "
        "| JS(random_matched) bits | action changed (random_matched) |",
        "|---|---|---|---|---|---|",
    ]
    for r, res in results.items():
        for name, m in res.communication.items():
            if "cic_bits" not in m:
                continue
            lines.append(
                f"| {r} | {name} | {_m(m, 'js_shuffle_batch_bits')} "
                f"| {_m(m, 'changed_shuffle_batch')} | {_m(m, 'js_random_matched_bits')} "
                f"| {_m(m, 'changed_random_matched')} |"
            )
    return lines


LABELS = [
    f"{d}/{f}" for d in ("ahead", "right", "behind", "left") for f in ("near", "far")
]


def codebook_table(results: dict[int, ExperimentResult]) -> list[str]:
    lines: list[str] = []
    for r, res in results.items():
        for run, table in zip(res.seeds, res.metadata.get("codebooks", [])):
            lines += [
                f"r = {r}, training run {run} (counts of live receivers; rows = code "
                "sent, columns = the receiver's nearest own-colour food):",
                "",
                "| code | " + " | ".join(LABELS + ["no food"]) + " |",
                "|---|" + "---|" * (len(LABELS) + 1),
            ]
            for code, row in enumerate(table):
                if sum(row):
                    cells = " | ".join(str(v) for v in row)
                    lines.append(f"| {code} | {cells} |")
            lines.append("")
    return lines


def per_run_table(results: dict[int, ExperimentResult]) -> list[str]:
    """Learned-centre system, one row per training run (codes differ between runs)."""
    lines = [
        "| r | run | team food (c) | (b) | (a) | H(M) bits | codes used | MI(M;T) "
        "| MI(M;direction) | MI(M;far) | CIC bits | drop when zeroed | drop when "
        "shuffled |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r, res in results.items():
        per = res.metadata.get("per_run_communication", {}).get(C)
        if not per:
            continue
        abl = {
            a.intervention: a.ablated.per_seed for a in res.ablations if a.system == C
        }
        food = {n: res.condition(n).per_seed for n in (A, B, C)}
        for i, run in enumerate(res.seeds):
            lines.append(
                f"| {r} | {run} | {food[C][i]:.2f} | {food[B][i]:.2f} "
                f"| {food[A][i]:.2f} | {per['message_entropy_bits'][i]:.2f} "
                f"| {per['distinct_codes'][i]:.0f} | {per['mi_bits'][i]:.2f} "
                f"| {per['mi_direction_bits'][i]:.2f} | {per['mi_far_bits'][i]:.2f} "
                f"| {per['cic_bits'][i]:.3f} | {food[C][i] - abl['zero'][i]:.2f} "
                f"| {food[C][i] - abl['shuffle_batch'][i]:.2f} |"
            )
    return lines


def curve_table(trained: list[dict[str, Any]], radii: list[int]) -> list[str]:
    """Validation team food of the mean parameters at a few checkpoints, mean over runs."""
    lines = []
    for r in radii:
        runs = [t for t in trained if t["radius"] == r]
        gens = sorted({row["generation"] for t in runs for row in t["curve"]})
        picks = sorted(
            {
                gens[int(round(q * (len(gens) - 1)))]
                for q in (0, 0.1, 0.25, 0.5, 0.75, 1)
            }
        )
        lines += [
            f"r = {r} (validation team food of the ES mean, mean over runs):",
            "",
            "| condition | "
            + " | ".join(f"gen {g}" for g in picks)
            + " | best ckpt gen (mean) |",
            "|---|" + "---|" * (len(picks) + 1),
        ]
        for mode in MODES:
            mine = [t for t in runs if t["mode"] == mode]
            if not mine:
                continue
            cells = []
            for g in picks:
                vals = [
                    row["val_team_food"]
                    for t in mine
                    for row in t["curve"]
                    if row["generation"] == g
                ]
                cells.append(f"{np.mean(vals):.2f}" if vals else "-")
            best = np.mean([t["best_generation"] for t in mine])
            lines.append(
                f"| {CONDITION[mode]} | " + " | ".join(cells) + f" | {best:.0f} |"
            )
        lines.append("")
    return lines


def write_curves(trained: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "radius", "condition", "run", "generation", "episodes", "wall_s",
        "pop_mean_return", "pop_max_return", "val_return", "val_team_food",
        "val_message_entropy",
    ]  # fmt: skip
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="nan")
        writer.writeheader()
        for t in trained:
            for row in t["curve"]:
                writer.writerow(
                    {
                        "radius": t["radius"],
                        "condition": CONDITION[t["mode"]],
                        "run": t["run"],
                        **{
                            k: (f"{v:.4f}" if isinstance(v, float) else v)
                            for k, v in row.items()
                        },
                    }
                )


def auto_interpretation(results: dict[int, ExperimentResult]) -> list[str]:
    out = []
    for r, res in results.items():
        mean = {c.name: c.mean for c in res.conditions}
        rows = {(row["a"], row["b"]): row for row in res.metadata["headline"]}

        def verdict(a: str, b: str, rows: dict = rows) -> str:
            row = rows.get((a, b))
            if row is None:
                return "n/a"
            low, high = (
                row["mean_diff"] - row["welch_ci"],
                row["mean_diff"] + row["welch_ci"],
            )
            sign = (
                "higher"
                if low > 0
                else ("lower" if high < 0 else "not distinguishable")
            )
            return f"{row['mean_diff']:+.2f} [{low:+.2f}, {high:+.2f}], {sign}, p(Holm) = {_p(row['p_holm'])}"

        comm = res.communication.get(C, {})
        listen = [
            a.intervention for a in res.ablations if a.system == C and a.listening
        ]
        out.append(
            f"- **r = {r}:** team food (a) silent {mean.get(A, float('nan')):.2f}, "
            f"(b) hand-coded {mean.get(B, float('nan')):.2f}, (c) learned "
            f"{mean.get(C, float('nan')):.2f}; scripted greedy+hand-coded "
            f"{mean[SCRIPTED]:.2f}, no-comm {mean['no-comm']:.2f}, oracle "
            f"{mean['oracle']:.2f}. (c) − (a): {verdict(C, A)}. (c) − (b): "
            f"{verdict(C, B)}. (b) − (a): {verdict(B, A)}. The learned centre uses "
            f"{comm.get('message_entropy_bits', float('nan')):.2f} of 3 bits with "
            f"MI(M;T) = {comm.get('mi_bits', float('nan')):.2f} bits; CIC "
            f"{comm.get('cic_bits', float('nan')):.3f} bits (means over runs); "
            "interventions whose drop CI excludes 0: "
            f"{', '.join(f'`{x}`' for x in listen) or 'none'}."
        )
    return out


def code_formation_flags(settings: TrainSettings) -> str:
    """The non-default code-formation flags of a run, with a trailing space."""
    out = ""
    if settings.entropy_bonus > 0:
        out += (
            f"--entropy-bonus {settings.entropy_bonus:g} "
            f"--entropy-anneal {settings.entropy_anneal:g} "
        )
    if settings.centre_temperature > 0:
        out += (
            f"--centre-temperature {settings.centre_temperature:g} "
            f"--temperature-anneal {settings.temperature_anneal:g} "
        )
    return out


def render(
    results: dict[int, ExperimentResult],
    trained: list[dict[str, Any]],
    tsettings: TrainSettings,
    esettings: EvalSettings,
    runs: int,
    workers: int,
    train_seconds: float,
    eval_seconds: float,
    stem: str,
    scale: str = "cpu",
) -> str:
    radii = list(results)
    n_params = {t["mode"]: t["n_params"] for t in trained}
    gens = [t["generations_done"] for t in trained]
    gen_range = str(min(gens)) if min(gens) == max(gens) else f"{min(gens)}–{max(gens)}"
    cpu_s = sum(t["seconds"] for t in trained)
    episodes = [t["curve"][-1]["episodes"] for t in trained]
    lines = [
        "# Squad: learned protocol vs hand-coded protocol at 3 bits (R6)",
        "",
        f"- Date: {datetime.date.today().isoformat()}",
        f"- Commit: {_commit()} (code; this report is committed on top of it)",
        f"- Compute: {_cpu_name()}, {os.cpu_count()} logical cores, {workers} worker "
        f"processes, numpy/numba only (no GPU). Training wall time {train_seconds / 60:.1f} "
        f"min ({cpu_s / 3600:.2f} CPU-hours), evaluation {eval_seconds / 60:.1f} min.",
        f"- Training: {runs} independent runs per condition and radius; "
        f"{gen_range} ES generations × population {tsettings.population} × "
        f"{tsettings.episodes_per_member} episodes per member "
        f"(≈{np.mean(episodes) / 1e3:.0f}k episodes per run).",
        f"- Held-out evaluation: run s plays episodes {TEST_SEED_BASE:,} + s × 10,000 + "
        f"i, i < {esettings.n_episodes}. Training runs are the independent units for "
        "every CI and test (n = number of runs), so the CIs include training-seed "
        "variance, not only episode noise.",
        "- Squad: 21×21 walls, 3 snakes, 2 food per colour, health 60, T = 300; channel "
        "= 1 symbol, V = 8 (3 bits) per receiver, sent every step (k = 1, the "
        "hand-coded centre's interval), read one step later.",
        "",
        f"Reproduce with `python scripts/learned_protocol.py --scale {scale} --radii {' '.join(map(str, radii))} "
        f"--runs {runs} --generations {tsettings.generations} --population "
        f"{tsettings.population} --episodes-per-member {tsettings.episodes_per_member} "
        f"--workers {workers} {code_formation_flags(tsettings)}"
        f"--eval-episodes {esettings.n_episodes} --out docs/results/{stem}.md`.",
        "",
        "## Design",
        "",
        "- **Algorithm:** OpenAI-ES (`learning.EvolutionStrategy`: antithetic "
        f"Gaussian perturbations σ = {tsettings.sigma}, centred-rank shaping, Adam "
        f"lr = {tsettings.lr}, weight decay {tsettings.weight_decay}). Centre and "
        "specialists are one parameter vector, so the joint system is optimised "
        "end-to-end from the team return through the discrete argmax channel, with no "
        "gradient through the message. All members of a generation play the same "
        "episodes and action noise. Fitness = team return (food + 0.01 per living "
        "snake per step). Same budget, sizes and hyperparameters for every condition; "
        "the three conditions of run s start from the same specialist weights.",
        "- **Specialist (shared by the 3 snakes):** egocentric window planes "
        "(4·(2r+1)²), health, one-hot received code (8) → "
        f"{tsettings.spec_hidden} tanh → 3 action logits, actions sampled from the "
        f"softmax ({n_params.get(SILENT, n_params.get(LEARNED))} parameters at r = "
        f"{radii[0]}).",
        "- **Learned centre (shared across receivers):** per receiver, its two "
        "own-colour food items as egocentric (ahead, right) / 21 + valid flag, sorted "
        f"near to far, and its health (7 inputs) → {tsettings.centre_hidden} tanh → 8 "
        "logits; the code is the argmax "
        f"({n_params.get(LEARNED, 0) - n_params.get(SILENT, 0)} parameters). It gets "
        "food positions, not directions or distance classes: which partition of the "
        "plane to name is learned. Sorting by distance only removes a slot "
        "permutation.",
        "- **Conditions:** (a) `silent+learned-spec`: the centre always sends code 0; "
        "(b) `hand-coded+learned-spec`: the fixed `HandCodedCentre` (`2·direction + "
        "far`); (c) `learned-centre+learned-spec`. Plus the scripted references "
        "`greedy+hand-coded` and the registered baselines (random actions, greedy + "
        "silent / random centre, no-comm greedy, full-view oracle).",
        "- **Checkpoint:** the ES mean is scored every "
        f"{tsettings.eval_every} generations on {tsettings.val_episodes} fixed "
        "validation episodes (seeds 4,000,000,000 + run × 10,000 + i, disjoint from "
        "training and test); the best checkpoint is evaluated on the test seeds.",
        "- **Metrics:** computed per training run and averaged (runs learn different "
        "codes, so messages are never pooled). T = the receiver's nearest own-colour "
        "food: direction relative to heading (4) × near/far (Manhattan > 5) + 'no "
        "food' (the hand-coded meaning). topsim = Spearman between Hamming distance of "
        "(direction, far) and of the code, on receivers with food. CIC from the "
        "specialists' exact softmax.",
        "",
        "## Interpretation",
        "",
        *auto_interpretation(results),
        "",
        "<!-- discussion -->",
        "",
        "## Team food per episode (held-out)",
        "",
        "Mean ± 95% t-CI over training runs (each run's score is its mean over the test "
        "episodes; scripted systems and baselines use the same per-run episode sets).",
        "",
        *performance_table(results),
        "",
        "## Headline comparisons",
        "",
        "(c) vs (a): is learned communication useful at all; (c) vs (b): learned vs "
        "hand-coded protocol for learned listeners; (b) vs (a): do learned listeners "
        "use the hand-coded channel. Unpaired over training runs.",
        "",
        *headline_table(results),
        "",
        "## Message ablations (learned systems)",
        "",
        "Drop = intact − ablated team food, paired by training run; p = Wilcoxon "
        "signed-rank, Holm across the four interventions per system.",
        "",
        *ablation_table(results),
        "",
        "## What the centre says and how much it moves actions",
        "",
        "Mean ± 95% CI over training runs; MI null q95 from 100 target shuffles (mean "
        "over runs).",
        "",
        *communication_table(results),
        "",
        *influence_table(results),
        "",
        "### Learned centre, run by run",
        "",
        "(a), (b), (c) = silent, hand-coded and learned centre with learned "
        "specialists trained from the same initial specialist weights.",
        "",
        *per_run_table(results),
        "",
        "### Learned codebooks",
        "",
        *codebook_table(results),
        "## Learning curves",
        "",
        f"Full curves (every {tsettings.eval_every} generations, every run) in "
        f"`{stem}_curves.csv`.",
        "",
        *curve_table(trained, radii),
        "## Harness output",
        "",
        "`evaluation.format_table` per radius.",
        "",
    ]
    for r, res in results.items():
        lines += [f"### r = {r}", "", "```", format_table(res), "```", ""]
    return "\n".join(lines)


SCALES: dict[str, dict[str, Any]] = {
    # This report: 4 CPUs, about 75 min of training
    "cpu": {
        "radii": [3],
        "runs": 4,
        "generations": 1000,
        "population": 64,
        "episodes_per_member": 8,
        "workers": 4,
        "eval_episodes": 100,
    },
    # Tiny smoke test (a minute or two)
    "smoke": {
        "radii": [3],
        "runs": 2,
        "generations": 6,
        "population": 8,
        "episodes_per_member": 2,
        "workers": 2,
        "eval_episodes": 4,
        "eval_every": 3,
    },
    # The 20-core PC: numpy ES parallelised over processes (no GPU needed)
    "gpu": {
        "radii": [3, 5],
        "runs": 10,
        "generations": 3000,
        "population": 128,
        "episodes_per_member": 16,
        "workers": 20,
        "eval_episodes": 200,
    },
}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scale",
        choices=sorted(SCALES),
        default="cpu",
        help="Preset; any explicit flag below overrides it",
    )
    parser.add_argument("--radii", type=int, nargs="+")
    parser.add_argument("--runs", type=int, help="Training runs per condition")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--population", type=int)
    parser.add_argument("--episodes-per-member", type=int)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--spec-hidden", type=int, default=16)
    parser.add_argument("--centre-hidden", type=int, default=16)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument(
        "--entropy-bonus",
        type=float,
        default=0.0,
        help="Code formation: fitness += w * H(M) bits of the learned centre's "
        "symbols, w annealed linearly from this value to 0 (default 0: off)",
    )
    parser.add_argument(
        "--entropy-anneal",
        type=float,
        default=0.5,
        help="Fraction of the generations over which the entropy bonus anneals to 0",
    )
    parser.add_argument(
        "--centre-temperature",
        type=float,
        default=0.0,
        help="Code formation: the learned centre samples its symbol from "
        "softmax(logits / T) in training, T annealed from this value to 0 (argmax) "
        "(default 0: always argmax)",
    )
    parser.add_argument(
        "--temperature-anneal",
        type=float,
        default=0.5,
        help="Fraction of the generations over which the temperature anneals to 0",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=float("inf"),
        help="Safety cap in seconds per training run",
    )
    parser.add_argument(
        "--progress",
        type=float,
        default=120.0,
        help="Seconds between progress lines per training run",
    )
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--reuse",
        type=Path,
        help="Earlier report (.md path) whose trained runs (_params.npz, _curves.csv, "
        ".json) are reused instead of retrained; must use the same training settings",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/results/learned_protocol.md")
    )
    args = parser.parse_args(argv)  # fmt: skip
    preset = {"eval_every": 10, **SCALES[args.scale]}
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def train_settings(args: argparse.Namespace) -> TrainSettings:
    """The training settings the parsed flags stand for (radius left at its default)."""
    return TrainSettings(
        generations=args.generations,
        population=args.population,
        episodes_per_member=args.episodes_per_member,
        sigma=args.sigma,
        lr=args.lr,
        spec_hidden=args.spec_hidden,
        centre_hidden=args.centre_hidden,
        eval_every=args.eval_every,
        time_limit_s=args.time_limit,
        progress_s=args.progress,
        entropy_bonus=args.entropy_bonus,
        entropy_anneal=args.entropy_anneal,
        centre_temperature=args.centre_temperature,
        temperature_anneal=args.temperature_anneal,
    )


def main(argv: Optional[list[str]] = None) -> None:
    """Train, evaluate and write the report; ``argv`` as on the command line."""
    args = parse_args(argv)
    base = train_settings(args)
    esettings = EvalSettings(n_episodes=args.eval_episodes)
    reused, reused_seconds = load_trained(args.reuse, base) if args.reuse else ([], 0.0)
    have = {(t["radius"], t["mode"], t["run"]) for t in reused}
    jobs = [
        (mode, run, replace(base, radius=r))
        for r in args.radii
        for run in range(args.runs)
        for mode in args.modes
        if (r, mode, run) not in have
    ]
    if reused:
        print(f"[reuse] {len(reused)} trained runs from {args.reuse}", flush=True)
    print(f"[start] scale={args.scale}: {len(jobs)} training runs on "
          f"{args.workers} workers; {base}", flush=True)  # fmt: skip
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        trained = list(pool.map(_train_job, jobs))
    wanted = {
        (r, m, k) for r in args.radii for m in args.modes for k in range(args.runs)
    }
    trained = [
        t for t in reused if (t["radius"], t["mode"], t["run"]) in wanted
    ] + trained
    trained.sort(key=lambda t: (t["radius"], t["run"], MODES.index(t["mode"])))
    train_seconds = time.perf_counter() - start + reused_seconds
    print(f"[train] done in {train_seconds:.0f} s; evaluating", flush=True)

    start = time.perf_counter()
    eval_jobs = [(r, trained, base, esettings) for r in args.radii]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(eval_jobs))) as pool:
        results = dict(zip(args.radii, pool.map(_eval_job, eval_jobs)))
    eval_seconds = time.perf_counter() - start

    stem = args.out.stem
    targets = [args.out.parent]
    if os.environ.get("GCA_RUN_DIR"):
        targets.append(Path(os.environ["GCA_RUN_DIR"]))
    payload = {
        "results": {str(r): res.to_dict() for r, res in results.items()},
        "training": [
            {k: v for k, v in t.items() if k not in ("theta", "theta_final", "curve")}
            for t in trained
        ],
        "scale": args.scale,
        "train_settings": asdict(base),
        "eval_settings": asdict(esettings),
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "workers": args.workers,
    }
    text = render(
        results, trained, base, esettings, args.runs, args.workers,
        train_seconds, eval_seconds, stem, args.scale,
    )  # fmt: skip
    for folder in targets:
        folder.mkdir(parents=True, exist_ok=True)
        write_curves(trained, folder / f"{stem}_curves.csv")
        np.savez_compressed(
            folder / f"{stem}_params.npz",
            **{
                f"r{t['radius']}_{t['mode']}_run{t['run']}": t["theta"].astype(
                    np.float32
                )
                for t in trained
            },
        )
        (folder / f"{stem}.json").write_text(
            json.dumps(_finite(payload), indent=1, allow_nan=False, default=float),
            encoding="utf-8",
        )
        (folder / f"{stem}.md").write_text(text, encoding="utf-8")
        print(
            f"Wrote {folder / stem}.{{md,json}}, _curves.csv, _params.npz", flush=True
        )
    for res in results.values():
        print(format_table(res))
    print(f"Done: train {train_seconds:.0f} s, eval {eval_seconds:.0f} s", flush=True)


if __name__ == "__main__":
    main()
