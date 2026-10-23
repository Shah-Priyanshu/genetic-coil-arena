"""Hand-designed byte baselines on the Rieth et al. runs, with calibrated false alarms (R8, step 2).

Re-measures the step-1 baselines on 500 independent simulation runs per fault, with
run-level splits, a stronger MLP centre whose hyperparameters are chosen on validation
runs only, and a normal-vs-fault margin calibrated on validation normal runs to a 5%
false-alarm target. Every number is mean ± 95% t-CI over independent replicates, each with
its own disjoint training, validation and test runs. Writes the Markdown report and the
per-cell JSON. Needs the data: run ``python scripts/fetch_rieth.py`` once.

    python scripts/plant_baselines_rieth.py --replicates 10 --workers 4 \
        --out docs/results/plant_bytes_baselines_rieth.md
"""

import os

# One BLAS thread per process: the machine is shared with long experiments
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import (  # noqa: E402
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

import numpy as np  # noqa: E402

from genetic_coil_arena.envs.plant.budget import (  # noqa: E402
    lorawan_airtime_s,
    min_uplink_period_s,
)
from genetic_coil_arena.envs.plant.replicates import (  # noqa: E402
    FAULT_SETS,
    Protocol,
    Replicate,
    summarise,
)
from genetic_coil_arena.envs.plant.rieth import RiethData  # noqa: E402
from genetic_coil_arena.envs.plant.tep import (  # noqa: E402
    FAULT_DESCRIPTIONS,
    NODE_VARIABLES,
    NODES,
    SAMPLE_MINUTES,
    describe_mapping,
)
from genetic_coil_arena.envs.plant.tuning import (  # noqa: E402
    STAGES,
    TOLERANCE,
    tune_centre,
)

# 1, 2 and 4 B are extra columns: the alarm byte's regime and where the curves leave the oracle
BUDGET_COLUMNS = (0, 1, 2, 4, 8, 16, 51)
ROWS = ("alarms", "summary", "features", "silent", "oracle", "noise")
LABELS = {
    "alarms": "(a) threshold alarms (1 B alarm byte per node)",
    "summary": "(b) quantised summary stats (mean, std, max, min)",
    "features": "(c) quantised hand features (PCA SPE/T2/scores, FFT bands)",
    "silent": "(d) silent (0 B, always reports normal)",
    "oracle": "(e) oracle (all raw variables, no byte limit)",
    "noise": "control: random bytes",
}
# Encoding -> byte budgets measured; the settings family it is tuned with
BUDGETS = {
    "alarms": (1,),
    "summary": (1, 2, 4, 8, 16, 51),
    "features": (1, 2, 4, 8, 16, 51),
    "noise": (8, 16, 51),
    "oracle": (0,),
    "silent": (0,),
}
# Cells that cost the most and add the least run on the first few replicates only (marked
# with a dagger in the tables): the control at 8 and 51 B, the centre ablation, the 20-fault set
SECONDARY_NOISE = (8, 51)
_N_FULL = (
    10  # replicates of the main cells, set by the report builder (for the daggers)
)
TUNE_CELLS = (("alarms", 1), ("summary", 16), ("features", 16), ("oracle", 0))
FAMILY = {"noise": "features"}  # the control reads random bytes with (c)'s settings
ALL_FAULT_CELLS = (
    ("alarms", 1),
    ("summary", 16),
    ("features", 16),
    ("oracle", 0),
    ("silent", 0),
)
# Step-1 centre: softmax regression, L2 1e-2, no temporal context
LINEAR_CENTRE = {
    "classifier": "logreg",
    "l2": 1e-2,
    "context": 1,
    "precision": "float64",
}
ABLATION_CELLS = (("features", 16), ("oracle", 0))
VARIANTS = ("tuned", "linear_k1", "mlp_k1")
VARIANT_LABELS = {
    "linear_k1": "softmax regression, no context (the step-1 centre)",
    "mlp_k1": "tuned MLP without temporal context",
    "tuned": "tuned MLP with the selected context",
}

_STATE: dict[str, Any] = {}


# ---------------------------------------------------------------------------- workers
def _init_worker(cache_dir: Optional[str], protocol: dict, smoke: bool = False) -> None:
    _STATE["stages"] = STAGES[:1] if smoke else STAGES
    _STATE["data"] = RiethData.from_cache(Path(cache_dir) if cache_dir else None)
    _STATE["protocol"] = protocol
    _STATE["replicate"] = None


def _protocol(fault_set: str) -> Protocol:
    base = dict(_STATE["protocol"])
    base["faults"] = FAULT_SETS[fault_set]
    return Protocol(**base)


def _replicate(fault_set: str, index: Optional[int]) -> Replicate:
    key = (fault_set, index)
    cached = _STATE.get("replicate")
    if cached is None or cached[0] != key:
        _STATE["replicate"] = None  # free the previous one before building the next
        _STATE["replicate"] = (
            key,
            Replicate(_STATE["data"], _protocol(fault_set), index),
        )
    return _STATE["replicate"][1]


def _tune_task(task: tuple[str, int]) -> dict[str, Any]:
    encoding, budget = task
    start = time.perf_counter()
    rep = _replicate("diagnosable", None)
    prep = rep.prepare(encoding, budget, seed=0)
    params, trace = tune_centre(rep, prep, stages=_STATE["stages"])
    return {
        "encoding": encoding,
        "budget": budget,
        "params": params,
        "trace": trace,
        "seconds": time.perf_counter() - start,
    }


def _family(encoding: str, variant: str) -> Optional[str]:
    """The tuning family a cell's centre settings come from (``None``: needs no tuning)."""
    if encoding == "silent" or variant == "linear_k1":
        return None
    return FAMILY.get(encoding, encoding)


def _params(encoding: str, variant: str, tuned: dict[str, Any]) -> dict[str, Any]:
    if variant == "linear_k1":
        return dict(LINEAR_CENTRE)
    if encoding == "silent":
        return {}
    params = dict(tuned[FAMILY.get(encoding, encoding)])
    return {**params, "context": 1} if variant == "mlp_k1" else params


def _cell_task(task: tuple[str, int, str, int, str, dict]) -> dict[str, Any]:
    fault_set, replicate, encoding, budget, variant, params = task
    start = time.perf_counter()
    rep = _replicate(fault_set, replicate)
    out = rep.evaluate(encoding, budget, replicate, params)
    out.update(
        fault_set=fault_set, variant=variant, seconds=time.perf_counter() - start
    )
    return out


def _cells(
    fault_sets: list[str], replicates: int, secondary: int
) -> list[tuple[str, int, str, int, str]]:
    """Every (fault set, replicate, encoding, budget, variant) cell, in replicate order.

    The main cells run on all ``replicates``; the secondary ones (see ``SECONDARY_NOISE``)
    on the first ``secondary``.
    """
    cells = []
    for fault_set in fault_sets:
        if fault_set == "diagnosable":
            for r in range(replicates):
                for enc in ROWS:
                    for budget in BUDGETS[enc]:
                        if (
                            enc == "noise"
                            and budget in SECONDARY_NOISE
                            and r >= secondary
                        ):
                            continue
                        cells.append((fault_set, r, enc, budget, "tuned"))
                if r < secondary:
                    for enc, budget in ABLATION_CELLS:
                        cells += [(fault_set, r, enc, budget, v) for v in VARIANTS[1:]]
        else:
            for r in range(min(replicates, secondary)):
                cells += [(fault_set, r, enc, b, "tuned") for enc, b in ALL_FAULT_CELLS]
    return cells


def _run_all(
    args: argparse.Namespace,
    protocol: dict,
    cache: Optional[str],
    fault_sets: list[str],
    tuning: dict[str, Any],
) -> tuple[dict[str, Any], list[dict]]:
    """Tune and evaluate on one pool: a family's cells start as soon as its tuning is done.

    The oracle's search is by far the longest, so it is submitted first and the cells that
    need no tuning (silent, the step-1 centre) fill the other workers meanwhile.
    """
    waiting: dict[Optional[str], list[tuple]] = {}
    for cell in _cells(fault_sets, args.replicates, args.secondary):
        waiting.setdefault(_family(cell[2], cell[4]), []).append(cell)
    total = sum(len(v) for v in waiting.values())
    rows: list[dict] = []
    start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(cache, protocol, args.smoke),
    ) as pool:
        pending: dict[Any, str] = {}

        def submit_cells(family: Optional[str], tuned: dict[str, Any]) -> None:
            for cell in waiting.pop(family, []):
                params = _params(cell[2], cell[4], tuned)
                pending[pool.submit(_cell_task, (*cell, params))] = "cell"

        if args.tuned_from is None:
            for task in sorted(TUNE_CELLS, key=lambda t: t[0] != "oracle"):
                pending[pool.submit(_tune_task, task)] = "tune"
            submit_cells(None, {})
        else:
            tuned = {f: i["params"] for f, i in tuning.items()}
            for family in list(waiting):
                submit_cells(family, tuned)
        print(f"{total} cells on {args.workers} workers", flush=True)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                kind = pending.pop(future)
                result = future.result()
                if kind == "tune":
                    family = result["encoding"]
                    tuning[family] = result
                    best = max(t["val_f1"] for t in result["trace"])
                    print(
                        f"  tuned {family}: {result['params']} val F1 {best:.3f} "
                        f"({result['seconds']:.0f}s)",
                        flush=True,
                    )
                    tuned = {f: i["params"] for f, i in tuning.items()}
                    submit_cells(family, tuned)
                else:
                    rows.append(result)
                    if len(rows) % 20 == 0 or len(rows) == total:
                        print(
                            f"  {len(rows)}/{total} cells, "
                            f"{time.perf_counter() - start:.0f}s",
                            flush=True,
                        )
    return tuning, rows


# ---------------------------------------------------------------------------- tables
def _fmt(values: list[float], digits: int = 3, scale: float = 1.0) -> str:
    arr = np.asarray(values, dtype=float) * scale
    if not np.isfinite(arr).any():
        return "never"
    mean, half = summarise(arr)
    return (
        f"{mean:.{digits}f} ± {half:.{digits}f}"
        if np.isfinite(half)
        else f"{mean:.{digits}f}"
    )


def _select(
    rows: list[dict], fault_set: str, enc: str, budget: int, variant: str = "tuned"
) -> list[dict]:
    """Rows of a cell, ordered by replicate; alarms repeat their 1 B cell at every budget."""
    if enc == "alarms" and budget >= 1:
        budget = 1
    if enc in ("silent", "oracle"):
        budget = 0
    out = [
        r
        for r in rows
        if r["fault_set"] == fault_set
        and r["encoding"] == enc
        and r["budget"] == budget
        and r["variant"] == variant
    ]
    return sorted(out, key=lambda r: r["replicate"])


def _values(rows, fault_set, enc, budget, metric, variant="tuned") -> list[float]:
    return [r[metric] for r in _select(rows, fault_set, enc, budget, variant)]


def _mean(rows, fault_set, enc, budget, metric, variant="tuned") -> float:
    return summarise(_values(rows, fault_set, enc, budget, metric, variant))[0]


def _table(
    rows: list[dict],
    fault_set: str,
    metric: str,
    digits: int,
    note: str,
    scale: float = 1.0,
) -> list[str]:
    lines = [
        f"| Encoding | {' | '.join(f'{b} B' for b in BUDGET_COLUMNS)} |",
        "|---|" + "---|" * len(BUDGET_COLUMNS),
    ]
    for enc in ROWS:
        cells = []
        for budget in BUDGET_COLUMNS:
            if enc in ("silent", "oracle"):
                have = budget == 0
            elif enc == "alarms":
                have = budget >= 1
            else:
                have = budget in BUDGETS[enc]
            vals = _values(rows, fault_set, enc, budget, metric) if have else []
            mark = "†" if 0 < len(vals) < _N_FULL else ""
            cells.append(_fmt(vals, digits, scale) + mark if vals else "-")
        lines.append(f"| {LABELS[enc]} | {' | '.join(cells)} |")
    return lines + ["", note, ""]


def _bar(rows: list[dict], fault_set: str, metric: str) -> dict[int, tuple[str, float]]:
    """Best hand-designed encoding (a, b, c) per byte budget by mean ``metric``."""
    best = {}
    for budget in (1, 2, 4, 8, 16, 51):
        scores = {
            e: _mean(rows, fault_set, e, budget, metric)
            for e in ("alarms", "summary", "features")
            if _values(rows, fault_set, e, budget, metric)
        }
        if scores:
            enc = max(scores, key=lambda e: scores[e])
            best[budget] = (enc, scores[enc])
    return best


def _paired(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mean and half-width of the replicate-paired difference ``a - b``."""
    return summarise(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def _oracle_check(
    rows: list[dict], fault_set: str, metric: str
) -> tuple[list[str], list[str]]:
    """Table lines and the cells where the oracle is significantly below an encoding."""
    lines = [
        "| Encoding | Budget | Oracle - encoding | Oracle at least as good? |",
        "|---|---|---|---|",
    ]
    failures = []
    oracle = {r["replicate"]: r[metric] for r in _select(rows, fault_set, "oracle", 0)}
    for enc in ("alarms", "summary", "features", "noise"):
        for budget in BUDGETS[enc] if enc != "alarms" else (1,):
            other = {
                r["replicate"]: r[metric] for r in _select(rows, fault_set, enc, budget)
            }
            common = sorted(set(oracle) & set(other))  # pair by replicate
            if len(common) < 2:
                continue
            mean, half = _paired(
                [oracle[k] for k in common], [other[k] for k in common]
            )
            ok = mean + half >= 0
            if not ok:
                failures.append(f"{enc} {budget} B")
            verdict = "yes" if mean >= 0 else ("yes, within the CI" if ok else "NO")
            lines.append(
                f"| {LABELS[enc]} | {budget} B | {mean:+.3f} ± {half:.3f} | {verdict} |"
            )
    return lines, failures


def _summary(rows: list[dict], args: argparse.Namespace, fault_set: str) -> list[str]:
    n = len(FAULT_SETS[fault_set])
    pc = f"{args.target_far:.0%}"
    lines = [f"## Summary ({n} faults)", ""]
    bar = _bar(rows, fault_set, "macro_f1")
    parts = []
    for budget, (enc, score) in bar.items():
        if fault_set == "diagnosable" and budget not in (4, 8, 16, 51):
            continue  # 1 and 2 B: see the tables; the bar lines stay short
        if fault_set == "all" and budget != 16:
            continue  # only 16 B was run with the 20-fault set
        far = _mean(rows, fault_set, enc, budget, "false_alarm_rate")
        delay = _mean(rows, fault_set, enc, budget, "detection_delay")
        parts.append(
            f"{budget} B: ({enc}) macro-F1 {score:.3f}, test false alarms {far:.1%}, delay {delay:.1f}"
        )
    lines.append(
        f"- **The bar for learned codes** (best hand design per budget, centre calibrated on "
        f"validation runs to {pc} false alarms): " + "; ".join(parts) + "."
    )
    oracle = _mean(rows, fault_set, "oracle", 0, "macro_f1")
    ofar = _mean(rows, fault_set, "oracle", 0, "false_alarm_rate")
    noise = _values(rows, fault_set, "noise", 16, "macro_f1")
    lines.append(
        f"- Oracle (all raw variables, no byte limit): macro-F1 {oracle:.3f}, test false alarms {ofar:.1%}; "
        f"silent 0.000"
        + (f"; random bytes {np.mean(noise):.3f} (16 B)." if noise else ".")
    )
    far_all = [
        _mean(rows, fault_set, e, b, "false_alarm_rate")
        for e in ("alarms", "summary", "features", "oracle", "noise")
        for b in BUDGETS[e]
        if _values(rows, fault_set, e, b, "false_alarm_rate")
    ]
    lines.append(
        f"- Realised test false-alarm rate against the {pc} target: from {min(far_all):.1%} to "
        f"{max(far_all):.1%} over the cells (mean {np.mean(far_all):.1%}); "
        "the table below splits it by source."
    )
    if fault_set == "diagnosable":
        gaps, room, none = [], [], []
        oracle = {
            r["replicate"]: r["macro_f1"] for r in _select(rows, fault_set, "oracle", 0)
        }
        for budget, (enc, _) in bar.items():
            best = {
                r["replicate"]: r["macro_f1"]
                for r in _select(rows, fault_set, enc, budget)
            }
            common = sorted(set(oracle) & set(best))
            if len(common) < 2:
                continue
            mean, half = _paired([oracle[k] for k in common], [best[k] for k in common])
            gaps.append(f"{budget} B {mean:+.3f} ± {half:.3f}")
            (room if mean - half > 0 else none).append(budget)
        lines.append(
            "- Headroom (oracle minus the best hand design, paired by replicate, macro-F1): "
            + ", ".join(gaps)
            + f". The best hand design is statistically indistinguishable from the oracle at "
            f"{', '.join(f'{b} B' for b in none) or 'no budget'}; there is measurable room for a "
            f"better code at {', '.join(f'{b} B' for b in room) or 'no budget'}."
        )
    for name, enc in (("(c)", "features"), ("(b)", "summary")):
        pairs = [
            (b, _mean(rows, fault_set, enc, b, "macro_f1"))
            for b in BUDGETS[enc]
            if _values(rows, fault_set, enc, b, "macro_f1")
        ]
        if len(pairs) > 1:
            lines.append(
                f"- Macro-F1 of {name} by budget: "
                + ", ".join(f"{b} B {v:.3f}" for b, v in pairs)
                + (
                    f"; the fixed one-byte alarm code (a): "
                    f"{_mean(rows, fault_set, 'alarms', 1, 'macro_f1'):.3f}."
                    if enc == "summary"
                    else "."
                )
            )
    return lines + [""]


def _fault_f1_table(
    rows: list[dict], fault_set: str, cells: list[tuple[str, int]]
) -> list[str]:
    faults = FAULT_SETS[fault_set]
    header = " | ".join(f"{e} {b} B" if e != "oracle" else "oracle" for e, b in cells)
    lines = [f"| Fault | {header} |", "|---|" + "---|" * len(cells)]
    for i, f in enumerate(faults):
        cols = []
        for enc, budget in cells:
            sel = _select(rows, fault_set, enc, budget)
            vals = [r["f1_per_class"][i] for r in sel]
            mark = "†" if 0 < len(vals) < _N_FULL else ""
            cols.append(_fmt(vals, 2) + mark if vals else "-")
        lines.append(f"| IDV({f}) | {' | '.join(cols)} |")
    return lines + [""]


def _tuning_text(tuning: dict[str, Any]) -> list[str]:
    lines = [
        "## Centre hyperparameters (chosen on validation runs only)",
        "",
        "A staged search on the tuning block of the run splits (its own training runs; validation "
        "runs = its validation and test slices, none of which any replicate uses). One setting at a "
        "time, in the order below, every other setting at its current best; the score is the macro-F1 "
        "on the validation windows at the candidate's own calibrated threshold. Candidates within "
        f"{TOLERANCE} of the best tie and the cheapest wins. Tuned once per encoding family at its "
        "reference budget (16 B; the alarm byte; the oracle) and reused at the other budgets; the "
        "random-bytes control uses (c)'s settings. Search space: "
        + "; ".join(f"{name} in {list(vals)}" for name, vals in STAGES)
        + ".",
        "",
        "| Family | Reference cell | Selected | Fits | Validation macro-F1 before -> after |",
        "|---|---|---|---|---|",
    ]
    for family, info in tuning.items():
        trace = info["trace"]
        first, best = trace[0]["val_f1"], max(t["val_f1"] for t in trace)
        sel = info["params"]
        picked = ", ".join(
            f"{k}={sel[k]}" for k in ("context", "hidden", "l2", "max_iter")
        )
        ref = (
            f"{info['encoding']} {info['budget']} B" if family != "oracle" else "oracle"
        )
        lines.append(
            f"| {family} | {ref} | {picked} | {len(trace)} | {first:.3f} -> {best:.3f} |"
        )
    lines += ["", "Full trace (stage, value, validation macro-F1):", ""]
    for family, info in tuning.items():
        lines.append(
            f"- {family}: "
            + "; ".join(
                f"{t['stage']}={t['value']} {t['val_f1']:.3f}" for t in info["trace"]
            )
        )
    return lines + [""]


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


def _report(saved: dict[str, Any], data: RiethData) -> str:
    global _N_FULL
    rows, proto, tuning = saved["rows"], saved["protocol"], saved["tuning"]
    args = argparse.Namespace(
        target_far=proto["target_far"],
        replicates=proto["n_replicates"],
        secondary=saved.get("secondary", proto["n_replicates"]),
        workers=saved["workers"],
    )
    _N_FULL = args.replicates
    src = data.source
    airtime = lorawan_airtime_s(51, 12)
    fault_sets = [
        f for f in ("diagnosable", "all") if any(r["fault_set"] == f for r in rows)
    ]
    n_fault = len(FAULT_SETS["diagnosable"])
    p = Protocol(**{**proto, "faults": FAULT_SETS["diagnosable"]})
    pc = f"{args.target_far:.0%}"
    lines = [
        "# Plant floor: diagnosis quality vs uplink bytes, run-level measuring stick (R8, step 2)",
        "",
        f"- Date: {datetime.date.today().isoformat()}; on top of base commit {saved['commit']}; "
        f"Python {sys.version.split()[0]}; {platform.processor() or platform.machine()}; "
        f"{args.workers} worker processes; {saved['seconds_wall'] / 60:.1f} min wall clock "
        f"({saved['cpu_seconds'] / 60:.0f} CPU-minutes of tuning and cell work)",
        f"- Data: {src.get('dataset', 'Rieth et al. 2017')} ({src.get('url', 'n/a')}, retrieved "
        f"{src.get('retrieved', 'n/a')}). 500 independent simulation runs for each of the fault-free "
        "condition and faults 1..20 in each of four files (fault-free/faulty x training/testing); "
        "training runs have 500 samples (25 h), testing runs 960 (48 h), one sample every 3 minutes, "
        "the same 52 variables as the Braatz data. Measured from the data: the first faulty sample "
        f"(0-based) is {src.get('first_faulty_sample_0based', {}).get('training', 20)} in the "
        f"training runs and {src.get('first_faulty_sample_0based', {}).get('testing', 160)} in the "
        "testing runs, i.e. the fault is introduced after 1 h and 8 h. Fault 21 of the Braatz set is "
        f"not in this data. Licence: {src.get('license', 'n/a')} The full download (1.4 GB) was used; "
        "no subset of the published runs. `python scripts/fetch_rieth.py` downloads it to a cache "
        "outside the repo, checks size, MD5 and SHA-256 of every file and converts it once to "
        "float32 `.npy` arrays; the conversion was cross-checked against pyreadr.",
        f"- Run-level replicates: **{args.replicates} independent replicates**, each with its own "
        f"disjoint simulation runs. Per replicate and fault: {p.train_runs} training runs (training "
        f"files), {p.val_runs} validation and {p.test_runs} test runs (testing files); fault-free: "
        f"{p.train_normal_runs} training, {p.val_normal_runs} validation, {p.test_normal_runs} test "
        f"runs. A replicate's test set is {n_fault * p.test_runs + p.test_normal_runs} runs; the "
        f"{args.replicates} replicates together use {args.replicates * p.test_runs} of the 500 testing "
        "runs of every fault. A separate tuning block (also disjoint) is used for hyperparameters. "
        "Every number is mean ± 95% t-CI over replicates (`evaluation.stats.mean_ci`), so the CIs "
        "cover the draw of training, calibration and test runs as well as classifier initialisation. "
        f"A dagger (†) marks a cell measured on the first {args.secondary} replicates only, to keep "
        "the run short: the random-bytes control at 8 and 51 B, the centre ablation and the 20-fault set.",
        f"- Protocol: window {p.window} samples ({p.window * SAMPLE_MINUTES} min), one uplink and one "
        f"diagnosis every {p.stride} samples ({p.stride * SAMPLE_MINUTES} min); test runs have the "
        "fault from sample 160 (95 windows per run, 15 of them before the onset); the centre never "
        "sees the clock or the onset. Window label = the run's fault once the window contains a "
        "faulty sample (>= 1 faulty sample), else normal. Training windows use the same stride.",
        f"- Centre: an MLP (one tanh hidden layer, class-balanced, L2, L-BFGS) on the decoded messages, "
        "optionally stacking the last few uplinks; hyperparameters in the tuning section. "
        f"Calibration: the normal margin `tau` is the {1 - args.target_far:.0%} quantile of the margin on "
        f"the validation normal windows (validation fault-free runs plus the pre-onset windows of "
        f"validation fault runs, the same mix as the test normal windows), so {pc} of them alarm; "
        "the same target for every encoding, nothing of the validation set is trained on.",
        f"- LoRaWAN EU868 reference: 51 B at SF12 is {airtime:.2f} s on air, so a 1% duty cycle allows "
        f"one uplink per {min_uplink_period_s(51, 12):.0f} s per node; our uplinks are "
        f"{p.stride * SAMPLE_MINUTES * 60} s apart, so every budget here is legal at SF12.",
        "",
        "## Variable mapping (52 TEP variables to 4 nodes)",
        "",
        "Each node owns the sensors and actuators of one unit operation or service; analysers are "
        "attached to the unit that the sampled stream leaves.",
        "",
    ]
    mapping = describe_mapping()
    for node in NODES:
        lines.append(
            f"- **{node}** ({len(NODE_VARIABLES[node])} variables): "
            + "; ".join(mapping[node])
        )
    lines.append("")
    for fault_set in fault_sets:
        lines += _summary(rows, args, fault_set)
    # Oracle sanity
    lines += [
        "## Oracle sanity check",
        "",
        "The oracle reads unquantised statistics of all 52 variables plus every PCA and band feature "
        "(c) can send, with no byte limit, and uses the same centre family and the same tuning effort "
        "as the encodings, so it should do at least as well as every encoding. Paired by replicate "
        "(same runs), oracle minus encoding; 'within the CI' means the mean is below zero but the 95% "
        "interval reaches it.",
        "",
    ]
    for metric, title in (
        ("macro_f1", f"Macro-F1 at the calibrated {pc} threshold"),
        (
            "macro_f1_matched",
            f"Macro-F1 re-thresholded to exactly {pc} test false alarms",
        ),
    ):
        table, failures = _oracle_check(rows, "diagnosable", metric)
        lines += [f"**{title} ({n_fault} faults)**", ""] + table + [""]
        lines.append(
            "Result: "
            + (
                "the oracle is significantly below " + ", ".join(failures) + "."
                if failures
                else "the oracle is at least as good as every encoding within the CIs."
            )
        )
        lines.append("")
    titles = {
        "diagnosable": f"{n_fault} diagnosable faults (faults 3, 9 and 15 left out: no data-driven method separates them from normal operation, see the 20-fault section)",
        "all": "all 20 faults",
    }
    for fault_set in fault_sets:
        n = len(FAULT_SETS[fault_set])
        lines += [f"## Results: {titles[fault_set]}", ""]
        lines += [
            f"### Macro-F1 over the fault classes at the calibrated {pc} threshold (higher is better)",
            "",
        ]
        lines += _table(
            rows,
            fault_set,
            "macro_f1",
            3,
            f"Mean ± 95% CI over {args.replicates} replicates. F1 of each of the {n} fault classes from "
            "the confusion of all test windows (normal windows mistaken for a fault count against that "
            "class), then averaged. Random-guess level is about 0.05. The alarm byte fits every budget, "
            "so (a) is one measurement shown in each column.",
        )
        lines += [
            "### False-alarm rate on the test normal windows (target " + pc + ")",
            "",
        ]
        lines += _table(
            rows,
            fault_set,
            "false_alarm_rate",
            3,
            f"Realised share of test normal windows (fault-free runs plus the pre-onset part of every "
            f"fault run) that raise an alarm, with `tau` calibrated on validation runs to {pc}.",
        )
        lines += [
            "### Detection delay in samples (lower is better; 1 sample = 3 min)",
            "",
        ]
        lines += _table(
            rows,
            fault_set,
            "detection_delay",
            1,
            "Samples from the fault onset to the end of the first window in which the centre raises "
            "any alarm, averaged over the detected fault runs (the minimum is 10, one stride).",
        )
        lines += ["### Detection rate (higher is better)", ""]
        lines += _table(
            rows,
            fault_set,
            "detection_rate",
            3,
            "Share of the fault runs in which the centre ever alarmed after the onset.",
        )
        lines += [
            f"### The same decisions re-thresholded to exactly {pc} false alarms on the test normal windows",
            "",
            "Each centre's threshold is moved so that exactly "
            f"{pc} of its test normal windows alarm. This looks at test labels, so it is not a "
            "deployable setting; it removes the residual calibration error so that encodings compare at "
            "one operating point. The silent row is unchanged.",
            "",
            "**Macro-F1 (higher is better)**",
            "",
        ]
        lines += _table(
            rows,
            fault_set,
            "macro_f1_matched",
            3,
            f"Mean ± 95% CI over {args.replicates} replicates.",
        )
        lines += ["**Detection delay in samples**", ""]
        lines += _table(
            rows,
            fault_set,
            "detection_delay_matched",
            1,
            "Mean over the detected fault runs.",
        )
        if fault_set == "diagnosable":
            lines += ["### Where the realised false alarms come from (tuned cells)", ""]
            lines += [
                "| Cell | Validation (at `tau`) | Test, all normal windows | Test, fault-free runs | Test, pre-onset windows |",
                "|---|---|---|---|---|",
            ]
            for enc, budget in (
                ("alarms", 1),
                ("summary", 16),
                ("features", 16),
                ("noise", 16),
                ("oracle", 0),
            ):
                cols = [
                    _fmt(_values(rows, fault_set, enc, budget, m), 3)
                    for m in (
                        "val_far",
                        "false_alarm_rate",
                        "far_fault_free",
                        "far_pre_onset",
                    )
                ]
                name = LABELS[enc].split(" (")[0] + (
                    f" {budget} B" if enc != "oracle" else ""
                )
                lines.append(f"| {name} | {' | '.join(cols)} |")
            lines += [
                "",
                "Validation runs are fault-free runs (960 samples) plus the first 160 samples of fault "
                "runs, exactly like the test normal windows, so a gap between the columns is the "
                "run-to-run difference between independent draws of runs.",
                "",
            ]
            lines += ["### Mean bytes actually used per node message", ""]
            used = [
                "| Encoding | " + " | ".join(f"{b} B" for b in BUDGET_COLUMNS) + " |",
                "|---|" + "---|" * len(BUDGET_COLUMNS),
            ]
            for enc in ROWS:
                cells = []
                for budget in BUDGET_COLUMNS:
                    ok = (
                        (budget == 0)
                        if enc in ("silent", "oracle")
                        else (budget in BUDGETS[enc])
                    )
                    vals = (
                        _values(rows, fault_set, enc, budget, "bytes_used")
                        if ok
                        else []
                    )
                    cells.append(f"{np.mean(vals):.1f}" if vals else "-")
                used.append(f"| {LABELS[enc]} | {' | '.join(cells)} |")
            lines += used + [""]
    # Centre ablation
    lines += [
        "## What the stronger centre buys (centre ablation, 16 B for (c))",
        "",
        f"Macro-F1 at the calibrated {pc} threshold with the same encoders, replicates and calibration; "
        "only the centre changes.",
        "",
        "| Centre | (c) features 16 B | (e) oracle |",
        "|---|---|---|",
    ]
    for variant in VARIANTS:
        cols = [
            _fmt(_values(rows, "diagnosable", enc, budget, "macro_f1", variant), 3)
            for enc, budget in ABLATION_CELLS
        ]
        lines.append(f"| {VARIANT_LABELS[variant]} | {' | '.join(cols)} |")
    lines.append("")
    # Per-fault
    lines += ["## Per-fault F1 (calibrated threshold, mean ± CI over replicates)", ""]
    lines += _fault_f1_table(
        rows,
        "diagnosable",
        [("alarms", 1), ("summary", 16), ("features", 16), ("oracle", 0)],
    )
    if "all" in fault_sets:
        lines += ["### All 20 faults: the faults that are not diagnosable", ""]
        lines += _fault_f1_table(rows, "all", [("features", 16), ("oracle", 0)])
    lines += _tuning_text(tuning)
    lines += [
        "## Caveats",
        "",
        "- All runs are simulations of one plant model with one sampling interval; the replicates differ "
        "in simulator random seeds only. The testing files hold 500 runs per condition, of which this "
        f"report uses {args.replicates * (p.val_runs + p.test_runs)} per fault plus the tuning block.",
        "- Windows within a run are strongly autocorrelated; replicates (not windows) are the unit of "
        "the CIs, which is why they are wide relative to the number of windows.",
        "- Hyperparameters were tuned per encoding family at one budget and reused; (a)-(c) and the "
        "oracle got the same search, but a per-budget search might move the 8 B and 51 B cells slightly.",
        "- The windows' labels follow the step-1 rule (a window is faulty once it contains one faulty "
        "sample), so the first windows after an onset are nearly unlabellable and cap macro-F1 below 1.",
        "- Fault classes 3, 9 and 15 are not separable from normal operation here either; the 17-fault "
        "tables leave them out, the 20-fault tables show what they cost. Fault 21 of step 1 does not exist "
        "in this data, so numbers are not comparable one to one with step 1 (different data, fault set, "
        "centre and calibration).",
        "- The alarm design is a fixed 1 B code, so its row does not change with the budget; the 0 B "
        "column of an encoding means no message, the same as (d).",
        "",
        "## Fault list",
        "",
    ]
    for f, text in FAULT_DESCRIPTIONS.items():
        if f <= 20:
            lines.append(f"- IDV({f}): {text}")
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/fetch_rieth.py",
        f"python scripts/plant_baselines_rieth.py --replicates {args.replicates} "
        f"--secondary {args.secondary} --workers {args.workers} "
        "--out docs/results/plant_bytes_baselines_rieth.md",
        "# re-render the report from the saved per-cell results, no computation:",
        "python scripts/plant_baselines_rieth.py --from-json docs/results/plant_bytes_baselines_rieth.json "
        "--out docs/results/plant_bytes_baselines_rieth.md",
        "```",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--replicates", type=int, default=10, help="Independent replicates"
    )
    parser.add_argument("--workers", type=int, default=4, help="Processes (max 4)")
    parser.add_argument(
        "--fault-set", choices=("diagnosable", "all", "both"), default="both"
    )
    parser.add_argument(
        "--secondary",
        type=int,
        default=5,
        help="Replicates for the secondary cells (control at 8/51 B, ablation, 20 faults)",
    )
    parser.add_argument("--train-runs", type=int, default=20)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--cache-dir", type=Path, default=None, help="TEP cache root")
    parser.add_argument("--out", type=Path, default=None, help="Markdown report path")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tune only the context length (a quick pipeline check)",
    )
    parser.add_argument(
        "--tuned-from",
        type=Path,
        default=None,
        help="Reuse the tuning stored in a previous run's JSON instead of searching again",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Re-render the report from a saved run's JSON instead of computing",
    )
    args = parser.parse_args()
    args.workers = max(1, min(args.workers, 4))
    data = RiethData.from_cache(args.cache_dir)  # fail fast with a clear message
    fault_sets = (
        ["diagnosable", "all"] if args.fault_set == "both" else [args.fault_set]
    )

    if args.from_json is not None:
        saved = json.loads(args.from_json.read_text(encoding="utf-8"))
        report = _report(saved, data)
        if args.out is not None:
            args.out.write_text(report, encoding="utf-8")
        print(f"rendered {args.out}")
        return

    protocol = dataclasses.asdict(
        Protocol(
            n_replicates=args.replicates,
            train_runs=args.train_runs,
            target_far=args.target_far,
        )
    )
    protocol.pop("faults")
    cache = str(args.cache_dir) if args.cache_dir else None

    tuning: dict[str, Any] = {}
    if args.tuned_from is not None:
        tuning = json.loads(args.tuned_from.read_text(encoding="utf-8"))["tuning"]
        print(f"reusing the tuning of {args.tuned_from}", flush=True)
    start = time.perf_counter()
    tuning, rows = _run_all(args, protocol, cache, fault_sets, tuning)
    seconds = time.perf_counter() - start
    tune_cpu = 0.0 if args.tuned_from else sum(i["seconds"] for i in tuning.values())
    saved = {
        "protocol": protocol,
        "workers": args.workers,
        "secondary": args.secondary,
        "commit": _commit(),
        "seconds_wall": seconds,
        "cpu_seconds": float(sum(r["seconds"] for r in rows) + tune_cpu),
        "tuning": tuning,
        "rows": rows,
    }
    report = _report(saved, data)
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        args.out.with_suffix(".json").write_text(json.dumps(saved), encoding="utf-8")
        print(f"wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
