"""Diagnosis quality vs uplink bytes for the hand-designed plant encodings (R8, step 1).

Replays the Tennessee Eastman test runs through ``PlantEnv`` for each encoding and byte
budget, with a centre classifier trained on the decoded messages, over several seeds
(each a different 80% block subsample of the training split), and writes a Markdown
report plus the per-seed JSON. Needs the data: run ``python scripts/fetch_tep.py`` once.

    python scripts/plant_baselines.py --seeds 8 --workers 4 \
        --out docs/results/plant_bytes_baselines.md
"""

import os

# At most a few BLAS threads per process: the machine is shared with long experiments
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

import numpy as np  # noqa: E402

from genetic_coil_arena.envs.plant.baselines import (  # noqa: E402
    ENCODINGS,
    FAULT_SETS,
    PlantContext,
    plant_system,
    plant_task,
    task_name,
)
from genetic_coil_arena.envs.plant.budget import (  # noqa: E402
    lorawan_airtime_s,
    min_uplink_period_s,
)
from genetic_coil_arena.envs.plant.tep import (  # noqa: E402
    FAULT_DESCRIPTIONS,
    NODE_VARIABLES,
    NODES,
    SAMPLE_MINUTES,
    TepData,
    describe_mapping,
)
from genetic_coil_arena.evaluation import (  # noqa: E402
    BaselineRegistry,
    format_table,
    run_experiment,
)
from genetic_coil_arena.evaluation.stats import mean_ci  # noqa: E402

BUDGET_COLUMNS = (0, 1, 8, 16, 51)  # 1 B is an extra column: the alarm byte's regime
LABELS = {
    "alarms": "(a) threshold alarms (1 B alarm byte per node)",
    "summary": "(b) quantised summary stats (mean, std, max, min)",
    "features": "(c) quantised hand features (PCA SPE/T2/scores, FFT bands)",
    "silent": "(d) silent (0 B, always reports normal)",
    "oracle": "(e) oracle (all raw variables, no byte limit)",
    "noise": "control: random bytes",
}
_CONTEXTS: dict[str, PlantContext] = {}
_SETTINGS: dict[str, Any] = {}


def _init_worker(cache_dir: Optional[str], settings: dict[str, Any]) -> None:
    _SETTINGS.update(settings)
    _SETTINGS["cache_dir"] = cache_dir


def _context(fault_set: str) -> PlantContext:
    if fault_set not in _CONTEXTS:
        cache = _SETTINGS["cache_dir"]
        data = TepData.from_cache(Path(cache) if cache else None)
        _CONTEXTS[fault_set] = PlantContext(
            data,
            FAULT_SETS[fault_set],
            window=_SETTINGS["window"],
            stride=_SETTINGS["stride"],
            train_stride=_SETTINGS["train_stride"],
            target_far=_SETTINGS["target_far"],
            l2=_SETTINGS["l2"],
            classifier=_SETTINGS["classifier"],
        )
    return _CONTEXTS[fault_set]


def _run_cell(task: tuple[str, str, int, int]) -> dict[str, Any]:
    fault_set, encoding, budget, seed = task
    start = time.perf_counter()
    out = _context(fault_set).evaluate(encoding, budget, seed)
    out["fault_set"] = fault_set
    out["seconds"] = time.perf_counter() - start
    out["f1_per_class"] = [float(x) for x in out["f1_per_class"]]
    return out


def _cells(fault_sets: list[str], seeds: int) -> list[tuple[str, str, int, int]]:
    cells = []
    for fault_set in fault_sets:
        for encoding in ENCODINGS:
            budgets = (0,) if encoding in ("silent", "oracle") else BUDGET_COLUMNS[1:]
            for budget in budgets:
                for seed in range(seeds):
                    cells.append((fault_set, encoding, budget, seed))
    return cells


def _fmt(values: list[float], digits: int = 3, scale: float = 1.0) -> str:
    arr = np.asarray(values, dtype=float) * scale
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "never"
    if arr.size < 2:
        return f"{arr[0]:.{digits}f}"
    mean, half = mean_ci(arr)
    return f"{mean:.{digits}f} ± {half:.{digits}f}"


def _table(
    rows: list[dict], fault_set: str, metric: str, digits: int, note: str
) -> list[str]:
    lines = [
        f"| Encoding | {' | '.join(f'{b} B' for b in BUDGET_COLUMNS)} |",
        "|---|" + "---|" * len(BUDGET_COLUMNS),
    ]
    for enc in ENCODINGS:
        cells = []
        for budget in BUDGET_COLUMNS:
            match = [
                r
                for r in rows
                if r["fault_set"] == fault_set
                and r["encoding"] == enc
                and (
                    r["budget"] == budget
                    or (enc in ("silent", "oracle") and budget == 0)
                )
            ]
            cells.append(_fmt([r[metric] for r in match], digits) if match else "-")
        lines.append(f"| {LABELS[enc]} | {' | '.join(cells)} |")
    return lines + ["", note, ""]


HARNESS_BUDGET = 16
HARNESS_SYSTEMS = ("alarms", "summary", "features")


def _harness_text(args: argparse.Namespace, rows: list[dict]) -> str:
    """Run the CommArena harness (baselines, Mann-Whitney, message ablations) at 16 B.

    Uses the 18-fault set. Also checks that the harness reproduces the per-cell
    macro-F1 of the pool run, so a silently failing evaluation would show up.
    """
    cache = args.cache_dir
    ctx = PlantContext(
        TepData.from_cache(cache),
        FAULT_SETS["diagnosable"],
        window=args.window,
        stride=args.stride,
        train_stride=args.train_stride,
        target_far=args.target_far,
        l2=args.l2,
        classifier=args.classifier,
    )
    registry = BaselineRegistry()
    name = task_name(ctx, HARNESS_BUDGET)
    registry.register(name, "silent", plant_system(ctx, "silent", 0))
    registry.register(name, "random_bytes", plant_system(ctx, "noise", HARNESS_BUDGET))
    task = plant_task(ctx, HARNESS_BUDGET, n_seeds=args.seeds)
    systems = {enc: plant_system(ctx, enc, HARNESS_BUDGET) for enc in HARNESS_SYSTEMS}
    result = run_experiment(
        task, systems, seeds=list(range(args.seeds)), registry=registry
    )
    checks = []
    for enc in HARNESS_SYSTEMS:
        pooled = [
            r["macro_f1"]
            for r in rows
            if r["fault_set"] == "diagnosable"
            and r["encoding"] == enc
            and r["budget"] == HARNESS_BUDGET
        ]
        harness_mean = result.condition(enc).mean
        checks.append(abs(float(np.mean(pooled)) - harness_mean) < 1e-9)
    lines = [
        f"## CommArena harness cross-check ({HARNESS_BUDGET} B, 18 faults)",
        "",
        "The same cells through `evaluation.run_experiment`: each system against the mandatory "
        "baselines (silent centre, random-bytes control) with Mann-Whitney tests (Holm-corrected), "
        "then the message ablations on the nodes' uplink (`zero`, `shuffle_batch`, "
        "`swap_receivers`, `random_matched`; paired Wilcoxon, `listening` = the performance "
        "drop's CI lies above zero). Score = macro-F1. Mean macro-F1 here equals the pool run's "
        f"for every system: {'yes' if all(checks) else 'NO - investigate'}.",
        "",
        "```",
        format_table(result),
        "```",
    ]
    return "\n".join(lines)


def _cpu_name() -> str:
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


def _mean(
    rows: list[dict], fault_set: str, enc: str, budget: int, metric: str
) -> float:
    vals = [
        r[metric]
        for r in rows
        if r["fault_set"] == fault_set
        and r["encoding"] == enc
        and (r["budget"] == budget or (enc in ("silent", "oracle") and budget == 0))
    ]
    arr = np.asarray(vals, dtype=float)
    return (
        float(np.nanmean(arr)) if arr.size and np.isfinite(arr).any() else float("nan")
    )


def _summary(rows: list[dict], args: argparse.Namespace, fault_set: str) -> list[str]:
    """Bullet points computed from the numbers, so they cannot drift from the tables."""
    n = len(FAULT_SETS[fault_set])
    lines = [f"## Summary ({n} faults)", ""]
    pc = f"{args.target_far:.0%}"
    for metric, label in (
        ("macro_f1", "own calibrated threshold"),
        ("macro_f1_matched", f"matched {pc} test false alarms"),
    ):
        parts = []
        for budget in (8, 16, 51):
            scores = {
                e: _mean(rows, fault_set, e, budget, metric)
                for e in ("alarms", "summary", "features")
            }
            best = max(scores, key=lambda e: scores[e])
            parts.append(f"{budget} B: {best} {scores[best]:.2f}")
        oracle = _mean(rows, fault_set, "oracle", 0, metric)
        lines.append(
            f"- Best hand design by macro-F1 at {label} ({', '.join(parts)}); "
            f"oracle {oracle:.2f}; silent 0.00; random bytes "
            f"{_mean(rows, fault_set, 'noise', 16, metric):.2f}."
        )
    curve = ", ".join(
        f"{b} B {_mean(rows, fault_set, 'features', b, 'macro_f1_matched'):.2f}"
        for b in (1, 8, 16, 51)
    )
    lines.append(
        f"- Macro-F1 of (c) at matched false alarms by budget: {curve}; the fixed one-byte "
        f"alarm code (a): {_mean(rows, fault_set, 'alarms', 1, 'macro_f1_matched'):.2f}."
    )
    best = max(
        _mean(rows, fault_set, e, b, "macro_f1_matched")
        for e in ("alarms", "summary", "features")
        for b in (1, 8, 16, 51)
    )
    oracle = _mean(rows, fault_set, "oracle", 0, "macro_f1_matched")
    if oracle < best:
        lines.append(
            f"- The oracle ({oracle:.2f}) is *below* the best byte-limited design ({best:.2f}) "
            "at matched false alarms, so it is not an upper bound for this centre: a linear "
            "classifier fitted on a few hundred strongly autocorrelated windows per class does "
            "worse with all the fields than with a compact subset (a likely cause, not "
            "tested here). Bandwidth does not limit this centre beyond about 16 B; a "
            "stronger centre or more training data (Rieth et al.) may change that."
        )
    else:
        lines.append(
            f"- The oracle ({oracle:.2f}) is at or above the best byte-limited design ({best:.2f}) "
            "at matched false alarms."
        )
    lines.append("")
    return lines


def _report(
    rows: list[dict],
    args: argparse.Namespace,
    fault_sets: list[str],
    seconds: float,
    harness: str = "",
) -> str:
    data = TepData.from_cache(args.cache_dir)
    src = data.source
    airtime = lorawan_airtime_s(51, 12)
    lines = [
        "# Plant floor: diagnosis quality vs uplink bytes (R8, step 1)",
        "",
        f"- Date: {datetime.date.today().isoformat()}; on top of base commit {_commit()}; "
        f"Python {sys.version.split()[0]}; {_cpu_name()}; "
        f"{args.workers} worker processes, {seconds / 60:.1f} min wall clock",
        f"- Data: Tennessee Eastman Process, Braatz-group benchmark d00-d21 ({src.get('url', 'n/a')}, "
        f"retrieved {src.get('retrieved', 'n/a')}). Licence: {src.get('license', 'n/a')}. "
        "Files are not committed; `python scripts/fetch_tep.py` downloads them to a cache "
        "outside the repo and checks every file's SHA-256 and shape.",
        f"- Seeds: {args.seeds} (0..{args.seeds - 1}); every number is mean ± 95% t-CI over "
        "seeds (`evaluation.stats.mean_ci`). A seed draws a different 80% subsample of the "
        "training blocks and a different classifier initialisation / noise stream. "
        "**The CI does not cover run-to-run plant variability**: the held-out test is one "
        "recorded run per fault (the same for every seed), see Caveats.",
        f"- Protocol: window {args.window} samples ({args.window * SAMPLE_MINUTES} min), one uplink and one "
        f"diagnosis every {args.stride} samples ({args.stride * SAMPLE_MINUTES} min); test runs are 960 "
        "samples with the fault from sample 160 (95 windows per run); the centre never sees the "
        "clock or the onset. Window label = the run's fault once the window contains a faulty "
        "sample (>= 1 faulty sample), else normal.",
        f"- Centre: softmax regression (L2 {args.l2}, class-balanced) on the decoded messages, trained "
        f"on the training split at stride {args.train_stride}; normal margin calibrated by "
        f"time-blocked cross-validation on training data to {args.target_far:.0%} false alarms on "
        "held-out normal windows (same target for every encoding).",
        f"- LoRaWAN EU868 reference: 51 B at SF12 is {airtime:.2f} s on air, so a 1% duty cycle "
        f"allows one uplink per {min_uplink_period_s(51, 12):.0f} s per node; our uplinks are "
        f"{args.stride * SAMPLE_MINUTES * 60} s apart, so every budget here is legal at SF12. "
        "JSON of the same readings would not fit (a 4-variable JSON object is already tens of bytes).",
        "",
        "## Variable mapping (52 TEP variables to 4 nodes)",
        "",
        "Each node owns the sensors and actuators of one unit operation or service; "
        "analysers are attached to the unit that the sampled stream leaves.",
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

    titles = {
        "diagnosable": "18 diagnosable faults (faults 3, 9 and 15 left out: no method separates them from normal operation in this data)",
        "all": "all 21 faults",
    }
    for fault_set in fault_sets:
        n = len(FAULT_SETS[fault_set])
        lines += [f"## Results: {titles[fault_set]}", ""]
        lines += ["### Macro-F1 over the fault classes (higher is better)", ""]
        lines += _table(
            rows,
            fault_set,
            "macro_f1",
            3,
            f"Mean ± 95% CI over {args.seeds} seeds. F1 of each of the {n} fault classes from the "
            "confusion of all test windows (normal windows mistaken for a fault count against that "
            "class), then averaged. Random-guess level is about 0.05.",
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
            "any alarm, averaged over the fault runs that were detected (the minimum is 10, one "
            "stride). `never` = no run detected.",
        )
        lines += ["### Detection rate (higher is better)", ""]
        lines += _table(
            rows,
            fault_set,
            "detection_rate",
            2,
            "Share of the fault runs in which the centre ever alarmed after the onset.",
        )
        lines += ["### False-alarm rate on normal windows (lower is better)", ""]
        lines += _table(
            rows,
            fault_set,
            "false_alarm_rate",
            3,
            "Share of normal test windows (the normal run plus the pre-onset part of every fault "
            "run) that raise an alarm. The calibration target was "
            f"{args.target_far:.0%} on held-out training windows; the test value is higher because "
            "the test runs sit at different offsets from the single 500-sample training run.",
        )
        lines += [
            f"### Same decisions at a matched {args.target_far:.0%} false-alarm rate on the test normal windows",
            "",
            "Each centre's alarm threshold is moved so that exactly "
            f"{args.target_far:.0%} of the test normal windows alarm (the fault class it reports "
            "is unchanged). This looks at the test normal labels, so it is not a deployable "
            "setting; it puts every encoding at one operating point so that macro-F1 and delay "
            "compare information content rather than how well the training calibration "
            "transferred. The silent row is unchanged (it never alarms).",
            "",
            "**Macro-F1 at matched false alarms (higher is better)**",
            "",
        ]
        lines += _table(
            rows,
            fault_set,
            "macro_f1_matched",
            3,
            f"Mean ± 95% CI over {args.seeds} seeds.",
        )
        lines += ["**Detection delay in samples at matched false alarms**", ""]
        lines += _table(
            rows,
            fault_set,
            "detection_delay_matched",
            1,
            "Mean over the detected fault runs; see the detection rate below.",
        )
        lines += ["**Detection rate at matched false alarms**", ""]
        lines += _table(
            rows,
            fault_set,
            "detection_rate_matched",
            2,
            "Share of fault runs detected.",
        )
        lines += ["### Mean bytes actually used per node message", ""]
        used = [
            "| Encoding | " + " | ".join(f"{b} B" for b in BUDGET_COLUMNS) + " |",
            "|---|" + "---|" * len(BUDGET_COLUMNS),
        ]
        for enc in ENCODINGS:
            cells = []
            for budget in BUDGET_COLUMNS:
                match = [
                    r["bytes_used"]
                    for r in rows
                    if r["fault_set"] == fault_set
                    and r["encoding"] == enc
                    and (
                        r["budget"] == budget
                        or (enc in ("silent", "oracle") and budget == 0)
                    )
                ]
                cells.append(f"{np.mean(match):.1f}" if match else "-")
            used.append(f"| {LABELS[enc]} | {' | '.join(cells)} |")
        lines += used + [""]
    if harness:
        lines += [harness, ""]
    lines += [
        "## Caveats",
        "",
        "- One recorded test run per fault (and one 500-sample normal training run). The seeds "
        "vary training only, so the CIs understate the uncertainty about a new plant run; the "
        "500-run Rieth et al. 2017 data set (doi:10.7910/DVN/6C3JR1) would give run-level CIs.",
        "- Fault classes 3, 9 and 15 are indistinguishable from normal operation in this data; "
        "the all-21-fault tables show what they cost.",
        "- Slow variables carry large run-to-run offsets. The pre-onset means of stripper "
        "temperature, stripper steam flow and stripper steam valve range from -0.5 to +1.5 "
        "training standard deviations across the 21 fault test runs (about +0.4 to +0.5 in the "
        "normal test run), so a threshold calibrated on the single 500-sample training run does "
        "not transfer cleanly: realised false-alarm rates exceed the calibration target and "
        "differ a lot between encodings (macro-F1 and delay at each centre's own threshold "
        "trade against that rate; hence the matched-false-alarm tables).",
        "- The alarm design is a fixed 1 B code, so its row does not change with the budget; "
        "the 0 B column of an encoding means no message, the same as (d).",
        "- The oracle uses the same centre family (softmax regression) on unquantised statistics "
        "of every variable plus every PCA/band feature (c) can send. It holds all the information "
        "but is a bound on nothing: with so few independent training windows a wider input "
        "tends to over-fit, so it can score below the compact feature set (c) (see the "
        "Summary for whether it does). Compare encodings at the matched false-alarm rate, "
        "where the thresholds agree.",
        "- The classifier L2 strength was fixed at one value after a coarse look at "
        "{1e-3, 1e-2, 1e-1} that moved macro-F1 by under 0.03; it is not tuned per encoding.",
        "",
        "## Fault list",
        "",
    ]
    for f, text in FAULT_DESCRIPTIONS.items():
        lines.append(f"- IDV({f}): {text}")
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/fetch_tep.py",
        f"python scripts/plant_baselines.py --seeds {args.seeds} --workers {args.workers} "
        "--out docs/results/plant_bytes_baselines.md",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=8, help="Seeds per cell (>= 5)")
    parser.add_argument("--workers", type=int, default=4, help="Processes (max 4)")
    parser.add_argument(
        "--fault-set", choices=("diagnosable", "all", "both"), default="both"
    )
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--target-far", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-2)
    parser.add_argument("--classifier", choices=("logreg", "mlp"), default="logreg")
    parser.add_argument("--cache-dir", type=Path, default=None, help="TEP cache")
    parser.add_argument("--out", type=Path, default=None, help="Markdown report path")
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Re-render the report from a saved run's JSON instead of computing",
    )
    args = parser.parse_args()
    args.workers = max(1, min(args.workers, 4))
    fault_sets = (
        ["diagnosable", "all"] if args.fault_set == "both" else [args.fault_set]
    )
    TepData.from_cache(args.cache_dir)  # fail fast with a clear message if not fetched

    if args.from_json is not None:
        saved = json.loads(args.from_json.read_text(encoding="utf-8"))
        args.seeds = saved["seeds"]
        for key, value in saved["settings"].items():
            setattr(args, key, value)
        rows = saved["rows"]
        fault_sets = sorted(
            {r["fault_set"] for r in rows}, key=["diagnosable", "all"].index
        )
        seconds = float(saved.get("seconds", 0.0))
        args.workers = int(saved.get("workers", args.workers))
        report = _report(rows, args, fault_sets, seconds, saved.get("harness", ""))
        if args.out is not None:
            args.out.write_text(report, encoding="utf-8")
        print(f"rendered {args.out}")
        return

    settings = {
        "window": args.window,
        "stride": args.stride,
        "train_stride": args.train_stride,
        "target_far": args.target_far,
        "l2": args.l2,
        "classifier": args.classifier,
    }
    cells = _cells(fault_sets, args.seeds)
    print(f"{len(cells)} cells on {args.workers} workers", flush=True)
    start = time.perf_counter()
    cache = str(args.cache_dir) if args.cache_dir else None
    rows: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(cache, settings)
    ) as pool:
        for i, row in enumerate(pool.map(_run_cell, cells, chunksize=1), 1):
            rows.append(row)
            if i % 10 == 0 or i == len(cells):
                print(
                    f"  {i}/{len(cells)} cells, {time.perf_counter() - start:.0f}s",
                    flush=True,
                )
    seconds = time.perf_counter() - start
    print("running the harness cross-check", flush=True)
    harness = _harness_text(args, rows) if "diagnosable" in fault_sets else ""

    report = _report(rows, args, fault_sets, seconds, harness)
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        args.out.with_suffix(".json").write_text(
            json.dumps(
                {
                    "settings": settings,
                    "seeds": args.seeds,
                    "seconds": seconds,
                    "workers": args.workers,
                    "harness": harness,
                    "rows": rows,
                }
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
