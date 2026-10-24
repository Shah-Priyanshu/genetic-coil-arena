"""Learned codes against the best hand design on the plant floor (R8 step 3, goal G2).

Trains learned node encoders (end-to-end with a discrete bottleneck, residual k-means, local
classifier) at 1-8 B per uplink on the Rieth et al. runs, compares each with step 2's best hand
design per replicate, and applies the G2 rule. Needs the data: ``python scripts/fetch_rieth.py``.

    # one slice of the experiment; writes partial JSON into $GCA_RUN_DIR (or --partial-dir)
    python scripts/plant_learned_codes.py --methods e2e --budgets 1,2 --workers 8

    # merge the partials into docs/results/plant_learned_codes.{md,json}
    python scripts/plant_learned_codes.py --from-json runs:r8s3-* --out docs/results/plant_learned_codes.md

``--from-json`` takes files, folders and glob patterns; ``runs:<pattern>`` means the
``artifacts`` folders of the sibling jobs ``<pattern>`` next to ``$GCA_RUN_DIR`` (how the
runner's aggregation job finds its inputs). ``GCA_WORKERS`` overrides ``--workers``.
"""

import os

# One BLAS thread per process; torch's own threads are set per worker (--torch-threads)
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

from genetic_coil_arena.experiments.plant_learned_codes import (  # noqa: E402
    build_report,
    default_protocol,
    env_workers,
    g2_tests,
    merge_partials,
    merged_verdict,
    normalise_baseline,
    read_partials,
    run_groups,
    torch_threads_for,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs" / "results" / "plant_bytes_baselines_rieth.json"
PARTIAL_KIND = "plant_learned_codes_partial"


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def resolve_sources(tokens: list[str]) -> list[Path]:
    """Files, folders, globs and ``runs:<pattern>`` tokens -> existing paths."""
    out: list[Path] = []
    for tok in tokens:
        if tok.startswith("runs:"):
            run_dir = os.environ.get("GCA_RUN_DIR")
            if not run_dir:
                raise SystemExit("'runs:' needs GCA_RUN_DIR (it is set by the runner)")
            runs = Path(run_dir).resolve().parents[0]  # .../runs/<this job>
            for job in sorted(runs.parent.glob(tok[len("runs:") :])):
                out.append(job / "artifacts")
        elif any(c in tok for c in "*?["):
            out += [Path(p) for p in sorted(glob.glob(tok))]
        else:
            out.append(Path(tok))
    return [p for p in out if p.exists()]


def render(
    merged: dict[str, Any], baseline: dict, out: Optional[Path], name: str
) -> None:
    report = build_report(merged, baseline, name)
    base_rows = normalise_baseline(baseline)
    tests = g2_tests(merged["rows"], base_rows)
    verdict, why = merged_verdict(merged, tests)
    print(report)
    print(f"G2 verdict: {verdict}. {why}")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        saved = {**merged, "g2_tests": tests, "verdict": verdict, "why": why}
        out.with_suffix(".json").write_text(json.dumps(saved), encoding="utf-8")
        print(f"wrote {out} and {out.with_suffix('.json')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--methods", default="e2e,vq,local", help="e2e, vq, local")
    ap.add_argument("--budgets", default="1,2,4,8", help="Bytes per uplink")
    ap.add_argument("--replicates", type=int, default=None, help="Replicates (10)")
    ap.add_argument("--block", default=None, help="Run only replicates a:b of them")
    ap.add_argument("--seeds", type=int, default=1, help="Encoder seeds per replicate")
    ap.add_argument(
        "--workers", type=int, default=8, help="CPU budget (processes x threads)"
    )
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument(
        "--ablate-budgets", default="1,2,4", help="Budgets that also run ablations"
    )
    ap.add_argument("--out", type=Path, default=None, help="Markdown report path")
    ap.add_argument(
        "--from-json", nargs="+", default=None, help="Merge partials; no training"
    )
    ap.add_argument("--partial-dir", type=Path, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--cache-dir", type=Path, default=None, help="Rieth cache root")
    ap.add_argument(
        "--smoke", action="store_true", help="Reduced scale: a pipeline check"
    )
    args = ap.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    name = args.baseline.name

    if args.from_json is not None:
        partials = read_partials(resolve_sources(args.from_json))
        if not partials:
            raise SystemExit(f"no partial results found in {args.from_json}")
        merged = merge_partials(partials)
        print(f"merged {len(partials)} partial files, {len(merged['rows'])} rows")
        render(merged, baseline, args.out, name)
        return

    scale = "smoke" if args.smoke else "full"
    replicates = args.replicates or (2 if args.smoke else 10)
    lo, hi = (0, replicates)
    if args.block:
        lo, hi = (int(x) for x in args.block.split(":"))
    if not 0 <= lo < hi <= replicates:
        raise SystemExit(f"--block {lo}:{hi} is outside 0:{replicates}")
    methods = [m for m in args.methods.split(",") if m]
    budgets = _ints(args.budgets)
    workers = env_workers(args.workers)
    procs, threads = torch_threads_for(workers, args.torch_threads)
    run_dir = os.environ.get("GCA_RUN_DIR")
    pdir = args.partial_dir or (Path(run_dir) if run_dir else Path("partial"))
    pdir.mkdir(parents=True, exist_ok=True)
    protocol = default_protocol(replicates)
    commit = _commit()

    def partial_path(method: str, budget: int) -> Path:
        return pdir / f"plant_learned_{method}_b{budget}_r{lo}-{hi}.json"

    groups = []
    for method in methods:
        for budget in budgets:
            if not args.no_resume and partial_path(method, budget).exists():
                print(
                    f"skip {method} {budget} B: {partial_path(method, budget)} exists"
                )
                continue
            groups.append((method, budget))

    def save(group: dict[str, Any]) -> None:
        payload = {
            "kind": PARTIAL_KIND,
            "method": group["method"],
            "budget": group["budget"],
            "block": [lo, hi],
            "scale": scale,
            "protocol": protocol,
            "commit": commit,
            "seconds": group["seconds"],
            "tuning": group["tuning"],
            "rows": group["rows"],
        }
        path = partial_path(group["method"], group["budget"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
        print(f"wrote {path} ({group['seconds'] / 60:.1f} min)", flush=True)

    print(
        f"scale {scale}: {len(groups)} groups, replicates {lo}:{hi} of {replicates}, "
        f"{args.seeds} seed(s); {procs} processes x {threads} torch thread(s) "
        f"<= {workers} workers",
        flush=True,
    )
    start = time.perf_counter()
    if groups:
        run_groups(
            groups,
            list(range(lo, hi)),
            args.seeds,
            _ints(args.ablate_budgets),
            procs,
            protocol,
            scale,
            str(args.cache_dir) if args.cache_dir else None,
            threads,
            save,
            log=lambda s: print(s, flush=True),
        )
    print(f"done in {(time.perf_counter() - start) / 60:.1f} min")
    if args.out is not None:
        merged = merge_partials(read_partials([pdir]))
        render(merged, baseline, args.out, name)


if __name__ == "__main__":
    main()
