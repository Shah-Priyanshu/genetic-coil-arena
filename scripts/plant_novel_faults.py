"""Can learned codes express faults they never saw? (R8 step 4, goal G2b)

Trains codes on 12 seen faults of the Rieth et al. runs, freezes them, and measures detection,
separability and few-shot extension on 5 unseen faults, over fixed seen/unseen splits and 10
replicates. Needs the data: ``python scripts/fetch_rieth.py``.

    # one slice of the experiment; writes partial JSON into $GCA_RUN_DIR (or --partial-dir)
    python scripts/plant_novel_faults.py --codes e2e --budgets 1,2 --workers 8

    # merge the partials into docs/results/plant_novel_faults.{md,json}
    python scripts/plant_novel_faults.py --from-json runs:r8s4-* --out docs/results/plant_novel_faults.md

``--from-json`` takes files, folders, globs and ``runs:<pattern>`` (the ``artifacts`` folders of
the sibling jobs next to ``$GCA_RUN_DIR``). ``GCA_WORKERS`` overrides ``--workers``.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402

from plant_learned_codes import resolve_sources  # noqa: E402

from genetic_coil_arena.experiments.plant_learned_codes import (  # noqa: E402
    default_protocol,
    env_workers,
    torch_threads_for,
)
from genetic_coil_arena.experiments.plant_novel_faults import (  # noqa: E402
    CODES,
    N_SPLITS,
    PARTIAL_KIND,
    build_report,
    g2b_tests,
    g2b_verdict,
    make_splits,
    merge_partials,
    read_partials,
    run_groups,
)

ROOT = Path(__file__).resolve().parents[1]


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
    """Integers separated by ',' or '+' (so a --grid value can hold several budgets)."""
    return [int(x) for x in text.replace("+", ",").split(",") if x.strip()]


def render(merged: dict[str, Any], out: Optional[Path]) -> None:
    report = build_report(merged)
    tests = g2b_tests(merged["rows"])
    verdict, why = g2b_verdict(tests)
    print(report)
    print(f"G2b verdict: {verdict}. {why}")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        saved = {**merged, "g2b_tests": tests, "verdict": verdict, "why": why}
        out.with_suffix(".json").write_text(json.dumps(saved), encoding="utf-8")
        print(f"wrote {out} and {out.with_suffix('.json')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--codes", default=",".join(CODES), help=", ".join(CODES))
    ap.add_argument("--budgets", default="1,2,4,8", help="Bytes per uplink")
    ap.add_argument("--replicates", type=int, default=None, help="Replicates (10)")
    ap.add_argument("--block", default=None, help="Run only replicates a:b of them")
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None, help="Markdown report path")
    ap.add_argument("--from-json", nargs="+", default=None)
    ap.add_argument("--partial-dir", type=Path, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--cache-dir", type=Path, default=None, help="Rieth cache root")
    ap.add_argument(
        "--smoke", action="store_true", help="Reduced scale: a pipeline check"
    )
    args = ap.parse_args()

    if args.from_json is not None:
        sources = resolve_sources(args.from_json)
        partials = read_partials(sources)
        if not partials:
            raise SystemExit(
                f"no partial results in {args.from_json} (searched {sources})"
            )
        merged = merge_partials(partials)
        print(f"merged {len(partials)} partial files, {len(merged['rows'])} rows")
        render(merged, args.out)
        return

    scale = "smoke" if args.smoke else "full"
    replicates = args.replicates or (2 if args.smoke else 10)
    lo, hi = 0, replicates
    if args.block:
        lo, hi = (int(x) for x in args.block.split(":"))
    if not 0 <= lo < hi <= replicates:
        raise SystemExit(f"--block {lo}:{hi} is outside 0:{replicates}")
    codes = [c for c in args.codes.split(",") if c]
    bad = [c for c in codes if c not in CODES]
    if bad:
        raise SystemExit(f"unknown codes {bad}; expected {CODES}")
    budgets = _ints(args.budgets)
    workers = env_workers(args.workers)
    procs, threads = torch_threads_for(workers, args.torch_threads)
    run_dir = os.environ.get("GCA_RUN_DIR")
    pdir = args.partial_dir or (Path(run_dir) if run_dir else Path("partial"))
    pdir.mkdir(parents=True, exist_ok=True)
    protocol = default_protocol(replicates)
    splits = make_splits(n_splits=args.n_splits + 1)  # the last one tunes only
    commit = _commit()

    def partial_path(code: str, budget: int) -> Path:
        return pdir / f"plant_novel_{code}_b{budget}_r{lo}-{hi}.json"

    groups = []
    for code in codes:
        for budget in [0] if code == "oracle" else budgets:
            if not args.no_resume and partial_path(code, budget).exists():
                print(f"skip {code} {budget} B: {partial_path(code, budget)} exists")
                continue
            groups.append((code, budget))

    def save(group: dict[str, Any]) -> None:
        payload = {
            "kind": PARTIAL_KIND,
            "code": group["code"],
            "budget": group["budget"],
            "block": [lo, hi],
            "scale": scale,
            "protocol": protocol,
            "splits": splits,
            "commit": commit,
            "seconds": group["seconds"],
            "tuning": group["tuning"],
            "rows": group["rows"],
        }
        path = partial_path(group["code"], group["budget"])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)
        print(f"wrote {path} ({group['seconds'] / 60:.1f} min)", flush=True)

    print(
        f"scale {scale}: {len(groups)} groups, replicates {lo}:{hi} of {replicates}; "
        f"{procs} processes x {threads} torch thread(s) <= {workers} workers; "
        f"splits (unseen faults): {splits}",
        flush=True,
    )
    start = time.perf_counter()
    if groups:
        run_groups(
            groups,
            list(range(lo, hi)),
            procs,
            protocol,
            scale,
            str(args.cache_dir) if args.cache_dir else None,
            threads,
            splits,
            save,
            log=lambda s: print(s, flush=True),
        )
    print(f"done in {(time.perf_counter() - start) / 60:.1f} min")
    if args.out is not None:
        render(merge_partials(read_partials([pdir])), args.out)


if __name__ == "__main__":
    main()
