"""G12 report: per-squad decoders read and write the learned language of the G7/G7c squads.

Thin wrapper around ``genetic_coil_arena.experiments.r7d_legibility``; see that module.
It needs only the trained runs of the G7 16-run report and G7c:

    python scripts/r7d_legibility.py --from-json runs:r7b3-train-* runs:r7b4-train-* \
        runs:r7b5-train-* runs:r7c-train-* --out docs/results/r7d_legibility.md
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from genetic_coil_arena.experiments import r7d_legibility as r7d  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import (  # noqa: E402
    env_workers,
    read_trained,
    resolve_sources,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from-json", nargs="+", required=True)
    p.add_argument("--modes", nargs="+", default=list(r7d.MODES), choices=r7d.MODES)
    p.add_argument(
        "--fit-per-goal", type=int, default=r7d.LegibilitySettings.fit_per_goal
    )
    p.add_argument(
        "--test-per-goal", type=int, default=r7d.LegibilitySettings.test_per_goal
    )
    p.add_argument(
        "--random-writers", type=int, default=r7d.LegibilitySettings.n_random_writers
    )
    p.add_argument("--workers", type=int, default=env_workers(4))
    p.add_argument("--out", type=Path, default=Path("docs/results/r7d_legibility.md"))
    args = p.parse_args()
    trained, settings_d = read_trained(resolve_sources(args.from_json))
    trained = [t for t in trained if t["mode"] in args.modes]
    ls = r7d.LegibilitySettings(
        fit_per_goal=args.fit_per_goal,
        test_per_goal=args.test_per_goal,
        n_random_writers=args.random_writers,
    )
    print(f"[g12] {len(trained)} trained runs, {args.workers} workers", flush=True)
    jobs = [(t, settings_d, ls) for t in trained]
    rows, start = [], time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(r7d.evaluate_run, jobs):
            rows.append(row)
            print(
                f"[g12] {row['mode']} run {row['run']}: eaters exact "
                f"{row['read']['eaters_exact']:.3f}, efficacy "
                f"{row['write']['efficacy']:.2f} ({len(rows)}/{len(jobs)}, "
                f"{time.time() - start:.0f} s)",
                flush=True,
            )
    rep = r7d.build_report(rows, settings_d, ls)
    md, js = r7d.render(rep), json.dumps(rep, indent=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(js, encoding="utf-8")
    run_dir = os.environ.get("GCA_RUN_DIR")
    if run_dir:  # copies into the job's artifacts
        art = Path(run_dir)
        (art / args.out.name).write_text(md, encoding="utf-8")
        (art / args.out.with_suffix(".json").name).write_text(js, encoding="utf-8")
    print(f"G12 verdict: {rep['g12']['verdict']}; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
