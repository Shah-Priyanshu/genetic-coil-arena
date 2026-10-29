"""R14 step 2: train and evaluate learned per-role swarm brains (ES).

Thin wrapper around ``genetic_coil_arena.experiments.r14_brains``; see that module.

    python scripts/r14_brains.py train --runs 0:4 [--generations 1500] [--workers 4]
    python scripts/r14_brains.py eval --from runs:r14b-train-* --out docs/results/r14_brains.md
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

from genetic_coil_arena.experiments import r14_brains as rb  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import resolve_sources  # noqa: E402


def _train(job):
    run, settings, out = job
    res = rb.train_run(run, settings)
    stem = rb.save_run(res, Path(out))
    print(
        f"[train] run {run}: best val {res['best_val']:.3f} at gen "
        f"{res['best_generation']} ({res['seconds'] / 3600:.2f} h) -> {stem}",
        flush=True,
    )
    return run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--runs", default="0:4")
    t.add_argument("--generations", type=int, default=rb.BrainSettings.generations)
    t.add_argument("--population", type=int, default=rb.BrainSettings.population)
    t.add_argument("--episodes", type=int, default=rb.BrainSettings.episodes)
    t.add_argument("--progress", type=float, default=rb.BrainSettings.progress_s)
    t.add_argument("--randomise", action="store_true",
                   help="step 3: a random condition every generation")  # fmt: skip
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--out", default=os.environ.get("GCA_RUN_DIR", "outputs/r14b"))
    e = sub.add_parser("eval")
    e.add_argument("--from", dest="sources", nargs="+", required=True)
    e.add_argument("--reps", type=int, default=10)
    e.add_argument("--per-rep", type=int, default=300)
    e.add_argument("--out", type=Path, default=Path("docs/results/r14_brains.md"))
    args = p.parse_args()
    if args.cmd == "train":
        a, b = (int(x) for x in args.runs.split(":"))
        settings = replace(
            rb.BrainSettings(),
            generations=args.generations,
            population=args.population,
            episodes=args.episodes,
            progress_s=args.progress,
            randomise=args.randomise,
        )
        jobs = [(r, settings, args.out) for r in range(a, b)]
        with ProcessPoolExecutor(min(args.workers, len(jobs))) as pool:
            list(pool.map(_train, jobs))
        return
    runs = rb.load_runs(resolve_sources(args.sources))
    if not runs:
        raise SystemExit("no trained runs found")
    rep = rb.evaluate(runs, args.reps, args.per_rep)
    md = rb.render(rep, runs)
    js = json.dumps(rep, indent=1, default=float)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(js, encoding="utf-8")
    run_dir = os.environ.get("GCA_RUN_DIR")
    if run_dir:
        (Path(run_dir) / args.out.name).write_text(md, encoding="utf-8")
        (Path(run_dir) / args.out.with_suffix(".json").name).write_text(
            js, encoding="utf-8"
        )
    print(md, flush=True)


if __name__ == "__main__":
    main()
