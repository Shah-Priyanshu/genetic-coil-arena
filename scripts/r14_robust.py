"""R14 step 3: learned swarm brains under worse conditions (and randomised training).

Thin wrapper around ``genetic_coil_arena.experiments.r14_robust``; see that module.

    python scripts/r14_robust.py --set step2=runs:r14b-train-* [--set random=runs:r14c-train-*]
        [--workers 8] [--out docs/results/r14_robust.md]
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from genetic_coil_arena.experiments import r14_brains as rb  # noqa: E402
from genetic_coil_arena.experiments import r14_robust as rr  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import resolve_sources  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--set", dest="sets", action="append", required=True,
                   help="name=source [source ...] (comma-separated sources)")  # fmt: skip
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--per-rep", type=int, default=300)
    p.add_argument("--workers", type=int, default=int(os.environ.get("GCA_WORKERS", 4)))
    p.add_argument("--out", type=Path, default=Path("docs/results/r14_robust.md"))
    args = p.parse_args()
    sets = {}
    for spec in args.sets:
        name, srcs = spec.split("=", 1)
        runs = rb.load_runs(resolve_sources(srcs.split(",")))
        if not runs:
            raise SystemExit(f"set {name}: no trained runs in {srcs}")
        sets[name] = runs
        print(f"[load] {name}: {len(runs)} runs", flush=True)
    todo = rr.tasks(sets, args.reps, args.per_rep, args.conditions)
    cells, start = {}, time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (key, res) in enumerate(pool.map(rr.play_task, todo), 1):
            cells[key] = res
            if i % 10 == 0 or i == len(todo):
                print(f"[progress] {i}/{len(todo)} cells, "
                      f"{time.perf_counter() - start:.0f} s", flush=True)  # fmt: skip
    rep = rr.summarise(cells, sets)
    md = rr.render(rep, {k: len(v) for k, v in sets.items()}, args.reps * args.per_rep)
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
