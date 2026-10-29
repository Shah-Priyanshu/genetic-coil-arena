"""R14 step 4a: the swarm brains in int8 (see experiments/r14_int8.py), and a C header.

python scripts/r14_int8.py --set step2=runs:r14b-train-* --set random=runs:r14c-train-*
    [--header-from random] [--workers 8] [--out docs/results/r14_int8.md]
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
from genetic_coil_arena.experiments import r14_int8 as ri  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import resolve_sources  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--set", dest="sets", action="append", required=True,
                   help="name=source [source ...] (comma-separated sources)")  # fmt: skip
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--per-rep", type=int, default=300)
    p.add_argument("--header-from", default=None,
                   help="set whose best-validation run becomes the C header")  # fmt: skip
    p.add_argument("--workers", type=int, default=int(os.environ.get("GCA_WORKERS", 4)))
    p.add_argument("--out", type=Path, default=Path("docs/results/r14_int8.md"))
    args = p.parse_args()
    sets = {}
    for spec in args.sets:
        name, srcs = spec.split("=", 1)
        sets[name] = rb.load_runs(resolve_sources(srcs.split(",")))
        if not sets[name]:
            raise SystemExit(f"set {name}: no runs")
        print(f"[load] {name}: {len(sets[name])} runs", flush=True)
    todo = ri.tasks(sets, args.reps, args.per_rep)
    cells, start = {}, time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (key, res) in enumerate(pool.map(ri.play, todo), 1):
            cells[key] = res
            if i % 10 == 0 or i == len(todo):
                print(f"[progress] {i}/{len(todo)} cells, "
                      f"{time.perf_counter() - start:.0f} s", flush=True)  # fmt: skip
    rep = ri.summarise(cells, sets)
    md = ri.render(rep, args.reps * args.per_rep)
    run_dir = os.environ.get("GCA_RUN_DIR")
    outs = [args.out] + ([Path(run_dir) / args.out.name] if run_dir else [])
    if args.header_from:
        best = max(sets[args.header_from], key=lambda r: r["best_val"])
        header = ri.c_header(best, rb.BrainSettings(**best["settings"]).hidden)
        rep["header_run"] = {"set": args.header_from, "run": best["run"]}
        for out in outs:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.with_name("swarm_brains_int8.h").write_text(header, encoding="utf-8")
    for out in outs:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()
