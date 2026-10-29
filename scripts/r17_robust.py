"""R17 step 1b: the trained search planners in a worse world (see experiments/r17_robust.py).

python scripts/r17_robust.py --from runs:r17-train-* [--workers 8] [--out docs/results/r17_robust.md]
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from genetic_coil_arena.experiments import r17_mission as r17  # noqa: E402
from genetic_coil_arena.experiments import r17_robust as rr  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import resolve_sources  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from", dest="sources", nargs="+", required=True)
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--episodes", type=int, default=rr.EPISODES)
    p.add_argument("--workers", type=int, default=int(os.environ.get("GCA_WORKERS", 4)))
    p.add_argument("--out", type=Path, default=Path("docs/results/r17_robust.md"))
    args = p.parse_args()
    folders = resolve_sources(args.sources)
    blind, cond = r17.load_runs(folders, "blind"), r17.load_runs(folders, "conditioned")
    print(f"[load] blind {len(blind)}, conditioned {len(cond)}", flush=True)
    if not blind or not cond:
        raise SystemExit("need blind and conditioned runs")
    todo = rr.tasks(blind, cond, args.episodes, args.conditions)
    cells, start = {}, time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (key, res) in enumerate(pool.map(rr.play_cell, todo, chunksize=16), 1):
            cells[key] = res
            if i % 500 == 0 or i == len(todo):
                print(f"[progress] {i}/{len(todo)} cells, "
                      f"{time.perf_counter() - start:.0f} s", flush=True)  # fmt: skip
    rep = rr.summarise(cells)
    md = rr.render(rep)
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
