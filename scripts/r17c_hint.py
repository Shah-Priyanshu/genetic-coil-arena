"""R17 step 3: the planner that reads the operator's moving hint (experiments/r17c_hint.py).

Training: python scripts/r17_mission.py train --kind conditioned --version 3 --runs 0:8
Report:   python scripts/r17c_hint.py --from runs:r17-train-cond-* runs:r17b-train-cond-* runs:r17c-train-*
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
from genetic_coil_arena.experiments import r17c_hint as rc  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import resolve_sources  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from", dest="sources", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=rc.EPISODES)
    p.add_argument("--workers", type=int, default=int(os.environ.get("GCA_WORKERS", 4)))
    p.add_argument("--out", type=Path, default=Path("docs/results/r17c_hint.md"))
    args = p.parse_args()
    folders = resolve_sources(args.sources)
    runs = {name: r17.load_runs(folders, "conditioned", version=v)
            for name, v in rc.SYSTEMS.items()}  # fmt: skip
    print("[load] " + ", ".join(f"{n}: {len(r)}" for n, r in runs.items()), flush=True)
    if not all(runs.values()):
        raise SystemExit("need conditioned runs of versions 1, 2 and 3")
    todo = rc.tasks(runs, args.episodes)
    cells, start = {}, time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (key, res) in enumerate(pool.map(rc.play_cell, todo, chunksize=16), 1):
            cells[key] = res
            if i % 500 == 0 or i == len(todo):
                print(f"[progress] {i}/{len(todo)} cells, "
                      f"{time.perf_counter() - start:.0f} s", flush=True)  # fmt: skip
    rep = rc.summarise(cells)
    md = rc.render(rep, args.episodes)
    js = json.dumps(rep, indent=1, default=float)
    outs = [args.out]
    if os.environ.get("GCA_RUN_DIR"):
        outs.append(Path(os.environ["GCA_RUN_DIR"]) / args.out.name)
    for out in outs:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        out.with_suffix(".json").write_text(js, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()
