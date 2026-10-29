"""R17 step 2: evaluate the hand-off and the version 2 planners (experiments/r17b_handoff.py).

Training: python scripts/r17_mission.py train --kind conditioned --version 2 --runs 0:8
Report:   python scripts/r17b_handoff.py --from runs:r17-train-* runs:r17b-train-*
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
from genetic_coil_arena.experiments import r17b_handoff as rh  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import resolve_sources  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from", dest="sources", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=rh.EPISODES)
    p.add_argument("--workers", type=int, default=int(os.environ.get("GCA_WORKERS", 4)))
    p.add_argument("--out", type=Path, default=Path("docs/results/r17b_handoff.md"))
    args = p.parse_args()
    folders = resolve_sources(args.sources)
    step1 = r17.load_runs(folders, "conditioned")
    blind2 = r17.load_runs(folders, "blind", version=2)
    cond2 = r17.load_runs(folders, "conditioned", version=2)
    print(f"[load] step-1 {len(step1)}, blind v2 {len(blind2)}, conditioned v2 "
          f"{len(cond2)}", flush=True)  # fmt: skip
    if not (step1 and blind2 and cond2):
        raise SystemExit(
            "need step-1 conditioned and version 2 blind and conditioned runs"
        )
    todo = rh.tasks(step1, blind2, cond2, args.episodes)
    cells, start = {}, time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (key, res) in enumerate(pool.map(rh.play_cell, todo, chunksize=16), 1):
            cells[key] = res
            if i % 500 == 0 or i == len(todo):
                print(f"[progress] {i}/{len(todo)} cells, "
                      f"{time.perf_counter() - start:.0f} s", flush=True)  # fmt: skip
    rep = rh.summarise(cells)
    md = rh.render(rep, args.episodes)
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
