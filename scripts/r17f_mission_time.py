"""R17 step 5: plan at mission time (experiments/r17f_mission_time.py).

python scripts/r17f_mission_time.py --from runs:r17-train-cond-* runs:r17b-train-cond-* runs:r17b-oracle
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
from genetic_coil_arena.experiments import r17f_mission_time as rf  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import resolve_sources  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from", dest="sources", nargs="+", required=True)
    p.add_argument("--missions", type=int, default=r17.N_TEST)
    p.add_argument("--generations", type=int, default=rf.GENERATIONS,
                   help="tuning budget (the budget curve reruns with other values)")  # fmt: skip
    p.add_argument("--workers", type=int, default=int(os.environ.get("GCA_WORKERS", 4)))
    p.add_argument(
        "--out", type=Path, default=Path("docs/results/r17f_mission_time.md")
    )
    args = p.parse_args()
    folders = resolve_sources(args.sources)
    step1 = r17.load_runs(folders, "conditioned", version=1)
    step2 = r17.load_runs(folders, "conditioned", version=2)
    oracle2 = r17.load_oracle(folders, version=2)
    print(f"[load] step-1 {len(step1)}, step-2 {len(step2)}, oracle {len(oracle2)}",
          flush=True)  # fmt: skip
    if not step1 or not step2:
        raise SystemExit("need step-1 and step-2 conditioned planners")
    todo = [(which, i, step2[0], args.generations)
            for which in rf.SETS for i in range(args.missions)]  # fmt: skip
    tuned, start = {}, time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for k, rec in enumerate(pool.map(rf.tune_task, todo), 1):
            tuned[(rec["set"], rec["mission"])] = rec
            if k % 10 == 0 or k == len(todo):
                print(f"[progress] {k}/{len(todo)} missions tuned, "
                      f"{time.perf_counter() - start:.0f} s", flush=True)  # fmt: skip
    cells = {}
    with ProcessPoolExecutor(args.workers) as pool:
        tasks = rf.eval_tasks(tuned, step1, step2, oracle2)
        for key, score in pool.map(rf.eval_task, tasks, chunksize=16):
            cells[key] = score
    rep = rf.summarise(cells, tuned)
    rep["tuned"] = [dict(v, set=k[0]) for k, v in tuned.items()]
    md = rf.render(rep)
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
