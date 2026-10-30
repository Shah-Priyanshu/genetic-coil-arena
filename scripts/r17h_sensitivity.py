"""R17 step 6b: sensitivity of the recommended search system (experiments/r17h_sensitivity.py).

python scripts/r17h_sensitivity.py --from runs:r17b-train-cond-*
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
from genetic_coil_arena.experiments import r17h_sensitivity as rh  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import resolve_sources  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from", dest="sources", nargs="+", required=True)
    p.add_argument("--variants", nargs="*", default=list(rh.VARIANTS))
    p.add_argument("--generations", type=int, default=rf.GENERATIONS)
    p.add_argument("--workers", type=int, default=int(os.environ.get("GCA_WORKERS", 4)))
    p.add_argument("--out", type=Path, default=Path("docs/results/r17h_sensitivity.md"))
    args = p.parse_args()
    step2 = r17.load_runs(resolve_sources(args.sources), "conditioned", version=2)
    if not step2:
        raise SystemExit("need the step-2 conditioned planners")
    todo = [(v, i, step2[0], args.generations)
            for v in args.variants for i in range(r17.N_TEST)]  # fmt: skip
    tuned, start = [], time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        for k, rec in enumerate(pool.map(rh.tune_task, todo), 1):
            tuned.append(rec)
            if k % 20 == 0 or k == len(todo):
                print(f"[progress] {k}/{len(todo)} missions tuned, "
                      f"{time.perf_counter() - start:.0f} s", flush=True)  # fmt: skip
    cells = {}
    with ProcessPoolExecutor(args.workers) as pool:
        tasks = [(r["variant"], r["mission"], r["u"]) for r in tuned]
        for key, res in pool.map(rh.eval_task, tasks, chunksize=8):
            cells[key] = res
    rep = rh.summarise(cells)
    md = rh.render(rep)
    js = json.dumps(rep, indent=1)
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
