"""R14 probe: with several gas sources, how much room does coordination leave?

Thin wrapper around ``genetic_coil_arena.experiments.r14_multi_probe``; see that module.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from genetic_coil_arena.experiments import r14_multi_probe as probe  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reps", type=int, default=probe.N_REPS)
    p.add_argument("--per-rep", type=int, default=probe.PER_REP)
    p.add_argument("--tune-episodes", type=int, default=probe.TUNE_EPISODES)
    p.add_argument("--out", type=Path, default=Path("docs/results/r14_multi_probe.md"))
    args = p.parse_args()
    start = time.time()
    knobs = probe.tune(episodes=args.tune_episodes)
    print(f"[probe] tuned in {time.time() - start:.0f} s: {knobs}", flush=True)
    cells = probe.run(knobs, args.reps, args.per_rep)
    rep = {"knobs": knobs, "reps": args.reps, "per_rep": args.per_rep,
           "summary": probe.summarise(cells), "cells": cells}  # fmt: skip
    md, js = probe.render(rep), json.dumps(rep, indent=1)
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
