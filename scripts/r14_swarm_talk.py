"""R14 step 1 report: does a drone swarm need to talk, and does its radio bind? (G13)

Thin wrapper around ``genetic_coil_arena.experiments.r14_swarm_talk``; see that module.

    python scripts/r14_swarm_talk.py --out docs/results/r14_swarm_talk.md
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from genetic_coil_arena.experiments import r14_swarm_talk as r14  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reps", type=int, default=r14.N_REPS)
    p.add_argument("--per-rep", type=int, default=r14.PER_REP)
    p.add_argument("--out", type=Path, default=Path("docs/results/r14_swarm_talk.md"))
    args = p.parse_args()
    start = time.time()
    hand = r14.tune(per_rep=args.per_rep)
    print(f"[r14] tuned in {time.time() - start:.0f} s: {hand}", flush=True)
    cells = r14.run_grid(hand, per_rep=args.per_rep, reps=args.reps)
    rep = r14.build_report(hand, cells)
    rep["grid"]["per_rep"] = args.per_rep
    md, js = r14.render(rep), json.dumps(rep, indent=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(js, encoding="utf-8")
    run_dir = os.environ.get("GCA_RUN_DIR")
    if run_dir:  # copies into the job's artifacts
        art = Path(run_dir)
        (art / args.out.name).write_text(md, encoding="utf-8")
        (art / args.out.with_suffix(".json").name).write_text(js, encoding="utf-8")
    print(
        f"G13: {rep['g13']['verdict']} ({time.time() - start:.0f} s); wrote {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
