"""Goal-conditioned learned command centre (R7b, goal G7).

Thin wrapper around ``genetic_coil_arena.experiments.r7b_report``; see that module.

    # a queued training block (writes into $GCA_RUN_DIR)
    python scripts/r7b_goal_conditioned.py --train-only --modes blind --runs 0:8

    # evaluate all trained runs and write docs/results/r7b_goal_conditioned.{md,json}
    python scripts/r7b_goal_conditioned.py --from-json runs:r7b-train-* \
        --out docs/results/r7b_goal_conditioned.md

    python scripts/r7b_goal_conditioned.py --scale smoke --out outputs/r7b_smoke.md
"""

import os

# One BLAS thread per process: the training runs are spread over processes
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from genetic_coil_arena.experiments.r7b_report import main  # noqa: E402

if __name__ == "__main__":
    main()
