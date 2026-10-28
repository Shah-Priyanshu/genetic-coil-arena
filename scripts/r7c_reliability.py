"""G7c report: (b) runs trained on fresh seeds under the pre-registered restart rule.

Thin wrapper around ``genetic_coil_arena.experiments.r7c_reliability``; see that module.
Training uses the G7 script (jobs ``r7c-train-*``):

    python scripts/r7b_goal_conditioned.py --train-only --modes centre-goal --runs 16:26 ...
    python scripts/r7c_reliability.py --from-json runs:r7c-train-* \
        --out docs/results/r7c_reliability.md
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from genetic_coil_arena.experiments import r7c_reliability as r7c  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import (  # noqa: E402
    read_trained,
    resolve_sources,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from-json", nargs="+", required=True)
    p.add_argument("--reference", type=Path, default=r7c.REFERENCE)
    p.add_argument("--out", type=Path, default=Path("docs/results/r7c_reliability.md"))
    args = p.parse_args()
    trained, settings_d = read_trained(resolve_sources(args.from_json))
    print(f"[eval] {len(trained)} trained runs", flush=True)
    rep = r7c.build_report(trained, settings_d, r7c.load_reference(args.reference))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(r7c.render(rep), encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(rep, indent=1), encoding="utf-8"
    )
    run_dir = os.environ.get("GCA_RUN_DIR")
    if run_dir:  # copies into the job's artifacts
        art = Path(run_dir)
        (art / args.out.name).write_text(r7c.render(rep), encoding="utf-8")
        (art / args.out.with_suffix(".json").name).write_text(
            json.dumps(rep, indent=1), encoding="utf-8"
        )
    print(f"G7c verdict: {rep['g7c']['verdict']}; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
