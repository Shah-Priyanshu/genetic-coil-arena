"""R7e probe: squad A's centre drives squad B's specialists through the G12 meaning space.

Thin wrapper around ``genetic_coil_arena.experiments.r7e_interlingua``; see that module.
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from genetic_coil_arena.experiments import r7e_interlingua as r7e  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import (  # noqa: E402
    env_workers,
    read_trained,
    resolve_sources,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from-json", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=0, help="first N pairs (smoke)")
    p.add_argument("--workers", type=int, default=env_workers(8))
    p.add_argument("--out", type=Path, default=Path("docs/results/r7e_interlingua.md"))
    args = p.parse_args()
    trained, settings_d = read_trained(resolve_sources(args.from_json))
    pairs = r7e.pairs(trained)
    if args.limit:
        pairs = pairs[: args.limit]
    start, rows = time.time(), []
    with ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(r7e.evaluate_pair, [(a, b, settings_d) for a, b in pairs]):
            rows.append(row)
            sc = row["scores"]
            print(
                f"[r7e] {row['a']}->{row['b']}: native {sc['native']:.1f}, raw "
                f"{sc['raw']:.1f}, bridged {sc['bridged']:.1f}, hand {sc['hand']:.1f}, "
                f"oracle {sc['oracle']:.1f} ({time.time() - start:.0f} s)",
                flush=True,
            )
    rep = {"n_pairs": len(rows), "summary": r7e.summarise(rows), "pairs": rows,
           "settings": settings_d}  # fmt: skip
    md, js = r7e.render(rep), json.dumps(rep, indent=1)
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
