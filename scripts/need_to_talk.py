"""Need-to-talk baselines for the squad: does the centre's information help?

Thin wrapper around ``genetic_coil_arena.analysis.need_to_talk`` (also available as
``genetic-coil squad-benchmark``).

    python scripts/need_to_talk.py --seeds 100 --out docs/results/need_to_talk.md
"""

import argparse
from pathlib import Path

from genetic_coil_arena.analysis.need_to_talk import RADII, run_need_to_talk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=100, help="Episodes per setting")
    parser.add_argument(
        "--radius",
        type=int,
        action="append",
        default=None,
        help=f"Window radius (repeatable; default {', '.join(map(str, RADII))})",
    )
    parser.add_argument("--batch", type=int, default=256, help="Throughput batch size")
    parser.add_argument(
        "--no-throughput", action="store_true", help="Skip the throughput measurement"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Write a Markdown report"
    )
    args = parser.parse_args()

    print(
        run_need_to_talk(
            args.seeds,
            args.radius or RADII,
            batch_size=args.batch,
            throughput=not args.no_throughput,
            out=args.out,
        )
    )


if __name__ == "__main__":
    main()
