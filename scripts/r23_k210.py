"""R23 (goal G19): score the K210-compiled scout brain against float (experiments/r23_k210.py).

The compiled models' raw outputs come from scripts/r23_k210_sim.py (its own venv).

python scripts/r23_k210.py report
"""

import argparse
import time

from genetic_coil_arena.experiments import r23_k210 as rk


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report")
    p.parse_args()
    rep = rk.report(log=lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True))
    print(rk.render(rep))


if __name__ == "__main__":
    main()
