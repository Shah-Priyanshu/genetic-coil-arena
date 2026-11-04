"""R23 (goal G19): score the K210-compiled scout brain against float (experiments/r23_k210.py).

The compiled models' raw outputs come from scripts/r23_k210_sim.py (its own venv).

python scripts/r23_k210.py report
python scripts/r23_k210.py report-b   # R23b: thresholds set on val
python scripts/r23_k210.py select-c   # R23c: pick the compile variant on val
python scripts/r23_k210.py report-c   # R23c: the chosen variant on test-dev
"""

import argparse
import time

from genetic_coil_arena.experiments import r23_k210 as rk


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report")
    sub.add_parser("report-b")
    sub.add_parser("select-c")
    sub.add_parser("report-c")
    args = p.parse_args()
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    if args.cmd == "report":
        print(rk.render(rk.report(log=log)))
    elif args.cmd == "report-b":
        print(rk.render_b(rk.report_b(log=log)))
    elif args.cmd == "select-c":
        sel = rk.select_c(log=log)
        print({v: round(r["recall_at_false"], 4) for v, r in sel["variants"].items()}, sel["chosen"])
    else:
        print(rk.render_c(rk.report_c(log=log)))


if __name__ == "__main__":
    main()
