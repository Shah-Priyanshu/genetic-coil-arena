"""R24: score the 448-px scout brain compiled for MaixCAM (cv181x) against float.

Simulation is ``scripts/r24_maixcam_sim.py`` in WSL; this runs in the main venv.

python scripts/r24_maixcam.py export --model r15b_coco_a0.5_seed1_s448
python scripts/r24_maixcam.py report
"""

from __future__ import annotations

import argparse
import time

from genetic_coil_arena.experiments import r23_k210 as r23
from genetic_coil_arena.experiments import r24_maixcam as r24


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="the split-head ONNX (R23c's 'heat' export) at 448")
    e.add_argument("--model", required=True)
    sub.add_parser("report")
    args = p.parse_args()
    if args.cmd == "export":
        log(str(r23.export_heat_onnx(args.model, size=r24.SIZE, out_dir=r24.OUT_DIR)))
    else:
        rep = r24.report(log=log)
        log(f"verdict: {rep['verdict']}")


if __name__ == "__main__":
    main()
