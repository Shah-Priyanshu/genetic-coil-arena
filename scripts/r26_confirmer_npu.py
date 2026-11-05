"""R26: R19's confirmer on MaixCAM's NPU (experiments/r26_confirmer_npu.py).

python scripts/r26_confirmer_npu.py export                 # the ONNX the simulator compiles
python scripts/r26_confirmer_npu.py prepare --split test   # the sample's uint8 views and fp32 scores
# in WSL: scripts/r26_confirmer_npu_sim.py compile; simulate --split test
python scripts/r26_confirmer_npu.py report

Runs on CPU (set CUDA_VISIBLE_DEVICES= beside a GPU job). Data go to
``D:/Projects/genetic-coil-arena-data/r26_confirmer_npu``; ``report`` writes
``docs/results/r26_confirmer_npu.{md,json}``.
"""

import argparse
import os
import time

os.environ.setdefault("HF_HOME", r"D:\hf_cache")

from genetic_coil_arena.experiments import r26_confirmer_npu as r26  # noqa: E402


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    c = sub.add_parser("prepare")
    c.add_argument("--split", choices=("val", "test"), required=True)
    sub.add_parser("report")
    args = p.parse_args()
    if args.cmd == "export":
        r26.export(log=log)
    elif args.cmd == "prepare":
        r26.prepare(args.split, log=log)
    else:
        log(f"verdict: {r26.report(log=log)['verdict']}")


if __name__ == "__main__":
    main()
