"""R25: the 448-px camera through R19's confirmer (experiments/r25_camera.py).

python scripts/r25_camera.py crops --scout s448_seed0 --split val   # also test; s224_seed1, s448_seed1
python scripts/r25_camera.py score --scout s448_seed0 --split val   # test-dev once per scout
python scripts/r25_camera.py report

Candidates, crops and confirmer scores go to ``D:/Projects/genetic-coil-arena-data/r25_camera``;
the 224 seed-0 scout is R19's own. ``report`` writes ``docs/results/r25_camera.{md,json}``.
"""

import argparse
import os
import time

os.environ.setdefault("HF_HOME", r"D:\hf_cache")

from genetic_coil_arena.experiments import r25_camera as r25  # noqa: E402


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("crops", "score"):
        c = sub.add_parser(name)
        c.add_argument("--scout", choices=sorted(r25.SCOUTS), required=True)
        c.add_argument("--split", choices=("val", "test"), required=True)
    sub.add_parser("report")
    args = p.parse_args()
    if args.cmd == "crops":
        r25.build(args.scout, args.split, log=log)
    elif args.cmd == "score":
        r25.score(args.scout, args.split, log=log)
    else:
        rep = r25.report()
        log(f"verdict: {rep['verdict']}, pooled ratio {rep['pooled_ratio']}")


if __name__ == "__main__":
    main()
