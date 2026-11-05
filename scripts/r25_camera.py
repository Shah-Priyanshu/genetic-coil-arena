"""R25: the 448-px camera through R19's confirmer (experiments/r25_camera.py).

python scripts/r25_camera.py crops --scout s448_seed0 --split val   # also test; s224_seed1, s448_seed1
python scripts/r25_camera.py score --scout s448_seed0 --split val   # test-dev once per scout
python scripts/r25_camera.py report
# R25b: the 448 camera with its own confirmer
python scripts/r25_camera.py crops --scout s448_seed0 --split train
python scripts/r25_camera.py fit --seed 0                          # seeds 0, 1, 2
python scripts/r25_camera.py score --scout s448_seed0 --split val --confirmer own
python scripts/r25_camera.py report-b
# R25c: clear non-persons, labelled blind
python scripts/r25_camera.py sample-c                              # sheets in <data>/r25_camera/visual_c
python scripts/r25_camera.py report-c                              # after the labels_*.json are in

Candidates, crops and confirmer scores go to ``D:/Projects/genetic-coil-arena-data/r25_camera``;
the 224 seed-0 scout is R19's own. ``report`` writes ``docs/results/r25_camera.{md,json}``.
"""

import argparse
import os
import time
from pathlib import Path

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
        splits = ("train", "val", "test") if name == "crops" else ("val", "test")
        c.add_argument("--split", choices=splits, required=True)
    c.add_argument("--confirmer", choices=("r19", r25.OWN), default="r19")
    f = sub.add_parser("fit", help="R25b: the confirmer on the 448 scout's candidates")
    f.add_argument("--seed", type=int, choices=r25.OWN_SEEDS, required=True)
    sub.add_parser("report")
    sub.add_parser("report-b")
    sub.add_parser("sample-c")
    sub.add_parser("report-c")
    args = p.parse_args()
    if args.cmd == "crops":
        r25.build(args.scout, args.split, log=log)
    elif args.cmd == "score":
        r25.score(args.scout, args.split, log=log, variant=args.confirmer)
    elif args.cmd == "fit":
        r25.fit_own(args.seed, log=log)
    elif args.cmd == "sample-c":
        r25.sample_sheets(log=log)
    elif args.cmd == "report-c":
        rep = r25.report_c()
        log(f"verdict: {rep['verdict']}, pooled ratio {rep['pooled_ratio']}")
    else:
        rep = (r25.report() if args.cmd == "report"
               else r25.report(Path("docs/results/r25b_camera_own.md"), r25.OWN))  # fmt: skip
        log(f"verdict: {rep['verdict']}, pooled ratio {rep['pooled_ratio']}")


if __name__ == "__main__":
    main()
