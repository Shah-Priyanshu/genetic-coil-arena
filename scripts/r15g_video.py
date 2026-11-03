"""R15g: the scout brain over VisDrone2019-VID (experiments/r15g_video.py).

python scripts/r15g_video.py detect --split val [--limit 1] [--device cuda]

Writes ``<split>.npz`` (detections and labelled people per frame) to
``D:/Projects/genetic-coil-arena-data/visdrone_vid_dets`` and, under the runner, a copy
to ``$GCA_RUN_DIR``.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from genetic_coil_arena.experiments import r15g_video as rg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--split", choices=sorted(rg.SPLITS), default="val")
    d.add_argument("--device", default="cuda")
    d.add_argument("--limit", type=int, default=0, help="first N sequences only")
    args = p.parse_args()
    start = time.perf_counter()
    arrays = rg.detect_split(args.split, args.device, args.limit,
                             log=lambda m: print(m, flush=True))  # fmt: skip
    outs = [rg.OUT_DIR / f"{args.split}{'_limit' if args.limit else ''}.npz"]
    if os.environ.get("GCA_RUN_DIR"):
        outs.append(Path(os.environ["GCA_RUN_DIR"]) / outs[0].name)
    for out in outs:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays)
    n = arrays["seq_frames"][:, 0].sum()
    print(f"{len(arrays['seq_names'])} sequences, {n} frames, {len(arrays['dets'])} "
          f"detections >= {rg.MIN_SCORE}, {len(arrays['gts'])} labelled people, "
          f"{time.perf_counter() - start:.0f} s -> {outs[0]}", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
