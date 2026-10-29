"""R15: the scout brain, a K210-sized person detector for OpenSwarm (COCO persons).

Thin wrapper around ``genetic_coil_arena.experiments.r15_scout_brain``; see that module.

    python scripts/r15_scout_brain.py prepare [--size 160]      # once, CPU
    python scripts/r15_scout_brain.py train --alphas 0.25 0.5 0.75 [--epochs 40]   # GPU
    python scripts/r15_scout_brain.py report --from runs:r15-train-* --out docs/results/r15_scout_brain.md
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from genetic_coil_arena.experiments import r15_scout_brain as r15  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import resolve_sources  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def render(results: list[dict]) -> str:
    lines = [
        "# R15: the scout brain, a person detector sized for OpenSwarm's K210",
        "",
        "COCO 2017 persons, letterboxed to the input size; trained on train2017, scored "
        "on the 5,000 val2017 images. AP50 = average precision at IoU 0.5 for persons "
        '("tall": persons at least 32 px high). Present AUC = how well the frame\'s top '
        "score separates frames with a person (≥ 0.5% of the image, VWW's rule) from "
        "frames without. int8 weights = one byte per parameter.",
        "",
        "| Width (alpha) | Input | Params | int8 weights | MACs per frame | AP50 | AP50, tall persons | Present AUC | Training |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: r["alpha"]):
        lines.append(
            f"| {r['alpha']} | {r['size']}×{r['size']} | {r['params'] / 1e3:.0f} k "
            f"| {r['int8_kb']:.0f} KB | {r['macs'] / 1e6:.1f} M | {r['ap50']:.3f} "
            f"| {r['ap50_tall']:.3f} | {r['present_auc']:.3f} | {r['seconds'] / 60:.0f} min |"
        )
    lines += [
        "",
        "Reproduce: `python scripts/r15_scout_brain.py train` then `report`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=r15.DATA_DIR)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--size", type=int, default=160)
    a.add_argument("--workers", type=int, default=6)
    a.add_argument("--limit", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--size", type=int, default=160)
    t.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--out", default=os.environ.get("GCA_RUN_DIR", "outputs/r15"))
    r = sub.add_parser("report")
    r.add_argument("--from", dest="sources", nargs="+", required=True)
    r.add_argument("--out", type=Path, default=Path("docs/results/r15_scout_brain.md"))
    args = p.parse_args()
    if args.cmd == "prepare":
        info = r15.prepare(args.data / "coco", args.data / f"coco_person{args.size}",
                           args.size, args.workers, args.limit)  # fmt: skip
        log(json.dumps(info))
        return
    if args.cmd == "train":
        import torch

        data = args.data / f"coco_person{args.size}"
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        models = args.data / "r15_models"
        models.mkdir(parents=True, exist_ok=True)
        for alpha in args.alphas:
            res = r15.train_model(
                alpha, data, args.epochs, args.batch, limit=args.limit, log=log
            )
            tag = f"scout_a{alpha:g}_s{args.size}"
            torch.save(res["model"].state_dict(), models / f"{tag}.pt")
            res["result"]["onnx"] = r15.export_onnx(
                res["model"], args.size, models / f"{tag}.onnx"
            )
            res["result"]["weights"] = str(models / f"{tag}.pt")
            (out / f"r15_{tag}.json").write_text(
                json.dumps(res["result"], indent=1), encoding="utf-8"
            )
            log(f"{tag}: {json.dumps(res['result'])}")
        return
    results = []
    for src in resolve_sources(args.sources):
        for f in sorted(Path(src).glob("r15_scout_*.json")):
            results.append(json.loads(f.read_text(encoding="utf-8")))
    md = render(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(results, indent=1), encoding="utf-8"
    )
    run_dir = os.environ.get("GCA_RUN_DIR")
    if run_dir:
        (Path(run_dir) / args.out.name).write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()
