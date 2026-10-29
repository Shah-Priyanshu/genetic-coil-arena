"""R15b: the scout brain from a drone's view (VisDrone tiles at OpenSwarm's 224 input).

Thin wrapper around ``genetic_coil_arena.experiments.r15b_aerial_scout``; see that module.

    python scripts/r15b_aerial_scout.py prepare                      # once, CPU
    python scripts/r15b_aerial_scout.py train --init coco --alphas 0.5 --seeds 0 1   # GPU
    python scripts/r15b_aerial_scout.py report --from runs:r15b-train-* --out docs/results/r15b_aerial_scout.md
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from genetic_coil_arena.experiments import r15_scout_brain as r15  # noqa: E402
from genetic_coil_arena.experiments import r15b_aerial_scout as r15b  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import resolve_sources  # noqa: E402

TILES = "visdrone_tiles224"
COCO_SIZE = 160


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def coco_weights(models: Path, alpha: float) -> Path:
    return models / f"scout_a{alpha:g}_s{COCO_SIZE}.pt"


def train(args: argparse.Namespace) -> None:
    import torch

    data = args.data / TILES
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = args.data / "r15_models"
    for alpha in args.alphas:
        for seed in args.seeds:
            init = coco_weights(models, alpha) if args.init == "coco" else None
            res = r15.train_model(
                alpha, data, args.epochs, args.batch, args.lr, seed, args.limit, log,
                init=init, eval_fn=r15b.val_metrics,
            )  # fmt: skip
            tag = f"r15b_{args.init}_a{alpha:g}_seed{seed}"
            torch.save(res["model"].state_dict(), models / f"{tag}.pt")
            res["result"].update({
                "condition": "fine-tuned" if args.init == "coco" else "scratch",
                "seed": seed, "lr": args.lr, "weights": str(models / f"{tag}.pt"),
                "onnx": r15.export_onnx(res["model"], 224, models / f"{tag}.onnx"),
            })  # fmt: skip
            (out / f"{tag}.json").write_text(
                json.dumps(res["result"], indent=1), encoding="utf-8"
            )
            log(f"{tag}: {json.dumps(res['result'])}")


def report(args: argparse.Namespace) -> None:
    import torch

    torch.cuda.set_per_process_memory_fraction(0.85)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_input = r15b.input_fn(dev)
    test = r15b.load_tiles(args.data / TILES, "test")
    coco_val = r15.load_split(args.data / f"coco_person{COCO_SIZE}", "val")
    runs = []
    for alpha in (0.25, 0.5, 0.75):
        runs.append({"condition": "zero-shot", "alpha": alpha, "seed": 0,
                     "weights": str(coco_weights(args.data / "r15_models", alpha))})  # fmt: skip
    for src in resolve_sources(args.sources):
        for f in sorted(Path(src).glob("r15b_*.json")):
            runs.append(json.loads(f.read_text(encoding="utf-8")))
    for r in runs:
        model = r15.build_model(r["alpha"])
        model.load_state_dict(torch.load(r["weights"], map_location="cpu"))
        model.to(dev)
        start = time.time()
        preds, top = r15b.predict(model, test, 224, dev, to_input, top=100)
        r["test"] = r15b.tile_metrics(preds, top, test, n_boot=args.boot)
        coco = r15.evaluate(model, coco_val, COCO_SIZE, dev, to_input)
        r["coco_val"] = {k: coco[k] for k in ("ap50", "ap50_tall", "present_auc")}
        log(f"{r['condition']} a{r['alpha']} seed {r['seed']}: test AP50 "
            f"{r['test']['pooled']['ap50 all']:.3f}, centre AP "
            f"{r['test']['pooled']['centre_ap all']:.3f}, COCO AP50 "
            f"{coco['ap50']:.3f} ({time.time() - start:.0f} s)")  # fmt: skip
    rep = r15b.verdict(runs)
    md = r15b.render(rep, runs, test_frames=int(len(np.unique(test["src"]))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps({"verdict": rep, "runs": runs}, indent=1), encoding="utf-8"
    )
    run_dir = os.environ.get("GCA_RUN_DIR")
    if run_dir:
        for p in (args.out, args.out.with_suffix(".json")):
            (Path(run_dir) / p.name).write_text(p.read_text(encoding="utf-8"),
                                                encoding="utf-8")  # fmt: skip
    print(md, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=r15.DATA_DIR)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--workers", type=int, default=6)
    a.add_argument("--limit", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--init", choices=["coco", "scratch"], required=True)
    t.add_argument("--alphas", type=float, nargs="+", default=[0.5])
    t.add_argument("--seeds", type=int, nargs="+", default=[0])
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--out", default=os.environ.get("GCA_RUN_DIR", "outputs/r15b"))
    r = sub.add_parser("report")
    r.add_argument("--from", dest="sources", nargs="+", required=True)
    r.add_argument("--boot", type=int, default=200)
    r.add_argument(
        "--out", type=Path, default=Path("docs/results/r15b_aerial_scout.md")
    )
    args = p.parse_args()
    if args.cmd == "prepare":
        info = r15b.prepare(args.data / "visdrone", args.data / TILES,
                            workers=args.workers, limit=args.limit)  # fmt: skip
        log(json.dumps(info))
    elif args.cmd == "train":
        train(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
