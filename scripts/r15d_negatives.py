"""R15d: the scout brain trained on every empty tile vs R15b's (experiments/r15d_negatives.py).

GPU (prediction on the VisDrone test tiles):
python scripts/r15d_negatives.py --from runs:r15b2-train-ft-a05 runs:r15d-train-neg
"""

import os
import sys

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from genetic_coil_arena.experiments import r15_scout_brain as r15  # noqa: E402
from genetic_coil_arena.experiments import r15b_aerial_scout as r15b  # noqa: E402
from genetic_coil_arena.experiments import r15d_negatives as r15d  # noqa: E402
from genetic_coil_arena.experiments.r7b_report import resolve_sources  # noqa: E402


def main() -> None:
    import torch

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--from", dest="sources", nargs="+", required=True)
    p.add_argument("--data", type=Path, default=r15.DATA_DIR)
    p.add_argument("--out", type=Path, default=Path("docs/results/r15d_negatives.md"))
    args = p.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.85)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_input = r15b.input_fn(dev)
    test = r15b.load_tiles(args.data / "visdrone_tiles224", "test")
    runs = []
    for src in resolve_sources(args.sources):
        for f in sorted(Path(src).glob("r15b_coco_a0.5_seed*.json")):
            meta = json.loads(f.read_text(encoding="utf-8"))
            if meta.get("size", 224) != 224:
                continue
            group = (
                "all empty tiles" if meta.get("tiles", "").endswith("_neg") else "R15b"
            )
            runs.append(
                {"group": group, "seed": meta["seed"], "weights": meta["weights"]}
            )
    print(f"[load] {len(runs)} brains", flush=True)
    for r in runs:
        start = time.time()
        model = r15.build_model(0.5)
        model.load_state_dict(torch.load(r["weights"], map_location="cpu"))
        model.to(dev)
        preds, top = r15b.predict(model, test, 224, dev, to_input, top=100)
        r["operating"] = r15b.tile_metrics(preds, top, test, n_boot=0)["pooled"][
            "operating"
        ]
        print(
            f"[done] {r['group']} seed {r['seed']}: {time.time() - start:.0f} s",
            flush=True,
        )
    rep = r15d.summarise(runs)
    rep["runs_operating"] = [
        {k: v for k, v in r.items() if k != "weights"} for r in runs
    ]
    md = r15d.render(rep)
    js = json.dumps(rep, indent=1, default=float)
    outs = [args.out]
    if os.environ.get("GCA_RUN_DIR"):
        outs.append(Path(os.environ["GCA_RUN_DIR"]) / args.out.name)
    for out in outs:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        out.with_suffix(".json").write_text(js, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()
