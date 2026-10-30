"""R15e: crops of the scout brain's detections for an operator labelling session.

Runs R15b's width-0.5 fine-tuned brain (seed 0) on the CPU over VisDrone *val* tiles
(native scale; the test split stays untouched), keeps detections at confidence >= 0.5
(a sweep threshold), and samples ``--n`` of them, half on a person (centre hit) and half
false alarms. Each crop is 72 x 72 tile pixels around the detection's centre, shown 3x,
with a ring on the centre: the question is whether that spot is on a person. Writes ``crops.json`` (id and PNG, shuffled; no labels) for the labelling
page, and ``truth.json`` (id, person or not, score) kept beside it, under
``D:/Projects/genetic-coil-arena-data/r15e_operator``.

python scripts/r15e_operator_crops.py [--n 200] [--tiles 4000]
"""

import argparse
import base64
import io
import json

import numpy as np
from PIL import Image, ImageDraw

from genetic_coil_arena.experiments import r15_scout_brain as r15
from genetic_coil_arena.experiments import r15b_aerial_scout as r15b

WEIGHTS = r15.DATA_DIR / "r15_models" / "r15b_coco_a0.5_seed0.pt"
OUT = r15.DATA_DIR / "r15e_operator"
HALF = 36  # crop half-size in tile pixels (72 x 72), shown 3x larger


def main() -> None:
    import torch

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--tiles", type=int, default=4000)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1519)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    val = r15b.load_tiles(r15.DATA_DIR / "visdrone_tiles224", "val")
    native = np.flatnonzero(np.asarray(val["scale"]) == 1.0)
    pick = np.sort(rng.choice(native, size=min(args.tiles, len(native)), replace=False))
    sub = {"images": np.asarray(val["images"])[pick],
           "boxes": [r15.image_boxes(val, int(i)) for i in pick]}  # fmt: skip
    model = r15.build_model(0.5)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    dev = torch.device("cpu")
    preds = []
    to_input = r15b.input_fn(dev)
    for a in range(0, len(pick), 128):
        split = {"images": sub["images"][a : a + 128],
                 "offsets": np.arange(len(sub["images"][a : a + 128]) + 1)}  # fmt: skip
        out, _ = r15b.predict(model, split, 224, dev, to_input, top=100)
        preds += out
    table = r15b.hit_table(preds, sub["boxes"], "centre")
    keep = table["score"] >= args.threshold
    true_idx = np.flatnonzero(keep & (table["det_gt"] >= 0))
    false_idx = np.flatnonzero(keep & (table["det_gt"] < 0))
    k = args.n // 2
    chosen = np.concatenate([rng.choice(true_idx, k, replace=False),
                             rng.choice(false_idx, k, replace=False)])  # fmt: skip
    rng.shuffle(chosen)
    flat = np.concatenate(preds)
    crops, truth = [], []
    for n, d in enumerate(chosen):
        tile = int(table["det_tile"][d])
        x, y, w, h, score = flat[d]
        cx, cy = int(round(x + w / 2)), int(round(y + h / 2))
        img = np.zeros((2 * HALF, 2 * HALF, 3), dtype=np.uint8)
        src = sub["images"][tile]
        x0, y0 = max(cx - HALF, 0), max(cy - HALF, 0)
        x1, y1 = min(cx + HALF, src.shape[1]), min(cy + HALF, src.shape[0])
        img[
            y0 - (cy - HALF) : y1 - (cy - HALF), x0 - (cx - HALF) : x1 - (cx - HALF)
        ] = src[y0:y1, x0:x1]
        big = Image.fromarray(img).resize((6 * HALF, 6 * HALF), Image.BICUBIC)
        ring = ImageDraw.Draw(
            big
        )  # mark the detection's centre: that spot is the question
        c, r = 3 * HALF, 16
        ring.ellipse((c - r, c - r, c + r, c + r), outline=(255, 230, 0), width=2)
        buf = io.BytesIO()
        big.save(buf, format="PNG")
        cid = f"c{n:03d}"
        crops.append({"id": cid, "png": base64.b64encode(buf.getvalue()).decode()})
        truth.append({"id": cid, "person": bool(table["det_gt"][d] >= 0),
                      "score": float(score), "size_px": float(np.sqrt(w * h))})  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crops.json").write_text(json.dumps(crops), encoding="utf-8")
    (OUT / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(
        f"{len(crops)} crops ({k} person, {k} false alarm) from {len(pick)} val tiles; "
        f"detections at >= {args.threshold}: {len(true_idx)} person, "
        f"{len(false_idx)} false -> {OUT}"
    )


if __name__ == "__main__":
    main()
