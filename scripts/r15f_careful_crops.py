"""R15f: crops for a careful-mode operator session (a second look at R15e's question).

R15e timed a quick single look (median 0.75 s per spot) and the operator confirmed 23% of
clearly empty spots. This draws 200 new spots the same way (R15b's width-0.5 brain, seed
0, on the CPU, VisDrone *val* at native scale, confidence >= 0.5, half on a person by
R15b's matching rule and half false alarms) from val tiles R15e did not draw. It drops
any spot within ``MIN_GAP`` frame pixels of an R15e spot or of an earlier pick (tiles of
one frame overlap). Each spot gets the R15e close-up (``2 * HALF`` pixels square) and a
wider view (``2 * WIDE``), both cut from the original frame (so no padding at tile
edges) with VisDrone's ignore regions greyed as the scout saw them. Writes ``crops.json`` (id, close-up and context JPEGs, shuffled; no labels),
``truth.json`` and contact sheets of the false alarms (for the visual check, made before
any answer is read) under ``D:/Projects/genetic-coil-arena-data/r15f_careful``.

python scripts/r15f_careful_crops.py [--n 200] [--tiles 4000] [--seed 2611]
"""

import argparse
import base64
import io
import json
import math
import zipfile
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

from genetic_coil_arena.experiments import r15_scout_brain as r15
from genetic_coil_arena.experiments import r15b_aerial_scout as r15b

WEIGHTS = r15.DATA_DIR / "r15_models" / "r15b_coco_a0.5_seed0.pt"
OUT = r15.DATA_DIR / "r15f_careful"
VAL_ZIP = r15.DATA_DIR / "visdrone" / "VisDrone2019-DET-val.zip"
HALF, WIDE, SHOW = 36, 72, 216  # close-up 72 px, context 144 px, both shown 216 px
RING = 16  # close-up ring radius in shown pixels (16/3 tile pixels, as in R15e)
RING_PX = RING * 2 * HALF / SHOW
MIN_GAP = 24.0
YELLOW = (255, 230, 0)


def detect(model, images: np.ndarray) -> list[np.ndarray]:
    import torch

    dev = torch.device("cpu")
    to_input = r15b.input_fn(dev)
    preds = []
    for a in range(0, len(images), 128):
        split = {"images": images[a : a + 128],
                 "offsets": np.arange(len(images[a : a + 128]) + 1)}  # fmt: skip
        preds += r15b.predict(model, split, 224, dev, to_input, top=100)[0]
    return preds


def draw(rng, val, native, model, tiles, threshold):
    """One draw of tiles and their thresholded detections, split by R15b's matching."""
    pick = np.sort(rng.choice(native, size=min(tiles, len(native)), replace=False))
    preds = detect(model, np.asarray(val["images"])[pick])
    boxes = [r15.image_boxes(val, int(i)) for i in pick]
    table = r15b.hit_table(preds, boxes, "centre")
    keep = table["score"] >= threshold
    return (pick, np.concatenate(preds), boxes, table,
            np.flatnonzero(keep & (table["det_gt"] >= 0)),
            np.flatnonzero(keep & (table["det_gt"] < 0)))  # fmt: skip


def tile_origins() -> tuple[list[tuple[int, float, int, int]], list[str], list]:
    """(frame, scale, x0, y0) of every val tile, rebuilt from ``plan_tiles`` (val keeps
    every tile, so no random draw), plus each frame's image member and ignore regions.
    """
    origins, members, ignores = [], [], []
    with zipfile.ZipFile(VAL_ZIP) as z:
        anns = sorted(
            n for n in z.namelist() if n.endswith(".txt") and "/annotations/" in n
        )
        for i, a in enumerate(anns):
            member = a.replace("/annotations/", "/images/")[:-4] + ".jpg"
            with z.open(member) as f:
                w, h = Image.open(f).size
            persons, ignore = r15b.parse_annotations(z.read(a).decode())
            plan = r15b.plan_tiles(w, h, persons, 224, 1.0, np.random.default_rng(0))
            origins += [(i, s, x0, y0) for s, x0, y0, _ in plan]
            members.append(member)
            ignores.append(ignore)
    return origins, members, ignores


def jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    import torch

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--tiles", type=int, default=4000)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2611)
    args = p.parse_args()
    val = r15b.load_tiles(r15.DATA_DIR / "visdrone_tiles224", "val")
    native = np.flatnonzero(np.asarray(val["scale"]) == 1.0)
    origins, members, ignores = tile_origins()
    assert len(origins) == len(
        val["scale"]
    ), "tile plan differs from the prepared tiles"
    model = r15.build_model(0.5)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))

    def spot(pick, flat, table, d):
        x, y, w, h, score = flat[d]
        t = int(pick[int(table["det_tile"][d])])
        f, s, x0, y0 = origins[t]
        assert s == 1.0
        return f, x0 + x + w / 2, y0 + y + h / 2

    # R15e's selection, re-run exactly as r15e_operator_crops.py drew it
    rng = np.random.default_rng(1519)
    pick1, flat1, _, table1, t1, f1 = draw(rng, val, native, model, 4000, 0.5)
    k = 100
    old = np.concatenate(
        [rng.choice(t1, k, replace=False), rng.choice(f1, k, replace=False)]
    )
    r15e = json.loads(
        (r15.DATA_DIR / "r15e_operator" / "truth.json").read_text("utf-8")
    )
    assert np.allclose(
        np.sort(flat1[old, 4]), np.sort([t["score"] for t in r15e])
    ), "R15e's draw did not reproduce"
    taken: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for d in old:
        f, fx, fy = spot(pick1, flat1, table1, d)
        taken[f].append((fx, fy))

    rng = np.random.default_rng(args.seed)
    fresh = np.setdiff1d(native, pick1)
    pick, flat, boxes, table, true_idx, false_idx = draw(
        rng, val, fresh, model, args.tiles, args.threshold
    )
    k = args.n // 2
    chosen = []
    for pool in (true_idx, false_idx):
        got = 0
        for d in rng.permutation(pool):
            f, fx, fy = spot(pick, flat, table, d)
            if any(math.hypot(fx - a, fy - b) < MIN_GAP for a, b in taken[f]):
                continue
            taken[f].append((fx, fy))
            chosen.append(d)
            got += 1
            if got == k:
                break
        assert got == k, "not enough separate spots"
    chosen = np.asarray(chosen)
    rng.shuffle(chosen)

    frames: dict[int, np.ndarray] = {}
    with zipfile.ZipFile(VAL_ZIP) as z:
        for f in sorted({spot(pick, flat, table, d)[0] for d in chosen}):
            with z.open(members[f]) as fh:
                arr = np.asarray(Image.open(fh).convert("RGB")).copy()
            for x, y, w, h in ignores[f]:
                arr[int(y) : int(math.ceil(y + h)), int(x) : int(math.ceil(x + w))] = (
                    r15.PAD
                )
            frames[f] = arr
    crops, truth = [], []
    for n, d in enumerate(chosen):
        tile = int(table["det_tile"][d])
        x, y, w, h, score = flat[d]
        cx, cy = x + w / 2, y + h / 2
        f, fx, fy = spot(pick, flat, table, d)
        t0 = origins[int(pick[tile])]
        src = np.asarray(val["images"][int(pick[tile])])
        assert np.array_equal(
            src[:8, :8], frames[f][t0[3] : t0[3] + 8, t0[2] : t0[2] + 8]
        ), "tile origin does not match the frame"
        views = []
        for half in (HALF, WIDE):  # both cut from the frame, as a video feed shows it
            img, px, py = frames[f], int(round(fx)), int(round(fy))
            out = np.full((2 * half, 2 * half, 3), r15.PAD, dtype=np.uint8)
            xa, ya = max(px - half, 0), max(py - half, 0)
            xb, yb = min(px + half, img.shape[1]), min(py + half, img.shape[0])
            out[
                ya - (py - half) : yb - (py - half), xa - (px - half) : xb - (px - half)
            ] = img[ya:yb, xa:xb]
            big = Image.fromarray(out).resize((SHOW, SHOW), Image.BICUBIC)
            r = RING * HALF / half  # the same ring in frame pixels
            c = SHOW / 2
            ImageDraw.Draw(big).ellipse((c - r, c - r, c + r, c + r), outline=YELLOW,
                                        width=2 if half == HALF else 1)  # fmt: skip
            views.append(big)
        g = boxes[tile]
        dist = float("inf")
        for bx, by, bw, bh in g[:, :4]:
            dx = max(bx - cx, 0.0, cx - (bx + bw))
            dy = max(by - cy, 0.0, cy - (by + bh))
            dist = min(dist, math.hypot(dx, dy))
        cid = f"k{n:03d}"
        crops.append({"id": cid, "close": jpeg(views[0]), "wide": jpeg(views[1])})
        truth.append({"id": cid, "person": bool(table["det_gt"][d] >= 0),
                      "near_person": dist <= RING_PX,
                      "dist_px": dist, "score": float(score), "size_px": float(np.sqrt(w * h)),
                      "frame": members[f].rsplit("/", 1)[-1], "fx": float(fx), "fy": float(fy),
                      "views": views})  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    sheets = OUT / "sheets"
    sheets.mkdir(exist_ok=True)
    empties = [t for t in truth if not t["person"]]
    for s in range(0, len(empties), 10):
        page = Image.new("RGB", (2 * (2 * SHOW + 12), 5 * (SHOW + 22)), (255, 255, 255))
        pen = ImageDraw.Draw(page)
        for j, t in enumerate(empties[s : s + 10]):
            ox, oy = (j % 2) * (2 * SHOW + 12), (j // 2) * (SHOW + 22)
            page.paste(t["views"][0], (ox, oy + 18))
            page.paste(t["views"][1], (ox + SHOW + 2, oy + 18))
            pen.text((ox + 2, oy + 3), t["id"], fill=(0, 0, 0))
        page.save(sheets / f"false_{s // 10:02d}.png")
    for t in truth:
        del t["views"]
    (OUT / "crops.json").write_text(json.dumps(crops), encoding="utf-8")
    (OUT / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(
        f"{len(crops)} crops ({k} person, {k} false alarm) from {len(pick)} fresh val "
        f"tiles; detections at >= {args.threshold}: {len(true_idx)} person, "
        f"{len(false_idx)} false; near a labelled person: "
        f"{sum(t['near_person'] for t in truth)} -> {OUT}"
    )


if __name__ == "__main__":
    main()
