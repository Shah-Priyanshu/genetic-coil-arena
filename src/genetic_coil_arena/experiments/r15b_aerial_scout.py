"""R15b: does the scout brain see people from a drone, and does aerial training help?

R15 trained K210-sized person detectors on COCO, whose photos are taken from the ground.
A SCOUT drone looks down from altitude, where people are small and seen from above. This
experiment measures that gap on VisDrone2019-DET (drone footage of 14 Chinese cities,
academic use; downloaded from the Ultralytics mirror of the official release) and how
much training on aerial images closes it.

Data (``prepare``). Persons are VisDrone's "pedestrian" and "people" (sitting, riding)
boxes; "ignored regions" are painted with the pad grey. Every frame is cut into
``size`` × ``size`` tiles (OpenSwarm's 224 input) at two scales: 1.0 (native pixels) and
0.5 (the frame halved first, i.e. people as from twice the height). Tiles cover the
frame on a grid whose last row and column are shifted back to the edge. A person cut by
a tile border counts if at least half of its box is inside; otherwise it is ignored
(neither a hit nor a false alarm). Training keeps every tile with a counted person and
``KEEP_EMPTY`` of the others; validation and test keep every tile.

Splits: VisDrone train (fine-tuning), val (training-time monitoring only, nothing is
selected on it) and test-dev (scored once, by the ``report`` job, after all training).

Conditions, all with R15's model and settings at widths 0.25, 0.5 and 0.75:

- **zero-shot**: R15's COCO models, unchanged (fully convolutional, so they run at 224);
- **fine-tuned**: R15's COCO weights trained further on aerial tiles;
- **scratch**: the same training from random weights (same epochs and learning rate).

Measures on test-dev tiles, per scale and pooled:

- AP50 (box IoU at least 0.5);
- centre AP: a detection hits a person when its centre lies inside the person's box,
  grown to at least 8 × 8 px. This is what the DETECTION packet needs (``cx, cy``);
- both by person size (square root of box area in tile pixels: < 8, 8-16, 16-32, ≥ 32),
  matching once and then stratifying, COCO-style: detections matched to a person outside
  the bin are ignored, and unmatched ones count only if their own size is in the bin;
- person-present AUC per tile (any counted person).

Matching is greedy by score, each detection taking the best unused counted person first
and an ignored one second (COCO's rule; R15's ``ap50`` is VOC's and is kept for COCO).
95% intervals bootstrap over source frames (tiles of one frame move together); model
differences are paired on the same resamples.

Bars (fixed before training):

1. **Aerial training helps:** at width 0.5 the fine-tuned brain's AP50 on test-dev
   (scales pooled) beats zero-shot by at least 10 points, with the paired interval above
   0 for every seed.
2. **COCO is worth starting from:** at width 0.5 fine-tuned beats scratch on AP50, the
   paired interval above 0 for every seed.
"""

from __future__ import annotations

import json
import math
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from . import r15_scout_brain as r15

SCALES = (1.0, 0.5)
KEEP_EMPTY = 0.2
PERSON_CATS = (1, 2)
IGNORE_CAT = 0
SIZE_BINS = ((0.0, 8.0), (8.0, 16.0), (16.0, 32.0), (32.0, math.inf))
MIN_HIT = 4.0  # centre hit: half-width of the smallest accepted box
SPLITS = {"train": "train", "val": "val", "test": "test-dev"}
GOOD_POINTS = 0.10
THRESHOLDS = (0.3, 0.5, 0.7)  # 0.7 = OpenSwarm's person_follow confidence threshold
BIN_NAMES = [("all", 0.0, math.inf)] + [
    (f"{lo:g}-{hi:g}" if hi < math.inf else f"{lo:g}+", lo, hi) for lo, hi in SIZE_BINS
]


# ------------------------------------------------------------------ data
def parse_annotations(text: str) -> tuple[np.ndarray, np.ndarray]:
    """VisDrone lines -> (persons (M, 4), ignore regions (K, 4)) as x, y, w, h."""
    persons, ignore = [], []
    for line in text.strip().splitlines():
        v = [int(x) for x in line.strip().strip(",").split(",")[:6]]
        if len(v) < 6 or v[2] < 1 or v[3] < 1:
            continue
        if v[5] in PERSON_CATS and v[4] == 1:
            persons.append(v[:4])
        elif v[5] == IGNORE_CAT:
            ignore.append(v[:4])
    return (
        np.asarray(persons, np.float32).reshape(-1, 4),
        np.asarray(ignore, np.float32).reshape(-1, 4),
    )


def tile_starts(length: int, size: int) -> list[int]:
    """Grid starts covering ``length``; the last tile is shifted back to the edge."""
    if length <= size:
        return [0]
    n = math.ceil(length / size)
    return [min(i * size, length - size) for i in range(n)]


def clip_boxes(boxes: np.ndarray, x0: float, y0: float, size: int) -> np.ndarray:
    """(M, 5) boxes clipped to the tile, in tile pixels, with column 4 = counted (at
    least half the box inside); boxes entirely outside are dropped."""
    if not len(boxes):
        return np.zeros((0, 5), np.float32)
    x1, y1 = boxes[:, 0] - x0, boxes[:, 1] - y0
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    cx1, cy1 = np.clip(x1, 0, size), np.clip(y1, 0, size)
    cx2, cy2 = np.clip(x2, 0, size), np.clip(y2, 0, size)
    inside = (cx2 - cx1) * (cy2 - cy1)
    frac = inside / (boxes[:, 2] * boxes[:, 3])
    ok = (cx2 - cx1 >= 1) & (cy2 - cy1 >= 1)
    out = np.stack(
        [cx1, cy1, cx2 - cx1, cy2 - cy1, (frac >= 0.5).astype(np.float32)], 1
    )
    return out[ok].astype(np.float32)


def plan_tiles(
    w: int,
    h: int,
    persons: np.ndarray,
    size: int,
    keep_empty: float,
    rng: np.random.Generator,
) -> list[tuple[float, int, int, np.ndarray]]:
    """(scale, x0, y0, boxes (M, 5)) for every kept tile of a ``w`` × ``h`` frame."""
    out = []
    for s in SCALES:
        sw, sh = max(1, round(w * s)), max(1, round(h * s))
        for y0 in tile_starts(sh, size):
            for x0 in tile_starts(sw, size):
                b = clip_boxes(persons * s, x0, y0, size)
                if keep_empty < 1 and not b[:, 4].any() and rng.random() >= keep_empty:
                    continue
                out.append((s, x0, y0, b))
    return out


def _render(
    job: tuple[str, str, list[tuple[float, int, int]], np.ndarray, int],
) -> np.ndarray:
    from PIL import Image

    zpath, member, tiles, ignore, size = job
    if zpath not in r15._ZIP:
        r15._ZIP[zpath] = zipfile.ZipFile(zpath)
    with r15._ZIP[zpath].open(member) as f:
        img = Image.open(f).convert("RGB")
    w, h = img.size
    out = np.full((len(tiles), size, size, 3), r15.PAD, dtype=np.uint8)
    cache: dict[float, np.ndarray] = {}
    for k, (s, x0, y0) in enumerate(tiles):
        if s not in cache:
            sw, sh = max(1, round(w * s)), max(1, round(h * s))
            arr = np.asarray(img.resize((sw, sh), Image.BILINEAR)).copy()
            for x, y, bw, bh in ignore * s:
                arr[
                    int(y) : int(math.ceil(y + bh)), int(x) : int(math.ceil(x + bw))
                ] = r15.PAD
            cache[s] = arr
        arr = cache[s]
        crop = arr[y0 : y0 + size, x0 : x0 + size]
        out[k, : crop.shape[0], : crop.shape[1]] = crop
    return out


def prepare(
    vd_dir: Path,
    out_dir: Path,
    size: int = 224,
    workers: int = 6,
    limit: int = 0,
    seed: int = 15,
) -> dict[str, Any]:
    """Tiles, boxes (x, y, w, h, counted), offsets, source frame and scale per split."""
    from multiprocessing import Pool

    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {
        "size": size,
        "stride": r15.STRIDE,
        "scales": SCALES,
        "keep_empty": KEEP_EMPTY,
        "splits": {},
    }
    for split, name in SPLITS.items():
        start = time.time()
        zpath = str(vd_dir / f"VisDrone2019-DET-{name}.zip")
        rng = np.random.default_rng([seed, len(split)])
        with zipfile.ZipFile(zpath) as z:
            anns = sorted(
                n for n in z.namelist() if n.endswith(".txt") and "/annotations/" in n
            )
            if limit:
                anns = anns[:limit]
            jobs, boxes, offsets, src, scale, frames = [], [], [0], [], [], []
            for i, a in enumerate(anns):
                member = a.replace("/annotations/", "/images/")[:-4] + ".jpg"
                with z.open(member) as f:
                    w, h = Image.open(f).size
                persons, ignore = parse_annotations(z.read(a).decode())
                keep = KEEP_EMPTY if split == "train" else 1.0
                plan = plan_tiles(w, h, persons, size, keep, rng)
                jobs.append(
                    (zpath, member, [(s, x, y) for s, x, y, _ in plan], ignore, size)
                )
                for s, _, _, b in plan:
                    boxes.append(b)
                    offsets.append(offsets[-1] + len(b))
                    src.append(i)
                    scale.append(s)
                frames.append(member.rsplit("/", 1)[-1])
        arr = np.lib.format.open_memmap(
            out_dir / f"{split}_images.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(len(src), size, size, 3),
        )
        pos = 0
        with Pool(workers) as pool:
            for tiles in pool.imap(_render, jobs, chunksize=8):
                arr[pos : pos + len(tiles)] = tiles
                pos += len(tiles)
        arr.flush()
        del arr
        np.save(
            out_dir / f"{split}_boxes.npy",
            np.concatenate(boxes).astype(np.float32).reshape(-1, 5),
        )
        np.save(out_dir / f"{split}_offsets.npy", np.asarray(offsets, np.int64))
        np.save(out_dir / f"{split}_src.npy", np.asarray(src, np.int64))
        np.save(out_dir / f"{split}_scale.npy", np.asarray(scale, np.float32))
        (out_dir / f"{split}_frames.json").write_text(
            json.dumps(frames), encoding="utf-8"
        )
        all_boxes = np.concatenate(boxes).reshape(-1, 5)
        info["splits"][split] = {
            "frames": len(anns),
            "tiles": len(src),
            "tiles_with_person": int(sum(b[:, 4].any() for b in boxes)),
            "counted_persons": int(all_boxes[:, 4].sum()),
            "seconds": round(time.time() - start, 1),
        }
    (out_dir / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def load_tiles(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    d = r15.load_split(data_dir, split)
    missing = [k for k in r15.EXTRAS if k not in d]
    if missing:
        raise FileNotFoundError(f"{data_dir} has no {split} {missing}: run prepare")
    return d


# ------------------------------------------------------------------ matching
def _sim(p: np.ndarray, g: np.ndarray, mode: str) -> np.ndarray:
    if mode == "iou":
        m = r15.iou(p[:, :4], g[:, :4])
        return np.where(m >= 0.5, m, 0.0)
    pcx, pcy = p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2
    gcx, gcy = g[:, 0] + g[:, 2] / 2, g[:, 1] + g[:, 3] / 2
    dx = np.abs(pcx[:, None] - gcx[None])
    dy = np.abs(pcy[:, None] - gcy[None])
    hit = (dx <= np.maximum(g[:, 2] / 2, MIN_HIT)[None]) & (
        dy <= np.maximum(g[:, 3] / 2, MIN_HIT)[None]
    )
    return np.where(hit, 1.0 / (1.0 + np.hypot(dx, dy)), 0.0)


def match(p: np.ndarray, g: np.ndarray, mode: str) -> np.ndarray:
    """Index of the person each detection (score order as given) hits, or -1. Greedy by
    score; a detection takes the best unused counted person, else the best unused
    ignored one (COCO's rule)."""
    out = np.full(len(p), -1, dtype=np.int64)
    if not len(p) or not len(g):
        return out
    sim = _sim(p, g, mode)
    counted = g[:, 4] > 0
    used = np.zeros(len(g), dtype=bool)
    for j in np.argsort(-p[:, 4], kind="stable"):
        for pool in (counted, ~counted):
            cand = np.where(pool & ~used, sim[j], 0.0)
            k = int(np.argmax(cand))
            if cand[k] > 0:
                used[k] = True
                out[j] = k
                break
    return out


def size_of(boxes: np.ndarray) -> np.ndarray:
    return np.sqrt(boxes[:, 2] * boxes[:, 3])


def hit_table(preds: list[np.ndarray], gts: list[np.ndarray], mode: str) -> dict:
    """Flat per-detection and per-person arrays for fast (bootstrapped) AP."""
    tile_d, score, det_size, gt_of = [], [], [], []
    tile_g, g_size, g_counted = [], [], []
    base = 0
    for t, (p, g) in enumerate(zip(preds, gts)):
        m = match(p, g, mode)
        tile_d.append(np.full(len(p), t))
        score.append(p[:, 4])
        det_size.append(size_of(p))
        gt_of.append(np.where(m >= 0, m + base, -1))
        base += len(g)
        tile_g.append(np.full(len(g), t))
        g_size.append(size_of(g))
        g_counted.append(g[:, 4] > 0)
    cat = np.concatenate
    return {
        "det_tile": cat(tile_d).astype(np.int64),
        "score": cat(score),
        "det_size": cat(det_size),
        "det_gt": cat(gt_of).astype(np.int64),
        "gt_tile": cat(tile_g).astype(np.int64),
        "gt_size": cat(g_size),
        "gt_counted": cat(g_counted),
    }


def ap_from(score: np.ndarray, tp: np.ndarray, n_gt: float) -> float:
    """All-point interpolated AP from per-detection scores and hits."""
    if n_gt <= 0:
        return float("nan")
    if not len(score):
        return 0.0
    order = np.argsort(-score, kind="stable")
    hits = tp[order].astype(np.float64)
    ctp = np.cumsum(hits)
    rec = ctp / n_gt
    prec = ctp / np.arange(1, len(hits) + 1)
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mrec = np.concatenate([[0.0], rec, [rec[-1]]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    return float(((mrec[1:] - mrec[:-1]) * mpre[1:]).sum())


def ap_parts(
    table: dict, lo: float = 0.0, hi: float = math.inf, tiles: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Per-detection (counts, is TP) and per-person (counts) masks for one size bin,
    restricted to the tiles where ``tiles`` (per tile) is true."""
    gt_in = table["gt_counted"] & (table["gt_size"] >= lo) & (table["gt_size"] < hi)
    det_on = np.ones(len(table["score"]), bool)
    if tiles is not None:
        gt_in &= tiles[table["gt_tile"]]
        det_on = tiles[table["det_tile"]]
    matched = table["det_gt"] >= 0
    tgt = np.where(matched, table["det_gt"], 0)
    tp = matched & gt_in[tgt]
    fp = ~matched & (table["det_size"] >= lo) & (table["det_size"] < hi)
    return {"det": (tp | fp) & det_on, "tp": tp, "gt": gt_in}


def ap_bin(
    table: dict,
    lo: float = 0.0,
    hi: float = math.inf,
    weights: tuple[np.ndarray, np.ndarray] | None = None,
    tiles: np.ndarray | None = None,
) -> float:
    """AP for persons with size in [lo, hi) on the selected ``tiles``; ``weights`` =
    (per-detection, per-person) multiplicities from a bootstrap resample."""
    part = ap_parts(table, lo, hi, tiles)
    if weights is None:
        sel = part["det"]
        return ap_from(table["score"][sel], part["tp"][sel], float(part["gt"].sum()))
    wd, wg = weights
    sel = part["det"] & (wd > 0)
    rep = wd[sel]
    return ap_from(
        np.repeat(table["score"][sel], rep),
        np.repeat(part["tp"][sel], rep),
        float((wg * part["gt"]).sum()),
    )


def resample_weights(
    frames: np.ndarray, rng: np.random.Generator, n: int
) -> list[np.ndarray]:
    """Per-frame multiplicities of ``n`` bootstrap resamples over source frames."""
    uniq = np.unique(frames)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        w = np.zeros(int(uniq.max()) + 1, np.int64)
        np.add.at(w, uniq[pick], 1)
        out.append(w)
    return out


# ------------------------------------------------------------------ evaluation
def input_fn(dev: Any) -> Any:
    import torch

    mean = torch.tensor([123.7, 116.3, 103.5], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([58.4, 57.1, 57.4], device=dev).view(1, 3, 1, 1)

    def to_input(imgs: np.ndarray) -> Any:
        x = torch.from_numpy(np.ascontiguousarray(imgs)).to(dev).permute(0, 3, 1, 2)
        return (x.float() - mean) / std

    return to_input


def predict(
    model: Any,
    split: dict,
    size: int,
    dev: Any,
    to_input: Any,
    batch: int = 256,
    top: int = 100,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Per-tile detections (x, y, w, h, score; at most ``top``, crowded aerial tiles
    hold dozens of people) and top heatmap score."""
    import torch

    model.eval()
    preds, best = [], []
    n = len(split["offsets"]) - 1
    with torch.no_grad():
        for a in range(0, n, batch):
            imgs = np.asarray(split["images"][a : a + batch])
            out = model(to_input(imgs)).float()
            heat = torch.sigmoid(out[:, 0]).cpu().numpy()
            preds += r15.decode(
                heat,
                out[:, 1:3].cpu().numpy(),
                torch.sigmoid(out[:, 3:5]).cpu().numpy(),
                top=top,
            )
            best.append(heat.reshape(len(heat), -1).max(1))
    return preds, np.concatenate(best)


def tile_metrics(
    preds: list[np.ndarray],
    top: np.ndarray,
    split: dict,
    n_boot: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """AP50 and centre AP (pooled, per scale, per size bin) and present AUC; with
    ``n_boot`` the per-resample values are kept under ``boot`` for paired differences.
    """
    gts = [r15.image_boxes(split, i) for i in range(len(preds))]
    frames, scales = split["src"], split["scale"]
    present = np.array([bool(g[:, 4].any()) for g in gts])
    tables = {mode: hit_table(preds, gts, mode) for mode in ("iou", "centre")}
    groups = {"pooled": np.ones(len(preds), bool)}
    groups.update({f"scale {s:g}": scales == s for s in SCALES})
    weights = resample_weights(frames, np.random.default_rng(seed), n_boot)
    out: dict[str, Any] = {"tiles": len(preds), "present_rate": float(present.mean())}
    for gname, gmask in groups.items():
        res: dict[str, Any] = {"present_auc": r15.auc(top[gmask], present[gmask])}
        for mode, t in tables.items():
            key = "ap50" if mode == "iou" else "centre_ap"
            for bname, lo, hi in BIN_NAMES:
                res[f"{key} {bname}"] = ap_bin(t, lo, hi, tiles=gmask)
                res[f"n_persons {bname}"] = int(ap_parts(t, lo, hi, gmask)["gt"].sum())
            if n_boot:
                df, gf = frames[t["det_tile"]], frames[t["gt_tile"]]
                res[f"boot {key}"] = [
                    ap_bin(t, weights=(w[df], w[gf]), tiles=gmask) for w in weights
                ]
        res["operating"] = operating_points(tables["centre"], top, present, gmask)
        out[gname] = res
    return out


def operating_points(
    table: dict, top: np.ndarray, present: np.ndarray, tiles: np.ndarray
) -> dict[str, dict[str, float]]:
    """What a scout sees at a fixed confidence threshold (centre hits): recall per person
    size bin, false detections per tile, and per tile the share of tiles with a person
    whose top score passes (hit rate) and of empty tiles that pass (false-alarm rate).
    Greedy matching runs in score order, so the matches above a threshold are those a
    run on the detections above it would make."""
    out = {}
    det_on = tiles[table["det_tile"]]
    for tau in THRESHOLDS:
        keep = det_on & (table["score"] >= tau)
        hit_gt = np.zeros(len(table["gt_size"]), bool)
        hit_gt[table["det_gt"][keep & (table["det_gt"] >= 0)]] = True
        row: dict[str, float] = {}
        for bname, lo, hi in BIN_NAMES:
            gt_in = ap_parts(table, lo, hi, tiles)["gt"]
            row[f"recall {bname}"] = (
                float(hit_gt[gt_in].mean()) if gt_in.any() else float("nan")
            )
        n_tiles = int(tiles.sum())
        false = int((keep & (table["det_gt"] < 0)).sum())
        row["false_per_tile"] = false / n_tiles if n_tiles else float("nan")
        pos, neg = present & tiles, ~present & tiles
        row["tile_hit_rate"] = float((top[pos] >= tau).mean()) if pos.any() else 0.0
        row["tile_false_alarm_rate"] = (
            float((top[neg] >= tau).mean()) if neg.any() else 0.0
        )
        out[f"{tau:g}"] = row
    return out


def val_metrics(
    model: Any, va: dict, size: int, dev: Any, to_input: Any
) -> dict[str, Any]:
    """Training-time monitoring on VisDrone val (``train_model``'s ``eval_fn``)."""
    preds, best = predict(model, va, size, dev, to_input)
    m = tile_metrics(preds, best, va)["pooled"]
    return {
        "val_ap50": m["ap50 all"],
        "val_centre_ap": m["centre_ap all"],
        "val_present_auc": m["present_auc"],
    }


def paired(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Mean and 95% percentile interval of the paired bootstrap difference a - b
    (resamples without persons, nan, are skipped)."""
    d = np.asarray(a) - np.asarray(b)
    d = d[~np.isnan(d)]
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ------------------------------------------------------------------ verdict and report
def _find(runs: list[dict], condition: str, alpha: float) -> list[dict]:
    return sorted(
        (r for r in runs if r["condition"] == condition and r["alpha"] == alpha),
        key=lambda r: r["seed"],
    )


def _diff(a: dict, b: dict, key: str = "ap50") -> dict[str, float]:
    """Observed pooled difference a - b and its paired bootstrap interval."""
    pa, pb = a["test"]["pooled"], b["test"]["pooled"]
    _, lo, hi = paired(pa[f"boot {key}"], pb[f"boot {key}"])
    return {"diff": pa[f"{key} all"] - pb[f"{key} all"], "lo": lo, "hi": hi}


def verdict(runs: list[dict]) -> dict[str, Any]:
    """The two pre-registered bars at width 0.5."""
    zs, fts = _find(runs, "zero-shot", 0.5), _find(runs, "fine-tuned", 0.5)
    scratch = {r["seed"]: r for r in _find(runs, "scratch", 0.5)}
    bar1 = [dict(seed=f["seed"], **_diff(f, zs[0])) for f in fts] if zs else []
    bar2 = [
        dict(seed=f["seed"], **_diff(f, scratch[f["seed"]]))
        for f in fts
        if f["seed"] in scratch
    ]
    return {
        "aerial_training_helps": bar1,
        "aerial_training_helps_pass": bool(bar1)
        and all(d["diff"] >= GOOD_POINTS and d["lo"] > 0 for d in bar1),
        "coco_start_helps": bar2,
        "coco_start_helps_pass": bool(bar2) and all(d["lo"] > 0 for d in bar2),
    }


def _avg(rs: list[dict], get: Any) -> str:
    vals = [get(r) for r in rs]
    txt = f"{np.mean(vals):.3f}"
    if len(vals) == 1:
        return txt
    return txt + " (" + ", ".join(f"{v:.3f}" for v in vals) + ")"


def _bar_line(name: str, ok: bool, what: str, diffs: list[dict], bar: str) -> str:
    parts = "; ".join(
        f"seed {d['seed']}: {100 * d['diff']:+.1f} points "
        f"[{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"
        for d in diffs
    )
    return f"**{name}: {'PASS' if ok else 'FAIL'}.** {what}: {parts} ({bar})."


def render(rep: dict[str, Any], runs: list[dict], test_frames: int,
           size: int = 224) -> str:  # fmt: skip
    conds = ("zero-shot", "fine-tuned", "scratch")
    alphas = sorted({r["alpha"] for r in runs})

    def pooled(key: str) -> Any:
        return lambda r: r["test"]["pooled"][key]

    def scaled(s: float, key: str) -> Any:
        return lambda r: r["test"][f"scale {s:g}"][f"{key} all"]

    lines = [
        "# R15b: the scout brain from a drone's view",
        "",
        _bar_line(
            "Bar 1, aerial training helps",
            rep["aerial_training_helps_pass"],
            "Width 0.5, fine-tuned minus zero-shot AP50",
            rep["aerial_training_helps"],
            f"bar: at least +{100 * GOOD_POINTS:.0f}, interval above 0",
        ),
        "",
        _bar_line(
            "Bar 2, COCO is worth starting from",
            rep["coco_start_helps_pass"],
            "Width 0.5, fine-tuned minus scratch AP50",
            rep["coco_start_helps"],
            "bar: interval above 0",
        ),
        "",
        f"VisDrone2019-DET test-dev: {test_frames} drone frames cut into {size}×{size} tiles "
        "at native and half resolution, scored once after all training. Intervals: 95% "
        "paired bootstrap over frames. Centre AP counts a detection when its centre "
        "falls inside the person's box (grown to at least 8×8 px), which is what "
        "OpenSwarm's DETECTION packet carries. Several seeds are shown as mean (seed "
        "values).",
        "",
        "| Brain | Width | int8 weights | AP50 | Centre AP | Present AUC "
        "| COCO val AP50 (ground photos) |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in conds:
        for a in alphas:
            rs = _find(runs, c, a)
            if not rs:
                continue
            kb = rs[0].get("int8_kb")
            lines.append(
                f"| {c} | {a:g} | {f'{kb:.0f} KB' if kb else '-'} "
                f"| {_avg(rs, pooled('ap50 all'))} "
                f"| {_avg(rs, pooled('centre_ap all'))} "
                f"| {_avg(rs, pooled('present_auc'))} "
                f"| {_avg(rs, lambda r: r['coco_val']['ap50'])} |"
            )
    bins = [b[0] for b in BIN_NAMES if b[0] != "all"]
    counts = runs[0]["test"]["pooled"]
    for key, label in (("centre_ap", "Centre AP"), ("ap50", "AP50")):
        lines += [
            "",
            f"{label} by person size (square root of box area, in pixels of the {size} "
            "tile; persons per bin in the header):",
            "",
            "| Brain | Width | "
            + " | ".join(f"{b} px ({counts[f'n_persons {b}']:,})" for b in bins)
            + " |",
            "|---|---|" + "---|" * len(bins),
        ]
        for c in conds:
            for a in alphas:
                rs = _find(runs, c, a)
                if rs:
                    cells = [_avg(rs, pooled(f"{key} {b}")) for b in bins]
                    lines.append(f"| {c} | {a:g} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "By tile scale (1 = native VisDrone pixels; 0.5 = frame halved, people as from "
        "twice the height):",
        "",
        "| Brain | Width | AP50, scale 1 | AP50, scale 0.5 | Centre AP, scale 1 "
        "| Centre AP, scale 0.5 |",
        "|---|---|---|---|---|---|",
    ]
    for c in conds:
        for a in alphas:
            rs = _find(runs, c, a)
            if rs:
                lines.append(
                    f"| {c} | {a:g} | {_avg(rs, scaled(1, 'ap50'))} "
                    f"| {_avg(rs, scaled(0.5, 'ap50'))} "
                    f"| {_avg(rs, scaled(1, 'centre_ap'))} "
                    f"| {_avg(rs, scaled(0.5, 'centre_ap'))} |"
                )
    lines += [
        "",
        "What a scout sees at a fixed confidence threshold (centre hits, scales "
        "pooled): recall by person size, false detections per tile, and the share of "
        "empty tiles that raise an alarm. 0.7 is OpenSwarm's person-follow threshold.",
        "",
        "| Brain | Width | Threshold | "
        + " | ".join(f"recall {b} px" for b in bins)
        + " | False detections per tile | Empty tiles alarmed |",
        "|---|---|---|" + "---|" * (len(bins) + 2),
    ]
    for c in conds:
        for a in alphas:
            rs = _find(runs, c, a)
            if not rs or "operating" not in rs[0]["test"]["pooled"]:
                continue
            for tau in ("0.5", "0.7"):

                def op(key: str, tau: str = tau) -> Any:
                    return lambda r: r["test"]["pooled"]["operating"][tau][key]

                cells = [_avg(rs, op(f"recall {b}")) for b in bins]
                cells += [_avg(rs, op("false_per_tile")),
                          _avg(rs, op("tile_false_alarm_rate"))]  # fmt: skip
                lines.append(f"| {c} | {a:g} | {tau} | " + " | ".join(cells) + " |")
    trained = [r for r in runs if r["condition"] != "zero-shot"]
    if trained:
        s = trained[0]
        lines += [
            "",
            f"Training: {s['epochs']} epochs on {s['train_images']:,} VisDrone train "
            f"tiles (every tile with a person, {100 * KEEP_EMPTY:.0f}% of the others), "
            f"batch 128, AdamW one-cycle, peak learning rate {s['lr']:g}, the same "
            "settings for fine-tuned and scratch. Zero-shot = R15's COCO models "
            "(trained at 160, run at 224).",
        ]
    lines += [
        "",
        "Reproduce: `python scripts/r15b_aerial_scout.py prepare`, `train --init coco` "
        "and `--init scratch`, then `report --from runs:r15b2-train-*`.",
        "",
    ]
    if size != 224:  # R15c probe: no bars; the title and the bar lines are replaced
        lines = [
            f"# R15c probe: the scout brain at a {size}×{size} input",
            "",
            "No pre-registered bars. The question: is recall at a given person size in "
            "pixels the same at this input as at 224? If it is (within about 5 points "
            "per size bin at threshold 0.5), pixels on the person summarise a camera "
            "and its resolution; compare the operating-point table with R15b's.",
            "",
        ] + lines[6:]
    return "\n".join(lines)
