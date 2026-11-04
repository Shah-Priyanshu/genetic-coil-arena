"""R22: does the scout brain learn better from labels a confirmer has cleaned?

R20b found that most of what VisDrone calls false alarms are people the labels miss: of
594 passed "false alarms" checked blind, 457 were people. R15b's scout brain was trained
on those labels, so part of its training taught it that people are background. This
module finds likely unlabelled people in the training frames with an out-of-fold
confirmer, adds them as person boxes and retrains the scout with R15b's recipe.

1. ``folds``: VisDrone train's 208 sequences split in two at random (``FOLD_SEED``); a
   sequence's frames look alike, so folds never share one.
2. ``fit``/``score`` per fold: DINOv2-small fully fine-tuned (R19's recipe) on R19's train
   candidates of the *other* fold, epoch picked on R19's val candidates, then scored on
   this fold's candidates: every train candidate gets a score from a model that never
   saw its sequence.
3. ``pseudo``: candidates VisDrone labels false alarms, with the centre outside every
   person box (R19's ``dist`` > 0), whose out-of-fold score passes at least ``KEEP_PSEUDO``
   of that fold's labelled people's threshold, become person boxes: a square of the
   scout's box size around the candidate's centre.
4. ``tiles``: R15b's training tiles with those boxes added (same tiles, same images; val
   linked unchanged), in ``visdrone_tiles224_clean``; then R15b's ``train`` with
   ``--tiles-suffix _clean``.
5. ``detect``: each scout over every val and test-dev frame at native scale (R19's way).
   Thresholds on val: each scout keeps the share of labelled val people that the control
   seed 0 keeps at OpenSwarm's 0.5. ``sample``: a random ``SAMPLE_N`` of each scout's
   test-dev false alarms (VisDrone labels) on blind sheets, scout hidden.
"""

from __future__ import annotations

import json
import math
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from . import r15_scout_brain as r15
from . import r15b_aerial_scout as r15b
from . import r19_confirmer as r19

OUT_DIR = r15.DATA_DIR / "r22_clean_labels"
TILES = r15.DATA_DIR / "visdrone_tiles224"
CLEAN_TILES = r15.DATA_DIR / "visdrone_tiles224_clean"
FOLD_SEED = 22
KEEP_PSEUDO = 0.80  # the pseudo-label threshold passes this share of the fold's people
BASE_THRESHOLD = 0.5  # OpenSwarm's: the control seed 0's people share sets the others
SAMPLE_N = 150
MODELS = {"control_s0": "r15b_coco_a0.5_seed0", "control_s1": "r15b_coco_a0.5_seed1",
          "clean_s0": "r15b_coco_a0.5_seed0_clean", "clean_s1": "r15b_coco_a0.5_seed1_clean"}  # fmt: skip
RATIO_PASS, RATIO_SEED, RATIO_KILL, RECALL_DROP = 0.7, 0.8, 0.9, 0.02
CLASSES = ("person", "not_person", "ambiguous")


def sequence(name: str) -> str:
    return name.split("_")[0]


def folds(names: np.ndarray) -> np.ndarray:
    """Fold (0 or 1) of each candidate, by its frame's sequence."""
    seqs = sorted({sequence(n) for n in names})
    perm = np.random.default_rng(FOLD_SEED).permutation(len(seqs))
    fold = {s: int(k < len(seqs) // 2) for s, k in zip(seqs, perm)}
    return np.array([fold[sequence(n)] for n in names])


def candidate_folds(meta: dict[str, np.ndarray]) -> np.ndarray:
    return folds(meta["names"])[meta["frame"]]


def pseudo_mask(meta: dict[str, np.ndarray], score: np.ndarray,
                fold: np.ndarray) -> tuple[np.ndarray, dict[int, float]]:  # fmt: skip
    """Candidates that become person boxes, and each fold's threshold."""
    person, out, thr = meta["person"].astype(bool), np.zeros(len(score), bool), {}
    for f in (0, 1):
        sel = fold == f
        if not (sel & person).any():
            continue
        thr[f] = r19.threshold_at(score[sel & person], KEEP_PSEUDO)
        out |= sel & ~person & (meta["dist"] > 0) & (score >= thr[f])
    return out, thr


def pseudo_boxes(meta: dict[str, np.ndarray], mask: np.ndarray) -> dict[int, np.ndarray]:
    """Frame index -> (K, 4) square boxes x, y, w, h in native pixels."""
    out: dict[int, list] = {}
    for i in np.flatnonzero(mask):
        s = float(meta["size"][i])
        out.setdefault(int(meta["frame"][i]), []).append(
            [meta["fx"][i] - s / 2, meta["fy"][i] - s / 2, s, s])
    return {k: np.asarray(v, np.float32) for k, v in out.items()}


def tile_plan(size: int = 224, seed: int = 15) -> list[tuple[int, float, int, int]]:
    """(frame, scale, x0, y0) of every R15b train tile, replaying ``r15b.prepare``'s
    random choice of empty tiles."""
    from PIL import Image

    rng = np.random.default_rng([seed, len("train")])
    out = []
    with zipfile.ZipFile(r19.det_zip("train")) as z:
        anns = sorted(n for n in z.namelist() if n.endswith(".txt") and "/annotations/" in n)
        for i, a in enumerate(anns):
            member = a.replace("/annotations/", "/images/")[:-4] + ".jpg"
            with z.open(member) as f:
                w, h = Image.open(f).size
            persons, _ = r15b.parse_annotations(z.read(a).decode())
            for s, x0, y0, _ in r15b.plan_tiles(w, h, persons, size, r15b.KEEP_EMPTY, rng):
                out.append((i, s, x0, y0))
    return out


def build_clean_tiles(boxes: dict[int, np.ndarray], size: int = 224,
                      log: Any = print) -> dict[str, Any]:  # fmt: skip
    """``CLEAN_TILES``: R15b's tiles (hard links), train boxes with ``boxes`` added."""
    plan = tile_plan(size)
    src = np.load(TILES / "train_src.npy")
    old_boxes, old_off = np.load(TILES / "train_boxes.npy"), np.load(TILES / "train_offsets.npy")
    if len(plan) != len(src) or any(p[0] != s for p, s in zip(plan, src)):
        raise RuntimeError("replayed tile plan does not match R15b's train tiles")
    CLEAN_TILES.mkdir(parents=True, exist_ok=True)
    for f in TILES.iterdir():
        dst = CLEAN_TILES / f.name
        if f.name not in ("train_boxes.npy", "train_offsets.npy", "info.json") and not dst.exists():
            os.link(f, dst)
    new, off, added, tiles_hit = [], [0], 0, 0
    for k, (i, s, x0, y0) in enumerate(plan):
        b = old_boxes[old_off[k] : old_off[k + 1]]
        if i in boxes:
            extra = r15b.clip_boxes(boxes[i] * s, x0, y0, size)
            if len(extra):
                b = np.concatenate([b, extra])
                added += int(extra[:, 4].sum())
                tiles_hit += 1
        new.append(b)
        off.append(off[-1] + len(b))
    np.save(CLEAN_TILES / "train_boxes.npy", np.concatenate(new).astype(np.float32))
    np.save(CLEAN_TILES / "train_offsets.npy", np.asarray(off, np.int64))
    info = json.loads((TILES / "info.json").read_text("utf-8"))
    info["r22"] = {"pseudo_frames": len(boxes), "pseudo_boxes": int(sum(len(v) for v in boxes.values())),
                   "counted_boxes_added": added, "tiles_with_added": tiles_hit}  # fmt: skip
    (CLEAN_TILES / "info.json").write_text(json.dumps(info, indent=1), "utf-8")
    log(f"clean tiles: {info['r22']}")
    return info["r22"]


# ----------------------------------------------------------------------------- scouts


def load_scout(tag: str, device: str = "cuda") -> tuple[Any, Any, Any]:
    import torch

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)
    model = r15.build_model(0.5)
    model.load_state_dict(torch.load(r15.DATA_DIR / "r15_models" / f"{tag}.pt",
                                     map_location="cpu"))  # fmt: skip
    return model.to(dev).eval(), dev, r15b.input_fn(dev)


def detect(tag: str, split: str, log: Any = print) -> dict[str, np.ndarray]:
    """Every frame of ``split``: detections (score >= the video module's minimum) with
    their frame, and the person boxes, as flat arrays with offsets."""
    from . import r15g_video as rg

    model, dev, to_input = load_scout(tag)
    dets, d_off, persons, p_off = [], [0], [], [0]
    for i, _, img, p in r19.frames(split):
        d = rg.detect_frame(model, dev, to_input, img)
        dets.append(d)
        d_off.append(d_off[-1] + len(d))
        persons.append(p)
        p_off.append(p_off[-1] + len(p))
        if (i + 1) % 200 == 0:
            log(f"[progress] {tag} {split}: {i + 1} frames")
    return {"dets": np.concatenate(dets).reshape(-1, 5), "d_off": np.asarray(d_off),
            "persons": np.concatenate(persons).reshape(-1, 4), "p_off": np.asarray(p_off)}  # fmt: skip


def per_frame(d: dict[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(d["dets"][d["d_off"][k] : d["d_off"][k + 1]],
             d["persons"][d["p_off"][k] : d["p_off"][k + 1]])
            for k in range(len(d["d_off"]) - 1)]  # fmt: skip


def outcomes(d: dict[str, np.ndarray], threshold: float) -> dict[str, np.ndarray]:
    """At ``threshold``: per detection kept, its frame, whether it is a VisDrone false
    alarm, its centre; per frame, people and people found (R15b's centre rule)."""
    frame, fa, cx, cy, n_p, n_hit = [], [], [], [], [], []
    for k, (dt, p) in enumerate(per_frame(d)):
        dt = dt[dt[:, 4] >= threshold]
        g = np.column_stack([p, np.ones(len(p), np.float32)])
        hit = r15b.match(dt, g, "centre")
        frame += [k] * len(dt)
        fa += list(hit < 0)
        cx += list(dt[:, 0] + dt[:, 2] / 2)
        cy += list(dt[:, 1] + dt[:, 3] / 2)
        n_p.append(len(p))
        n_hit.append(int((hit >= 0).sum()))
    return {"frame": np.asarray(frame, np.int64), "fa": np.asarray(fa, bool),
            "cx": np.asarray(cx, np.float32), "cy": np.asarray(cy, np.float32),
            "people": np.asarray(n_p), "found": np.asarray(n_hit)}  # fmt: skip


def recall(d: dict[str, np.ndarray], threshold: float) -> float:
    o = outcomes(d, threshold)
    return float(o["found"].sum() / max(o["people"].sum(), 1))


def matched_threshold(d: dict[str, np.ndarray], target: float) -> float:
    """The highest threshold (0.005 grid) at which the scout finds ``target`` of the
    labelled people."""
    lo, hi = 0.05, 0.95
    for _ in range(12):  # recall falls as the threshold rises
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if recall(d, mid) >= target else (lo, mid)
    return math.floor(lo / 0.005) * 0.005


def sample_spots(fas: dict[str, np.ndarray], n: int = SAMPLE_N,
                 seed: int = FOLD_SEED) -> list[tuple[str, int]]:  # fmt: skip
    """(model, false-alarm index) pairs: ``n`` per model at random, in one random order."""
    rng = np.random.default_rng(seed)
    pool = [(m, int(i)) for m, idx in fas.items()
            for i in rng.choice(idx, min(n, len(idx)), replace=False)]  # fmt: skip
    return [pool[k] for k in rng.permutation(len(pool))]


def contact_sheets(crops: np.ndarray, out: Path, per: int = 12) -> int:
    """R20's sheets: per spot the centre at 4x and the whole view with the centre marked."""
    from PIL import Image, ImageDraw

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("sheet_*.png"):
        old.unlink()
    h = r19.HALF
    for sheet in range(0, len(crops), per):
        img = Image.new("RGB", (3 * 600, 4 * 310), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        for k, c in enumerate(crops[sheet : sheet + per]):
            x0, y0 = (k % 3) * 600, (k // 3) * 310
            view = Image.fromarray(np.asarray(c))
            img.paste(view.crop((h // 2, h // 2, 3 * h // 2, 3 * h // 2)).resize((288, 288)),
                      (x0, y0 + 20))  # fmt: skip
            img.paste(view.resize((288, 288)), (x0 + 296, y0 + 20))
            cx, cy = x0 + 296 + 144, y0 + 20 + 144
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                draw.line((cx + 30 * dx, cy + 30 * dy, cx + 48 * dx, cy + 48 * dy),
                          fill=(255, 0, 255), width=2)  # fmt: skip
            draw.text((x0 + 4, y0 + 4), f"#{sheet + k:03d}  centre x4 | closer look",
                      fill=(0, 0, 0))  # fmt: skip
        img.save(out / f"sheet_{sheet // per:02d}.png")
    return (len(crops) + per - 1) // per


def clear_rate(n_fa: np.ndarray, sampled: np.ndarray, clear: np.ndarray,
               weights: np.ndarray) -> np.ndarray:  # fmt: skip
    """Per bootstrap draw (frame weights), clear non-person false alarms per frame: the
    false alarms per frame times the share of the sample that is clear.
    ``n_fa``: false alarms per frame; ``sampled``/``clear``: per frame, sampled spots and
    clear ones among them."""
    fa = weights @ n_fa / weights.sum(1)
    share = (weights @ clear) / np.maximum(weights @ sampled, 1e-9)
    return fa * share
