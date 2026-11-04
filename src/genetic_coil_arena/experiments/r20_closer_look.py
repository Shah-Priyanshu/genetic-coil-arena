"""R20: does a closer look from the confirming drone fail on different false alarms?

R19's fine-tuned confirmer and the operator fail on the same false alarms (the operator
accepted 4 of the 5 clear non-persons the model passed), and two confirmers on the same
view fail together too (DINOv2 and Qwen scores correlate 0.85 on val; together they pass
33.5% against 33.9% for DINOv2 alone). In R17's mission the confirming drone flies to
each detection and hovers far lower than the scout sweeps, so it sees the spot in more
detail. This module asks whether that closer look removes false alarms that a second
model on the scout's own view cannot.

VisDrone2019-DET frames stand in for both views. The scout's view is the frame reduced
``FACTOR`` times (box filter: the scout flying twice as high); the closer look is the
frame at native resolution (the confirmer at half the scout's height: the same 144-px
window covers half the ground at twice the detail). R15b's brain runs on the reduced
frame (``r15g_video.detect_frame``). Each candidate gets both 144-px views around its
centre and R15b's centre-rule label against the reduced person boxes. Both views come
from one photograph, so pose, light and motion blur are shared: the test isolates what
resolution alone adds, and a real second look (another angle, another moment) can only
be less correlated.

DINOv2-small is fully fine-tuned on each view separately with R19's chosen recipe
(``r19_confirmer.train_dino``, 10 epochs, 336-px input), two seeds per view. Two chains
are compared at the same share of val people kept (``chain_thresholds``): the control
checks the scout's view with two models (far seeds 0 and 1); the closer look checks it
with one model and the native view with another (far seed 0, close seed 0; replicated
with seed 1 of each).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import r15_scout_brain as r15
from . import r15b_aerial_scout as r15b
from . import r19_confirmer as r19

OUT_DIR = r15.DATA_DIR / "r20_closer_look"
FACTOR = 2  # the scout flies this many times higher than the confirmer hovers
VIEWS = ("far", "close")
META = ("frame", "fx", "fy", "score", "size", "person", "dist")
RATIO_PASS, RATIO_KILL = 0.5, 0.8
KEEP_TEST = 0.90


def reduce(img: np.ndarray) -> np.ndarray:
    """The scout's view: ``img`` reduced ``FACTOR`` times by a box filter."""
    from PIL import Image

    return np.asarray(Image.fromarray(img).reduce(FACTOR)).copy()


def build(split: str, threshold: float, cap: int = 0, seed: int = 0,
          device: str = "cuda", log: Any = print) -> dict[str, Any]:  # fmt: skip
    """Run the scout over the reduced frames of ``split``; write ``<split>_far.u8``,
    ``<split>_close.u8`` (144-px views around each detection) and ``<split>_meta.npz``
    (``fx``, ``fy`` in reduced pixels, ``dist`` to the nearest person box in native
    pixels)."""
    from . import r15g_video as rg

    model, dev, to_input = r19.scout(device)
    rng = np.random.default_rng(seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict[str, list] = {k: [] for k in META}
    names = []
    n = 0
    with open(OUT_DIR / f"{split}_far.u8", "wb") as ff, \
            open(OUT_DIR / f"{split}_close.u8", "wb") as fc:  # fmt: skip
        for i, name, img, persons in r19.frames(split):
            names.append(name)
            far = reduce(img)
            d = rg.detect_frame(model, dev, to_input, far)
            d = d[d[:, 4] >= threshold]
            g = np.column_stack([persons / FACTOR, np.ones(len(persons), np.float32)])
            hit = r15b.match(d, g, "centre") >= 0
            idx = np.arange(len(d))
            if cap and len(idx) > cap:
                idx = np.sort(rng.choice(idx, cap, replace=False))
            for j in idx:
                x, y, w, h, s = (float(v) for v in d[j])
                cx, cy = x + w / 2, y + h / 2
                ff.write(r19.cut(far, cx, cy).tobytes())
                fc.write(r19.cut(img, FACTOR * cx, FACTOR * cy).tobytes())
                for k, v in zip(META, (i, cx, cy, s, math.sqrt(w * h), bool(hit[j]),
                                       r19.box_dist(persons, FACTOR * cx,
                                                    FACTOR * cy))):  # fmt: skip
                    meta[k].append(v)
                n += 1
            if (i + 1) % 200 == 0:
                log(f"[progress] {split}: {i + 1} frames, {n} candidates")
    arrays = {k: np.asarray(v) for k, v in meta.items()}
    np.savez(OUT_DIR / f"{split}_meta.npz", names=np.asarray(names), **arrays)
    log(f"{split}: {len(names)} frames, {n} candidates, "
        f"{arrays['person'].mean():.1%} on a person")  # fmt: skip
    return arrays


def load(split: str, view: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    meta = dict(np.load(OUT_DIR / f"{split}_meta.npz"))
    n = len(meta["frame"])
    crops = np.memmap(OUT_DIR / f"{split}_{view}.u8", np.uint8, "r",
                      shape=(n, 2 * r19.HALF, 2 * r19.HALF, 3))  # fmt: skip
    return crops, meta


# ----------------------------------------------------------------------------- metrics


def chain_thresholds(a: np.ndarray, b: np.ndarray, person: np.ndarray,
                     keep: float = r19.KEEP) -> tuple[float, float, float]:  # fmt: skip
    """Thresholds for "both models pass": each model keeps the same share ``q`` of the
    val people, the smallest ``q`` on a 0.0005 grid with which both together keep
    ``keep`` of them. Returns (threshold a, threshold b, q)."""
    for q in np.arange(keep, 1.0, 0.0005):
        ta = r19.threshold_at(a[person], q)
        tb = r19.threshold_at(b[person], q)
        if ((a[person] >= ta) & (b[person] >= tb)).mean() >= keep:
            return ta, tb, float(q)
    return float(a[person].min()), float(b[person].min()), 1.0


def boot_ratio(num: np.ndarray, den: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per bootstrap draw, the weighted count of ``num`` over that of ``den``."""
    return (weights @ num.astype(float)) / np.maximum(weights @ den.astype(float), 1e-9)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
