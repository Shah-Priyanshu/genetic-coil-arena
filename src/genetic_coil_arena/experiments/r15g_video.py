"""R15g: the scout brain on drone video (VisDrone2019-VID), for filters across frames.

R15b scored the scout brain on single frames. The person search needs about ten times
fewer false detections than it gives (R17 step 4), and a careful operator alone does not
close the gap (R15f). This module runs the same brain (R15b's width-0.5 fine-tuned
model, seed 0) over every frame of the VID sequences at native scale: each frame is cut
into 224-px tiles as in R15b, VisDrone's ignore regions are greyed first, and detections
from overlapping tiles are merged. It stores, per frame, the detections in frame pixels
and the labelled people with their track ids, so filters that ask a spot to persist
across frames can be scored without rerunning the brain.
"""

from __future__ import annotations

import io
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import r15_scout_brain as r15
from . import r15b_aerial_scout as r15b

VID_DIR = r15.DATA_DIR / "visdrone"
OUT_DIR = r15.DATA_DIR / "visdrone_vid_dets"
WEIGHTS = r15.DATA_DIR / "r15_models" / "r15b_coco_a0.5_seed0.pt"
SPLITS = {"val": "val", "test": "test-dev", "train": "train"}
SIZE = 224
MIN_SCORE = 0.3  # the lowest threshold any later filter uses
MERGE_PX = 4.0  # detections of overlapping tiles closer than this are one


def vid_zip(split: str) -> Path:
    return VID_DIR / f"VisDrone2019-VID-{SPLITS[split]}.zip"


def parse_vid(text: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """VID annotation lines (frame, track, x, y, w, h, score, category, truncation,
    occlusion) -> per frame (people (M, 5): x, y, w, h, track; ignore regions (K, 4)).
    """
    people: dict[int, list] = defaultdict(list)
    ignore: dict[int, list] = defaultdict(list)
    for line in text.strip().splitlines():
        v = [int(x) for x in line.strip().strip(",").split(",")[:8]]
        if len(v) < 8 or v[4] < 1 or v[5] < 1:
            continue
        if v[7] in r15b.PERSON_CATS and v[6] == 1:
            people[v[0]].append(v[2:6] + [v[1]])
        elif v[7] == r15b.IGNORE_CAT:
            ignore[v[0]].append(v[2:6])
    return {f: (np.asarray(people.get(f, []), np.float32).reshape(-1, 5),
                np.asarray(ignore.get(f, []), np.float32).reshape(-1, 4))
            for f in set(people) | set(ignore)}  # fmt: skip


def sequences(z: zipfile.ZipFile) -> dict[str, tuple[str, list[str]]]:
    """Sequence name -> (annotation member, sorted frame members)."""
    frames: dict[str, list[str]] = defaultdict(list)
    anns = {}
    for n in z.namelist():
        parts = n.split("/")
        if "sequences" in parts and n.lower().endswith(".jpg"):
            frames[parts[parts.index("sequences") + 1]].append(n)
        elif "annotations" in parts and n.endswith(".txt"):
            anns[parts[-1][:-4]] = n
    return {s: (anns[s], sorted(frames[s])) for s in sorted(frames) if s in anns}


def merge(dets: np.ndarray) -> np.ndarray:
    """Greedy by score: drop a detection whose centre is within ``MERGE_PX`` of a kept one."""
    if len(dets) < 2:
        return dets
    dets = dets[np.argsort(-dets[:, 4])]
    c = dets[:, :2] + dets[:, 2:4] / 2
    keep: list[int] = []
    for i in range(len(dets)):
        if all(math.hypot(*(c[i] - c[j])) >= MERGE_PX for j in keep):
            keep.append(i)
    return dets[keep]


def detect_frame(model: Any, dev: Any, to_input: Any, img: np.ndarray) -> np.ndarray:
    """(K, 5) detections x, y, w, h, score in frame pixels, score >= ``MIN_SCORE``."""
    h, w = img.shape[:2]
    tiles, origins = [], []
    for y0 in r15b.tile_starts(h, SIZE):
        for x0 in r15b.tile_starts(w, SIZE):
            t = np.full((SIZE, SIZE, 3), r15.PAD, np.uint8)
            crop = img[y0 : y0 + SIZE, x0 : x0 + SIZE]
            t[: crop.shape[0], : crop.shape[1]] = crop
            tiles.append(t)
            origins.append((x0, y0))
    split = {"images": np.stack(tiles), "offsets": np.arange(len(tiles) + 1)}
    preds = r15b.predict(model, split, SIZE, dev, to_input, top=100)[0]
    out = [p[p[:, 4] >= MIN_SCORE] + np.array([x0, y0, 0, 0, 0], np.float32)
           for p, (x0, y0) in zip(preds, origins)]  # fmt: skip
    return merge(np.concatenate(out)) if out else np.zeros((0, 5), np.float32)


def detect_split(split: str, device: str = "cuda", limit: int = 0,
                 log: Any = print) -> dict[str, np.ndarray]:  # fmt: skip
    """Run the scout brain over every frame of ``split``; arrays for ``np.savez``."""
    import torch
    from PIL import Image

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)
    model = r15.build_model(0.5)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.to(dev)
    to_input = r15b.input_fn(dev)
    dets, gts, info = [], [], []
    with zipfile.ZipFile(vid_zip(split)) as z:
        seqs = sequences(z)
        for s, (name, (ann, members)) in enumerate(seqs.items()):
            if limit and s >= limit:
                break
            labels = parse_vid(z.read(ann).decode())
            size = (0, 0)
            for member in members:
                f = int(Path(member).stem)
                img = np.asarray(
                    Image.open(io.BytesIO(z.read(member))).convert("RGB")
                ).copy()
                size = img.shape[1], img.shape[0]
                people, ignore = labels.get(f, (np.zeros((0, 5)), np.zeros((0, 4))))
                for x, y, bw, bh in ignore:
                    img[
                        int(y) : int(math.ceil(y + bh)), int(x) : int(math.ceil(x + bw))
                    ] = r15.PAD
                d = detect_frame(model, dev, to_input, img)
                dets.append(np.column_stack([np.full((len(d), 2), (s, f)), d]))
                gts.append(np.column_stack([np.full((len(people), 2), (s, f)), people]))
            info.append((name, len(members), *size))
            log(f"[progress] {split} {s + 1}/{len(seqs)} {name}: {len(members)} frames")
    return {
        "dets": np.concatenate(dets).astype(
            np.float32
        ),  # seq, frame, x, y, w, h, score
        "gts": np.concatenate(gts).astype(np.float32),  # seq, frame, x, y, w, h, track
        "seq_names": np.array([i[0] for i in info]),
        "seq_frames": np.array(
            [i[1:] for i in info], np.int64
        ),  # frames, width, height
    }
