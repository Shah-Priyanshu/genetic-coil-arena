"""R15: the scout brain, a person detector sized for OpenSwarm's K210 vision module.

OpenSwarm (the user's drone platform, ``D:\\Projects\\drone``) gives a SCOUT drone a K210 +
OV2640 module that sends ``DETECTION`` packets (class, normalised centre ``cx, cy``,
confidence) to the swarm. This experiment asks how accurate a detector can be at the
sizes the K210 can run, and how accuracy trades against weights and compute.

Data: COCO 2017 persons. Every image is letterboxed (aspect kept, top-left, grey pad) to
``size`` × ``size`` and stored once as uint8 (``prepare``). Crowd boxes are dropped.
Training uses train2017 and evaluation val2017 (5,000 images, never trained on).

Model (``build_model``): a MobileNetV1-style backbone of width ``alpha``, cut after the
stride-16 stage, with a small top-down merge to stride 8. Only K210-friendly ops are
used: 3×3 and depthwise convolutions, 1×1 convolutions, BatchNorm (folded at export),
ReLU and nearest upsampling. A CenterNet head predicts a person-centre heatmap, box
size and centre offset per cell. Decoding is a 3×3 peak search, cheap enough for the
ESP32-S3.

Measures: AP50 for persons (all, and those at least 32 px tall), frame-level "person
present" ROC AUC of the top heatmap score (VWW's rule: a person box covering at least
0.5% of the image),
parameters, int8 weight size and multiply-accumulates per frame.

The numpy parts (letterbox geometry, targets, decoding, AP) are importable without
torch; torch is imported inside the functions that need it.
"""

from __future__ import annotations

import json
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

DATA_DIR = Path(r"D:\Projects\genetic-coil-arena-data")
STRIDE = 8
PAD = 114
PERSON = 1
VWW_AREA = 0.005
TALL_PX = 32


# ------------------------------------------------------------------ data
def letterbox_scale(w: int, h: int, size: int) -> float:
    return size / max(w, h)


def person_boxes(anns: Sequence[dict], scale: float) -> np.ndarray:
    """(M, 4) x, y, w, h in letterboxed pixels of non-crowd person boxes."""
    out = [
        [a["bbox"][0] * scale, a["bbox"][1] * scale, a["bbox"][2] * scale,
         a["bbox"][3] * scale]
        for a in anns
        if a["category_id"] == PERSON and not a.get("iscrowd", 0)
    ]  # fmt: skip
    arr = np.asarray(out, dtype=np.float32).reshape(-1, 4)
    return arr[(arr[:, 2] >= 1) & (arr[:, 3] >= 1)]


_ZIP: dict[str, zipfile.ZipFile] = {}


def _load_image(job: tuple[str, str, int]) -> np.ndarray:
    from PIL import Image

    zpath, member, size = job
    if zpath not in _ZIP:
        _ZIP[zpath] = zipfile.ZipFile(zpath)
    with _ZIP[zpath].open(member) as f:
        img = Image.open(f).convert("RGB")
    w, h = img.size
    s = letterbox_scale(w, h, size)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    out = np.full((size, size, 3), PAD, dtype=np.uint8)
    out[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR))
    return out


def prepare(coco_dir: Path, out_dir: Path, size: int = 160, workers: int = 6,
            limit: int = 0) -> dict[str, Any]:  # fmt: skip
    """Letterboxed uint8 images, person boxes and offsets for train2017 and val2017."""
    from multiprocessing import Pool

    out_dir.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {"size": size, "stride": STRIDE, "splits": {}}
    with zipfile.ZipFile(coco_dir / "annotations_trainval2017.zip") as z:
        anns = {
            split: json.loads(z.read(f"annotations/instances_{split}2017.json"))
            for split in ("val", "train")
        }
    for split, data in anns.items():
        start = time.time()
        images = sorted(data["images"], key=lambda im: im["id"])
        if limit:
            images = images[:limit]
        by_img: dict[int, list] = {}
        for a in data["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        boxes, offsets = [], [0]
        for im in images:
            b = person_boxes(by_img.get(im["id"], []),
                             letterbox_scale(im["width"], im["height"], size))  # fmt: skip
            boxes.append(b)
            offsets.append(offsets[-1] + len(b))
        arr = np.lib.format.open_memmap(
            out_dir / f"{split}_images.npy", mode="w+", dtype=np.uint8,
            shape=(len(images), size, size, 3),
        )  # fmt: skip
        zpath = str(coco_dir / f"{split}2017.zip")
        jobs = [(zpath, f"{split}2017/{im['file_name']}", size) for im in images]
        with Pool(workers) as pool:
            for i, img in enumerate(pool.imap(_load_image, jobs, chunksize=64)):
                arr[i] = img
        arr.flush()
        del arr
        np.save(
            out_dir / f"{split}_boxes.npy", np.concatenate(boxes).astype(np.float32)
        )
        np.save(out_dir / f"{split}_offsets.npy", np.asarray(offsets, dtype=np.int64))
        np.save(out_dir / f"{split}_ids.npy", np.asarray([im["id"] for im in images]))
        n_pos = sum(len(b) > 0 for b in boxes)
        info["splits"][split] = {
            "images": len(images), "with_person": n_pos, "boxes": offsets[-1],
            "seconds": round(time.time() - start, 1),
        }  # fmt: skip
    (out_dir / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def load_split(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    return {
        "images": np.load(data_dir / f"{split}_images.npy", mmap_mode="r"),
        "boxes": np.load(data_dir / f"{split}_boxes.npy"),
        "offsets": np.load(data_dir / f"{split}_offsets.npy"),
    }


def image_boxes(split: dict[str, np.ndarray], i: int) -> np.ndarray:
    o = split["offsets"]
    return split["boxes"][o[i] : o[i + 1]]


# ------------------------------------------------------------------ targets
def targets(boxes_list: Sequence[np.ndarray], size: int, flip: np.ndarray) -> dict:
    """CenterNet targets on the stride-8 grid for a batch.

    Returns heat (B, G, G) gaussian peaks at person centres, and per-centre ``idx``
    (flat cell), ``wh`` (log of box size in cells), ``off`` (sub-cell centre) with
    ``batch`` indices."""
    g = size // STRIDE
    b = len(boxes_list)
    heat = np.zeros((b, g, g), dtype=np.float32)
    yy, xx = np.mgrid[0:g, 0:g].astype(np.float32)
    bi, idx, wh, off = [], [], [], []
    for k, boxes in enumerate(boxes_list):
        for x, y, w, h in boxes:
            if flip[k]:
                x = size - x - w
            cx, cy = (x + w / 2) / STRIDE, (y + h / 2) / STRIDE
            ix, iy = min(int(cx), g - 1), min(int(cy), g - 1)
            sigma = max(0.5, np.sqrt(w * h) / STRIDE / 6.0)
            gauss = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * sigma**2))
            np.maximum(heat[k], gauss, out=heat[k])
            bi.append(k)
            idx.append(iy * g + ix)
            wh.append([np.log(max(w, 1.0) / STRIDE), np.log(max(h, 1.0) / STRIDE)])
            off.append([cx - ix, cy - iy])
    return {
        "heat": heat,
        "batch": np.asarray(bi, dtype=np.int64),
        "idx": np.asarray(idx, dtype=np.int64),
        "wh": np.asarray(wh, dtype=np.float32).reshape(-1, 2),
        "off": np.asarray(off, dtype=np.float32).reshape(-1, 2),
    }


def decode(heat: np.ndarray, wh: np.ndarray, off: np.ndarray, top: int = 20,
           min_score: float = 0.05) -> list[np.ndarray]:  # fmt: skip
    """Per image (K, 5) boxes x, y, w, h, score from sigmoid heatmaps (B, G, G), log
    sizes (B, 2, G, G) and offsets (B, 2, G, G); peaks are 3×3 local maxima."""
    b, g, _ = heat.shape
    pad = np.pad(heat, ((0, 0), (1, 1), (1, 1)), constant_values=-1)
    neigh = np.max(
        [pad[:, dy : dy + g, dx : dx + g] for dy in range(3) for dx in range(3)], axis=0
    )
    peaks = np.where(heat >= neigh, heat, 0.0).reshape(b, -1)
    out = []
    for k in range(b):
        cand = np.argsort(-peaks[k])[:top]
        cand = cand[peaks[k, cand] >= min_score]
        iy, ix = np.divmod(cand, g)
        w = np.exp(wh[k, 0, iy, ix]) * STRIDE
        h = np.exp(wh[k, 1, iy, ix]) * STRIDE
        cx = (ix + off[k, 0, iy, ix]) * STRIDE
        cy = (iy + off[k, 1, iy, ix]) * STRIDE
        out.append(np.stack([cx - w / 2, cy - h / 2, w, h, peaks[k, cand]], 1))
    return out


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix of xywh boxes (N, 4) and (M, 4)."""
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(
        np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]),
        0,
        None,
    )
    ih = np.clip(
        np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]),
        0,
        None,
    )
    inter = iw * ih
    union = a[:, 2:3] * a[:, 3:4] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def ap50(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray],
         keep: Any = None) -> float:  # fmt: skip
    """VOC-style AP at IoU 0.5 (all-point interpolation). ``keep(gt)`` selects the
    ground truth that counts; detections matched to other ground truth are ignored."""
    scores, hits, n_gt = [], [], 0
    for p, g in zip(preds, gts):
        counted = np.ones(len(g), dtype=bool) if keep is None else keep(g)
        n_gt += int(counted.sum())
        order = np.argsort(-p[:, 4]) if len(p) else np.zeros(0, int)
        used = np.zeros(len(g), dtype=bool)
        m = iou(p[order, :4], g) if len(g) and len(p) else np.zeros((len(p), len(g)))
        for r, j in enumerate(order):
            best = int(np.argmax(m[r])) if len(g) else -1
            if best >= 0 and m[r, best] >= 0.5 and not used[best]:
                used[best] = True
                if counted[best]:
                    scores.append(p[j, 4])
                    hits.append(1)
            else:
                scores.append(p[j, 4])
                hits.append(0)
    if n_gt == 0 or not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.cumsum(np.asarray(hits)[order])
    fp = np.cumsum(1 - np.asarray(hits)[order])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    steps = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(((mrec[steps + 1] - mrec[steps]) * mpre[steps + 1]).sum())


# ------------------------------------------------------------------ model (torch)
def build_model(alpha: float) -> Any:
    import torch
    from torch import nn

    def ch(c: int) -> int:
        return max(8, int(round(c * alpha / 8)) * 8)

    def conv_bn(cin: int, cout: int, s: int) -> nn.Sequential:
        return nn.Sequential(nn.Conv2d(cin, cout, 3, s, 1, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))  # fmt: skip

    def dw(cin: int, cout: int, s: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cin, 3, s, 1, groups=cin, bias=False), nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True), nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )  # fmt: skip

    class ScoutNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.s8 = nn.Sequential(
                conv_bn(3, ch(32), 2), dw(ch(32), ch(64), 1), dw(ch(64), ch(128), 2),
                dw(ch(128), ch(128), 1), dw(ch(128), ch(256), 2), dw(ch(256), ch(256), 1),
            )  # fmt: skip
            self.s16 = nn.Sequential(
                dw(ch(256), ch(512), 2), *[dw(ch(512), ch(512), 1) for _ in range(5)]
            )
            head = max(24, ch(128))
            self.lat8 = nn.Conv2d(ch(256), head, 1)
            self.lat16 = nn.Conv2d(ch(512), head, 1)
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.merge = dw(head, head, 1)
            self.out = nn.Conv2d(head, 5, 1)  # heat, log w, log h, off x, off y
            with torch.no_grad():
                self.out.bias[0] = -2.19  # prior 0.1 for the heatmap

        def forward(self, x: Any) -> Any:
            f8 = self.s8(x)
            f16 = self.s16(f8)
            return self.out(self.merge(self.lat8(f8) + self.up(self.lat16(f16))))

    return ScoutNet()


def count_macs(model: Any, size: int) -> int:
    import torch

    macs = [0]

    def hook(m: Any, inp: Any, out: Any) -> None:
        k = m.kernel_size[0] * m.kernel_size[1]
        macs[0] += out.numel() * (m.in_channels // m.groups) * k

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, torch.nn.Conv2d)]  # fmt: skip
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    for h in hs:
        h.remove()
    return macs[0]


def _batches(split: dict, idx: np.ndarray, batch: int, size: int, rng: Any,
             train: bool):  # fmt: skip
    """Yields (uint8 images (B, S, S, 3), targets) with horizontal flips when training."""
    for a in range(0, len(idx), batch):
        sel = np.sort(idx[a : a + batch])
        imgs = np.asarray(split["images"][sel])
        flip = rng.random(len(sel)) < 0.5 if train else np.zeros(len(sel), bool)
        imgs[flip] = imgs[flip, :, ::-1]
        boxes = [image_boxes(split, i)[:, :4] for i in sel]
        yield imgs, targets(boxes, size, flip), sel


def prefetch(gen: Any, depth: int = 6) -> Any:
    """Run a batch generator in a background thread, ``depth`` batches ahead."""
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()

    def work() -> None:
        for item in gen:
            q.put(item)
        q.put(done)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not done:
        yield item


def focal(pred: Any, gt: Any) -> Any:
    import torch

    p = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p**2 * (1 - gt) ** 4 * neg
    n = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / n


def train_model(alpha: float, data_dir: Path, epochs: int = 40, batch: int = 128,
                lr: float = 2e-3, seed: int = 0, limit: int = 0, log: Any = print,
                init: Path | None = None, eval_fn: Any = None) -> dict[str, Any]:  # fmt: skip
    """Train on ``data_dir``'s train split and score its val split with ``eval_fn``
    (default ``evaluate``); ``init`` starts from saved weights instead of random ones.
    """
    import torch

    torch.cuda.set_per_process_memory_fraction(0.85)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = json.loads((data_dir / "info.json").read_text(encoding="utf-8"))
    size = int(info["size"])
    tr, va = load_split(data_dir, "train"), load_split(data_dir, "val")
    n_tr = len(tr["offsets"]) - 1 if not limit else min(limit, len(tr["offsets"]) - 1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(alpha)
    if init is not None:
        model.load_state_dict(torch.load(init, map_location="cpu"))
    macs = count_macs(model, size)
    model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    steps = epochs * ((n_tr + batch - 1) // batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, lr, total_steps=steps, pct_start=0.05
    )
    mean = torch.tensor([123.7, 116.3, 103.5], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([58.4, 57.1, 57.4], device=dev).view(1, 3, 1, 1)

    def to_input(imgs: np.ndarray) -> Any:
        x = torch.from_numpy(np.ascontiguousarray(imgs)).to(dev).permute(0, 3, 1, 2)
        return (x.float() - mean) / std

    start = time.time()
    for ep in range(epochs):
        model.train()
        total, nb = 0.0, 0
        for imgs, t, _ in prefetch(
            _batches(tr, rng.permutation(n_tr), batch, size, rng, True)
        ):
            out = model(to_input(imgs))
            heat = torch.from_numpy(t["heat"]).to(dev)
            loss = focal(out[:, 0], heat)
            if len(t["idx"]):
                flat = out.flatten(2)  # (B, 5, G*G)
                bi = torch.from_numpy(t["batch"]).to(dev)
                ci = torch.from_numpy(t["idx"]).to(dev)
                sel = flat[bi, :, ci]  # (K, 5)
                loss = loss + 0.1 * torch.nn.functional.l1_loss(
                    sel[:, 1:3], torch.from_numpy(t["wh"]).to(dev)
                ) + torch.nn.functional.l1_loss(
                    torch.sigmoid(sel[:, 3:5]), torch.from_numpy(t["off"]).to(dev)
                )  # fmt: skip
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())
            nb += 1
        log(f"[alpha {alpha}] epoch {ep + 1}/{epochs} loss {total / max(nb, 1):.4f} "
            f"({time.time() - start:.0f} s)")  # fmt: skip
    res = (eval_fn or evaluate)(model, va, size, dev, to_input)
    res.update({
        "alpha": alpha, "params": int(n_params), "int8_kb": n_params / 1024,
        "macs": int(macs), "epochs": epochs, "train_images": int(n_tr),
        "seconds": time.time() - start, "size": size,
    })  # fmt: skip
    return {"result": res, "model": model}


def evaluate(
    model: Any, va: dict, size: int, dev: Any, to_input: Any
) -> dict[str, Any]:
    import torch

    model.eval()
    preds, gts, present, score = [], [], [], []
    n = len(va["offsets"]) - 1
    with torch.no_grad():
        for imgs, _, sel in _batches(va, np.arange(n), 256, size, None, False):
            out = model(to_input(imgs)).float()
            heat = torch.sigmoid(out[:, 0]).cpu().numpy()
            wh = out[:, 1:3].cpu().numpy()
            off = torch.sigmoid(out[:, 3:5]).cpu().numpy()
            for k, i in enumerate(sel):
                g = image_boxes(va, int(i))
                gts.append(g)
                present.append(
                    bool(((g[:, 2] * g[:, 3]) >= VWW_AREA * size * size).any())
                )
                score.append(float(heat[k].max()))
            preds += decode(heat, wh, off)
    return {
        "ap50": ap50(preds, gts),
        "ap50_tall": ap50(preds, gts, keep=lambda g: g[:, 3] >= TALL_PX),
        "present_auc": auc(np.asarray(score), np.asarray(present)),
        "present_rate": float(np.mean(present)),
        "val_images": int(n),
    }


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """ROC AUC by ranks (ties averaged): how well the top heatmap score separates
    frames with a person from frames without."""
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):
        tie = score == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    pos = label.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def export_onnx(model: Any, size: int, path: Path) -> bool:
    """ONNX for the K210 toolchain (nncase); False if the exporter is unavailable."""
    import torch

    try:
        model.eval().cpu()
        torch.onnx.export(model, torch.zeros(1, 3, size, size), str(path),
                          opset_version=11, dynamo=False)  # fmt: skip
        return True
    except Exception:  # noqa: BLE001 - optional artefact
        return False
