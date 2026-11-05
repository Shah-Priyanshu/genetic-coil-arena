"""R19: an open-weight vision model as the confirmer of the scout's detections.

R15e/R15f showed the operator cannot remove the scout brain's false alarms alone, and
R17 step 4 asks for 10-30x fewer of them. This module tests whether an open-weight model
that looks at each detection before the operator does can remove them, and whether
editing its weights (fine-tuning on drone footage) matters against using it as is.

Candidates are R15b's width-0.5 brain (seed 0) run over VisDrone2019-DET frames at
native scale (``r15g_video.detect_frame``: 224-px tiles, ignore regions greyed, tiles
merged). Each candidate is the 144-px view around the detection centre, cut from the
frame as R15f's wider view (no ring drawn), labelled "person" when R15b's centre rule
matches it to a counted person. Fitting uses train frames (threshold ``TRAIN_THRESHOLD``,
at most ``TRAIN_CAP`` per frame), selection uses val (threshold ``EVAL_THRESHOLD``)
minus the spots near the 400 R15e/R15f operator spots, and test-dev and the operator
spots are scored once. The operator spots carry R15b's label, Claude's visual check of
the false alarms and the operator's answer, so the model can be compared with the
operator on the same spots and chained in front of them.

Models (all open weights, loaded from ``D:/hf_cache``):

- DINOv2-small (Apache-2.0): ``probe`` keeps the weights and fits a linear head on
  frozen features; ``full`` fine-tunes every weight.
- Qwen3-VL-2B-Instruct (Apache-2.0): ``zero`` asks it, unedited, whether a person is at
  the centre (score = logit "Yes" minus "No"); ``lora`` fine-tunes LoRA adapters on its
  language model with the same question.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from . import r15_scout_brain as r15
from . import r15b_aerial_scout as r15b

OUT_DIR = r15.DATA_DIR / "r19_confirmer"
HF_CACHE = Path(r"D:\hf_cache")
WEIGHTS = r15.DATA_DIR / "r15_models" / "r15b_coco_a0.5_seed0.pt"
HALF = 72  # the 144-px view of R15f
EVAL_THRESHOLD = 0.5  # R15e/R15f's spots
TRAIN_THRESHOLD = 0.4
TRAIN_CAP = 12  # random candidates kept per train frame
GAP = 24.0  # val candidates this close to an operator spot are kept out of selection
KEEP = 0.95  # the confirmer threshold keeps this share of val people
DINO = "facebook/dinov2-small"
QWEN = "Qwen/Qwen3-VL-2B-Instruct"
PROMPT = ("A drone camera looks down at this scene. Is there a person (standing, walking, "
          "sitting, or riding a bike, scooter or motorbike) at the exact centre of the "
          "image? Answer Yes or No.")  # fmt: skip
META = ("frame", "fx", "fy", "score", "size", "person", "dist", "reserved")


def det_zip(split: str) -> Path:
    return r15.DATA_DIR / "visdrone" / f"VisDrone2019-DET-{r15b.SPLITS[split]}.zip"


def frames(split: str) -> Iterator[tuple[int, str, np.ndarray, np.ndarray]]:
    """(index, image name, frame with ignore regions greyed, persons (M, 4))."""
    from PIL import Image

    with zipfile.ZipFile(det_zip(split)) as z:
        anns = sorted(
            n for n in z.namelist() if n.endswith(".txt") and "/annotations/" in n
        )
        for i, a in enumerate(anns):
            member = a.replace("/annotations/", "/images/")[:-4] + ".jpg"
            img = np.asarray(
                Image.open(io.BytesIO(z.read(member))).convert("RGB")
            ).copy()
            persons, ignore = r15b.parse_annotations(z.read(a).decode())
            grey(img, ignore)
            yield i, member.rsplit("/", 1)[-1], img, persons


def grey(img: np.ndarray, ignore: np.ndarray) -> None:
    for x, y, w, h in ignore:
        img[int(y) : int(math.ceil(y + h)), int(x) : int(math.ceil(x + w))] = r15.PAD


def cut(img: np.ndarray, fx: float, fy: float, half: int = HALF) -> np.ndarray:
    """The ``2 * half`` square around (fx, fy), padded with ``PAD`` outside the frame."""
    px, py = int(round(fx)), int(round(fy))
    out = np.full((2 * half, 2 * half, 3), r15.PAD, dtype=np.uint8)
    xa, ya = max(px - half, 0), max(py - half, 0)
    xb, yb = min(px + half, img.shape[1]), min(py + half, img.shape[0])
    if xb > xa and yb > ya:
        out[
            ya - (py - half) : yb - (py - half), xa - (px - half) : xb - (px - half)
        ] = img[ya:yb, xa:xb]
    return out


def box_dist(persons: np.ndarray, cx: float, cy: float) -> float:
    """Distance from (cx, cy) to the nearest person box (0 inside one)."""
    if not len(persons):
        return float("inf")
    dx = np.maximum.reduce([persons[:, 0] - cx, np.zeros(len(persons)),
                            cx - persons[:, 0] - persons[:, 2]])  # fmt: skip
    dy = np.maximum.reduce([persons[:, 1] - cy, np.zeros(len(persons)),
                            cy - persons[:, 1] - persons[:, 3]])  # fmt: skip
    return float(np.hypot(dx, dy).min())


def scout(device: str = "cuda", weights: Path = WEIGHTS) -> tuple[Any, Any, Any]:
    import torch

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)
    model = r15.build_model(0.5)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.to(dev), dev, r15b.input_fn(dev)


def build(split: str, threshold: float, cap: int = 0, seed: int = 0,
          reserved_spots: dict[str, list[tuple[float, float]]] | None = None,
          device: str = "cuda", log: Any = print, weights: Path = WEIGHTS, size: int = 224,
          out_dir: Path = OUT_DIR) -> dict[str, Any]:  # fmt: skip
    """Run the scout over ``split`` and write ``<split>_crops.u8`` and ``<split>_meta.npz``.

    ``weights``, ``size``, ``out_dir``: another scout brain and its tile (R25)."""
    from . import r15g_video as rg

    model, dev, to_input = scout(device, weights)
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, list] = {k: [] for k in META}
    names, n_persons = [], []
    n = 0
    with open(out_dir / f"{split}_crops.u8", "wb") as fh:
        for i, name, img, persons in frames(split):
            names.append(name)
            n_persons.append(len(persons))
            d = rg.detect_frame(model, dev, to_input, img, size)
            d = d[d[:, 4] >= threshold]
            g = np.column_stack([persons, np.ones(len(persons), np.float32)])
            hit = r15b.match(d, g, "centre") >= 0
            idx = np.arange(len(d))
            if cap and len(idx) > cap:
                idx = np.sort(rng.choice(idx, cap, replace=False))
            near = (reserved_spots or {}).get(name, [])
            for j in idx:
                x, y, w, h, s = (float(v) for v in d[j])
                cx, cy = x + w / 2, y + h / 2
                fh.write(cut(img, cx, cy).tobytes())
                for k, v in zip(META, (i, cx, cy, s, math.sqrt(w * h), bool(hit[j]),
                                       box_dist(persons, cx, cy),
                                       any(math.hypot(cx - a, cy - b) < GAP
                                           for a, b in near))):  # fmt: skip
                    meta[k].append(v)
                n += 1
            if (i + 1) % 200 == 0:
                log(f"[progress] {split}: {i + 1} frames, {n} candidates")
    arrays = {k: np.asarray(v) for k, v in meta.items()}
    np.savez(out_dir / f"{split}_meta.npz", names=np.asarray(names),
             n_persons=np.asarray(n_persons), **arrays)  # fmt: skip
    log(f"{split}: {len(names)} frames, {n} candidates, {arrays['person'].mean():.1%} "
        f"on a person, {arrays['reserved'].sum()} reserved")  # fmt: skip
    return arrays


def load(split: str, out_dir: Path = OUT_DIR) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    meta = dict(np.load(out_dir / f"{split}_meta.npz"))
    n = len(meta["frame"])
    crops = np.memmap(out_dir / f"{split}_crops.u8", np.uint8, "r",
                      shape=(n, 2 * HALF, 2 * HALF, 3))  # fmt: skip
    return crops, meta


# ----------------------------------------------------------------------------- metrics


def threshold_at(pos: np.ndarray, keep: float = KEEP) -> float:
    """The score threshold that keeps ``keep`` of the ``pos`` scores."""
    return float(np.quantile(pos, 1 - keep))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    s = np.concatenate([pos, neg])
    r = np.empty(len(s))
    r[np.argsort(s, kind="stable")] = np.arange(1, len(s) + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))  # fmt: skip


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if not n:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def rates(scores: np.ndarray, person: np.ndarray, thr: float) -> dict[str, Any]:
    keep = scores >= thr
    return {"keep": wilson(int(keep[person].sum()), int(person.sum())),
            "pass": wilson(int(keep[~person].sum()), int((~person).sum()))}  # fmt: skip


def frame_bootstrap(frame: np.ndarray, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """(n_boot, N) candidate weights from resampling frames."""
    rng = np.random.default_rng(seed)
    ids, inv = np.unique(frame, return_inverse=True)
    counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=n_boot)
    return counts[:, inv]


# ------------------------------------------------------------------------------ models


def _gpu() -> Any:
    import torch

    torch.cuda.set_per_process_memory_fraction(0.85)
    return torch.device("cuda")


def _views(crops: np.ndarray, idx: np.ndarray, dev: Any, size: int, train: bool,
           rng: np.random.Generator | None = None) -> Any:  # fmt: skip
    """Crops -> float (B, 3, size, size) in 0..255; training adds flips, quarter turns,
    a random 132-px window (the scout's centre is a few pixels off anyway) and light
    brightness/contrast jitter."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.ascontiguousarray(crops[np.sort(idx)])).to(dev)
    order = np.argsort(np.argsort(idx))  # back to the order of idx
    x = x[torch.from_numpy(order).to(dev)].permute(0, 3, 1, 2).float()
    w = 2 * HALF - 12
    if train:
        assert rng is not None
        ox, oy = rng.integers(0, 13, 2)
        x = x[:, :, oy : oy + w, ox : ox + w]
        x = torch.rot90(x, int(rng.integers(4)), (2, 3))
        flip = torch.from_numpy(rng.random(len(x)) < 0.5).to(dev)
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        a = torch.from_numpy(rng.uniform(0.8, 1.2, (len(x), 1, 1, 1))).to(dev).float()
        b = torch.from_numpy(rng.uniform(-20, 20, (len(x), 1, 1, 1))).to(dev).float()
        x = (x * a + b).clamp(0, 255)
    else:
        x = x[:, :, 6 : 6 + w, 6 : 6 + w]
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


class DinoConfirmer:
    """DINOv2-small with a linear head on [CLS, mean patch]."""

    def __init__(self, dev: Any, size: int = 224) -> None:
        import torch
        from transformers import AutoModel

        self.dev, self.size = dev, size
        self.net = AutoModel.from_pretrained(DINO, cache_dir=HF_CACHE / "hub").to(dev)
        self.head = torch.nn.Linear(2 * self.net.config.hidden_size, 1).to(dev)
        self.mean = (
            torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1) * 255
        )
        self.std = (
            torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1) * 255
        )

    def features(self, x: Any) -> Any:
        import torch

        h = self.net(pixel_values=(x - self.mean) / self.std).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)

    def score(self, crops: np.ndarray, batch: int = 256) -> np.ndarray:
        import torch

        self.net.eval()
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for a in range(0, len(crops), batch):
                x = _views(crops, np.arange(a, min(a + batch, len(crops))), self.dev,
                           self.size, False)  # fmt: skip
                out.append(self.head(self.features(x)).float().squeeze(1).cpu().numpy())
        return np.concatenate(out)


def train_dino(mode: str, crops: np.ndarray, person: np.ndarray, val: tuple[np.ndarray,
               np.ndarray], seed: int, epochs: int = 3, batch: int = 64, size: int = 224,
               log: Any = print) -> tuple[DinoConfirmer, list[dict]]:  # fmt: skip
    """``probe``: frozen backbone, head only; ``full``: every weight. The epoch with the
    lowest val false-alarm pass at ``KEEP`` is kept."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = _gpu()
    m = DinoConfirmer(dev, size)
    full = mode == "full"
    m.net.requires_grad_(full)
    groups = [{"params": m.head.parameters(), "lr": 1e-3}]
    if full:
        groups.append({"params": m.net.parameters(), "lr": 2e-5})
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    steps = epochs * (len(crops) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.05
    )
    y_all = torch.from_numpy(person.astype(np.float32)).to(dev)
    vc, vp = val
    hist, best = [], None
    for ep in range(epochs):
        m.net.train(full)
        perm = rng.permutation(len(crops))
        for s in range(len(crops) // batch):
            idx = perm[s * batch : (s + 1) * batch]
            x = _views(crops, idx, dev, m.size, True, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if full:
                    f = m.features(x)
                else:
                    with torch.no_grad():
                        f = m.features(x)
                logit = m.head(f.float()).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit.float(), y_all[torch.from_numpy(idx).to(dev)]
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if s % 200 == 0:
                log(f"[progress] dino {mode} seed {seed} epoch {ep + 1} step {s}: "
                    f"loss {loss.item():.3f}")  # fmt: skip
        sv = m.score(vc)
        thr = threshold_at(sv[vp])
        rec = {"epoch": ep + 1, "auc": auc(sv[vp], sv[~vp]),
               "pass": float((sv[~vp] >= thr).mean())}  # fmt: skip
        hist.append(rec)
        log(f"dino {mode} seed {seed} epoch {ep + 1}: val AUC {rec['auc']:.3f}, "
            f"false alarms passed at {KEEP:.0%} keep {rec['pass']:.1%}")  # fmt: skip
        if best is None or rec["pass"] < best[0]:
            best = (rec["pass"], {k: v.detach().clone() for k, v in m.net.state_dict().items()},
                    {k: v.detach().clone() for k, v in m.head.state_dict().items()})  # fmt: skip
    assert best is not None
    m.net.load_state_dict(best[1])
    m.head.load_state_dict(best[2])
    return m, hist


class QwenConfirmer:
    """Qwen3-VL-2B-Instruct asked ``PROMPT``; score = logit "Yes" minus logit "No" for
    the first answer token (case variants pooled by log-sum-exp)."""

    def __init__(self, dev: Any, size: int = 448) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.dev, self.size = dev, size
        self.proc = AutoProcessor.from_pretrained(QWEN, cache_dir=HF_CACHE / "hub")
        self.proc.tokenizer.padding_side = "left"
        self.net = AutoModelForImageTextToText.from_pretrained(
            QWEN, cache_dir=HF_CACHE / "hub", dtype=torch.bfloat16
        ).to(dev)
        tok = self.proc.tokenizer
        self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")]
        self.no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")]

    def inputs(self, crops: np.ndarray, idx: np.ndarray, train: bool,
               rng: np.random.Generator | None = None) -> Any:  # fmt: skip
        from PIL import Image

        x = _views(crops, idx, self.dev, self.size, train, rng)
        imgs = [Image.fromarray(a) for a in
                x.round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()]  # fmt: skip
        msgs = [[{"role": "user", "content": [{"type": "image"},
                                              {"type": "text", "text": PROMPT}]}]
                for _ in imgs]  # fmt: skip
        text = [self.proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in msgs]  # fmt: skip
        px = self.size * self.size
        return self.proc(text=text, images=imgs, padding=True, return_tensors="pt",
                         size={"shortest_edge": px, "longest_edge": px}).to(self.dev)  # fmt: skip

    def logit(self, batch: Any) -> Any:
        import torch

        out = self.net(**batch, logits_to_keep=1).logits[:, -1].float()  # last position only
        return torch.logsumexp(out[:, self.yes], 1) - torch.logsumexp(
            out[:, self.no], 1
        )

    def score(self, crops: np.ndarray, batch: int = 32) -> np.ndarray:
        import torch

        self.net.eval()
        out = []
        with torch.no_grad():
            for a in range(0, len(crops), batch):
                idx = np.arange(a, min(a + batch, len(crops)))
                out.append(self.logit(self.inputs(crops, idx, False)).cpu().numpy())
        return np.concatenate(out)


def train_lora(q: QwenConfirmer, crops: np.ndarray, person: np.ndarray,
               val: tuple[np.ndarray, np.ndarray], seed: int, steps: int = 1500,
               batch: int = 16, eval_every: int = 500,
               log: Any = print) -> list[dict]:  # fmt: skip
    """LoRA (r 16) on the language model's attention and MLP; the vision tower is kept.
    Loss: binary cross-entropy on the Yes-minus-No logit. The checkpoint with the lowest
    val false-alarm pass at ``KEEP`` is kept (adapters only)."""
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                     "up_proj", "down_proj"])  # fmt: skip
    q.net = get_peft_model(q.net, cfg)
    q.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in q.net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-4, total_steps=steps,
                                                pct_start=0.05)  # fmt: skip
    y_all = torch.from_numpy(person.astype(np.float32))
    vc, vp = val
    hist, best = [], None
    for s in range(steps):
        q.net.train()
        idx = rng.choice(len(crops), batch, replace=False)
        logit = q.logit(q.inputs(crops, idx, True, rng))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, y_all[idx].to(q.dev)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if s % 100 == 0:
            log(f"[progress] lora seed {seed} step {s}: loss {loss.item():.3f}")
        if (s + 1) % eval_every == 0 or s + 1 == steps:
            sv = q.score(vc)
            thr = threshold_at(sv[vp])
            rec = {"step": s + 1, "auc": auc(sv[vp], sv[~vp]),
                   "pass": float((sv[~vp] >= thr).mean())}  # fmt: skip
            hist.append(rec)
            log(f"lora seed {seed} step {s + 1}: val AUC {rec['auc']:.3f}, false alarms "
                f"passed at {KEEP:.0%} keep {rec['pass']:.1%}")  # fmt: skip
            if best is None or rec["pass"] < best[0]:
                best = (rec["pass"], {k: v.detach().clone() for k, v in
                                      q.net.state_dict().items() if "lora_" in k})  # fmt: skip
    assert best is not None
    q.net.load_state_dict(best[1], strict=False)
    return hist


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")
