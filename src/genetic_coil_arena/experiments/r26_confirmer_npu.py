"""R26 (goal G22): does R19's confirmer keep its decisions on MaixCAM's NPU?

R19's chosen confirmer (DINOv2-small fine-tuned, e10s336 seed 2) runs on a GPU, at the
ground station or on a Jetson-class companion computer on the confirming drone. If it ran
on the drone's own camera module (a MaixCAM-class SG2002, a cv181x NPU, which R24b showed
runs the 448 scout brain), the confirming drone would need no companion computer. This
step compiles the confirmer with tpu-mlir 1.30.2 for cv181x in BF16 (the camera's uint8
RGB in, preprocessing fused; R19's eval view: the centre 132 px of the 144-px crop,
bilinear to 336 and rounded to uint8) and runs it in tpu-mlir's cv181x simulator.

A val probe before pre-registration (64 crops): BF16 scores correlate 0.9999 with float
(mean |difference| 0.03 logit); INT8 with tpu-mlir's default calibration does not (0.95,
1.7 logit), so INT8 is not tested here. The simulator takes about 85 s per crop per
worker, so the test uses a fixed sample instead of all 11,639 test-dev candidates.

Read-out, fixed before any test-dev crop was simulated:

- **Sample:** ``N_PER_CLASS`` people and ``N_PER_CLASS`` non-persons drawn at random
  (seed ``SAMPLE_SEED``) from R19's test-dev candidates (the 224 seed-0 scout's).
- **Reference:** the same model in fp32 on CPU on the same uint8 views. **Threshold:**
  R19's, from its stored val scores (keeps 95% of the val selection set's people).
- **PASS** if BF16's accept/reject decision agrees with float's on >= 99% of the sampled
  people and >= 99% of the sampled non-persons, and its AUC on the sample is within 0.005
  of float's. **KILL** if agreement is below 95% for either class.
- Measured with no bar: model size, NPU (ION) memory, the logit differences. Speed needs a
  module: the simulator does not time the NPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import r15_scout_brain as r15
from . import r19_confirmer as rc

OUT_DIR = r15.DATA_DIR / "r26_confirmer_npu"
CONFIRMER = "dino-full-e10s336_s2"  # R19's chosen
SIZE = 336
N_PER_CLASS, SAMPLE_SEED = 600, 26
N_VAL_PER_CLASS = 64  # a val check of the pipeline, no bar
AGREE, KILL_AGREE, AUC_TOL = 0.99, 0.95, 0.005
VARIANT = "bf16"


def r19_threshold() -> float:
    """R19's threshold for its chosen confirmer, from its stored val scores."""
    _, vm = rc.load("val")
    sel = ~vm["reserved"]
    sv = np.load(rc.OUT_DIR / "scores" / f"{CONFIRMER}_val.npy")[sel]
    return rc.threshold_at(sv[vm["person"][sel]])


def sample(person: np.ndarray, n_per_class: int = N_PER_CLASS,
           seed: int = SAMPLE_SEED) -> np.ndarray:  # fmt: skip
    """Sorted indices: ``n_per_class`` people and as many non-persons, at random."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(person), np.flatnonzero(~person)
    return np.sort(np.concatenate([rng.choice(pos, n_per_class, replace=False),
                                   rng.choice(neg, n_per_class, replace=False)]))  # fmt: skip


def views_u8(crops: np.ndarray) -> np.ndarray:
    """R19's eval view (centre 132 px, bilinear to 336), rounded to uint8 (N, 336, 336, 3)."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.ascontiguousarray(crops)).permute(0, 3, 1, 2).float()
    w = 2 * rc.HALF - 12
    x = F.interpolate(x[:, :, 6 : 6 + w, 6 : 6 + w], size=(SIZE, SIZE), mode="bilinear",
                      align_corners=False)  # fmt: skip
    return x.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def load_confirmer() -> Any:
    """The chosen confirmer on CPU in fp32 as one module: normalised (B, 3, 336, 336) to
    logit (B, 1), with the position embedding precomputed for 336 (no resize in the graph)."""
    import torch
    from transformers import AutoModel

    net = AutoModel.from_pretrained(rc.DINO, cache_dir=rc.HF_CACHE / "hub",
                                    attn_implementation="eager")  # fmt: skip
    head = torch.nn.Linear(2 * net.config.hidden_size, 1)
    state = torch.load(rc.OUT_DIR / "models" / CONFIRMER / "model.pt", map_location="cpu")
    net.load_state_dict(state["net"])
    head.load_state_dict(state["head"])

    class Confirmer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net, self.head = net, head
            emb = net.embeddings
            with torch.no_grad():
                tokens = torch.zeros(1, 1 + (SIZE // 14) ** 2, net.config.hidden_size)
                pos = emb.interpolate_pos_encoding(tokens, SIZE, SIZE)
            self.register_buffer("pos", pos.clone())
            emb.interpolate_pos_encoding = lambda e, h, w: self.pos

        def forward(self, x: Any) -> Any:
            h = self.net(pixel_values=x).last_hidden_state
            return self.head(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1))

    return Confirmer().eval()


MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


def float_scores(model: Any, views: np.ndarray, batch: int = 32) -> np.ndarray:
    import torch

    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for a in range(0, len(views), batch):
            x = torch.from_numpy(views[a : a + batch]).permute(0, 3, 1, 2).float()
            out.append(model((x - mean) / std).squeeze(1).numpy())
    return np.concatenate(out)


def export(log: Any = print) -> Path:
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "confirmer336.onnx"
    torch.onnx.export(load_confirmer(), torch.zeros(1, 3, SIZE, SIZE), str(path),
                      input_names=["image"], output_names=["logit"], opset_version=17,
                      dynamo=False)  # fmt: skip
    log(f"ONNX {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def prepare(split: str, log: Any = print) -> None:
    """The sample's uint8 views (``{split}_views.npy``, for the simulator) and fp32 scores."""
    if split == "test" and (OUT_DIR / f"test_{VARIANT}.npy").exists():
        raise SystemExit("test-dev has been simulated: the sample is fixed")
    crops, meta = rc.load(split)
    person = meta["person"].astype(bool)
    pool = ~meta["reserved"] if split == "val" else np.ones(len(person), bool)  # R19's sets
    n = N_PER_CLASS if split == "test" else N_VAL_PER_CLASS
    idx = np.flatnonzero(pool)[sample(person[pool], n)]
    views = views_u8(crops[idx])
    np.save(OUT_DIR / f"{split}_views.npy", views)
    from PIL import Image

    Image.fromarray(views[0]).save(OUT_DIR / f"{split}_first.png")
    s = float_scores(load_confirmer(), views)
    np.save(OUT_DIR / f"{split}_float.npy", s)
    (OUT_DIR / f"{split}_sample.json").write_text(json.dumps(
        {"idx": idx.tolist(), "person": person[idx].tolist(), "seed": SAMPLE_SEED}),
        "utf-8")  # fmt: skip
    log(f"{split}: {len(idx)} views ({int(person[idx].sum())} people), fp32 scores saved")


def agreement(ref: np.ndarray, sim: np.ndarray, thr: float) -> dict[str, Any]:
    same = (ref >= thr) == (sim >= thr)
    return {"wilson": rc.wilson(int(same.sum()), len(same)), "n": len(same),
            "disagree": int((~same).sum())}  # fmt: skip


def evaluate(ref: np.ndarray, sim: np.ndarray, person: np.ndarray,
             thr: float) -> dict[str, Any]:  # fmt: skip
    d = sim - ref
    a_ref, a_sim = rc.auc(ref[person], ref[~person]), rc.auc(sim[person], sim[~person])
    res = {"threshold": thr, "people": agreement(ref[person], sim[person], thr),
           "non_persons": agreement(ref[~person], sim[~person], thr),
           "auc": {"float": a_ref, VARIANT: a_sim},
           "keep": {"float": float((ref[person] >= thr).mean()),
                    VARIANT: float((sim[person] >= thr).mean())},
           "pass": {"float": float((ref[~person] >= thr).mean()),
                    VARIANT: float((sim[~person] >= thr).mean())},
           "logit_diff": {"mean_abs": float(np.abs(d).mean()), "max_abs": float(np.abs(d).max()),
                          "mean": float(d.mean()),
                          "corr": float(np.corrcoef(ref, sim)[0, 1])}}  # fmt: skip
    res["verdict"] = verdict(res)
    return res


def verdict(res: dict[str, Any]) -> str:
    agree = [res[c]["wilson"][0] for c in ("people", "non_persons")]
    if min(agree) < KILL_AGREE:
        return "KILL"
    auc_ok = abs(res["auc"][VARIANT] - res["auc"]["float"]) <= AUC_TOL
    return "PASS" if min(agree) >= AGREE and auc_ok else "NOT PASSED"


def compile_record() -> dict[str, Any] | None:
    p = OUT_DIR / f"{VARIANT}_compile.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def render(rep: dict[str, Any]) -> str:
    t, rec = rep["test"], rep.get("compile") or {}

    def pct(c: str) -> str:
        w = t[c]["wilson"]
        return (f"{100 * w[0]:.1f}% [{100 * w[1]:.1f}, {100 * w[2]:.1f}] "
                f"({t[c]['disagree']} of {t[c]['n']} differ)")  # fmt: skip

    size = f"{rec['cvimodel_bytes'] / 1e6:.1f} MB" if rec else "-"
    ion = f"{rec['ion_mb']:.1f} MB" if rec.get("ion_mb") else "-"
    d = t["logit_diff"]
    lines = ["# R26: R19's confirmer on MaixCAM's NPU (tpu-mlir, cv181x, BF16)", "",
             f"**{rep['verdict']}.** Bars: at R19's threshold ({t['threshold']:.3f} logit), the "
             f"compiled model's accept/reject decision agrees with float's on >= {AGREE:.0%} of "
             f"the sampled people and of the sampled non-persons, and its AUC is within "
             f"{AUC_TOL} of float's; KILL below {KILL_AGREE:.0%} agreement.", "",
             f"Sample: {t['people']['n']} people and {t['non_persons']['n']} non-persons drawn "
             f"at random (seed {SAMPLE_SEED}) from R19's {rep['n_test']:,} test-dev candidates. "
             "Float is the same model in fp32 on the same uint8 views.", "",
             "| | Float (fp32) | cv181x BF16 |", "|---|---|---|",
             f"| Decisions agree, people | | {pct('people')} |",
             f"| Decisions agree, non-persons | | {pct('non_persons')} |",
             f"| AUC on the sample | {t['auc']['float']:.4f} | {t['auc'][VARIANT]:.4f} |",
             f"| People kept | {100 * t['keep']['float']:.1f}% | {100 * t['keep'][VARIANT]:.1f}% |",
             f"| Non-persons passed | {100 * t['pass']['float']:.1f}% | "
             f"{100 * t['pass'][VARIANT]:.1f}% |", "",
             f"Logit difference (BF16 - float): mean {d['mean']:+.3f}, mean absolute "
             f"{d['mean_abs']:.3f}, largest {d['max_abs']:.3f}; correlation {d['corr']:.5f}.", "",
             f"cvimodel {size}, NPU (ION) memory {ion}. The simulator does not time the NPU, so "
             "frames per second needs a module.", "",
             "Reproduce: `python scripts/r26_confirmer_npu.py export`, `prepare --split test`; "
             "in WSL `scripts/r26_confirmer_npu_sim.py compile`, `simulate --split test`; then "
             "`python scripts/r26_confirmer_npu.py report`.", ""]  # fmt: skip
    return "\n".join(lines)


def report(path: Path = Path("docs/results/r26_confirmer_npu.md"),
           log: Any = print) -> dict[str, Any]:  # fmt: skip
    info = json.loads((OUT_DIR / "test_sample.json").read_text("utf-8"))
    ref = np.load(OUT_DIR / "test_float.npy")
    sim = np.load(OUT_DIR / f"test_{VARIANT}.npy")
    person = np.array(info["person"], bool)
    _, tm = rc.load("test")
    res = evaluate(ref, sim, person, r19_threshold())
    rep = {"verdict": res["verdict"], "test": res, "compile": compile_record(),
           "n_test": int(len(tm["person"])), "confirmer": CONFIRMER}  # fmt: skip
    path.write_text(render(rep), "utf-8")
    path.with_suffix(".json").write_text(json.dumps(rep, indent=1), "utf-8")
    log(f"wrote {path}")
    return rep
