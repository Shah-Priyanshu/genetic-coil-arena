"""R23 (goal G19): does the scout brain keep its accuracy through the K210 toolchain?

OpenSwarm's vision module is a Kendryte K210 running an INT8 model at 224 x 224
converted with nncase (``D:\\Projects\\drone\\ai_module\\README.md``). Every R15-R22
number so far comes from the float brain. This step compiles R15b's deployed brain
(width 0.5, COCO then VisDrone, seed 0; seed 1 as the replicate) with nncase 1.9.0 for
the K210: uint8 weights and activations, post-training quantisation calibrated on
``N_CALIB`` random VisDrone train tiles, the input normalisation folded into the model.
nncase's simulator runs the compiled model on every VisDrone2019-DET test-dev tile
(``scripts/r23_k210_sim.py``); the float brain runs on the same tiles with R15b's own
``predict``. Both go through R15b's decode and ``tile_metrics``, with a paired frame
bootstrap.

Bars, fixed before any test-dev tile was simulated:

- **PASS** if, for both seeds, the int8 model loses at most 2 points of centre AP
  (pooled, R15b's main metric for what OpenSwarm's DETECTION packet carries) with the
  bootstrap interval's worst end at most 3 points, at most 2 points of AP50 and at most
  0.01 present AUC; and at the 0.5 threshold its false detections per tile are within
  20% of float's and its recall (all sizes) within 3 points.
- **KILL** if centre AP drops by 5 points or more for either seed.
- Measured with no bar: the compiled model's size and memory, R17's detector table at
  0.3 / 0.5 / 0.7 for int8 against float, and whether the 448-px brain (R15c) compiles
  for the K210 and how much memory it needs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import r15_scout_brain as r15
from . import r15b_aerial_scout as r15b

OUT_DIR = r15.DATA_DIR / "r23_k210"
TILES = r15.DATA_DIR / "visdrone_tiles224"
MODELS = ("r15b_coco_a0.5_seed0", "r15b_coco_a0.5_seed1")
EXTRA = ("r15b_coco_a0.5_seed0_s448",)  # compiled only (memory), no bar
N_BOOT, BOOT_SEED = 500, 23
AP_DROP, AP_WORST, AUC_DROP, FALSE_REL, RECALL_DROP, KILL_DROP = (
    0.02, 0.03, 0.01, 0.20, 0.03, 0.05)  # fmt: skip
TOP = 100  # r15b.predict's detections per tile


def int8_predictions(out: np.ndarray, batch: int = 1024) -> tuple[list[np.ndarray], np.ndarray]:
    """R15b's ``predict`` on stored raw outputs (N, 5, G, G) of the compiled model."""
    preds, best = [], []
    for a in range(0, len(out), batch):
        o = np.asarray(out[a : a + batch], dtype=np.float32)
        heat = 1.0 / (1.0 + np.exp(-o[:, 0]))
        preds += r15.decode(heat, o[:, 1:3], 1.0 / (1.0 + np.exp(-o[:, 3:5])), top=TOP)
        best.append(heat.reshape(len(heat), -1).max(1))
    return preds, np.concatenate(best)


def float_predictions(model_tag: str, split: dict) -> tuple[list[np.ndarray], np.ndarray]:
    """The float brain on the CPU (the GPU may be busy), exactly as R15b scored it."""
    import torch

    model = r15.build_model(0.5)
    model.load_state_dict(torch.load(r15.DATA_DIR / "r15_models" / f"{model_tag}.pt",
                                     map_location="cpu"))  # fmt: skip
    dev = torch.device("cpu")
    return r15b.predict(model, split, 224, dev, r15b.input_fn(dev), batch=128, top=TOP)


def compare(f: dict, q: dict) -> dict[str, Any]:
    """Float (``f``) against int8 (``q``) tile metrics on the same tiles and resamples."""
    pf, pq = f["pooled"], q["pooled"]
    out: dict[str, Any] = {}
    for key in ("centre_ap", "ap50"):
        _, lo, hi = r15b.paired(pf[f"boot {key}"], pq[f"boot {key}"])
        out[key] = {"float": pf[f"{key} all"], "int8": pq[f"{key} all"],
                    "drop": pf[f"{key} all"] - pq[f"{key} all"], "drop_lo": lo, "drop_hi": hi}  # fmt: skip
    out["present_auc"] = {"float": pf["present_auc"], "int8": pq["present_auc"],
                          "drop": pf["present_auc"] - pq["present_auc"]}  # fmt: skip
    table = {}
    for tau in ("0.3", "0.5", "0.7"):
        a, b = pf["operating"][tau], pq["operating"][tau]
        table[tau] = {k: {"float": a[k], "int8": b[k]} for k in a}
    out["operating"] = table
    op = table["0.5"]
    out["false_rel"] = op["false_per_tile"]["int8"] / op["false_per_tile"]["float"] - 1.0
    out["recall_drop"] = op["recall all"]["float"] - op["recall all"]["int8"]
    out["passes"] = bool(
        out["centre_ap"]["drop"] <= AP_DROP and out["centre_ap"]["drop_hi"] <= AP_WORST
        and out["ap50"]["drop"] <= AP_DROP and out["present_auc"]["drop"] <= AUC_DROP
        and abs(out["false_rel"]) <= FALSE_REL and out["recall_drop"] <= RECALL_DROP
    )  # fmt: skip
    out["kill"] = bool(out["centre_ap"]["drop"] >= KILL_DROP)
    return out


def verdict(per_model: dict[str, dict]) -> str:
    if any(c["kill"] for c in per_model.values()):
        return "KILL"
    if len(per_model) == len(MODELS) and all(c["passes"] for c in per_model.values()):
        return "PASS"
    return "NOT PASSED"


def compile_records() -> dict[str, Any]:
    out = {}
    for tag in MODELS + EXTRA:
        p = OUT_DIR / f"{tag}_compile.json"
        if p.exists():
            out[tag] = json.loads(p.read_text("utf-8"))
        elif (OUT_DIR / f"{tag}_compile.log").exists():  # the compiler raised
            text = (OUT_DIR / f"{tag}_compile.log").read_text("utf-8")
            err = [x for x in text.splitlines() if x.startswith("RuntimeError")]
            size = 448 if tag.endswith("_s448") else 224
            out[tag] = {"size": size, "failed": err[-1] if err else "failed"}
    return out


def render(rep: dict[str, Any]) -> str:
    lines = ["# R23: the scout brain through the K210 toolchain", ""]
    if "verdict" in rep:
        lines += [f"**{rep['verdict']}.** R15b's width-0.5 brain compiled by nncase 1.9.0 "
                  "for the K210 (uint8 weights and activations, calibrated on "
                  f"{rep['calibration']} VisDrone train tiles) and run in nncase's simulator "
                  "on every VisDrone2019-DET test-dev tile, against the float brain on the "
                  "same tiles. Bars: centre AP and AP50 drop <= 2 points (worst end of the "
                  "centre-AP interval <= 3), present AUC drop <= 0.01, at threshold 0.5 "
                  "false detections per tile within 20% and recall within 3 points.", ""]  # fmt: skip
        lines += ["| Seed | Centre AP float / int8 (drop [95% CI]) | AP50 float / int8 | "
                  "Present AUC float / int8 | False per tile at 0.5 | Recall at 0.5 | Passes |",
                  "|---|---|---|---|---|---|---|"]  # fmt: skip
        for tag, c in rep["models"].items():
            ca, ap, au = c["centre_ap"], c["ap50"], c["present_auc"]
            op = c["operating"]["0.5"]
            lines.append(
                f"| {tag[-1]} | {ca['float']:.3f} / {ca['int8']:.3f} ({100 * ca['drop']:+.1f} "
                f"[{100 * ca['drop_lo']:+.1f}, {100 * ca['drop_hi']:+.1f}]) | "
                f"{ap['float']:.3f} / {ap['int8']:.3f} | {au['float']:.3f} / {au['int8']:.3f} | "
                f"{op['false_per_tile']['float']:.3f} / {op['false_per_tile']['int8']:.3f} "
                f"({100 * c['false_rel']:+.0f}%) | {100 * op['recall all']['float']:.1f}% / "
                f"{100 * op['recall all']['int8']:.1f}% | {'yes' if c['passes'] else 'no'} |"
            )  # fmt: skip
        lines += ["", "Drops are float minus int8 (positive = int8 worse), in points.", ""]
        lines += ["R17's detector table (seed 0), recall per person size (px) and false "
                  "detections per tile, float / int8:", "",
                  "| Threshold | 0-8 | 8-16 | 16-32 | 32+ | False per tile |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
        c = rep["models"][MODELS[0]]
        for tau, row in c["operating"].items():
            cells = [f"{100 * row[f'recall {b}']['float']:.1f} / "
                     f"{100 * row[f'recall {b}']['int8']:.1f}"
                     for b in ("0-8", "8-16", "16-32", "32+")]  # fmt: skip
            lines.append(f"| {tau} | " + " | ".join(cells) + f" | "
                         f"{row['false_per_tile']['float']:.3f} / "
                         f"{row['false_per_tile']['int8']:.3f} |")  # fmt: skip
        lines.append("")
    if rep.get("compile"):
        lines += ["Compiled models (nncase's memory plan; the K210 has 6 MB of general SRAM "
                  "and 2 MB for the KPU):", "",
                  "| Model | Input | kmodel | Total memory |", "|---|---|---|---|"]  # fmt: skip
        for tag, r in rep["compile"].items():
            if "failed" in r:
                lines.append(f"| {tag} | {r['size']} px | does not compile: {r['failed']} | - |")
                continue
            total = r["memory"].get("TOTAL")
            lines.append(f"| {tag} | {r['size']} px | {r['kmodel_bytes'] / 1024:.0f} KB | "
                         + (f"{total / 2**20:.2f} MB" if total else "-") + " |")
        lines.append("")
    lines += ["Reproduce: `D:\\venvs\\nncase\\Scripts\\python.exe scripts/r23_k210_sim.py "
              "compile|simulate ...`, then `python scripts/r23_k210.py report`.", ""]  # fmt: skip
    return "\n".join(lines)


def load_split() -> dict:
    return r15b.load_tiles(TILES, "test")


def raw_outputs(tag: str) -> np.ndarray | None:
    p = OUT_DIR / f"{tag}_test.npy"
    return np.load(p, mmap_mode="r") if p.exists() else None


def report(path: Path = Path("docs/results/r23_k210.md"), log: Any = print) -> dict[str, Any]:
    rep: dict[str, Any] = {"compile": compile_records(), "models": {}}
    split = None
    for tag in MODELS:
        out = raw_outputs(tag)
        if out is None:
            continue
        split = split if split is not None else load_split()
        if len(out) != len(split["offsets"]) - 1:
            raise ValueError(f"{tag}: {len(out)} outputs for {len(split['offsets']) - 1} tiles")
        log(f"{tag}: int8 metrics")
        q = r15b.tile_metrics(*int8_predictions(out), split, n_boot=N_BOOT, seed=BOOT_SEED)
        log(f"{tag}: float predictions on the CPU")
        f = r15b.tile_metrics(*float_predictions(tag, split), split, n_boot=N_BOOT,
                              seed=BOOT_SEED)  # fmt: skip
        rep["models"][tag] = compare(f, q)
    if rep["models"]:
        rep["verdict"] = verdict(rep["models"])
        rep["calibration"] = rep["compile"][MODELS[0]]["calibration"]["tiles"]
    path.write_text(render(rep), "utf-8")
    path.with_suffix(".json").write_text(json.dumps(rep, indent=1), "utf-8")
    return rep
