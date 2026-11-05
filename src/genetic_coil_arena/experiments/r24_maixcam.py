"""R24 (goal G20): does the 448-px scout brain keep its accuracy through MaixCAM's toolchain?

R15c's 448-px brain finds more people per look than the 224-px one (R17 step 4 picked it
for the recommended setup), but it runs the K210 out of memory (R23). MaixCAM-class
modules (SG2002, a cv181x NPU) take it. This step compiles the 448 brain (seed 0, and a
seed-1 replicate trained the same way) with tpu-mlir 1.30.2 for cv181x: R23c's export
with the heatmap in its own clamped output, INT8 symmetric, preprocessing fused so the
model takes the camera's uint8 RGB, calibrated with tpu-mlir's defaults on ``N_CALIB``
random 448 train tiles (seed 23). tpu-mlir's cv181x simulator runs it on every
VisDrone2019-DET val and test-dev 448 tile (``scripts/r24_maixcam_sim.py``); the float
brain runs on the same tiles with R15b's ``predict``.

Bars, fixed before any val or test-dev tile was simulated (R23b's rule and bars):

- Per seed, the int8 threshold is the lowest whose false detections per val tile do not
  exceed float's at 0.5.
- **PASS** if, for both seeds, int8 finds at least float's share of test-dev people at
  0.5 minus 2 points (all sizes), with false detections per tile within 10% of float's.
- **KILL** if it finds 5 points fewer or more for either seed: the brain would need
  quantisation-aware training for this toolchain.
- Measured with no bar: the compiled model's size and NPU memory, recall per person size.

R24b (goal G20b): R24 found that tpu-mlir runs the clamp after the heat conv as its own
bf16 op, so the conv output was quantised to int8 with its calibrated threshold (about
24), a step of about 0.19 in logit: only a few score levels near 0.5, and ties inside
each. R24b recompiles both seeds with R24's calibration table and one change, the heat
conv's threshold set to the clamp bound 4 (a step of 0.031), then repeats R24 exactly:
same tiles, threshold rule and bars. Pre-registered after R24's test-dev numbers were
known, from a val-only diagnosis; R24b's compiled models are new systems and run on
test-dev once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import r15_scout_brain as r15
from . import r15b_aerial_scout as r15b
from . import r23_k210 as r23

OUT_DIR = r15.DATA_DIR / "r24_maixcam"
TILES = r15.DATA_DIR / "visdrone_tiles448"
SIZE = 448
MODELS = ("r15b_coco_a0.5_seed0_s448", "r15b_coco_a0.5_seed1_s448")
HEATCAL = "__heatcal"  # R24b: the compiled model is f"{tag}{HEATCAL}"
TITLES = {
    "": "# R24: the 448-px scout brain through MaixCAM's toolchain (tpu-mlir, cv181x)",
    HEATCAL: "# R24b: the MaixCAM scout brain with the heat conv calibrated to its clamp",
}
N_CALIB = 500


def load_split(split: str) -> dict:
    return r15b.load_tiles(TILES, split)


def raw_outputs(tag: str, split: str) -> np.ndarray | None:
    p = OUT_DIR / f"{tag}_{split}.npy"
    return np.load(p, mmap_mode="r") if p.exists() else None


def compile_record(tag: str) -> dict[str, Any] | None:
    p = OUT_DIR / f"{tag}_compile.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def verdict(cs: list[dict]) -> str:
    if any(c["kill"] for c in cs):
        return "KILL"
    return "PASS" if len(cs) == len(MODELS) and all(c["passes"] for c in cs) else "NOT PASSED"


def render(rep: dict[str, Any]) -> str:
    lines = [TITLES[rep.get("variant", "")], "",
             f"**{rep['verdict']}.** Per seed the compiled model's threshold is set on VisDrone "
             "val to float's false detections per tile at 0.5; test-dev bars: people found "
             "at most 2 points below float's at 0.5, false detections per tile within 10%.", "",
             "| Seed | int8 threshold | Recall float / int8 | False per tile float / int8 | "
             "0-8 px | 8-16 px | 16-32 px | 32+ px | cvimodel | NPU memory | Passes |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    levels = []
    for tag, c in rep["models"].items():
        t, rec = c["test"], c.get("compile") or {}
        bins = [f"{100 * t[f'recall {b}']['float']:.1f} / {100 * t[f'recall {b}']['int8']:.1f}"
                for b in ("0-8", "8-16", "16-32", "32+")]  # fmt: skip
        size = f"{rec['cvimodel_bytes'] / 1024:.0f} KB" if rec else "-"
        ion = f"{rec['ion_mb']:.2f} MB" if rec.get("ion_mb") else "-"
        lines.append(f"| {tag.split('_seed')[1][0]} | {c['int8_tau']:.3f} | "
                     f"{100 * t['recall all']['float']:.1f}% / {100 * t['recall all']['int8']:.1f}% | "
                     f"{t['false_per_tile']['float']:.3f} / {t['false_per_tile']['int8']:.3f} "
                     f"({100 * c['false_rel']:+.0f}%) | " + " | ".join(bins)
                     + f" | {size} | {ion} | {'yes' if c['passes'] else 'no'} |")  # fmt: skip
        if "val_levels" in c:
            levels.append(str(c["val_levels"]))
    lines += ["", "Recall by person size in px of the 448 tile. NPU memory is the ION memory "
              "`model_tool --info` reports."
              + (f" Distinct int8 scores between 0.45 and 0.56 on val: {' / '.join(levels)}."
                 if levels else ""), "",
              "Reproduce: `scripts/r24_maixcam_sim.py` (WSL, tpu-mlir venv) `calib`, `compile"
              + (" --heat-threshold 4" if rep.get("variant") else "") + "`, `simulate`, then "
              "`python scripts/r24_maixcam.py "
              + ("report-b" if rep.get("variant") else "report") + "`.", ""]  # fmt: skip
    return "\n".join(lines)


def report(path: Path = Path("docs/results/r24_maixcam.md"), log: Any = print,
           variant: str = "") -> dict[str, Any]:  # fmt: skip
    """``variant``: the compiled models' suffix (R24b: ``HEATCAL``); float is the same."""
    rep: dict[str, Any] = {"variant": variant, "models": {}}
    splits = {s: load_split(s) for s in ("val", "test")}
    for tag in MODELS:
        outs = {s: raw_outputs(tag + variant, s) for s in splits}
        if any(o is None for o in outs.values()):
            continue
        got = {}
        for s, split in splits.items():
            log(f"{tag} {s}")
            got[("int8", s)] = r23.tables(*r23.int8_predictions(outs[s]), split)
            got[("float", s)] = r23.tables(*r23.float_predictions(tag, split, SIZE), split)
        rep["models"][tag] = r23.compare_b(*[got[(k, s)] for s in ("val", "test")
                                             for k in ("float", "int8")])  # fmt: skip
        rep["models"][tag]["compile"] = compile_record(tag + variant)
        sc = got[("int8", "val")][0]["score"]
        rep["models"][tag]["val_levels"] = len(np.unique(sc[(sc > 0.45) & (sc < 0.56)]))
    rep["verdict"] = verdict(list(rep["models"].values()))
    path.write_text(render(rep), "utf-8")
    path.with_suffix(".json").write_text(json.dumps(rep, indent=1), "utf-8")
    return rep
