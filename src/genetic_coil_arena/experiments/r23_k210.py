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


def load_split(split: str = "test") -> dict:
    return r15b.load_tiles(TILES, split)


def raw_outputs(tag: str, split: str = "test") -> np.ndarray | None:
    p = OUT_DIR / f"{tag}_{split}.npy"
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


# ----------------------------------------------------------------------------- R23b
# R23b (goal G19b): the compiled model with its own threshold. R23 found that int8 keeps
# the ranking but shifts the score scale, so a threshold carried over from float passes
# more false detections. The ground station would set the threshold on the compiled
# model: per seed, the int8 threshold is the lowest at which the compiled model's false
# detections per val tile do not exceed float's at FLOAT_TAU; test-dev is then compared
# at those thresholds. R23's test-dev outputs are reused (its fixed-threshold numbers
# were known when these bars were set; the matched-threshold numbers were not).
# PASS if, for both seeds on test-dev, int8 at its val threshold finds at least float's
# share of people at FLOAT_TAU minus 2 points (all sizes) with false detections per tile
# within 10% of float's; KILL if it finds 5 points fewer or more for either seed.
FLOAT_TAU = 0.5
B_RECALL_DROP, B_FALSE_REL, B_KILL_DROP = 0.02, 0.10, 0.05


def tables(preds: list[np.ndarray], top: np.ndarray, split: dict) -> tuple:
    """(centre hit table, top score per tile, tile has a person)."""
    gts = [r15.image_boxes(split, i) for i in range(len(preds))]
    present = np.array([bool(g[:, 4].any()) for g in gts])
    return r15b.hit_table(preds, gts, "centre"), top, present


def matched_threshold(table: dict, n_tiles: int, target: float) -> float:
    """The lowest threshold whose false detections per tile do not exceed ``target``."""
    false = np.sort(table["score"][table["det_gt"] < 0])[::-1]
    allowed = int(np.floor(target * n_tiles + 1e-9))
    if allowed >= len(false):
        return 0.0
    return float(np.nextafter(false[allowed], np.inf))


def at(table: dict, top: np.ndarray, present: np.ndarray, tau: float) -> dict[str, float]:
    tiles = np.ones(len(top), bool)
    return r15b.operating_points(table, top, present, tiles, taus=[tau])[f"{tau:g}"]


def compare_b(f_val: tuple, q_val: tuple, f_test: tuple, q_test: tuple) -> dict[str, Any]:
    """Each argument: (hit table, top scores, present) for float / int8 on val / test."""
    target = at(*f_val, FLOAT_TAU)["false_per_tile"]
    tau = matched_threshold(q_val[0], len(q_val[1]), target)
    f, q = at(*f_test, FLOAT_TAU), at(*q_test, tau)
    out = {"float_tau": FLOAT_TAU, "int8_tau": tau, "val_false_target": target,
           "val_false_int8": at(*q_val, tau)["false_per_tile"],
           "test": {k: {"float": f[k], "int8": q[k]} for k in f}}  # fmt: skip
    out["recall_drop"] = f["recall all"] - q["recall all"]
    out["false_rel"] = q["false_per_tile"] / f["false_per_tile"] - 1.0
    out["passes"] = bool(out["recall_drop"] <= B_RECALL_DROP and abs(out["false_rel"]) <= B_FALSE_REL)
    out["kill"] = bool(out["recall_drop"] >= B_KILL_DROP)
    return out


def render_b(rep: dict[str, Any]) -> str:
    lines = ["# R23b: the K210 scout brain with its own threshold", "",
             f"**{rep['verdict']}.** Per seed the compiled model's threshold is set on VisDrone "
             f"val so that its false detections per tile do not exceed the float brain's at "
             f"{FLOAT_TAU}; test-dev is then compared at those thresholds. Bars: people found "
             f"(all sizes) at most {100 * B_RECALL_DROP:.0f} points below float's, false "
             f"detections per tile within {100 * B_FALSE_REL:.0f}%.", "",
             "| Seed | int8 threshold | Recall float / int8 | False per tile float / int8 | "
             "0-8 px | 8-16 px | 16-32 px | 32+ px | Passes |",
             "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for tag, c in rep["models"].items():
        t = c["test"]
        bins = [f"{100 * t[f'recall {b}']['float']:.1f} / {100 * t[f'recall {b}']['int8']:.1f}"
                for b in ("0-8", "8-16", "16-32", "32+")]  # fmt: skip
        lines.append(f"| {tag[-1]} | {c['int8_tau']:.3f} | {100 * t['recall all']['float']:.1f}% / "
                     f"{100 * t['recall all']['int8']:.1f}% | {t['false_per_tile']['float']:.3f} / "
                     f"{t['false_per_tile']['int8']:.3f} ({100 * c['false_rel']:+.0f}%) | "
                     + " | ".join(bins) + f" | {'yes' if c['passes'] else 'no'} |")  # fmt: skip
    lines += ["", "Recall by person size (px in the 224 tile), float at 0.5 / int8 at its "
              "threshold. Reproduce: `python scripts/r23_k210.py report-b`.", ""]  # fmt: skip
    return "\n".join(lines)


def report_b(path: Path = Path("docs/results/r23b_k210_threshold.md"),
             log: Any = print) -> dict[str, Any]:  # fmt: skip
    rep: dict[str, Any] = {"models": {}}
    splits = {s: load_split(s) for s in ("val", "test")}
    for tag in MODELS:
        outs = {s: raw_outputs(tag, s) for s in splits}
        if any(o is None for o in outs.values()):
            continue
        got = {}
        for s, split in splits.items():
            log(f"{tag} {s}: int8 and float hit tables")
            got[("int8", s)] = tables(*int8_predictions(outs[s]), split)
            got[("float", s)] = tables(*float_predictions(tag, split), split)
        rep["models"][tag] = compare_b(*[got[(k, s)] for s in ("val", "test")
                                         for k in ("float", "int8")])  # fmt: skip
    if rep["models"]:
        cs = rep["models"].values()
        rep["verdict"] = ("KILL" if any(c["kill"] for c in cs) else
                          "PASS" if len(rep["models"]) == len(MODELS) and all(c["passes"] for c in cs)
                          else "NOT PASSED")  # fmt: skip
    path.write_text(render_b(rep), "utf-8")
    path.with_suffix(".json").write_text(json.dumps(rep, indent=1), "utf-8")
    return rep


# ----------------------------------------------------------------------------- R23c
# R23c (goal G19c): two compile-side fixes that leave the float brain unchanged, chosen
# on val. R23b found that the five output channels share one 8-bit scale whose steps are
# too coarse near the threshold, and that large people lose most (VisDrone's random
# train tiles, used for calibration, rarely hold one).
# - "heat": the head's last convolution split in two, the heatmap's own (clamped to
#   logits HEAT_CLAMP, which only moves scores below 0.018 or above 0.982; the decoder
#   ignores peaks under 0.05) and the size and offset channels', so each gets its own
#   8-bit range. Calibration as in R23.
# - "heat_sized": "heat" calibrated on N_CALIB train tiles, half random and half drawn
#   from tiles holding a person of at least LARGE_PX (square root of box area).
# Selection on seed 0's val: the variant (R23's "base" included) with the most people
# found at float's val false-detection rate at 0.5, interpolated between the int8 score
# steps; a later variant must beat an earlier one by 0.5 points to be chosen. The chosen
# variant is then compiled for seed 1 and both run once on test-dev, with R23b's
# threshold rule and bars.
VARIANTS = ("base", "heat", "heat_sized")
HEAT_CLAMP = (-4.0, 4.0)
LARGE_PX = 32.0
N_CALIB = 500
SELECT_MARGIN = 0.005


def variant_tag(tag: str, variant: str) -> str:
    return tag if variant == "base" else f"{tag}__{variant}"


def export_heat_onnx(tag: str) -> Path:
    """The brain with the heatmap in its own clamped output (same scores in float)."""
    import torch
    from torch import nn

    model = r15.build_model(0.5)
    model.load_state_dict(torch.load(r15.DATA_DIR / "r15_models" / f"{tag}.pt",
                                     map_location="cpu"))  # fmt: skip
    model.eval()

    class Split(nn.Module):
        def __init__(self, m: Any) -> None:
            super().__init__()
            self.m = m
            w, b = m.out.weight.data, m.out.bias.data
            self.heat = nn.Conv2d(w.shape[1], 1, 1)
            self.rest = nn.Conv2d(w.shape[1], 4, 1)
            with torch.no_grad():
                self.heat.weight.copy_(w[:1]), self.heat.bias.copy_(b[:1])
                self.rest.weight.copy_(w[1:]), self.rest.bias.copy_(b[1:])

        def forward(self, x: Any) -> Any:
            m = self.m
            f8 = m.s8(x)
            h = m.merge(m.lat8(f8) + m.up(m.lat16(m.s16(f8))))
            return torch.clamp(self.heat(h), *HEAT_CLAMP), self.rest(h)

    net = Split(model).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref = model(x)
        heat, rest = net(x)
    assert torch.allclose(torch.cat([heat, rest], 1), torch.cat(
        [ref[:, :1].clamp(*HEAT_CLAMP), ref[:, 1:]], 1), atol=1e-4)  # fmt: skip
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{variant_tag(tag, 'heat')}.onnx"
    torch.onnx.export(net, x, str(path), opset_version=11, input_names=["input"],
                      output_names=["heat", "rest"], do_constant_folding=True,
                      dynamo=False)  # fmt: skip
    return path


def sized_calibration(seed: int = 23) -> Path:
    """Half random train tiles, half from tiles holding a person >= LARGE_PX."""
    tr = r15b.load_tiles(TILES, "train")
    n = len(tr["offsets"]) - 1
    largest = np.zeros(n)
    for i in range(n):
        g = r15.image_boxes(tr, i)
        g = g[g[:, 4] > 0]
        if len(g):
            largest[i] = np.sqrt(g[:, 2] * g[:, 3]).max()
    rng = np.random.default_rng(seed)
    big = rng.choice(np.flatnonzero(largest >= LARGE_PX), N_CALIB // 2, replace=False)
    rest = rng.choice(np.setdiff1d(np.arange(n), big), N_CALIB - len(big), replace=False)
    path = OUT_DIR / "calib_sized.npy"
    np.save(path, np.sort(np.concatenate([big, rest])))
    return path


def recall_at_false(table: dict, top: np.ndarray, present: np.ndarray, target: float,
                    lo: float = 0.2, hi: float = 0.9) -> float:  # fmt: skip
    """People found (all sizes) at ``target`` false detections per tile, interpolated
    linearly between the int8 score steps around it."""
    levels = np.unique(table["score"][(table["score"] >= lo) & (table["score"] <= hi)])
    pts = [at(table, top, present, float(t)) for t in levels[::-1]]
    false = np.array([p["false_per_tile"] for p in pts])
    rec = np.array([p["recall all"] for p in pts])
    return float(np.interp(target, false, rec))


def select_variant(scores: dict[str, float]) -> str:
    """The first variant in ``VARIANTS`` order unless a later one beats the current
    choice by ``SELECT_MARGIN``."""
    best = VARIANTS[0]
    for v in VARIANTS[1:]:
        if v in scores and scores[v] >= scores[best] + SELECT_MARGIN:
            best = v
    return best


SELECT_JSON = Path("docs/results/r23c_k210_select.json")


def select_c(log: Any = print) -> dict[str, Any]:
    """R23c step 1, on seed 0's val only: score each variant and pick one."""
    tag, split = MODELS[0], load_split("val")
    f = tables(*float_predictions(tag, split), split)
    target = at(*f, FLOAT_TAU)["false_per_tile"]
    rep: dict[str, Any] = {"model": tag, "float_val_false": target,
                           "float_val_recall": at(*f, FLOAT_TAU)["recall all"], "variants": {}}  # fmt: skip
    for v in VARIANTS:
        out = raw_outputs(variant_tag(tag, v), "val")
        if out is None:
            continue
        log(f"{v}: val")
        q = tables(*int8_predictions(out), split)
        tau = matched_threshold(q[0], len(q[1]), target)
        rep["variants"][v] = {"recall_at_false": recall_at_false(*q, target), "matched_tau": tau,
                              "at_matched": at(*q, tau)}  # fmt: skip
    rep["chosen"] = select_variant({v: r["recall_at_false"] for v, r in rep["variants"].items()})
    SELECT_JSON.write_text(json.dumps(rep, indent=1), "utf-8")
    return rep


def render_c(rep: dict[str, Any]) -> str:
    sel = rep["select"]
    lines = ["# R23c: the K210 scout brain with a finer heatmap and size-aware calibration", "",
             f"**{rep['verdict']}.** Variant chosen on seed 0's val: **{sel['chosen']}** "
             "(people found at float's val false-detection rate, interpolated between the "
             "int8 score steps):", "",
             "| Variant | People found at float's val false rate | Threshold set on val | "
             "32+ px people at that threshold |", "|---|---|---|---|"]  # fmt: skip
    for v, r in sel["variants"].items():
        lines.append(f"| {v} | {100 * r['recall_at_false']:.1f}% | {r['matched_tau']:.3f} | "
                     f"{100 * r['at_matched']['recall 32+']:.1f}% |")
    lines += ["", f"Float on val at 0.5: {100 * sel['float_val_recall']:.1f}% of people, "
              f"{sel['float_val_false']:.3f} false detections per tile.", "",
              "Test-dev, the chosen variant with R23b's threshold rule and bars (people found "
              "at most 2 points below float's at 0.5, false detections per tile within 10%):", "",
              "| Seed | int8 threshold | Recall float / int8 | False per tile float / int8 | "
              "0-8 px | 8-16 px | 16-32 px | 32+ px | Passes |",
              "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for tag, c in rep["models"].items():
        t = c["test"]
        bins = [f"{100 * t[f'recall {b}']['float']:.1f} / {100 * t[f'recall {b}']['int8']:.1f}"
                for b in ("0-8", "8-16", "16-32", "32+")]  # fmt: skip
        lines.append(f"| {tag.split('__')[0][-1]} | {c['int8_tau']:.3f} | "
                     f"{100 * t['recall all']['float']:.1f}% / {100 * t['recall all']['int8']:.1f}% | "
                     f"{t['false_per_tile']['float']:.3f} / {t['false_per_tile']['int8']:.3f} "
                     f"({100 * c['false_rel']:+.0f}%) | " + " | ".join(bins)
                     + f" | {'yes' if c['passes'] else 'no'} |")  # fmt: skip
    lines += ["", "Reproduce: `python scripts/r23_k210.py select-c`, then `report-c`.", ""]
    return "\n".join(lines)


def report_c(path: Path = Path("docs/results/r23c_k210_fixes.md"),
             log: Any = print) -> dict[str, Any]:  # fmt: skip
    """R23c step 2: the chosen variant on test-dev for both seeds, R23b's rule and bars."""
    sel = json.loads(SELECT_JSON.read_text("utf-8"))
    rep: dict[str, Any] = {"select": sel, "models": {}}
    splits = {s: load_split(s) for s in ("val", "test")}
    for tag in MODELS:
        name = variant_tag(tag, sel["chosen"])
        outs = {s: raw_outputs(name, s) for s in splits}
        if any(o is None for o in outs.values()):
            continue
        got = {}
        for s, split in splits.items():
            log(f"{name} {s}")
            got[("int8", s)] = tables(*int8_predictions(outs[s]), split)
            got[("float", s)] = tables(*float_predictions(tag, split), split)
        rep["models"][name] = compare_b(*[got[(k, s)] for s in ("val", "test")
                                          for k in ("float", "int8")])  # fmt: skip
    cs = list(rep["models"].values())
    rep["verdict"] = ("KILL" if any(c["kill"] for c in cs) else
                      "PASS" if len(cs) == len(MODELS) and all(c["passes"] for c in cs)
                      else "NOT PASSED")  # fmt: skip
    path.write_text(render_c(rep), "utf-8")
    path.with_suffix(".json").write_text(json.dumps(rep, indent=1), "utf-8")
    return rep
