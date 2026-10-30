"""R15d: can training on every empty tile cut the scout brain's false detections?

R17 step 4 found that searches with the R15b brain only work once false detections fall
10-30 times. R15b trained on every tile with a person but only ``KEEP_EMPTY`` (20%) of the
empty ones. This brain trains on all of them (training folder ``visdrone_tiles224_neg``),
with R15b's recipe otherwise (width 0.5, from COCO, 20 epochs), and is compared with
R15b's width-0.5 fine-tuned brains on the same VisDrone test tiles.

Comparison at matched recall (``matched_false``): the threshold at which a brain finds
``TARGET_RECALL`` of 8-16 px people (R15b's recall there at 0.5, so roughly people at
4-8 m) is interpolated on the ``FINE_THRESHOLDS`` grid, and the false detections per tile
are read at that threshold (interpolated in log space).

Bar (fixed before the run): averaged over seeds, the all-empty brain raises at most half
the false detections per tile of the R15b brain at matched recall.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .r15b_aerial_scout import FINE_THRESHOLDS

TARGET_BIN, TARGET_RECALL = "8-16", 0.288
BAR_RATIO = 0.5


def matched_false(op: dict[str, dict[str, float]], target: float = TARGET_RECALL,
                  bin_name: str = TARGET_BIN) -> dict[str, float]:  # fmt: skip
    """Threshold, false detections per tile and other bins' recall where the recall of
    ``bin_name`` equals ``target`` (recall falls as the threshold rises)."""
    taus = [t for t in FINE_THRESHOLDS if f"{t:g}" in op]
    rec = [op[f"{t:g}"][f"recall {bin_name}"] for t in taus]
    if not (rec[-1] <= target <= rec[0]):
        return {"threshold": float("nan"), "false_per_tile": float("nan")}
    for k in range(1, len(taus)):
        if rec[k] <= target:
            w = (rec[k - 1] - target) / max(rec[k - 1] - rec[k], 1e-12)
            a, b = op[f"{taus[k - 1]:g}"], op[f"{taus[k]:g}"]
            fa = math.exp((1 - w) * math.log(max(a["false_per_tile"], 1e-9))
                          + w * math.log(max(b["false_per_tile"], 1e-9)))  # fmt: skip
            out = {"threshold": taus[k - 1] + w * (taus[k] - taus[k - 1]),
                   "false_per_tile": fa}  # fmt: skip
            for key in a:
                if key.startswith("recall "):
                    out[key] = (1 - w) * a[key] + w * b[key]
            return out
    return {"threshold": float("nan"), "false_per_tile": float("nan")}


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """``runs``: dicts with ``group`` ("R15b" or "all empty tiles"), ``seed`` and the
    test ``operating`` points."""
    out: dict[str, Any] = {"groups": {}}
    for group in ("R15b", "all empty tiles"):
        rows = [dict(seed=r["seed"], **matched_false(r["operating"]))
                for r in runs if r["group"] == group]  # fmt: skip
        out["groups"][group] = {
            "runs": rows,
            "false_per_tile": float(np.mean([r["false_per_tile"] for r in rows]))
            if rows else float("nan"),
        }  # fmt: skip
    base = out["groups"]["R15b"]["false_per_tile"]
    new = out["groups"]["all empty tiles"]["false_per_tile"]
    out["ratio"] = new / base if base > 0 else float("nan")
    out["pass"] = bool(out["ratio"] <= BAR_RATIO)
    return out


def render(rep: dict[str, Any]) -> str:
    lines = ["# R15d: training on every empty tile against false detections", "",
             f"**{'PASS' if rep['pass'] else 'FAIL'}.** At matched recall "
             f"({100 * TARGET_RECALL:.1f}% of {TARGET_BIN} px people), the brain trained on "
             f"every empty tile raises {rep['ratio']:.2f}× the false detections per tile "
             f"of R15b's brain (bar: at most {BAR_RATIO:g}×).", "",
             "| Brain | Seed | Threshold | False detections per tile | Recall 0-8 | "
             "Recall 8-16 | Recall 16-32 | Recall 32+ |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for group, g in rep["groups"].items():
        for r in g["runs"]:
            rec = [f"{100 * r.get(f'recall {b}', float('nan')):.1f}%"
                   for b in ("0-8", "8-16", "16-32", "32+")]  # fmt: skip
            lines.append(f"| {group} | {r['seed']} | {r['threshold']:.3f} | "
                         f"{r['false_per_tile']:.4f} | " + " | ".join(rec) + " |")  # fmt: skip
    lines += ["", "VisDrone2019-DET test-dev tiles (224, native and half scale), centre "
              "hits; width 0.5, fine-tuned from COCO, 20 epochs; the only change is the "
              "share of empty training tiles (20% → 100%).", "",
              "Reproduce: `python scripts/r15d_negatives.py --from runs:r15b2-train-ft-a05 "
              "runs:r15d-train-neg`.", ""]  # fmt: skip
    return "\n".join(lines)
