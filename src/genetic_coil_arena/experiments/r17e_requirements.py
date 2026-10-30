"""R17 step 4: what the scout brain must reach before a search works.

R15b measured the K210-sized scout brain on drone footage: per frame it finds 11-61% of
people depending on their size in pixels, and raises about 0.1 false detections per
224-px tile at threshold 0.5 (0.002 at 0.7). This step puts that measured table into
the search simulator (``Mission.table``) and asks how many fewer false detections the
scout needs (from a temporal filter on the drone or a better model) before a tuned
search finds people without sending the team to the wrong place.

Everything uses step 1's 40 held-out missions (area, drones; a still person) with the
detector replaced by the measured one. Assumptions, stated rather than measured:

- a person seen from above is ``PERSON_AREA_M`` (0.34 m, the square root of a 0.45 x
  0.25 m box) wide in R15b's size measure; the camera keeps the OV2640's ~60 degrees;
- one second of video holds ``frames`` independent frames (1 or 3);
- the operator confirms candidates from the confirmer's view (``OP_HIT``, ``OP_FALSE``);
- ``fa_mult`` scales the false detections and leaves recall unchanged.

Per condition (camera, frames, fa_mult) each mission's plan is tuned alone
(``r17_mission.tune_plan``) on its own seeds and scored on the test seeds; OpenSwarm's
default and the hand two-tier plan (threshold 0.6) are scored too.

No pre-registered bar: this measures a requirement. Read-out, fixed in advance: the
smallest false-detection cut at which tuned searches find the person in at least 50% of
episodes with at most 5% wrong finds (averaged over missions), per camera and frames.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..envs.swarm.mission import (
    FOOTPRINT,
    OPENSWARM_DEFAULT,
    PERSON_AREA_M,
    Mission,
    Plan,
    hand_two_tier,
    run_missions,
)
from . import r17_mission as r17

FA_MULTS = (1.0, 0.3, 0.1, 0.03, 0.01)
FRAMES = (1.0, 3.0)
EPISODES = 2000
FOUND_TARGET, WRONG_TARGET = 0.5, 0.05
SIZE_BINS = ("0-8", "8-16", "16-32", "32+")
THRESHOLDS = ("0.3", "0.5", "0.7")


def camera_table(report_json: Path, alpha: float = 0.5,
                 condition: str = "fine-tuned") -> dict[str, Any]:  # fmt: skip
    """Recall table and false detections per tile, averaged over the seeds of one brain
    in an R15b-style report (``r15b_aerial_scout.json`` or a probe at another size)."""
    data = json.loads(Path(report_json).read_text(encoding="utf-8"))
    runs = [
        r for r in data["runs"] if r["condition"] == condition and r["alpha"] == alpha
    ]
    if not runs:
        raise ValueError(f"no {condition} width-{alpha:g} brain in {report_json}")
    op = [r["test"]["pooled"]["operating"] for r in runs]
    recall = [float(np.mean([o[t][f"recall {b}"] for o in op]))
              for t in THRESHOLDS for b in SIZE_BINS]  # fmt: skip
    false = [float(np.mean([o[t]["false_per_tile"] for o in op])) for t in THRESHOLDS]
    return {"recall": recall, "false": false, "size": int(runs[0].get("size", 224))}


def missions_for(
    camera: dict[str, Any], frames: float, fa_mult: float
) -> list[Mission]:
    table = tuple(camera["recall"]) + tuple(f * fa_mult for f in camera["false"])
    px_m = camera["size"] * PERSON_AREA_M / FOOTPRINT
    return [replace(m, table=table, px_m=px_m, frames_per_look=frames,
                    operator_confirms=True) for m in r17.held_out_missions(1)]  # fmt: skip


def hand_plan(drones: int) -> Plan:
    return replace(hand_two_tier(drones), sweep_thr=0.6)


def tune_task(task: tuple) -> dict[str, Any]:
    camera, frames, fa_mult, index, generations = task
    m = missions_for(camera, frames, fa_mult)[index]
    out = r17.tune_plan(m, index, generations=generations)
    out.update(frames=frames, fa_mult=fa_mult, camera_size=camera["size"])
    return out


def eval_task(task: tuple) -> tuple[tuple, dict[str, float]]:
    key, camera, frames, fa_mult, index, plan, episodes = task
    m = missions_for(camera, frames, fa_mult)[index]
    res = run_missions(m, plan, r17.mission_seeds(r17.TEST_BASE, index, episodes))
    return key, {k: float(res[k].mean()) for k in ("score", "found", "wrong")}


def eval_tasks(cameras: dict[str, dict], tuned: dict[tuple, dict],
               episodes: int = EPISODES) -> list[tuple]:  # fmt: skip
    """``tuned`` maps (camera name, frames, fa_mult, mission) to a tuned record."""
    out = []
    for (name, frames, mult, i), rec in tuned.items():
        cam = cameras[name]
        drones = r17.held_out_missions(1)[i].drones
        plans = {"OpenSwarm default": OPENSWARM_DEFAULT, "hand two-tier": hand_plan(drones),
                 "tuned per mission": r17.decode_plan(np.array(rec["u"]), drones)}  # fmt: skip
        for system, plan in plans.items():
            out.append(((name, frames, mult, system, i), cam, frames, mult, i, plan,
                        episodes))  # fmt: skip
    return out


def summarise(cells: dict[tuple, dict]) -> dict[str, Any]:
    conds = sorted({k[:3] for k in cells}, key=lambda c: (c[0], c[1], -c[2]))
    rows = []
    for name, frames, mult in conds:
        row: dict[str, Any] = {"camera": name, "frames": frames, "fa_mult": mult}
        for system in ("OpenSwarm default", "hand two-tier", "tuned per mission"):
            vals = [cells[k] for k in cells if k[:4] == (name, frames, mult, system)]
            row[system] = {m: float(np.mean([v[m] for v in vals]))
                           for m in ("score", "found", "wrong")}  # fmt: skip
        rows.append(row)
    need: dict[str, Any] = {}
    for name, frames in sorted({(r["camera"], r["frames"]) for r in rows}):
        ok = [r["fa_mult"] for r in rows if r["camera"] == name and r["frames"] == frames
              and r["tuned per mission"]["found"] >= FOUND_TARGET
              and r["tuned per mission"]["wrong"] <= WRONG_TARGET]  # fmt: skip
        need[f"{name}, {frames:g} frames/look"] = max(ok) if ok else None
    return {"rows": rows, "needed_fa_mult": need}


def render(rep: dict[str, Any], episodes: int = EPISODES) -> str:
    lines = ["# R17 step 4: what the scout brain must reach before a search works", "",
             f"Read-out (fixed in advance): the smallest cut in false detections at which "
             f"tuned searches find the person in ≥ {100 * FOUND_TARGET:.0f}% of episodes "
             f"with ≤ {100 * WRONG_TARGET:.0f}% wrong finds.", ""]  # fmt: skip
    for cond, mult in rep["needed_fa_mult"].items():
        lines.append(f"- **{cond}:** " + (
            f"false detections cut to {mult:g}× ({1 / mult:g} times fewer)" if mult
            else "not reached at any tested cut (down to 100 times fewer)"))  # fmt: skip
    lines += ["", f"Step 1's 40 held-out missions × {episodes} episodes; operator confirms; "
              "found / wrong find, averaged over missions.", "",
              "| Camera | Frames per look | False detections | OpenSwarm default | Hand "
              "two-tier (0.6) | Tuned per mission |", "|---|---|---|---|---|---|"]  # fmt: skip
    for r in rep["rows"]:
        cells = [f"{100 * r[s]['found']:.0f}% / {100 * r[s]['wrong']:.1f}%"
                 for s in ("OpenSwarm default", "hand two-tier", "tuned per mission")]  # fmt: skip
        lines.append(f"| {r['camera']} | {r['frames']:g} | ×{r['fa_mult']:g} | "
                     + " | ".join(cells) + " |")  # fmt: skip
    lines += ["", "Reproduce: `python scripts/r17e_requirements.py report --from "
              "runs:r17e-tune-*`.", ""]  # fmt: skip
    return "\n".join(lines)
