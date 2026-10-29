"""R14 step 4a (bench preparation): do the swarm brains keep their gain in int8?

The ESP32-S3 runs integer arithmetic far faster than float, and its ML libraries
(ESP-DL, TFLite Micro) run int8 networks. ``learning.Int8MLP`` runs a brain the way the
chip would: int8 weights per layer, int8 inputs (fixed scale ``INPUT_SCALE``), int32
accumulators and an int8 tanh table. This module plays the trained brains (R14 step 2
and the randomised set of step 3) both ways on the same test episodes, and writes the
chosen brains as a C header for the bench (``c_header``).

Bar (fixed before the run): at the nominal setting the int8 brains find all three leaks
within 1 point of the float brains (mean over each set's runs), and still beat the hand
design with the replicate CI above 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..envs.swarm import SwarmEnv, run_swarm
from ..envs.swarm.brains import N_FEATURES
from ..evaluation import mean_ci
from ..learning.quant import Int8MLP
from . import r14_brains as rb
from . import r14_robust as rr

INPUT_SCALE = 1 / 64  # inputs in [-2, 2)
CONDITIONS = ("nominal", "loss 70%", "2 drones down")
LOSS_BAR = 0.01


def play(task: tuple) -> tuple[tuple, list[float]]:
    """One (condition, set, run, int8) cell -> per-replicate all-found rates."""
    key, cond, run, int8, reps, per_rep = task
    over = rr.CONDITIONS[cond]
    eps = rr.seeds(reps, per_rep)
    if run is None:
        got = rr.play_ref("hand 2B", over, eps)
    else:
        settings = rb.BrainSettings(**run["settings"])
        cfg, pol = rb.make_policy(settings, rr.K, over)
        if int8:
            pol.gas_net = Int8MLP(pol.gas_net, INPUT_SCALE)
            pol.cam_net = Int8MLP(pol.cam_net, INPUT_SCALE)
        gas_p, cam_p = rb.split(run["theta"], pol)
        pol.set_params(gas_p, cam_p, len(eps))
        got = run_swarm(SwarmEnv(cfg), pol, eps)
    return key, got["confirmed"].astype(float).reshape(reps, per_rep).mean(1).tolist()


def tasks(sets: dict[str, list[dict]], reps: int, per_rep: int) -> list[tuple]:
    out = []
    for cond in CONDITIONS:
        out.append(((cond, "hand", -1, False), cond, None, False, reps, per_rep))
        for name, runs in sets.items():
            for r in runs:
                for int8 in (False, True):
                    out.append(((cond, name, r["run"], int8), cond, r, int8, reps,
                                per_rep))  # fmt: skip
    return out


def summarise(cells: dict[tuple, list[float]], sets: dict[str, list[dict]]) -> dict:
    out: dict[str, Any] = {"by_condition": {}, "sets": list(sets)}
    passes = []
    for cond in CONDITIONS:
        hand = np.array(cells[(cond, "hand", -1, False)])
        row: dict[str, Any] = {"hand": float(hand.mean())}
        for name, runs in sets.items():
            f = np.array([cells[(cond, name, r["run"], False)] for r in runs])
            q = np.array([cells[(cond, name, r["run"], True)] for r in runs])
            row[name] = {
                "float": float(f.mean()), "int8": float(q.mean()),
                "int8_minus_float": mean_ci((q.mean(0) - f.mean(0)).tolist()),
                "int8_minus_hand": mean_ci((q.mean(0) - hand).tolist()),
                "runs_loss": (q.mean(1) - f.mean(1)).tolist(),
            }  # fmt: skip
            if cond == "nominal":
                d, h = row[name]["int8_minus_float"], row[name]["int8_minus_hand"]
                passes.append(d[0] >= -LOSS_BAR and h[0] - h[1] > 0)
        out["by_condition"][cond] = row
    out["pass"] = bool(passes and all(passes))
    return out


def render(rep: dict[str, Any], episodes: int) -> str:
    lines = ["# R14 step 4a: the swarm brains in int8", "",
             f"**{'PASS' if rep['pass'] else 'FAIL'}.** At nominal the int8 brains find all "
             f"leaks within {100 * LOSS_BAR:.0f} point of the float brains and still beat the "
             "hand design (replicate CI above 0)." if rep["pass"] else
             f"**FAIL.** At nominal the int8 brains lose more than {100 * LOSS_BAR:.0f} point "
             "or no longer beat the hand design.", "",
             f"All three leaks found, 3 sources, {episodes} episodes per cell (int8 − float: "
             "replicate mean ± 95% CI).", "",
             "| Condition | Hand | " + " | ".join(
                 f"{s} float | {s} int8 (int8 − float)" for s in rep["sets"]) + " |",
             "|---|---|" + "---|---|" * len(rep["sets"])]  # fmt: skip
    for cond, row in rep["by_condition"].items():
        cells = [f"{100 * row['hand']:.1f}%"]
        for s in rep["sets"]:
            d = row[s]["int8_minus_float"]
            cells += [f"{100 * row[s]['float']:.1f}%",
                      f"{100 * row[s]['int8']:.1f}% ({100 * d[0]:+.1f} ± {100 * d[1]:.1f})"]  # fmt: skip
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    lines += ["", "Reproduce: `python scripts/r14_int8.py --set step2=runs:r14b-train-* "
              "--set random=runs:r14c-train-*`.", ""]  # fmt: skip
    return "\n".join(lines)


FEATURES = (
    "0-1 offset to the candidate / arena", "2 distance / arena", "3 sqrt(value)",
    "4-6 kind one-hot (explore, hotspot, own best)", "7 hotspot rank / m",
    "8 distance to current destination / arena", "9 step / max_steps",
    "10 confirmed sources / n_sources", "11-13 index within role one-hot",
    "14 open hotspots / m",
)  # fmt: skip


def c_header(run: dict[str, Any], hidden: int) -> str:
    """The run's gas and camera brains as int8 C arrays for an ESP32."""
    net = Int8MLP(rb.brain(hidden), INPUT_SCALE)
    ng = net.n_params
    theta = np.asarray(run["theta"], dtype=np.float64)
    lines = [
        "// Swarm brains (R14), int8. Generated by scripts/r14_int8.py; do not edit.",
        f"// Training run {run['run']} (best validation {run['best_val']:.3f}).",
        "// Per drone and candidate destination: score = net(features); fly to the",
        "// valid candidate with the highest score. Features (float, then int8 with",
        f"// q = clamp(round(x * {round(1 / INPUT_SCALE)}), -128, 127)):",
        *[f"//   {f}" for f in FEATURES],
        "// Layer: acc = sum(q_in * w) + b (int32); real = acc * scale; hidden layers",
        "// use q = clamp(round(tanh(real) * 127), -127, 127).",
        "#pragma once",
        "#include <stdint.h>",
        f"#define BRAIN_IN {N_FEATURES}",
        f"#define BRAIN_HIDDEN {hidden}",
    ]
    for name, params in (("gas", theta[:ng]), ("cam", theta[ng:])):
        for k, layer in enumerate(net.quantised(params)):
            w = layer["w"]
            lines.append(
                f"static const int8_t {name}_w{k}[{w.shape[0]}][{w.shape[1]}] = {{"
            )
            lines += ["  {" + ", ".join(str(int(v)) for v in row) + "}," for row in w]
            lines.append("};")
            lines.append(f"static const int32_t {name}_b{k}[{len(layer['b'])}] = {{"
                         + ", ".join(str(int(v)) for v in layer["b"]) + "};")  # fmt: skip
            lines.append(f"static const float {name}_scale{k} = {layer['scale']:.9g}f;")
    return "\n".join(lines) + "\n"
