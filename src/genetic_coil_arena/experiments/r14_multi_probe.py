"""R14 probe: with several gas sources, how much room does coordination leave?

Exploratory probe, not a goal test. G13 found that one source needs only 2 bytes of
radio. With ``K`` sources the drones must also share *which* hotspots are already
confirmed and split the cameras, which is harder to hand-design. Systems:

- ``silent``: G13's silent swarm;
- ``hand``: ``MultiTargetSwarm`` (hotspot and confirmation gossip), its knobs
  (``one_each``, camera radius, exclusion radius) tuned on a separate seed block;
  2-byte packets, and the same knobs at 8 bytes;
- ``sensed oracle``: a source becomes known exactly to every drone the moment any gas
  drone smells it, cameras are split greedily and gas drones move on. It bounds what
  better messages and allocation could add to the swarm's own sensing;
- ``oracle``: cameras know every source from the start (no search at all).

The coordination headroom is sensed oracle minus hand, paired by replicate. 30% packet
loss and flow drift 0.03 m per step throughout; 240 steps (2 minutes).
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any

from ..envs.swarm import ScriptedSwarm, SwarmConfig, SwarmEnv, run_swarm
from ..envs.swarm.multi import (
    MultiCodec,
    MultiTargetSwarm,
    OracleSwarm,
    SensedOracleSwarm,
)
from ..evaluation import mean_ci
from ..messages import MessageSpec

KS = (1, 3)
SYSTEMS = ("silent", "hand 2B", "hand 8B", "sensed oracle", "oracle")
GRID = {
    "one_each": (False, True),
    "camera_radius": (2.0, 4.5),
    "exclude": (5.0, 8.0, 11.0),
}
POS_BITS = {2: 5, 8: 20}
LOSS, DRIFT = 0.3, 0.03
TUNE_BASE, TEST_BASE, STRIDE = 8_000_000_000, 8_500_000_000, 10_000
N_REPS, PER_REP, TUNE_EPISODES = 10, 300, 600


def config(k: int, budget: int = 2) -> SwarmConfig:
    msg = MessageSpec("discrete", budget, 256)
    return replace(SwarmConfig(), n_sources=k, loss=LOSS, drift=DRIFT, message=msg)


def make(system: str, k: int, knobs: dict[str, Any]) -> tuple[SwarmConfig, Any]:
    budget = 8 if system.endswith("8B") else 2
    cfg = config(k, budget)
    if system == "silent":
        return cfg, ScriptedSwarm(cfg)
    if system.startswith("hand"):
        codec = MultiCodec(budget, POS_BITS[budget], cfg.arena)
        return cfg, MultiTargetSwarm(cfg, codec, **knobs)
    if system == "sensed oracle":
        return cfg, SensedOracleSwarm(cfg)
    return cfg, OracleSwarm(cfg)


def play(system: str, k: int, knobs: dict[str, Any], seeds: list[int]) -> dict:
    cfg, policy = make(system, k, knobs)
    return run_swarm(SwarmEnv(cfg), policy, seeds)


def tune(k: int = 3, episodes: int = TUNE_EPISODES) -> dict[str, Any]:
    """Hand knobs with the most sources found at 2 B on the tuning block."""
    seeds = [TUNE_BASE + i for i in range(episodes)]
    best: tuple[float, dict[str, Any]] = (-1.0, {})
    for values in product(*GRID.values()):
        knobs = dict(zip(GRID, values))
        found = float(play("hand 2B", k, knobs, seeds)["n_found"].mean())
        if found > best[0]:
            best = (found, knobs)
    return {**best[1], "tune_found": best[0]}


def run(knobs: dict[str, Any], reps: int = N_REPS, per_rep: int = PER_REP) -> dict:
    hand = {k: v for k, v in knobs.items() if k in GRID}
    out: dict[str, Any] = {}
    for k in KS:
        for system in SYSTEMS:
            rows = []
            for r in range(reps):
                seeds = [TEST_BASE + r * STRIDE + i for i in range(per_rep)]
                got = play(system, k, hand, seeds)
                rows.append(
                    {
                        "found": float(got["n_found"].mean()),
                        "all": float(got["confirmed"].mean()),
                        "steps": float(got["steps"].mean()),
                    }
                )
            out[f"K={k}|{system}"] = rows
    return out


def summarise(cells: dict[str, list[dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, rows in cells.items():
        out[key] = {m: mean_ci([r[m] for r in rows]) for m in ("found", "all", "steps")}
    for k in KS:
        hand = cells[f"K={k}|hand 2B"]
        for ref in ("sensed oracle", "silent"):
            other = cells[f"K={k}|{ref}"]
            for m in ("found", "all", "steps"):
                d = [o[m] - h[m] for o, h in zip(other, hand)]
                out[f"K={k}|{ref} - hand 2B|{m}"] = mean_ci(d)
    return out


def render(rep: dict[str, Any]) -> str:
    s = rep["summary"]
    lines = [
        "# R14 probe: several gas sources, how much room does coordination leave?",
        "",
        "Exploratory probe (not a pre-registered goal). 3 gas and 3 camera drones, 30 m "
        "arena, 240 steps (2 min), 30% packet loss, drift 0.03 m per step. Mean ± 95% "
        f"t-CI over {rep['reps']} replicates of {rep['per_rep']} episodes (same seeds "
        "for every system).",
        "",
        "| Sources | System | Sources found | All found | Steps |",
        "|---|---|---|---|---|",
    ]
    for k in KS:
        for system in SYSTEMS:
            c = s[f"K={k}|{system}"]
            lines.append(
                f"| {k} | {system} | {c['found'][0]:.2f} ± {c['found'][1]:.2f} "
                f"| {100 * c['all'][0]:.1f} ± {100 * c['all'][1]:.1f}% "
                f"| {c['steps'][0]:.0f} ± {c['steps'][1]:.0f} |"
            )
    lines += ["", "Differences, paired by replicate:", ""]
    for k in KS:
        for ref in ("sensed oracle", "silent"):
            parts = []
            for m, scale, unit in (
                ("found", 1, ""),
                ("all", 100, " points"),
                ("steps", 1, " steps"),
            ):
                v = s[f"K={k}|{ref} - hand 2B|{m}"]
                parts.append(f"{m} {scale * v[0]:+.2f} ± {scale * v[1]:.2f}{unit}")
            lines.append(f"- K = {k}, {ref} minus hand 2 B: " + "; ".join(parts) + ".")
    k = rep["knobs"]
    lines += [
        "",
        f"Hand knobs tuned at K = 3 on a separate block: one camera per hotspot "
        f"{k['one_each']}, camera radius {k['camera_radius']:g} m, exclusion "
        f"{k['exclude']:g} m (tuning: {k['tune_found']:.2f} sources found).",
        "",
        "Reproduce: `python scripts/r14_multi_probe.py`.",
        "",
    ]
    return "\n".join(lines)
