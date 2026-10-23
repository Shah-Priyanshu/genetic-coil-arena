"""Need-to-talk baselines for the squad: does the centre's information help?

Runs scripted specialists for each window radius r and centre policy, plus the
full-view oracle, on held-out seeds, and reports team food per episode with 95%
confidence intervals (``evaluation.stats.mean_ci``). Optionally measures
single-process throughput in agent-steps per second.

Used by ``genetic-coil squad-benchmark`` and ``scripts/need_to_talk.py``.
"""

import datetime
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from ..envs.squad import (
    GreedyPolicy,
    HandCodedCentre,
    NoCommPolicy,
    OraclePolicy,
    RandomCentre,
    RandomPolicy,
    SilentCentre,
    SquadConfig,
    SquadEnv,
    measure_throughput,
    run_episodes,
)
from ..evaluation.stats import mean_ci
from ..evo.baselines import BENCHMARK_SEED_OFFSET

RADII = (3, 5, 7)


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _fmt(values) -> str:
    mean, half = mean_ci(values)
    return f"{mean:.2f} ± {half:.2f}"


def need_to_talk(n_seeds: int = 100, radii: Sequence[int] = RADII) -> list[dict]:
    """Rows of (setting, team food, survivors, steps) over ``n_seeds`` held-out seeds."""
    seeds = list(range(BENCHMARK_SEED_OFFSET, BENCHMARK_SEED_OFFSET + n_seeds))
    rows = []

    def add(r, specialist, centre, label, r_label=None):
        env = SquadEnv(SquadConfig(obs_radius=r))
        res = run_episodes(env, specialist, centre, seeds)
        rows.append(
            {
                "r": r if r_label is None else r_label,
                "setting": label,
                "food": _fmt(res["team_food"]),
                "survivors": _fmt(res["survivors"]),
                "steps": _fmt(res["steps"]),
            }
        )

    for r in radii:
        add(r, GreedyPolicy(), HandCodedCentre(), "greedy + hand-coded centre (3 bits)")
        add(r, GreedyPolicy(), SilentCentre(), "greedy + silent centre")
        add(r, GreedyPolicy(), RandomCentre(), "greedy + random centre")
        add(r, NoCommPolicy(), None, "no-comm greedy (ignores messages)")
    add(5, OraclePolicy(), None, "oracle (specialists see the full board)", "full")
    add(5, RandomPolicy(), None, "random actions", "-")
    return rows


def squad_throughput(batch_size: int = 256) -> dict[str, dict[str, float]]:
    """Agent-steps per second for the main rollouts (one process)."""
    return {
        "greedy + hand-coded centre": measure_throughput(
            None, GreedyPolicy(), HandCodedCentre(), batch_size=batch_size
        ),
        "oracle": measure_throughput(None, OraclePolicy(), None, batch_size=batch_size),
        "engine only (random actions)": measure_throughput(None, batch_size=batch_size),
    }


def run_need_to_talk(
    n_seeds: int = 100,
    radii: Sequence[int] = RADII,
    batch_size: int = 256,
    throughput: bool = True,
    out: Optional[Path] = None,
) -> str:
    """Run the benchmark and return a Markdown report, also written to ``out``."""
    rows = need_to_talk(n_seeds, radii)
    speeds = squad_throughput(batch_size) if throughput else {}

    lines = [
        "# Squad: need to talk",
        "",
        f"- Date: {datetime.date.today().isoformat()}",
        f"- Commit: {_commit()}",
        f"- Seeds: {n_seeds} held-out episodes per row, seeds "
        f"{BENCHMARK_SEED_OFFSET}..{BENCHMARK_SEED_OFFSET + n_seeds - 1}",
        f"- CPU: {_cpu_name()} ({os.cpu_count()} logical cores visible; "
        "throughput is one process on one core)",
        f"- Python {sys.version.split()[0]}",
        "- Squad defaults: 21×21 walls, 3 snakes, 2 food per colour, health 60, T = 300, "
        "message = 1 discrete symbol, V = 8, every step",
        "",
        "Team food per episode (all snakes), mean ± 95% CI over episodes.",
        "",
        "| r | Setting | Team food | Survivors at end | Episode steps |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['r']} | {row['setting']} | {row['food']} | {row['survivors']} "
            f"| {row['steps']} |"
        )
    if speeds:
        lines += [
            "",
            f"Throughput (batch {batch_size}, best of 3, living-snake steps per second, "
            "one process):",
            "",
            "| Rollout | Agent-steps/s |",
            "|---|---|",
        ]
        for name, t in speeds.items():
            lines.append(f"| {name} | {t['agent_steps_per_sec']:,.0f} |")
    report = "\n".join(lines) + "\n"
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
    return report
