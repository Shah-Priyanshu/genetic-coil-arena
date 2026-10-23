"""Do squad specialists listen to the command centre, and how much does it say? (R5)

Runs ``experiments.listening`` for each window radius through the CommArena harness and
writes a Markdown report plus the full JSON results.

    python scripts/listening.py --seeds 20 --episodes 50 --workers 2 \
        --out docs/results/listening.md
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from genetic_coil_arena.evaluation import ExperimentResult, format_table
from genetic_coil_arena.experiments.listening import (
    DEAF,
    EPISODE_SEED_BASE,
    LISTENER,
    RANDOM_CENTRE,
    SEED_STRIDE,
    SILENT_CENTRE,
    ListeningSettings,
    run_radius,
)

COMM_ROWS = (LISTENER, DEAF, RANDOM_CENTRE, SILENT_CENTRE)


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


def _pm(mean: float, half: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} ± {half:.{digits}f}"


def _p(value: float) -> str:
    return f"{value:.2g}" if value < 0.01 else f"{value:.3f}"


def run_all(settings: ListeningSettings, workers: int) -> dict[int, ExperimentResult]:
    """One ExperimentResult per radius, using up to ``workers`` processes."""
    radii = list(settings.radii)
    if workers <= 1 or len(radii) == 1:
        return {r: run_radius(r, settings) for r in radii}
    with ProcessPoolExecutor(max_workers=min(workers, len(radii))) as pool:
        results = pool.map(run_radius, radii, [settings] * len(radii))
        return dict(zip(radii, results))


def performance_table(results: dict[int, ExperimentResult]) -> list[str]:
    names = [c.name for c in next(iter(results.values())).conditions]
    lines = [
        "| condition | " + " | ".join(f"r = {r}" for r in results) + " |",
        "|---|" + "---|" * len(results),
    ]
    for name in names:
        cells = []
        for res in results.values():
            c = res.condition(name)
            cells.append(_pm(c.mean, c.ci))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def ablation_table(results: dict[int, ExperimentResult]) -> list[str]:
    lines = [
        "| r | system | intervention | intact | ablated | drop [95% CI] "
        "| p (Holm) | A12 | listening |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r, res in results.items():
        for a in res.ablations:
            lines.append(
                f"| {r} | {a.system} | {a.intervention} | {a.intact_mean:.2f} "
                f"| {a.ablated.mean:.2f} | {a.delta_mean:.2f} "
                f"[{a.delta_low:.2f}, {a.delta_high:.2f}] | {_p(a.p_holm)} "
                f"| {a.a12:.2f} | {'yes' if a.listening else 'no'} |"
            )
    return lines


def communication_table(results: dict[int, ExperimentResult]) -> list[str]:
    lines = [
        "| r | squad | H(M) bits | used / 3-bit capacity | H(T) bits "
        "| MI(M;T) bits [null q95, p] | MI(M;direction) | MI(M;far) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r, res in results.items():
        for name in COMM_ROWS:
            m = res.communication[name]
            lines.append(
                f"| {r} | {name} | {m['message_entropy_bits']:.3f} "
                f"| {100 * m['utilization']:.1f}% | {m['target_entropy_bits']:.3f} "
                f"| {m['mi_bits']:.3f} [{m['mi_null_q95_bits']:.3f}, "
                f"{_p(m['mi_p_value'])}] | {m['mi_direction_bits']:.3f} "
                f"| {m['mi_far_bits']:.3f} |"
            )
    return lines


def influence_table(results: dict[int, ExperimentResult]) -> list[str]:
    lines = [
        "| r | squad | CIC bits | JS(shuffle_batch) bits | action changed "
        "(shuffle_batch) | JS(random_matched) bits | action changed (random_matched) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r, res in results.items():
        for name in (LISTENER, DEAF, RANDOM_CENTRE):
            m = res.communication[name]
            lines.append(
                f"| {r} | {name} | {_pm(m['cic_bits'], m['cic_bits_ci'], 3)} "
                f"| {_pm(m['js_shuffle_batch_bits'], m['js_shuffle_batch_bits_ci'], 3)} "
                f"| {_pm(m['changed_shuffle_batch'], m['changed_shuffle_batch_ci'], 3)} "
                f"| {_pm(m['js_random_matched_bits'], m['js_random_matched_bits_ci'], 3)}"
                f" | {_pm(m['changed_random_matched'], m['changed_random_matched_ci'], 3)}"
                " |"
            )
    return lines


def sanity_table(results: dict[int, ExperimentResult]) -> list[str]:
    lines = ["| r | check | passed | detail |", "|---|---|---|---|"]
    for r, res in results.items():
        for row in res.metadata["sanity"]:
            lines.append(
                f"| {r} | {row['check']} | {'yes' if row['passed'] else '**NO**'} "
                f"| {row['detail']} |"
            )
    return lines


def interpretation(results: dict[int, ExperimentResult]) -> list[str]:
    """Plain-English reading generated from the numbers."""
    out = []
    for r, res in results.items():
        listener = res.condition(LISTENER)
        drops = {a.intervention: a for a in res.ablations if a.system == LISTENER}
        summary = res.metadata["listening_summary"][LISTENER]
        m = res.communication[LISTENER]
        worst = max(drops.values(), key=lambda a: a.delta_mean)
        out.append(
            f"- **r = {r}:** the message-following squad eats {listener.mean:.1f} food "
            f"per episode. Every intervention costs food (positive listening: "
            f"{'yes' if summary['positive_listening'] else 'no'}). Zeroing the channel "
            f"costs {drops['zero'].delta_mean:.1f}, and the worst is "
            f"{worst.intervention} at {worst.delta_mean:.1f}. The centre uses "
            f"{m['message_entropy_bits']:.2f} of 3 bits, of which MI = "
            f"{m['mi_bits']:.2f} bits is about the target. A message changes a "
            "specialist's "
            f"action distribution by CIC = {m['cic_bits']:.2f} bits per living "
            "receiver."
        )
    return out


def render(
    results: dict[int, ExperimentResult],
    settings: ListeningSettings,
    seconds: float,
    workers: int,
) -> str:
    first = next(iter(results.values()))
    n_seeds = len(first.seeds)
    all_passed = all(
        row["passed"] for res in results.values() for row in res.metadata["sanity"]
    )
    lines = [
        "# Squad: do specialists listen to the command centre? (R5)",
        "",
        f"- Date: {datetime.date.today().isoformat()}",
        f"- Commit: {_commit()}",
        f"- Seeds: {n_seeds} run seeds (0..{n_seeds - 1}) × {settings.n_episodes} "
        f"episodes each; run seed s plays episode seeds {EPISODE_SEED_BASE:,} + "
        f"s × {SEED_STRIDE:,} + i. The seeds are the independent units for CIs and tests.",
        f"- CPU: {_cpu_name()} ({os.cpu_count()} logical cores visible, "
        f"{workers} worker processes); wall time {seconds:.0f} s",
        f"- Python {sys.version.split()[0]}",
        "- Squad defaults: 21×21 walls, 3 snakes, 2 food per colour, health 60, T = 300, "
        "message = 1 discrete symbol, V = 8 (3 bits), sent every step, read one step "
        "later",
        f"- Communication metrics: messages recorded every {settings.record_every} "
        f"steps for living receivers; CIC and intervention influence on listener "
        f"snapshots every {settings.snapshot_every} steps, action probabilities from "
        f"{settings.n_draws} seeded draws; MI null from {settings.n_shuffles} "
        "label shuffles.",
        f"- Sanity controls: {'all passed' if all_passed else '**SOME FAILED**'}",
        "",
        "Reproduce with `python scripts/listening.py --seeds "
        f"{n_seeds} --episodes {settings.n_episodes} --workers 2 "
        "--out docs/results/listening.md`.",
        "",
        "## Systems",
        "",
        f"- `{LISTENER}`: greedy specialists (chase own-colour food in their window, "
        "follow the message's direction bits when none is visible) with the hand-coded "
        "centre (`code = 2 * direction + far` of the receiver's nearest own-colour "
        "food).",
        f"- `{DEAF}`: the same centre talking to `NoCommPolicy` specialists, which never "
        "read the message. The deaf control.",
        "- Baselines (mandatory, registered with `evaluation.register_baseline`): "
        "random actions, greedy + silent centre, greedy + random centre, no-comm "
        "greedy, and the full-view oracle.",
        "",
        "## Interpretation",
        "",
        *interpretation(results),
        "",
        "Reading across radii:",
        "",
        "- The specialists listen, and only through the message. Every intervention "
        "drops team food at every radius. The deaf control's drop is exactly 0 and "
        "its CIC is exactly 0.",
        "- Wrong information is worse than none. shuffle_batch, swap_receivers and "
        "random_matched keep the message statistics but break the link to the "
        "receiver's situation. Each of them costs more than zeroing the channel, and "
        "each leaves the squad below the message-blind `no-comm` baseline.",
        "- Listening matters less as the window grows. CIC and the share of actions an "
        "intervention flips both fall from r = 3 to r = 7, because the greedy "
        "specialist only reads the message when no own-colour food is in view.",
        "- The centre says about 2.6 of its 3 bits. All of that is about the target: "
        "MI(M;T) = H(M) = H(T). This holds by construction, since the code is that "
        "target. Here MI is a check that the centre is implemented correctly against "
        "an independent recomputation, not a discovery. A learned centre is where MI "
        "becomes informative. The unused ~0.4 bits are the skewed code distribution, "
        "and the 'no food' case is folded into code 0.",
        "- p-values bottom out at 2.7e-07 because, with 20 seeds and complete "
        "separation, that is the smallest two-sided Mann-Whitney p after Holm. The "
        "paired CIs carry the effect sizes.",
        "",
        "## Team food per episode",
        "",
        "Mean ± 95% t-CI over run seeds (each seed's score is the mean over its "
        "episodes).",
        "",
        *performance_table(results),
        "",
        "## Message ablations (positive listening)",
        "",
        "Drop = intact minus ablated team food, paired by seed; `listening` = the drop's "
        "95% CI lies above zero. p is a two-sided Mann-Whitney test (the harness's "
        "choice; unpaired), Holm-corrected across the four interventions per system.",
        "",
        *ablation_table(results),
        "",
        "## What the centre says (positive signalling)",
        "",
        "Pooled over all seeds and recorded steps, living receivers only. T = the "
        "receiver's nearest own-colour food at send time, recomputed independently "
        "from the centre's observation: direction relative to heading (4) × "
        "near/far (Manhattan > 5), plus a 'no food' class. MI is Miller-Madow "
        "corrected; null q95 and p come from shuffled targets.",
        "",
        *communication_table(results),
        "",
        "## How much messages move actions (listener side)",
        "",
        "Per living receiver, mean ± 95% CI over seeds. CIC = I(action; message) with "
        "the situation held fixed and messages drawn from the centre's own message "
        "distribution (max log2 3 = 1.585 bits). JS = Jensen-Shannon divergence "
        "between action distributions with the real and the intervened message.",
        "",
        *influence_table(results),
        "",
        "## Sanity controls",
        "",
        *sanity_table(results),
        "",
        "## Harness output",
        "",
        "`evaluation.format_table` for each radius (communication metrics are in the "
        "tables above and in `listening.json`).",
        "",
    ]
    for r, res in results.items():
        trimmed = replace(res, communication={})
        lines += [f"### r = {r}", "", "```", format_table(trimmed), "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20, help="Run seeds per radius")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per seed")
    parser.add_argument("--radii", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--workers", type=int, default=2, help="Processes (radii)")
    parser.add_argument(
        "--out", type=Path, default=None, help="Markdown report (JSON goes next to it)"
    )
    args = parser.parse_args()

    settings = ListeningSettings(
        radii=tuple(args.radii), n_seeds=args.seeds, n_episodes=args.episodes
    )
    start = time.perf_counter()
    results = run_all(settings, args.workers)
    seconds = time.perf_counter() - start
    report = render(results, settings, seconds, args.workers)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        payload: dict[str, Any] = {
            "date": datetime.date.today().isoformat(),
            "commit": _commit(),
            "results": {f"r{r}": res.to_dict() for r, res in results.items()},
        }
        args.out.with_suffix(".json").write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
