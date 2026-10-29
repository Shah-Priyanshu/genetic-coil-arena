"""R14 step 2: can learned per-role brains coordinate a swarm better than the hand design?

Exploratory test (the user allows exploration without approval; the bar below was fixed
before any training). The 3-source probe found that a tuned hand-designed swarm finds
all three leaks in 61.9% of episodes, against 93.1% for ideal sharing and allocation of
what the swarm smells. ``LearnedSwarm`` keeps the hand design's radio and memory and
learns only the destination choice, with one brain for gas drones and one for cameras.

Training: OpenAI-ES (``learning.EvolutionStrategy``) on 3-source episodes (30% loss,
drift 0.03, 2-byte packets). Fitness = sources found + 1 if all were found, averaged
over ``episodes`` common-random-number episodes per member. The checkpoint is kept by
validation on separate seeds.

Evaluation: the probe's test seeds (10 replicates × 300 episodes), with 1 to 4 sources
(trained on 3 only). Learned brains of every run against the hand design (the probe's
tuned knobs) and the sensed oracle on the same episodes.

Good sign (fixed in advance): at 3 sources, the learned brains' all-found rate, averaged
over training runs, beats the hand design by at least 10 points, with the run CI's lower
bound above 0.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..envs.swarm import SwarmEnv
from ..envs.swarm.brains import LearnedSwarm, brain
from ..envs.swarm.multi import MultiCodec
from ..evaluation import mean_ci
from ..learning import EvolutionStrategy
from . import r14_multi_probe as probe

TRAIN_BASE, VAL_BASE, RUN_STRIDE = 9_000_000_000, 9_500_000_000, 10_000_000
HAND_KNOBS = {"one_each": False, "camera_radius": 4.5, "exclude": 8.0}  # probe-tuned
EVAL_KS = (1, 2, 3, 4)
GOOD_SIGN_POINTS = 0.10


@dataclass(frozen=True)
class BrainSettings:
    generations: int = 1500
    population: int = 64
    episodes: int = 16
    sigma: float = 0.1
    lr: float = 0.03
    weight_decay: float = 0.001
    hidden: int = 16
    eval_every: int = 25
    val_episodes: int = 120
    k: int = 3
    progress_s: float = 600.0


def make_policy(settings: BrainSettings, k: int) -> tuple[Any, LearnedSwarm]:
    cfg = probe.config(k, 2)
    pol = LearnedSwarm(
        cfg, MultiCodec(2, probe.POS_BITS[2], cfg.arena),
        brain(settings.hidden), brain(settings.hidden), **HAND_KNOBS,
    )  # fmt: skip
    return cfg, pol


def split(theta: np.ndarray, pol: LearnedSwarm) -> tuple[np.ndarray, np.ndarray]:
    theta = np.atleast_2d(theta)
    ng = pol.gas_net.n_params
    return theta[:, :ng], theta[:, ng:]


def rollout(
    settings: BrainSettings, k: int, thetas: np.ndarray, seeds: list[int]
) -> np.ndarray:
    """(P, E) episode scores (found + all found) of every member on the same seeds."""
    thetas = np.atleast_2d(thetas)
    cfg, pol = make_policy(settings, k)
    gas_p, cam_p = split(thetas, pol)
    pol.set_params(gas_p, cam_p, len(seeds))
    all_seeds = list(seeds) * len(thetas)
    env = SwarmEnv(cfg)
    obs = env.reset(all_seeds)
    pol.reset(all_seeds)
    while True:
        actions, packets = pol.act(obs)
        obs, _, done, info = env.step(actions, packets)
        if done.all():
            break
    score = info["n_found"] + info["confirmed"].astype(float)
    return score.reshape(len(thetas), len(seeds))


def train_run(run: int, settings: BrainSettings) -> dict[str, Any]:
    _, pol = make_policy(settings, settings.k)
    rng = np.random.default_rng([run, 14])
    theta0 = np.concatenate([pol.gas_net.init(rng), pol.cam_net.init(rng)])
    es = EvolutionStrategy(
        theta0, population=settings.population, sigma=settings.sigma, lr=settings.lr,
        weight_decay=settings.weight_decay, seed=run,
    )  # fmt: skip
    val_seeds = [VAL_BASE + run * 10_000 + i for i in range(settings.val_episodes)]
    start = time.perf_counter()
    best = (-np.inf, es.theta.copy(), 0)
    curve: list[dict[str, float]] = []
    last = 0.0
    for g in range(settings.generations + 1):
        if g % settings.eval_every == 0 or g == settings.generations:
            val = float(rollout(settings, settings.k, es.theta, val_seeds).mean())
            curve.append({"generation": g, "val": val,
                          "wall_s": time.perf_counter() - start})  # fmt: skip
            if val > best[0]:
                best = (val, es.theta.copy(), g)
            now = time.perf_counter() - start
            if settings.progress_s and (now - last >= settings.progress_s or g == 0):
                last = now
                print(f"[progress] run {run}: gen {g}/{settings.generations}, val "
                      f"{val:.3f}, best {best[0]:.3f} at {best[2]}, {now:.0f} s",
                      flush=True)  # fmt: skip
        if g == settings.generations:
            break
        seeds = [
            TRAIN_BASE + run * RUN_STRIDE + g * settings.episodes + i
            for i in range(settings.episodes)
        ]
        cands = es.ask()
        es.tell(rollout(settings, settings.k, cands, seeds).mean(axis=1))
    return {
        "run": run,
        "theta": best[1],
        "best_val": float(best[0]),
        "best_generation": best[2],
        "curve": curve,
        "seconds": time.perf_counter() - start,
        "n_params": int(theta0.size),
        "settings": asdict(settings),
    }


def save_run(out: dict[str, Any], folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"r14b_brains_run{out['run']}"
    np.savez(stem.with_suffix(".npz"), theta=out["theta"])
    meta = {k: v for k, v in out.items() if k != "theta"}
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return stem


def load_runs(folders: list[Path]) -> list[dict[str, Any]]:
    runs = []
    for folder in folders:
        for meta in sorted(folder.glob("r14b_brains_run*.json")):
            d = json.loads(meta.read_text(encoding="utf-8"))
            with np.load(meta.with_suffix(".npz")) as z:
                d["theta"] = z["theta"]
            runs.append(d)
    return sorted(runs, key=lambda r: r["run"])


def evaluate(
    runs: list[dict[str, Any]], reps: int = probe.N_REPS, per_rep: int = probe.PER_REP
) -> dict[str, Any]:
    settings = BrainSettings(**runs[0]["settings"])
    out: dict[str, Any] = {"by_k": {}, "runs": [r["run"] for r in runs]}
    for k in EVAL_KS:
        seeds = [probe.TEST_BASE + r * probe.STRIDE + i
                 for r in range(reps) for i in range(per_rep)]  # fmt: skip
        learned = []
        for r in runs:
            cfg, pol = make_policy(settings, k)
            gas_p, cam_p = split(r["theta"], pol)
            pol.set_params(gas_p, cam_p, len(seeds))
            from ..envs.swarm import run_swarm

            got = run_swarm(SwarmEnv(cfg), pol, seeds)
            learned.append({"found": float(got["n_found"].mean()),
                            "all": float(got["confirmed"].mean()),
                            "steps": float(got["steps"].mean())})  # fmt: skip
        refs = {}
        for system in ("silent", "hand 2B", "sensed oracle"):
            got = probe.play(system, k, HAND_KNOBS, seeds)
            refs[system] = {"found": float(got["n_found"].mean()),
                            "all": float(got["confirmed"].mean()),
                            "steps": float(got["steps"].mean())}  # fmt: skip
        diff = [x["all"] - refs["hand 2B"]["all"] for x in learned]
        out["by_k"][k] = {
            "learned": {
                m: mean_ci([x[m] for x in learned]) for m in ("found", "all", "steps")
            },
            "learned_runs": learned,
            "refs": refs,
            "all_minus_hand": mean_ci(diff),
        }
    d = out["by_k"][3]["all_minus_hand"]
    out["good_sign"] = bool(d[0] >= GOOD_SIGN_POINTS and d[0] - d[1] > 0)
    return out


def render(rep: dict[str, Any], runs: list[dict[str, Any]]) -> str:
    g = rep["by_k"][3]["all_minus_hand"]
    lines = [
        "# R14 step 2: learned per-role brains coordinate the swarm",
        "",
        f"**{'GOOD SIGN' if rep['good_sign'] else 'NO GOOD SIGN'}.** At 3 sources the learned "
        f"brains find all leaks {100 * g[0]:+.1f} ± {100 * g[1]:.1f} points more often than "
        f"the tuned hand design (bar: ≥ +{100 * GOOD_SIGN_POINTS:.0f} with the CI above 0); "
        f"{len(runs)} training runs, each evaluated on the probe's 3,000 test episodes.",
        "",
        "| Sources | Learned brains (mean ± CI over runs) | Hand design | Sensed oracle | Silent |",
        "|---|---|---|---|---|",
    ]
    for k, r in rep["by_k"].items():
        lr = r["learned"]
        cells = [
            f"{lr['found'][0]:.2f} found, {100 * lr['all'][0]:.1f} ± {100 * lr['all'][1]:.1f}% all"
        ]
        for s in ("hand 2B", "sensed oracle", "silent"):
            x = r["refs"][s]
            cells.append(f"{x['found']:.2f} found, {100 * x['all']:.1f}% all")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Trained on 3 sources only; 1, 2 and 4 test generalisation to other mission sizes.",
        "",
        "| Run | Best validation (found + all) | at generation | Training time |",
        "|---|---|---|---|",
    ]
    for r in runs:
        lines.append(
            f"| {r['run']} | {r['best_val']:.3f} | {r['best_generation']} "
            f"| {r['seconds'] / 3600:.1f} h |"
        )
    lines += [
        "",
        f"Settings: `{json.dumps(runs[0]['settings'])}`; hand knobs `{json.dumps(HAND_KNOBS)}`.",
        "",
        "Reproduce: `python scripts/r14_brains.py train --runs 0:8` then "
        "`python scripts/r14_brains.py eval --from runs:r14b-train-*`.",
        "",
    ]
    return "\n".join(lines)
