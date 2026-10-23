"""Roll out scripted squad policies and time the engine."""

import time
from collections.abc import Sequence
from typing import Any, Callable, Optional

import numpy as np

from ..base import Observations
from .env import SquadConfig, SquadEnv
from .policies import CentrePolicy, SilentCentre, SpecialistPolicy

# hook(messages, step, mask) -> messages; mask is the (B, N) live-receiver mask
Hook = Callable[..., np.ndarray]
# observer(step, obs, sent, done) is called every step before the specialists act
Observer = Callable[[int, Observations, Optional[np.ndarray], np.ndarray], Any]


def run_episodes(
    env: SquadEnv,
    specialist: SpecialistPolicy,
    centre: Optional[CentrePolicy],
    seeds: Sequence[int],
    hook: Optional[Hook] = None,
    observer: Optional[Observer] = None,
) -> dict[str, np.ndarray]:
    """Play one batched episode per seed to the end.

    Args:
        hook: Applied to the centre's messages between send and receive as
            ``hook(messages, step, mask)``, where ``mask`` = alive snakes in unfinished
            episodes (an ``evaluation`` ``MessageHook`` fits)
        observer: Called as ``observer(step, obs, sent, done)`` with the messages as
            sent (before the hook), e.g. to record them

    Returns:
        Per-episode arrays: ``team_food`` (B,), ``food`` (B, N), ``steps`` (B,),
        ``survivors`` (B,), ``return`` (B,) team reward, ``death_reason`` (B, N),
        and the scalar ``agent_steps`` (living-snake steps simulated)
    """
    centre = centre or SilentCentre()
    obs = env.reset(seeds)
    specialist.reset(seeds)
    centre.reset(seeds)
    total = np.zeros(len(seeds), dtype=np.float64)
    agent_steps = 0
    done = np.zeros(len(seeds), dtype=bool)
    step = 0
    while not done.all():
        agent_steps += int(env.alive.sum())
        sent = centre.send(obs)
        if observer is not None:
            observer(step, obs, sent, done)
        delivered = sent
        if hook is not None and sent is not None:
            delivered = hook(sent, step, obs.centre["alive"] & ~done[:, None])
        actions = specialist.act(obs)
        obs, rewards, done, info = env.step(actions, delivered)
        total += rewards[:, 0]
        step += 1
    return {
        "team_food": env.score.sum(axis=1).astype(np.float64),
        "food": env.score.copy(),
        "steps": env.steps.copy(),
        "survivors": env.alive.sum(axis=1),
        "return": total,
        "death_reason": env.death.copy(),
        "agent_steps": np.int64(agent_steps),
    }


def measure_throughput(
    config: Optional[SquadConfig] = None,
    specialist: Optional[SpecialistPolicy] = None,
    centre: Optional[CentrePolicy] = None,
    batch_size: int = 256,
    repeats: int = 3,
    seed: int = 0,
) -> dict[str, float]:
    """Single-process agent-steps per second (best of ``repeats``, after a warm-up).

    With ``specialist=None`` actions are uniformly random and no messages are sent,
    which times the engine and observation building alone.
    """
    env = SquadEnv(config)
    seeds = list(range(seed, seed + batch_size))

    def once() -> dict[str, float]:
        start = time.perf_counter()
        if specialist is None:
            gen = np.random.default_rng(seed)
            env.reset(seeds)
            agent_steps = 0
            while not env.done.all():
                agent_steps += int(env.alive.sum())
                env.step(gen.integers(0, 3, size=(batch_size, env.config.n_snakes)))
        else:
            agent_steps = int(
                run_episodes(env, specialist, centre, seeds)["agent_steps"]
            )
        elapsed = time.perf_counter() - start
        return {"agent_steps": agent_steps, "seconds": elapsed}

    once()  # compile and warm caches
    runs = [once() for _ in range(repeats)]
    best = min(runs, key=lambda r: r["seconds"])
    return {
        "agent_steps": float(best["agent_steps"]),
        "seconds": best["seconds"],
        "agent_steps_per_sec": best["agent_steps"] / best["seconds"],
    }
