"""Do squad specialists listen to the command centre, and how much does it say?

Runs the cooperative squad through the CommArena evaluation harness end to end:

- ``SquadSystem`` adapts a (specialist, centre) pair into an evaluation system
  ``system(inputs, seed=, hook=)``. ``inputs`` are the episode seeds of one run seed;
  the hook is applied to the centre's messages between send and receive.
- ``register_squad_baselines`` makes random actions, a silent centre, a random centre,
  message-blind ("no-comm") specialists and the full-view oracle mandatory baselines.
- ``run_radius`` runs one window radius: performance vs baselines, every message
  intervention (positive listening), and, from instrumented reruns of the same
  episodes, the message-side metrics (entropy, bits used vs capacity, MI with the
  receiver's nearest-own-food direction and distance) and the listener-side metrics
  (CIC and intervention influence on specialist actions).
- ``sanity_checks`` checks the controls that must come out a known way.

Seeds: run seed s plays episodes ``EPISODE_SEED_BASE + s * SEED_STRIDE + i``, which
lie above the step-0 benchmark's held-out range (2,000,000,000 + i).
"""

import copy
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from ..envs.base import Observations
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
)
from ..envs.squad.engine import DIR_DX, DIR_DY
from ..envs.squad.policies import HAND_CODED_VOCAB, CentrePolicy, SpecialistPolicy
from ..evaluation import (
    BaselineRegistry,
    ExperimentResult,
    MessageHook,
    Task,
    TaskSpec,
    apply_hook,
    causal_influence,
    communication_metrics,
    entropy,
    intervention_influence,
    listening_summary,
    mean_ci,
    mutual_information,
    register_baseline,
    run_experiment,
)

EPISODE_SEED_BASE = 3_000_000_000
SEED_STRIDE = 10_000
INTERVENTIONS = ("zero", "shuffle_batch", "swap_receivers", "random_matched")
NO_FOOD = 2 * 4  # Target label when the receiver has no own-colour food on the board
NEAR_DISTANCE = 5  # Same threshold as HandCodedCentre's default

LISTENER = "greedy+hand-coded"
DEAF = "no-comm+hand-coded"
RANDOM_CENTRE = "greedy+random-centre"
SILENT_CENTRE = "greedy+silent-centre"


def task_name(radius: int) -> str:
    """Evaluation task name for window radius ``radius``."""
    return f"squad-listening-r{radius}"


def episode_seeds(seed: int, n_episodes: int) -> list[int]:
    """The episode seeds one run seed plays."""
    if not 1 <= n_episodes <= SEED_STRIDE:
        raise ValueError(f"n_episodes must lie in [1, {SEED_STRIDE}]")
    start = EPISODE_SEED_BASE + int(seed) * SEED_STRIDE
    return list(range(start, start + n_episodes))


def team_food(inputs: Any, outputs: dict[str, np.ndarray]) -> np.ndarray:
    """Checker: team food per episode."""
    return np.asarray(outputs["team_food"], dtype=float)


# ------------------------------------------------------------------ targets
def food_targets(
    centre_obs: dict[str, np.ndarray], near_distance: int = NEAR_DISTANCE
) -> dict[str, np.ndarray]:
    """What the centre's code is supposed to encode, recomputed from its observation.

    For each receiver: the egocentric direction (0 ahead, 1 right, 2 behind, 3 left, by
    the dominant axis) of its nearest own-colour food, whether that food is farther than
    ``near_distance`` (Manhattan), and ``label = 2 * direction + far``, or ``NO_FOOD``
    when the receiver is dead or has no own-colour food. Shapes (B, N).
    """
    heads = np.asarray(centre_obs["heads"])
    heading = np.asarray(centre_obs["heading"])
    food = np.asarray(centre_obs["food"])
    batch, n = heading.shape
    per_colour = food.shape[1] // n
    fx = food[:, :, 0].reshape(batch, n, per_colour)
    fy = food[:, :, 1].reshape(batch, n, per_colour)
    hx, hy = heads[:, :, 0:1], heads[:, :, 1:2]

    big = np.iinfo(np.int64).max
    dist = np.where(fx >= 0, np.abs(fx - hx) + np.abs(fy - hy), big)
    nearest = dist.argmin(axis=2)[:, :, None]  # First minimum, like the centre
    best = np.take_along_axis(dist, nearest, 2)[:, :, 0]
    ox = np.take_along_axis(fx, nearest, 2)[:, :, 0] - heads[:, :, 0]
    oy = np.take_along_axis(fy, nearest, 2)[:, :, 0] - heads[:, :, 1]
    valid = (heads[:, :, 0] >= 0) & (best < big)

    ahead = ox * DIR_DX[heading] + oy * DIR_DY[heading]
    right = ox * DIR_DX[(heading + 1) % 4] + oy * DIR_DY[(heading + 1) % 4]
    direction = np.where(
        np.abs(ahead) >= np.abs(right),
        np.where(ahead > 0, 0, 2),
        np.where(right > 0, 1, 3),
    )
    far = (best > near_distance).astype(np.int64)
    label = np.where(valid, 2 * direction + far, NO_FOOD)
    return {
        "label": label.astype(np.int64),
        "direction": np.where(valid, direction, 4).astype(np.int64),
        "far": np.where(valid, far, 2).astype(np.int64),
        "valid": valid,
    }


# ------------------------------------------------------------------ rollout
@dataclass
class Recorder:
    """Collects what the centre sends and listener snapshots during a rollout.

    Args:
        record_every: Record sent messages and targets every this many steps
        snapshot_every: Keep the specialists' observations every this many steps (for
            CIC); 0 keeps none
    """

    record_every: int = 1
    snapshot_every: int = 0
    near_distance: int = NEAR_DISTANCE
    messages: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    directions: list[np.ndarray] = field(default_factory=list)
    fars: list[np.ndarray] = field(default_factory=list)
    snapshots: list[tuple[Observations, np.ndarray]] = field(default_factory=list)

    def observe(
        self,
        step: int,
        obs: Observations,
        sent: Optional[np.ndarray],
        done: np.ndarray,
    ) -> None:
        alive = obs.centre["heads"][:, :, 0] >= 0
        if step % self.record_every == 0:
            active = alive & ~done[:, None]
            if sent is None:  # A silent centre: receivers keep reading the zero payload
                sent = np.zeros_like(obs.specialists["message"])
            targets = food_targets(obs.centre, self.near_distance)
            self.messages.append(np.asarray(sent)[active])
            self.labels.append(targets["label"][active])
            self.directions.append(targets["direction"][active])
            self.fars.append(targets["far"][active])
        if self.snapshot_every and step % self.snapshot_every == 0:
            keep = ~done & alive.any(axis=1)
            if keep.any():
                snap = Observations(
                    specialists={k: v[keep] for k, v in obs.specialists.items()},
                    centre={k: v[keep] for k, v in obs.centre.items()},
                )
                self.snapshots.append((snap, alive[keep]))

    def stacked(self) -> dict[str, np.ndarray]:
        """Recorded messages (n, L) and targets (n,) for active receivers."""
        if not self.messages:
            raise ValueError("Nothing recorded")
        return {
            "messages": np.concatenate(self.messages),
            "label": np.concatenate(self.labels),
            "direction": np.concatenate(self.directions),
            "far": np.concatenate(self.fars),
        }

    def snapshot(self) -> tuple[Observations, np.ndarray]:
        """All snapshots concatenated along the episode axis, with the alive mask."""
        if not self.snapshots:
            raise ValueError("No snapshots recorded")
        obs = [s for s, _ in self.snapshots]
        return (
            Observations(
                specialists={
                    k: np.concatenate([o.specialists[k] for o in obs])
                    for k in obs[0].specialists
                },
                centre={
                    k: np.concatenate([o.centre[k] for o in obs]) for k in obs[0].centre
                },
            ),
            np.concatenate([a for _, a in self.snapshots]),
        )


def rollout(
    env: SquadEnv,
    specialist: SpecialistPolicy,
    centre: Optional[CentrePolicy],
    seeds: Sequence[int],
    hook: Optional[MessageHook] = None,
    recorder: Optional[Recorder] = None,
) -> dict[str, np.ndarray]:
    """``envs.squad.run_episodes`` with the hook applied between send and receive.

    A silent centre sends ``None``; there is nothing to intervene on, so the hook is
    not called and the receivers keep their zero payload.
    """
    centre = centre or SilentCentre()
    obs = env.reset(seeds)
    specialist.reset(seeds)
    centre.reset(seeds)
    done = np.zeros(len(seeds), dtype=bool)
    info: dict[str, np.ndarray] = {}
    step = 0
    while not done.all():
        sent = centre.send(obs)
        delivered = None if sent is None else apply_hook(hook, sent, step)
        if recorder is not None:
            recorder.observe(step, obs, sent, done)
        actions = specialist.act(obs)
        obs, _, done, info = env.step(actions, delivered)
        step += 1
    return {
        "team_food": info["score"].sum(axis=1).astype(np.float64),
        "food": info["score"],
        "steps": info["steps"],
        "survivors": info["alive"].sum(axis=1),
    }


class SquadSystem:
    """A squad (specialist policy + centre policy) as an evaluation system."""

    def __init__(
        self,
        config: SquadConfig,
        specialist: SpecialistPolicy,
        centre: Optional[CentrePolicy] = None,
    ):
        self.config = config
        self.specialist = specialist
        self.centre = centre or SilentCentre()

    def __call__(
        self,
        inputs: Sequence[int],
        seed: int = 0,
        hook: Optional[MessageHook] = None,
        recorder: Optional[Recorder] = None,
    ) -> dict[str, np.ndarray]:
        env = SquadEnv(self.config)
        return rollout(env, self.specialist, self.centre, inputs, hook, recorder)

    def __repr__(self) -> str:
        return f"SquadSystem({self.specialist.name}, {self.centre.name})"


def squad_systems(config: SquadConfig) -> dict[str, SquadSystem]:
    """Systems under test: message-following specialists and the deaf control."""
    return {
        LISTENER: SquadSystem(config, GreedyPolicy(), HandCodedCentre()),
        DEAF: SquadSystem(config, NoCommPolicy(), HandCodedCentre()),
    }


def squad_baselines(config: SquadConfig) -> dict[str, SquadSystem]:
    """Mandatory baselines for a squad listening task."""
    return {
        "random-actions": SquadSystem(config, RandomPolicy()),
        SILENT_CENTRE: SquadSystem(config, GreedyPolicy(), SilentCentre()),
        RANDOM_CENTRE: SquadSystem(config, GreedyPolicy(), RandomCentre()),
        "no-comm": SquadSystem(config, NoCommPolicy()),
        "oracle": SquadSystem(config, OraclePolicy()),
    }


def register_squad_baselines(
    name: str, config: SquadConfig, registry: Optional[BaselineRegistry] = None
) -> list[str]:
    """Register the squad baselines for task ``name``; returns their names."""
    baselines = squad_baselines(config)
    for baseline, system in baselines.items():
        register_baseline(name, baseline, system, registry=registry)
    return list(baselines)


def make_task(
    radius: int,
    n_seeds: int = 20,
    n_episodes: int = 50,
    registry: Optional[BaselineRegistry] = None,
) -> Task:
    """The squad task at window radius ``radius``, with its baselines registered."""
    config = SquadConfig(obs_radius=radius)
    spec = TaskSpec(
        name=task_name(radius),
        description=(
            "Cooperative squad (21x21 walls, 3 snakes, T=300); score = team food per "
            f"episode; specialist window radius {radius}"
        ),
        metric="team_food",
        n_seeds=n_seeds,
        params={
            "obs_radius": radius,
            "n_episodes": n_episodes,
            "episode_seed_base": EPISODE_SEED_BASE,
            "seed_stride": SEED_STRIDE,
            "message_capacity_bits": config.message.bits,
        },
    )
    register_squad_baselines(spec.name, config, registry)
    return Task(spec, lambda seed: episode_seeds(seed, n_episodes), team_food)


# ------------------------------------------------------------------ listener metrics
def make_listener(
    specialist: SpecialistPolicy,
    obs: Observations,
    alive: np.ndarray,
    n_draws: int = 8,
    seed: int = 0,
):
    """Action probabilities of ``specialist`` in fixed situations, as a function of the
    message it reads.

    The scripted policies only expose a sampling ``act`` with internal per-episode
    state, so probabilities are estimated from ``n_draws`` independently seeded copies.
    Dead receivers get a constant distribution so they add nothing to CIC or JS.
    """
    policy = copy.deepcopy(specialist)
    batch, n = alive.shape
    n_actions = 3

    def listener(messages: np.ndarray) -> np.ndarray:
        view = Observations(
            specialists={**obs.specialists, "message": np.asarray(messages)},
            centre=obs.centre,
        )
        probs = np.zeros((batch, n, n_actions))
        rows, cols = np.indices((batch, n))
        for k in range(n_draws):
            base = (seed * n_draws + k) * batch
            policy.reset(range(base, base + batch))
            probs[rows, cols, policy.act(view)] += 1.0
        probs /= n_draws
        probs[~alive] = np.eye(n_actions)[0]
        return probs

    return listener


def listener_metrics(
    specialist: SpecialistPolicy,
    recorder: Recorder,
    n_draws: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    """CIC and intervention influence per living receiver, from one seed's snapshots.

    Candidate messages for CIC are the centre's empirical message distribution over
    the same run. Values are rescaled from the harness's average over every (episode,
    receiver) slot to an average over living receivers.
    """
    obs, alive = recorder.snapshot()
    listener = make_listener(specialist, obs, alive, n_draws, seed)
    sent = recorder.stacked()["messages"]
    candidates, counts = np.unique(sent, axis=0, return_counts=True)
    scale = alive.size / max(int(alive.sum()), 1)
    received = obs.specialists["message"]

    cic = causal_influence(listener, received, candidates, counts)
    out = {"cic_bits": cic["cic_bits"] * scale}
    for name in ("shuffle_batch", "random_matched"):
        if name == "shuffle_batch" and received.shape[0] < 2:
            continue
        infl = intervention_influence(listener, received, name, seed=seed)
        out[f"js_{name}_bits"] = infl["js_bits"] * scale
        out[f"changed_{name}"] = infl["changed_fraction"] * scale
    return out


def message_metrics(
    recorded: dict[str, np.ndarray],
    vocab_size: int = HAND_CODED_VOCAB,
    n_shuffles: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    """Entropy, bits used vs capacity, and MI with the nearest-own-food target."""
    msgs = recorded["messages"]
    out = communication_metrics(
        msgs,
        targets=recorded["label"],
        vocab_size=vocab_size,
        n_shuffles=n_shuffles,
        seed=seed,
    )
    out["target_entropy_bits"] = entropy(recorded["label"])
    out["mi_direction_bits"] = mutual_information(msgs, recorded["direction"])
    out["mi_far_bits"] = mutual_information(msgs, recorded["far"])
    out["n_messages"] = float(msgs.shape[0])
    return out


def communication_pass(
    task: Task,
    system: SquadSystem,
    seeds: Sequence[int],
    listens: bool,
    record_every: int = 3,
    snapshot_every: int = 25,
    n_draws: int = 8,
    n_shuffles: int = 100,
) -> tuple[dict[str, float], list[float]]:
    """Instrumented intact reruns of every seed; returns (metrics, per-seed scores).

    The harness's ``evaluate`` exposes only per-seed scores, so messages and listener
    snapshots come from a second, identical run of the same episodes.
    """
    pooled: list[dict[str, np.ndarray]] = []
    per_seed_mi: list[float] = []
    per_seed_listen: dict[str, list[float]] = {}
    scores: list[float] = []
    for seed in seeds:
        recorder = Recorder(record_every, snapshot_every if listens else 0)
        outputs = system(task.inputs(seed), seed=seed, recorder=recorder)
        scores.append(float(np.mean(task.checker(None, outputs))))
        recorded = recorder.stacked()
        pooled.append(recorded)
        per_seed_mi.append(mutual_information(recorded["messages"], recorded["label"]))
        if listens:
            for key, value in listener_metrics(
                system.specialist, recorder, n_draws, seed
            ).items():
                per_seed_listen.setdefault(key, []).append(value)

    merged = {k: np.concatenate([p[k] for p in pooled]) for k in pooled[0]}
    metrics = message_metrics(merged, n_shuffles=n_shuffles)
    metrics["mi_bits_seed_mean"], metrics["mi_bits_seed_ci"] = mean_ci(
        per_seed_mi
    )
    for key, values in per_seed_listen.items():
        metrics[key], metrics[f"{key}_ci"] = mean_ci(values)
    return metrics, scores


# ------------------------------------------------------------------ experiment
@dataclass(frozen=True)
class ListeningSettings:
    """Size of the experiment."""

    radii: tuple[int, ...] = (3, 5, 7)
    n_seeds: int = 20
    n_episodes: int = 50
    interventions: tuple[str, ...] = INTERVENTIONS
    record_every: int = 3
    snapshot_every: int = 25
    n_draws: int = 8
    n_shuffles: int = 100
    n_boot: int = 2000


def run_radius(
    radius: int,
    settings: ListeningSettings = ListeningSettings(),
    registry: Optional[BaselineRegistry] = None,
) -> ExperimentResult:
    """The full listening experiment at one window radius."""
    registry = BaselineRegistry() if registry is None else registry
    task = make_task(radius, settings.n_seeds, settings.n_episodes, registry)
    config = SquadConfig(obs_radius=radius)
    systems = squad_systems(config)
    seeds = task.spec.seeds()
    result = run_experiment(
        task,
        systems,
        seeds=seeds,
        ablations=settings.interventions,
        registry=registry,
        n_boot=settings.n_boot,
        metadata={"settings": asdict(settings), "radius": radius},
    )
    result.require_complete()

    baselines = registry.baselines(task.name)
    reproduced: dict[str, float] = {}
    for name, system, listens in (
        (LISTENER, systems[LISTENER], True),
        (DEAF, systems[DEAF], True),
        (RANDOM_CENTRE, baselines[RANDOM_CENTRE], True),
        (SILENT_CENTRE, baselines[SILENT_CENTRE], False),
    ):
        metrics, scores = communication_pass(
            task,
            system,
            seeds,
            listens,
            settings.record_every,
            settings.snapshot_every,
            settings.n_draws,
            settings.n_shuffles,
        )
        result.add_communication(name, metrics)
        harness = np.asarray(result.condition(name).per_seed)
        reproduced[name] = float(np.max(np.abs(harness - np.asarray(scores))))
    result.metadata["instrumented_rerun_max_abs_diff"] = reproduced

    result.metadata["listening_summary"] = {
        name: listening_summary(
            {"cic_bits": result.communication[name]["cic_bits"]},
            {
                a.intervention: {"listening": a.listening}
                for a in result.ablations
                if a.system == name
            },
        )
        for name in (LISTENER, DEAF)
    }
    result.metadata["sanity"] = sanity_checks(result)
    return result


def sanity_checks(result: ExperimentResult) -> list[dict[str, Any]]:
    """Controls with a known answer; each row has check, passed and detail."""
    rows: list[dict[str, Any]] = []
    deaf = [a for a in result.ablations if a.system == DEAF]
    worst = max((abs(a.delta_mean) for a in deaf), default=float("nan"))
    rows.append(
        {
            "check": "deaf specialists: no ablation changes team food",
            "passed": bool(deaf)
            and all(not a.listening for a in deaf)
            and worst < 0.5,
            "detail": f"max |drop| = {worst:.3f} food",
        }
    )
    cic_deaf = result.communication.get(DEAF, {}).get("cic_bits", float("nan"))
    rows.append(
        {
            "check": "deaf specialists: CIC ~ 0",
            "passed": bool(abs(cic_deaf) < 1e-9),
            "detail": f"CIC = {cic_deaf:.4f} bits",
        }
    )
    rand = result.communication.get(RANDOM_CENTRE, {})
    mi_rand = rand.get("mi_bits", float("nan"))
    rows.append(
        {
            "check": "random centre: MI(message; target) ~ 0",
            "passed": bool(mi_rand < max(0.01, rand.get("mi_null_q95_bits", 0.0))),
            "detail": f"MI = {mi_rand:.4f} bits (null q95 "
            f"{rand.get('mi_null_q95_bits', float('nan')):.4f})",
        }
    )
    mi_silent = result.communication.get(SILENT_CENTRE, {}).get(
        "mi_bits", float("nan")
    )
    rows.append(
        {
            "check": "silent centre: MI(message; target) = 0",
            "passed": bool(abs(mi_silent) < 1e-9),
            "detail": f"MI = {mi_silent:.4f} bits",
        }
    )
    zero = [
        a for a in result.ablations if a.system == LISTENER and a.intervention == "zero"
    ]
    if zero:
        silent = np.asarray(result.condition(SILENT_CENTRE).per_seed)
        diff = float(np.max(np.abs(np.asarray(zero[0].ablated.per_seed) - silent)))
        rows.append(
            {
                "check": "zero ablation == silent-centre baseline, seed by seed",
                "passed": diff < 1e-9,
                "detail": f"max |diff| = {diff:.3g}",
            }
        )
    rerun = result.metadata.get("instrumented_rerun_max_abs_diff", {})
    if rerun:
        diff = max(rerun.values())
        rows.append(
            {
                "check": "instrumented reruns reproduce the harness scores",
                "passed": diff < 1e-9,
                "detail": f"max |diff| = {diff:.3g}",
            }
        )
    return rows
