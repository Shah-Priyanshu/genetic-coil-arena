"""Do squad specialists listen to the command centre, and how much does it say?

Runs the cooperative squad through the CommArena evaluation harness end to end:

- ``SquadSystem`` adapts a (specialist, centre) pair into an evaluation system
  ``system(inputs, seed=, hook=)``. ``inputs`` are the episode seeds of one run seed;
  ``envs.squad.run_episodes`` applies the hook to the centre's messages between send
  and receive, with the live-receiver mask (alive snakes in unfinished episodes). With
  a ``Recording`` it also returns a ``Recorder`` of what was sent and listener
  snapshots, so metrics come from the scored runs themselves.
- ``register_squad_baselines`` makes random actions, a silent centre, a random centre,
  message-blind ("no-comm") specialists and the full-view oracle mandatory baselines.
- ``run_radius`` runs one window radius: performance vs baselines, every default
  message ablation (positive listening), the message-side metrics (entropy, bits used
  vs capacity, MI with the receiver's nearest-own-food direction and distance) and the
  listener-side metrics (CIC and intervention influence on specialist actions, from
  the specialists' exact action probabilities).
- ``sanity_checks`` checks the controls that must come out a known way.

Seeds: run seed s plays episodes ``EPISODE_SEED_BASE + s * SEED_STRIDE + i``, which
lie above the step-0 benchmark's held-out range (2,000,000,000 + i).
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
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
    run_episodes,
)
from ..envs.squad.engine import DIR_DX, DIR_DY
from ..envs.squad.policies import HAND_CODED_VOCAB, CentrePolicy, SpecialistPolicy
from ..evaluation import (
    ABLATIONS,
    BaselineRegistry,
    ConditionResult,
    ExperimentResult,
    MessageHook,
    Task,
    TaskSpec,
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


# ------------------------------------------------------------------ recording
@dataclass(frozen=True)
class Recording:
    """What a recording system keeps from each run.

    Args:
        record_every: Record sent messages and targets every this many steps
        snapshot_every: Keep the specialists' observations every this many steps (for
            CIC); 0 keeps none
    """

    record_every: int = 3
    snapshot_every: int = 25


@dataclass
class Recorder:
    """Collects what the centre sends and listener snapshots during a rollout.

    Used as the ``observer`` of ``envs.squad.run_episodes``; only live receivers
    (alive snakes in unfinished episodes) are recorded.
    """

    record_every: int = 1
    snapshot_every: int = 0
    near_distance: int = NEAR_DISTANCE
    messages: list[np.ndarray] = field(default_factory=list)
    labels: list[np.ndarray] = field(default_factory=list)
    directions: list[np.ndarray] = field(default_factory=list)
    fars: list[np.ndarray] = field(default_factory=list)
    snapshots: list[tuple[Observations, np.ndarray]] = field(default_factory=list)

    def __call__(
        self, step: int, obs: Observations, sent: np.ndarray, done: np.ndarray
    ) -> None:
        live = obs.centre["alive"] & ~done[:, None]
        if step % self.record_every == 0:
            targets = food_targets(obs.centre, self.near_distance)
            self.messages.append(np.asarray(sent)[live])
            self.labels.append(targets["label"][live])
            self.directions.append(targets["direction"][live])
            self.fars.append(targets["far"][live])
        if self.snapshot_every and step % self.snapshot_every == 0:
            keep = live.any(axis=1)
            if keep.any():
                snap = Observations(
                    specialists={k: v[keep] for k, v in obs.specialists.items()},
                    centre={k: v[keep] for k, v in obs.centre.items()},
                )
                self.snapshots.append((snap, live[keep]))

    def stacked(self) -> dict[str, np.ndarray]:
        """Recorded messages (n, L) and targets (n,) for live receivers."""
        if not self.messages:
            raise ValueError("Nothing recorded")
        return {
            "messages": np.concatenate(self.messages),
            "label": np.concatenate(self.labels),
            "direction": np.concatenate(self.directions),
            "far": np.concatenate(self.fars),
        }

    def snapshot(self) -> tuple[Observations, np.ndarray]:
        """All snapshots concatenated along the episode axis, with the live mask."""
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
            np.concatenate([m for _, m in self.snapshots]),
        )


class SquadSystem:
    """A squad (specialist policy + centre policy) as an evaluation system.

    With ``recording`` set, the outputs also carry ``"recorder"``: the ``Recorder`` of
    that very run (``run_experiment(keep_outputs=True)`` keeps them per seed).
    """

    def __init__(
        self,
        config: SquadConfig,
        specialist: SpecialistPolicy,
        centre: Optional[CentrePolicy] = None,
        recording: Optional[Recording] = None,
    ):
        self.config = config
        self.specialist = specialist
        self.centre = centre or SilentCentre()
        self.recording = recording

    def __call__(
        self,
        inputs: Sequence[int],
        seed: int = 0,
        hook: Optional[MessageHook] = None,
    ) -> dict[str, Any]:
        rec = None
        if self.recording is not None:
            rec = Recorder(self.recording.record_every, self.recording.snapshot_every)
        env = SquadEnv(self.config)
        out: dict[str, Any] = dict(
            run_episodes(env, self.specialist, self.centre, inputs, hook, rec)
        )
        if rec is not None:
            out["recorder"] = rec
        return out

    def __repr__(self) -> str:
        return f"SquadSystem({self.specialist.name}, {self.centre.name})"


def squad_systems(
    config: SquadConfig, recording: Optional[Recording] = None
) -> dict[str, SquadSystem]:
    """Systems under test: message-following specialists and the deaf control."""
    return {
        LISTENER: SquadSystem(config, GreedyPolicy(), HandCodedCentre(), recording),
        DEAF: SquadSystem(config, NoCommPolicy(), HandCodedCentre(), recording),
    }


def squad_baselines(
    config: SquadConfig, recording: Optional[Recording] = None
) -> dict[str, SquadSystem]:
    """Mandatory baselines for a squad listening task.

    ``recording`` applies to the two baselines with a talking centre; the silent one
    records messages but no listener snapshots (its only message is the zero code).
    """
    silent = None if recording is None else replace(recording, snapshot_every=0)
    return {
        "random-actions": SquadSystem(config, RandomPolicy()),
        SILENT_CENTRE: SquadSystem(config, GreedyPolicy(), SilentCentre(), silent),
        RANDOM_CENTRE: SquadSystem(config, GreedyPolicy(), RandomCentre(), recording),
        "no-comm": SquadSystem(config, NoCommPolicy()),
        "oracle": SquadSystem(config, OraclePolicy()),
    }


def register_squad_baselines(
    name: str,
    config: SquadConfig,
    registry: Optional[BaselineRegistry] = None,
    recording: Optional[Recording] = None,
) -> list[str]:
    """Register the squad baselines for task ``name``; returns their names."""
    baselines = squad_baselines(config, recording)
    for baseline, system in baselines.items():
        register_baseline(name, baseline, system, registry=registry)
    return list(baselines)


def make_task(
    radius: int,
    n_seeds: int = 20,
    n_episodes: int = 50,
    registry: Optional[BaselineRegistry] = None,
    recording: Optional[Recording] = None,
) -> Task:
    """The squad task at window radius ``radius``, with its baselines registered."""
    config = SquadConfig(obs_radius=radius)
    channel = SquadEnv(config).spec.message  # n_receivers = n_snakes
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
        message=channel,
    )
    register_squad_baselines(spec.name, config, registry, recording)
    return Task(spec, lambda seed: episode_seeds(seed, n_episodes), team_food)


# ------------------------------------------------------------------ listener metrics
def make_listener(specialist: SpecialistPolicy, obs: Observations):
    """Exact action probabilities (B, N, 3) of ``specialist`` in fixed situations, as
    a function of the message it reads."""

    def listener(messages: np.ndarray) -> np.ndarray:
        view = Observations(
            specialists={**obs.specialists, "message": np.asarray(messages)},
            centre=obs.centre,
        )
        return specialist.probs(view)

    return listener


def listener_metrics(
    specialist: SpecialistPolicy, recorder: Recorder, seed: int = 0
) -> dict[str, float]:
    """CIC and intervention influence per live receiver, from one run's snapshots.

    Candidate messages for CIC are the centre's empirical message distribution over
    the same run.
    """
    obs, live = recorder.snapshot()
    listener = make_listener(specialist, obs)
    candidates, counts = np.unique(
        recorder.stacked()["messages"], axis=0, return_counts=True
    )
    received = obs.specialists["message"]

    cic = causal_influence(listener, received, candidates, counts, mask=live)
    out = {"cic_bits": cic["cic_bits"]}
    for name in ("shuffle_batch", "random_matched"):
        infl = intervention_influence(listener, received, name, seed=seed, mask=live)
        out[f"js_{name}_bits"] = infl["js_bits"]
        out[f"changed_{name}"] = infl["changed_fraction"]
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


def communication_metrics_for(
    condition: ConditionResult,
    specialist: SpecialistPolicy,
    listens: bool,
    n_shuffles: int = 100,
) -> dict[str, float]:
    """Message and listener metrics from the recorders of a condition's scored runs."""
    if condition.outputs is None:
        raise ValueError(f"{condition.name!r} was run without keep_outputs")
    pooled: list[dict[str, np.ndarray]] = []
    per_seed_mi: list[float] = []
    per_seed_listen: dict[str, list[float]] = {}
    for seed, outputs in zip(condition.seeds, condition.outputs):
        recorder: Recorder = outputs["recorder"]
        recorded = recorder.stacked()
        pooled.append(recorded)
        per_seed_mi.append(mutual_information(recorded["messages"], recorded["label"]))
        if listens:
            for key, value in listener_metrics(specialist, recorder, seed).items():
                per_seed_listen.setdefault(key, []).append(value)

    merged = {k: np.concatenate([p[k] for p in pooled]) for k in pooled[0]}
    metrics = message_metrics(merged, n_shuffles=n_shuffles)
    metrics["mi_bits_seed_mean"], metrics["mi_bits_seed_ci"] = mean_ci(
        per_seed_mi
    )
    for key, values in per_seed_listen.items():
        metrics[key], metrics[f"{key}_ci"] = mean_ci(values)
    return metrics


# ------------------------------------------------------------------ experiment
@dataclass(frozen=True)
class ListeningSettings:
    """Size of the experiment."""

    radii: tuple[int, ...] = (3, 5, 7)
    n_seeds: int = 20
    n_episodes: int = 50
    interventions: tuple[str, ...] = ABLATIONS
    record_every: int = 3
    snapshot_every: int = 25
    n_shuffles: int = 100
    n_boot: int = 2000


def run_radius(
    radius: int,
    settings: ListeningSettings = ListeningSettings(),
    registry: Optional[BaselineRegistry] = None,
) -> ExperimentResult:
    """The full listening experiment at one window radius."""
    registry = BaselineRegistry() if registry is None else registry
    recording = Recording(settings.record_every, settings.snapshot_every)
    task = make_task(radius, settings.n_seeds, settings.n_episodes, registry, recording)
    systems = squad_systems(SquadConfig(obs_radius=radius), recording)
    result = run_experiment(
        task,
        systems,
        ablations=settings.interventions,
        registry=registry,
        n_boot=settings.n_boot,
        metadata={"settings": asdict(settings), "radius": radius},
        keep_outputs=True,
    )
    result.require_complete()

    baselines = registry.baselines(task.name)
    for name, system, listens in (
        (LISTENER, systems[LISTENER], True),
        (DEAF, systems[DEAF], True),
        (RANDOM_CENTRE, baselines[RANDOM_CENTRE], True),
        (SILENT_CENTRE, baselines[SILENT_CENTRE], False),
    ):
        result.add_communication(
            name,
            communication_metrics_for(
                result.condition(name), system.specialist, listens, settings.n_shuffles
            ),
        )

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
    return rows
