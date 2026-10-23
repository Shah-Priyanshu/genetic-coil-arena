"""Does a learned protocol match a hand-coded one at the same bit budget? (R6)

A command centre and three colour specialists are trained jointly from team reward
with evolution strategies (``learning.EvolutionStrategy``), over the same channel the
hand-coded centre uses: one symbol from V = 8 (3 bits) per receiver, sent every step
(k = 1) and read one step later.

Conditions, trained with the same algorithm, network sizes and budget:

- ``silent+learned-spec`` (a): learned specialists, the centre always sends code 0
- ``hand-coded+learned-spec`` (b): learned specialists listening to the fixed
  ``HandCodedCentre``
- ``learned-centre+learned-spec`` (c): learned centre and learned specialists

Networks (shared across the three specialists, and across receivers for the centre):

- specialist: egocentric window planes (4 (2r+1)²), health, one-hot received message
  (V) -> tanh hidden -> 3 action logits; actions are sampled from the softmax
  (``probs`` returns it exactly, so CIC is computed on the real policy)
- centre, per receiver: the receiver's two own-colour food items as egocentric
  (ahead, right) offsets / board size plus a valid flag, sorted near to far, and its
  health -> tanh hidden -> V logits; the message is the argmax. Sorting is a
  permutation canonicalisation, not the code: the centre still has to learn which
  partition of the plane to name and the specialists what each name means.

Training: OpenAI-ES (antithetic, centred ranks, Adam). Every member of a generation
plays the same episode seeds and the same action noise (common random numbers);
fitness is the environment's team return (food + 0.01 per living snake per step).
The mean parameters are scored on fixed validation seeds every ``eval_every``
generations (the learning curve); the best validation checkpoint is kept.

Seeds: training episodes ``TRAIN_SEED_BASE + run * 10^7 + ...``, validation
``VAL_SEED_BASE + run * SEED_STRIDE + i``, held-out test ``TEST_SEED_BASE + s *
SEED_STRIDE + i``. Each training run is one evaluation seed of the harness, so the
CIs and tests are over independent training runs.
"""

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
from scipy import stats as scipy_stats

from ..envs.base import Observations
from ..envs.squad import (
    GreedyPolicy,
    HandCodedCentre,
    SquadConfig,
    SquadEnv,
    run_episodes,
)
from ..envs.squad.engine import DIR_DX, DIR_DY
from ..envs.squad.policies import CentrePolicy, SilentCentre, SpecialistPolicy
from ..evaluation import (
    ABLATIONS,
    BaselineRegistry,
    ConditionResult,
    ExperimentResult,
    MessageHook,
    Task,
    TaskSpec,
    communication_metrics,
    compare_conditions,
    entropy,
    mean_ci,
    mutual_information,
    run_experiment,
)
from ..learning import MLP, EvolutionStrategy, softmax
from .listening import (
    NO_FOOD,
    Recorder,
    Recording,
    listener_metrics,
    register_squad_baselines,
    team_food,
)

TRAIN_SEED_BASE = 1_000_000_000
TRAIN_RUN_STRIDE = 10_000_000
VAL_SEED_BASE = 4_000_000_000
TEST_SEED_BASE = 3_500_000_000
SEED_STRIDE = 10_000

LEARNED = "learned"
HAND_CODED = "hand-coded"
SILENT = "silent"
MODES = (SILENT, HAND_CODED, LEARNED)
CONDITION = {
    SILENT: "silent+learned-spec",
    HAND_CODED: "hand-coded+learned-spec",
    LEARNED: "learned-centre+learned-spec",
}
SCRIPTED = "greedy+hand-coded"
FOOD_FEATURES = 3  # ahead, right, valid per own-colour food item


# ------------------------------------------------------------------ settings
@dataclass(frozen=True)
class TrainSettings:
    """One training run's budget and hyperparameters (the same for every condition).

    ``time_limit_s`` stops a run early (after the generation in progress); the curve
    records how far it got.
    """

    radius: int = 3
    generations: int = 300
    population: int = 64
    episodes_per_member: int = 8
    sigma: float = 0.05
    lr: float = 0.03
    weight_decay: float = 0.005
    spec_hidden: int = 16
    centre_hidden: int = 16
    eval_every: int = 10
    val_episodes: int = 32
    max_steps: int = 300
    time_limit_s: float = float("inf")
    progress_s: float = 0.0  # Print a progress line at most this often (0: never)

    def squad_config(self) -> SquadConfig:
        return SquadConfig(obs_radius=self.radius, max_steps=self.max_steps)


# ------------------------------------------------------------------ features
def specialist_inputs(specialists: dict[str, np.ndarray], vocab: int) -> np.ndarray:
    """(B, N, 4 (2r+1)² + 1 + V) float32: window planes, health, one-hot message."""
    window = specialists["window"]
    batch, n = window.shape[:2]
    msg = np.asarray(specialists["message"])
    onehot = np.zeros((batch, n, vocab), dtype=np.float32)
    if msg.ndim == 3 and msg.shape[2] > 0:
        codes = np.clip(msg[:, :, 0], 0, vocab - 1)
        np.put_along_axis(onehot, codes[:, :, None], 1.0, axis=2)
    else:
        onehot[:, :, 0] = 1.0
    return np.concatenate(
        [
            window.reshape(batch, n, -1).astype(np.float32),
            np.asarray(specialists["health"], dtype=np.float32).reshape(batch, n, 1),
            onehot,
        ],
        axis=2,
    )


def specialist_input_size(radius: int, vocab: int) -> int:
    return 4 * (2 * radius + 1) ** 2 + 1 + vocab


def centre_features(centre: dict[str, np.ndarray]) -> np.ndarray:
    """(B, N, 3F + 1) float32 per-receiver features from the centre's observation.

    For each of the receiver's F own-colour food items, sorted by Manhattan distance:
    egocentric (ahead, right) offsets divided by the board side, and a valid flag;
    then the receiver's health. A dead receiver gets all zeros.
    """
    heads = np.asarray(centre["heads"])
    heading = np.asarray(centre["heading"])
    food = np.asarray(centre["food"])
    health = np.asarray(centre["health"], dtype=np.float32)
    board = np.asarray(centre["board"]).shape[-1]
    batch, n = heading.shape
    per = food.shape[1] // n
    fx = food[:, :, 0].reshape(batch, n, per)
    fy = food[:, :, 1].reshape(batch, n, per)
    alive = heads[:, :, 0] >= 0
    valid = (fx >= 0) & alive[:, :, None]
    ox = fx - heads[:, :, 0:1]
    oy = fy - heads[:, :, 1:2]
    h = heading[:, :, None]
    ahead = ox * DIR_DX[h] + oy * DIR_DY[h]
    right = ox * DIR_DX[(h + 1) % 4] + oy * DIR_DY[(h + 1) % 4]
    dist = np.where(valid, np.abs(ox) + np.abs(oy), np.iinfo(np.int64).max)
    order = np.argsort(dist, axis=2, kind="stable")
    ahead = np.take_along_axis(ahead, order, 2)
    right = np.take_along_axis(right, order, 2)
    valid = np.take_along_axis(valid, order, 2)
    feats = np.zeros((batch, n, FOOD_FEATURES * per + 1), dtype=np.float32)
    feats[:, :, 0 : FOOD_FEATURES * per : 3] = np.where(valid, ahead / board, 0.0)
    feats[:, :, 1 : FOOD_FEATURES * per : 3] = np.where(valid, right / board, 0.0)
    feats[:, :, 2 : FOOD_FEATURES * per : 3] = valid
    feats[:, :, -1] = np.where(alive, health, 0.0)
    return feats


# ------------------------------------------------------------------ networks
@dataclass(frozen=True)
class Nets:
    """The specialist and centre networks of one condition."""

    spec: MLP
    centre: Optional[MLP]
    vocab: int

    @classmethod
    def build(cls, settings: TrainSettings, mode: str) -> "Nets":
        config = settings.squad_config()
        vocab = config.message.vocab_size
        spec = MLP(
            (specialist_input_size(settings.radius, vocab), settings.spec_hidden, 3)
        )
        centre = None
        if mode == LEARNED:
            n_in = FOOD_FEATURES * config.food_per_colour + 1
            centre = MLP((n_in, settings.centre_hidden, vocab))
        return cls(spec, centre, vocab)

    @property
    def n_params(self) -> int:
        return self.spec.n_params + (self.centre.n_params if self.centre else 0)

    def split(self, theta: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """(spec params, centre params) from joint parameters (last axis)."""
        k = self.spec.n_params
        return theta[..., :k], (theta[..., k:] if self.centre else None)

    def init(self, run: int) -> np.ndarray:
        """Initial parameters; the specialist part depends on the run only, so the
        three conditions of one run start from the same specialist."""
        spec = self.spec.init(np.random.default_rng([run, 1]))
        if self.centre is None:
            return spec
        return np.concatenate([spec, self.centre.init(np.random.default_rng([run, 2]))])


# ------------------------------------------------------------------ rollouts
def _messages(
    mode: str,
    nets: Nets,
    centre_params: Optional[np.ndarray],
    obs: Observations,
    members: int,
) -> np.ndarray:
    """(P B, N, 1) codes for the population, member-major."""
    heading = obs.centre["heading"]
    if mode == SILENT:
        return np.zeros(heading.shape + (1,), dtype=np.int64)
    if mode == HAND_CODED:
        return HandCodedCentre().send(obs)
    assert nets.centre is not None and centre_params is not None
    feats = centre_features(obs.centre)
    total, n = feats.shape[:2]
    logits = nets.centre.forward(
        centre_params, feats.reshape(members, (total // members) * n, -1)
    )
    return logits.argmax(axis=2).reshape(total, n, 1).astype(np.int64)


def population_rollout(
    settings: TrainSettings,
    nets: Nets,
    mode: str,
    params: np.ndarray,
    seeds: Sequence[int],
    noise_seed: int,
) -> dict[str, np.ndarray]:
    """Every member of ``params`` (P, n) plays the same ``seeds`` with the same action
    noise; returns ``return`` and ``team_food`` as (P, B)."""
    params = np.atleast_2d(params)
    members, batch = params.shape[0], len(seeds)
    spec_p, centre_p = nets.split(params)
    env = SquadEnv(settings.squad_config())
    obs = env.reset(np.tile(np.asarray(seeds, dtype=np.int64), members))
    n = env.config.n_snakes
    gen = np.random.default_rng(noise_seed)
    total = np.zeros(members * batch, dtype=np.float64)
    done = np.zeros(members * batch, dtype=bool)
    while not done.all():
        msgs = _messages(mode, nets, centre_p, obs, members)
        x = specialist_inputs(obs.specialists, nets.vocab)
        logits = nets.spec.forward(spec_p, x.reshape(members, batch * n, -1))
        gumbel = gen.gumbel(size=(batch * n, 3)).astype(np.float32)
        actions = (logits + gumbel).argmax(axis=2).reshape(members * batch, n)
        obs, rewards, done, _ = env.step(actions, msgs)
        total += rewards[:, 0]
    return {
        "return": total.reshape(members, batch),
        "team_food": env.score.sum(axis=1).reshape(members, batch).astype(float),
    }


# ------------------------------------------------------------------ training
def train_run(
    mode: str, run: int, settings: TrainSettings = TrainSettings()
) -> dict[str, Any]:
    """Train one condition from one seed; returns parameters and the learning curve.

    Returns:
        ``theta`` (best validation checkpoint), ``theta_final``, ``curve`` (rows of
        generation, episodes, wall_s, pop_mean_return, pop_max_return,
        val_return, val_team_food), ``best_generation``, ``generations_done``,
        ``seconds``, ``n_params``
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
    nets = Nets.build(settings, mode)
    es = EvolutionStrategy(
        nets.init(run),
        population=settings.population,
        sigma=settings.sigma,
        lr=settings.lr,
        weight_decay=settings.weight_decay,
        seed=run * 1000 + MODES.index(mode),
    )
    val_seeds = [
        VAL_SEED_BASE + run * SEED_STRIDE + i for i in range(settings.val_episodes)
    ]
    b = settings.episodes_per_member
    start = time.perf_counter()
    curve: list[dict[str, float]] = []
    best = (-np.inf, es.theta.copy(), 0)
    episodes = 0
    gen_done = 0
    last_print = [0.0]

    def validate(generation: int, pop: Optional[np.ndarray]) -> None:
        nonlocal best
        out = population_rollout(
            settings, nets, mode, es.theta, val_seeds, VAL_SEED_BASE + run
        )
        val_food = float(out["team_food"].mean())
        val_ret = float(out["return"].mean())
        curve.append(
            {
                "generation": generation,
                "episodes": episodes,
                "wall_s": time.perf_counter() - start,
                "pop_mean_return": float(pop.mean()) if pop is not None else np.nan,
                "pop_max_return": float(pop.max()) if pop is not None else np.nan,
                "val_return": val_ret,
                "val_team_food": val_food,
            }
        )
        if val_ret > best[0]:
            best = (val_ret, es.theta.copy(), generation)
        if settings.progress_s > 0 and (
            curve[-1]["wall_s"] - last_print[0] >= settings.progress_s
            or generation == settings.generations
        ):
            last_print[0] = curve[-1]["wall_s"]
            print(
                f"[progress] r={settings.radius} {mode} run {run}: gen {generation}/"
                f"{settings.generations}, val team food {val_food:.2f}, best gen "
                f"{best[2]}, elapsed {curve[-1]['wall_s']:.0f} s",
                flush=True,
            )

    validate(0, None)
    for g in range(settings.generations):
        seeds = [TRAIN_SEED_BASE + run * TRAIN_RUN_STRIDE + g * b + i for i in range(b)]
        cands = es.ask()
        out = population_rollout(
            settings, nets, mode, cands, seeds, TRAIN_SEED_BASE + run * 7919 + g
        )
        fitness = out["return"].mean(axis=1)
        es.tell(fitness)
        episodes += settings.population * b
        gen_done = g + 1
        last = gen_done == settings.generations
        overtime = time.perf_counter() - start > settings.time_limit_s
        if gen_done % settings.eval_every == 0 or last or overtime:
            validate(gen_done, fitness)
        if overtime:
            break
    return {
        "mode": mode,
        "run": run,
        "radius": settings.radius,
        "theta": best[1],
        "theta_final": es.theta.copy(),
        "best_generation": best[2],
        "generations_done": gen_done,
        "curve": curve,
        "seconds": time.perf_counter() - start,
        "n_params": nets.n_params,
    }


# ------------------------------------------------------------------ policies
class LearnedSpecialist(SpecialistPolicy):
    """A trained specialist network as a squad ``SpecialistPolicy``.

    ``act`` samples from the softmax that ``probs`` returns.
    """

    name = "learned-spec"
    salt = 606

    def __init__(self, net: MLP, params: np.ndarray, vocab: int):
        self.net = net
        self.params = np.asarray(params)
        self.vocab = vocab

    def reset(self, seeds: Sequence[int]) -> None:
        self._gen = np.random.default_rng([int(s) for s in seeds] + [self.salt])

    def _logits(self, obs: Observations) -> np.ndarray:
        x = specialist_inputs(obs.specialists, self.vocab)
        batch, n = x.shape[:2]
        return self.net.forward(self.params, x.reshape(batch * n, -1)).reshape(
            batch, n, -1
        )

    def probs(self, obs: Observations) -> np.ndarray:
        return softmax(self._logits(obs))

    def act(self, obs: Observations) -> np.ndarray:
        logits = self._logits(obs)
        gumbel = self._gen.gumbel(size=logits.shape)
        return (logits + gumbel).argmax(axis=2).astype(np.int64)


class LearnedCentre(CentrePolicy):
    """A trained centre network: per receiver, the argmax code of its logits."""

    name = "learned-centre"

    def __init__(self, net: MLP, params: np.ndarray):
        self.net = net
        self.params = np.asarray(params)

    def send(self, obs: Observations) -> np.ndarray:
        feats = centre_features(obs.centre)
        batch, n = feats.shape[:2]
        logits = self.net.forward(self.params, feats.reshape(batch * n, -1))
        return logits.argmax(axis=1).reshape(batch, n, 1).astype(np.int64)


def policies_for(
    mode: str, settings: TrainSettings, theta: np.ndarray
) -> tuple[SpecialistPolicy, CentrePolicy]:
    """The (specialist, centre) pair a trained parameter vector stands for."""
    nets = Nets.build(settings, mode)
    spec_p, centre_p = nets.split(np.asarray(theta))
    specialist = LearnedSpecialist(nets.spec, spec_p, nets.vocab)
    if mode == LEARNED:
        assert nets.centre is not None and centre_p is not None
        return specialist, LearnedCentre(nets.centre, centre_p)
    if mode == HAND_CODED:
        return specialist, HandCodedCentre()
    return specialist, SilentCentre()


# ------------------------------------------------------------------ evaluation
class PerRunSquadSystem:
    """An evaluation system whose run seed s plays training run s's policies."""

    def __init__(
        self,
        config: SquadConfig,
        runs: dict[int, tuple[SpecialistPolicy, CentrePolicy]],
        recording: Optional[Recording] = None,
    ):
        self.config = config
        self.runs = runs
        self.recording = recording

    def policies(self, seed: int) -> tuple[SpecialistPolicy, CentrePolicy]:
        if int(seed) not in self.runs:
            raise KeyError(f"No trained run for seed {seed}")
        return self.runs[int(seed)]

    def __call__(
        self, inputs: Sequence[int], seed: int = 0, hook: Optional[MessageHook] = None
    ) -> dict[str, Any]:
        specialist, centre = self.policies(seed)
        rec = None
        if self.recording is not None:
            rec = Recorder(self.recording.record_every, self.recording.snapshot_every)
        out: dict[str, Any] = dict(
            run_episodes(SquadEnv(self.config), specialist, centre, inputs, hook, rec)
        )
        if rec is not None:
            out["recorder"] = rec
        return out


def test_episode_seeds(seed: int, n_episodes: int) -> list[int]:
    """Held-out test episodes of run seed ``seed``."""
    if not 1 <= n_episodes <= SEED_STRIDE:
        raise ValueError(f"n_episodes must lie in [1, {SEED_STRIDE}]")
    start = TEST_SEED_BASE + int(seed) * SEED_STRIDE
    return list(range(start, start + n_episodes))


test_episode_seeds.__test__ = False  # type: ignore[attr-defined]  # not a pytest test


def make_task(
    radius: int,
    runs: Sequence[int],
    n_episodes: int,
    registry: BaselineRegistry,
    recording: Optional[Recording] = None,
    max_steps: int = 300,
) -> Task:
    """The held-out evaluation task; seeds are training-run indices."""
    config = SquadConfig(obs_radius=radius, max_steps=max_steps)
    spec = TaskSpec(
        name=f"squad-learned-protocol-r{radius}",
        description=(
            "Cooperative squad (21x21 walls, 3 snakes); score = team food per episode "
            f"on held-out seeds; window radius {radius}; seed = training run"
        ),
        metric="team_food",
        n_seeds=len(runs),
        params={
            "obs_radius": radius,
            "n_episodes": n_episodes,
            "test_seed_base": TEST_SEED_BASE,
            "seed_stride": SEED_STRIDE,
            "message_capacity_bits": config.message.bits,
            "message_interval": config.message.interval,
        },
        message=SquadEnv(config).spec.message,
    )
    register_squad_baselines(spec.name, config, registry, recording)
    return Task(spec, lambda seed: test_episode_seeds(seed, n_episodes), team_food)


def message_metrics_per_run(
    condition: ConditionResult,
    systems: PerRunSquadSystem,
    listens: bool = True,
    n_shuffles: int = 100,
    topsim_samples: int = 1500,
    per_run: Optional[dict[str, list[float]]] = None,
) -> dict[str, float]:
    """Message- and listener-side metrics, computed per training run then averaged.

    Each run may learn a different code, so messages are never pooled across runs.
    Returns the mean and the 95% CI half-width (``<key>_ci``) of each metric over runs;
    a ``per_run`` dict, when given, is filled with each metric's per-run values.
    """
    if condition.outputs is None:
        raise ValueError(f"{condition.name!r} was run without keep_outputs")
    per: dict[str, list[float]] = {}
    for seed, outputs in zip(condition.seeds, condition.outputs):
        recorder: Recorder = outputs["recorder"]
        rec = recorder.stacked()
        msgs, label = rec["messages"], rec["label"]
        valid = label != NO_FOOD
        meanings = np.stack([rec["direction"], rec["far"]], axis=1)[valid]
        gen = np.random.default_rng(seed)
        idx = np.flatnonzero(valid)
        if idx.size > topsim_samples:
            idx = np.sort(gen.choice(idx, topsim_samples, replace=False))
        m = communication_metrics(
            msgs, targets=label, vocab_size=8, n_shuffles=n_shuffles, seed=int(seed)
        )
        row = {
            "message_entropy_bits": m["message_entropy_bits"],
            "utilization": m["utilization"],
            "mi_bits": m["mi_bits"],
            "mi_null_q95_bits": m["mi_null_q95_bits"],
            "mi_p_value": m["mi_p_value"],
            "target_entropy_bits": entropy(label),
            "mi_direction_bits": mutual_information(msgs, rec["direction"]),
            "mi_far_bits": mutual_information(msgs, rec["far"]),
            "distinct_codes": float(np.unique(msgs).size),
        }
        if idx.size >= 3:
            meanings = np.stack([rec["direction"], rec["far"]], axis=1)[idx]
            row["topsim"] = communication_metrics(msgs[idx], meanings=meanings)[
                "topsim"
            ]
        if listens:
            specialist, _ = systems.policies(seed)
            row.update(listener_metrics(specialist, recorder, int(seed)))
        for key, value in row.items():
            per.setdefault(key, []).append(float(value))
    if per_run is not None:
        per_run.update(per)
    out: dict[str, float] = {}
    for key, values in per.items():
        finite = [v for v in values if np.isfinite(v)]
        if finite:
            out[key], out[f"{key}_ci"] = mean_ci(finite)
        else:
            out[key], out[f"{key}_ci"] = float("nan"), float("nan")
    return out


def codebook(condition: ConditionResult, seed_index: int = 0) -> list[list[int]]:
    """Counts of code (rows, V) by target label (columns, 2 * direction + far, then
    'no food') for one run: what each learned symbol is used for."""
    if condition.outputs is None:
        raise ValueError("Need outputs")
    rec = condition.outputs[seed_index]["recorder"].stacked()
    table = np.zeros((8, NO_FOOD + 1), dtype=np.int64)
    np.add.at(table, (rec["messages"][:, 0], rec["label"]), 1)
    return table.tolist()


@dataclass(frozen=True)
class EvalSettings:
    """Size of the held-out evaluation."""

    n_episodes: int = 100
    interventions: tuple[str, ...] = ABLATIONS
    record_every: int = 3
    snapshot_every: int = 25
    n_shuffles: int = 100
    n_boot: int = 2000


def evaluate_radius(
    radius: int,
    trained: Sequence[dict[str, Any]],
    train_settings: TrainSettings,
    settings: EvalSettings = EvalSettings(),
) -> ExperimentResult:
    """Score every trained condition at one radius against the scripted references.

    Args:
        trained: ``train_run`` outputs for this radius (any modes, runs 0..R-1 each)
    """
    by_mode: dict[str, dict[int, np.ndarray]] = {}
    for t in trained:
        if t["radius"] == radius:
            by_mode.setdefault(t["mode"], {})[int(t["run"])] = t["theta"]
    runs = sorted(next(iter(by_mode.values())))
    if any(sorted(v) != runs for v in by_mode.values()):
        raise ValueError("Every condition needs the same training runs")
    config = SquadConfig(obs_radius=radius, max_steps=train_settings.max_steps)
    recording = Recording(settings.record_every, settings.snapshot_every)
    registry = BaselineRegistry()
    task = make_task(
        radius, runs, settings.n_episodes, registry, recording, train_settings.max_steps
    )
    tsettings = TrainSettings(**{**asdict(train_settings), "radius": radius})
    systems: dict[str, Any] = {}
    for mode in MODES:
        if mode in by_mode:
            systems[CONDITION[mode]] = PerRunSquadSystem(
                config,
                {
                    r: policies_for(mode, tsettings, th)
                    for r, th in by_mode[mode].items()
                },
                recording,
            )
    systems[SCRIPTED] = PerRunSquadSystem(
        config, {r: (GreedyPolicy(), HandCodedCentre()) for r in runs}, recording
    )
    learned = [CONDITION[m] for m in MODES if m in by_mode]
    result = run_experiment(
        task,
        systems,
        seeds=runs,
        ablations=settings.interventions,
        ablate=learned,
        registry=registry,
        n_boot=settings.n_boot,
        metadata={
            "radius": radius,
            "train_settings": asdict(tsettings),
            "eval_settings": asdict(settings),
        },
        keep_outputs=True,
    )
    result.require_complete()

    result.metadata["per_run_communication"] = {}
    for name in (*learned, SCRIPTED):
        per: dict[str, list[float]] = {}
        result.add_communication(
            name,
            message_metrics_per_run(
                result.condition(name),
                systems[name],
                listens=name != CONDITION[SILENT],
                n_shuffles=settings.n_shuffles,
                per_run=per,
            ),
        )
        result.metadata["per_run_communication"][name] = per
    if CONDITION[LEARNED] in learned:
        cond = result.condition(CONDITION[LEARNED])
        result.metadata["codebooks"] = [codebook(cond, i) for i in range(len(runs))]
    result.metadata["headline"] = headline(result)
    for name in list(systems):
        result.condition(name).outputs = None  # Recorders are large and not JSON
    for name in registry.baselines(task.name):
        result.condition(name).outputs = None
    return result


def headline(result: ExperimentResult) -> list[dict[str, Any]]:
    """The comparisons the experiment is for, Mann-Whitney over training runs + Holm.

    (c) vs (a): is learned communication useful at all; (c) vs (b): learned protocol vs
    hand-coded protocol for learned listeners; (b) vs (a): do learned listeners use
    the hand-coded channel; and each learned system vs the scripted squad.
    """
    names = {c.name for c in result.conditions}
    a, b, c = CONDITION[SILENT], CONDITION[HAND_CODED], CONDITION[LEARNED]
    wanted = [(c, a), (c, b), (b, a), (c, SCRIPTED), (b, SCRIPTED)]
    pairs = [(x, y) for x, y in wanted if x in names and y in names]
    if not pairs:
        return []
    samples = {n: np.asarray(result.condition(n).per_seed) for n in names}
    rows = compare_conditions(samples, pairs=pairs)
    for row in rows:
        row["welch_ci"] = welch_ci(samples[row["a"]], samples[row["b"]])
    return rows


def welch_ci(a: np.ndarray, b: np.ndarray, confidence: float = 0.95) -> float:
    """Half-width of the Welch t-interval for mean(a) - mean(b) (NaN if undefined)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if min(a.size, b.size) < 2:
        return float("nan")
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se2 = va + vb
    if se2 == 0:
        return 0.0
    df = se2**2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    return float(scipy_stats.t.ppf(0.5 + confidence / 2, df) * np.sqrt(se2))
