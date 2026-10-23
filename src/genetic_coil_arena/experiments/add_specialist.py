"""Can a new specialist join a squad that already speaks a learned language? (R7)

Start from a squad trained by ``learned_protocol`` (R6: three colour specialists that
share one network, and a centre), add a fourth snake with a new, fourth colour (the
board gets that colour's two food items), and train only the newcomer's network.
The old specialists and the centre are frozen: ``EvolutionStrategy`` gets one
parameter vector [old specialist, centre (learned only), newcomer] and a mask that is
1 on the newcomer's block only.

Why the frozen centre can talk to the newcomer at all: both R6 networks are
receiver-agnostic. The centre maps each receiver's own-colour food (egocentric, sorted)
and health to a code, and the specialist sees own / other planes rather than colours.
With N = 4 the centre computes the newcomer's features from colour 3's food the same
way and sends it a code in the same language (``tests/test_add_specialist.py`` checks
that permuting snakes permutes the codes).

Conditions (``CONDITIONS``):

- ``learned`` (1): frozen learned centre + its frozen learned specialists, newcomer trains
- ``hand-coded`` (2): frozen ``HandCodedCentre`` + its frozen learned listeners
- ``joint`` (3): all four snakes (one shared network, as in R6) and a learned centre
  trained together from scratch: the reference for compute and newcomer food
- ``silent`` (4): the silent squad + newcomer (no channel: what does listening buy)

Newcomer initialisations (``INITS``): ``scratch`` (fresh random weights: a new kind
of specialist that must learn the language) and ``copy`` (a copy of the squad's
shared specialist weights; colour-agnostic, so this is a zero-shot clone at
generation 0).

Training is R6's: OpenAI-ES, common random numbers within a generation. Fitness is
the team return for the joint squad (as in R6) and, by default, the newcomer's own
return (its food + survival reward) for a newcomer (``AddSettings.fitness``): with the
team return, the frozen snakes' food (about 30 per episode) varies between members
as soon as the newcomer changes their trajectories, and that noise swamps the
newcomer's first food items (a from-scratch newcomer stayed at ~0.06 food after 200
generations in the pilot). Whether the newcomer then harms the old snakes is measured
(``old_food_check``). The validation curve records the newcomer's food, the old
specialists' food and the per-snake mean every ``eval_every`` generations; the best
checkpoint is chosen on the validation value of the fitness.

Seeds: training ``TRAIN_SEED_BASE + run * TRAIN_RUN_STRIDE + g * episodes + i``,
validation ``VAL_SEED_BASE + run * SEED_STRIDE + i`` (both disjoint from R6's),
held-out test ``learned_protocol.test_episode_seeds`` (shared with R6). Run k of every
newcomer condition starts from R6 source run k; run k of ``joint`` from R6's initial
weights of run k.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..envs.base import Observations
from ..envs.squad import (
    HandCodedCentre,
    NoCommPolicy,
    OraclePolicy,
    RandomPolicy,
    SilentCentre,
    SquadConfig,
    SquadEnv,
    run_episodes,
)
from ..envs.squad.policies import CentrePolicy, SpecialistPolicy
from ..evaluation import (
    BaselineRegistry,
    ExperimentResult,
    MessageHook,
    Task,
    TaskSpec,
    mean_ci,
    paired_difference,
    register_baseline,
    run_experiment,
    wilcoxon_signed_rank,
)
from ..learning import EvolutionStrategy
from .learned_protocol import (
    HAND_CODED,
    LEARNED,
    SILENT,
    LearnedCentre,
    LearnedSpecialist,
    Nets,
    TrainSettings,
    _messages,
    specialist_inputs,
    test_episode_seeds,
)

JOINT = "joint"
CONDITIONS = (LEARNED, HAND_CODED, JOINT, SILENT)  # numbered 1-4 in the design
SOURCE_MODES = (LEARNED, HAND_CODED, SILENT)
SCRATCH, COPY = "scratch", "copy"
INITS = (SCRATCH, COPY)
FITNESS = ("newcomer", "team")
ABLATIONS = ("zero", "shuffle_batch", "random_matched")  # on the newcomer's channel

TRAIN_SEED_BASE = 1_500_000_000
TRAIN_RUN_STRIDE = 10_000_000
VAL_SEED_BASE = 4_200_000_000
SEED_STRIDE = 10_000


# ------------------------------------------------------------------ settings
@dataclass(frozen=True)
class AddSettings:
    """One newcomer (or joint) training run's budget and hyperparameters.

    ``n_old`` snakes come from the source squad; the newcomer is snake ``n_old``.
    ``fitness`` is what a newcomer is trained on: ``newcomer`` (its own return:
    its food + the survival reward while it lives) or ``team`` (the team return).
    The joint squad always trains on the team return.
    """

    radius: int = 3
    n_old: int = 3
    generations: int = 300
    population: int = 64
    episodes_per_member: int = 8
    sigma: float = 0.05
    lr: float = 0.03
    weight_decay: float = 0.005
    spec_hidden: int = 16
    centre_hidden: int = 16
    eval_every: int = 10
    val_episodes: int = 64
    max_steps: int = 300
    time_limit_s: float = float("inf")
    progress_s: float = 0.0
    fitness: str = "newcomer"

    def __post_init__(self) -> None:
        if self.fitness not in FITNESS:
            raise ValueError(f"fitness must be one of {FITNESS}")

    @property
    def newcomer(self) -> int:
        return self.n_old

    def squad_config(self, n_snakes: Optional[int] = None) -> SquadConfig:
        n = self.n_old + 1 if n_snakes is None else n_snakes
        return SquadConfig(n_snakes=n, obs_radius=self.radius, max_steps=self.max_steps)

    def source_settings(self) -> TrainSettings:
        """The R6 settings the source networks were built with."""
        return TrainSettings(
            radius=self.radius,
            spec_hidden=self.spec_hidden,
            centre_hidden=self.centre_hidden,
            max_steps=self.max_steps,
        )


def source_key(radius: int, mode: str, run: int) -> str:
    """Key of an R6 trained run in ``learned_protocol_params.npz``."""
    return f"r{radius}_{mode}_run{run}"


# ------------------------------------------------------------------ layout
@dataclass(frozen=True)
class Layout:
    """How one flat ES parameter vector maps onto the squad's networks.

    ``snake_blocks[i]`` is the slice of snake i's specialist parameters (snakes that
    share a network share a slice); ``centre_block`` that of the learned centre.
    ``mask`` is 1 where ES may change parameters.
    """

    nets: Nets
    mode: str  # centre: silent, hand-coded or learned
    theta0: np.ndarray
    mask: np.ndarray
    snake_blocks: tuple[slice, ...]
    centre_block: Optional[slice]

    @property
    def n_params(self) -> int:
        return int(self.theta0.size)

    @property
    def n_trainable(self) -> int:
        return int(self.mask.sum())

    def centre_params(self, params: np.ndarray) -> Optional[np.ndarray]:
        return None if self.centre_block is None else params[..., self.centre_block]

    def groups(self) -> list[tuple[slice, list[int]]]:
        """(parameter block, snakes using it), in order of first use."""
        out: dict[tuple[int, int], tuple[slice, list[int]]] = {}
        for i, block in enumerate(self.snake_blocks):
            out.setdefault((block.start, block.stop), (block, []))[1].append(i)
        return list(out.values())


def newcomer_layout(
    settings: AddSettings,
    mode: str,
    source_theta: np.ndarray,
    run: int,
    init: str = SCRATCH,
) -> Layout:
    """Frozen source squad (``mode`` in ``SOURCE_MODES``) + a trainable newcomer.

    Args:
        source_theta: The R6 parameters of that mode (specialist, then centre)
        init: ``scratch`` (random, ``MLP.init`` with rng [run, 3]) or ``copy`` (the
            source's shared specialist weights)
    """
    if mode not in SOURCE_MODES:
        raise ValueError(
            f"Unknown source mode {mode!r}; expected one of {SOURCE_MODES}"
        )
    nets = Nets.build(settings.source_settings(), mode)
    source_theta = np.asarray(source_theta, dtype=np.float64)
    if source_theta.size != nets.n_params:
        raise ValueError(
            f"Source has {source_theta.size} parameters, {mode} nets need {nets.n_params}"
        )
    k = nets.spec.n_params
    old = slice(0, k)
    if init == SCRATCH:
        new = nets.spec.init(np.random.default_rng([run, 3]))
    elif init == COPY:
        new = source_theta[:k].copy()
    else:
        raise ValueError(f"Unknown init {init!r}; expected one of {INITS}")
    theta0 = np.concatenate([source_theta, new])
    start = source_theta.size
    mask = np.zeros(theta0.size)
    mask[start:] = 1.0
    blocks = (old,) * settings.n_old + (slice(start, start + k),)
    centre = slice(k, start) if nets.centre is not None else None
    return Layout(nets, mode, theta0, mask, blocks, centre)


def joint_layout(settings: AddSettings, run: int) -> Layout:
    """All ``n_old + 1`` snakes share one network, trained with a learned centre from
    R6's initial weights of ``run`` (everything trainable)."""
    nets = Nets.build(settings.source_settings(), LEARNED)
    theta0 = nets.init(run)
    k = nets.spec.n_params
    blocks = (slice(0, k),) * (settings.n_old + 1)
    return Layout(
        nets, LEARNED, theta0, np.ones(theta0.size), blocks, slice(k, theta0.size)
    )


# ------------------------------------------------------------------ rollouts
def population_rollout(
    settings: AddSettings,
    layout: Layout,
    params: np.ndarray,
    seeds: Sequence[int],
    noise_seed: int,
) -> dict[str, np.ndarray]:
    """Every member of ``params`` (P, n) plays the same ``seeds`` with the same action
    noise. Returns ``return`` (team), ``newcomer_return`` (snake ``n_old``'s food +
    survival reward) and ``team_food`` (P, B), and ``food`` (P, B, N).

    A block the mask freezes is the same for every member, so it is evaluated once
    for the whole population (one matmul instead of P).
    """
    params = np.atleast_2d(params)
    members, batch = params.shape[0], len(seeds)
    env = SquadEnv(settings.squad_config())
    n = env.config.n_snakes
    if len(layout.snake_blocks) != n:
        raise ValueError(f"Layout has {len(layout.snake_blocks)} snakes, env has {n}")
    obs = env.reset(np.tile(np.asarray(seeds, dtype=np.int64), members))
    gen = np.random.default_rng(noise_seed)
    nets = layout.nets
    centre_p = layout.centre_params(params)
    centre_members = members
    if layout.centre_block is not None and not layout.mask[layout.centre_block].any():
        assert centre_p is not None
        centre_p, centre_members = centre_p[:1], 1  # frozen: one forward for all
    groups = [
        (block, idx, not layout.mask[block].any()) for block, idx in layout.groups()
    ]
    logits = np.empty((members, batch, n, 3), dtype=np.float32)
    total = np.zeros(members * batch, dtype=np.float64)
    own = np.zeros(members * batch, dtype=np.float64)
    nc = settings.newcomer
    survival = env.config.survival_reward
    done = np.zeros(members * batch, dtype=bool)
    while not done.all():
        msgs = _messages(layout.mode, nets, centre_p, obs, centre_members)
        x = specialist_inputs(obs.specialists, nets.vocab)
        x = x.reshape(members, batch, n, -1)
        for block, idx, frozen in groups:
            xs = x[:, :, idx]
            if frozen:
                out = nets.spec.forward(params[0, block], xs.reshape(-1, xs.shape[-1]))
            else:
                out = nets.spec.forward(
                    params[:, block], xs.reshape(members, -1, xs.shape[-1])
                )
            logits[:, :, idx] = out.reshape(members, batch, len(idx), 3)
        gumbel = gen.gumbel(size=(batch, n, 3)).astype(np.float32)
        actions = (logits + gumbel).argmax(axis=3).reshape(members * batch, n)
        was_done = done
        obs, rewards, done, info = env.step(actions, msgs)
        total += rewards[:, 0]
        step_own = info["food_eaten"][:, nc] + survival * info["alive"][:, nc]
        own += np.where(was_done, 0.0, step_own)
    food = env.score.reshape(members, batch, n).astype(float)
    return {
        "return": total.reshape(members, batch),
        "newcomer_return": own.reshape(members, batch),
        "team_food": food.sum(axis=2),
        "food": food,
    }


# ------------------------------------------------------------------ training
def _seed_offset(condition: str, init: str) -> int:
    return 10 + 2 * CONDITIONS.index(condition) + INITS.index(init)


def train_run(
    condition: str,
    run: int,
    settings: AddSettings = AddSettings(),
    source_theta: Optional[np.ndarray] = None,
    init: str = SCRATCH,
) -> dict[str, Any]:
    """Train one condition's newcomer (or the joint squad) for one run.

    Args:
        condition: One of ``CONDITIONS``
        source_theta: The source squad's R6 parameters (not used by ``joint``)
        init: Newcomer initialisation (ignored by ``joint``)

    Returns:
        ``theta`` (best validation checkpoint, full vector), ``theta_final``,
        ``curve`` rows (generation, episodes, wall_s, pop_mean_return (mean fitness),
        val_return, val_newcomer_return,
        val_team_food, val_newcomer_food, val_old_food, val_per_snake_food; food per
        episode), ``best_generation``, ``generations_done``, ``seconds``,
        ``n_params``, ``n_trainable``
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected {CONDITIONS}")
    if condition == JOINT:
        init = SCRATCH
        layout = joint_layout(settings, run)
    else:
        if source_theta is None:
            raise ValueError(f"Condition {condition!r} needs the source parameters")
        layout = newcomer_layout(settings, condition, source_theta, run, init)
    es = EvolutionStrategy(
        layout.theta0,
        population=settings.population,
        sigma=settings.sigma,
        lr=settings.lr,
        weight_decay=settings.weight_decay,
        seed=run * 1000 + _seed_offset(condition, init),
        mask=layout.mask,
    )
    val_seeds = [
        VAL_SEED_BASE + run * SEED_STRIDE + i for i in range(settings.val_episodes)
    ]
    b = settings.episodes_per_member
    nc = settings.newcomer
    start = time.perf_counter()
    curve: list[dict[str, float]] = []
    best = (-np.inf, es.theta.copy(), 0)
    episodes = 0
    gen_done = 0
    last_print = [-np.inf]
    label = condition if condition == JOINT else f"{condition}/{init}"
    team = condition == JOINT or settings.fitness == "team"
    fit_key = "return" if team else "newcomer_return"
    val_key = "val_return" if team else "val_newcomer_return"

    def validate(generation: int, fitness: Optional[np.ndarray]) -> None:
        nonlocal best
        out = population_rollout(
            settings, layout, es.theta, val_seeds, VAL_SEED_BASE + run
        )
        food = out["food"][0]  # (B, N)
        row = {
            "generation": generation,
            "episodes": episodes,
            "wall_s": time.perf_counter() - start,
            "pop_mean_return": float(fitness.mean()) if fitness is not None else np.nan,
            "val_return": float(out["return"].mean()),
            "val_newcomer_return": float(out["newcomer_return"].mean()),
            "val_team_food": float(food.sum(axis=1).mean()),
            "val_newcomer_food": float(food[:, nc].mean()),
            "val_old_food": float(food[:, :nc].sum(axis=1).mean()),
            "val_per_snake_food": float(food.mean()),
        }
        curve.append(row)
        if row[val_key] > best[0]:
            best = (row[val_key], es.theta.copy(), generation)
        if settings.progress_s > 0 and (
            row["wall_s"] - last_print[0] >= settings.progress_s
            or generation == settings.generations
        ):
            last_print[0] = row["wall_s"]
            print(
                f"[progress] {label} run {run}: gen {generation}/{settings.generations}"
                f", val newcomer food {row['val_newcomer_food']:.2f}, team "
                f"{row['val_team_food']:.2f}, elapsed {row['wall_s']:.0f} s",
                flush=True,
            )

    validate(0, None)
    for g in range(settings.generations):
        seeds = [TRAIN_SEED_BASE + run * TRAIN_RUN_STRIDE + g * b + i for i in range(b)]
        cands = es.ask()
        out = population_rollout(
            settings, layout, cands, seeds, TRAIN_SEED_BASE + run * 7919 + g
        )
        fitness = out[fit_key].mean(axis=1)
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
        "condition": condition,
        "init": init,
        "run": run,
        "theta": best[1],
        "theta_final": es.theta.copy(),
        "best_generation": best[2],
        "generations_done": gen_done,
        "curve": curve,
        "seconds": time.perf_counter() - start,
        "n_params": layout.n_params,
        "n_trainable": layout.n_trainable,
    }


def generations_to(
    curve: Sequence[dict[str, float]], key: str, threshold: float, smooth: int = 3
) -> Optional[dict[str, float]]:
    """First validation point whose trailing ``smooth``-point mean of ``key`` reaches
    ``threshold``; returns its generation, episodes and wall_s (None if never).

    Smoothing over consecutive checkpoints guards against one lucky validation
    score; at the start the window is shorter (so a zero-shot newcomer counts at
    generation 0).
    """
    values = [row[key] for row in curve]
    for i, row in enumerate(curve):
        window = values[max(0, i - smooth + 1) : i + 1]
        if float(np.mean(window)) >= threshold:
            return {
                "generation": float(row["generation"]),
                "episodes": float(row["episodes"]),
                "wall_s": float(row["wall_s"]),
            }
    return None


# ------------------------------------------------------------------ policies
class ComposedSpecialist(SpecialistPolicy):
    """Old specialists' policy for snakes ``< newcomer`` and another for the newcomer.

    Both policies see the whole batch; each one's columns are taken. ``act`` and
    ``probs`` stay consistent because each sub-policy's are.
    """

    name = "composed"

    def __init__(self, old: SpecialistPolicy, new: SpecialistPolicy, newcomer: int):
        self.old, self.new, self.newcomer = old, new, newcomer

    def reset(self, seeds: Sequence[int]) -> None:
        self.old.reset(seeds)
        self.new.reset(seeds)

    def _merge(self, old: np.ndarray, new: np.ndarray) -> np.ndarray:
        out = old.copy()
        out[:, self.newcomer] = new[:, self.newcomer]
        return out

    def act(self, obs: Observations) -> np.ndarray:
        return self._merge(self.old.act(obs), self.new.act(obs))

    def probs(self, obs: Observations) -> np.ndarray:
        return self._merge(self.old.probs(obs), self.new.probs(obs))


def source_policies(
    settings: AddSettings, mode: str, source_theta: np.ndarray
) -> tuple[SpecialistPolicy, CentrePolicy]:
    """The frozen source squad's (specialist, centre)."""
    nets = Nets.build(settings.source_settings(), mode)
    theta = np.asarray(source_theta, dtype=np.float64)
    spec = LearnedSpecialist(nets.spec, theta[: nets.spec.n_params], nets.vocab)
    if mode == LEARNED:
        assert nets.centre is not None
        centre: CentrePolicy = LearnedCentre(nets.centre, theta[nets.spec.n_params :])
    elif mode == HAND_CODED:
        centre = HandCodedCentre()
    else:
        centre = SilentCentre()
    return spec, centre


def trained_policies(
    settings: AddSettings, condition: str, theta: np.ndarray
) -> tuple[SpecialistPolicy, CentrePolicy]:
    """(specialist, centre) a trained ``train_run`` vector stands for, at N = n_old+1."""
    theta = np.asarray(theta, dtype=np.float64)
    if condition == JOINT:
        nets = Nets.build(settings.source_settings(), LEARNED)
        k = nets.spec.n_params
        assert nets.centre is not None
        return (
            LearnedSpecialist(nets.spec, theta[:k], nets.vocab),
            LearnedCentre(nets.centre, theta[k:]),
        )
    nets = Nets.build(settings.source_settings(), condition)
    k = nets.spec.n_params
    old, centre = source_policies(settings, condition, theta[:-k])
    new = LearnedSpecialist(nets.spec, theta[-k:], nets.vocab)
    return ComposedSpecialist(old, new, settings.newcomer), centre


# ------------------------------------------------------------------ evaluation
def newcomer_hook(hook: Optional[MessageHook], newcomer: int) -> Optional[MessageHook]:
    """``hook`` restricted to the newcomer's channel: the live mask is cleared for
    every other receiver, and interventions leave masked-out entries alone."""
    if hook is None:
        return None

    def wrapped(
        messages: np.ndarray, step: int = 0, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        only = np.zeros(messages.shape[:2], dtype=bool)
        only[:, newcomer] = True if mask is None else mask[:, newcomer]
        return hook(messages, step, only)

    return wrapped


class AddSquadSystem:
    """Evaluation system: run seed s plays run s's squad; hooks act on the newcomer's
    channel only."""

    def __init__(
        self,
        config: SquadConfig,
        runs: dict[int, tuple[SpecialistPolicy, CentrePolicy]],
        newcomer: int,
    ):
        self.config, self.runs, self.newcomer = config, runs, newcomer

    def __call__(
        self, inputs: Sequence[int], seed: int = 0, hook: Optional[MessageHook] = None
    ) -> dict[str, Any]:
        if int(seed) not in self.runs:
            raise KeyError(f"No run for seed {seed}")
        specialist, centre = self.runs[int(seed)]
        return run_episodes(
            SquadEnv(self.config),
            specialist,
            centre,
            inputs,
            newcomer_hook(hook, self.newcomer),
        )


def system_name(condition: str, init: str = SCRATCH) -> str:
    """Display name of a trained condition."""
    if condition == JOINT:
        return "(3) joint-retrain"
    number = {LEARNED: 1, HAND_CODED: 2, SILENT: 4}[condition]
    return f"({number}) {condition}+newcomer[{init}]"


def make_task(
    settings: AddSettings,
    runs: Sequence[int],
    n_episodes: int,
    learned_sources: dict[int, np.ndarray],
    registry: BaselineRegistry,
) -> Task:
    """Held-out task: score = the newcomer's food per episode; seeds = run indices.

    Mandatory baselines put a scripted newcomer into run s's frozen learned squad:
    random actions, message-blind greedy (``NoCommPolicy``) and the full-view oracle.
    """
    config = settings.squad_config()
    nc = settings.newcomer
    spec = TaskSpec(
        name=f"squad-add-specialist-r{settings.radius}",
        description=(
            f"Cooperative squad (21x21 walls, {nc} frozen snakes + 1 newcomer); score "
            "= newcomer food per episode on held-out seeds; seed = training run"
        ),
        metric="newcomer_food",
        n_seeds=len(runs),
        params={
            "obs_radius": settings.radius,
            "n_snakes": config.n_snakes,
            "newcomer": nc,
            "n_episodes": n_episodes,
        },
        message=SquadEnv(config).spec.message,
    )
    for name, scripted in (
        ("newcomer:random", RandomPolicy()),
        ("newcomer:no-comm-greedy", NoCommPolicy()),
        ("newcomer:oracle", OraclePolicy()),
    ):
        squads = {}
        for run, theta in learned_sources.items():
            old, centre = source_policies(settings, LEARNED, theta)
            squads[int(run)] = (ComposedSpecialist(old, scripted, nc), centre)
        register_baseline(
            spec.name, name, AddSquadSystem(config, squads, nc), registry=registry
        )

    def checker(inputs: Any, outputs: dict[str, Any]) -> np.ndarray:
        return np.asarray(outputs["food"])[:, nc]

    return Task(spec, lambda seed: test_episode_seeds(seed, n_episodes), checker)


def evaluate(
    settings: AddSettings,
    trained: Sequence[dict[str, Any]],
    sources: dict[tuple[str, int], np.ndarray],
    n_episodes: int = 100,
    n_boot: int = 2000,
) -> ExperimentResult:
    """Newcomer food of every trained condition vs the scripted-newcomer baselines on
    held-out seeds, with ablations of the newcomer's channel.

    Args:
        trained: ``train_run`` outputs (every condition over the same runs)
        sources: (mode, run) -> R6 parameters
    """
    runs = sorted({int(t["run"]) for t in trained})
    registry = BaselineRegistry()
    learned_sources = {r: sources[(LEARNED, r)] for r in runs}
    task = make_task(settings, runs, n_episodes, learned_sources, registry)
    config = settings.squad_config()
    systems: dict[str, AddSquadSystem] = {}
    for t in sorted(
        trained, key=lambda t: (CONDITIONS.index(t["condition"]), t["init"])
    ):
        name = system_name(t["condition"], t["init"])
        system = systems.setdefault(name, AddSquadSystem(config, {}, settings.newcomer))
        system.runs[int(t["run"])] = trained_policies(
            settings, t["condition"], t["theta"]
        )
    for name, system in systems.items():
        if sorted(system.runs) != runs:
            raise ValueError(f"{name} is missing runs: has {sorted(system.runs)}")
    result = run_experiment(
        task,
        systems,
        seeds=runs,
        ablations=ABLATIONS,
        registry=registry,
        n_boot=n_boot,
        keep_outputs=True,
    )
    result.require_complete()
    per_run_food: dict[str, dict[str, list[float]]] = {}
    for cond in result.conditions:
        assert cond.outputs is not None
        food = [np.asarray(o["food"], dtype=float) for o in cond.outputs]
        per_run_food[cond.name] = {
            "team": [float(f.sum(axis=1).mean()) for f in food],
            "old": [float(f[:, : settings.n_old].sum(axis=1).mean()) for f in food],
            "per_snake": [float(f.mean()) for f in food],
        }
        cond.outputs = None
    result.metadata["per_run_food"] = per_run_food
    return result


def old_food_check(
    settings: AddSettings,
    trained: Sequence[dict[str, Any]],
    sources: dict[tuple[str, int], np.ndarray],
    n_episodes: int = 100,
) -> list[dict[str, Any]]:
    """Are the old specialists' food preserved when the newcomer joins?

    Per source mode and run: the old snakes' summed food per held-out episode in the
    original 3-snake squad, in the 4-snake squad with a random newcomer (the board
    changes: a fourth body and two more food items), and with each trained newcomer.
    Differences are paired by run (t-CI, Wilcoxon).
    """
    n_old, nc = settings.n_old, settings.newcomer
    runs = sorted({int(t["run"]) for t in trained})
    by_key = {(t["condition"], t["init"], int(t["run"])): t for t in trained}
    rows = []
    for mode in SOURCE_MODES:
        inits = sorted({k[1] for k in by_key if k[0] == mode})
        if not inits:
            continue
        three, rand = [], []
        trained_food: dict[str, list[float]] = {i: [] for i in inits}
        for run in runs:
            seeds = test_episode_seeds(run, n_episodes)
            spec, centre = source_policies(settings, mode, sources[(mode, run)])
            out = run_episodes(
                SquadEnv(settings.squad_config(n_old)), spec, centre, seeds
            )
            three.append(float(out["food"].sum(axis=1).mean()))
            mixed = ComposedSpecialist(spec, RandomPolicy(), nc)
            out = run_episodes(SquadEnv(settings.squad_config()), mixed, centre, seeds)
            rand.append(float(out["food"][:, :n_old].sum(axis=1).mean()))
            for init in inits:
                pol, cen = trained_policies(
                    settings, mode, by_key[(mode, init, run)]["theta"]
                )
                out = run_episodes(SquadEnv(settings.squad_config()), pol, cen, seeds)
                trained_food[init].append(
                    float(out["food"][:, :n_old].sum(axis=1).mean())
                )
        for init in inits:
            row: dict[str, Any] = {"mode": mode, "init": init, "runs": runs}
            for name, values in (
                ("three_snake", three),
                ("random_newcomer", rand),
                ("trained_newcomer", trained_food[init]),
            ):
                row[name] = values
                row[f"{name}_mean"], row[f"{name}_ci"] = mean_ci(values)
            for ref in ("three_snake", "random_newcomer"):
                if len(runs) >= 2:
                    diff = paired_difference(trained_food[init], row[ref])
                    _, p = wilcoxon_signed_rank(trained_food[init], row[ref])
                else:
                    diff = {"mean": trained_food[init][0] - row[ref][0], "ci": np.nan}
                    p = np.nan
                row[f"vs_{ref}"] = {"mean": diff["mean"], "ci": diff["ci"], "p": p}
            rows.append(row)
    return rows
