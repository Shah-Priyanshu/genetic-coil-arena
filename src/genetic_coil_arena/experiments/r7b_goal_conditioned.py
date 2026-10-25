"""R7b: a learned command centre that follows a GoalSpec (goal G7).

R7a made a translator turn text into a validated ``squad.mission`` GoalSpec and honoured
it by wrapping *scripted* policies. R6 trained a squad protocol with evolution
strategies, but for one fixed task. Here the learned centre is told the GoalSpec and the
squad must do what different goals ask. Can the goal travel through a 3-bit message, and
does that hold for goal combinations never trained on?

Goal space (``squad.mission`` of R7a, validated by ``SpecRegistry.make``):

- ``eaters``: any non-empty subset of the three colours (7 subsets);
- ``caution``: low / normal / high (3 levels).

That is 21 goals. ``step_limit`` and ``food_target`` stay at their defaults (300 and 0):
they are valid spec fields but fixed, so they are neither encoded nor varied here.

Spec encoding for networks (``SPEC_DIM`` = 6, fixed order)::

    [eater_red, eater_green, eater_blue, caution_low, caution_normal, caution_high]

(0/1 floats). A network that acts for one snake also gets ``own_eater`` (that snake's
bit), so ``GoalArrays.extra`` is (B, N, 7) = ``[own_eater, spec(6)]``. The centre has no
receiver identity (as in R6), so it can only use the colour-indexed bits through
``own_eater``.

Mission score per episode (higher is better)::

    sum_{eaters} food  -  NON_TARGET_PENALTY * sum_{others} food
                       -  DEATH_PENALTY[caution] * (# eaters dead at the end)

with ``NON_TARGET_PENALTY`` = 1 and ``DEATH_PENALTY`` = 0 / 2 / 5 for low / normal /
high. The R7a score is eater food (capped) + 0.5 per surviving eater; this variant
penalises non-target eating (a network cannot "blank its window" as the scripted
wrapper does) and makes caution a weight on losing a snake. Snakes that are not ordered
to eat starve after about 60 steps; their deaths are free. Caveat: the greedy squad
rarely dies for another reason, so the caution term has a weak behavioural lever.

Conditions (all trained with OpenAI-ES, ``learning.EvolutionStrategy``, on the 15
training goals; fitness = mean mission score, common random numbers):

- ``blind`` (a): spec-blind learned centre + learned specialists
- ``centre-goal`` (b): the centre sees the spec; specialists are goal-agnostic, so the
  goal must travel through the messages
- ``both-goal`` (c): centre and specialists see the spec
- ``hand-goal`` (d): hand-coded spec-aware centre + learned goal-agnostic listeners.
  R6's code (2 * direction + far) is sent to target colours with code 0 remapped to 1;
  code 0 is sent to non-targets, so "0" means "do not eat"
- scripted reference (e): the R7a wrapper (``NoCommPolicy``, non-eaters' own-food plane
  blanked, wander rate by caution), no training

Generalisation: ``split_goals`` holds out the eater subset {green, blue} (all 3
cautions) and three (subset, caution) combinations whose parts were all seen.
Checkpoints are selected on training goals only.
"""

import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import numpy as np

from ..envs.base import Observations
from ..envs.squad import (
    GreedyPolicy,
    HandCodedCentre,
    NoCommPolicy,
    SquadConfig,
    SquadEnv,
    run_episodes,
)
from ..envs.squad.engine import ALIVE
from ..envs.squad.policies import (
    CentrePolicy,
    RandomPolicy,
    SilentCentre,
    SpecialistPolicy,
)
from ..evaluation import (
    BaselineRegistry,
    Task,
    TaskSpec,
    entropy,
    mi_null_test,
    register_baseline,
)
from ..learning import MLP, EvolutionStrategy
from ..protocol.goalspec import GoalSpec, SpecRegistry
from .learned_protocol import (
    FOOD_FEATURES,
    SEED_STRIDE,
    TRAIN_RUN_STRIDE,
    TRAIN_SEED_BASE,
    VAL_SEED_BASE,
    LearnedSpecialist,
    centre_features,
    entropy_bits,
    specialist_inputs,
    test_episode_seeds,
)
from .r7a_domains import CAUTIONS, COLOURS, EXPLORE, SQUAD_MISSION, build_registry

N_COLOURS = len(COLOURS)
SPEC_DIM = N_COLOURS + len(CAUTIONS)  # 6
EXTRA_DIM = SPEC_DIM + 1  # own_eater + spec
NON_TARGET_PENALTY = 1.0
DEATH_PENALTY = {"low": 0.0, "normal": 2.0, "high": 5.0}
DEFAULT_STEP_LIMIT = 300
DEFAULT_FOOD_TARGET = 0

BLIND = "blind"
CENTRE_GOAL = "centre-goal"
BOTH_GOAL = "both-goal"
HAND_GOAL = "hand-goal"
MODES = (BLIND, CENTRE_GOAL, BOTH_GOAL, HAND_GOAL)
CONDITION = {
    BLIND: "(a) blind-centre+learned-spec",
    CENTRE_GOAL: "(b) goal-centre+learned-spec",
    BOTH_GOAL: "(c) goal-centre+goal-spec",
    HAND_GOAL: "(d) hand-goal-centre+learned-spec",
}
SCRIPTED = "(e) scripted-wrapper (R7a)"
BLIND_GREEDY = "greedy+hand-coded (goal-blind)"
RANDOM_ACTIONS = "random-actions"
SEEN, HELDOUT = "seen", "heldout"
NOVEL_SUBSET, NOVEL_COMBO = "novel_subset", "novel_combo"


# ------------------------------------------------------------------ goals
@dataclass(frozen=True)
class Goal:
    """One point of the goal space: eaters (colour order) and caution."""

    eaters: tuple[str, ...]
    caution: str = "normal"

    def __post_init__(self) -> None:
        order = tuple(c for c in COLOURS if c in self.eaters)
        if not order or len(order) != len(self.eaters):
            raise ValueError(f"eaters must be a non-empty subset of {COLOURS}")
        object.__setattr__(self, "eaters", order)
        if self.caution not in CAUTIONS:
            raise ValueError(f"caution must be one of {CAUTIONS}")

    def params(self) -> dict[str, Any]:
        """``squad.mission`` params (step_limit and food_target at their defaults)."""
        return {
            "eaters": list(self.eaters),
            "caution": self.caution,
            "step_limit": DEFAULT_STEP_LIMIT,
            "food_target": DEFAULT_FOOD_TARGET,
        }

    @property
    def mask(self) -> np.ndarray:
        return np.array([c in self.eaters for c in COLOURS], dtype=bool)

    def spec_vector(self) -> np.ndarray:
        """The documented 6-vector: eater bits, then one-hot caution (float32)."""
        vec = np.zeros(SPEC_DIM, dtype=np.float32)
        vec[:N_COLOURS] = self.mask
        vec[N_COLOURS + CAUTIONS.index(self.caution)] = 1.0
        return vec

    @classmethod
    def from_spec(cls, spec: GoalSpec) -> "Goal":
        """The goal a validated ``squad.mission`` GoalSpec asks for.

        Raises ValueError for a spec whose step_limit or food_target is not the default
        (this experiment does not vary them).
        """
        p = spec.params
        if spec.spec != SQUAD_MISSION:
            raise ValueError(f"expected a {SQUAD_MISSION} spec, got {spec.spec}")
        if (
            int(p["step_limit"]) != DEFAULT_STEP_LIMIT
            or int(p["food_target"]) != DEFAULT_FOOD_TARGET
        ):
            raise ValueError("R7b goals use the default step_limit and food_target")
        return cls(tuple(p["eaters"]), p["caution"])

    def label(self) -> str:
        return f"{'+'.join(c[0] for c in self.eaters)}/{self.caution}"


def all_goals() -> list[Goal]:
    """The 21 goals, subset-major (bitmask 1..7), caution-minor."""
    subsets = [
        tuple(c for i, c in enumerate(COLOURS) if mask >> i & 1)
        for mask in range(1, 2**N_COLOURS)
    ]
    return [Goal(s, c) for s in subsets for c in CAUTIONS]


HELDOUT_SUBSET = ("green", "blue")
HELDOUT_COMBOS = (
    (("red",), "high"),
    (("red", "green", "blue"), "low"),
    (("red", "blue"), "normal"),
)


def split_goals() -> dict[str, list[Goal]]:
    """``train`` (15), ``novel_subset`` (3: {green, blue} at every caution) and
    ``novel_combo`` (3: seen eater subsets and cautions, never together)."""
    combos = {Goal(s, c) for s, c in HELDOUT_COMBOS}
    novel_subset = [g for g in all_goals() if g.eaters == HELDOUT_SUBSET]
    novel_combo = [g for g in all_goals() if g in combos]
    held = set(novel_subset) | set(novel_combo)
    return {
        "train": [g for g in all_goals() if g not in held],
        NOVEL_SUBSET: novel_subset,
        NOVEL_COMBO: novel_combo,
    }


def goal_sets() -> dict[str, list[Goal]]:
    """``seen`` (the training goals) and ``heldout`` (novel subset, then novel combos)."""
    s = split_goals()
    return {SEEN: s["train"], HELDOUT: s[NOVEL_SUBSET] + s[NOVEL_COMBO]}


def make_goal_spec(goal: Goal, registry: Optional[SpecRegistry] = None) -> GoalSpec:
    """The goal as a validated ``squad.mission`` GoalSpec (the only way goals are built)."""
    registry = registry or build_registry()
    return registry.make(SQUAD_MISSION, goal.params())


@dataclass(frozen=True)
class GoalArrays:
    """Per-episode goal arrays for a batch (episode b plays ``goals[b]``)."""

    eaters: np.ndarray  # (B, N) bool
    spec: np.ndarray  # (B, SPEC_DIM) float32
    caution: np.ndarray  # (B,) int index into CAUTIONS
    penalty: np.ndarray  # (B,) float64 death penalty per dead eater

    @classmethod
    def from_goals(cls, goals: Sequence[Goal]) -> "GoalArrays":
        return cls(
            np.stack([g.mask for g in goals]),
            np.stack([g.spec_vector() for g in goals]),
            np.array([CAUTIONS.index(g.caution) for g in goals], dtype=np.int64),
            np.array([DEATH_PENALTY[g.caution] for g in goals], dtype=np.float64),
        )

    def tile(self, reps: int) -> "GoalArrays":
        """``reps`` copies in sequence (member-major, as the population rollouts tile)."""
        return GoalArrays(
            np.tile(self.eaters, (reps, 1)),
            np.tile(self.spec, (reps, 1)),
            np.tile(self.caution, reps),
            np.tile(self.penalty, reps),
        )

    def extra(self) -> np.ndarray:
        """(B, N, EXTRA_DIM) float32: ``[own_eater, spec]`` per snake."""
        b, n = self.eaters.shape
        out = np.empty((b, n, EXTRA_DIM), dtype=np.float32)
        out[:, :, 0] = self.eaters
        out[:, :, 1:] = self.spec[:, None, :]
        return out


def mission_score(
    food: np.ndarray, alive: np.ndarray, arrays: GoalArrays
) -> np.ndarray:
    """(B,) mission score from per-snake food (B, N) and alive flags (B, N)."""
    food = np.asarray(food, dtype=np.float64)
    eat = arrays.eaters
    target = (food * eat).sum(axis=1)
    other = (food * ~eat).sum(axis=1)
    dead_eaters = (eat & ~np.asarray(alive, dtype=bool)).sum(axis=1)
    return target - NON_TARGET_PENALTY * other - arrays.penalty * dead_eaters


def swapped_goals(goals: Sequence[Goal]) -> list[Goal]:
    """For each goal, the next goal (cyclically) with a different eater set; if every
    goal has the same eaters, the next with a different caution."""
    n = len(goals)
    out = []
    for j, g in enumerate(goals):
        pick = None
        for key in (lambda x: x.eaters, lambda x: x.caution):
            for k in range(1, n + 1):
                cand = goals[(j + k) % n]
                if key(cand) != key(g):
                    pick = cand
                    break
            if pick is not None:
                break
        out.append(pick or g)
    return out


# ------------------------------------------------------------------ settings
@dataclass(frozen=True)
class GoalTrainSettings:
    """One training run's budget; the same for every condition.

    A member plays every training goal ``reps`` times per generation.
    """

    radius: int = 3
    generations: int = 2000
    population: int = 128
    reps: int = 1
    sigma: float = 0.05
    lr: float = 0.03
    weight_decay: float = 0.005
    spec_hidden: int = 24
    centre_hidden: int = 24
    eval_every: int = 20
    val_reps: int = 2
    max_steps: int = 300
    time_limit_s: float = float("inf")
    progress_s: float = 0.0

    def squad_config(self) -> SquadConfig:
        return SquadConfig(obs_radius=self.radius, max_steps=self.max_steps)


# ------------------------------------------------------------------ networks
@dataclass(frozen=True)
class GoalNets:
    """Specialist and centre networks of one condition."""

    spec: MLP
    centre: Optional[MLP]
    vocab: int
    spec_goal: bool
    centre_goal: bool

    @classmethod
    def build(cls, settings: GoalTrainSettings, mode: str) -> "GoalNets":
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
        config = settings.squad_config()
        vocab = config.message.vocab_size
        spec_goal = mode == BOTH_GOAL
        centre_goal = mode in (CENTRE_GOAL, BOTH_GOAL)
        n_spec = 4 * (2 * settings.radius + 1) ** 2 + 1 + vocab
        n_spec += EXTRA_DIM if spec_goal else 0
        spec = MLP((n_spec, settings.spec_hidden, 3))
        centre = None
        if mode != HAND_GOAL:
            n_in = FOOD_FEATURES * config.food_per_colour + 1
            n_in += EXTRA_DIM if centre_goal else 0
            centre = MLP((n_in, settings.centre_hidden, vocab))
        return cls(spec, centre, vocab, spec_goal, centre_goal)

    @property
    def n_params(self) -> int:
        return self.spec.n_params + (self.centre.n_params if self.centre else 0)

    def split(self, theta: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        k = self.spec.n_params
        return theta[..., :k], (theta[..., k:] if self.centre else None)

    def init(self, run: int) -> np.ndarray:
        spec = self.spec.init(np.random.default_rng([run, 1]))
        if self.centre is None:
            return spec
        return np.concatenate([spec, self.centre.init(np.random.default_rng([run, 2]))])


def hand_goal_codes(obs: Observations, eaters: np.ndarray) -> np.ndarray:
    """(B, N, 1) spec-aware hand-coded messages: R6's code with 0 remapped to 1 for
    target colours, and 0 ("do not eat") for the others."""
    codes = HandCodedCentre().send(obs)
    codes = np.where(codes == 0, 1, codes)
    return np.where(eaters[:, :, None], codes, 0).astype(np.int64)


def _centre_input(centre_obs: dict[str, np.ndarray], arrays: GoalArrays, goal: bool):
    feats = centre_features(centre_obs)
    return np.concatenate([feats, arrays.extra()], axis=2) if goal else feats


def _goal_messages(
    mode: str,
    nets: GoalNets,
    centre_params: Optional[np.ndarray],
    obs: Observations,
    members: int,
    arrays: GoalArrays,
) -> np.ndarray:
    """(P B, N, 1) codes for the population, member-major."""
    if mode == HAND_GOAL:
        return hand_goal_codes(obs, arrays.eaters)
    assert nets.centre is not None and centre_params is not None
    feats = _centre_input(obs.centre, arrays, nets.centre_goal)
    total, n = feats.shape[:2]
    logits = nets.centre.forward(
        centre_params, feats.reshape(members, (total // members) * n, -1)
    )
    return logits.argmax(axis=2).reshape(total, n, 1).astype(np.int64)


def goal_population_rollout(
    settings: GoalTrainSettings,
    nets: GoalNets,
    mode: str,
    params: np.ndarray,
    seeds: Sequence[int],
    goals: Sequence[Goal],
    noise_seed: int,
) -> dict[str, np.ndarray]:
    """Every member of ``params`` (P, n) plays the same ``seeds`` and ``goals`` (episode
    b plays ``goals[b]``) with the same action noise; returns ``mission``,
    ``team_food`` as (P, B) and ``message_entropy`` (P,) in bits."""
    if len(seeds) != len(goals):
        raise ValueError("one goal per episode seed")
    params = np.atleast_2d(params)
    members, batch = params.shape[0], len(seeds)
    arrays = GoalArrays.from_goals(goals).tile(members)
    spec_p, centre_p = nets.split(params)
    env = SquadEnv(settings.squad_config())
    obs = env.reset(np.tile(np.asarray(seeds, dtype=np.int64), members))
    n = env.config.n_snakes
    gen = np.random.default_rng(noise_seed)
    vocab = nets.vocab
    counts = np.zeros(members * vocab, dtype=np.float64)
    member = np.repeat(np.arange(members), batch)[:, None] * vocab
    extra = arrays.extra() if nets.spec_goal else None
    done = np.zeros(members * batch, dtype=bool)
    while not done.all():
        msgs = _goal_messages(mode, nets, centre_p, obs, members, arrays)
        live = obs.centre["alive"] & ~done[:, None]
        counts += np.bincount((member + msgs[:, :, 0])[live], minlength=members * vocab)
        x = specialist_inputs(obs.specialists, vocab)
        if extra is not None:
            x = np.concatenate([x, extra], axis=2)
        logits = nets.spec.forward(spec_p, x.reshape(members, batch * n, -1))
        gumbel = gen.gumbel(size=(batch * n, 3)).astype(np.float32)
        actions = (logits + gumbel).argmax(axis=2).reshape(members * batch, n)
        obs, _, done, _ = env.step(actions, msgs)
    mission = mission_score(env.score, env.death == ALIVE, arrays)
    return {
        "mission": mission.reshape(members, batch),
        "team_food": env.score.sum(axis=1).reshape(members, batch).astype(float),
        "message_entropy": entropy_bits(counts.reshape(members, vocab)),
    }


# ------------------------------------------------------------------ training
def train_run(
    mode: str, run: int, settings: GoalTrainSettings = GoalTrainSettings()
) -> dict[str, Any]:
    """Train one condition from one seed on the 15 training goals.

    Returns ``theta`` (best validation checkpoint, validated on training goals only),
    ``theta_final``, ``curve`` (rows of generation, episodes, wall_s, pop_mean,
    pop_max, val_mission, val_team_food, val_message_entropy), ``best_generation``,
    ``generations_done``, ``seconds``, ``n_params``.
    """
    nets = GoalNets.build(settings, mode)
    train_goals = split_goals()["train"]
    gen_goals = train_goals * settings.reps
    val_goals = train_goals * settings.val_reps
    es = EvolutionStrategy(
        nets.init(run),
        population=settings.population,
        sigma=settings.sigma,
        lr=settings.lr,
        weight_decay=settings.weight_decay,
        seed=run * 1000 + MODES.index(mode),
    )
    val_seeds = [VAL_SEED_BASE + run * SEED_STRIDE + i for i in range(len(val_goals))]
    b = len(gen_goals)
    start = time.perf_counter()
    curve: list[dict[str, float]] = []
    best = (-np.inf, es.theta.copy(), 0)
    episodes = 0
    last_print = [0.0]

    def validate(generation: int, pop: Optional[np.ndarray]) -> None:
        nonlocal best
        out = goal_population_rollout(
            settings, nets, mode, es.theta, val_seeds, val_goals, VAL_SEED_BASE + run
        )
        val = float(out["mission"].mean())
        curve.append(
            {
                "generation": generation,
                "episodes": episodes,
                "wall_s": time.perf_counter() - start,
                "pop_mean": float(pop.mean()) if pop is not None else math.nan,
                "pop_max": float(pop.max()) if pop is not None else math.nan,
                "val_mission": val,
                "val_team_food": float(out["team_food"].mean()),
                "val_message_entropy": float(out["message_entropy"][0]),
            }
        )
        if val > best[0]:
            best = (val, es.theta.copy(), generation)
        if settings.progress_s > 0 and (
            curve[-1]["wall_s"] - last_print[0] >= settings.progress_s
            or generation == settings.generations
        ):
            last_print[0] = curve[-1]["wall_s"]
            print(
                f"[progress] {mode} run {run}: gen {generation}/{settings.generations},"
                f" val mission {val:.2f}, best gen {best[2]}, "
                f"elapsed {curve[-1]['wall_s']:.0f} s",
                flush=True,
            )

    validate(0, None)
    gen_done = 0
    for g in range(settings.generations):
        seeds = [TRAIN_SEED_BASE + run * TRAIN_RUN_STRIDE + g * b + i for i in range(b)]
        cands = es.ask()
        out = goal_population_rollout(
            settings,
            nets,
            mode,
            cands,
            seeds,
            gen_goals,
            TRAIN_SEED_BASE + run * 7919 + g,
        )
        fitness = out["mission"].mean(axis=1)
        es.tell(fitness)
        episodes += settings.population * b
        gen_done = g + 1
        overtime = time.perf_counter() - start > settings.time_limit_s
        last = gen_done == settings.generations
        if gen_done % settings.eval_every == 0 or last or overtime:
            validate(gen_done, fitness)
        if overtime:
            break
    return {
        "mode": mode,
        "run": run,
        "theta": best[1],
        "theta_final": es.theta.copy(),
        "best_generation": best[2],
        "best_val_mission": float(best[0]),
        "generations_done": gen_done,
        "curve": curve,
        "seconds": time.perf_counter() - start,
        "n_params": nets.n_params,
    }


# ------------------------------------------------------------------ policies
class _GoalAware:
    """Mixin: the per-episode goals a policy is told, set by ``GoalSystem``."""

    arrays: Optional[GoalArrays] = None

    def set_goals(self, arrays: GoalArrays) -> None:
        self.arrays = arrays

    def _arrays(self, batch: int) -> GoalArrays:
        if self.arrays is None or self.arrays.eaters.shape[0] != batch:
            raise RuntimeError("set_goals() with one goal per episode before use")
        return self.arrays


class GoalSpecialist(_GoalAware, LearnedSpecialist):
    """A trained specialist network; with ``use_goal`` it also reads ``[own_eater, spec]``."""

    name = "goal-spec"

    def __init__(self, net: MLP, params: np.ndarray, vocab: int, use_goal: bool):
        super().__init__(net, params, vocab)
        self.use_goal = use_goal

    def _logits(self, obs: Observations) -> np.ndarray:
        x = specialist_inputs(obs.specialists, self.vocab)
        batch, n = x.shape[:2]
        if self.use_goal:
            x = np.concatenate([x, self._arrays(batch).extra()], axis=2)
        return self.net.forward(self.params, x.reshape(batch * n, -1)).reshape(
            batch, n, -1
        )


class GoalCentre(_GoalAware, CentrePolicy):
    """A trained centre network (argmax code per receiver); optionally goal-conditioned."""

    name = "goal-centre"

    def __init__(self, net: MLP, params: np.ndarray, use_goal: bool):
        self.net = net
        self.params = np.asarray(params)
        self.use_goal = use_goal

    def send(self, obs: Observations) -> np.ndarray:
        batch = obs.centre["heading"].shape[0]
        arrays = self._arrays(batch) if self.use_goal else None
        feats = centre_features(obs.centre)
        if arrays is not None:
            feats = np.concatenate([feats, arrays.extra()], axis=2)
        n = feats.shape[1]
        logits = self.net.forward(self.params, feats.reshape(batch * n, -1))
        return logits.argmax(axis=1).reshape(batch, n, 1).astype(np.int64)


class GoalHandCodedCentre(_GoalAware, CentrePolicy):
    """(d): R6's hand-coded code restricted to target colours (see ``hand_goal_codes``)."""

    name = "hand-goal"

    def send(self, obs: Observations) -> np.ndarray:
        batch = obs.centre["heading"].shape[0]
        return hand_goal_codes(obs, self._arrays(batch).eaters)


class ScriptedGoalSpecialist(_GoalAware, SpecialistPolicy):
    """(e) the R7a wrapper: ``NoCommPolicy`` with non-eaters' own-food plane blanked and
    the wander rate chosen per episode by caution (``EXPLORE``)."""

    name = "scripted-wrapper"

    def __init__(self) -> None:
        self.policies = [NoCommPolicy(explore=EXPLORE[c]) for c in CAUTIONS]

    def reset(self, seeds: Sequence[int]) -> None:
        for p in self.policies:
            p.reset(seeds)

    def act(self, obs: Observations) -> np.ndarray:
        windows = obs.specialists["window"]
        arrays = self._arrays(windows.shape[0])
        windows = windows.copy()
        windows[~arrays.eaters, 2] = 0  # blind to own-colour food: they just wander
        seen = Observations(
            specialists={**obs.specialists, "window": windows}, centre=obs.centre
        )
        out = np.zeros(windows.shape[:2], dtype=np.int64)
        for idx, policy in enumerate(self.policies):
            pick = arrays.caution == idx
            if pick.any():
                out[pick] = policy.act(seen)[pick]
        return out


def policies_for(
    mode: str, settings: GoalTrainSettings, theta: np.ndarray
) -> tuple[SpecialistPolicy, CentrePolicy]:
    """The (specialist, centre) pair a trained parameter vector stands for."""
    nets = GoalNets.build(settings, mode)
    spec_p, centre_p = nets.split(np.asarray(theta))
    specialist = GoalSpecialist(nets.spec, spec_p, nets.vocab, nets.spec_goal)
    if mode == HAND_GOAL:
        return specialist, GoalHandCodedCentre()
    assert nets.centre is not None and centre_p is not None
    return specialist, GoalCentre(nets.centre, centre_p, nets.centre_goal)


# ------------------------------------------------------------------ evaluation
class GoalRecorder:
    """``run_episodes`` observer: live receivers' messages, with episode and receiver."""

    def __init__(self, every: int = 3):
        self.every = every
        self._m: list[np.ndarray] = []
        self._e: list[np.ndarray] = []
        self._r: list[np.ndarray] = []

    def __call__(
        self, step: int, obs: Observations, sent: np.ndarray, done: np.ndarray
    ) -> None:
        if step % self.every:
            return
        live = obs.centre["alive"] & ~done[:, None]
        eps, recv = np.nonzero(live)
        self._m.append(np.asarray(sent)[eps, recv, 0])
        self._e.append(eps)
        self._r.append(recv)

    def stacked(self) -> dict[str, np.ndarray]:
        def cat(parts: list[np.ndarray]) -> np.ndarray:
            return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

        return {
            "messages": cat(self._m).astype(np.int64),
            "episode": cat(self._e).astype(np.int64),
            "receiver": cat(self._r).astype(np.int64),
        }


class GoalSystem:
    """An evaluation system for the harness: episode j of a call plays
    ``goals[j % len(goals)]``; run seed s plays ``policies(s)``.

    Args:
        policies: ``seed -> (specialist, centre)``
        swap: Tell the policies the goal of ``swapped_goals`` instead (the score still
            uses the true goal): the goal-swap ablation
        record_every: Keep the centre's messages every this many steps (0: none)
    """

    def __init__(
        self,
        config: SquadConfig,
        goals: Sequence[Goal],
        policies: Callable[[int], tuple[SpecialistPolicy, CentrePolicy]],
        swap: bool = False,
        record_every: int = 0,
    ):
        self.config = config
        self.goals = list(goals)
        self.policies = policies
        self.swap = swap
        self.record_every = record_every

    def __call__(
        self, inputs: Sequence[int], seed: int = 0, hook: Any = None
    ) -> dict[str, Any]:
        specialist, centre = self.policies(int(seed))
        n = len(inputs)
        index = np.arange(n) % len(self.goals)
        true = [self.goals[i] for i in index]
        told = swapped_goals(true) if self.swap else true
        told_arrays = GoalArrays.from_goals(told)
        for policy in (specialist, centre):
            if isinstance(policy, _GoalAware):
                policy.set_goals(told_arrays)
        rec = GoalRecorder(self.record_every) if self.record_every else None
        out = run_episodes(SquadEnv(self.config), specialist, centre, inputs, hook, rec)
        alive = out["death_reason"] == ALIVE
        arrays = GoalArrays.from_goals(true)
        result: dict[str, Any] = {
            "mission": mission_score(out["food"], alive, arrays),
            "team_food": out["team_food"],
            "goal_index": index,
            "arrays": arrays,
        }
        if rec is not None:
            result["recorder"] = rec.stacked()
        return result


def mission_checker(inputs: Any, outputs: dict[str, Any]) -> np.ndarray:
    return np.asarray(outputs["mission"], dtype=float)


def make_goal_task(
    name: str,
    goals: Sequence[Goal],
    runs: Sequence[int],
    per_goal: int,
    config: SquadConfig,
    registry: BaselineRegistry,
    record_every: int = 0,
) -> Task:
    """The held-out task for one goal set: ``per_goal`` test episodes of every goal;
    the run seed is the training-run index. Goal-blind baselines are registered."""
    n = len(goals) * per_goal
    spec = TaskSpec(
        name=name,
        description=(
            f"Squad mission score over {len(goals)} goals ({'; '.join(g.label() for g in goals)}), "
            f"{per_goal} held-out episodes per goal; seed = training run"
        ),
        metric="mission_score",
        n_seeds=len(runs),
        params={
            "goals": [g.params() for g in goals],
            "per_goal": per_goal,
            "non_target_penalty": NON_TARGET_PENALTY,
            "death_penalty": DEATH_PENALTY,
            "obs_radius": config.obs_radius,
            "message_capacity_bits": config.message.bits,
        },
        message=SquadEnv(config).spec.message,
    )
    fixed = {
        RANDOM_ACTIONS: lambda s: (RandomPolicy(), SilentCentre()),
        BLIND_GREEDY: lambda s: (GreedyPolicy(), HandCodedCentre()),
        SCRIPTED: lambda s: (ScriptedGoalSpecialist(), SilentCentre()),
    }
    for label, factory in fixed.items():
        register_baseline(
            name, label, GoalSystem(config, goals, factory), registry=registry
        )
    return Task(spec, lambda seed: test_episode_seeds(seed, n), mission_checker)


# ------------------------------------------------------------------ message metrics
def _probe_accuracy(msgs: np.ndarray, labels: np.ndarray, episode: np.ndarray):
    """Held-out-episode plug-in probe: majority label per message code, fitted on even
    episodes and scored on odd ones. Returns (accuracy, majority-class accuracy)."""
    train, test = episode % 2 == 0, episode % 2 == 1
    if not train.any() or not test.any():
        return math.nan, math.nan
    k = int(labels.max()) + 1
    table = np.zeros((int(msgs.max()) + 1, k))
    np.add.at(table, (msgs[train], labels[train]), 1)
    fallback = int(np.bincount(labels[train], minlength=k).argmax())
    guess = np.where(table.sum(axis=1) > 0, table.argmax(axis=1), fallback)
    acc = float((guess[msgs[test]] == labels[test]).mean())
    base = float((labels[test] == fallback).mean())
    return acc, base


def goal_message_metrics(
    outputs: dict[str, Any], n_shuffles: int = 100, seed: int = 0
) -> dict[str, float]:
    """Do the messages carry the goal? For one run's outputs (recorded system).

    MI (bits, Miller-Madow) between a live receiver's message and: its own eater bit
    (``own_eater``), the episode's caution (``caution``) and the goal id (``goal``),
    each with its 95th-percentile shuffle null; probe accuracy on held-out episodes
    against the majority-class accuracy; and the message entropy in bits.
    """
    rec, arrays = outputs["recorder"], outputs["arrays"]
    msgs, eps, recv = rec["messages"], rec["episode"], rec["receiver"]
    if msgs.size < 4:
        return {}
    labels = {
        "own_eater": arrays.eaters[eps, recv].astype(np.int64),
        "caution": arrays.caution[eps],
        "goal": outputs["goal_index"][eps],
    }
    out = {"message_entropy_bits": entropy(msgs)}
    for name, lab in labels.items():
        test = mi_null_test(msgs, lab, n_shuffles=n_shuffles, seed=seed)
        out[f"mi_{name}_bits"] = float(test["mi"])
        out[f"mi_{name}_null_q95_bits"] = float(test["null_q95"])
        acc, base = _probe_accuracy(msgs, lab, eps)
        out[f"probe_{name}_acc"] = acc
        out[f"probe_{name}_chance"] = base
    return out


def settings_dict(settings: GoalTrainSettings) -> dict[str, Any]:
    return asdict(settings)
