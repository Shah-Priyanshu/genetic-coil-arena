"""Task runner: systems and mandatory baselines across seeds, plus message ablations.

A task is a spec, a way to build the inputs (episodes) for a seed, and a checker that
scores a system's outputs on those inputs. A system is any callable

    system(inputs, seed=seed, hook=hook) -> outputs

where ``hook`` is ``None`` for an intact run or a ``MessageHook`` the system's
environment loop must apply to messages between send and receive (see
``interventions.apply_hook``). Baselines are systems too; they may ignore the hook.

Each seed yields one score (the mean over that seed's episodes), so the seeds are the
independent runs that confidence intervals and tests are computed over. Systems,
baselines and ablations all see the same inputs for a given seed, which makes the
ablation deltas paired: they are tested with the Wilcoxon signed-rank test, while
system-vs-baseline comparisons use Mann-Whitney.

``evaluate(..., keep_outputs=True)`` keeps each seed's system outputs (per-episode
artifacts) and ``record=True`` wraps each seed's hook in a ``RecordingHook``, so metrics
can be computed from the scored runs without replaying them.

Nothing here imports an environment: tasks plug in through callables.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..messages import MessageSpec
from .interventions import (
    ABLATIONS,
    INTERVENTIONS,
    InterventionHook,
    MessageHook,
    RecordingHook,
    default_ablations,
)
from .report import BASELINE, SYSTEM, AblationResult, ConditionResult, ExperimentResult
from .stats import (
    compare_conditions,
    holm_correction,
    paired_difference,
    vargha_delaney_a12,
    wilcoxon_signed_rank,
)

System = Callable[..., Any]
Checker = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class TaskSpec:
    """Static description of a task: what is measured and how runs are seeded.

    ``message`` optionally describes the channel the ablations act on; when given,
    intervention hooks validate messages against it, and with ``n_receivers == 1``
    the default ablations leave out ``swap_receivers``.
    """

    name: str
    description: str = ""
    metric: str = "score"
    higher_is_better: bool = True
    n_seeds: int = 10
    seed_offset: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    message: Optional[MessageSpec] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TaskSpec needs a name")
        if self.n_seeds < 1:
            raise ValueError("n_seeds must be positive")
        if self.seed_offset < 0:
            raise ValueError("seed_offset must be non-negative")

    def seeds(self, n_seeds: Optional[int] = None) -> list[int]:
        """The task's run seeds: seed_offset, seed_offset + 1, ..."""
        n = self.n_seeds if n_seeds is None else n_seeds
        return [self.seed_offset + i for i in range(n)]


@dataclass
class Task:
    """(spec, inputs, checker).

    Args:
        spec: Static description
        episodes: Either a callable ``seed -> inputs`` (preferred: fresh episodes per
            seed) or fixed inputs shared by every seed
        checker: ``(inputs, outputs) -> scores``: a scalar or one score per episode
        baselines: Baselines this task always requires, name -> system; merged with
            those in the registry
    """

    spec: TaskSpec
    episodes: Any
    checker: Checker
    baselines: dict[str, System] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.name

    def inputs(self, seed: int) -> Any:
        return self.episodes(seed) if callable(self.episodes) else self.episodes

    def score(
        self, system: System, seed: int, hook: Optional[MessageHook] = None
    ) -> tuple[float, int]:
        """Run ``system`` on this seed's inputs; returns (mean score, n_episodes)."""
        scores, _ = self.run(system, seed, hook)
        return float(scores.mean()), int(scores.size)

    def run(
        self, system: System, seed: int, hook: Optional[MessageHook] = None
    ) -> tuple[np.ndarray, Any]:
        """Run ``system`` on this seed's inputs; returns (per-episode scores, outputs)."""
        inputs = self.inputs(seed)
        outputs = system(inputs, seed=seed, hook=hook)
        scores = np.asarray(self.checker(inputs, outputs), dtype=float).ravel()
        if scores.size == 0:
            raise ValueError(f"Checker for task {self.name!r} returned no scores")
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                f"Checker for task {self.name!r} returned non-finite scores; "
                "refusing to average over a silently failing evaluation"
            )
        return scores, outputs


class BaselineRegistry:
    """Mandatory baselines per task name. A report without them is incomplete."""

    def __init__(self) -> None:
        self._baselines: dict[str, dict[str, System]] = {}

    def register(self, task_name: str, name: str, system: System) -> None:
        """Make ``name`` a mandatory baseline for ``task_name``."""
        if not callable(system):
            raise TypeError("A baseline must be callable")
        self._baselines.setdefault(task_name, {})[name] = system

    def unregister(self, task_name: str, name: str) -> None:
        self._baselines.get(task_name, {}).pop(name, None)

    def baselines(self, task_name: str) -> dict[str, System]:
        return dict(self._baselines.get(task_name, {}))

    def required(self, task_name: str) -> list[str]:
        return list(self._baselines.get(task_name, {}))

    def __contains__(self, task_name: str) -> bool:
        return bool(self._baselines.get(task_name))


# Process-wide registry that environment packages register their baselines into
BASELINES = BaselineRegistry()


def register_baseline(
    task_name: str,
    name: str,
    system: Optional[System] = None,
    registry: Optional[BaselineRegistry] = None,
) -> Any:
    """Register a mandatory baseline; usable directly or as a decorator."""
    target = BASELINES if registry is None else registry

    if system is not None:
        target.register(task_name, name, system)
        return system

    def decorator(fn: System) -> System:
        target.register(task_name, name, fn)
        return fn

    return decorator


def required_baselines(
    task: Task, registry: Optional[BaselineRegistry] = None
) -> dict[str, System]:
    """Registry baselines for the task, overridden by the task's own."""
    merged = (BASELINES if registry is None else registry).baselines(task.name)
    merged.update(task.baselines)
    return merged


def evaluate(
    task: Task,
    system: System,
    seeds: Sequence[int],
    name: str = "system",
    role: str = SYSTEM,
    hook_factory: Optional[Callable[[int], MessageHook]] = None,
    confidence: float = 0.95,
    n_boot: int = 2000,
    keep_outputs: bool = False,
    record: bool = False,
) -> ConditionResult:
    """Score one condition on every seed.

    Args:
        hook_factory: ``seed -> hook`` for intervened runs; ``None`` for intact runs
        keep_outputs: Keep each seed's system outputs in ``result.outputs``
        record: Wrap each seed's hook (or ``None``) in a ``RecordingHook``, kept in
            ``result.recordings``; it records messages as sent, before intervention

    Returns:
        A ``ConditionResult``; ``outputs`` and ``recordings`` are in seed order and
        stay in memory only
    """
    if len(seeds) == 0:
        raise ValueError("Need at least one seed")
    per_seed, n_episodes = [], 0
    outputs: list[Any] = []
    recordings: list[RecordingHook] = []
    for seed in seeds:
        hook = None if hook_factory is None else hook_factory(int(seed))
        if record:
            hook = RecordingHook(hook)
            recordings.append(hook)
        scores, out = task.run(system, int(seed), hook)
        per_seed.append(float(scores.mean()))
        n_episodes += int(scores.size)
        if keep_outputs:
            outputs.append(out)
    result = ConditionResult.from_scores(
        name, role, seeds, per_seed, n_episodes, confidence=confidence, n_boot=n_boot
    )
    result.outputs = outputs if keep_outputs else None
    result.recordings = recordings if record else None
    return result


def _intervention_stream(name: str) -> int:
    """Stable per-intervention stream id so different ablations draw independently."""
    names = sorted(INTERVENTIONS)
    return names.index(name) if name in names else len(names)


def run_ablations(
    task: Task,
    system: System,
    seeds: Sequence[int],
    interventions: Optional[Sequence[str]] = None,
    intact: Optional[ConditionResult] = None,
    name: str = "system",
    confidence: float = 0.95,
    n_boot: int = 2000,
    keep_outputs: bool = False,
    record: bool = False,
) -> list[AblationResult]:
    """Performance drop when each intervention is applied to the system's messages.

    Deltas are intact minus ablated per seed (sign flipped when lower is better), so a
    positive delta always means performance was lost. ``listening`` is true when the
    delta's CI lies above zero. p-values come from the Wilcoxon signed-rank test on the
    per-seed pairs, Holm-corrected across the interventions.

    Args:
        interventions: Names from ``INTERVENTIONS``; default
            ``default_ablations(task.spec.message.n_receivers)``
        keep_outputs, record: Passed to ``evaluate`` for the ablated runs
    """
    if interventions is None:
        interventions = _default_ablations(task)
    seeds = [int(s) for s in seeds]
    if intact is None:
        intact = evaluate(
            task, system, seeds, name=name, confidence=confidence, n_boot=n_boot
        )
    if list(intact.seeds) != seeds:
        raise ValueError("Intact and ablated runs must use the same seeds")
    sign = 1.0 if task.spec.higher_is_better else -1.0

    results: list[AblationResult] = []
    for intervention in interventions:
        stream = _intervention_stream(intervention)

        def factory(seed: int, _name: str = intervention, _stream: int = stream) -> Any:
            return InterventionHook(
                _name, seed=seed, spec=task.spec.message, stream=_stream
            )

        ablated = evaluate(
            task,
            system,
            seeds,
            name=f"{name}|{intervention}",
            role="ablation",
            hook_factory=factory,
            confidence=confidence,
            n_boot=n_boot,
            keep_outputs=keep_outputs,
            record=record,
        )
        intact_scores = sign * np.asarray(intact.per_seed)
        ablated_scores = sign * np.asarray(ablated.per_seed)
        diff = paired_difference(intact_scores, ablated_scores, confidence)
        _, p = wilcoxon_signed_rank(intact_scores, ablated_scores)
        results.append(
            AblationResult(
                system=name,
                intervention=intervention,
                intact_mean=intact.mean,
                ablated=ablated,
                delta_mean=diff["mean"],
                delta_ci=diff["ci"],
                p=p,
                p_holm=p,
                a12=vargha_delaney_a12(intact_scores, ablated_scores),
                listening=bool(np.isfinite(diff["low"]) and diff["low"] > 0),
            )
        )

    for result, adjusted in zip(results, holm_correction([r.p for r in results])):
        result.p_holm = float(adjusted)
    return results


def run_experiment(
    task: Task,
    systems: Mapping[str, System],
    seeds: Optional[Sequence[int]] = None,
    ablations: Optional[Sequence[str]] = ABLATIONS,
    ablate: Optional[Sequence[str]] = None,
    registry: Optional[BaselineRegistry] = None,
    confidence: float = 0.95,
    n_boot: int = 2000,
    strict: bool = True,
    metadata: Optional[dict[str, Any]] = None,
    keep_outputs: bool = False,
    record: bool = False,
) -> ExperimentResult:
    """Run systems and every mandatory baseline, compare them, and ablate messages.

    Args:
        task: The task
        systems: Name -> system under test
        seeds: Run seeds (default ``task.spec.seeds()``)
        ablations: Interventions to apply to each ablated system; ``None`` or empty
            skips ablations. The default ``ABLATIONS`` drops ``swap_receivers`` when
            ``task.spec.message`` has a single receiver
        ablate: Which systems to ablate (default all of them; baselines never)
        registry: Baseline registry (default the global ``BASELINES``)
        strict: Raise if the task has no mandatory baseline at all
        metadata: Free-form context stored in the result
        keep_outputs, record: Passed to ``evaluate`` for the intact runs of systems
            and baselines (not ablations), so their per-seed outputs / recordings are
            on ``result.condition(name)``

    Returns:
        An ``ExperimentResult``; comparisons are every system vs every baseline,
        Holm-corrected as one family, oriented so A12 > 0.5 means the system is better
    """
    seeds = task.spec.seeds() if seeds is None else [int(s) for s in seeds]
    baselines = required_baselines(task, registry)
    if strict and not baselines:
        raise ValueError(
            f"Task {task.name!r} has no mandatory baselines; register some "
            "with register_baseline() or pass strict=False"
        )
    overlap = set(systems) & set(baselines)
    if overlap:
        raise ValueError(
            f"Names used for both systems and baselines: {sorted(overlap)}"
        )

    result = ExperimentResult(
        task=_spec_dict(task.spec),
        seeds=list(seeds),
        required_baselines=list(baselines),
        metadata=dict(metadata or {}),
    )
    for name, system in systems.items():
        result.conditions.append(
            evaluate(
                task,
                system,
                seeds,
                name,
                SYSTEM,
                confidence=confidence,
                n_boot=n_boot,
                keep_outputs=keep_outputs,
                record=record,
            )
        )
    for name, baseline in baselines.items():
        result.conditions.append(
            evaluate(
                task,
                baseline,
                seeds,
                name,
                BASELINE,
                confidence=confidence,
                n_boot=n_boot,
                keep_outputs=keep_outputs,
                record=record,
            )
        )

    sign = 1.0 if task.spec.higher_is_better else -1.0
    samples = {c.name: sign * np.asarray(c.per_seed) for c in result.conditions}
    pairs = [(s, b) for s in systems for b in baselines]
    if pairs:
        result.comparisons = compare_conditions(samples, pairs=pairs)
        for row in result.comparisons:
            row["mean_diff"] *= sign  # Report the raw-score difference

    if ablations is ABLATIONS:
        ablations = _default_ablations(task)
    if ablations:
        targets = list(systems) if ablate is None else list(ablate)
        for name in targets:
            if name not in systems:
                raise KeyError(f"Cannot ablate unknown system {name!r}")
            result.ablations.extend(
                run_ablations(
                    task,
                    systems[name],
                    seeds,
                    ablations,
                    intact=result.condition(name),
                    name=name,
                    confidence=confidence,
                    n_boot=n_boot,
                )
            )
    return result


def _default_ablations(task: Task) -> tuple[str, ...]:
    message = task.spec.message
    return default_ablations(None if message is None else message.n_receivers)


def _spec_dict(spec: TaskSpec) -> dict[str, Any]:
    data = asdict(spec)  # A MessageSpec becomes a plain dict
    data["params"] = dict(data["params"])
    return data
