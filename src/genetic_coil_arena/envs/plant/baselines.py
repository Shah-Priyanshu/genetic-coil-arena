"""Diagnosis quality of the hand-designed encodings at equal bytes (step 1 baselines).

One *cell* is an (encoding, byte budget, seed) triple. ``PlantContext.evaluate`` fits the
encoder's calibration and the centre's classifier on a seeded 80% block subsample of the
training split (normal training data only for calibration), then replays every test run
through the environment and scores the diagnoses. Seeds therefore vary the training
set, the classifier and the noise control; the held-out test runs are the same recorded
runs for every seed.

Encodings: ``alarms`` (a), ``summary`` (b), ``features`` (c), ``silent`` (d), ``oracle``
(e, ignores the byte budget) and ``noise`` (a random-bytes control).
"""

from typing import Any, Callable, Optional

import numpy as np

from ...evaluation import Task, TaskSpec, register_baseline
from .centres import (
    DEFAULT_TARGET_FAR,
    DecodedCentre,
    OracleCentre,
    SilentCentre,
    block_subsample,
)
from .encoders import FeatureEncoder, make_encoder
from .env import PlantConfig, PlantEnv
from .runner import (
    diagnosis_metrics,
    matched_far_tau,
    rethreshold,
    run_episodes,
)
from .tep import FAULT_IDS, UNOBSERVABLE_FAULTS, TepData

ENCODINGS = ("alarms", "summary", "features", "silent", "oracle", "noise")
DIAGNOSABLE_FAULTS = tuple(f for f in FAULT_IDS if f not in UNOBSERVABLE_FAULTS)
FAULT_SETS = {"diagnosable": DIAGNOSABLE_FAULTS, "all": FAULT_IDS}


class PlantContext:
    """Data, windows and environments for one fault set, shared by many cells.

    Args:
        data: The TEP runs
        faults: Fault ids present (classes are normal plus these)
        window: Samples per node window
        stride: Samples between uplinks at test time
        train_stride: Samples between training windows (denser than ``stride``)
        target_far: Nominal held-out false-alarm rate of the centre's normal margin
        l2: L2 strength of the centre's classifier
        fraction: Share of training blocks each seed keeps
        block: Windows per training block
        classifier: "logreg" or "mlp"
    """

    def __init__(
        self,
        data: TepData,
        faults: tuple[int, ...] = DIAGNOSABLE_FAULTS,
        window: int = 20,
        stride: int = 10,
        train_stride: int = 4,
        target_far: Optional[float] = DEFAULT_TARGET_FAR,
        l2: float = 1e-2,
        fraction: float = 0.8,
        block: int = 10,
        classifier: str = "logreg",
    ):
        self.data = data
        self.faults = tuple(faults)
        self.window = window
        self.stride = stride
        self.target_far = target_far
        self.l2 = l2
        self.fraction = fraction
        self.block = block
        self.classifier = classifier
        self.n_classes = len(self.faults) + 1
        # Training windows overlap their neighbours up to ceil(window / stride) positions
        self.gap = -(-window // train_stride)
        self._envs: dict[int, PlantEnv] = {}
        base = self.env(0)
        self.layout = base.layout
        self.train = base.labelled_windows("train", stride=train_stride)

    def env(self, budget: int) -> PlantEnv:
        """The test-split environment at a byte budget (cached)."""
        if budget not in self._envs:
            cfg = PlantConfig(
                budget=budget,
                window=self.window,
                stride=self.stride,
                split="test",
                faults=self.faults,
            )
            self._envs[budget] = PlantEnv(cfg, self.data)
        return self._envs[budget]

    def fit(self, encoding: str, budget: int, seed: int) -> tuple[PlantEnv, Any, Any]:
        """Train one cell on this seed's training subsample.

        Returns:
            ``(env, encoder, centre)``; the encoder is ``None`` for silent and oracle
            (which run on a 0-byte channel)
        """
        if encoding not in ENCODINGS:
            raise ValueError(f"Unknown encoding {encoding!r}; expected {ENCODINGS}")
        rng = np.random.default_rng(10_000 + seed)
        keep = block_subsample(self.train["run"], self.fraction, rng, self.block)
        windows = self.train["windows"][keep]
        labels = self.train["labels"][keep]
        run = self.train["run"][keep]
        normal = windows[labels == 0]
        params = {"target_far": self.target_far, "gap": self.gap, "l2": self.l2}

        if encoding == "silent":
            return self.env(0), None, SilentCentre()
        if encoding == "oracle":
            features = FeatureEncoder(51, self.layout).fit(normal)
            centre = OracleCentre(
                self.layout.mask,
                self.n_classes,
                self.classifier,
                seed,
                features=features,
                **params,
            ).fit(windows, labels, run)
            return self.env(0), None, centre
        if budget == 0:
            raise ValueError(f"{encoding!r} needs a budget >= 1 byte")
        encoder = make_encoder(encoding, budget, self.layout, seed).fit(normal)
        centre = DecodedCentre(
            encoder, self.n_classes, self.classifier, seed, **params
        ).fit(windows, labels, run)
        return self.env(budget), encoder, centre

    def rollout(
        self, env: PlantEnv, encoder: Any, centre: Any, hook: Optional[Any] = None
    ) -> dict[str, np.ndarray]:
        """Replay every test run once (``run_episodes`` over ``range(env.n_runs)``)."""
        return run_episodes(env, encoder, centre, range(env.n_runs), hook)

    def evaluate(self, encoding: str, budget: int, seed: int) -> dict[str, Any]:
        """Fit on this seed's training subsample and score the test runs.

        Returns:
            The ``diagnosis_metrics`` at the centre's own calibrated threshold, the same
            metrics with a ``_matched`` suffix at a threshold that gives the target false-
            alarm rate on the test normal windows, ``bytes_used`` (mean bytes per node
            message), ``tau`` / ``tau_matched`` (the normal margins) and the cell's identity
        """
        env, encoder, centre = self.fit(encoding, budget, seed)
        res = self.rollout(env, encoder, centre)
        out = diagnosis_metrics(res, self.n_classes)
        # The same decisions at one shared operating point: the normal-margin threshold
        # that gives `target_far` false alarms on the test normal windows. It peeks at the
        # test labels, so it is for comparing encodings, not a deployable setting.
        if hasattr(centre, "tau"):
            tau_matched = matched_far_tau(res, self.target_far or DEFAULT_TARGET_FAR)
            matched = diagnosis_metrics(rethreshold(res, tau_matched), self.n_classes)
        else:
            tau_matched, matched = float("nan"), out
        out.update(
            {f"{k}_matched": v for k, v in matched.items() if k != "f1_per_class"}
        )
        out.update(
            encoding=encoding,
            budget=0 if encoding in ("silent", "oracle") else budget,
            seed=seed,
            bytes_used=float(res["bytes_per_node"]),
            tau=float(getattr(centre, "tau", 0.0)),
            tau_matched=tau_matched,
        )
        return out


# ---------------------------------------------------------------------- harness adapter
def plant_system(ctx: PlantContext, encoding: str, budget: int) -> Callable[..., Any]:
    """A CommArena system: ``system(inputs, seed=seed, hook=hook) -> outputs``.

    Trains each seed's cell once and replays it under any message hook, so the
    harness's ablations reuse the trained centre.
    """
    cache: dict[int, tuple[PlantEnv, Any, Any]] = {}

    def system(inputs: Any, seed: int = 0, hook: Optional[Any] = None) -> Any:
        if seed not in cache:
            cache[seed] = ctx.fit(encoding, budget, seed)
        env, encoder, centre = cache[seed]
        return ctx.rollout(env, encoder, centre, hook)

    return system


def plant_task(ctx: PlantContext, budget: int, n_seeds: int = 10) -> Task:
    """The diagnosis task at a byte budget for the CommArena harness.

    One score per seed: macro-F1 over the fault classes. The task's message channel is
    the uplink, so the default ablations (zero, shuffle_batch, swap_receivers,
    random_matched) act on the nodes' messages. Its mandatory baselines are the silent
    centre and the random-bytes control (also registered with
    ``evaluation.register_baseline`` by ``register_plant_baselines``).
    """
    env = ctx.env(budget)
    spec = TaskSpec(
        name=task_name(ctx, budget),
        description="Tennessee Eastman fault diagnosis from byte-budgeted node uplinks",
        metric="macro_f1",
        n_seeds=n_seeds,
        params={"budget": budget, "faults": list(ctx.faults)},
        message=env.spec.message if budget > 0 else None,
    )
    n_classes = ctx.n_classes

    def checker(inputs: Any, outputs: Any) -> float:
        return float(diagnosis_metrics(outputs, n_classes)["macro_f1"])

    return Task(spec=spec, episodes=lambda seed: seed, checker=checker)


def task_name(ctx: PlantContext, budget: int) -> str:
    """Harness task name for a fault set and budget."""
    return f"plant_diagnosis_{len(ctx.faults)}faults_{budget}B"


def register_plant_baselines(ctx: PlantContext, budget: int) -> None:
    """Register the mandatory baselines of ``plant_task`` with the global registry."""
    name = task_name(ctx, budget)
    register_baseline(name, "silent", plant_system(ctx, "silent", 0))
    register_baseline(name, "random_bytes", plant_system(ctx, "noise", budget))
