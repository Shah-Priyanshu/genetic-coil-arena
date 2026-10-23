"""Run-level replicates of the plant diagnosis experiment on the Rieth et al. data (R8 step 2).

A *replicate* is one complete, independent draw of the experiment: its own training runs
(from the Rieth ``training`` files), validation runs and test runs (from the longer
``testing`` files), all different simulation runs (``rieth.RunSplits``). On it a cell
(encoding, byte budget) is evaluated in four steps:

1. the encoder's calibration (alarm thresholds, PCA, band baselines) is fitted on the
   replicate's normal training windows;
2. the centre (an MLP on the decoded messages, optionally stacking the last few uplinks)
   is trained on the training windows;
3. its normal-vs-fault margin ``tau`` is set on the validation runs so that the target
   share (5%) of validation normal windows raise an alarm; nothing of the validation set
   is trained on, and the test runs are never touched;
4. every test run is replayed through ``PlantEnv`` and scored.

Because replicates share no simulation run, the mean ± t-CI over replicates covers
run-to-run variability of the plant (training draw, calibration draw, test draw), not only
the classifier's initialisation. The realised false-alarm rate of the test runs is
reported next to the target, and the same decisions re-thresholded to exactly the target
on the test normal windows (which peeks at test labels) show how much the calibration
cost.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ...evaluation.stats import mean_ci
from .centres import (
    DEFAULT_TARGET_FAR,
    DecodedCentre,
    OracleCentre,
    SilentCentre,
    _LearnedCentre,
)
from .encoders import FeatureEncoder, UplinkEncoder, make_encoder
from .env import PlantConfig, PlantEnv
from .rieth import (
    N_RUNS,
    RIETH_FAULTS,
    RiethData,
    RunSelection,
    RunSplits,
    to_runs,
)
from .runner import (
    diagnosis_metrics,
    macro_f1,
    matched_far_tau,
    rethreshold,
    run_episodes,
)
from .tep import UNOBSERVABLE_FAULTS, Normaliser
from .windows import labelled_windows, make_layout

ENCODINGS = ("alarms", "summary", "features", "silent", "oracle", "noise")
DIAGNOSABLE_FAULTS = tuple(f for f in RIETH_FAULTS if f not in UNOBSERVABLE_FAULTS)
FAULT_SETS = {"diagnosable": DIAGNOSABLE_FAULTS, "all": RIETH_FAULTS}

# The centre every cell starts from; ``tuning.tune_centre`` moves it on validation runs
DEFAULT_CENTRE: dict[str, Any] = {
    "classifier": "mlp",
    "hidden": 64,
    "l2": 1e-3,
    "max_iter": 100,
    "context": 1,
    "precision": "float32",
}


@dataclass(frozen=True)
class Protocol:
    """Sizes of the experiment (runs per fault unless noted) and its windows.

    Args:
        faults: Fault ids; classes are normal (0) then these in order
        window: Samples per node window (20 = 1 hour)
        stride: Samples between uplinks, and between windows of every split
        n_replicates: Independent replicates (disjoint runs)
        train_runs, val_runs, test_runs: Faulty runs per fault in each split
        train_normal_runs, val_normal_runs, test_normal_runs: Fault-free runs per split
        target_far: Share of validation normal windows allowed to alarm
        split_seed: Seed of the run permutations (``rieth.RunSplits``)
        n_available: Simulation runs per condition in the data (500 for the real data)
    """

    faults: tuple[int, ...] = DIAGNOSABLE_FAULTS
    window: int = 20
    stride: int = 10
    n_replicates: int = 10
    train_runs: int = 20
    train_normal_runs: int = 40
    val_runs: int = 12
    val_normal_runs: int = 20
    test_runs: int = 25
    test_normal_runs: int = 25
    target_far: float = DEFAULT_TARGET_FAR
    split_seed: int = 0
    n_available: int = N_RUNS

    @property
    def n_classes(self) -> int:
        return len(self.faults) + 1

    @property
    def class_of_fault(self) -> dict[int, int]:
        return {f: i + 1 for i, f in enumerate(self.faults)}

    def splits(self) -> RunSplits:
        return RunSplits(
            faults=tuple(self.faults),
            n_replicates=self.n_replicates,
            train_runs=self.train_runs,
            val_runs=self.val_runs,
            test_runs=self.test_runs,
            train_normal_runs=self.train_normal_runs,
            val_normal_runs=self.val_normal_runs,
            test_normal_runs=self.test_normal_runs,
            split_seed=self.split_seed,
            n_available=self.n_available,
        )


@dataclass
class Prepared:
    """One cell's encoder and the features its centre reads, ready to fit centres on.

    Args:
        encoding: Encoding name
        budget: Bytes per node message (0 for silent and oracle)
        seed: Centre initialisation / noise seed
        encoder: The fitted encoder (``None`` for silent and oracle)
        make_centre: ``params -> untrained centre`` (``params`` as ``DEFAULT_CENTRE``)
        x_train, x_val: Decoded features of the training and validation windows
    """

    encoding: str
    budget: int
    seed: int
    encoder: Optional[UplinkEncoder]
    make_centre: Optional[Callable[[dict[str, Any]], _LearnedCentre]]
    x_train: Optional[np.ndarray] = None
    x_val: Optional[np.ndarray] = None
    extra: dict[str, Any] = field(default_factory=dict)


class Replicate:
    """The runs, windows and test environment of one replicate (or of the tuning stage).

    Args:
        data: The converted Rieth et al. runs
        protocol: Sizes and window settings
        index: Replicate number, or ``None`` for the tuning stage, whose validation set
            is the validation and test slices of its block together and which has no
            test environment (hyperparameters are chosen there and never see a
            replicate's test runs)
    """

    def __init__(self, data: RiethData, protocol: Protocol, index: Optional[int]):
        self.data, self.protocol, self.index = data, protocol, index
        sel = protocol.splits().selection(index)
        if index is None:
            sel = RunSelection(
                train_normal=sel.train_normal,
                train_fault=sel.train_fault,
                val_normal=np.concatenate([sel.val_normal, sel.test_normal]),
                val_fault={
                    f: np.concatenate([sel.val_fault[f], sel.test_fault[f]])
                    for f in protocol.faults
                },
                test_normal=sel.test_normal,
                test_fault=sel.test_fault,
            )
        self.selection = sel
        self.layout = make_layout()
        self.normaliser = Normaliser.fit(
            data.normal_runs("training", np.sort(sel.train_normal)).reshape(-1, 52)
        )
        faults, cls = tuple(protocol.faults), protocol.class_of_fault
        w, s = protocol.window, protocol.stride
        self.train = labelled_windows(
            to_runs(data, self.normaliser, sel, "train", faults), cls, self.layout, w, s
        )
        self.val_runs = to_runs(data, self.normaliser, sel, "val", faults)
        self.val = labelled_windows(self.val_runs, cls, self.layout, w, s)
        self._env: Optional[PlantEnv] = None

    # ------------------------------------------------------------------ environments
    def env(self, budget: int) -> PlantEnv:
        """The test-run environment at a byte budget (windows are built once)."""
        if self.index is None:
            raise ValueError("The tuning stage has no test runs")
        if self._env is None:
            p = self.protocol
            cfg = PlantConfig(
                budget=0, window=p.window, stride=p.stride, faults=tuple(p.faults)
            )
            runs = to_runs(
                self.data, self.normaliser, self.selection, "test", tuple(p.faults)
            )
            self._env = PlantEnv(cfg, runs=runs)
        return self._env.with_budget(budget)

    # --------------------------------------------------------------------- cells
    def prepare(
        self, encoding: str, budget: int, seed: int, features: bool = True
    ) -> Prepared:
        """Fit the encoder on normal training windows and extract the centre's features.

        Args:
            encoding: One of ``ENCODINGS``
            budget: Bytes per node message (ignored by silent and oracle)
            seed: Centre initialisation and noise seed
            features: Also extract train and validation features (skip for silent)
        """
        if encoding not in ENCODINGS:
            raise ValueError(f"Unknown encoding {encoding!r}; expected {ENCODINGS}")
        n_classes = self.protocol.n_classes
        normal = self.train["windows"][self.train["labels"] == 0]
        target = self.protocol.target_far
        if encoding == "silent":
            return Prepared("silent", 0, seed, None, None)
        if encoding == "oracle":
            reference = FeatureEncoder(51, self.layout).fit(normal)

            def make_oracle(params: dict[str, Any]) -> _LearnedCentre:
                kind = params["classifier"]
                rest = {k: v for k, v in params.items() if k != "classifier"}
                return OracleCentre(
                    self.layout.mask,
                    n_classes,
                    kind,
                    seed,
                    target_far=target,
                    features=reference,
                    **rest,
                )

            prep = Prepared("oracle", 0, seed, None, make_oracle)
        else:
            if budget < 1:
                raise ValueError(f"{encoding!r} needs a budget >= 1 byte")
            encoder = make_encoder(encoding, budget, self.layout, seed).fit(normal)

            def make_decoded(params: dict[str, Any]) -> _LearnedCentre:
                kind = params["classifier"]
                rest = {k: v for k, v in params.items() if k != "classifier"}
                return DecodedCentre(
                    encoder, n_classes, kind, seed, target_far=target, **rest
                )

            prep = Prepared(encoding, budget, seed, encoder, make_decoded)
        if features:
            assert prep.make_centre is not None
            probe = prep.make_centre(DEFAULT_CENTRE)  # only to extract features
            prep.x_train = probe.extract(self.train["windows"])
            prep.x_val = probe.extract(self.val["windows"])
        return prep

    def fit_centre(self, prep: Prepared, params: dict[str, Any]) -> _LearnedCentre:
        """Train a centre with ``params`` and calibrate it on the validation runs."""
        if prep.make_centre is None or prep.x_train is None or prep.x_val is None:
            raise ValueError("Nothing to fit for this cell")
        centre = prep.make_centre(params)
        cal = {"features": prep.x_val, **{k: self.val[k] for k in ("labels", "run")}}
        return centre.fit_features(
            prep.x_train, self.train["labels"], self.train["run"], cal
        )

    def validation_f1(self, centre: _LearnedCentre, prep: Prepared) -> float:
        """Macro-F1 over the validation windows at the centre's calibrated threshold."""
        assert prep.x_val is not None
        margin, fault = centre.score_features(prep.x_val, self.val["run"])
        decision = np.where(margin > centre.tau, fault, 0)
        return macro_f1(self.val["labels"], decision, self.protocol.n_classes)[0]

    def evaluate(
        self,
        encoding: str,
        budget: int,
        seed: int,
        params: dict[str, Any],
        prep: Optional[Prepared] = None,
    ) -> dict[str, Any]:
        """Train, calibrate and replay the test runs; return the cell's metrics.

        Returns:
            ``diagnosis_metrics`` at the calibrated threshold (``tau``), the same with a
            ``_matched`` suffix after re-thresholding to exactly ``target_far`` on the
            test normal windows, ``val_far`` (the validation false-alarm rate at ``tau``),
            the test false-alarm rate split by source (``far_fault_free`` for fault-free
            runs, ``far_pre_onset`` for the normal start of fault runs), ``bytes_used``,
            ``val_f1`` and the cell's identity
        """
        p = self.protocol
        prep = prep or self.prepare(
            encoding, budget, seed, features=encoding != "silent"
        )
        if encoding == "silent":
            centre: Any = SilentCentre()
            val_far = val_f1 = float("nan")
        else:
            centre = self.fit_centre(prep, params)
            margin, fault = centre.score_features(prep.x_val, self.val["run"])
            normal = self.val["labels"] == 0
            val_far = float(np.mean(margin[normal] > centre.tau))
            val_f1 = self.validation_f1(centre, prep)
        env = self.env(prep.budget)
        res = run_episodes(env, prep.encoder, centre, range(env.n_runs))
        out = diagnosis_metrics(res, p.n_classes)
        if hasattr(centre, "tau"):
            tau_matched = matched_far_tau(res, p.target_far)
            matched = diagnosis_metrics(rethreshold(res, tau_matched), p.n_classes)
        else:
            tau_matched, matched = float("nan"), out
        out.update(
            {f"{k}_matched": v for k, v in matched.items() if k != "f1_per_class"}
        )
        valid, label = res["valid"], res["labels"]
        alarm = res["diagnoses"] != 0
        free = (res["episode_class"] == 0)[:, None]
        for name, mask in (("far_fault_free", free), ("far_pre_onset", ~free)):
            normal = valid & (label == 0) & mask
            out[name] = float(alarm[normal].mean()) if normal.any() else float("nan")
        out.update(
            encoding=encoding,
            budget=int(prep.budget),
            seed=seed,
            replicate=self.index,
            params=dict(params) if encoding != "silent" else {},
            bytes_used=float(res["bytes_per_node"]),
            tau=float(getattr(centre, "tau", 0.0)),
            tau_matched=tau_matched,
            val_far=val_far,
            val_f1=val_f1,
            n_test_runs=int(env.n_runs),
        )
        out["f1_per_class"] = [float(x) for x in out["f1_per_class"]]
        return out


def summarise(values: Sequence[float]) -> tuple[float, float]:
    """Mean and 95% t half-width over replicates, ignoring NaN (``(nan, nan)`` if none)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size < 2:
        return float(arr[0]), float("nan")
    mean, half = mean_ci(arr)
    return float(mean), float(half)
