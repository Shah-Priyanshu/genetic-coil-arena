"""Gateway centres: diagnose the plant state from the nodes' uplink messages.

A centre maps the delivered ``(B, N, L)`` uplink to a ``(B,)`` class (0 normal, else a
fault). The honest centres see only the messages. The silent centre reports normal; the
decoded centre decodes the bytes with its encoder's codec and classifies them with a
classifier trained on the training split. The oracle is the one exception: it reads the
raw windows (``needs_raw``) and so ignores the bandwidth limit, as an upper bound.

Normal operation is rare in the training data (one 500-sample run) and a plain argmax
over balanced classes raises alarms on much of it. So every learned centre has a normal
margin ``tau``: it alarms only when the best fault logit beats the normal logit by more
than ``tau``, and then reports that fault. ``tau`` is set on training data alone, as the
``1 - target_far`` quantile of that margin on held-out normal windows from time-blocked
cross-validation (neighbouring windows are dropped so no window overlaps its own
held-out fold). All encoders therefore share one nominal false-alarm target.
"""

from typing import Any, Optional

import numpy as np

from .classifiers import make_classifier
from .encoders import FeatureEncoder, UplinkEncoder, window_stats

DEFAULT_TARGET_FAR = 0.05


def blocked_folds(
    run: np.ndarray, n_folds: int = 4, gap: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-blocked cross-validation folds over windows ordered in time within each run.

    Each run is cut into ``n_folds`` contiguous parts; fold ``f`` holds out part ``f`` of
    every run and trains on the rest, minus ``gap`` windows either side of the held-out
    part (so overlapping windows never straddle the split).

    Args:
        run: (n,) run index of each window; windows of a run are consecutive and in time
            order
        n_folds: Number of folds
        gap: Windows dropped on each side of a held-out part
    """
    run = np.asarray(run)
    starts = {r: np.flatnonzero(run == r) for r in np.unique(run)}
    folds = []
    for f in range(n_folds):
        test: list[np.ndarray] = []
        train: list[np.ndarray] = []
        for idx in starts.values():
            m = len(idx)
            pos = np.arange(m)
            part = (pos * n_folds) // m
            held = part == f
            if not held.any():
                train.append(idx)
                continue
            lo, hi = pos[held].min(), pos[held].max()
            keep = (pos < lo - gap) | (pos > hi + gap)
            test.append(idx[held])
            train.append(idx[keep])
        folds.append((np.concatenate(train), np.concatenate(test)))
    return folds


def block_subsample(
    run: np.ndarray, fraction: float, rng: np.random.Generator, block: int = 10
) -> np.ndarray:
    """Indices of a random ``fraction`` of the contiguous blocks of every run, sorted.

    Used to give each seed a different training set while keeping windows in blocks (a
    window and its neighbours stay together).
    """
    run = np.asarray(run)
    picked: list[np.ndarray] = []
    for r in np.unique(run):
        idx = np.flatnonzero(run == r)
        blocks = [idx[i : i + block] for i in range(0, len(idx), block)]
        k = max(1, int(round(fraction * len(blocks))))
        chosen = rng.choice(len(blocks), size=k, replace=False)
        picked.extend(blocks[j] for j in sorted(chosen))
    return np.concatenate(picked)


class SilentCentre:
    """No telemetry, no information: always reports normal operation."""

    needs_raw = False
    name = "silent"

    def diagnose(
        self, messages: np.ndarray, raw: Optional[dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        return np.zeros(messages.shape[0], dtype=np.int64)


class _LearnedCentre:
    """Classifier on features with a calibrated normal margin (see the module docstring).

    Args:
        n_classes: Normal plus faults
        classifier: "logreg" or "mlp"
        seed: Classifier initialisation seed
        target_far: Held-out normal false-alarm rate that sets ``tau``; ``None`` skips
            calibration (``tau = 0`` is the plain argmax)
        folds, gap: Calibration cross-validation (see ``blocked_folds``)
        params: Classifier hyperparameters (``l2``, ``normal_weight``, ...)
    """

    needs_raw = False
    name = "learned"

    def __init__(
        self,
        n_classes: int,
        classifier: str = "logreg",
        seed: int = 0,
        target_far: Optional[float] = DEFAULT_TARGET_FAR,
        folds: int = 4,
        gap: int = 5,
        **params: Any,
    ):
        self.n_classes = n_classes
        self.kind = classifier
        self.seed = seed
        self.target_far = target_far
        self.folds = folds
        self.gap = gap
        self.params = params
        self.tau = 0.0
        self.last_margin: Optional[np.ndarray] = None  # (B,) of the last diagnose call
        self.last_fault: Optional[np.ndarray] = None  # (B,) best fault class there
        self.classifier = make_classifier(classifier, n_classes, seed, **params)

    def _from_windows(self, windows: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _from_messages(
        self, messages: np.ndarray, raw: Optional[dict[str, np.ndarray]]
    ) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _margin(logits: np.ndarray) -> np.ndarray:
        return logits[:, 1:].max(axis=1) - logits[:, 0]

    def fit(
        self, windows: np.ndarray, labels: np.ndarray, run: Optional[np.ndarray] = None
    ) -> "_LearnedCentre":
        """Train on labelled windows, in time order within each run.

        Args:
            windows: (n, N, Vmax, W) training windows
            labels: (n,) classes
            run: (n,) run index of each window; needed to calibrate ``tau``
        """
        x = self._from_windows(windows)
        y = np.asarray(labels, dtype=np.int64)
        self.tau = 0.0
        if self.target_far is not None:
            if run is None:
                raise ValueError("Calibrating the normal margin needs the run indices")
            margins = []
            for train, test in blocked_folds(run, self.folds, self.gap):
                held = test[y[test] == 0]
                model = make_classifier(
                    self.kind, self.n_classes, self.seed, **self.params
                ).fit(x[train], y[train])
                margins.append(self._margin(model.decision_function(x[held])))
            self.tau = float(np.quantile(np.concatenate(margins), 1 - self.target_far))
        self.classifier = make_classifier(
            self.kind, self.n_classes, self.seed, **self.params
        ).fit(x, y)
        return self

    def _decide(self, features: np.ndarray) -> np.ndarray:
        logits = self.classifier.decision_function(features)
        fault = logits[:, 1:].argmax(axis=1) + 1
        margin = self._margin(logits)
        # Kept so a rollout can re-threshold the same decisions (see runner.rethreshold)
        self.last_margin, self.last_fault = margin, fault.astype(np.int64)
        return np.where(margin > self.tau, fault, 0).astype(np.int64)

    def diagnose(
        self, messages: np.ndarray, raw: Optional[dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        return self._decide(self._from_messages(messages, raw))


class DecodedCentre(_LearnedCentre):
    """Decodes the uplink with the encoder's codec, then classifies the features.

    Args:
        encoder: The (fitted) encoder whose messages this centre reads
        n_classes: Normal plus faults
        (other arguments as ``_LearnedCentre``)
    """

    def __init__(
        self, encoder: UplinkEncoder, n_classes: int, *args: Any, **kwargs: Any
    ):
        super().__init__(n_classes, *args, **kwargs)
        self.encoder = encoder
        self.name = encoder.name

    def _from_windows(self, windows: np.ndarray) -> np.ndarray:
        # Trained on exactly what the centre would receive: encoded, then decoded
        return self.encoder.decode(self.encoder.encode_windows(windows))

    def _from_messages(
        self, messages: np.ndarray, raw: Optional[dict[str, np.ndarray]]
    ) -> np.ndarray:
        return self.encoder.decode(messages)


def raw_features(windows: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean, std, max, min and slope of every real variable of every node, unquantised."""
    stats = window_stats(windows)
    cols = [stats[k][:, mask] for k in ("mean", "std", "max", "min", "slope")]
    return np.concatenate(cols, axis=1).astype(np.float32)


class OracleCentre(_LearnedCentre):
    """Upper bound: classifies full-precision features of every raw variable.

    It reads the raw windows of all 52 variables and so ignores the bandwidth limit. Its
    features are the unquantised window statistics of every variable and, if
    ``features`` is given (a fitted ``FeatureEncoder``), also every PCA and band-energy
    feature that encoder can send, so no hand-designed encoding carries information the
    oracle lacks.
    """

    needs_raw = True
    name = "oracle"

    def __init__(
        self,
        mask: np.ndarray,
        n_classes: int,
        *args: Any,
        features: Optional[FeatureEncoder] = None,
        **kwargs: Any,
    ):
        super().__init__(n_classes, *args, **kwargs)
        self.mask = np.asarray(mask, dtype=bool)
        self.features = features

    def _features(self, windows: np.ndarray) -> np.ndarray:
        out = raw_features(windows, self.mask)
        if self.features is not None:
            out = np.concatenate([out, self.features.unquantised(windows)], axis=1)
        return out

    def _from_windows(self, windows: np.ndarray) -> np.ndarray:
        return self._features(windows)

    def _from_messages(
        self, messages: np.ndarray, raw: Optional[dict[str, np.ndarray]]
    ) -> np.ndarray:
        if raw is None:
            raise ValueError("The oracle needs the raw specialist observations")
        return self._features(raw["window"])
