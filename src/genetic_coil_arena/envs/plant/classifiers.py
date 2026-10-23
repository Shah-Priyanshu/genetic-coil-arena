"""Small numpy classifiers for the centre: softmax regression and a one-hidden-layer MLP.

Both standardise their inputs with the training statistics, weight classes equally
(balanced), regularise with L2 and fit with L-BFGS (``scipy.optimize``), so a fit is
deterministic given its seed. They have no dependency beyond numpy and scipy.
"""

from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class _Classifier:
    """Shared standardisation, class weighting and prediction plumbing."""

    def __init__(
        self,
        n_classes: int,
        l2: float,
        max_iter: int,
        seed: int,
        normal_weight: float = 1.0,
    ):
        self.n_classes = n_classes
        self.normal_weight = normal_weight
        self.l2 = l2
        self.max_iter = max_iter
        self.seed = seed
        self._mean: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._theta: Optional[np.ndarray] = None

    # Subclasses implement the model on standardised inputs
    def _size(self, n_features: int) -> int:
        raise NotImplementedError

    def _init(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def _loss_grad(
        self, theta: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> tuple[float, np.ndarray]:
        raise NotImplementedError

    def _logits(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "_Classifier":
        """Fit on (n, F) features and (n,) integer labels in [0, n_classes)."""
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        if x.ndim != 2 or y.shape != (x.shape[0],):
            raise ValueError("features must be (n, F) and labels (n,)")
        self._mean = x.mean(axis=0) if x.shape[1] else np.zeros(0)
        std = x.std(axis=0) if x.shape[1] else np.zeros(0)
        self._scale = np.where(std > 1e-8, std, 1.0)
        z = (x - self._mean) / self._scale
        counts = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        class_w = np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
        class_w[0] *= self.normal_weight
        w = class_w[y]
        w = w / w.sum()
        rng = np.random.default_rng(self.seed)
        theta0 = self._init(z.shape[1], rng)
        result = minimize(
            self._loss_grad,
            theta0,
            args=(z, y, w),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )
        self._theta = result.x
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """(n, n_classes) logits."""
        if self._theta is None:
            raise RuntimeError("fit() first")
        x = np.asarray(features, dtype=np.float64)
        z = (x - self._mean) / self._scale
        return self._logits(self._theta, z)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """(n, n_classes) class probabilities."""
        return _softmax(self.decision_function(features))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """(n,) most probable class."""
        return self.predict_proba(features).argmax(axis=1).astype(np.int64)


class SoftmaxRegression(_Classifier):
    """Multinomial logistic regression, balanced and L2-regularised."""

    def __init__(
        self,
        n_classes: int,
        l2: float = 1e-3,
        max_iter: int = 300,
        seed: int = 0,
        normal_weight: float = 1.0,
    ):
        super().__init__(n_classes, l2, max_iter, seed, normal_weight)

    def _init(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        return np.zeros((n_features + 1) * self.n_classes)

    def _unpack(self, theta: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray]:
        k = self.n_classes
        return theta[: f * k].reshape(f, k), theta[f * k :]

    def _logits(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        w, b = self._unpack(theta, x.shape[1])
        return x @ w + b

    def _loss_grad(
        self, theta: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> tuple[float, np.ndarray]:
        f = x.shape[1]
        weights, bias = self._unpack(theta, f)
        p = _softmax(x @ weights + bias)
        n = x.shape[0]
        loss = -np.sum(w * np.log(p[np.arange(n), y] + 1e-12))
        loss += 0.5 * self.l2 * np.sum(weights**2)
        d = p
        d[np.arange(n), y] -= 1.0
        d *= w[:, None]
        gw = x.T @ d + self.l2 * weights
        gb = d.sum(axis=0)
        return float(loss), np.concatenate([gw.ravel(), gb])


class MLPClassifier(_Classifier):
    """One tanh hidden layer, balanced and L2-regularised."""

    def __init__(
        self,
        n_classes: int,
        hidden: int = 32,
        l2: float = 1e-3,
        max_iter: int = 200,
        seed: int = 0,
        normal_weight: float = 1.0,
    ):
        super().__init__(n_classes, l2, max_iter, seed, normal_weight)
        self.hidden = hidden

    def _init(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        h, k = self.hidden, self.n_classes
        w1 = rng.normal(0.0, 1.0 / np.sqrt(max(n_features, 1)), size=(n_features, h))
        w2 = rng.normal(0.0, 1.0 / np.sqrt(h), size=(h, k))
        return np.concatenate([w1.ravel(), np.zeros(h), w2.ravel(), np.zeros(k)])

    def _unpack(self, theta: np.ndarray, f: int) -> tuple[np.ndarray, ...]:
        h, k = self.hidden, self.n_classes
        i = 0
        w1 = theta[i : i + f * h].reshape(f, h)
        i += f * h
        b1 = theta[i : i + h]
        i += h
        w2 = theta[i : i + h * k].reshape(h, k)
        i += h * k
        return w1, b1, w2, theta[i : i + k]

    def _logits(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        w1, b1, w2, b2 = self._unpack(theta, x.shape[1])
        return np.tanh(x @ w1 + b1) @ w2 + b2

    def _loss_grad(
        self, theta: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> tuple[float, np.ndarray]:
        f = x.shape[1]
        w1, b1, w2, b2 = self._unpack(theta, f)
        a = np.tanh(x @ w1 + b1)
        p = _softmax(a @ w2 + b2)
        n = x.shape[0]
        loss = -np.sum(w * np.log(p[np.arange(n), y] + 1e-12))
        loss += 0.5 * self.l2 * (np.sum(w1**2) + np.sum(w2**2))
        d = p
        d[np.arange(n), y] -= 1.0
        d *= w[:, None]
        gw2 = a.T @ d + self.l2 * w2
        gb2 = d.sum(axis=0)
        da = (d @ w2.T) * (1.0 - a**2)
        gw1 = x.T @ da + self.l2 * w1
        gb1 = da.sum(axis=0)
        return float(loss), np.concatenate([gw1.ravel(), gb1, gw2.ravel(), gb2])


def make_classifier(
    kind: str, n_classes: int, seed: int = 0, **params: Any
) -> _Classifier:
    """``"logreg"`` (softmax regression) or ``"mlp"`` (one hidden layer).

    ``params`` are passed on (``l2``, ``normal_weight``, ...).
    """
    if kind == "logreg":
        return SoftmaxRegression(n_classes, seed=seed, **params)
    if kind == "mlp":
        return MLPClassifier(n_classes, seed=seed, **params)
    raise ValueError(f"Unknown classifier {kind!r}; expected 'logreg' or 'mlp'")
