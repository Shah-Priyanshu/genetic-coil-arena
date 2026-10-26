"""A small numpy classifier: softmax regression (``hidden=0``) or a one-hidden-layer tanh MLP.

Full-batch Adam on class-balanced cross-entropy with L2 on the weights; inputs are
standardised with the training mean and standard deviation. ``n_models`` independently
initialised networks are averaged (an ensemble), which matters on a few hundred rows.
Everything is plain numpy, so the fitted state is a dict of arrays that round-trips
through JSON (:meth:`MLPClassifier.to_dict`).
"""

from typing import Any, Optional

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLPClassifier:
    """Class-balanced softmax regression or one-hidden-layer MLP, trained with Adam.

    Args:
        n_classes: Number of classes (labels are ``0 .. n_classes - 1``)
        hidden: Hidden units; 0 gives multinomial logistic regression
        l2: L2 penalty on the weights (not the biases)
        epochs: Full-batch Adam steps
        lr: Adam learning rate
        n_models: Ensemble size (probabilities are averaged)
        seed: Seeds the initialisation
        balanced: Weight every class equally (rows weighted by ``1 / class count``)
    """

    def __init__(
        self,
        n_classes: int,
        hidden: int = 0,
        l2: float = 1e-2,
        epochs: int = 300,
        lr: float = 0.02,
        n_models: int = 1,
        seed: int = 0,
        balanced: bool = True,
    ) -> None:
        if n_classes < 2:
            raise ValueError("need at least two classes")
        self.n_classes = n_classes
        self.hidden = hidden
        self.l2 = l2
        self.epochs = epochs
        self.lr = lr
        self.n_models = n_models
        self.seed = seed
        self.balanced = balanced
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.nets: list[list[np.ndarray]] = []

    # ------------------------------------------------------------------ fitting

    def _init(self, d: int, rng: np.random.Generator) -> list[np.ndarray]:
        k, h = self.n_classes, self.hidden
        if h == 0:
            return [np.zeros((d, k)), np.zeros(k)]
        return [
            rng.normal(0, 1 / np.sqrt(d), (d, h)),
            np.zeros(h),
            rng.normal(0, 1 / np.sqrt(h), (h, k)),
            np.zeros(k),
        ]

    def _forward(self, net: list[np.ndarray], x: np.ndarray) -> tuple[Any, np.ndarray]:
        if self.hidden == 0:
            return None, _softmax(x @ net[0] + net[1])
        a = np.tanh(x @ net[0] + net[1])
        return a, _softmax(a @ net[2] + net[3])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if x.ndim != 2 or len(x) != len(y) or len(y) == 0:
            raise ValueError("x must be (N, d) with N labels, N > 0")
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError("labels outside 0 .. n_classes - 1")
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-6
        z = (x - self.mean) / self.std
        n, d = z.shape
        onehot = np.eye(self.n_classes)[y]
        w = np.ones(n)
        if self.balanced:
            counts = np.bincount(y, minlength=self.n_classes).astype(float)
            w = (n / (np.count_nonzero(counts) * np.maximum(counts, 1)))[y]
        w = w / w.sum()
        self.nets = []
        for m in range(self.n_models):
            rng = np.random.default_rng(self.seed * 1000 + m)
            net = self._init(d, rng)
            mom = [np.zeros_like(p) for p in net]
            vel = [np.zeros_like(p) for p in net]
            for t in range(1, self.epochs + 1):
                act, p = self._forward(net, z)
                err = (p - onehot) * w[:, None]
                if self.hidden == 0:
                    grads = [z.T @ err + self.l2 * net[0], err.sum(axis=0)]
                else:
                    back = (err @ net[2].T) * (1 - act**2)
                    grads = [
                        z.T @ back + self.l2 * net[0],
                        back.sum(axis=0),
                        act.T @ err + self.l2 * net[2],
                        err.sum(axis=0),
                    ]
                for i, g in enumerate(grads):
                    mom[i] = 0.9 * mom[i] + 0.1 * g
                    vel[i] = 0.999 * vel[i] + 0.001 * g * g
                    mh = mom[i] / (1 - 0.9**t)
                    vh = vel[i] / (1 - 0.999**t)
                    net[i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
            self.nets.append(net)
        return self

    # --------------------------------------------------------------- prediction

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None or not self.nets:
            raise RuntimeError("fit the classifier first")
        z = (np.atleast_2d(np.asarray(x, dtype=np.float64)) - self.mean) / self.std
        return np.mean([self._forward(net, z)[1] for net in self.nets], axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    # -------------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        if self.mean is None or self.std is None:
            raise RuntimeError("fit the classifier first")
        return {
            "n_classes": self.n_classes,
            "hidden": self.hidden,
            "l2": self.l2,
            "epochs": self.epochs,
            "lr": self.lr,
            "n_models": self.n_models,
            "seed": self.seed,
            "balanced": self.balanced,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "nets": [[p.tolist() for p in net] for net in self.nets],
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "MLPClassifier":
        out = cls(
            state["n_classes"],
            hidden=state["hidden"],
            l2=state["l2"],
            epochs=state["epochs"],
            lr=state["lr"],
            n_models=state["n_models"],
            seed=state["seed"],
            balanced=state["balanced"],
        )
        out.mean = np.asarray(state["mean"])
        out.std = np.asarray(state["std"])
        out.nets = [[np.asarray(p) for p in net] for net in state["nets"]]
        return out
