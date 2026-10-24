"""Learned node encoders at a fixed byte budget (R8, step 3).

Each encoder maps a node's local window to exactly ``budget`` bytes (``8 * budget`` bits)
and back to the float features the centre classifies, so it drops into the same
``DecodedCentre`` as the hand-designed encoders of ``encoders.py``. The three methods:

- ``e2e``: every node has a small MLP whose ``8 * budget`` outputs pass a binary
  bottleneck (straight-through sign, or a binary-concrete relaxation with logistic noise)
  and are packed into bytes. The nodes' bits feed a classifier head and the whole thing is
  trained end to end with a class-balanced cross-entropy (torch, CPU). The head is dropped
  afterwards; the deployed centre is the same calibrated MLP as for every other method.
  After training, encoding is plain numpy (no torch needed to run an encoder).
- ``vq``: unsupervised residual vector quantisation. Per node, PCA of the local features and
  ``budget`` stages of 256-centroid k-means on the residual; each byte is one stage's
  centroid index. Labels are never used.
- ``local``: each node trains its own fault classifier on local features and sends its top
  ``budget`` classes, one byte each: 5 bits of class id and 3 bits of quantised posterior.

The local features are the same for all methods: per variable the window mean, std, slope,
min, max and the log energy of three FFT bands, standardised on the training windows. Torch
is imported only inside ``train_end_to_end``.
"""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from .budget import VOCAB_SIZE as VOCAB
from .classifiers import make_classifier
from .encoders import N_BANDS, UplinkEncoder, band_energies, window_stats
from .windows import NodeLayout

METHODS = ("e2e", "vq", "local")
N_STATS = 5 + N_BANDS  # mean, std, slope, min, max + band energies, per variable
CLIP = 8.0
LEVEL_EDGES = np.array([0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95])
LEVEL_VALUES = np.array([0.02, 0.10, 0.22, 0.40, 0.60, 0.78, 0.90, 0.98])
MAX_CLASSES = 32  # 5 bits of class id in the local-classifier code


# ------------------------------------------------------------------------------ features
def node_features(windows: np.ndarray, layout: NodeLayout) -> np.ndarray:
    """(B, N, Vmax, W) windows -> (B, N, 8 * Vmax) local features, zero on padding."""
    s = window_stats(windows)
    energy = np.log10(band_energies(windows) + 1e-9)  # (B, N, V, bands)
    parts = [s["mean"], s["std"], s["slope"], s["min"], s["max"]]
    parts += [energy[..., k] for k in range(N_BANDS)]
    feats = np.concatenate(parts, axis=-1)
    return (feats * np.tile(layout.mask, (1, N_STATS))[None]).astype(np.float32)


class Featuriser:
    """Local features, standardised per node on the windows it was fitted on."""

    def __init__(self, layout: NodeLayout):
        self.layout = layout
        self.mask = np.tile(layout.mask, (1, N_STATS))  # (N, 8 * Vmax)
        self.mean = np.zeros(self.mask.shape, np.float32)
        self.scale = np.ones(self.mask.shape, np.float32)

    def fit(self, windows: np.ndarray) -> "Featuriser":
        f = node_features(windows, self.layout)
        self.mean = f.mean(axis=0)
        self.scale = np.where(self.mask, np.maximum(f.std(axis=0), 1e-6), 1.0)
        return self

    def standardise(self, feats: np.ndarray) -> np.ndarray:
        z = (feats - self.mean[None]) / self.scale[None]
        return (np.clip(z, -CLIP, CLIP) * self.mask[None]).astype(np.float32)

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        return self.standardise(node_features(windows, self.layout))


# ---------------------------------------------------------------------------- bit packing
def pack_bits(bits: np.ndarray) -> np.ndarray:
    """(..., 8 * L) booleans -> (..., L) int64 bytes, least significant bit first."""
    b = np.asarray(bits, dtype=np.int64)
    if b.shape[-1] % 8:
        raise ValueError("The number of bits must be a multiple of 8")
    b = b.reshape(*b.shape[:-1], -1, 8)
    return (b * (1 << np.arange(8))).sum(axis=-1)


def unpack_bits(messages: np.ndarray) -> np.ndarray:
    """(..., L) bytes -> (..., 8 * L) float32 in {-1, +1}."""
    m = np.asarray(messages, dtype=np.int64)[..., None]
    bits = (m >> np.arange(8)) & 1
    return (2.0 * bits - 1.0).reshape(*m.shape[:-2], -1).astype(np.float32)


# ------------------------------------------------------------------------------- encoders
class _LearnedEncoder(UplinkEncoder):
    """Common plumbing: featurise, encode features, expose the byte accounting."""

    def __init__(self, budget: int, layout: NodeLayout, featuriser: Featuriser):
        if budget < 1:
            raise ValueError("A learned code needs a budget of at least 1 byte")
        super().__init__(budget, layout)
        self.featuriser = featuriser

    def encode_windows(self, windows: np.ndarray) -> np.ndarray:
        return self.encode_features(self.featuriser(windows))

    def encode_features(self, feats: np.ndarray) -> np.ndarray:
        """(B, N, F) standardised features -> (B, N, budget) int64 bytes in [0, 256)."""
        raise NotImplementedError

    @property
    def bytes_used(self) -> tuple[int, ...]:
        return (self.budget,) * self.layout.n_nodes


class EndToEndEncoder(_LearnedEncoder):
    """Per-node MLP -> ``8 * budget`` sign bits, trained jointly with a classifier head."""

    name = "e2e"

    def __init__(
        self,
        budget: int,
        layout: NodeLayout,
        featuriser: Featuriser,
        weights: Optional[dict[str, np.ndarray]] = None,
    ):
        super().__init__(budget, layout, featuriser)
        n, f = layout.n_nodes, featuriser.mask.shape[1]
        self.weights = weights or {
            "w1": np.zeros((n, f, 1), np.float32),
            "b1": np.zeros((n, 1), np.float32),
            "w2": np.zeros((n, 1, 8 * budget), np.float32),
            "b2": np.zeros((n, 8 * budget), np.float32),
        }

    def encode_features(self, feats: np.ndarray) -> np.ndarray:
        w = self.weights
        h = np.tanh(np.einsum("bnf,nfh->bnh", feats, w["w1"]) + w["b1"][None])
        z = np.einsum("bnh,nhk->bnk", h, w["w2"]) + w["b2"][None]
        return pack_bits(z > 0)

    def decode(self, messages: np.ndarray) -> np.ndarray:
        return unpack_bits(messages).reshape(messages.shape[0], -1)

    @property
    def n_features(self) -> int:
        return self.layout.n_nodes * 8 * self.budget


class ResidualVQEncoder(_LearnedEncoder):
    """Unsupervised residual vector quantisation of a PCA projection of the local features."""

    name = "vq"

    def __init__(
        self,
        budget: int,
        layout: NodeLayout,
        featuriser: Featuriser,
        dim: int = 16,
        whiten: bool = False,
    ):
        super().__init__(budget, layout, featuriser)
        self.dim, self.whiten = int(dim), bool(whiten)
        n, f = layout.n_nodes, featuriser.mask.shape[1]
        self.mean = np.zeros((n, f), np.float32)
        self.components = np.zeros((n, f, self.dim), np.float32)
        self.sigma = np.ones((n, self.dim), np.float32)
        self.codebooks = np.zeros((n, budget, VOCAB, self.dim), np.float32)

    def fit(self, feats: np.ndarray, seed: int = 0) -> "ResidualVQEncoder":  # type: ignore[override]
        """Fit on (n, N, F) standardised features of any windows (labels are not used)."""
        rng = np.random.default_rng(seed)
        for i in range(self.layout.n_nodes):
            x = feats[:, i, :].astype(np.float64)
            self.mean[i] = x.mean(axis=0)
            xc = x - self.mean[i]
            vals, vecs = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
            order = np.argsort(vals)[::-1][: self.dim]
            self.components[i, :, : len(order)] = vecs[:, order]
            self.sigma[i, : len(order)] = np.sqrt(np.maximum(vals[order], 1e-8))
            res = self._project(xc.astype(np.float32), i)
            for s in range(self.budget):
                centres = kmeans(res, VOCAB, rng)
                self.codebooks[i, s] = centres
                res = res - centres[nearest(res, centres)]
        return self

    def _project(self, centred: np.ndarray, node: int) -> np.ndarray:
        z = centred @ self.components[node]
        return z / self.sigma[node] if self.whiten else z

    def encode_features(self, feats: np.ndarray) -> np.ndarray:
        b = feats.shape[0]
        out = np.zeros((b, self.layout.n_nodes, self.budget), np.int64)
        for i in range(self.layout.n_nodes):
            res = self._project(feats[:, i, :] - self.mean[i], i)
            for s in range(self.budget):
                code = nearest(res, self.codebooks[i, s])
                out[:, i, s] = code
                res = res - self.codebooks[i, s][code]
        return out

    def decode(self, messages: np.ndarray) -> np.ndarray:
        b = messages.shape[0]
        out = np.zeros((b, self.layout.n_nodes, self.dim), np.float32)
        for i in range(self.layout.n_nodes):
            for s in range(self.budget):
                out[:, i, :] += self.codebooks[i, s][messages[:, i, s]]
        return out.reshape(b, -1)

    @property
    def n_features(self) -> int:
        return self.layout.n_nodes * self.dim


class LocalClassifierEncoder(_LearnedEncoder):
    """Each node sends the quantised top posteriors of its own fault classifier.

    A byte is ``class_id << 3 | level`` (5 bits of class, 3 bits of posterior). The ``budget``
    bytes list the node's most probable classes, best first; with more bytes than classes
    the list repeats.
    """

    name = "local"

    def __init__(
        self,
        budget: int,
        layout: NodeLayout,
        featuriser: Featuriser,
        n_classes: int,
        hidden: int = 64,
        l2: float = 1e-2,
        max_iter: int = 100,
    ):
        super().__init__(budget, layout, featuriser)
        if n_classes > MAX_CLASSES:
            raise ValueError(f"At most {MAX_CLASSES} classes fit in 5 bits")
        self.n_classes = n_classes
        self.params = {"hidden": hidden, "l2": l2, "max_iter": max_iter}
        self.models: list[Any] = []

    def fit(  # type: ignore[override]
        self, feats: np.ndarray, labels: np.ndarray, seed: int = 0
    ) -> "LocalClassifierEncoder":
        self.models = []
        for i in range(self.layout.n_nodes):
            cols = self.featuriser.mask[i]
            model = make_classifier(
                "mlp", self.n_classes, seed + i, precision="float32", **self.params
            )
            self.models.append(model.fit(feats[:, i, cols], labels))
        return self

    def encode_features(self, feats: np.ndarray) -> np.ndarray:
        b = feats.shape[0]
        out = np.zeros((b, self.layout.n_nodes, self.budget), np.int64)
        take = np.arange(self.budget) % self.n_classes
        rows = np.arange(b)[:, None]
        for i, model in enumerate(self.models):
            p = model.predict_proba(feats[:, i, self.featuriser.mask[i]])
            top = np.argsort(-p, axis=1, kind="stable")[:, take]
            level = np.searchsorted(LEVEL_EDGES, p[rows, top])
            out[:, i, :] = (top << 3) | level
        return out

    def decode(self, messages: np.ndarray) -> np.ndarray:
        b, n, _ = messages.shape
        post = np.zeros((b, n, self.n_classes), np.float32)
        cls = np.minimum(messages >> 3, self.n_classes - 1)
        val = LEVEL_VALUES[messages & 7].astype(np.float32)
        for k in range(messages.shape[2]):
            np.maximum.at(
                post,
                (np.arange(b)[:, None], np.arange(n)[None, :], cls[:, :, k]),
                val[:, :, k],
            )
        return post.reshape(b, -1)

    @property
    def n_features(self) -> int:
        return self.layout.n_nodes * self.n_classes

    def crossfit_decoded(
        self,
        feats: np.ndarray,
        labels: np.ndarray,
        run: np.ndarray,
        folds: int = 3,
        seed: int = 0,
    ) -> np.ndarray:
        """Decoded messages of ``feats`` from classifiers that never saw their own run.

        A classifier is over-confident on the windows it was trained on, which would teach
        the centre to trust the code more than it can on new runs. Runs are split into
        ``folds`` groups; each group is encoded by classifiers fitted on the others.
        """
        rng = np.random.default_rng(seed)
        runs = np.unique(run)
        group = dict(zip(rng.permutation(runs), np.arange(len(runs)) % folds))
        fold = np.array([group[r] for r in run])
        out = np.zeros((len(feats), self.n_features), np.float32)
        for f in range(folds):
            held = fold == f
            if not held.any() or held.all():
                continue
            twin = LocalClassifierEncoder(
                self.budget,
                self.layout,
                self.featuriser,
                self.n_classes,
                **self.params,
            ).fit(feats[~held], labels[~held], seed + 10 * (f + 1))
            out[held] = twin.decode(twin.encode_features(feats[held]))
        return out


# ------------------------------------------------------------------------------- k-means
def nearest(x: np.ndarray, centres: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Index of the closest centre for every row of ``x``."""
    c2 = (centres.astype(np.float64) ** 2).sum(axis=1)
    out = np.empty(len(x), np.int64)
    for a in range(0, len(x), chunk):
        part = x[a : a + chunk].astype(np.float64)
        out[a : a + chunk] = (
            c2[None] - 2.0 * part @ centres.T.astype(np.float64)
        ).argmin(axis=1)
    return out


def kmeans(
    x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 20
) -> np.ndarray:
    """Lloyd's k-means from random data points; empty clusters restart at random points."""
    n = len(x)
    centres = x[rng.choice(n, size=k, replace=n < k)].astype(np.float64)
    for _ in range(iters):
        idx = nearest(x, centres)
        counts = np.bincount(idx, minlength=k)
        sums = np.zeros_like(centres)
        np.add.at(sums, idx, x)
        full = counts > 0
        centres[full] = sums[full] / counts[full, None]
        if (~full).any():
            centres[~full] = x[rng.choice(n, size=int((~full).sum()))]
    return centres.astype(np.float32)


# ---------------------------------------------------------------------------- end to end
def train_end_to_end(
    feats: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    budget: int,
    seed: int = 0,
    hidden: int = 64,
    head: int = 128,
    epochs: int = 20,
    lr: float = 2e-3,
    batch: int = 256,
    weight_decay: float = 1e-4,
    bottleneck: str = "ste",
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Train the node encoders and a classifier head jointly (torch, CPU).

    Args:
        feats: (n, N, F) standardised local features of the training windows
        labels: (n,) classes
        n_classes: Normal plus faults
        budget: Bytes per node, so ``8 * budget`` bits
        seed: Initialisation, shuffling and noise seed
        hidden, head: Width of a node encoder and of the head
        epochs, lr, batch, weight_decay: Adam settings
        bottleneck: "ste" (sign forward, tanh gradient) or "gumbel" (logistic noise on
            the logits and an annealed temperature while training; hard bits forward)

    Returns:
        ``(weights, info)``: numpy weights ``w1, b1, w2, b2`` for ``EndToEndEncoder`` and
        the final training loss and accuracy
    """
    import torch

    if bottleneck not in ("ste", "gumbel"):
        raise ValueError("bottleneck must be 'ste' or 'gumbel'")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    n, nodes, f = feats.shape
    k = 8 * budget
    x = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    counts = torch.bincount(y, minlength=n_classes).float().clamp(min=1)
    cw = (n / (n_classes * counts)).float()

    def init(*shape: int, fan_in: int) -> Any:
        return torch.nn.Parameter(torch.randn(*shape, generator=gen) / np.sqrt(fan_in))

    w1, b1 = (
        init(nodes, f, hidden, fan_in=f),
        torch.nn.Parameter(torch.zeros(nodes, hidden)),
    )
    w2, b2 = (
        init(nodes, hidden, k, fan_in=hidden),
        torch.nn.Parameter(torch.zeros(nodes, k)),
    )
    head_net = torch.nn.Sequential(
        torch.nn.Linear(nodes * k, head),
        torch.nn.Tanh(),
        torch.nn.Linear(head, n_classes),
    )
    params = [w1, b1, w2, b2, *head_net.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def forward(xb: Any, tau: float, noisy: bool) -> Any:
        h = torch.tanh(torch.einsum("bnf,nfh->bnh", xb, w1) + b1)
        z = torch.einsum("bnh,nhk->bnk", h, w2) + b2
        if noisy:
            u = torch.rand(z.shape, generator=gen).clamp(1e-6, 1 - 1e-6)
            z = z + torch.log(u) - torch.log1p(-u)
        soft = torch.tanh(z / tau)
        hard = torch.where(z > 0, 1.0, -1.0)
        bits = soft + (hard - soft).detach()
        return head_net(bits.reshape(bits.shape[0], -1))

    loss_value, acc = float("nan"), float("nan")
    steps = max(1, epochs)
    for epoch in range(epochs):
        tau = 1.0 - 0.75 * epoch / steps if bottleneck == "gumbel" else 1.0
        perm = torch.randperm(n, generator=gen)
        total, hit = 0.0, 0
        for a in range(0, n, batch):
            idx = perm[a : a + batch]
            logits = forward(x[idx], tau, bottleneck == "gumbel")
            loss = torch.nn.functional.cross_entropy(logits, y[idx], weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            hit += int((logits.argmax(1) == y[idx]).sum())
        loss_value, acc = total / n, hit / n
    weights = {
        "w1": w1.detach().numpy().copy(),
        "b1": b1.detach().numpy().copy(),
        "w2": w2.detach().numpy().copy(),
        "b2": b2.detach().numpy().copy(),
    }
    return weights, {"train_loss": loss_value, "train_acc": acc}


# ------------------------------------------------------------------------------ factory
DEFAULT_HP: dict[str, dict[str, Any]] = {
    "e2e": {
        "bottleneck": "ste",
        "hidden": 64,
        "head": 128,
        "epochs": 20,
        "lr": 2e-3,
    },
    "vq": {"whiten": False, "dim": 16},
    "local": {"hidden": 64, "l2": 1e-2, "max_iter": 100, "folds": 3},
}


def fit_encoder(
    method: str,
    budget: int,
    layout: NodeLayout,
    featuriser: Featuriser,
    train: dict[str, np.ndarray],
    n_classes: int,
    hp: dict[str, Any],
    seed: int = 0,
) -> tuple[UplinkEncoder, Optional[np.ndarray]]:
    """Fit a learned encoder on the training windows.

    Args:
        method: One of ``METHODS``
        budget: Bytes per node message
        layout: Node layout
        featuriser: A fitted ``Featuriser``
        train: ``labelled_windows`` output of the training runs (``windows``, ``labels``,
            ``run``)
        n_classes: Normal plus faults
        hp: Method hyperparameters (``DEFAULT_HP`` shows the keys)
        seed: Seed of all random choices

    Returns:
        ``(encoder, x_train)``; ``x_train`` replaces the centre's training features when
        the code must be cross-fitted (the local classifiers), else ``None``
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected {METHODS}")
    hp = {**DEFAULT_HP[method], **hp}
    feats = featuriser(train["windows"])
    labels = np.asarray(train["labels"], dtype=np.int64)
    if method == "e2e":
        kwargs = {k: hp[k] for k in ("bottleneck", "hidden", "head", "epochs", "lr")}
        weights, _ = train_end_to_end(feats, labels, n_classes, budget, seed, **kwargs)
        return EndToEndEncoder(budget, layout, featuriser, weights), None
    if method == "vq":
        enc = ResidualVQEncoder(budget, layout, featuriser, hp["dim"], hp["whiten"])
        return enc.fit(feats, seed), None
    local = LocalClassifierEncoder(
        budget,
        layout,
        featuriser,
        n_classes,
        hidden=hp["hidden"],
        l2=hp["l2"],
        max_iter=hp["max_iter"],
    ).fit(feats, labels, seed)
    x_train = local.crossfit_decoded(
        feats, labels, np.asarray(train["run"]), int(hp["folds"]), seed
    )
    return local, x_train


def max_message(messages: np.ndarray) -> tuple[int, int]:
    """``(bytes per message, largest symbol)`` of an uplink array, for budget checks."""
    return int(messages.shape[-1]), int(messages.max()) if messages.size else 0


__all__: Sequence[str] = [
    "DEFAULT_HP",
    "METHODS",
    "EndToEndEncoder",
    "Featuriser",
    "LocalClassifierEncoder",
    "ResidualVQEncoder",
    "fit_encoder",
    "kmeans",
    "node_features",
    "pack_bits",
    "train_end_to_end",
    "unpack_bits",
]
