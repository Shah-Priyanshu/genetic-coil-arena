"""Hand-designed node encoders at a fixed byte budget, with their matching decoders.

An encoder maps the nodes' normalised windows ``(B, N, Vmax, W)`` to the uplink
``(B, N, L)`` int64 (L = byte budget, one symbol in [0, 256) per byte) and back to the
float features the centre classifies. Everything an encoder calibrates (alarm
thresholds, PCA, band-energy baselines) comes from normal *training* windows via ``fit``.
The designs:

- ``AlarmEncoder``: one byte of alarm bits per node (budget-independent);
- ``SummaryEncoder``: window mean, std, max and min of every variable, quantised;
- ``FeatureEncoder``: process-monitoring features, quantised: PCA residual (SPE) and
  Hotelling T2 of the node, its leading PCA scores, then FFT band energies per variable;
- ``SilentEncoder``: no bytes;
- ``NoiseEncoder``: uniformly random bytes (a control that carries no information).

The quantised designs fill the node's byte budget in priority order. If all fields fit
at 8 bits they are sent at 8 bits; otherwise at 4 bits (two per byte) and the list is cut
off at ``2 * budget`` fields. Quantisers are fixed (a signed asinh scale, or a linear
scale for log energy ratios), not fitted.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .budget import VOCAB_SIZE
from .windows import NodeLayout

N_BANDS = 3


# ---------------------------------------------------------------------------- features
def window_stats(windows: np.ndarray) -> dict[str, np.ndarray]:
    """Per-variable statistics of (B, N, V, W) windows; each result is (B, N, V)."""
    w = windows.astype(np.float64)
    t = np.arange(w.shape[-1], dtype=np.float64)
    t -= t.mean()
    return {
        "mean": w.mean(axis=-1),
        "std": w.std(axis=-1),
        "max": w.max(axis=-1),
        "min": w.min(axis=-1),
        "slope": (w * t).sum(axis=-1) / (t**2).sum(),
    }


def band_slices(window: int, n_bands: int = N_BANDS) -> list[np.ndarray]:
    """FFT bins 1..W/2 (DC excluded) split into contiguous bands, low to high."""
    bins = np.arange(1, window // 2 + 1)
    return list(np.array_split(bins, n_bands))


def band_energies(windows: np.ndarray) -> np.ndarray:
    """Detrended FFT energy per band: (B, N, V, W) -> (B, N, V, bands), energy / W."""
    w = windows.astype(np.float64)
    w = w - w.mean(axis=-1, keepdims=True)
    spec = np.abs(np.fft.rfft(w, axis=-1)) ** 2 / w.shape[-1]
    return np.stack(
        [spec[..., b].sum(axis=-1) for b in band_slices(w.shape[-1])], axis=-1
    )


# ---------------------------------------------------------------------------- quantisers
@dataclass(frozen=True)
class Quantiser:
    """Fixed uniform quantiser on a transformed scale.

    Args:
        kind: "asinh" (t = asinh(x / scale), fine near zero and coarse for big values)
            or "linear" (t = x)
        lo: Smallest representable x
        hi: Largest representable x
        scale: asinh scale in x units
    """

    kind: str
    lo: float
    hi: float
    scale: float = 1.0

    def _t(self, x: np.ndarray) -> np.ndarray:
        return np.arcsinh(x / self.scale) if self.kind == "asinh" else x

    def _bounds(self) -> tuple[float, float]:
        lo, hi = self._t(np.array([self.lo, self.hi], dtype=np.float64))
        return float(lo), float(hi)

    def levels(self, x: np.ndarray, bits: int) -> np.ndarray:
        """Integer levels in [0, 2^bits)."""
        lo, hi = self._bounds()
        u = (np.clip(self._t(np.asarray(x, dtype=np.float64)), lo, hi) - lo) / (hi - lo)
        return np.rint(u * (2**bits - 1)).astype(np.int64)

    def values(self, levels: np.ndarray, bits: int) -> np.ndarray:
        """Representative x of each level."""
        lo, hi = self._bounds()
        t = lo + np.asarray(levels, dtype=np.float64) / (2**bits - 1) * (hi - lo)
        return np.sinh(t) * self.scale if self.kind == "asinh" else t


SIGNED = Quantiser("asinh", -30.0, 30.0, 0.5)
UNSIGNED_STD = Quantiser("asinh", 0.0, 20.0, 0.25)
LOG_RATIO = Quantiser("linear", -1.5, 3.0)


class FieldCodec:
    """Packs a priority-ordered list of quantised fields per node into ``budget`` bytes.

    Args:
        slots_per_node: Per node, the (group, variable) slots in priority order
        quantisers: One quantiser per group
        budget: Bytes per node message (L)
    """

    def __init__(
        self,
        slots_per_node: Sequence[Sequence[tuple[int, int]]],
        quantisers: Sequence[Quantiser],
        budget: int,
    ):
        self.budget = budget
        self.quantisers = list(quantisers)
        self.bits: list[int] = []
        self.slots: list[list[tuple[int, int]]] = []
        for slots in slots_per_node:
            if len(slots) <= budget:
                self.bits.append(8)
                self.slots.append(list(slots))
            else:
                self.bits.append(4)
                self.slots.append(list(slots[: 2 * budget]))

    @property
    def n_fields(self) -> int:
        """Decoded features per message batch row, summed over nodes."""
        return sum(len(s) for s in self.slots)

    @property
    def bytes_used(self) -> tuple[int, ...]:
        """Bytes each node actually fills (the rest of its budget is zero padding)."""
        return tuple(math.ceil(len(s) * b / 8) for s, b in zip(self.slots, self.bits))

    def encode(self, feats: np.ndarray) -> np.ndarray:
        """(B, N, G, V) features -> (B, N, budget) int64 bytes."""
        b, n = feats.shape[:2]
        out = np.zeros((b, n, self.budget), dtype=np.int64)
        for i, (slots, bits) in enumerate(zip(self.slots, self.bits)):
            if not slots:
                continue
            levels = np.stack(
                [self.quantisers[g].levels(feats[:, i, g, v], bits) for g, v in slots],
                axis=1,
            )  # (B, fields)
            if bits == 8:
                out[:, i, : levels.shape[1]] = levels
            else:
                if levels.shape[1] % 2:
                    levels = np.concatenate([levels, np.zeros((b, 1), np.int64)], 1)
                packed = levels[:, 0::2] * 16 + levels[:, 1::2]
                out[:, i, : packed.shape[1]] = packed
        return out

    def decode(self, messages: np.ndarray) -> np.ndarray:
        """(B, N, budget) bytes -> (B, n_fields) float32 dequantised features."""
        cols: list[np.ndarray] = []
        for i, (slots, bits) in enumerate(zip(self.slots, self.bits)):
            if not slots:
                continue
            msg = messages[:, i, :]
            if bits == 8:
                levels = msg[:, : len(slots)]
            else:
                packed = msg[:, : math.ceil(len(slots) / 2)]
                levels = np.stack([packed // 16, packed % 16], axis=2).reshape(
                    msg.shape[0], -1
                )[:, : len(slots)]
            for j, (g, _) in enumerate(slots):
                cols.append(self.quantisers[g].values(levels[:, j], bits))
        if not cols:
            return np.zeros((messages.shape[0], 0), dtype=np.float32)
        return np.stack(cols, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------- encoders
class UplinkEncoder:
    """Base class: windows -> bytes (``encode``) and bytes -> features (``decode``)."""

    name = "base"

    def __init__(self, budget: int, layout: NodeLayout):
        if budget < 0:
            raise ValueError("budget must be >= 0")
        self.budget = int(budget)
        self.layout = layout

    def fit(self, windows: np.ndarray) -> "UplinkEncoder":
        """Calibrate on normal training windows (n, N, Vmax, W); default: nothing."""
        return self

    def encode_windows(self, windows: np.ndarray) -> np.ndarray:
        """(B, N, Vmax, W) windows -> (B, N, budget) int64."""
        raise NotImplementedError

    def encode(self, specialist_obs: dict[str, np.ndarray]) -> np.ndarray:
        """The uplink for an environment's specialist observations."""
        return self.encode_windows(specialist_obs["window"])

    def decode(self, messages: np.ndarray) -> np.ndarray:
        """(B, N, budget) bytes -> (B, F) float32 features for the centre."""
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        raise NotImplementedError

    @property
    def bytes_used(self) -> tuple[int, ...]:
        """Bytes each node actually sends (<= budget)."""
        raise NotImplementedError

    def _empty(self, windows: np.ndarray) -> np.ndarray:
        return np.zeros((windows.shape[0], windows.shape[1], self.budget), np.int64)


class SilentEncoder(UplinkEncoder):
    """(d) Sends nothing: zero bytes whatever the channel allows."""

    name = "silent"

    def encode_windows(self, windows: np.ndarray) -> np.ndarray:
        return self._empty(windows)

    def decode(self, messages: np.ndarray) -> np.ndarray:
        return np.zeros((messages.shape[0], 0), dtype=np.float32)

    @property
    def n_features(self) -> int:
        return 0

    @property
    def bytes_used(self) -> tuple[int, ...]:
        return (0,) * self.layout.n_nodes


class NoiseEncoder(UplinkEncoder):
    """Control: uniformly random bytes, independent of the data (seeded)."""

    name = "noise"

    def __init__(self, budget: int, layout: NodeLayout, seed: int = 0):
        super().__init__(budget, layout)
        self._rng = np.random.default_rng(seed)

    def encode_windows(self, windows: np.ndarray) -> np.ndarray:
        shape = (windows.shape[0], windows.shape[1], self.budget)
        return self._rng.integers(0, VOCAB_SIZE, size=shape, dtype=np.int64)

    def decode(self, messages: np.ndarray) -> np.ndarray:
        return (messages.reshape(messages.shape[0], -1) / 255.0).astype(np.float32)

    @property
    def n_features(self) -> int:
        return self.layout.n_nodes * self.budget

    @property
    def bytes_used(self) -> tuple[int, ...]:
        return (self.budget,) * self.layout.n_nodes


class AlarmEncoder(UplinkEncoder):
    """(a) One byte of alarm bits per node.

    Each node's variables are cut into 8 contiguous groups, one alarm bit per group:
    set when any variable's window mean is farther from the normal mean than that
    variable's threshold, the ``1 - false_alarm`` quantile of its absolute window mean on
    normal training windows (two-sided). The message is 1 byte whatever the budget (0 at
    budget 0); the other bytes of a larger budget stay zero.
    """

    name = "alarms"
    BITS = 8

    def __init__(self, budget: int, layout: NodeLayout, false_alarm: float = 0.005):
        super().__init__(budget, layout)
        self.false_alarm = false_alarm
        self.threshold: Optional[np.ndarray] = None
        n_nodes, width = layout.index.shape
        self.group = np.zeros((n_nodes, width), dtype=np.int64)
        for i in range(n_nodes):
            nv = int(layout.mask[i].sum())
            for g, chunk in enumerate(np.array_split(np.arange(nv), self.BITS)):
                self.group[i, chunk] = g

    def fit(self, windows: np.ndarray) -> "AlarmEncoder":
        means = np.abs(windows.mean(axis=-1))  # (n, N, V)
        self.threshold = np.quantile(means, 1.0 - self.false_alarm, axis=0)
        return self

    def _bits(self, windows: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("fit() the encoder on normal training windows first")
        over = (np.abs(windows.mean(axis=-1)) > self.threshold) & self.layout.mask
        bits = np.zeros(windows.shape[:2] + (self.BITS,), dtype=bool)
        for g in range(self.BITS):
            bits[:, :, g] = (over & (self.group == g)).any(axis=-1)
        return bits

    def encode_windows(self, windows: np.ndarray) -> np.ndarray:
        out = self._empty(windows)
        if self.budget >= 1:
            bits = self._bits(windows).astype(np.int64)
            out[:, :, 0] = (bits * (1 << np.arange(self.BITS))).sum(axis=-1)
        return out

    def decode(self, messages: np.ndarray) -> np.ndarray:
        if self.budget < 1:
            return np.zeros((messages.shape[0], 0), dtype=np.float32)
        byte = messages[:, :, 0]
        bits = (byte[:, :, None] >> np.arange(self.BITS)) & 1
        return bits.reshape(messages.shape[0], -1).astype(np.float32)

    @property
    def n_features(self) -> int:
        return self.layout.n_nodes * self.BITS if self.budget >= 1 else 0

    @property
    def bytes_used(self) -> tuple[int, ...]:
        return (min(self.budget, 1),) * self.layout.n_nodes


class _CodecEncoder(UplinkEncoder):
    """Shared plumbing for encoders built on a ``FieldCodec``."""

    codec: FieldCodec

    def features(self, windows: np.ndarray) -> np.ndarray:
        """(B, N, G, V) raw feature groups before quantisation."""
        raise NotImplementedError

    def encode_windows(self, windows: np.ndarray) -> np.ndarray:
        if self.budget == 0:
            return self._empty(windows)
        return self.codec.encode(self.features(windows))

    def decode(self, messages: np.ndarray) -> np.ndarray:
        return self.codec.decode(messages)

    @property
    def n_features(self) -> int:
        return self.codec.n_fields

    @property
    def bytes_used(self) -> tuple[int, ...]:
        return self.codec.bytes_used


class SummaryEncoder(_CodecEncoder):
    """(b) Quantised window mean, std, max and min of every variable.

    Fields are ordered mean of all variables, then std, max, min, and cut to the budget.
    """

    name = "summary"
    GROUPS = ("mean", "std", "max", "min")

    def __init__(self, budget: int, layout: NodeLayout):
        super().__init__(budget, layout)
        slots = [
            [(g, v) for g in range(len(self.GROUPS)) for v in range(int(nv))]
            for nv in layout.n_vars
        ]
        self.codec = FieldCodec(slots, [SIGNED, UNSIGNED_STD, SIGNED, SIGNED], budget)

    def features(self, windows: np.ndarray) -> np.ndarray:
        s = window_stats(windows)
        return np.stack([s[g] for g in self.GROUPS], axis=2)


class FeatureEncoder(_CodecEncoder):
    """(c) Quantised process-monitoring features.

    Per node, in priority order: the log PCA residual (SPE) and log Hotelling T2 of the
    window-mean vector, the leading PCA scores (enough components for 90% of the normal
    variance, at most 6), then the log ratio of each variable's low, mid and high FFT
    band energy to its normal-training average. PCA and the energy baselines are fitted on
    normal training windows.
    """

    name = "features"
    MAX_COMPONENTS = 6

    def __init__(self, budget: int, layout: NodeLayout, variance: float = 0.9):
        super().__init__(budget, layout)
        self.variance = variance
        self._fitted = False
        n_nodes, width = layout.index.shape
        self.mu = np.zeros((n_nodes, width))
        self.components: list[np.ndarray] = []
        self.inv_std: list[np.ndarray] = []
        self.spe_ref = np.ones(n_nodes)
        self.energy_ref = np.ones((n_nodes, width, N_BANDS))
        self.all_slots: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
        self.codec = FieldCodec([[] for _ in range(n_nodes)], [], budget)

    def fit(self, windows: np.ndarray) -> "FeatureEncoder":
        n_nodes = self.layout.n_nodes
        means = windows.astype(np.float64).mean(axis=-1)  # (n, N, V)
        self.components, self.inv_std = [], []
        for i in range(n_nodes):
            nv = int(self.layout.n_vars[i])
            x = means[:, i, :nv]
            self.mu[i, :nv] = x.mean(axis=0)
            xc = x - self.mu[i, :nv]
            vals, vecs = np.linalg.eigh(xc.T @ xc / max(len(xc) - 1, 1))
            order = np.argsort(vals)[::-1]
            vals, vecs = np.maximum(vals[order], 1e-12), vecs[:, order]
            share = np.cumsum(vals) / vals.sum()
            k = int(np.searchsorted(share, self.variance) + 1)
            k = max(1, min(k, self.MAX_COMPONENTS, nv))
            self.components.append(vecs[:, :k])
            self.inv_std.append(1.0 / np.sqrt(vals[:k]))
            resid = xc - (xc @ vecs[:, :k]) @ vecs[:, :k].T
            self.spe_ref[i] = max(float((resid**2).sum(axis=1).mean()), 1e-9)
        self.energy_ref = np.maximum(band_energies(windows).mean(axis=0), 1e-9)
        slots = []
        for i in range(n_nodes):
            nv = int(self.layout.n_vars[i])
            pca = [(0, j) for j in range(2 + self.components[i].shape[1])]
            bands = [(1 + b, v) for b in range(N_BANDS) for v in range(nv)]
            slots.append(pca + bands)
        self.all_slots = slots
        self.codec = FieldCodec(
            slots, [SIGNED, LOG_RATIO, LOG_RATIO, LOG_RATIO], self.budget
        )
        self._fitted = True
        return self

    def unquantised(self, windows: np.ndarray) -> np.ndarray:
        """Every feature of every node at full precision, (B, total slots) float32.

        What a centre with unlimited bandwidth could read (the oracle uses it).
        """
        feats = self.features(windows)
        cols = [
            feats[:, i, g, v]
            for i, slots in enumerate(self.all_slots)
            for g, v in slots
        ]
        return np.stack(cols, axis=1).astype(np.float32)

    def features(self, windows: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit() the encoder on normal training windows first")
        b, n_nodes, width, _ = windows.shape
        means = windows.astype(np.float64).mean(axis=-1)
        out = np.zeros((b, n_nodes, 1 + N_BANDS, width))
        for i in range(n_nodes):
            nv = int(self.layout.n_vars[i])
            xc = means[:, i, :nv] - self.mu[i, :nv]
            p, inv = self.components[i], self.inv_std[i]
            scores = (xc @ p) * inv
            resid = xc - (xc @ p) @ p.T
            spe = (resid**2).sum(axis=1)
            t2 = (scores**2).sum(axis=1)
            k = p.shape[1]
            out[:, i, 0, 0] = np.log10((spe + 1e-9) / self.spe_ref[i])
            out[:, i, 0, 1] = np.log10((t2 + 1e-9) / k)
            out[:, i, 0, 2 : 2 + k] = scores
        energy = band_energies(windows)  # (B, N, V, bands)
        ratio = np.log10((energy + 1e-9) / self.energy_ref[None])
        for band in range(N_BANDS):
            out[:, :, 1 + band, :] = ratio[..., band]
        return out


def make_encoder(
    name: str, budget: int, layout: NodeLayout, seed: int = 0
) -> UplinkEncoder:
    """Encoder by name: alarms, summary, features, silent or noise."""
    if name == "alarms":
        return AlarmEncoder(budget, layout)
    if name == "summary":
        return SummaryEncoder(budget, layout)
    if name == "features":
        return FeatureEncoder(budget, layout)
    if name == "silent":
        return SilentEncoder(budget, layout)
    if name == "noise":
        return NoiseEncoder(budget, layout, seed)
    raise ValueError(f"Unknown encoder {name!r}")


ENCODERS = ("alarms", "summary", "features", "silent", "noise")
