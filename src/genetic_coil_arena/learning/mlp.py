"""A small tanh MLP with flat parameters, evaluated for a whole population at once."""

from collections.abc import Sequence

import numpy as np


class MLP:
    """Fully connected net: tanh hidden layers, linear output.

    The parameters are one flat float vector, laid out layer by layer as ``W`` (in,
    out, row-major) then ``b``. ``forward`` takes a population of such vectors.

    Args:
        sizes: Layer widths, input first and output last, e.g. ``(10, 16, 3)``
    """

    def __init__(self, sizes: Sequence[int]):
        sizes = tuple(int(s) for s in sizes)
        if len(sizes) < 2 or min(sizes) < 1:
            raise ValueError("MLP needs at least an input and an output size >= 1")
        self.sizes = sizes
        self._slices: list[tuple[slice, slice, tuple[int, int]]] = []
        start = 0
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            w = slice(start, start + n_in * n_out)
            start += n_in * n_out
            b = slice(start, start + n_out)
            start += n_out
            self._slices.append((w, b, (n_in, n_out)))
        self.n_params = start

    def init(self, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
        """Random parameters: weights N(0, scale² / fan_in), zero biases."""
        theta = np.zeros(self.n_params, dtype=np.float64)
        for w, _, (n_in, n_out) in self._slices:
            theta[w] = rng.normal(0.0, scale / np.sqrt(n_in), n_in * n_out)
        return theta

    def forward(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Outputs for a population.

        Args:
            params: (P, n_params), or (n_params,) for a single net
            x: (P, M, n_in) inputs per member, or (M, n_in) shared by all members

        Returns:
            (P, M, n_out) float32, or (M, n_out) when ``params`` is one vector
        """
        params = np.asarray(params)
        single = params.ndim == 1
        p = params[None] if single else params
        if p.shape[1] != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {p.shape[1]}")
        h = np.asarray(x, dtype=np.float32)
        if h.shape[-1] != self.sizes[0]:
            raise ValueError(f"Expected {self.sizes[0]} inputs, got {h.shape[-1]}")
        if h.ndim == 2:
            h = np.broadcast_to(h, (p.shape[0],) + h.shape)
        last = len(self._slices) - 1
        for k, (w, b, shape) in enumerate(self._slices):
            weights = p[:, w].reshape((p.shape[0],) + shape).astype(np.float32)
            h = np.matmul(h, weights) + p[:, None, b].astype(np.float32)
            if k < last:
                h = np.tanh(h)
        return h[0] if single else h


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax in float64."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
