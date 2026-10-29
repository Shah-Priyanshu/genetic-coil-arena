"""Integer (int8) inference of an ``MLP``, as a microcontroller would run it.

Weights are int8 per layer and member (symmetric, scale = max|W| / 127); inputs are int8
with a fixed scale; products accumulate in int32; biases are int32 at the accumulator's
scale; hidden tanh outputs are re-quantised to int8 (a 256-entry table on the chip).
The arithmetic is simulated exactly in float64: every integer here is far below 2**53.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .mlp import MLP


class Int8MLP:
    """``MLP.forward`` with int8 weights and activations (same flat parameters)."""

    def __init__(self, net: MLP, input_scale: float = 1 / 64):
        self.net, self.sizes, self.n_params = net, net.sizes, net.n_params
        self.input_scale = float(input_scale)

    def quantised(self, params: np.ndarray) -> list[dict[str, Any]]:
        """One net's layers as integers: int8 ``w`` (in, out), int32 ``b`` and the
        float ``scale`` of the layer's accumulator."""
        layers, s_in = [], self.input_scale
        for w, b, shape in self.net._slices:
            weights = np.asarray(params[w], dtype=np.float64).reshape(shape)
            s_w = float(np.abs(weights).max()) / 127 or 1.0
            layers.append({
                "w": np.round(weights / s_w).astype(np.int8),
                "b": np.round(np.asarray(params[b]) / (s_in * s_w)).astype(np.int32),
                "scale": s_in * s_w,
            })  # fmt: skip
            s_in = 1 / 127
        return layers

    def forward(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """As ``MLP.forward``: (P, n_params) or (n_params,) with (P, M, in) or (M, in)."""
        params = np.asarray(params, dtype=np.float64)
        single = params.ndim == 1
        p = params[None] if single else params
        h = np.asarray(x, dtype=np.float64)
        if h.ndim == 2:
            h = np.broadcast_to(h, (p.shape[0],) + h.shape)
        s_in = self.input_scale
        q = np.clip(np.round(h / s_in), -128, 127)
        last = len(self.net._slices) - 1
        out = q
        for k, (w, b, shape) in enumerate(self.net._slices):
            weights = p[:, w].reshape((p.shape[0],) + shape)
            s_w = np.abs(weights).max(axis=(1, 2), keepdims=True) / 127
            s_w = np.where(s_w > 0, s_w, 1.0)
            acc = np.matmul(q, np.round(weights / s_w))
            bias = np.round(p[:, None, b] / (s_in * s_w))
            z = (acc + bias) * (s_in * s_w)
            if k < last:
                q = np.clip(np.round(np.tanh(z) * 127), -127, 127)
                s_in = 1 / 127
            else:
                out = z
        out = out.astype(np.float32)
        return out[0] if single else out
