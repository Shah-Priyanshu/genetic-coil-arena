"""learning.quant: int8 inference of an MLP."""

import numpy as np

from genetic_coil_arena.learning import MLP
from genetic_coil_arena.learning.quant import Int8MLP


def test_int8_tracks_float_and_quantised_layers_reproduce_forward():
    net = MLP((15, 16, 1))
    rng = np.random.default_rng(0)
    params = np.stack(
        [net.init(rng) + rng.normal(0, 0.1, net.n_params) for _ in range(3)]
    )
    x = rng.uniform(-1, 1, (200, 15)).astype(np.float32)
    f, q = net.forward(params, x), Int8MLP(net).forward(params, x)
    assert q.shape == f.shape and q.dtype == np.float32
    assert np.abs(q - f).max() < 0.05 * np.abs(f).max() + 0.02
    agree = np.mean(f[:, :, 0].argmax(1) == q[:, :, 0].argmax(1))
    assert agree == 1.0
    # integer layers, run by hand, give the same scores as forward
    layers = Int8MLP(net).quantised(params[1])
    assert layers[0]["w"].dtype == np.int8 and layers[0]["b"].dtype == np.int32
    h = np.clip(np.round(x.astype(np.float64) * 64), -128, 127)
    for k, layer in enumerate(layers):
        real = (h @ layer["w"].astype(np.int64) + layer["b"]) * layer["scale"]
        h = np.clip(np.round(np.tanh(real) * 127), -127, 127) if k == 0 else real
    assert np.allclose(h, Int8MLP(net).forward(params[1], x), atol=1e-5)
