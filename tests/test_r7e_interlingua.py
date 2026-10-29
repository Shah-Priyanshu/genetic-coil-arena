"""R7e probe: status table and bridge plumbing."""

import numpy as np

from genetic_coil_arena.experiments import r7e_interlingua as r7e
from genetic_coil_arena.experiments.r7d_legibility import N_SITUATIONS


class _Rec:
    situation = np.array([[0, 3]])


def test_status_table_is_normalised():
    rec = {
        "symbols": np.array([[[1, 2]]]),
        "situation": np.array([[[0, 0]]]),
        "live": np.ones((1, 1, 2), bool),
        "eaters": np.array([[True, False]]),
    }
    t = r7e.status_table(rec, 8)
    assert t.shape == (2, N_SITUATIONS, 8)
    assert np.allclose(np.exp(t).sum(axis=2), 1.0)
    assert t[1, 0, 1] > t[0, 0, 1] and t[0, 0, 2] > t[1, 0, 2]


def test_bridge_writes_the_target_dialect():
    writer = np.zeros((2, N_SITUATIONS), np.int64)
    writer[1] = 5
    writer[0] = 7
    msgs = np.array([[[1], [0]]])
    out = r7e.Bridge(_Rec(), writer, status=lambda sym, s: sym > 0)(msgs, 0)
    assert out[0, :, 0].tolist() == [5, 7]
    table = np.log(np.full((2, N_SITUATIONS, 8), 1 / 8))
    table[1, :, 1] += 1.0
    bridge = r7e.Bridge(_Rec(), writer, reader=(table, 0.0))
    assert bridge(msgs, 0)[0, :, 0].tolist() == [5, 7]
