"""R13b probe: the adapter starts at the frozen specialist and learns from labels."""

import numpy as np

from genetic_coil_arena.experiments import r13b_kshot as probe


def test_adapter_prior_reproduces_frozen_and_k0_is_identity():
    rng = np.random.default_rng(0)
    logp = np.log(rng.dirichlet(np.ones(probe.N_F), size=20))
    x = np.concatenate([logp, rng.normal(size=(20, 3))], axis=1)
    w0 = probe.prior_weights(x.shape[1])
    assert (probe.predict(w0, x) == logp.argmax(axis=1)).all()
    w = probe.fit_adapter(x[:0], np.zeros(0, int), w0)
    assert np.array_equal(w, w0)


def test_adapter_learns_a_feature_the_specialist_ignores():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 60)
    logp = np.log(np.full((60, probe.N_F), 1 / probe.N_F))
    signal = np.where(y[:, None] == 1, 3.0, -3.0)
    x = np.concatenate([logp, signal], axis=1)
    w = probe.fit_adapter(x[:40], y[:40], probe.prior_weights(x.shape[1]))
    assert (probe.predict(w, x[40:]) == y[40:]).mean() > 0.9
