"""R26: sample, agreement, verdict and report rendering for the confirmer on cv181x."""

import numpy as np

from genetic_coil_arena.experiments import r26_confirmer_npu as r26


def test_sample_is_stratified_and_fixed():
    person = np.arange(5000) % 3 == 0
    a, b = r26.sample(person, 100), r26.sample(person, 100)
    assert np.array_equal(a, b) and np.all(np.diff(a) > 0)
    assert person[a].sum() == 100 and (~person[a]).sum() == 100
    assert not np.array_equal(a, r26.sample(person, 100, seed=1))


def _scores(noise: float, n: int = 1200, seed: int = 0):
    rng = np.random.default_rng(seed)
    person = np.arange(n) < n // 2
    ref = np.where(person, rng.normal(2, 2, n), rng.normal(-2, 2, n))
    return ref, ref + rng.normal(0, noise, n), person


def test_close_scores_pass_and_noisy_ones_fail():
    ref, sim, person = _scores(0.01)
    res = r26.evaluate(ref, sim, person, 0.4)
    assert res["verdict"] == "PASS"
    assert res["people"]["n"] == 600 and res["non_persons"]["n"] == 600
    ref, sim, person = _scores(1.5)
    assert r26.evaluate(ref, sim, person, 0.4)["verdict"] == "KILL"


def test_verdict_needs_both_classes_and_the_auc():
    res = {"people": {"wilson": (0.995, 0.99, 0.999)},
           "non_persons": {"wilson": (0.985, 0.97, 0.99)},
           "auc": {"float": 0.9, r26.VARIANT: 0.9}}  # fmt: skip
    assert r26.verdict(res) == "NOT PASSED"
    res["non_persons"]["wilson"] = (0.995, 0.991, 0.999)
    assert r26.verdict(res) == "PASS"
    res["auc"][r26.VARIANT] = 0.89
    assert r26.verdict(res) == "NOT PASSED"


def test_render_names_the_bars_and_memory():
    ref, sim, person = _scores(0.01)
    res = r26.evaluate(ref, sim, person, 0.4)
    rep = {"verdict": res["verdict"], "test": res, "n_test": 11639,
           "compile": {"cvimodel_bytes": 50_098_288, "ion_mb": 56.48}}  # fmt: skip
    text = r26.render(rep)
    assert text.startswith("# R26:") and "**PASS.**" in text
    assert "50.1 MB, NPU (ION) memory 56.5 MB" in text and "11,639" in text
