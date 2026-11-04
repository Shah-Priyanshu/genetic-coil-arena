"""R23: the int8 comparison and verdict (no nncase or data needed)."""

import numpy as np

from genetic_coil_arena.experiments import r23_k210 as rk


def _metrics(centre: float, ap50: float, auc: float, false: float, recall: float) -> dict:
    row = {"false_per_tile": false, "recall all": recall}
    row.update({f"recall {b}": recall for b in ("0-8", "8-16", "16-32", "32+")})
    return {"pooled": {"centre_ap all": centre, "ap50 all": ap50, "present_auc": auc,
                       "boot centre_ap": list(centre + np.linspace(-0.01, 0.01, 21)),
                       "boot ap50": list(ap50 + np.linspace(-0.01, 0.01, 21)),
                       "operating": {t: dict(row) for t in ("0.3", "0.5", "0.7")}}}


def test_compare_passes_and_kills():
    f = _metrics(0.315, 0.145, 0.889, 0.10, 0.40)
    ok = rk.compare(f, _metrics(0.305, 0.140, 0.885, 0.11, 0.39))
    assert ok["passes"] and not ok["kill"]
    assert abs(ok["centre_ap"]["drop"] - 0.010) < 1e-9
    more_false = rk.compare(f, _metrics(0.305, 0.140, 0.885, 0.13, 0.39))
    assert not more_false["passes"]  # +30% false detections
    bad = rk.compare(f, _metrics(0.260, 0.120, 0.86, 0.10, 0.35))
    assert bad["kill"] and rk.verdict({"a": ok, "b": bad}) == "KILL"
    assert rk.verdict(dict.fromkeys(rk.MODELS, ok)) == "PASS"
    assert rk.verdict({rk.MODELS[0]: ok}) == "NOT PASSED"


def test_int8_predictions_match_decode():
    rng = np.random.default_rng(0)
    out = rng.normal(size=(3, 5, 28, 28)).astype(np.float32)
    preds, best = rk.int8_predictions(out, batch=2)
    assert len(preds) == 3 and best.shape == (3,)
    assert np.allclose(best, 1 / (1 + np.exp(-out[:, 0].reshape(3, -1).max(1))))


def test_matched_threshold_keeps_false_detections_at_target():
    table = {"score": np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
             "det_gt": np.array([-1, 0, -1, -1, 1, -1])}
    # false scores 0.9, 0.7, 0.6, 0.4 over 10 tiles; target 0.2 per tile = 2 allowed
    tau = rk.matched_threshold(table, 10, 0.2)
    assert 0.6 < tau <= 0.7 + 1e-12
    assert ((table["score"] >= tau) & (table["det_gt"] < 0)).sum() == 2
    assert rk.matched_threshold(table, 10, 1.0) == 0.0


def test_select_variant_needs_a_margin():
    assert rk.select_variant({"base": 0.20, "heat": 0.204, "heat_sized": 0.203}) == "base"
    assert rk.select_variant({"base": 0.20, "heat": 0.21, "heat_sized": 0.212}) == "heat"
    assert rk.select_variant({"base": 0.20, "heat": 0.21, "heat_sized": 0.216}) == "heat_sized"
