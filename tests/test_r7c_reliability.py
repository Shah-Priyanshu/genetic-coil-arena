"""G7c: restart rule, verdict and report on synthetic evaluation results."""

from types import SimpleNamespace

import pytest

from genetic_coil_arena.experiments import r7c_reliability as r7c
from genetic_coil_arena.experiments.r7b_report import HELDOUT, SEEN


def _training(vals, start=16):
    return [
        {"mode": "centre-goal", "run": start + i, "best_val_mission": v}
        for i, v in enumerate(vals)
    ]


def test_select_runs_skips_flagged_and_keeps_first_n():
    vals = [20.0] * 20
    vals[2], vals[5] = 5.0, 11.99
    kept, flagged = r7c.select_runs(_training(vals))
    assert flagged == [18, 21]
    assert kept == [r for r in range(16, 36) if r not in (18, 21)][: r7c.N_KEEP]
    assert len(kept) == r7c.N_KEEP


def test_select_runs_threshold_is_strict_and_short_pools_keep_all():
    kept, flagged = r7c.select_runs(_training([r7c.FLAG_BELOW, 3.0, 4.0]))
    assert flagged == [17, 18] and kept == [16]


@pytest.mark.parametrize(
    "kept, verdict",
    [([21.0, 21.0], "PASS"), ([17.5, 17.5], "NOT PASSED"), ([16.0, 16.0], "KILL")],
)
def test_verdict_lines(kept, verdict):
    assert r7c.g7c_verdict(kept, [20.0, 20.0])["verdict"] == verdict


def _sets(scores):
    def result(scale):
        conds = [
            SimpleNamespace(
                name=r7c.B,
                seeds=list(scores),
                per_seed=[scale * v for v in scores.values()],
            ),
            SimpleNamespace(
                name="random-actions", seeds=list(scores), per_seed=[-4.0] * len(scores)
            ),
        ]
        return {"result": SimpleNamespace(conditions=conds)}

    return {HELDOUT: result(1.0), SEEN: result(0.9)}


def test_build_report_and_render():
    vals = [20.0] * 20
    vals[3] = 6.0
    trained = _training(vals)
    held = {16 + i: (7.0 if i == 3 else 22.0) for i in range(20)}
    ref = {
        "per": {
            w: {r7c.B: [21.0] * 16, r7c.D: [23.0, 24.0] * 8, r7c.A: [-3.4, -3.3] * 8}
            for w in (HELDOUT, SEEN)
        },
        "seeds": list(range(16)),
        "training": _training([20.0] * 14 + [5.7, 8.2], start=0),
    }
    rep = r7c.build_report(trained, {"generations": 10}, ref, sets=_sets(held))
    assert 19 not in rep["kept"] and len(rep["kept"]) == 16
    assert rep["flagged"] == [{"run": 19, "best_val": 6.0, "heldout": 7.0}]
    assert rep["flag_rate"]["flagged"] == 3 and rep["flag_rate"]["runs"] == 36
    assert rep["g7c"]["verdict"] == "PASS"  # 22 / 23.5 = 94%
    text = r7c.render(rep)
    assert "G7c verdict: PASS" in text and "| 19 | 6.0 | 7.0 |" in text
    with pytest.raises(ValueError):
        r7c.build_report(
            trained + [{"mode": "blind", "run": 0, "best_val_mission": 0}],
            {},
            ref,
            sets=_sets(held),
        )
