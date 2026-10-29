"""R17: the OpenSwarm person-search simulator and the mission planners."""

import numpy as np
import pytest

from genetic_coil_arena.envs.swarm.mission import (
    OPENSWARM_DEFAULT,
    Detector,
    Mission,
    Plan,
    hand_two_tier,
    run_missions,
)
from genetic_coil_arena.experiments import r17_mission as r17

SEEDS = list(range(100, 400))
PERFECT = Detector(px50=0.01, slope=3.0, fa50=1e-12)
BLIND = Detector(px50=1e6, slope=3.0, fa50=0.05)


def test_episodes_are_deterministic_per_seed():
    m, plan = Mission(), hand_two_tier(6)
    a, b = run_missions(m, plan, SEEDS), run_missions(m, plan, SEEDS)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_perfect_detector_finds_everyone_in_a_small_area():
    m = Mission(40.0, 40.0, 6, 300.0, PERFECT)
    r = run_missions(m, OPENSWARM_DEFAULT, SEEDS)
    assert r["found"].all() and r["swept"].max() <= 1.0 + 1e-9
    # confirmers hover over the reported spot: localisation error can put the person
    # outside a low camera's small view, and the candidate is dropped
    r = run_missions(m, hand_two_tier(6), SEEDS)
    assert 0.8 < r["found"].mean() < 1 and r["wrong"].mean() == 0


def test_false_alarms_mislead_the_default_but_confirmers_check_them():
    m = Mission(120.0, 120.0, 6, 300.0, BLIND)
    assert run_missions(m, OPENSWARM_DEFAULT, SEEDS)["wrong"].mean() > 0.95
    checked = Plan(confirmers=2, confirm_thr=0.95, confirm_looks=2)
    r = run_missions(m, checked, SEEDS)
    assert r["found"].mean() == 0 and r["wrong"].mean() < 0.6


def test_battery_bounds_the_sortie_and_scores_are_bounded():
    m = Mission(240.0, 240.0, 3, 60.0, PERFECT)
    r = run_missions(m, OPENSWARM_DEFAULT, SEEDS)
    assert r["t"].max() <= 60 + 200 and r["swept"].max() < 0.2
    assert set(np.unique(r["score"])) <= {1.0, 0.0, -2.0}
    with pytest.raises(ValueError):
        run_missions(m, Plan(confirmers=3), SEEDS)


def test_plan_encoding_round_trips_and_planners_start_at_the_hand_plan():
    plan = hand_two_tier(6)
    back = r17.decode_plan(r17.encode_plan(plan, 6), 6)
    assert (
        back.confirmers == plan.confirmers and back.confirm_looks == plan.confirm_looks
    )
    assert abs(back.sweep_alt - plan.sweep_alt) < 1e-6
    missions = r17.held_out_missions()[:5]
    for conditioned in (False, True):
        planner = r17.Planner(r17.PlannerSettings(conditioned=conditioned))
        theta = planner.init(np.random.default_rng(0))
        alts = [p.sweep_alt for p in planner.plans(theta, missions)]
        assert np.allclose(alts, plan.sweep_alt, atol=0.6)
    assert r17.held_out_missions() == r17.held_out_missions()
    assert r17.features(missions[0]).shape == (r17.N_FEATURES,)


def test_summary_bars():
    plans = {"blind": [[Plan()] * r17.N_TEST], "conditioned": [[Plan()] * r17.N_TEST],
             "per-mission tuned": [[Plan()] * r17.N_TEST]}  # fmt: skip
    rng = np.random.default_rng(0)
    cells = {}
    for name, shift in (
        ("blind", 0.0),
        ("conditioned", 0.06),
        ("per-mission tuned", 0.02),
    ):
        for i in range(r17.N_TEST):
            s = 0.3 + shift + rng.normal(0, 0.01)
            cells[(name, 0, i)] = {"score": s, "found": s, "wrong": 0.0, "swept": 0.5}
    rep = r17.summarise(cells, plans)
    assert rep["good_sign"]["pass"] and not rep["room"]["pass"]
    assert "PASS" in r17.render(rep) and "FAIL" in r17.render(rep)
