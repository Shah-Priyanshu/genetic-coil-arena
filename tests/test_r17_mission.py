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


def test_robustness_conditions_change_the_world_not_the_defaults():
    from dataclasses import replace

    from genetic_coil_arena.experiments import r17_robust as rr

    m = Mission(120.0, 120.0, 6, 300.0, Detector(px50=8, fa50=0.001))
    base = run_missions(m, hand_two_tier(6), SEEDS)
    same = run_missions(replace(m, person_speed=0.0, loc_error=1.0, speed=4.0),
                        hand_two_tier(6), SEEDS)  # fmt: skip
    assert np.array_equal(base["score"], same["score"])
    walking = run_missions(replace(m, person_speed=1.2), hand_two_tier(6), SEEDS)
    assert walking["found"].mean() < base["found"].mean()
    slow = run_missions(replace(m, speed=2.0), hand_two_tier(6), SEEDS)
    assert slow["swept"].mean() < base["swept"].mean()
    big = rr.missions_for("areas x1.6")[0]
    assert big.width == pytest.approx(1.6 * r17.held_out_missions()[0].width)
    assert rr.missions_for("10 drones")[0].drones == 10


def test_robustness_bar_needs_size_and_ci_everywhere():
    from genetic_coil_arena.experiments import r17_robust as rr

    rng = np.random.default_rng(1)
    cells = {}
    for cond, gain in (("nominal", 0.10), ("GPS error 3 m", 0.02)):
        for name, shift in (("blind", 0.0), ("conditioned", gain),
                            ("OpenSwarm default", -0.5), ("hand two-tier", 0.0)):  # fmt: skip
            for i in range(r17.N_TEST):
                s = 0.4 + shift + rng.normal(0, 0.005)
                cells[(cond, name, 0, i)] = {"score": s, "found": s, "wrong": 0.0}
    rep = rr.summarise(cells)
    assert rep["by_condition"]["nominal"]["pass"]
    assert not rep["by_condition"]["GPS error 3 m"]["pass"]  # CI above 0 but < 3 points
    assert not rep["pass"] and rep["fails"] == ["GPS error 3 m"]
    assert "FAIL" in rr.render(rep)


def test_handoff_keeps_a_walking_person_and_version_2_plans():
    from dataclasses import replace

    from genetic_coil_arena.experiments import r17b_handoff as rh

    m = Mission(80.0, 80.0, 6, 300.0, Detector(px50=8, fa50=1e-9), person_speed=0.5)
    plain = run_missions(m, hand_two_tier(6), SEEDS)
    handoff = run_missions(m, replace(hand_two_tier(6), track_s=60.0), SEEDS)
    assert handoff["found"].mean() > plain["found"].mean() + 0.3
    plan = replace(hand_two_tier(6), track_s=45.0)
    back = r17.decode_plan(r17.encode_plan(plan, 6, 2), 6, 2)
    assert abs(back.track_s - 45.0) < 1e-6
    assert r17.decode_plan(r17.encode_plan(plan, 6), 6).track_s == 0.0
    ms = r17.held_out_missions(2)
    assert 0 < sum(x.person_speed > 0 for x in ms) < len(ms)
    assert r17.features(ms[0], 2).shape == (r17.N_FEATURES_V2,)
    for conditioned in (False, True):
        settings = r17.PlannerSettings(conditioned=conditioned, version=2)
        planner = r17.Planner(settings)
        plans = planner.plans(planner.init(np.random.default_rng(0)), ms[:4])
        assert all(abs(p.track_s - 30.0) < 3 for p in plans)
    assert set(rh.BARS) == set(rh.LABELS)


def test_handoff_summary_bars():
    from genetic_coil_arena.experiments import r17b_handoff as rh

    rng = np.random.default_rng(2)
    cells = {}
    shifts = {"OpenSwarm default": -0.5, "hand two-tier": 0.0,
              "hand two-tier + hand-off": 0.2, "step-1 planner": 0.0, "blind v2": 0.1,
              "conditioned v2": 0.25}  # fmt: skip
    for name, shift in shifts.items():
        for i in range(r17.N_TEST):
            s = 0.3 + shift + rng.normal(0, 0.01)
            cells[(2, name, 0, i)] = {"score": s, "found": s, "wrong": 0.0}
    for name, shift in (("step-1 planner", 0.0), ("conditioned v2", -0.01)):
        for i in range(r17.N_TEST):
            cells[(1, name, 0, i)] = {"score": 0.6 + shift, "found": 0.6, "wrong": 0.0}
    rep = rh.summarise(cells)
    assert rep["passed"] and all(rep["bars"].values())
    assert "GOOD SIGN" in rh.render(rep)


def test_version_3_hint_feature_and_conditions():
    from genetic_coil_arena.experiments import r17c_hint as rc

    v2, v3 = r17.held_out_missions(2), r17.held_out_missions(3)
    assert [m.person_speed for m in v2] == [m.person_speed for m in v3]
    right = [(m.moving_hint > 0) == (m.person_speed > 0) for m in v3]
    assert 0.6 < np.mean(right) < 1.0
    assert r17.features(v3[0], 3).shape == (r17.N_FEATURES_V3,)
    assert r17.features(v3[0], 3)[-1] == v3[0].moving_hint
    wrong = rc.missions_for("hint always wrong")
    assert all((m.moving_hint > 0) != (m.person_speed > 0) for m in wrong)
    assert all(m.moving_hint == -1 for m in rc.missions_for("still person, hint still"))
    planner = r17.Planner(r17.PlannerSettings(version=3))
    plans = planner.plans(planner.init(np.random.default_rng(0)), v3[:3])
    assert all(abs(p.track_s - 30.0) < 3 for p in plans)


def test_hint_summary_bars():
    from genetic_coil_arena.experiments import r17c_hint as rc

    rng = np.random.default_rng(3)
    cells = {}
    shift = {"step-1 planner": 0.0, "planner v2": -0.1, "planner v3 (hint)": 0.0}
    for cond in rc.CONDITIONS:
        for name, d in shift.items():
            for i in range(r17.N_TEST):
                s = 0.4 + d + rng.normal(0, 0.005)
                cells[(cond, name, 0, i)] = {"score": s, "found": s, "wrong": 0.0}
    rep = rc.summarise(cells)
    assert rep["passed"] and "GOOD SIGN" in rc.render(rep)
