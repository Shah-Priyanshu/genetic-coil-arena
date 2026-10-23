"""Tests for the learned-protocol experiment (R6) and the domain-agnostic learning package."""

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from genetic_coil_arena.envs.squad import HandCodedCentre, SquadConfig, SquadEnv
from genetic_coil_arena.experiments.learned_protocol import (
    CONDITION,
    HAND_CODED,
    LEARNED,
    MODES,
    SCRIPTED,
    SILENT,
    EvalSettings,
    LearnedCentre,
    LearnedSpecialist,
    Nets,
    TrainSettings,
    centre_features,
    entropy_bits,
    evaluate_radius,
    policies_for,
    population_rollout,
    specialist_input_size,
    specialist_inputs,
    test_episode_seeds,
    train_run,
    welch_ci,
)
from genetic_coil_arena.learning import MLP, EvolutionStrategy, centered_ranks, softmax

TINY = TrainSettings(
    radius=3,
    generations=2,
    population=4,
    episodes_per_member=2,
    eval_every=1,
    val_episodes=2,
    max_steps=40,
)


# ------------------------------------------------------------------ learning package
def test_mlp_population_matches_single_members():
    net = MLP((5, 4, 3))
    rng = np.random.default_rng(0)
    params = np.stack([net.init(rng) for _ in range(3)])
    x = rng.normal(size=(3, 7, 5)).astype(np.float32)
    out = net.forward(params, x)
    assert out.shape == (3, 7, 3)
    for p in range(3):
        np.testing.assert_allclose(net.forward(params[p], x[p]), out[p], rtol=1e-5)
    assert net.forward(params, x[0]).shape == (3, 7, 3)  # shared inputs broadcast
    with pytest.raises(ValueError):
        net.forward(params[:, :-1], x)


def test_mlp_parameter_count():
    assert MLP((5, 4, 3)).n_params == 5 * 4 + 4 + 4 * 3 + 3


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1000.0, 0.0, -5.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_allclose(p[1], 1 / 3)


def test_centered_ranks():
    r = centered_ranks(np.array([10.0, -3.0, 4.0]))
    np.testing.assert_allclose(r, [0.5, -0.5, 0.0])
    assert centered_ranks(np.array([1.0]))[0] == 0.0


def test_es_maximises_a_quadratic():
    target = np.array([1.0, -2.0, 0.5])
    es = EvolutionStrategy(np.zeros(3), population=20, sigma=0.1, lr=0.1, seed=0,
                           weight_decay=0.0)  # fmt: skip
    for _ in range(200):
        cands = es.ask()
        es.tell(-((cands - target) ** 2).sum(axis=1))
    np.testing.assert_allclose(es.theta, target, atol=0.15)


def test_es_mask_freezes_parameters_and_validates():
    es = EvolutionStrategy(np.ones(4), population=6, mask=np.array([1, 1, 0, 0]))
    cands = es.ask()
    assert np.all(cands[:, 2:] == 1.0)
    es.tell(cands[:, 0])
    assert np.all(es.theta[2:] == 1.0)
    with pytest.raises(ValueError):
        EvolutionStrategy(np.zeros(2), population=3)
    with pytest.raises(RuntimeError):
        EvolutionStrategy(np.zeros(2), population=2).tell([0.0, 1.0])


def test_learning_package_is_domain_agnostic():
    code = (
        "import sys, genetic_coil_arena.learning; "
        "bad = [m for m in sys.modules if m.startswith(('genetic_coil_arena.envs', "
        "'genetic_coil_arena.snake_core', 'numba'))]; print(bad); assert not bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ------------------------------------------------------------------ features
def test_centre_features_are_egocentric_and_sorted():
    env = SquadEnv(SquadConfig(obs_radius=3, n_snakes=1, food_per_colour=2))
    env.reset([0])
    # Snake 0 heading north (0) at (10, 10); food 2 cells ahead and 5 cells to its right
    obs = env.set_state(0, [[(10, 10), (10, 11)]], [0], foods=[(15, 10, 0), (10, 8, 0)])
    feats = centre_features(obs.centre)
    assert feats.shape == (1, 1, 7)
    np.testing.assert_allclose(feats[0, 0, :3], [2 / 21, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(feats[0, 0, 3:6], [0.0, 5 / 21, 1.0], atol=1e-6)
    assert feats[0, 0, 6] == pytest.approx(1.0)
    # Heading east (1): the food at (15, 10) is now 5 ahead, (10, 8) is 2 to the left
    obs = env.set_state(0, [[(10, 10), (9, 10)]], [1], foods=[(15, 10, 0), (10, 8, 0)])
    np.testing.assert_allclose(
        centre_features(obs.centre)[0, 0, :6],
        [0.0, -2 / 21, 1.0, 5 / 21, 0.0, 1.0],
        atol=1e-6,
    )


def test_specialist_inputs_one_hot_message():
    env = SquadEnv(SquadConfig(obs_radius=3))
    obs = env.reset([1, 2])
    obs.specialists["message"][:, :, 0] = np.array([[0, 3, 7], [1, 1, 2]])
    x = specialist_inputs(obs.specialists, 8)
    assert x.shape == (2, 3, specialist_input_size(3, 8))
    np.testing.assert_array_equal(x[:, :, -8:].argmax(axis=2), [[0, 3, 7], [1, 1, 2]])
    assert np.all(x[:, :, -8:].sum(axis=2) == 1)


# ------------------------------------------------------------------ rollouts
@pytest.mark.parametrize("mode", MODES)
def test_population_rollout_members_match_single_rollouts(mode):
    nets = Nets.build(TINY, mode)
    rng = np.random.default_rng(3)
    params = np.stack([nets.init(0) + 0.3 * rng.normal(size=nets.n_params)
                       for _ in range(3)])  # fmt: skip
    seeds = [11, 12]
    pop = population_rollout(TINY, nets, mode, params, seeds, noise_seed=5)
    assert pop["return"].shape == (3, 2)
    for p in range(3):
        one = population_rollout(TINY, nets, mode, params[p], seeds, noise_seed=5)
        np.testing.assert_allclose(one["return"][0], pop["return"][p])


def test_train_run_returns_curve_and_parameters():
    out = train_run(LEARNED, 0, TINY)
    nets = Nets.build(TINY, LEARNED)
    assert out["theta"].shape == (nets.n_params,)
    assert out["generations_done"] == 2
    assert [row["generation"] for row in out["curve"]] == [0, 1, 2]
    assert out["curve"][-1]["episodes"] == 2 * 4 * 2
    with pytest.raises(ValueError):
        train_run("telepathy", 0, TINY)


def test_conditions_share_the_initial_specialist():
    spec = {m: Nets.build(TINY, m).split(Nets.build(TINY, m).init(4))[0] for m in MODES}
    np.testing.assert_array_equal(spec[SILENT], spec[LEARNED])
    np.testing.assert_array_equal(spec[SILENT], spec[HAND_CODED])


def test_time_limit_stops_early():
    out = train_run(SILENT, 0, replace(TINY, generations=50, time_limit_s=0.0))
    assert out["generations_done"] == 1


# ------------------------------------------------------------------ policies
def test_learned_policies_are_consistent():
    nets = Nets.build(TINY, LEARNED)
    theta = nets.init(1)
    specialist, centre = policies_for(LEARNED, TINY, theta)
    assert isinstance(specialist, LearnedSpecialist)
    assert isinstance(centre, LearnedCentre)
    env = SquadEnv(TINY.squad_config())
    obs = env.reset([5, 6, 7])
    specialist.reset([5, 6, 7])
    probs = specialist.probs(obs)
    assert probs.shape == (3, 3, 3)
    np.testing.assert_allclose(probs.sum(axis=2), 1.0)
    acts = specialist.act(obs)
    assert acts.shape == (3, 3) and acts.min() >= 0 and acts.max() < 3
    codes = centre.send(obs)
    assert codes.shape == (3, 3, 1) and codes.min() >= 0 and codes.max() < 8
    assert isinstance(policies_for(HAND_CODED, TINY, theta[:-10])[1], HandCodedCentre)


def test_test_seeds_are_disjoint():
    assert not set(test_episode_seeds(0, 10)) & set(test_episode_seeds(1, 10))
    with pytest.raises(ValueError):
        test_episode_seeds(0, 0)


def test_welch_ci():
    assert np.isnan(welch_ci(np.array([1.0]), np.array([2.0, 3.0])))
    assert welch_ci(np.ones(3), np.ones(3)) == 0.0
    assert welch_ci(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 3.0])) > 0


# ------------------------------------------------------------------ end to end
def test_evaluate_radius_end_to_end():
    trained = [train_run(m, r, TINY) for m in MODES for r in (0, 1)]
    result = evaluate_radius(
        3, trained, TINY, EvalSettings(n_episodes=2, n_shuffles=5, n_boot=50)
    )
    assert result.complete
    names = {c.name for c in result.conditions}
    assert {CONDITION[m] for m in MODES} | {SCRIPTED, "oracle", "no-comm"} <= names
    # The silent system ignores every intervention: its code is constant
    silent = [a for a in result.ablations if a.system == CONDITION[SILENT]]
    assert silent and all(a.delta_mean == 0 for a in silent)
    assert result.communication[CONDITION[SILENT]]["message_entropy_bits"] == 0
    assert "cic_bits" in result.communication[CONDITION[LEARNED]]
    assert len(result.metadata["headline"]) == 5
    result.to_json()  # serialisable once the recorders are dropped


# ------------------------------------------------------------------ code-formation aids
def test_code_formation_aids_are_off_by_default_and_anneal():
    s = TrainSettings(generations=100)
    assert s.entropy_weight(0) == 0.0 and s.temperature(0) == 0.0
    s = replace(s, entropy_bonus=2.0, entropy_anneal=0.5, centre_temperature=1.0,
                temperature_anneal=0.25)  # fmt: skip
    assert s.entropy_weight(0) == 2.0
    assert s.entropy_weight(25) == pytest.approx(1.0)
    assert s.entropy_weight(50) == 0.0 and s.entropy_weight(99) == 0.0
    assert s.temperature(0) == 1.0 and s.temperature(25) == 0.0
    assert replace(s, entropy_anneal=0.0).entropy_weight(0) == 0.0


def test_entropy_bits():
    h = entropy_bits(np.array([[4, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [2, 2, 0, 0]]))
    np.testing.assert_allclose(h, [0.0, 2.0, 0.0, 1.0])


def test_rollout_reports_message_entropy():
    seeds = [11, 12]
    silent = Nets.build(TINY, SILENT)
    out = population_rollout(TINY, silent, SILENT, silent.init(0), seeds, 5)
    assert out["message_entropy"].shape == (1,) and out["message_entropy"][0] == 0.0
    nets = Nets.build(TINY, LEARNED)
    params = np.stack([nets.init(0), nets.init(0)])
    greedy = population_rollout(TINY, nets, LEARNED, params, seeds, 5)
    assert 0.0 <= greedy["message_entropy"][0] <= 3.0
    # Identical members see identical (common) centre noise
    hot = population_rollout(TINY, nets, LEARNED, params, seeds, 5, temperature=1e6)
    assert hot["message_entropy"][0] == hot["message_entropy"][1]
    assert hot["message_entropy"][0] > 2.5  # Near-uniform symbols at a huge T
    # T -> 0 is the argmax centre, and the centre's noise never touches the actions
    cold = population_rollout(TINY, nets, LEARNED, params, seeds, 5, temperature=1e-6)
    np.testing.assert_array_equal(cold["return"], greedy["return"])


def test_entropy_bonus_changes_training_only_when_on():
    base = train_run(LEARNED, 0, TINY)
    again = train_run(LEARNED, 0, replace(TINY, entropy_anneal=0.9))  # bonus still 0
    np.testing.assert_array_equal(base["theta_final"], again["theta_final"])
    bonus = train_run(LEARNED, 0, replace(TINY, entropy_bonus=100.0))
    assert not np.array_equal(base["theta_final"], bonus["theta_final"])
    hot = train_run(LEARNED, 0, replace(TINY, centre_temperature=5.0))
    assert "val_message_entropy" in hot["curve"][-1]
    # The aids apply to the learned centre only
    silent = train_run(SILENT, 0, replace(TINY, entropy_bonus=100.0,
                                          centre_temperature=5.0))  # fmt: skip
    np.testing.assert_array_equal(
        silent["theta_final"], train_run(SILENT, 0, TINY)["theta_final"]
    )


def test_report_driver_saves_runs_and_resumes(tmp_path):
    from genetic_coil_arena.experiments.learned_protocol_report import main

    argv = [
        "--scale", "smoke", "--runs", "1", "--modes", SILENT, "--generations", "1",
        "--population", "2", "--episodes-per-member", "1", "--eval-every", "1",
        "--eval-episodes", "2", "--workers", "1", "--cache", str(tmp_path / "cache"),
        "--out", str(tmp_path / "lp.md"),
    ]  # fmt: skip
    main(argv)
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == [
        "r3_silent_run0.json",
        "r3_silent_run0.npz",
    ]
    first = np.load(tmp_path / "lp_params.npz")["r3_silent_run0"]
    main(argv)  # Loaded from the cache: same parameters, no retraining
    data = json.loads((tmp_path / "lp.json").read_text(encoding="utf-8"))
    assert data["training"][0]["cached"] is True
    again = np.load(tmp_path / "lp_params.npz")["r3_silent_run0"]
    np.testing.assert_array_equal(again, first)
