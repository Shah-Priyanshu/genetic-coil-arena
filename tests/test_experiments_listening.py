"""Test the squad listening experiment: the squad wired into the evaluation harness."""

import numpy as np
import pytest

from genetic_coil_arena.envs.squad import (
    GreedyPolicy,
    HandCodedCentre,
    NoCommPolicy,
    SilentCentre,
    SquadConfig,
    SquadEnv,
    run_episodes,
)
from genetic_coil_arena.evaluation import BaselineRegistry, InterventionHook
from genetic_coil_arena.experiments.listening import (
    DEAF,
    LISTENER,
    NO_FOOD,
    ListeningSettings,
    Recorder,
    SquadSystem,
    episode_seeds,
    food_targets,
    listener_metrics,
    make_task,
    register_squad_baselines,
    rollout,
    run_radius,
    sanity_checks,
)

CONFIG = SquadConfig(obs_radius=3, max_steps=60)


def test_episode_seeds_are_disjoint_per_run_seed():
    a, b = episode_seeds(0, 5), episode_seeds(1, 5)
    assert len(a) == 5 and not set(a) & set(b)
    with pytest.raises(ValueError):
        episode_seeds(0, 0)


def test_food_targets_match_hand_coded_centre():
    env = SquadEnv(CONFIG)
    obs = env.reset(range(16))
    centre = HandCodedCentre()
    gen = np.random.default_rng(0)
    checked = 0
    for _ in range(20):
        sent = centre.send(obs)[:, :, 0]
        targets = food_targets(obs.centre)
        valid = targets["valid"]
        assert np.array_equal(sent[valid], targets["label"][valid])
        assert np.all(targets["label"][~valid] == NO_FOOD)
        checked += int(valid.sum())
        obs = env.step(gen.integers(0, 3, size=(16, CONFIG.n_snakes))).obs
    assert checked > 0


def test_rollout_without_hook_matches_run_episodes():
    seeds = list(range(8))
    ours = rollout(SquadEnv(CONFIG), GreedyPolicy(), HandCodedCentre(), seeds)
    ref = run_episodes(SquadEnv(CONFIG), GreedyPolicy(), HandCodedCentre(), seeds)
    assert np.array_equal(ours["team_food"], ref["team_food"])


def test_zero_hook_equals_silent_centre():
    seeds = list(range(8))
    system = SquadSystem(CONFIG, GreedyPolicy(), HandCodedCentre())
    zeroed = system(seeds, seed=0, hook=InterventionHook("zero"))
    silent = SquadSystem(CONFIG, GreedyPolicy(), SilentCentre())(seeds, seed=0)
    assert np.array_equal(zeroed["team_food"], silent["team_food"])


def test_hook_does_not_affect_deaf_specialists():
    seeds = list(range(8))
    system = SquadSystem(CONFIG, NoCommPolicy(), HandCodedCentre())
    intact = system(seeds, seed=0)
    shuffled = system(seeds, seed=0, hook=InterventionHook("shuffle_batch", seed=3))
    assert np.array_equal(intact["team_food"], shuffled["team_food"])


def test_baselines_are_registered():
    registry = BaselineRegistry()
    names = register_squad_baselines("squad-test", CONFIG, registry)
    assert set(names) == set(registry.required("squad-test"))
    assert {"random-actions", "no-comm", "oracle"} <= set(names)
    task = make_task(3, n_seeds=2, n_episodes=3, registry=registry)
    assert task.inputs(1) == episode_seeds(1, 3)
    assert registry.required(task.name)


def test_listener_metrics_separate_listener_from_deaf():
    seeds = list(range(6))
    results = {}
    for name, policy in (("greedy", GreedyPolicy()), ("deaf", NoCommPolicy())):
        recorder = Recorder(record_every=2, snapshot_every=10)
        SquadSystem(CONFIG, policy, HandCodedCentre())(seeds, recorder=recorder)
        results[name] = listener_metrics(policy, recorder, n_draws=4)
    assert results["greedy"]["cic_bits"] > 0.05
    assert results["deaf"]["cic_bits"] == pytest.approx(0.0, abs=1e-12)
    assert results["deaf"]["js_shuffle_batch_bits"] == pytest.approx(0.0, abs=1e-12)


def test_run_radius_end_to_end_passes_sanity_checks():
    settings = ListeningSettings(
        radii=(3,), n_seeds=3, n_episodes=6, n_shuffles=20, n_boot=100
    )
    result = run_radius(3, settings)
    assert result.complete
    assert {a.intervention for a in result.ablations if a.system == LISTENER} == set(
        settings.interventions
    )
    assert all(a.delta_mean == 0 for a in result.ablations if a.system == DEAF)
    for row in sanity_checks(result):
        if "random centre" in row["check"]:
            continue  # Too few messages here for a reliable MI null
        assert row["passed"], row
    assert result.metadata["listening_summary"][LISTENER]["positive_listening"]
    assert not result.metadata["listening_summary"][DEAF]["positive_listening"]
