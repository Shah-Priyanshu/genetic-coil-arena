"""Tests for the "add a specialist" experiment (R7) and the per-run cache."""

import json
from dataclasses import replace

import numpy as np
import pytest

from genetic_coil_arena.envs.squad import (
    HandCodedCentre,
    RandomPolicy,
    SquadEnv,
    run_episodes,
)
from genetic_coil_arena.evaluation import InterventionHook
from genetic_coil_arena.experiments import run_cache
from genetic_coil_arena.experiments.add_specialist import (
    COPY,
    HAND_CODED,
    JOINT,
    LEARNED,
    SCRATCH,
    SILENT,
    SOURCE_MODES,
    AddSettings,
    ComposedSpecialist,
    generations_to,
    joint_layout,
    newcomer_hook,
    newcomer_layout,
    population_rollout,
    source_key,
    source_policies,
    train_run,
    trained_policies,
)
from genetic_coil_arena.experiments.add_specialist_report import main
from genetic_coil_arena.experiments.learned_protocol import (
    LearnedCentre,
    Nets,
    centre_features,
)

TINY = AddSettings(
    generations=2,
    population=4,
    episodes_per_member=2,
    eval_every=1,
    val_episodes=2,
    max_steps=40,
)


def source(mode: str, run: int = 0, noise: float = 0.5) -> np.ndarray:
    """A stand-in for a trained R6 parameter vector of ``mode``."""
    nets = Nets.build(TINY.source_settings(), mode)
    theta = nets.init(run + 100)
    return theta + noise * np.random.default_rng(run).normal(size=theta.size)


# ------------------------------------------------------------------ environment
def test_four_snake_env_has_a_fourth_colour():
    env = SquadEnv(TINY.squad_config())
    assert env.config.n_snakes == 4
    obs = env.reset([1, 2, 3])
    assert obs.specialists["window"].shape[:2] == (3, 4)
    assert obs.centre["food"].shape == (3, 8, 2)
    assert {int(c) for c in env.foods(0)[:, 2]} == {0, 1, 2, 3}
    assert centre_features(obs.centre).shape == (3, 4, 7)
    env.step(np.zeros((3, 4), dtype=np.int64), np.zeros((3, 4, 1), dtype=np.int64))


LAYOUT = {
    "bodies": [
        [(3, 3), (3, 4), (3, 5)],
        [(10, 10), (11, 10), (12, 10)],
        [(17, 5), (17, 4), (17, 3)],
        [(6, 15), (5, 15), (4, 15)],
    ],
    "headings": [0, 3, 2, 1],
    "health": [60, 12, 35, 5],
    "foods": [(3, 0, 0), (8, 3, 0), (10, 18, 1), (2, 10, 1), (17, 12, 2), (12, 2, 2),
              (9, 15, 3), (6, 19, 3)],  # fmt: skip
}


def _codes(centre, order):
    """Codes sent when the snakes of LAYOUT are listed in ``order``."""
    n = len(order)
    env = SquadEnv(replace(TINY, n_old=n - 1).squad_config())
    env.reset([0])
    colour = {old: new for new, old in enumerate(order)}
    obs = env.set_state(
        0,
        [LAYOUT["bodies"][i] for i in order],
        [LAYOUT["headings"][i] for i in order],
        [(x, y, colour[c]) for x, y, c in LAYOUT["foods"] if c in colour],
        [LAYOUT["health"][i] for i in order],
    )
    return centre.send(obs)[0, :, 0]


@pytest.mark.parametrize("kind", ["learned", "hand-coded"])
def test_frozen_centre_addresses_the_newcomer_like_any_receiver(kind):
    if kind == "learned":
        nets = Nets.build(TINY.source_settings(), LEARNED)
        params = nets.centre.init(np.random.default_rng(0), scale=3.0)
        w1 = params[: 7 * 16].reshape(7, 16)  # (inputs, hidden), a view
        w1[[2, 5]] = 0.0  # Drop the valid flags so the offsets decide the code
        centre = LearnedCentre(nets.centre, params)
    else:
        centre = HandCodedCentre()
    base = _codes(centre, [0, 1, 2, 3])
    # Receiver-agnostic: permuting snakes (with their colours) permutes the codes
    order = [3, 1, 0, 2]
    np.testing.assert_array_equal(_codes(centre, order), base[order])
    # The newcomer does not change what the old snakes are told
    np.testing.assert_array_equal(_codes(centre, [0, 1, 2]), base[:3])
    if kind == "learned":
        assert len(set(base.tolist())) > 1  # The test layout is informative


# ------------------------------------------------------------------ layouts
def test_newcomer_layout_freezes_everything_but_the_newcomer():
    theta = source(LEARNED)
    layout = newcomer_layout(TINY, LEARNED, theta, run=0)
    k = layout.nets.spec.n_params
    np.testing.assert_array_equal(layout.theta0[: theta.size], theta)
    assert layout.mask[: theta.size].sum() == 0 and layout.mask[theta.size :].all()
    assert layout.n_trainable == k
    assert len(layout.groups()) == 2 and layout.groups()[0][1] == [0, 1, 2]
    copy = newcomer_layout(TINY, LEARNED, theta, run=0, init=COPY)
    np.testing.assert_array_equal(copy.theta0[-k:], theta[:k])
    silent = newcomer_layout(TINY, SILENT, source(SILENT), run=0)
    assert silent.centre_block is None
    with pytest.raises(ValueError):
        newcomer_layout(TINY, LEARNED, theta[:-1], run=0)
    with pytest.raises(ValueError):
        newcomer_layout(TINY, JOINT, theta, run=0)


def test_joint_layout_trains_one_shared_network_and_the_centre():
    layout = joint_layout(TINY, run=2)
    assert layout.mask.all() and len(layout.groups()) == 1
    np.testing.assert_array_equal(
        layout.theta0, Nets.build(TINY.source_settings(), LEARNED).init(2)
    )


@pytest.mark.parametrize("mode", SOURCE_MODES)
def test_population_rollout_members_match_single_rollouts(mode):
    layout = newcomer_layout(TINY, mode, source(mode), run=0)
    rng = np.random.default_rng(3)
    params = np.stack(
        [layout.theta0 + 0.3 * rng.normal(size=layout.n_params) * layout.mask
         for _ in range(3)]
    )  # fmt: skip
    pop = population_rollout(TINY, layout, params, [11, 12], noise_seed=5)
    assert pop["food"].shape == (3, 2, 4)
    for p in range(3):
        one = population_rollout(TINY, layout, params[p], [11, 12], noise_seed=5)
        np.testing.assert_allclose(one["return"][0], pop["return"][p])
        np.testing.assert_allclose(one["food"][0], pop["food"][p])


# ------------------------------------------------------------------ training
@pytest.mark.parametrize("mode", SOURCE_MODES)
def test_frozen_parameters_never_change(mode):
    theta = source(mode)
    out = train_run(mode, 0, TINY, theta)
    k = Nets.build(TINY.source_settings(), mode).spec.n_params
    for vec in (out["theta"], out["theta_final"]):
        np.testing.assert_array_equal(vec[: theta.size], theta)
    start = newcomer_layout(TINY, mode, theta, run=0).theta0[-k:]
    assert not np.array_equal(out["theta_final"][-k:], start)
    assert out["n_trainable"] == k
    assert [row["generation"] for row in out["curve"]] == [0, 1, 2]
    assert {"val_newcomer_food", "val_old_food", "val_per_snake_food"} <= set(
        out["curve"][0]
    )


def test_joint_training_changes_everything_and_needs_no_source():
    out = train_run(JOINT, 1, TINY)
    start = joint_layout(TINY, 1).theta0
    nets = Nets.build(TINY.source_settings(), LEARNED)
    k = nets.spec.n_params
    assert not np.array_equal(out["theta_final"][:k], start[:k])
    assert not np.array_equal(out["theta_final"][k:], start[k:])
    with pytest.raises(ValueError):
        train_run(LEARNED, 0, TINY)  # A newcomer needs its source squad
    with pytest.raises(ValueError):
        train_run("telepathy", 0, TINY, source(LEARNED))


def test_copied_newcomer_is_a_clone_and_policies_compose():
    theta = source(HAND_CODED)
    layout = newcomer_layout(TINY, HAND_CODED, theta, run=0, init=COPY)
    spec, centre = trained_policies(TINY, HAND_CODED, layout.theta0)
    old, _ = source_policies(TINY, HAND_CODED, theta)
    assert isinstance(spec, ComposedSpecialist)
    assert isinstance(centre, HandCodedCentre)
    env = SquadEnv(TINY.squad_config())
    obs = env.reset([4, 5])
    np.testing.assert_allclose(spec.probs(obs), old.probs(obs))
    mixed = ComposedSpecialist(old, RandomPolicy(), 3)
    probs = mixed.probs(obs)
    np.testing.assert_allclose(probs[:, :3], old.probs(obs)[:, :3])
    np.testing.assert_allclose(probs[:, 3], 1 / 3)
    out = run_episodes(env, mixed, centre, [4, 5])
    assert out["food"].shape == (2, 4)


def test_newcomer_hook_only_touches_the_newcomer():
    msgs = np.arange(1, 13, dtype=np.int64).reshape(3, 4, 1)
    hook = newcomer_hook(InterventionHook("zero", seed=0), newcomer=3)
    out = hook(msgs, 0, np.ones((3, 4), dtype=bool))
    np.testing.assert_array_equal(out[:, :3], msgs[:, :3])
    assert (out[:, 3] == 0).all()
    dead = np.ones((3, 4), dtype=bool)
    dead[1, 3] = False
    out = hook(msgs, 0, dead)
    assert out[1, 3, 0] == msgs[1, 3, 0] and out[0, 3, 0] == 0
    assert newcomer_hook(None, 3) is None


def test_generations_to_smooths_and_reports_never():
    curve = [
        {"generation": g, "episodes": 10 * g, "wall_s": g, "food": v}
        for g, v in enumerate([0.0, 5.0, 0.0, 4.0, 5.0, 6.0])
    ]
    hit = generations_to(curve, "food", 4.0, smooth=3)
    assert hit == {"generation": 5.0, "episodes": 50.0, "wall_s": 5.0}
    assert generations_to(curve, "food", 5.0, smooth=1)["generation"] == 1
    assert generations_to(curve, "food", 0.0)["generation"] == 0
    assert generations_to(curve, "food", 99.0) is None


# ------------------------------------------------------------------ cache + driver
def test_run_cache_round_trip_and_settings_check(tmp_path):
    settings = {"a": 1, "limit": float("inf")}
    assert run_cache.load(tmp_path, "k", settings) is None
    run_cache.save(tmp_path, "k", settings, {"theta": np.arange(3.0)}, {"run": 2})
    hit = run_cache.load(tmp_path, "k", settings)
    assert hit["run"] == 2 and hit["cached"]
    np.testing.assert_array_equal(hit["theta"], np.arange(3.0))
    assert run_cache.load(tmp_path, "k", {**settings, "a": 2}) is None
    assert not list(tmp_path.glob("*.tmp*"))


def test_driver_end_to_end_and_resumes_from_cache(tmp_path):
    src = tmp_path / "params.npz"
    np.savez(
        src,
        **{source_key(3, m, r): source(m, r).astype(np.float32)
           for m in SOURCE_MODES for r in range(2)},
    )  # fmt: skip
    argv = [
        "--scale", "smoke", "--runs", "2", "--inits", SCRATCH, "--workers", "1",
        "--population", "4", "--episodes-per-member", "1", "--generations", "2",
        "--joint-generations", "2", "--val-episodes", "2", "--eval-episodes", "3",
        "--source", str(src), "--cache", str(tmp_path / "cache"),
        "--out", str(tmp_path / "report" / "add.md"),
    ]  # fmt: skip
    main(argv)
    text = (tmp_path / "report" / "add.md").read_text(encoding="utf-8")
    assert "Kill criterion" in text and "(1) learned+newcomer[scratch]" in text
    data = json.loads((tmp_path / "report" / "add.json").read_text(encoding="utf-8"))
    assert {r["condition"] for r in data["analysis"]["rows"]} == {
        LEARNED,
        HAND_CODED,
        SILENT,
        JOINT,
    }
    assert len(data["old_food"]) == 3
    assert len(list((tmp_path / "cache").glob("*.json"))) == 8
    assert not any(t.get("cached") for t in data["training"])
    main(argv)  # Everything is loaded from the cache this time
    data = json.loads((tmp_path / "report" / "add.json").read_text(encoding="utf-8"))
    assert len(data["training"]) == 8
    assert all(t["cached"] for t in data["training"])
