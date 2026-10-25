"""Tests for R7b: the goal-conditioned learned centre (goal G7)."""

import json

import numpy as np
import pytest

from genetic_coil_arena.envs.squad import SquadConfig, SquadEnv
from genetic_coil_arena.experiments import r7b_report
from genetic_coil_arena.experiments.r7a_domains import build_registry
from genetic_coil_arena.experiments.r7b_goal_conditioned import (
    BLIND,
    BOTH_GOAL,
    CENTRE_GOAL,
    CONDITION,
    DEATH_PENALTY,
    EXTRA_DIM,
    HAND_GOAL,
    HELDOUT,
    MODES,
    NON_TARGET_PENALTY,
    NOVEL_COMBO,
    NOVEL_SUBSET,
    SEEN,
    SPEC_DIM,
    Goal,
    GoalArrays,
    GoalNets,
    GoalSystem,
    GoalTrainSettings,
    ScriptedGoalSpecialist,
    all_goals,
    goal_message_metrics,
    goal_population_rollout,
    goal_sets,
    hand_goal_codes,
    make_goal_spec,
    mission_score,
    policies_for,
    split_goals,
    swapped_goals,
    train_run,
)
from genetic_coil_arena.protocol.goalspec import SpecValidationError

TINY = GoalTrainSettings(
    generations=2,
    population=4,
    eval_every=1,
    val_reps=1,
    max_steps=40,
)


# ------------------------------------------------------------------ goals and spec
def test_spec_vector_is_the_documented_fixed_encoding():
    g = Goal(("blue", "red"), "high")
    assert g.eaters == ("red", "blue")  # colour order
    assert g.spec_vector().tolist() == [1, 0, 1, 0, 0, 1]
    assert Goal(("green",), "low").spec_vector().tolist() == [0, 1, 0, 1, 0, 0]
    assert g.spec_vector().dtype == np.float32 and SPEC_DIM == 6


def test_goal_space_has_21_goals_all_validating_through_the_registry():
    registry = build_registry()
    goals = all_goals()
    assert len(goals) == len(set(goals)) == 21
    for g in goals:
        spec = make_goal_spec(g, registry)
        assert Goal.from_spec(spec) == g
    with pytest.raises(SpecValidationError):
        registry.make("squad.mission", {**goals[0].params(), "caution": "reckless"})


def test_invalid_goals_and_non_default_specs_are_refused():
    with pytest.raises(ValueError):
        Goal((), "normal")
    with pytest.raises(ValueError):
        Goal(("purple",), "normal")
    with pytest.raises(ValueError):
        Goal(("red",), "extreme")
    registry = build_registry()
    spec = registry.make(
        "squad.mission",
        {"eaters": ["red"], "caution": "low", "step_limit": 120, "food_target": 5},
    )
    with pytest.raises(ValueError):
        Goal.from_spec(spec)


def test_heldout_split_is_disjoint_complete_and_compositional():
    s = split_goals()
    train, subset, combo = s["train"], s[NOVEL_SUBSET], s[NOVEL_COMBO]
    assert (len(train), len(subset), len(combo)) == (15, 3, 3)
    assert set(train) | set(subset) | set(combo) == set(all_goals())
    assert not set(train) & (set(subset) | set(combo))
    assert {g.eaters for g in subset} == {("green", "blue")}
    assert ("green", "blue") not in {g.eaters for g in train}
    # novel combos: every eater subset and every caution appears in training
    for g in combo:
        assert g.eaters in {t.eaters for t in train}
        assert g.caution in {t.caution for t in train}
    # every colour is an eater and a non-eater, and every caution occurs, in training
    for i in range(3):
        assert any(g.mask[i] for g in train) and any(not g.mask[i] for g in train)
    assert {g.caution for g in train} == {"low", "normal", "high"}
    sets = goal_sets()
    assert sets[SEEN] == train and sets[HELDOUT] == subset + combo


# ------------------------------------------------------------------ mission score
def test_mission_score_matches_its_documented_definition():
    arrays = GoalArrays.from_goals(
        [
            Goal(("red",), "normal"),
            Goal(("red", "green"), "high"),
            Goal(("blue",), "low"),
        ]
    )
    food = np.array([[5, 3, 2], [4, 2, 7], [1, 1, 6]])
    alive = np.array([[True, False, False], [True, False, True], [False, False, False]])
    got = mission_score(food, alive, arrays)
    assert NON_TARGET_PENALTY == 1.0
    assert DEATH_PENALTY == {"low": 0.0, "normal": 2.0, "high": 5.0}
    # 5 - (3 + 2) - 0 dead eaters; 4 + 2 - 7 - 5 * 1; 6 - 2 - 0 * 1
    np.testing.assert_allclose(got, [0.0, -6.0, 4.0])


def test_goal_arrays_tile_and_extra():
    goals = [Goal(("red",), "low"), Goal(("green", "blue"), "high")]
    arrays = GoalArrays.from_goals(goals)
    extra = arrays.extra()
    assert extra.shape == (2, 3, EXTRA_DIM)
    assert extra[0, :, 0].tolist() == [1, 0, 0]  # own_eater per snake
    assert extra[1, :, 0].tolist() == [0, 1, 1]
    np.testing.assert_array_equal(extra[1, 2, 1:], goals[1].spec_vector())
    tiled = arrays.tile(3)
    assert tiled.eaters.shape == (6, 3)
    np.testing.assert_array_equal(tiled.eaters[2:4], arrays.eaters)


def test_swapped_goals_change_the_eater_set():
    goals = goal_sets()[HELDOUT]
    for orig, new in zip(goals, swapped_goals(goals)):
        assert new.eaters != orig.eaters
    same = [Goal(("red",), c) for c in ("low", "high")]
    assert [g.caution for g in swapped_goals(same)] == ["high", "low"]


# ------------------------------------------------------------------ conditions
def test_hand_goal_code_restricts_the_r6_code_to_target_colours():
    env = SquadEnv(SquadConfig(obs_radius=3, max_steps=20))
    obs = env.reset(np.arange(6))
    eaters = np.tile(np.array([[True, False, True]]), (6, 1))
    codes = hand_goal_codes(obs, eaters)
    assert codes.shape == (6, 3, 1) and codes.dtype == np.int64
    assert (codes[:, 1] == 0).all()  # non-target: "do not eat"
    assert (codes[:, [0, 2]] >= 1).all() and (codes <= 7).all()


def test_network_inputs_follow_the_condition():
    n = {m: GoalNets.build(TINY, m) for m in MODES}
    base = n[BLIND]
    assert not base.spec_goal and not base.centre_goal
    assert n[CENTRE_GOAL].centre_goal and not n[CENTRE_GOAL].spec_goal
    assert n[BOTH_GOAL].centre_goal and n[BOTH_GOAL].spec_goal
    assert n[HAND_GOAL].centre is None and not n[HAND_GOAL].spec_goal
    def sizes(net):
        return net.spec.sizes[0], net.centre.sizes[0] if net.centre else 0

    (sb, cb), (sc, cc), (sw, cw) = (
        sizes(n[m]) for m in (BLIND, CENTRE_GOAL, BOTH_GOAL)
    )
    assert (sc, cc) == (sb, cb + EXTRA_DIM)  # b: the goal goes to the centre only
    assert (sw, cw) == (sb + EXTRA_DIM, cb + EXTRA_DIM)  # c: to both


@pytest.mark.parametrize("mode", MODES)
def test_train_run_and_policies_at_tiny_scale(mode):
    out = train_run(mode, 0, TINY)
    nets = GoalNets.build(TINY, mode)
    assert out["theta"].shape == (nets.n_params,)
    assert out["generations_done"] == 2 and len(out["curve"]) == 3
    assert np.isfinite(out["curve"][-1]["val_mission"])
    goals = goal_sets()[SEEN][:4]
    system = GoalSystem(
        TINY.squad_config(), goals, lambda s: policies_for(mode, TINY, out["theta"])
    )
    res = system(list(range(8)), seed=0)
    assert res["mission"].shape == (8,) and np.isfinite(res["mission"]).all()
    assert res["goal_index"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_goal_aware_policies_refuse_to_run_without_goals():
    specialist, centre = policies_for(
        CENTRE_GOAL, TINY, GoalNets.build(TINY, CENTRE_GOAL).init(0)
    )
    obs = SquadEnv(TINY.squad_config()).reset(np.arange(2))
    with pytest.raises(RuntimeError):
        centre.send(obs)


def test_population_rollout_scores_the_goal_it_plays():
    nets = GoalNets.build(TINY, BLIND)
    theta = nets.init(0)
    goals = goal_sets()[SEEN][:3]
    out = goal_population_rollout(TINY, nets, BLIND, theta, [1, 2, 3], goals, 0)
    assert out["mission"].shape == (1, 3)
    again = goal_population_rollout(TINY, nets, BLIND, theta, [1, 2, 3], goals, 0)
    np.testing.assert_array_equal(out["mission"], again["mission"])  # CRN
    with pytest.raises(ValueError):
        goal_population_rollout(TINY, nets, BLIND, theta, [1, 2], goals, 0)


def test_goal_swap_changes_what_policies_are_told_not_the_score():
    told = []

    class Spy(ScriptedGoalSpecialist):
        def act(self, obs):
            told.append(self.arrays)
            return super().act(obs)

    goals = goal_sets()[HELDOUT]
    cfg = TINY.squad_config()
    from genetic_coil_arena.envs.squad import SilentCentre

    for swap in (False, True):
        told.clear()
        system = GoalSystem(cfg, goals, lambda s: (Spy(), SilentCentre()), swap=swap)
        out = system(list(range(len(goals))), seed=0)
        true = GoalArrays.from_goals(goals)
        np.testing.assert_array_equal(out["arrays"].eaters, true.eaters)
        same = np.array_equal(told[0].eaters, true.eaters)
        assert same is (not swap)


def test_scripted_wrapper_blanks_non_eaters_and_follows_caution():
    cfg = SquadConfig(obs_radius=3, max_steps=60)
    goals = [Goal(("red",), "normal"), Goal(("green", "blue"), "high")]
    system = GoalSystem(cfg, goals, lambda s: (ScriptedGoalSpecialist(), _silent()))
    a = system([5, 6], seed=0)
    b = system([5, 6], seed=0)
    np.testing.assert_array_equal(a["mission"], b["mission"])
    assert a["mission"].shape == (2,)


def _silent():
    from genetic_coil_arena.envs.squad import SilentCentre

    return SilentCentre()


# ------------------------------------------------------------------ message metrics
def _recorded(code_of_eater: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    goals = goal_sets()[HELDOUT]
    arrays = GoalArrays.from_goals(goals * 20)
    n_ep = len(goals) * 20
    eps = np.repeat(np.arange(n_ep), 3 * 10)
    recv = np.tile(np.repeat(np.arange(3), 10), n_ep)
    if code_of_eater:
        msgs = arrays.eaters[eps, recv].astype(np.int64) * 3 + 1
    else:
        msgs = rng.integers(0, 8, eps.size)
    return {
        "recorder": {"messages": msgs, "episode": eps, "receiver": recv},
        "arrays": arrays,
        "goal_index": np.arange(n_ep) % len(goals),
    }


def test_message_metrics_detect_a_goal_carrying_and_a_goal_blind_code():
    carrying = goal_message_metrics(_recorded(True), n_shuffles=50)
    blind = goal_message_metrics(_recorded(False), n_shuffles=50)
    assert carrying["mi_own_eater_bits"] > 0.9
    assert carrying["probe_own_eater_acc"] > 0.99
    assert blind["mi_own_eater_bits"] < 0.02
    assert blind["mi_own_eater_bits"] <= blind["mi_own_eater_null_q95_bits"] + 0.01
    assert carrying["message_entropy_bits"] == pytest.approx(1.0, abs=0.1)


# ------------------------------------------------------------------ G7 verdict
def _held(a, b, c, d, spread=0.1, n=8, seed=0):
    rng = np.random.default_rng(seed)
    names = [CONDITION[m] for m in (BLIND, CENTRE_GOAL, BOTH_GOAL, HAND_GOAL)]
    return {
        name: list(mean + spread * rng.standard_normal(n))
        for name, mean in zip(names, (a, b, c, d))
    }


def test_g7_pass_when_a_goal_conditioned_centre_beats_blind_and_reaches_90pct_of_hand():
    v = r7b_report.g7_verdict(_held(2.0, 8.5, 2.0, 9.0))
    assert v["verdict"] == "pass"
    row = v["rows"][0]
    assert row["passes"] and row["fraction_of_hand"] > 0.9
    assert not v["rows"][1]["beats_blind"]


def test_g7_kill_when_neither_beats_blind():
    v = r7b_report.g7_verdict(_held(2.0, 2.05, 1.9, 9.0, spread=0.5))
    assert v["verdict"] == "kill"
    assert not any(r["beats_blind"] for r in v["rows"])


def test_g7_partial_when_better_than_blind_but_below_90pct_of_hand():
    v = r7b_report.g7_verdict(_held(2.0, 5.0, 2.0, 9.0))
    assert v["rows"][0]["beats_blind"] and not v["rows"][0]["reaches_90pct_of_hand"]
    assert v["verdict"] == "partial"


def test_g7_cannot_pass_when_the_hand_reference_is_not_positive():
    v = r7b_report.g7_verdict(_held(-4.0, -1.0, -3.0, -0.5))
    assert v["verdict"] != "pass"
    assert np.isnan(v["rows"][0]["fraction_of_hand"])


# ------------------------------------------------------------------ pipeline
def test_cli_smoke_trains_reads_back_and_reports(tmp_path):
    cache = tmp_path / "runs"
    argv = ["--scale", "smoke", "--train-only", "--workers", "1", "--cache", str(cache)]
    r7b_report.main(argv + ["--modes", "blind", "hand-goal", "--runs", "0:2"])
    r7b_report.main(argv + ["--modes", "centre-goal", "both-goal", "--runs", "0:2"])
    trained, settings = r7b_report.read_trained([cache])
    assert len(trained) == 8
    out = tmp_path / "report.md"
    payload = r7b_report.evaluate_and_report(
        trained, settings, r7b_report.SCALES["smoke"]["eval"], out, "smoke"
    )
    assert payload["g7"]["verdict"] in {"pass", "kill", "partial"}
    text = out.read_text(encoding="utf-8")
    assert "G7 verdict" in text and "Do the messages carry the goal" in text
    saved = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert set(saved["sets"]) == {SEEN, HELDOUT}
    held = saved["sets"][HELDOUT]["result"]
    names = {c["name"] for c in held["conditions"]}
    assert {CONDITION[m] for m in MODES} <= names
    assert any("scripted" in n for n in names)  # mandatory baselines ran
