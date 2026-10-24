"""Tests for the run-level replicates: temporal context, calibration on validation runs, tuning."""

import numpy as np
import pytest

from genetic_coil_arena.envs.plant import PlantConfig, PlantEnv, synthetic_tep
from genetic_coil_arena.envs.plant.centres import stack_context
from genetic_coil_arena.envs.plant.replicates import (
    DEFAULT_CENTRE,
    Protocol,
    Replicate,
    summarise,
)
from genetic_coil_arena.envs.plant.rieth import synthetic_rieth, to_runs
from genetic_coil_arena.envs.plant.runner import run_episodes
from genetic_coil_arena.envs.plant.tuning import tune_centre
from genetic_coil_arena.envs.plant.windows import build_runs, labelled_windows

FAULTS = (1, 2, 3)
SMALL = dict(  # noqa: C408
    faults=FAULTS,
    n_replicates=2,
    train_runs=3,
    val_runs=2,
    test_runs=2,
    train_normal_runs=3,
    val_normal_runs=2,
    test_normal_runs=2,
    n_available=12,
)
FAST = {"classifier": "logreg", "l2": 1e-2, "max_iter": 40, "context": 1}


@pytest.fixture(scope="module")
def data():
    return synthetic_rieth(n_runs=12, n_faults=3, seed=0, strength=5.0)


@pytest.fixture(scope="module")
def rep(data):
    return Replicate(data, Protocol(**SMALL), 0)


class TestStackContext:
    def test_newest_first_and_padded_at_the_start_of_each_run(self):
        x = np.arange(1.0, 8.0)[:, None]  # windows 1..7
        run = np.array([0, 0, 0, 0, 1, 1, 1])
        out = stack_context(x, run, 3)
        assert out.shape == (7, 3)
        assert out[3].tolist() == [4.0, 3.0, 2.0]
        assert out[0].tolist() == [1.0, 1.0, 1.0]  # earliest window repeated
        assert out[1].tolist() == [2.0, 1.0, 1.0]
        assert out[4].tolist() == [5.0, 5.0, 5.0]  # never reaches into the previous run
        assert out[6].tolist() == [7.0, 6.0, 5.0]

    def test_context_one_is_the_identity_and_zero_is_refused(self):
        x = np.ones((4, 2))
        assert stack_context(x, np.zeros(4), 1) is x
        with pytest.raises(ValueError, match="context"):
            stack_context(x, np.zeros(4), 0)


class TestEnvFromRuns:
    def test_runs_replace_the_braatz_data(self):
        tep = synthetic_tep(seed=3, faults=(1, 2))
        cfg = PlantConfig(budget=8, faults=(1, 2), split="test")
        reference = PlantEnv(cfg, tep)
        runs = build_runs(tep, reference.normaliser, "test", (1, 2))
        direct = PlantEnv(cfg, runs=runs)
        assert direct.n_runs == reference.n_runs == 3
        a = direct.reset(range(3)).specialists["window"]
        b = reference.reset(range(3)).specialists["window"]
        assert np.array_equal(a, b)
        with pytest.raises(ValueError, match="built from runs"):
            direct.labelled_windows("train")
        with pytest.raises(ValueError, match="at least one run"):
            PlantEnv(cfg, runs=[])

    def test_with_budget_shares_the_windows_and_swaps_the_channel(self):
        tep = synthetic_tep(seed=3, faults=(1,))
        env = PlantEnv(PlantConfig(budget=0, faults=(1,)), tep)
        other = env.with_budget(16)
        assert np.shares_memory(env._windows, other._windows)
        assert env.config.budget == 0 and other.config.budget == 16
        assert other.spec.message.payload_shape == (16,)
        assert other.spec.max_steps == env.spec.max_steps
        obs = other.reset([0, 1])
        assert obs.centre["uplink"].shape == (2, 4, 16)
        with pytest.raises(ValueError, match="budget"):
            env.with_budget(500)


class TestProtocol:
    def test_classes_and_splits(self):
        p = Protocol(**SMALL)
        assert p.n_classes == 4 and p.class_of_fault == {1: 1, 2: 2, 3: 3}
        assert p.splits().n_available == 12 and p.splits().faults == FAULTS

    def test_summarise_ignores_nan_and_handles_small_samples(self):
        mean, half = summarise([1.0, 3.0, float("nan")])
        assert mean == 2.0 and half > 0
        assert summarise([2.0])[0] == 2.0 and np.isnan(summarise([2.0])[1])
        assert np.isnan(summarise([float("nan")])[0])


class TestReplicate:
    def test_windows_come_from_disjoint_runs_with_the_right_labels(self, rep):
        assert rep.train["windows"].shape[1:] == (4, 18, 20)
        # training runs: fault from sample 20 -> only the first window of a fault run is normal
        fault_rows = rep.train["run"] >= 3  # runs 0..2 are fault-free
        first = np.r_[True, np.diff(rep.train["run"]) != 0]
        assert (rep.train["labels"][fault_rows & first] == 0).all()
        assert (rep.train["labels"][fault_rows & ~first] > 0).all()
        # validation runs: 15 normal windows then faulty, like the test runs
        v = rep.val
        n_normal = np.bincount(v["run"], weights=(v["labels"] == 0))
        assert n_normal[:2].tolist() == [95.0, 95.0] and set(n_normal[2:]) == {15.0}
        sel = rep.selection
        assert sel.keys("val").isdisjoint(sel.keys("test"))

    def test_the_tuning_stage_has_one_big_validation_set_and_no_test_runs(self, data):
        tune = Replicate(data, Protocol(**SMALL), None)
        plain = Replicate(data, Protocol(**SMALL), 0)
        assert len(tune.val_runs) == 2 * len(plain.val_runs)  # validation + test slices
        with pytest.raises(ValueError, match="no test runs"):
            tune.env(8)
        assert tune.selection.keys("val").isdisjoint(plain.selection.keys("val"))

    def test_online_context_equals_offline_stacking(self, rep, data):
        prep = rep.prepare("summary", 8, 0)
        centre = rep.fit_centre(prep, {**FAST, "context": 3})
        env = rep.env(8)
        res = run_episodes(env, prep.encoder, centre, range(env.n_runs))
        runs = to_runs(data, rep.normaliser, rep.selection, "test", FAULTS)
        test = labelled_windows(runs, rep.protocol.class_of_fault, rep.layout, 20, 10)
        margin, fault = centre.score_features(
            centre.extract(test["windows"]), test["run"]
        )
        assert np.allclose(res["margins"], margin.reshape(env.n_runs, -1), atol=1e-4)
        agree = res["fault_pred"] == fault.reshape(env.n_runs, -1)
        assert agree.mean() > 0.999  # ties in a float32 argmax may differ
        # a second batch starts with no history (the runner resets the centre)
        again = run_episodes(env, prep.encoder, centre, range(env.n_runs))
        assert np.allclose(again["margins"], res["margins"])

    def test_oracle_stacks_raw_features_the_same_way(self, rep, data):
        prep = rep.prepare("oracle", 0, 0)
        centre = rep.fit_centre(prep, {**FAST, "context": 2})
        env = rep.env(0)
        res = run_episodes(env, None, centre, range(env.n_runs))
        runs = to_runs(data, rep.normaliser, rep.selection, "test", FAULTS)
        test = labelled_windows(runs, rep.protocol.class_of_fault, rep.layout, 20, 10)
        margin, _ = centre.score_features(centre.extract(test["windows"]), test["run"])
        assert np.allclose(res["margins"], margin.reshape(env.n_runs, -1), atol=1e-4)


class TestCalibration:
    def test_threshold_gives_the_target_false_alarm_rate_on_validation_normals(
        self, rep
    ):
        prep = rep.prepare("summary", 8, 0)
        for target in (0.02, 0.05, 0.2):
            centre = rep.fit_centre(prep, FAST)
            centre.calibrate_features(
                {
                    "features": prep.x_val,
                    "labels": rep.val["labels"],
                    "run": rep.val["run"],
                },
                target,
            )
            margin, _ = centre.score_features(prep.x_val, rep.val["run"])
            normal = rep.val["labels"] == 0
            realised = float(np.mean(margin[normal] > centre.tau))
            assert abs(realised - target) <= 1.0 / normal.sum() + 1e-12

    def test_validation_runs_set_the_threshold_but_are_never_trained_on(self, rep):
        prep = rep.prepare("summary", 8, 0)
        cal_a = {
            "features": prep.x_val,
            "labels": rep.val["labels"],
            "run": rep.val["run"],
        }
        keep = rep.val["run"] < 6  # a different, smaller validation set
        cal_b = {k: v[keep] for k, v in cal_a.items()}
        a = prep.make_centre(FAST).fit_features(
            prep.x_train, rep.train["labels"], rep.train["run"], cal_a
        )
        b = prep.make_centre(FAST).fit_features(
            prep.x_train, rep.train["labels"], rep.train["run"], cal_b
        )
        assert np.array_equal(a.classifier._theta, b.classifier._theta)
        assert a.tau != b.tau

    def test_fit_without_validation_runs_still_calibrates_by_cross_validation(
        self, rep
    ):
        prep = rep.prepare("summary", 8, 0)
        centre = prep.make_centre(FAST).fit_features(
            prep.x_train, rep.train["labels"], rep.train["run"]
        )
        assert centre.tau != 0.0
        with pytest.raises(ValueError, match="run indices"):
            prep.make_centre(FAST).fit_features(prep.x_train, rep.train["labels"])

    def test_needs_normal_windows(self, rep):
        prep = rep.prepare("summary", 8, 0)
        centre = rep.fit_centre(prep, FAST)
        faulty = rep.val["labels"] > 0
        with pytest.raises(ValueError, match="normal windows"):
            centre.calibrate_features(
                {
                    "features": prep.x_val[faulty],
                    "labels": rep.val["labels"][faulty],
                    "run": rep.val["run"][faulty],
                },
                0.05,
            )

    def test_evaluate_reports_realised_against_target_false_alarms(self, rep):
        out = rep.evaluate("summary", 8, 0, FAST)
        assert abs(out["val_far"] - 0.05) < 0.01
        assert 0.0 <= out["false_alarm_rate"] <= 1.0
        assert out["false_alarm_rate_matched"] == pytest.approx(0.05, abs=0.01)
        for key in (
            "far_fault_free",
            "far_pre_onset",
            "macro_f1",
            "detection_delay",
            "val_f1",
            "tau",
        ):
            assert key in out
        assert out["macro_f1"] > 0.5 and out["detection_rate"] == 1.0
        assert out["n_test_runs"] == 3 * 2 + 2 and out["bytes_used"] <= 8.0
        assert len(out["f1_per_class"]) == 3 and out["replicate"] == 0


class TestCells:
    def test_silent_noise_and_oracle_cells(self, rep):
        silent = rep.evaluate("silent", 0, 0, {})
        assert silent["macro_f1"] == 0.0 and silent["detection_rate"] == 0.0
        assert silent["false_alarm_rate"] == 0.0 and np.isnan(silent["val_far"])
        noise = rep.evaluate("noise", 8, 0, FAST)
        assert noise["macro_f1"] < 0.3
        oracle = rep.evaluate("oracle", 0, 0, FAST)
        assert oracle["budget"] == 0 and oracle["macro_f1"] > 0.5

    def test_unknown_encodings_and_zero_budgets_are_refused(self, rep):
        with pytest.raises(ValueError, match="Unknown encoding"):
            rep.prepare("telepathy", 8, 0)
        with pytest.raises(ValueError, match="budget"):
            rep.prepare("summary", 0, 0)

    def test_budget_zero_environment_has_no_uplink(self, rep):
        env = rep.env(0)
        assert env.reset([0]).centre["uplink"].shape == (1, 4, 0)


class _FakeReplicate:
    """Scores a candidate by a table so the search logic can be checked exactly."""

    def __init__(self, table):
        self.table, self.fits = table, []

    def fit_centre(self, prep, params):
        self.fits.append(dict(params))
        return dict(params)

    def validation_f1(self, centre, prep):
        return self.table(centre)


class TestTuning:
    def test_picks_the_best_and_the_cheapest_within_the_tolerance(self):
        def score(p):
            return 0.9 + 0.01 * p["context"] + (0.0005 if p["hidden"] == 128 else 0.0)

        fake = _FakeReplicate(score)
        stages = (("context", (1, 2, 3)), ("hidden", (32, 64, 128)))
        best, trace = tune_centre(
            fake, None, start={"context": 1, "hidden": 64}, stages=stages, tol=0.002
        )
        assert best == {
            "context": 3,
            "hidden": 32,
        }  # 128 is only 0.0005 better: the cheapest tie wins
        assert [t["stage"] for t in trace] == ["context"] * 3 + ["hidden"] * 2
        assert len(fake.fits) == 5  # the incumbent is never refitted
        assert trace[0]["params"] == {"context": 1, "hidden": 64}

    def test_a_clear_win_beats_the_cheap_candidates(self):
        fake = _FakeReplicate(lambda p: 0.8 if p["l2"] != 1e-2 else 0.9)
        best, _ = tune_centre(
            fake, None, start={"l2": 1e-3}, stages=(("l2", (1e-4, 1e-3, 1e-2)),)
        )
        assert best["l2"] == 1e-2

    def test_runs_on_a_real_replicate_without_touching_its_test_runs(self, data):
        tune = Replicate(data, Protocol(**SMALL), None)
        prep = tune.prepare("summary", 8, 0)
        best, trace = tune_centre(tune, prep, start=FAST, stages=(("context", (1, 2)),))
        assert best["context"] in (1, 2) and len(trace) == 2
        assert all(0.0 <= t["val_f1"] <= 1.0 for t in trace)
        assert DEFAULT_CENTRE["precision"] == "float32"
