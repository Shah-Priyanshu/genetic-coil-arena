"""Tests for the centre classifiers, rollouts, metrics and the baseline/harness wiring."""

import subprocess
import sys

import numpy as np
import pytest

from genetic_coil_arena.envs.plant import (
    DecodedCentre,
    MLPClassifier,
    OracleCentre,
    PlantConfig,
    PlantEnv,
    SilentCentre,
    SoftmaxRegression,
    block_subsample,
    blocked_folds,
    diagnosis_metrics,
    lorawan_airtime_s,
    macro_f1,
    make_classifier,
    make_encoder,
    min_uplink_period_s,
    run_episodes,
    synthetic_tep,
)
from genetic_coil_arena.envs.plant.baselines import (
    ENCODINGS,
    PlantContext,
    plant_system,
    plant_task,
    register_plant_baselines,
    task_name,
)
from genetic_coil_arena.envs.plant.runner import matched_far_tau, rethreshold
from genetic_coil_arena.evaluation import (
    BaselineRegistry,
    InterventionHook,
    RecordingHook,
    run_experiment,
)

FAULTS = (1, 2, 5)


@pytest.fixture(scope="module")
def data():
    return synthetic_tep(seed=0, faults=FAULTS)


@pytest.fixture(scope="module")
def ctx(data):
    return PlantContext(data, FAULTS)


class TestClassifiers:
    def blobs(self, seed=0, n=300, k=3):
        rng = np.random.default_rng(seed)
        centres = rng.normal(0, 4, size=(k, 5))
        y = rng.integers(0, k, size=n)
        return centres[y] + rng.normal(size=(n, 5)), y

    @pytest.mark.parametrize("kind", ["logreg", "mlp"])
    def test_separable_blobs_are_learned(self, kind):
        x, y = self.blobs()
        clf = make_classifier(kind, 3, seed=0).fit(x, y)
        assert (clf.predict(x) == y).mean() > 0.97
        p = clf.predict_proba(x)
        assert p.shape == (300, 3) and np.allclose(p.sum(1), 1.0)
        assert clf.decision_function(x).shape == (300, 3)

    @pytest.mark.parametrize("kind", ["logreg", "mlp"])
    def test_fit_is_deterministic_given_the_seed(self, kind):
        x, y = self.blobs(seed=1)
        a = make_classifier(kind, 3, seed=2).fit(x, y).decision_function(x)
        b = make_classifier(kind, 3, seed=2).fit(x, y).decision_function(x)
        assert np.array_equal(a, b)
        assert isinstance(make_classifier("logreg", 3), SoftmaxRegression)
        assert isinstance(make_classifier("mlp", 3), MLPClassifier)

    def test_gradients_match_finite_differences(self):
        x, y = self.blobs(n=40)
        w = np.full(40, 1 / 40)
        for clf in (SoftmaxRegression(3, l2=0.1), MLPClassifier(3, hidden=4, l2=0.1)):
            theta = clf._init(5, np.random.default_rng(0)) + 0.1
            _, grad = clf._loss_grad(theta, x, y, w)
            for i in np.random.default_rng(1).choice(theta.size, 6, replace=False):
                d = np.zeros_like(theta)
                d[i] = 1e-6
                num = (
                    clf._loss_grad(theta + d, x, y, w)[0]
                    - clf._loss_grad(theta - d, x, y, w)[0]
                ) / 2e-6
                assert num == pytest.approx(grad[i], abs=1e-5)

    def test_no_features_learns_only_the_prior(self):
        y = np.array([0, 1, 2, 1, 2, 2])
        clf = SoftmaxRegression(3).fit(np.zeros((6, 0)), y)
        assert len(set(clf.predict(np.zeros((4, 0))))) == 1

    def test_bad_input_and_kinds(self):
        with pytest.raises(ValueError):
            SoftmaxRegression(3).fit(np.zeros((4, 2)), np.zeros(3, dtype=int))
        with pytest.raises(RuntimeError):
            SoftmaxRegression(3).predict(np.zeros((1, 2)))
        with pytest.raises(ValueError):
            make_classifier("forest", 3)


class TestFoldsAndSubsampling:
    def test_blocked_folds_never_overlap_and_keep_a_gap(self):
        run = np.repeat([0, 1, 2], [40, 40, 30])
        folds = blocked_folds(run, n_folds=4, gap=3)
        assert len(folds) == 4
        pos = np.concatenate([np.arange(c) for c in (40, 40, 30)])
        for train, test in folds:
            assert not set(train) & set(test)
            for r in range(3):
                t = pos[test[run[test] == r]]
                tr = pos[train[run[train] == r]]
                assert np.abs(tr[:, None] - t[None, :]).min() > 3  # a gap each side
        covered = np.sort(np.concatenate([test for _, test in folds]))
        assert np.array_equal(covered, np.arange(110))  # every window held out once

    def test_block_subsample_keeps_whole_blocks_per_run(self):
        run = np.repeat([0, 1], [100, 60])
        idx = block_subsample(run, 0.5, np.random.default_rng(0), block=10)
        assert np.all(np.diff(idx) > 0)
        assert (run[idx] == 0).sum() == 50 and (run[idx] == 1).sum() == 30
        assert set(idx // 10 * 10) <= set(range(0, 160, 10))
        again = block_subsample(run, 0.5, np.random.default_rng(0), block=10)
        assert np.array_equal(idx, again)
        other = block_subsample(run, 0.5, np.random.default_rng(1), block=10)
        assert not np.array_equal(idx, other)


class TestRethreshold:
    @pytest.fixture(scope="class")
    def rollout(self, ctx):
        env, encoder, centre = ctx.fit("features", 16, 0)
        return env, ctx.rollout(env, encoder, centre), centre

    def test_array_metrics_agree_with_the_environments_own_tracking(self, rollout, ctx):
        env, res, _ = rollout
        m = diagnosis_metrics(res, ctx.n_classes)
        fault_run = res["episode_class"] != 0
        det = res["detection_delay"][fault_run]
        dia = res["diagnosis_delay"][fault_run]
        assert m["detection_delay"] == pytest.approx(np.nanmean(det))
        assert m["diagnosis_delay"] == pytest.approx(np.nanmean(dia))
        assert m["detection_rate"] == pytest.approx(np.isfinite(det).mean())
        assert m["false_alarm_rate"] == pytest.approx(
            res["false_alarms"].sum() / res["normal_windows"].sum()
        )

    def test_the_calibrated_threshold_reproduces_the_rollout(self, rollout):
        _, res, centre = rollout
        again = rethreshold(res, centre.tau)
        assert np.array_equal(
            again["diagnoses"][res["valid"]], res["diagnoses"][res["valid"]]
        )

    def test_extreme_thresholds_alarm_always_or_never(self, rollout, ctx):
        _, res, _ = rollout
        always = diagnosis_metrics(rethreshold(res, -np.inf), ctx.n_classes)
        never = diagnosis_metrics(rethreshold(res, np.inf), ctx.n_classes)
        assert always["false_alarm_rate"] == 1.0 and always["detection_rate"] == 1.0
        assert never["false_alarm_rate"] == 0.0 and never["detection_rate"] == 0.0
        assert never["macro_f1"] == 0.0

    @pytest.mark.parametrize("target", [0.02, 0.1, 0.3])
    def test_matched_threshold_hits_the_target_false_alarm_rate(
        self, rollout, ctx, target
    ):
        _, res, _ = rollout
        out = rethreshold(res, matched_far_tau(res, target))
        far = diagnosis_metrics(out, ctx.n_classes)["false_alarm_rate"]
        n_normal = int((res["valid"] & (res["labels"] == 0)).sum())
        assert abs(far - target) <= 1.5 / n_normal

    def test_a_silent_centre_has_no_margin_to_move(self, ctx):
        env, encoder, centre = ctx.fit("silent", 0, 0)
        res = ctx.rollout(env, encoder, centre)
        with pytest.raises(ValueError, match="margins"):
            matched_far_tau(res, 0.05)
        with pytest.raises(ValueError, match="margins"):
            rethreshold(res, 0.0)

    def test_evaluate_reports_the_matched_operating_point(self, ctx):
        out = ctx.evaluate("summary", 16, 0)
        assert out["false_alarm_rate_matched"] == pytest.approx(0.05, abs=0.02)
        assert 0.0 <= out["macro_f1_matched"] <= 1.0
        silent = ctx.evaluate("silent", 0, 0)
        assert silent["macro_f1_matched"] == 0.0 and np.isnan(silent["tau_matched"])


class TestMetrics:
    def test_macro_f1_over_fault_classes_only(self):
        labels = np.array([0, 0, 1, 1, 2, 2, 2])
        perfect, per = macro_f1(labels, labels, 3)
        assert perfect == 1.0 and per.tolist() == [1.0, 1.0]
        # Always predicting normal scores zero
        assert macro_f1(labels, np.zeros_like(labels), 3)[0] == 0.0
        # Normal windows called faults count against precision
        preds = np.array([1, 1, 1, 1, 2, 2, 2])
        f1, per = macro_f1(labels, preds, 3)
        assert per[0] == pytest.approx(2 * 2 / (2 * 2 + 2 + 0))  # tp 2, fp 2
        assert per[1] == 1.0 and f1 == pytest.approx(per.mean())

    def test_diagnosis_metrics_from_a_rollout(self, data):
        env = PlantEnv(PlantConfig(faults=FAULTS, budget=0), data)

        class Truth:
            needs_raw = False

            def diagnose(self, messages, raw=None):
                return env.labels()

        res = run_episodes(env, None, Truth(), range(env.n_runs))
        m = diagnosis_metrics(res, env.config.n_classes)
        assert m["macro_f1"] == 1.0 and m["false_alarm_rate"] == 0.0
        assert m["detection_rate"] == 1.0 and m["detection_delay"] == 10.0
        assert res["labels"].shape == res["diagnoses"].shape == (4, 95)
        assert res["valid"].all() and res["onset"].tolist() == [-1, 160, 160, 160]
        silent = run_episodes(env, None, SilentCentre(), range(env.n_runs))
        ms = diagnosis_metrics(silent, env.config.n_classes)
        assert ms["macro_f1"] == 0.0 and ms["detection_rate"] == 0.0
        assert np.isnan(ms["detection_delay"]) and ms["false_alarm_rate"] == 0.0


class TestAirtime:
    def test_lorawan_numbers(self):
        # 51 B payload + 13 B overhead: 2.79 s at SF12, 0.118 s at SF7 (Semtech formula)
        assert lorawan_airtime_s(51, 12) == pytest.approx(2.793, abs=0.01)
        assert lorawan_airtime_s(51, 7) == pytest.approx(0.118, abs=0.002)
        assert min_uplink_period_s(51, 12) == pytest.approx(279.3, abs=1.0)
        assert lorawan_airtime_s(8, 12) < lorawan_airtime_s(51, 12)


class TestCentres:
    def test_calibrated_margin_controls_held_out_false_alarms(self, ctx):
        env, encoder, centre = ctx.fit("features", 16, 0)
        assert centre.tau != 0.0
        res = ctx.rollout(env, encoder, centre)
        m = diagnosis_metrics(res, ctx.n_classes)
        assert m["false_alarm_rate"] < 0.2 and m["detection_rate"] == 1.0

    def test_target_none_is_plain_argmax(self, ctx):
        enc = make_encoder("features", 16, ctx.layout).fit(
            ctx.train["windows"][ctx.train["labels"] == 0]
        )
        centre = DecodedCentre(enc, ctx.n_classes, "logreg", 0, target_far=None).fit(
            ctx.train["windows"], ctx.train["labels"]
        )
        assert centre.tau == 0.0
        feats = enc.decode(enc.encode_windows(ctx.train["windows"][:50]))
        assert np.array_equal(
            centre.classifier.predict(feats),
            centre.diagnose(enc.encode_windows(ctx.train["windows"][:50])),
        )

    def test_calibration_needs_run_ids(self, ctx):
        enc = make_encoder("alarms", 8, ctx.layout).fit(
            ctx.train["windows"][ctx.train["labels"] == 0]
        )
        with pytest.raises(ValueError, match="run"):
            DecodedCentre(enc, ctx.n_classes).fit(
                ctx.train["windows"], ctx.train["labels"]
            )

    def test_oracle_needs_raw_windows(self, ctx):
        _, _, oracle = ctx.fit("oracle", 0, 0)
        assert oracle.needs_raw and isinstance(oracle, OracleCentre)
        with pytest.raises(ValueError, match="raw"):
            oracle.diagnose(np.zeros((1, 4, 0), dtype=np.int64))

    def test_honest_centres_never_get_the_raw_windows(self, ctx):
        env, encoder, centre = ctx.fit("summary", 8, 0)
        seen = []
        original = centre.diagnose

        def spy(messages, raw=None):
            seen.append(raw)
            return original(messages, raw)

        centre.diagnose = spy
        ctx.rollout(env, encoder, centre)
        assert seen and all(r is None for r in seen)


class TestEvaluate:
    @pytest.mark.parametrize("encoding", ENCODINGS)
    def test_every_encoding_runs_end_to_end(self, ctx, encoding):
        out = ctx.evaluate(encoding, 0 if encoding in ("silent", "oracle") else 16, 0)
        assert 0.0 <= out["macro_f1"] <= 1.0 and 0.0 <= out["false_alarm_rate"] <= 1.0
        assert out["encoding"] == encoding and out["seed"] == 0
        assert len(out["f1_per_class"]) == len(FAULTS)

    def test_ordering_on_clean_synthetic_faults(self, ctx):
        f1 = {e: ctx.evaluate(e, 16, 0)["macro_f1"] for e in ("summary", "noise")}
        assert f1["summary"] > 0.6 > f1["noise"]
        assert ctx.evaluate("silent", 0, 0)["macro_f1"] == 0.0
        assert ctx.evaluate("oracle", 0, 0)["macro_f1"] > 0.6

    def test_evaluate_is_deterministic_and_seeds_differ(self, ctx):
        a, b = ctx.evaluate("summary", 8, 3), ctx.evaluate("summary", 8, 3)
        assert a["macro_f1"] == b["macro_f1"] and a["tau"] == b["tau"]
        taus = {ctx.evaluate("summary", 8, s)["tau"] for s in range(3)}
        assert len(taus) > 1

    def test_budget_zero_needs_a_silent_encoding(self, ctx):
        with pytest.raises(ValueError, match="budget"):
            ctx.evaluate("summary", 0, 0)
        with pytest.raises(ValueError, match="Unknown"):
            ctx.evaluate("psychic", 8, 0)

    def test_bytes_used_never_exceeds_the_budget(self, ctx):
        for encoding in ("alarms", "summary", "features", "noise"):
            for budget in (8, 16, 51):
                used = ctx.evaluate(encoding, budget, 0)["bytes_used"]
                assert used <= budget
        assert ctx.evaluate("alarms", 51, 0)["bytes_used"] == 1.0
        assert ctx.evaluate("silent", 0, 0)["bytes_used"] == 0.0


class TestHarnessIntegration:
    def test_hooks_act_on_the_uplink(self, ctx):
        env, encoder, centre = ctx.fit("features", 16, 0)
        sent_total = []

        def watch(step, obs, sent, done):
            sent_total.append(np.array(sent))

        rec = RecordingHook(InterventionHook("zero", spec=env.spec.message))
        res = run_episodes(
            env, encoder, centre, range(env.n_runs), hook=rec, observer=watch
        )
        assert len(rec.records) == len(sent_total) == 95
        assert rec.records[0].shape == (env.n_runs, 4, 16)
        assert rec.records[0].any() and not env.last_message.any()  # delivered: zeros
        zeroed = diagnosis_metrics(res, ctx.n_classes)
        intact = diagnosis_metrics(ctx.rollout(env, encoder, centre), ctx.n_classes)
        assert zeroed["macro_f1"] < intact["macro_f1"] - 0.2

    def test_task_with_registered_baselines_runs_through_run_experiment(self, ctx):
        task = plant_task(ctx, 8, n_seeds=3)
        assert task.spec.message == ctx.env(8).spec.message
        assert task.spec.metric == "macro_f1" and task.name == task_name(ctx, 8)
        registry = BaselineRegistry()
        registry.register(task.name, "silent", plant_system(ctx, "silent", 0))
        registry.register(task.name, "random_bytes", plant_system(ctx, "noise", 8))
        result = run_experiment(
            task,
            {"summary": plant_system(ctx, "summary", 8)},
            seeds=[0, 1, 2],
            ablations=("zero", "shuffle_batch", "swap_receivers", "random_matched"),
            registry=registry,
        )
        assert set(result.required_baselines) == {"silent", "random_bytes"}
        assert result.condition("summary").mean > 0.6
        assert result.condition("silent").mean == 0.0
        assert len(result.ablations) == 4
        assert all(a.listening for a in result.ablations)  # the centre uses the bytes
        # The harness reproduces a direct evaluation of the same cells
        direct = np.mean([ctx.evaluate("summary", 8, s)["macro_f1"] for s in range(3)])
        assert result.condition("summary").mean == pytest.approx(direct)

    def test_global_registration(self, ctx):
        register_plant_baselines(ctx, 8)
        from genetic_coil_arena.evaluation import BASELINES

        assert set(BASELINES.required(task_name(ctx, 8))) == {"silent", "random_bytes"}
        for name in ("silent", "random_bytes"):
            BASELINES.unregister(task_name(ctx, 8), name)


def test_plant_package_is_domain_agnostic():
    code = (
        "import sys, genetic_coil_arena.envs.plant as p, genetic_coil_arena.envs; "
        "bad = [m for m in sys.modules if m.startswith(('genetic_coil_arena.snake_core', "
        "'genetic_coil_arena.evo', 'genetic_coil_arena.analysis', "
        "'genetic_coil_arena.envs.squad', 'numba', 'torch', 'pygame'))]; "
        "print(bad); assert not bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
