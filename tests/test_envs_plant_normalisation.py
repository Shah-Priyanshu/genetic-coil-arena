"""Normalisation and calibration use normal training data only: no test leakage."""

import copy

import numpy as np
import pytest

from genetic_coil_arena.envs.plant import (
    NODES,
    Normaliser,
    PlantConfig,
    PlantEnv,
    make_encoder,
    synthetic_tep,
)
from genetic_coil_arena.envs.plant.baselines import PlantContext
from genetic_coil_arena.envs.plant.tep import N_VARIABLES

FAULTS = (1, 2)


@pytest.fixture(scope="module")
def data():
    return synthetic_tep(seed=3, faults=FAULTS)


def env_for(data):
    return PlantEnv(PlantConfig(faults=FAULTS, budget=8), data)


class TestNormaliser:
    def test_statistics_are_those_of_the_normal_training_run(self, data):
        norm = Normaliser.fit(data.normal_train)
        assert np.allclose(norm.mean, data.normal_train.mean(axis=0))
        assert np.allclose(norm.std, data.normal_train.std(axis=0, ddof=1))
        z = norm.transform(data.normal_train)
        assert np.allclose(z.mean(axis=0), 0.0, atol=1e-4)
        assert np.allclose(z.std(axis=0, ddof=1), 1.0, atol=1e-4)

    def test_test_data_is_not_standardised_by_its_own_statistics(self, data):
        z = Normaliser.fit(data.normal_train).transform(data.normal_test)
        # Own-statistics scaling would give mean 0 / std 1 exactly; this does not
        assert not np.allclose(z.mean(axis=0), 0.0, atol=1e-3)
        assert not np.allclose(z.std(axis=0, ddof=1), 1.0, atol=1e-3)

    def test_transform_is_float32_and_never_refits(self, data):
        norm = Normaliser.fit(data.normal_train)
        mean, std = norm.mean.copy(), norm.std.copy()
        out = norm.transform(data.fault_test[1] * 1000.0)
        assert out.dtype == np.float32
        assert np.array_equal(norm.mean, mean) and np.array_equal(norm.std, std)

    def test_constant_variable_does_not_blow_up(self):
        x = np.tile(np.arange(5.0)[:, None], (1, N_VARIABLES))
        x[:, 3] = 7.0
        norm = Normaliser.fit(x)
        assert norm.std[3] > 0 and np.all(np.isfinite(norm.transform(x)))

    def test_fit_rejects_degenerate_input(self):
        with pytest.raises(ValueError):
            Normaliser.fit(np.zeros((1, N_VARIABLES)))


class TestEnvironmentUsesTrainingStatisticsOnly:
    def test_env_normaliser_equals_normal_train_fit(self, data):
        env = env_for(data)
        ref = Normaliser.fit(data.normal_train)
        assert np.array_equal(env.normaliser.mean, ref.mean)
        assert np.array_equal(env.normaliser.std, ref.std)

    def test_changing_any_test_data_leaves_the_statistics_unchanged(self, data):
        changed = copy.deepcopy(data)
        changed.normal_test = changed.normal_test * 50.0 + 9.0
        changed.fault_test = {f: x * 50.0 + 9.0 for f, x in changed.fault_test.items()}
        base, other = env_for(data), env_for(changed)
        assert np.array_equal(base.normaliser.mean, other.normaliser.mean)
        assert np.array_equal(base.normaliser.std, other.normaliser.std)
        # ... while the normalised test windows do reflect the change
        a = base.reset([0]).specialists["window"]
        b = other.reset([0]).specialists["window"]
        assert not np.allclose(a, b)

    def test_changing_fault_training_runs_leaves_the_statistics_unchanged(self, data):
        changed = copy.deepcopy(data)
        changed.fault_train = {f: x + 100.0 for f, x in changed.fault_train.items()}
        assert np.array_equal(
            env_for(data).normaliser.mean, env_for(changed).normaliser.mean
        )

    def test_changing_normal_training_data_does_change_the_statistics(self, data):
        changed = copy.deepcopy(data)
        changed.normal_train = changed.normal_train + 5.0
        assert not np.allclose(
            env_for(data).normaliser.mean, env_for(changed).normaliser.mean
        )

    def test_test_windows_are_normalised_with_the_training_statistics(self, data):
        env = env_for(data)
        norm = Normaliser.fit(data.normal_train)
        expected = norm.transform(data.normal_test)
        out = env.labelled_windows("test")
        first = out["windows"][out["run"] == 0][0]  # first window of the normal run
        # Node 0 (reactor), first real variable, first 20 samples
        col = int(env.layout.index[0, 0])
        assert np.allclose(first[0, 0], expected[:20, col])

    def test_train_and_test_windows_share_one_normaliser(self, data):
        env = env_for(data)
        train = env.labelled_windows("train")
        norm = Normaliser.fit(data.normal_train)
        z = norm.transform(data.normal_train)
        col = int(env.layout.index[2, 4])
        first = train["windows"][train["run"] == 0][0]
        assert np.allclose(first[2, 4], z[:20, col])


class TestCalibrationUsesNormalTrainingWindowsOnly:
    def test_alarm_thresholds_depend_on_the_fitted_windows_only(self, data):
        env = env_for(data)
        out = env.labelled_windows("train", stride=4)
        normal = out["windows"][out["labels"] == 0]
        enc = make_encoder("alarms", 8, env.layout).fit(normal)
        again = make_encoder("alarms", 8, env.layout).fit(normal.copy())
        assert np.array_equal(enc.threshold, again.threshold)
        manual = np.quantile(np.abs(normal.mean(axis=-1)), 1 - 0.005, axis=0)
        assert np.allclose(enc.threshold, manual)

    def test_context_trains_and_calibrates_on_the_training_split(self, data):
        ctx = PlantContext(data, FAULTS)
        # Perturbing every test run must not change the training windows or, therefore,
        # what a seed fits on
        changed = copy.deepcopy(data)
        changed.normal_test = changed.normal_test * 3.0
        changed.fault_test = {f: x * 3.0 for f, x in changed.fault_test.items()}
        other = PlantContext(changed, FAULTS)
        assert np.array_equal(ctx.train["windows"], other.train["windows"])
        assert np.array_equal(ctx.train["labels"], other.train["labels"])
        a, b = ctx.fit("features", 8, 0), other.fit("features", 8, 0)
        assert np.allclose(a[1].spe_ref, b[1].spe_ref)
        assert np.allclose(a[1].energy_ref, b[1].energy_ref)
        assert a[2].tau == pytest.approx(b[2].tau)

    def test_feature_encoder_must_be_fitted(self, data):
        env = env_for(data)
        obs = env.reset([0])
        for name in ("alarms", "features"):
            enc = make_encoder(name, 8, env.layout)
            with pytest.raises(RuntimeError, match="fit"):
                enc.encode(obs.specialists)

    def test_oracle_features_come_from_the_training_fit(self, data):
        # The oracle's PCA/energy references are fitted on normal *training* windows
        ctx = PlantContext(data, FAULTS)
        env, encoder, centre = ctx.fit("oracle", 0, 1)
        assert encoder is None and centre.features is not None
        out = ctx.train
        keep = out["labels"] == 0
        assert centre.features.energy_ref.shape == (len(NODES), 18, 3)
        assert np.all(centre.features.energy_ref > 0)
        assert keep.sum() > 0
