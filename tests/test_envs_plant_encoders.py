"""Tests for the hand-designed uplink encoders: byte budget, codecs, determinism."""

import numpy as np
import pytest

from genetic_coil_arena.envs.plant import (
    BUDGETS,
    ENCODERS,
    NODES,
    PlantConfig,
    PlantEnv,
    make_encoder,
    synthetic_tep,
)
from genetic_coil_arena.envs.plant.encoders import (
    LOG_RATIO,
    SIGNED,
    FieldCodec,
    Quantiser,
    band_energies,
    band_slices,
    window_stats,
)

FAULTS = (1, 2, 5)
N = len(NODES)


@pytest.fixture(scope="module")
def env():
    return PlantEnv(PlantConfig(faults=FAULTS, budget=0), synthetic_tep(0, FAULTS))


@pytest.fixture(scope="module")
def train(env):
    return env.labelled_windows("train", stride=4)


@pytest.fixture(scope="module")
def normal(train):
    return train["windows"][train["labels"] == 0]


def fitted(name, budget, env, normal, seed=0):
    return make_encoder(name, budget, env.layout, seed).fit(normal)


class TestQuantiser:
    def test_round_trip_error_is_within_one_level(self):
        x = np.linspace(-25, 25, 2001)
        for bits in (4, 8):
            q = SIGNED.levels(x, bits)
            assert q.min() >= 0 and q.max() <= 2**bits - 1
            back = SIGNED.values(q, bits)
            # The asinh scale is fine near zero and coarse for large values
            assert np.all(np.abs(back - x) <= 0.5 * np.maximum(1.0, np.abs(x)) + 1e-9)
            assert np.allclose(SIGNED.values(SIGNED.levels(back, bits), bits), back)

    def test_8_bits_are_finer_than_4_bits(self):
        x = np.random.default_rng(0).normal(0, 2, 500)
        err = {
            b: np.abs(SIGNED.values(SIGNED.levels(x, b), b) - x).mean() for b in (4, 8)
        }
        assert err[8] < err[4] / 4

    def test_values_clip_to_the_range_and_levels_are_monotone(self):
        q = SIGNED.levels(np.array([-1e6, -30.0, 0.0, 30.0, 1e6]), 8)
        assert q.tolist()[:2] == [0, 0] and q.tolist()[3:] == [255, 255]
        assert q[2] in (127, 128)  # zero sits at the middle of a symmetric range
        assert np.all(np.diff(SIGNED.levels(np.linspace(-40, 40, 300), 4)) >= 0)
        lin = Quantiser("linear", 0.0, 1.0)
        assert lin.levels(np.array([0.0, 0.5, 1.0]), 4).tolist() == [0, 8, 15]
        assert LOG_RATIO.levels(np.array([-9.0, 9.0]), 4).tolist() == [0, 15]


class TestFieldCodec:
    def feats(self, b=6, n=2, g=2, v=5, seed=0):
        return np.random.default_rng(seed).normal(0, 2, size=(b, n, g, v))

    def slots(self, n_slots, groups=2, v=5):
        return [(i // v % groups, i % v) for i in range(n_slots)]

    def test_8_bit_fields_when_they_all_fit(self):
        codec = FieldCodec([self.slots(6), self.slots(4)], [SIGNED, SIGNED], budget=8)
        assert codec.bits == [8, 8] and codec.bytes_used == (6, 4)
        msgs = codec.encode(self.feats())
        assert msgs.shape == (6, 2, 8) and msgs.dtype == np.int64
        assert not msgs[:, 0, 6:].any() and not msgs[:, 1, 4:].any()  # zero padding
        assert codec.n_fields == 10 and codec.decode(msgs).shape == (6, 10)

    def test_4_bit_packing_truncates_to_twice_the_budget(self):
        codec = FieldCodec([self.slots(10), self.slots(10)], [SIGNED, SIGNED], budget=3)
        assert codec.bits == [4, 4] and codec.bytes_used == (3, 3)
        msgs = codec.encode(self.feats())
        assert msgs.shape == (6, 2, 3) and msgs.max() <= 255
        assert codec.n_fields == 12  # 6 fields per node survive

    def test_decode_inverts_encode_to_within_the_quantisation_step(self):
        slots = [self.slots(8)] * 2
        feats = self.feats(b=40)
        for budget, bits in ((8, 8), (4, 4)):
            codec = FieldCodec(slots, [SIGNED, SIGNED], budget)
            assert codec.bits == [bits, bits]
            decoded = codec.decode(codec.encode(feats))
            expected = np.stack(
                [feats[:, i, g, v] for i in range(2) for g, v in codec.slots[i]], axis=1
            )
            tol = 0.12 if bits == 8 else 1.2
            assert np.abs(decoded - expected).max() < tol * 3

    def test_zero_budget_has_no_fields(self):
        codec = FieldCodec([self.slots(6)], [SIGNED, SIGNED], budget=0)
        assert codec.n_fields == 0 and codec.bytes_used == (0,)
        assert codec.decode(np.zeros((3, 1, 0), dtype=np.int64)).shape == (3, 0)


class TestWindowFeatures:
    def test_statistics_match_numpy(self):
        w = np.random.default_rng(0).normal(size=(3, 2, 4, 20)).astype(np.float32)
        s = window_stats(w)
        assert np.allclose(s["mean"], w.mean(-1)) and np.allclose(s["std"], w.std(-1))
        assert np.allclose(s["max"], w.max(-1)) and np.allclose(s["min"], w.min(-1))
        ramp = np.broadcast_to(np.arange(20.0), (1, 1, 1, 20))
        assert np.allclose(window_stats(ramp)["slope"], 1.0)

    def test_band_energies_are_detrended_and_partition_the_spectrum(self):
        w = np.random.default_rng(1).normal(size=(2, 1, 3, 20))
        total = band_energies(w).sum(-1)
        centred = w - w.mean(-1, keepdims=True)
        # Parseval over bins 1..10 (the one-sided spectrum double counts all but Nyquist)
        full = (np.abs(np.fft.rfft(centred, axis=-1)) ** 2).sum(-1) / 20
        assert np.all(total <= full + 1e-9)
        assert np.allclose(band_energies(w + 7.0), band_energies(w))  # DC removed
        bins = np.concatenate(band_slices(20))
        assert bins.tolist() == list(range(1, 11))


@pytest.mark.parametrize("name", ["alarms", "summary", "features", "silent", "noise"])
class TestEveryEncoder:
    @pytest.mark.parametrize("budget", [1, 8, 16, 51])
    def test_messages_obey_the_byte_budget(self, name, budget, env, normal, train):
        enc = fitted(name, budget, env, normal)
        msgs = enc.encode_windows(train["windows"][:50])
        assert msgs.shape == (50, N, budget) and msgs.dtype == np.int64
        assert msgs.min() >= 0 and msgs.max() < 256
        assert len(enc.bytes_used) == N and max(enc.bytes_used) <= budget
        assert enc.decode(msgs).shape == (50, enc.n_features)
        assert enc.decode(msgs).dtype == np.float32

    def test_zero_budget_sends_nothing(self, name, env, normal, train):
        enc = fitted(name, 0, env, normal)
        msgs = enc.encode_windows(train["windows"][:10])
        assert msgs.shape == (10, N, 0)
        assert enc.decode(msgs).shape == (10, 0)

    def test_env_accepts_the_encoders_messages(self, name, env, normal):
        e = PlantEnv(PlantConfig(faults=FAULTS, budget=16), env.data)
        enc = fitted(name, 16, e, normal)
        obs = e.reset(range(e.n_runs))
        obs = e.step(None, enc.encode(obs.specialists), None).obs
        assert obs.centre["uplink"].shape == (e.n_runs, N, 16)

    def test_deterministic_given_the_same_inputs(self, name, env, normal, train):
        a = fitted(name, 16, env, normal, seed=3).encode_windows(train["windows"][:20])
        b = fitted(name, 16, env, normal, seed=3).encode_windows(train["windows"][:20])
        assert np.array_equal(a, b)


class TestSpecificDesigns:
    def test_alarms_use_one_byte_whatever_the_budget(self, env, normal, train):
        w = train["windows"][:200]
        base = fitted("alarms", 1, env, normal).encode_windows(w)[:, :, 0]
        for budget in (8, 16, 51):
            enc = fitted("alarms", budget, env, normal)
            msgs = enc.encode_windows(w)
            assert np.array_equal(msgs[:, :, 0], base)
            assert not msgs[:, :, 1:].any()
            assert enc.bytes_used == (1,) * N
        assert enc.n_features == N * 8

    def test_alarm_bits_decode_to_the_byte(self, env, normal):
        enc = fitted("alarms", 8, env, normal)
        msgs = np.zeros((2, N, 8), dtype=np.int64)
        msgs[0, 0, 0], msgs[1, 3, 0] = 0b10100101, 0b00000010
        bits = enc.decode(msgs).reshape(2, N, 8)
        assert bits[0, 0].tolist() == [1, 0, 1, 0, 0, 1, 0, 1]
        assert bits[1, 3].tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
        assert bits.sum() == 5

    def test_alarms_fire_on_a_shifted_variable_and_not_on_normal(self, env, normal):
        enc = fitted("alarms", 1, env, normal)
        quiet = enc.encode_windows(normal)[:, :, 0]
        assert (quiet > 0).mean() < 0.15  # false-alarm target 0.5% per variable
        shifted = normal.copy()
        shifted[:, 0, 2, :] += 10.0  # reactor, third variable, 10 sigma
        loud = enc.encode_windows(shifted)[:, 0, 0]
        group = int(enc.group[0, 2])
        assert np.all((loud >> group) & 1 == 1)

    def test_silent_sends_zero_bytes(self, env, normal, train):
        enc = fitted("silent", 51, env, normal)
        assert not enc.encode_windows(train["windows"][:5]).any()
        assert enc.bytes_used == (0,) * N and enc.n_features == 0

    def test_noise_is_independent_of_the_data(self, env, normal, train):
        enc = fitted("noise", 16, env, normal, seed=1)
        a = enc.encode_windows(train["windows"][:100])
        b = enc.encode_windows(np.zeros_like(train["windows"][:100]))
        assert not np.array_equal(a, b)  # fresh draws, not a function of the input
        assert 100 < a.mean() < 155

    def test_summary_fills_the_budget_by_priority(self, env, normal, train):
        small = fitted("summary", 8, env, normal)
        big = fitted("summary", 51, env, normal)
        assert small.codec.bits == [
            4,
            4,
            4,
            4,
        ]  # 4 stats of every variable never fit 8 B
        assert big.n_features > small.n_features
        # Reactor at 8 B: 16 4-bit fields = means of the first 16 variables
        assert [g for g, _ in small.codec.slots[0]] == [0] * 16
        w = train["windows"][:30]
        assert small.decode(small.encode_windows(w)).shape == (30, small.n_features)

    def test_summary_means_decode_to_the_window_mean(self, env, normal, train):
        w = train["windows"][:60]
        enc = fitted("summary", 51, env, normal)
        decoded = enc.decode(enc.encode_windows(w))
        means = w.mean(-1)
        # Node 2 (stripper, 10 variables) fits 4 x 10 fields at 8 bits: its means lead
        assert enc.codec.bits[2] == 8
        offset = sum(len(s) for s in enc.codec.slots[:2])
        got = decoded[:, offset : offset + 10]
        assert np.abs(got - means[:, 2, :10]).max() < 0.5

    def test_features_lead_with_pca_and_use_normal_training_only(self, env, normal):
        enc = fitted("features", 8, env, normal)
        assert [g for g, _ in enc.codec.slots[0]][:2] == [0, 0]
        assert all(1 <= p.shape[1] <= 6 for p in enc.components)
        ok = enc.unquantised(normal[:20])
        assert ok.shape[0] == 20 and np.all(np.isfinite(ok))
        shifted = normal[:20].copy()
        shifted[:, 0, :6, :] += 8.0
        spe = enc.features(shifted)[:, 0, 0, 0]
        assert spe.mean() > enc.features(normal[:20])[:, 0, 0, 0].mean() + 1.0

    def test_unknown_encoder_name_fails(self, env):
        with pytest.raises(ValueError):
            make_encoder("telepathy", 8, env.layout)
        assert set(ENCODERS) == {"alarms", "summary", "features", "silent", "noise"}
        assert BUDGETS == (0, 8, 16, 51)
