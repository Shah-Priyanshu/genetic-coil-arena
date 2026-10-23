"""Test message interventions, the deterministic RNG API and the hook protocol."""

import numpy as np
import pytest

from genetic_coil_arena.evaluation.interventions import (
    ABLATIONS,
    INTERVENTIONS,
    InterventionHook,
    MessageSpec,
    RecordingHook,
    apply_hook,
    get_intervention,
    no_intervention,
    random_matched,
    shuffle_batch,
    step_rng,
    swap_receivers,
    zero_messages,
)
from genetic_coil_arena.evaluation.metrics import entropy, mutual_information

B, R, L, V, D = 64, 3, 4, 5, 6


@pytest.fixture
def discrete():
    return np.random.default_rng(0).integers(0, V, size=(B, R, L)).astype(np.int64)


@pytest.fixture
def continuous():
    rng = np.random.default_rng(1)
    return (rng.normal(2.0, 3.0, size=(B, R, D))).astype(np.float32)


class TestMessageSpec:
    """Shape, dtype and range validation."""

    def test_capacity(self):
        assert MessageSpec.discrete(2, 3, 8).capacity_bits == 9.0
        assert MessageSpec.continuous(2, 4).capacity_bits == float("inf")

    def test_validate_accepts_conforming(self, discrete, continuous):
        MessageSpec.discrete(R, L, V).validate(discrete)
        MessageSpec.continuous(R, D).validate(continuous)

    @pytest.mark.parametrize(
        "bad",
        [
            np.zeros((B, R, L + 1), dtype=np.int64),
            np.zeros((B, R, L), dtype=np.int32),
            np.full((B, R, L), V, dtype=np.int64),
            np.full((B, R, L), -1, dtype=np.int64),
            np.zeros((R, L), dtype=np.int64),
        ],
    )
    def test_validate_rejects(self, bad):
        with pytest.raises((ValueError, TypeError)):
            MessageSpec.discrete(R, L, V).validate(bad)

    def test_continuous_must_be_float32(self):
        with pytest.raises(TypeError):
            MessageSpec.continuous(R, D).validate(np.zeros((B, R, D)))

    def test_invalid_specs(self):
        with pytest.raises(ValueError):
            MessageSpec("analog", 1, 1)
        with pytest.raises(ValueError):
            MessageSpec("discrete", 1, 1)
        with pytest.raises(ValueError):
            MessageSpec("continuous", 1, 1, vocab_size=3)

    def test_empty(self):
        msgs = MessageSpec.discrete(R, L, V).empty(7)
        assert msgs.shape == (7, R, L) and msgs.dtype == np.int64


class TestInterventions:
    """Each intervention keeps shape and dtype, never mutates, and does its job."""

    @pytest.mark.parametrize("name", sorted(INTERVENTIONS))
    def test_shape_dtype_and_no_mutation(self, name, discrete, continuous):
        for msgs in (discrete, continuous):
            before = msgs.copy()
            out = get_intervention(name)(msgs, step_rng(0), None)
            assert out.shape == msgs.shape and out.dtype == msgs.dtype
            np.testing.assert_array_equal(msgs, before)
            assert not np.shares_memory(out, msgs)

    def test_none_is_identity(self, discrete):
        np.testing.assert_array_equal(no_intervention(discrete, step_rng(0)), discrete)

    def test_zero(self, discrete, continuous):
        assert not zero_messages(discrete, step_rng(0)).any()
        assert not zero_messages(continuous, step_rng(0)).any()

    def test_shuffle_batch_is_a_derangement_of_episodes(self):
        msgs = np.arange(B, dtype=np.int64)[:, None, None].repeat(R, 1).repeat(L, 2)
        out = shuffle_batch(msgs, step_rng(3))
        moved_from = out[:, 0, 0]
        assert sorted(moved_from) == list(range(B))  # Same multiset of messages
        assert np.all(moved_from != np.arange(B))  # Nobody keeps their own
        assert np.all(out == out[:, :1, :1])  # Whole episodes move together

    def test_shuffle_batch_needs_two_episodes(self, discrete):
        with pytest.raises(ValueError):
            shuffle_batch(discrete[:1], step_rng(0))

    def test_swap_receivers(self):
        msgs = np.arange(R, dtype=np.int64)[None, :, None].repeat(B, 0).repeat(L, 2)
        out = swap_receivers(msgs, step_rng(0))
        assert np.all(out[:, :, 0] != np.arange(R))
        assert sorted(out[0, :, 0]) == list(range(R))
        with pytest.raises(ValueError):
            swap_receivers(msgs[:, :1], step_rng(0))

    def test_random_matched_discrete_matches_marginals_breaks_dependence(self):
        rng = np.random.default_rng(0)
        n = 20000
        target = rng.integers(0, 4, size=n)
        msgs = np.stack([target, (target + 1) % 4], axis=1)[:, None, :].astype(np.int64)
        out = random_matched(msgs, step_rng(0))
        for pos in range(2):
            assert entropy(out[:, 0, pos]) == pytest.approx(
                entropy(msgs[:, 0, pos]), abs=0.01
            )
            assert set(np.unique(out[:, 0, pos])) <= set(np.unique(msgs[:, 0, pos]))
        assert mutual_information(msgs, target) == pytest.approx(2.0, abs=0.01)
        assert mutual_information(out, target) == pytest.approx(0.0, abs=0.01)

    def test_random_matched_constant_stays_constant(self):
        msgs = np.full((B, R, L), 3, dtype=np.int64)
        np.testing.assert_array_equal(random_matched(msgs, step_rng(0)), msgs)

    def test_random_matched_continuous_moments(self):
        rng = np.random.default_rng(0)
        msgs = rng.normal([1.0, -5.0], [0.5, 4.0], size=(50000, 1, 2)).astype(
            np.float32
        )
        out = random_matched(msgs, step_rng(1))
        assert out.dtype == np.float32
        np.testing.assert_allclose(out.mean(axis=0), msgs.mean(axis=0), atol=0.06)
        np.testing.assert_allclose(out.std(axis=0), msgs.std(axis=0), rtol=0.02)
        assert abs(np.corrcoef(out[:, 0, 1], msgs[:, 0, 1])[0, 1]) < 0.02

    def test_spec_is_enforced(self, discrete):
        with pytest.raises(ValueError):
            zero_messages(discrete, step_rng(0), MessageSpec.discrete(R, L, 2))

    def test_unknown_name(self):
        with pytest.raises(KeyError):
            get_intervention("scramble")

    def test_default_ablations_registered(self):
        assert set(ABLATIONS) <= set(INTERVENTIONS)


class TestDeterminism:
    """The RNG API and hooks replay exactly."""

    def test_step_rng_depends_only_on_keys(self):
        a = step_rng(5, 2, 1).integers(0, 1 << 30, size=4)
        np.random.default_rng(123).random(10)  # Unrelated global activity
        b = step_rng(5, 2, 1).integers(0, 1 << 30, size=4)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, step_rng(5, 3, 1).integers(0, 1 << 30, size=4))
        assert not np.array_equal(a, step_rng(6, 2, 1).integers(0, 1 << 30, size=4))

    def test_step_rng_rejects_negative(self):
        with pytest.raises(ValueError):
            step_rng(-1)

    def test_hook_replays_after_reset(self, discrete):
        hook = InterventionHook("shuffle_batch", seed=7)
        first = [hook(discrete, step) for step in range(3)]
        hook.reset()
        second = [hook(discrete, step) for step in range(3)]
        for x, y in zip(first, second):
            np.testing.assert_array_equal(x, y)

    def test_hook_seeds_differ(self, discrete):
        a = InterventionHook("shuffle_batch", seed=1)(discrete)
        b = InterventionHook("shuffle_batch", seed=2)(discrete)
        assert not np.array_equal(a, b)

    def test_hook_validates_with_spec(self, discrete):
        hook = InterventionHook("zero", spec=MessageSpec.discrete(R, L + 1, V))
        with pytest.raises(ValueError):
            hook(discrete)


class TestHooks:
    """Hooks as an environment loop uses them."""

    def test_apply_hook_none_passthrough(self, discrete):
        assert apply_hook(None, discrete) is discrete

    def test_plain_function_is_a_hook(self, discrete):
        out = apply_hook(lambda m, step=0: m + 1, discrete, step=4)
        np.testing.assert_array_equal(out, discrete + 1)

    def test_recording_hook_records_pre_intervention(self, discrete):
        rec = RecordingHook(InterventionHook("zero"))
        out = apply_hook(rec, discrete, step=2)
        assert not out.any()
        np.testing.assert_array_equal(rec.records[0], discrete)
        assert rec.steps == [2]
        apply_hook(rec, discrete, step=3)
        assert rec.stacked().shape == (2 * B, R, L)
        rec.reset()
        assert rec.records == [] and rec.inner.calls == 0
        with pytest.raises(ValueError):
            rec.stacked()

    def test_recording_hook_without_inner_is_transparent(self, discrete):
        rec = RecordingHook()
        np.testing.assert_array_equal(rec(discrete), discrete)

    def test_env_loop_integration(self):
        """A toy multi-step loop: intervention changes what receivers get, per step."""

        def loop(hook, steps=4):
            received = []
            for step in range(steps):
                sent = (
                    np.full((8, 2, 1), step, dtype=np.int64)
                    + np.arange(8)[:, None, None]
                )
                received.append(apply_hook(hook, sent, step))
            return received

        intact = loop(None)
        a = loop(InterventionHook("shuffle_batch", seed=3))
        b = loop(InterventionHook("shuffle_batch", seed=3))
        for x, y, z in zip(intact, a, b):
            np.testing.assert_array_equal(y, z)
            assert not np.array_equal(x, y)
