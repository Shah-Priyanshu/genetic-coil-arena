"""Validate the communication metrics on synthetic senders with known answers.

Meanings are 2 attributes x 4 values (16 meanings, 4 bits). The senders:

- perfect encoder: message = the meaning index (one symbol, V = 16)
- compositional: one position per attribute (L = 2, V = 4), and a variant with
  disjoint symbol sets per attribute (V = 8) for bag-of-symbols
- holistic: a random bijection from meanings to 2-symbol messages (V = 4)
- constant: always the same message
- random noise: uniform symbols independent of the meaning
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from genetic_coil_arena.evaluation.metrics import (
    as_labels,
    bag_of_symbols_disentanglement,
    causal_influence,
    channel_usage,
    communication_metrics,
    entropy,
    entropy_from_counts,
    flatten_messages,
    intervention_influence,
    learnability,
    listening_summary,
    message_entropy,
    mi_null_test,
    mutual_information,
    positional_disentanglement,
    quantize_continuous,
    steps_to_threshold,
    symbol_entropy,
    topographic_similarity,
)

N = 4000
N_ATTR, N_VAL = 2, 4


def _meanings(n=N, seed=0):
    return np.random.default_rng(seed).integers(0, N_VAL, size=(n, N_ATTR))


def _index(meanings):
    return meanings[:, 0] * N_VAL + meanings[:, 1]


def perfect(meanings):
    return _index(meanings)[:, None].astype(np.int64)


def compositional(meanings):
    return meanings.astype(np.int64)


def compositional_disjoint(meanings):
    return np.stack([meanings[:, 0], meanings[:, 1] + N_VAL], axis=1).astype(np.int64)


def holistic(meanings, seed=42):
    perm = np.random.default_rng(seed).permutation(N_VAL**N_ATTR)
    codes = np.stack([perm // N_VAL, perm % N_VAL], axis=1)
    return codes[_index(meanings)].astype(np.int64)


def constant(meanings):
    return np.full((len(meanings), 2), 3, dtype=np.int64)


def noise(meanings, seed=9):
    return np.random.default_rng(seed).integers(0, N_VAL, size=(len(meanings), 2))


@pytest.fixture(scope="module")
def meanings():
    return _meanings()


class TestEntropy:
    """Entropy estimators."""

    def test_uniform_and_point_mass(self):
        assert entropy(np.repeat(np.arange(8), 10)) == pytest.approx(3.0)
        assert entropy([5, 5, 5]) == 0.0

    def test_miller_madow_adds_bias_term(self):
        counts = [10, 10, 20]
        plain = entropy_from_counts(counts)
        corrected = entropy_from_counts(counts, "miller_madow")
        assert corrected - plain == pytest.approx((3 - 1) / (2 * 40) / np.log(2))

    def test_miller_madow_reduces_small_sample_bias(self):
        rng = np.random.default_rng(0)
        true_h = np.log2(16)
        plain, corrected = [], []
        for _ in range(200):
            x = rng.integers(0, 16, size=40)
            plain.append(entropy(x))
            corrected.append(entropy(x, "miller_madow"))
        assert abs(np.mean(corrected) - true_h) < abs(np.mean(plain) - true_h)

    def test_invalid(self):
        with pytest.raises(ValueError):
            entropy_from_counts([0, 0])
        with pytest.raises(ValueError):
            entropy([1, 2], correction="jackknife")

    def test_message_entropy_uses_whole_messages(self, meanings):
        assert message_entropy(compositional(meanings)) == pytest.approx(4.0, abs=0.02)
        np.testing.assert_allclose(
            symbol_entropy(compositional(meanings)), 2.0, atol=0.01
        )
        assert message_entropy(constant(meanings)) == 0.0

    def test_flatten_and_labels(self):
        msgs = np.arange(24).reshape(2, 3, 4)
        assert flatten_messages(msgs).shape == (6, 4)
        assert flatten_messages(np.arange(5)).shape == (5, 1)
        labels = as_labels(np.array([[1, 2], [1, 2], [0, 0]]))
        assert labels[0] == labels[1] != labels[2]
        with pytest.raises(ValueError):
            as_labels([1, 2, 3], n=4)


class TestMutualInformation:
    """Positive signalling: I(M; T) against a shuffled-label null."""

    @pytest.mark.parametrize("sender", [perfect, compositional, holistic])
    def test_informative_senders_carry_all_bits(self, sender, meanings):
        mi = mutual_information(sender(meanings), _index(meanings))
        assert mi == pytest.approx(4.0, abs=0.05)

    @pytest.mark.parametrize("sender", [constant, noise])
    def test_uninformative_senders_carry_none(self, sender, meanings):
        assert mutual_information(sender(meanings), _index(meanings)) == pytest.approx(
            0.0, abs=0.02
        )

    def test_miller_madow_shrinks_noise_mi(self, meanings):
        small = meanings[:200]
        raw = mutual_information(noise(small), _index(small), correction="none")
        corrected = mutual_information(noise(small), _index(small))
        assert corrected < raw

    def test_null_test_separates_signal_from_noise(self, meanings):
        sub = meanings[:400]
        signal = mi_null_test(compositional(sub), _index(sub), n_shuffles=100)
        assert signal["mi"] > signal["null_q95"]
        assert signal["p_value"] == pytest.approx(1 / 101)
        null = mi_null_test(noise(sub), _index(sub), n_shuffles=100)
        assert null["p_value"] > 0.05

    def test_per_attribute_mi(self, meanings):
        msgs = compositional(meanings)
        assert mutual_information(msgs[:, :1], meanings[:, 0]) == pytest.approx(
            2.0, abs=0.02
        )
        assert mutual_information(msgs[:, :1], meanings[:, 1]) == pytest.approx(
            0.0, abs=0.02
        )

    def test_batched_shape(self):
        target = np.random.default_rng(0).integers(0, 4, size=(500, 3))
        msgs = target[:, :, None].astype(
            np.int64
        )  # (B, R, 1): receiver r gets target r
        assert mutual_information(msgs, target) == pytest.approx(2.0, abs=0.02)


class TestChannelUsage:
    """Bits used vs capacity."""

    def test_full_and_partial_use(self, meanings):
        full = channel_usage(compositional(meanings), vocab_size=4)
        assert full["capacity_bits"] == 4.0
        assert full["utilization"] == pytest.approx(1.0, abs=0.01)
        half = channel_usage(compositional_disjoint(meanings), vocab_size=8)
        assert half["capacity_bits"] == 6.0
        assert half["utilization"] == pytest.approx(4 / 6, abs=0.01)
        assert half["distinct_messages"] == 16

    def test_constant_uses_nothing(self, meanings):
        usage = channel_usage(constant(meanings), vocab_size=4)
        assert usage["entropy_bits"] == 0.0 and usage["utilization"] == 0.0


class TestStructure:
    """Topographic similarity, posdis and bosdis."""

    def test_topsim(self, meanings):
        sub = meanings[:300]
        assert topographic_similarity(sub, compositional(sub)) == pytest.approx(1.0)
        assert topographic_similarity(sub, holistic(sub)) < 0.5
        assert abs(topographic_similarity(sub, noise(sub))) < 0.1
        assert np.isnan(topographic_similarity(sub, constant(sub)))
        with pytest.raises(ValueError):
            topographic_similarity(sub[:2], compositional(sub[:2]))

    def test_posdis(self, meanings):
        assert positional_disentanglement(meanings, compositional(meanings)) > 0.98
        assert positional_disentanglement(meanings, holistic(meanings)) < 0.5
        assert positional_disentanglement(meanings, noise(meanings)) < 0.05
        assert np.isnan(positional_disentanglement(meanings, constant(meanings)))

    def test_bosdis(self, meanings):
        disjoint = compositional_disjoint(meanings)
        assert bag_of_symbols_disentanglement(meanings, disjoint, 8) > 0.98
        # Shared symbols across positions: counts no longer identify the attribute
        assert (
            bag_of_symbols_disentanglement(meanings, compositional(meanings), 4) < 0.2
        )
        assert bag_of_symbols_disentanglement(meanings, holistic(meanings), 4) < 0.5

    def test_bosdis_exclude_symbols(self, meanings):
        disjoint = compositional_disjoint(meanings)
        value = bag_of_symbols_disentanglement(
            meanings, disjoint, 8, exclude_symbols=[0]
        )
        assert value > 0.98

    def test_need_two_attributes(self, meanings):
        with pytest.raises(ValueError):
            positional_disentanglement(meanings[:, :1], compositional(meanings))
        with pytest.raises(ValueError):
            bag_of_symbols_disentanglement(meanings[:, :1], compositional(meanings), 4)


class TestQuantize:
    """Continuous messages through the discrete metrics."""

    def test_codes_are_balanced_and_informative(self):
        rng = np.random.default_rng(0)
        target = rng.integers(0, 4, size=5000)
        msgs = (target[:, None] + 0.01 * rng.normal(size=(5000, 1))).astype(np.float32)
        codes = quantize_continuous(msgs[:, None, :], n_bins=4)
        assert codes.shape == (5000, 1, 1) and codes.dtype == np.int64
        assert codes.min() >= 0 and codes.max() < 4
        np.testing.assert_allclose(np.bincount(codes.ravel()) / 5000, 0.25, atol=0.01)
        order = np.argsort(msgs[:, 0])
        assert np.all(np.diff(codes.ravel()[order]) >= 0)  # Monotone in the value
        # Quantile edges need not fall exactly between clusters, so a little is lost
        assert 1.85 < mutual_information(codes, target) <= 2.0
        noise_codes = quantize_continuous(
            np.random.default_rng(1).normal(size=(5000, 1, 1))
        )
        assert mutual_information(noise_codes, target) == pytest.approx(0.0, abs=0.02)


def _listener(policy):
    """Listener for messages (B, R, 1) over 4 actions."""

    def listen(msgs):
        m = np.asarray(msgs)[..., 0]
        probs = np.full(m.shape + (4,), 0.0)
        if policy == "obey":
            np.put_along_axis(probs, (m % 4)[..., None], 1.0, axis=-1)
        elif policy == "ignore":
            probs[..., 0] = 1.0
        elif policy == "soft":
            probs += 0.1
            np.put_along_axis(probs, (m % 4)[..., None], 0.7, axis=-1)
        return probs

    return listen


class TestListening:
    """Positive listening: causal influence of communication on actions."""

    @pytest.fixture
    def msgs(self):
        return (
            np.random.default_rng(0).integers(0, 4, size=(200, 2, 1)).astype(np.int64)
        )

    def test_obedient_listener_has_full_cic(self, msgs):
        cic = causal_influence(_listener("obey"), msgs)
        assert cic["cic_bits"] == pytest.approx(entropy(msgs.ravel()), abs=1e-9)
        assert cic["n_candidates"] == 4
        assert len(cic["per_receiver_bits"]) == 2

    def test_ignoring_listener_has_zero_cic(self, msgs):
        assert causal_influence(_listener("ignore"), msgs)["cic_bits"] == pytest.approx(
            0.0
        )

    def test_soft_listener_in_between(self, msgs):
        cic = causal_influence(
            _listener("soft"), msgs, candidates=np.arange(4)[:, None]
        )
        p = np.array([0.7, 0.1, 0.1, 0.1])
        expected = np.log2(4) - (-(p * np.log2(p)).sum())  # H(A) - H(A|M)
        assert cic["cic_bits"] == pytest.approx(expected)

    def test_zero_weight_candidates_ignored(self, msgs):
        cic = causal_influence(
            _listener("obey"),
            msgs,
            candidates=np.arange(4)[:, None],
            weights=[1, 1, 0, 0],
        )
        assert cic["cic_bits"] == pytest.approx(1.0)
        assert cic["n_candidates"] == 2

    def test_bad_listener_output(self, msgs):
        with pytest.raises(ValueError):
            causal_influence(lambda m: np.ones((3, 4)), msgs)

    def test_intervention_influence(self, msgs):
        obey = intervention_influence(
            _listener("obey"), msgs, "shuffle_batch", n_repeats=3
        )
        assert 0.6 < obey["changed_fraction"] < 0.9  # ~3/4 of shuffled symbols differ
        assert obey["js_bits"] == pytest.approx(obey["changed_fraction"])
        ignore = intervention_influence(_listener("ignore"), msgs, "random_matched")
        assert ignore["js_bits"] == 0.0 and ignore["changed_fraction"] == 0.0

    def test_mask_averages_over_live_receivers_only(self, msgs):
        """Dead receivers (constant action, junk message 0) must not dilute the mean."""
        mask = np.ones(msgs.shape[:2], dtype=bool)
        mask[::2, 1] = False
        dead = msgs.copy()
        dead[~mask] = 0

        def listener(m):
            probs = _listener("obey")(m)
            probs[~mask] = np.eye(4)[0]
            return probs

        cic = causal_influence(listener, dead, mask=mask)
        live_msgs = msgs[mask]
        assert cic["cic_bits"] == pytest.approx(entropy(live_msgs.ravel()), abs=1e-9)
        assert cic["per_receiver_bits"][1] == pytest.approx(cic["cic_bits"], abs=0.1)
        unmasked = causal_influence(listener, dead)["cic_bits"]
        assert unmasked < cic["cic_bits"]

        infl = intervention_influence(
            listener, dead, "shuffle_batch", n_repeats=3, mask=mask
        )
        assert 0.6 < infl["changed_fraction"] < 0.9
        assert infl["js_bits"] == pytest.approx(infl["changed_fraction"])
        with pytest.raises(ValueError, match="live"):
            causal_influence(listener, dead, mask=np.zeros_like(mask))

    def test_listening_summary(self):
        cic = {"cic_bits": 1.5}
        verdict = listening_summary(
            cic, {"zero": {"listening": True}, "shuffle_batch": {"listening": False}}
        )
        assert verdict["positive_listening"]
        assert verdict["ablations_with_drop"] == ["zero"]
        assert not listening_summary({"cic_bits": 0.0}, {"zero": {"listening": True}})[
            "positive_listening"
        ]


class TestLearnability:
    """Steps for a fresh learner to reach a fraction of a reference."""

    def test_steps_to_threshold(self):
        steps = [0, 10, 20, 30, 40]
        assert steps_to_threshold(steps, [0, 0.5, 0.9, 1.0, 1.0], 0.8) == 20
        assert steps_to_threshold(steps, [0, 0.1, 0.2, 0.3, 0.4], 0.8) is None
        # A single spike does not count once smoothed
        assert steps_to_threshold(steps, [0, 1.0, 0, 0.9, 0.9], 0.8, window=2) == 40
        assert steps_to_threshold([], [], 0.5) is None
        with pytest.raises(ValueError):
            steps_to_threshold([0, 1], [0], 0.5)

    def test_learnability(self):
        steps = np.arange(0, 100, 10)
        fast = (steps, np.minimum(1.0, steps / 30))
        slow = (steps, np.minimum(1.0, steps / 60))
        never = (steps, np.zeros_like(steps, dtype=float))
        out = learnability([fast, slow, never], reference=1.0, fraction=0.5)
        assert out["threshold"] == 0.5
        assert out["steps_per_curve"] == [20.0, 30.0, None]
        assert out["fraction_reached"] == pytest.approx(2 / 3)
        assert out["mean_steps"] == 25.0


class TestCommunicationMetrics:
    """The standard bundle on every synthetic sender."""

    def test_bundle_known_answers(self, meanings):
        sub, target = meanings[:500], _index(meanings[:500])
        comp = communication_metrics(
            compositional(sub), target, sub, vocab_size=4, n_shuffles=50
        )
        assert comp["mi_bits"] > comp["mi_null_q95_bits"]
        assert comp["topsim"] == pytest.approx(1.0)
        assert comp["posdis"] > 0.95
        assert comp["utilization"] > 0.95

        const = communication_metrics(
            constant(sub), target, sub, vocab_size=4, n_shuffles=50
        )
        assert const["message_entropy_bits"] == 0.0 and const["mi_bits"] == 0.0
        assert np.isnan(const["topsim"]) and np.isnan(const["posdis"])

        rand = communication_metrics(
            noise(sub), target, sub, vocab_size=4, n_shuffles=50
        )
        assert rand["mi_p_value"] > 0.05
        assert rand["utilization"] > 0.95  # Uses the channel, says nothing

    def test_minimal_bundle(self, meanings):
        out = communication_metrics(compositional(meanings))
        assert set(out) == {"message_entropy_bits"}


def test_package_is_domain_agnostic():
    """evaluation/ must not import any environment, protocol or runtime package."""
    package = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "genetic_coil_arena"
        / "evaluation"
    )
    forbidden = {
        "snake_core",
        "envs",
        "protocol",
        "runtime",
        "evo",
        "analysis",
        "config",
    }
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 2 and node.module == "messages":
                    continue  # The shared, dependency-free message conventions
                parts = set((node.module or "").split("."))
                if node.level >= 2 or node.module is None:
                    parts |= {alias.name for alias in node.names}
                assert not parts & forbidden, f"{path.name} imports {node.module}"
                assert node.level <= 1, f"{path.name} imports from outside the package"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not set(alias.name.split(".")) & forbidden, path.name
