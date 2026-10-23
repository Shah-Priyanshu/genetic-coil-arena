"""Tests for the plant-floor sensor squad environment (envs.plant): API, budget, labels."""

import copy

import numpy as np
import pytest

from genetic_coil_arena.envs.base import (
    EnvSpec,
    MessageSpec,
    MultiAgentEnv,
    Observations,
    StepResult,
)
from genetic_coil_arena.envs.plant import (
    NODE_VARIABLES,
    NODES,
    ONSET_TEST,
    PlantConfig,
    PlantEnv,
    synthetic_tep,
)

FAULTS = (1, 2, 5)
N = len(NODES)


@pytest.fixture(scope="module")
def data():
    return synthetic_tep(seed=0, faults=FAULTS)


def make_env(data, **kwargs):
    kwargs.setdefault("faults", FAULTS)
    return PlantEnv(PlantConfig(**kwargs), data)


def silent_step(env, diagnosis=None, messages=None):
    return env.step(None, messages, diagnosis)


class TestApi:
    def test_is_the_shared_multi_agent_env_with_spec(self, data):
        env = make_env(data, budget=16)
        assert isinstance(env, MultiAgentEnv) and isinstance(env.spec, EnvSpec)
        assert env.spec.n_specialists == N == 4
        assert env.spec.n_actions == 1
        assert env.spec.message == MessageSpec.discrete(16, 256, n_receivers=4)
        assert env.spec.max_steps == 95  # (960 - 20) / 10 + 1 windows per test run

    def test_message_channel_follows_the_shared_conventions(self, data):
        for budget in (8, 16, 51):
            spec = make_env(data, budget=budget).spec.message
            assert spec.kind == "discrete" and spec.vocab_size == 256
            assert spec.payload_shape == (budget,) and spec.dtype == np.int64
            assert spec.bits == pytest.approx(8.0 * budget)  # one byte per symbol

    def test_zero_bytes_is_a_none_channel(self, data):
        env = make_env(data, budget=0)
        assert env.spec.message.kind == "none"
        obs = env.reset(range(env.n_runs))
        assert obs.centre["uplink"].shape == (env.n_runs, N, 0)

    def test_observation_shapes_and_dtypes(self, data):
        env = make_env(data, budget=16)
        obs = env.reset(range(env.n_runs))
        b = env.n_runs
        assert isinstance(obs, Observations) and obs.batch_size == b == 4
        win, mask = obs.specialists["window"], obs.specialists["mask"]
        assert win.shape == (b, N, 18, 20) and win.dtype == np.float32
        assert mask.shape == (b, N, 18) and mask.dtype == np.bool_
        assert mask[0].sum(axis=1).tolist() == [len(NODE_VARIABLES[n]) for n in NODES]
        assert np.all(win[~mask] == 0)  # padded rows carry nothing
        up = obs.centre["uplink"]
        assert up.shape == (b, N, 16) and up.dtype == np.int64 and not up.any()
        for name, (shape, dtype) in env.spec.specialist_obs.items():
            assert obs.specialists[name].shape[2:] == shape
            assert obs.specialists[name].dtype == dtype
        assert obs.centre["uplink"].shape[1:] == env.spec.centre_obs["uplink"][0]

    def test_the_centre_sees_only_the_uplink(self, data):
        env = make_env(data, budget=16)
        obs = env.reset([0, 1])
        assert set(obs.centre) == {"uplink"}
        assert set(obs.specialists) == {"window", "mask"}

    def test_step_result_conventions(self, data):
        env = make_env(data, budget=8)
        env.reset(range(env.n_runs))
        msgs = np.zeros((env.n_runs, N, 8), dtype=np.int64)
        result = env.step(np.zeros((env.n_runs, N), dtype=np.int64), msgs)
        assert isinstance(result, StepResult)
        obs, rewards, dones, info = result
        assert rewards.shape == (env.n_runs, N) and rewards.dtype == np.float32
        assert dones.shape == (env.n_runs,) and dones.dtype == np.bool_
        for key in ("label", "diagnosis", "n_seen", "steps"):
            assert info[key].shape == (env.n_runs,) and info[key].dtype == np.int64
        for key in ("correct", "alarm", "false_alarm"):
            assert info[key].dtype == np.bool_

    def test_nodes_take_only_the_noop_action(self, data):
        env = make_env(data, budget=8)
        env.reset([0])
        env.step(np.zeros((1, N), dtype=np.int64))
        with pytest.raises(ValueError):
            env.step(np.ones((1, N), dtype=np.int64))
        with pytest.raises(ValueError):
            env.step(np.zeros((1, N + 1), dtype=np.int64))

    def test_step_before_reset_fails(self, data):
        with pytest.raises(RuntimeError):
            make_env(data).step(None)

    def test_seeds_wrap_onto_runs(self, data):
        env = make_env(data)
        a = env.reset([0, 1, 2, 3])
        b = env.reset([4, 5, 6, 7])
        assert np.array_equal(a.specialists["window"], b.specialists["window"])
        with pytest.raises(ValueError):
            env.reset([])

    def test_config_validation(self, data):
        for bad in (
            {"budget": -1},
            {"budget": 223},
            {"window": 1},
            {"stride": 0},
            {"split": "val"},
            {"faults": (1, 1)},
            {"faults": (22,)},
            {"faults": (), "include_normal": False},
        ):
            with pytest.raises(ValueError):
                PlantConfig(**bad)


class TestBudgetIsEnforced:
    @pytest.mark.parametrize("budget", [8, 16, 51])
    def test_exact_size_accepted_and_delivered(self, data, budget):
        env = make_env(data, budget=budget)
        env.reset([0, 1])
        msgs = np.random.default_rng(0).integers(0, 256, size=(2, N, budget))
        obs = silent_step(env, messages=msgs).obs
        assert np.array_equal(obs.centre["uplink"], msgs)
        assert obs.centre["uplink"].dtype == np.int64

    @pytest.mark.parametrize("budget", [8, 16, 51])
    def test_too_many_or_too_few_bytes_rejected(self, data, budget):
        env = make_env(data, budget=budget)
        env.reset([0, 1])
        for length in (budget + 1, budget - 1, 2 * budget):
            with pytest.raises(ValueError, match="shape"):
                silent_step(env, messages=np.zeros((2, N, length), dtype=np.int64))

    def test_symbols_must_be_bytes(self, data):
        env = make_env(data, budget=8)
        env.reset([0, 1])
        shape = (2, N, 8)
        with pytest.raises(ValueError, match=r"\[0, 256\)"):
            silent_step(env, messages=np.full(shape, 256))
        with pytest.raises(ValueError, match=r"\[0, 256\)"):
            silent_step(env, messages=np.full(shape, -1))
        with pytest.raises(ValueError, match="integers"):
            silent_step(env, messages=np.zeros(shape, dtype=np.float32))
        ok = silent_step(env, messages=np.full(shape, 255)).obs
        assert ok.centre["uplink"].max() == 255

    def test_wrong_node_or_batch_count_rejected(self, data):
        env = make_env(data, budget=8)
        env.reset([0, 1])
        with pytest.raises(ValueError):
            silent_step(env, messages=np.zeros((2, N - 1, 8), dtype=np.int64))
        with pytest.raises(ValueError):
            silent_step(env, messages=np.zeros((3, N, 8), dtype=np.int64))

    def test_zero_budget_allows_no_bytes(self, data):
        env = make_env(data, budget=0)
        env.reset([0, 1])
        with pytest.raises(ValueError, match="0 bytes"):
            silent_step(env, messages=np.zeros((2, N, 1), dtype=np.int64))
        obs = silent_step(env, messages=np.zeros((2, N, 0), dtype=np.int64)).obs
        assert obs.centre["uplink"].shape == (2, N, 0)
        silent_step(env)  # None is silence

    def test_silent_nodes_send_zeros(self, data):
        env = make_env(data, budget=8)
        env.reset([0, 1])
        silent_step(env, messages=np.full((2, N, 8), 7))
        obs = silent_step(env).obs  # messages=None: silence replaces the last uplink
        assert not obs.centre["uplink"].any()

    def test_diagnosis_is_checked(self, data):
        env = make_env(data, budget=8)
        env.reset([0, 1])
        n_classes = env.config.n_classes
        for bad in (np.full(2, n_classes), np.full(2, -1), np.zeros(3, np.int64)):
            with pytest.raises(ValueError):
                silent_step(env, diagnosis=bad)
        with pytest.raises(ValueError):
            silent_step(env, diagnosis=np.zeros(2, dtype=np.float32))


class TestLabelsAndOnset:
    def spiked(self, data, fault=1, row=ONSET_TEST):
        """Copy of the data with a recognisable spike at one sample of a test run."""
        spiked = copy.deepcopy(data)
        spiked.fault_test[fault][row, :] = 1e6
        return spiked

    @pytest.mark.parametrize("window,stride", [(20, 10), (15, 7), (20, 20), (10, 1)])
    def test_labels_flip_exactly_when_a_window_contains_sample_160(
        self, data, window, stride
    ):
        env = make_env(self.spiked(data), window=window, stride=stride, faults=(1,))
        out = env.labelled_windows("test")
        r = int(np.flatnonzero(env.run_faults == 1)[0])
        sel = out["run"] == r
        labels, seen = out["labels"][sel], out["n_seen"][sel]
        windows = out["windows"][sel]
        spike = windows.max(axis=(1, 2, 3)) > 1e3
        first = int(np.argmax(labels != 0))
        assert labels[first] == 1  # the fault class (index 1 in (0, 1))
        assert first > 0 and not np.any(labels[:first])
        assert np.all(labels[first:] == 1)
        # The first faulty window is the first one that has seen a faulty sample
        assert seen[first - 1] <= ONSET_TEST < seen[first]
        assert seen[first] == min(s for s in seen if s > ONSET_TEST)
        # ... and that window is also the first one that contains the spike
        assert spike[first] and not spike[:first].any()

    @pytest.mark.parametrize(
        "row,first_seen,label", [(159, 160, 0), (160, 170, 1), (161, 170, 1)]
    )
    def test_onset_is_sample_160_not_159(self, data, row, first_seen, label):
        # Window k covers samples [n_seen - 20, n_seen): a spike at sample 159 first shows
        # in the window that has seen 160 samples, which is still normal; one at sample 160
        # first shows in the window that has seen 170, which is the first faulty one.
        env = make_env(self.spiked(data, row=row), window=20, stride=10, faults=(1,))
        out = env.labelled_windows("test")
        r = int(np.flatnonzero(env.run_faults == 1)[0])
        sel = out["run"] == r
        spike = out["windows"][sel].max(axis=(1, 2, 3)) > 1e3
        first = int(np.argmax(spike))
        assert out["n_seen"][sel][first] == first_seen
        assert out["labels"][sel][first] == label

    def test_normal_run_is_all_normal_and_fault_runs_start_normal(self, data):
        env = make_env(data)
        out = env.labelled_windows("test")
        assert env.run_faults.tolist() == [0, 1, 2, 5]
        normal = out["labels"][out["run"] == 0]
        assert normal.shape == (95,) and not normal.any()
        for r in range(1, env.n_runs):
            labels = out["labels"][out["run"] == r]
            seen = out["n_seen"][out["run"] == r]
            assert np.all(labels[seen <= ONSET_TEST] == 0)
            assert np.all(labels[seen > ONSET_TEST] == r)  # class index of the run
            assert (seen <= ONSET_TEST).sum() == 15

    def test_train_runs_are_faulty_from_the_first_sample(self, data):
        env = make_env(data, split="train")
        out = env.labelled_windows("train")
        for r in range(1, env.n_runs):
            assert np.all(out["labels"][out["run"] == r] == r)
        assert not out["labels"][out["run"] == 0].any()

    def test_class_indices_follow_the_fault_order(self, data):
        env = make_env(data, faults=(5, 1, 2))
        assert env.config.classes == (0, 5, 1, 2)
        out = env.labelled_windows("test")
        assert set(np.unique(out["labels"])) == {0, 1, 2, 3}
        r = int(np.flatnonzero(env.run_faults == 5)[0])
        assert out["labels"][out["run"] == r].max() == 1

    def test_env_labels_match_the_window_just_observed(self, data):
        env = make_env(data)
        env.reset(range(env.n_runs))
        seen_at_start = env.labels()
        assert not seen_at_start.any()  # first window (20 samples seen) is normal
        for _ in range(env.spec.max_steps):
            assert env.labels().shape == (env.n_runs,)
            _, _, done, _ = silent_step(env)
        assert done.all()


class TestScoringAndDelay:
    def play(self, env, policy):
        env.reset(range(env.n_runs))
        total = np.zeros(env.n_runs)
        while True:
            diagnosis = policy(env)
            _, rewards, done, info = silent_step(env, diagnosis)
            total += rewards[:, 0]
            if done.all():
                return total, info

    def test_truthful_centre_is_always_right_and_detects_one_stride_late(self, data):
        env = make_env(data)
        total, info = self.play(env, lambda e: e.labels())
        assert np.all(total == 95)
        delay = env.detection_delay
        assert np.isnan(delay[0])  # the normal run has nothing to detect
        assert np.all(delay[1:] == 10)  # first faulty window has seen 170 samples
        assert np.array_equal(env.diagnosis_delay[1:], delay[1:])
        assert not env.false_alarms.any()
        assert env.normal_windows.tolist() == [95, 15, 15, 15]

    def test_silence_never_detects_and_never_falsely_alarms(self, data):
        env = make_env(data)
        total, _ = self.play(env, lambda e: np.zeros(e.n_runs, dtype=np.int64))
        assert (
            np.isnan(env.detection_delay).all() and np.isnan(env.diagnosis_delay).all()
        )
        assert not env.false_alarms.any()
        assert total.tolist() == [95, 15, 15, 15]  # only the normal windows are right

    def test_constant_alarm_is_all_false_alarms_on_normal_windows(self, data):
        env = make_env(data)
        self.play(env, lambda e: np.ones(e.n_runs, dtype=np.int64))
        assert env.false_alarms.tolist() == [95, 15, 15, 15]
        assert np.all(env.detection_delay[1:] == 10)
        # Class 1 is only right for the fault-1 run
        assert env.diagnosis_delay[1] == 10 and np.isnan(env.diagnosis_delay[2:]).all()

    def test_episodes_end_after_the_last_window_and_then_freeze(self, data):
        env = make_env(data)
        env.reset(range(env.n_runs))
        for _ in range(94):
            _, _, done, _ = silent_step(env)
            assert not done.any()
        _, _, done, _ = silent_step(env, env.labels())
        assert done.all()
        before = (env.false_alarms.copy(), env.correct_windows.copy())
        _, rewards, done, _ = silent_step(env, np.ones(env.n_runs, dtype=np.int64))
        assert done.all() and not rewards.any()
        assert np.array_equal(before[0], env.false_alarms)
        assert np.array_equal(before[1], env.correct_windows)

    def test_train_split_runs_have_different_lengths(self, data):
        env = make_env(data, split="train")
        assert env.spec.max_steps == 49  # the 500-sample normal run
        env.reset(range(env.n_runs))
        dones = []
        for _ in range(49):
            dones.append(silent_step(env).dones)
        first_done = [int(np.argmax([d[i] for d in dones])) for i in range(env.n_runs)]
        assert first_done == [48, 46, 46, 46]  # 49 windows vs 47


class TestDeterminism:
    def test_same_seeds_same_observations(self, data):
        a, b = make_env(data), make_env(data)
        oa, ob = a.reset([3, 0, 2]), b.reset([3, 0, 2])
        assert np.array_equal(oa.specialists["window"], ob.specialists["window"])
        rng = np.random.default_rng(0)
        for _ in range(5):
            msgs = rng.integers(0, 256, size=(3, N, 16))
            oa = silent_step(a, np.array([0, 1, 2]), msgs).obs
            ob = silent_step(b, np.array([0, 1, 2]), msgs).obs
            assert np.array_equal(oa.specialists["window"], ob.specialists["window"])
            assert np.array_equal(oa.centre["uplink"], ob.centre["uplink"])

    def test_reset_replays_the_episode(self, data):
        env = make_env(data)
        first = env.reset([1]).specialists["window"].copy()
        for _ in range(10):
            silent_step(env)
        again = env.reset([1]).specialists["window"]
        assert np.array_equal(first, again)

    def test_observations_are_copies(self, data):
        env = make_env(data)
        obs = env.reset([0])
        obs.specialists["window"][:] = 123.0
        assert not np.any(env.observe().specialists["window"] == 123.0)

    def test_environment_does_not_alter_the_data(self, data):
        before = data.fault_test[1].copy()
        env = make_env(data)
        env.reset(range(env.n_runs))
        silent_step(env)
        assert np.array_equal(before, data.fault_test[1])


class TestSharedLayers:
    def test_uplink_round_trips_through_the_protocol_envelope_codecs(self, data):
        from genetic_coil_arena.protocol import codes_to_envelopes, envelopes_to_codes

        env = make_env(data, budget=16)
        env.reset(range(env.n_runs))
        msgs = np.random.default_rng(0).integers(0, 256, size=(env.n_runs, N, 16))
        uplink = silent_step(env, messages=msgs).obs.centre["uplink"]
        # One envelope per (episode, node): node i is the sender, the gateway the recipient
        batches = [
            codes_to_envelopes(uplink[:, i : i + 1], 256, NODES[i], ["gateway"])
            for i in range(N)
        ]
        assert batches[0][0][0].payload_encoding["V"] == 256
        assert batches[0][0][0].payload_encoding["L"] == 16
        back = np.concatenate([envelopes_to_codes(b, ["gateway"]) for b in batches], 1)
        assert back.dtype == np.int64 and np.array_equal(back, uplink)
