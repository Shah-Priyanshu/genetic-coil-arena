"""Test the task runner, mandatory baselines, ablations and the standard report.

The toy task is a referential game: each episode has a target in [0, 4); a sender
puts it in a message, the environment loop applies the hook, and the receiver
guesses. The checker scores a correct guess as 1.
"""

import json

import numpy as np
import pytest

from genetic_coil_arena.evaluation import (
    ABLATIONS,
    BaselineRegistry,
    ExperimentResult,
    MessageSpec,
    Task,
    TaskSpec,
    apply_hook,
    communication_metrics,
    default_ablations,
    evaluate,
    mann_whitney,
    register_baseline,
    run_ablations,
    run_experiment,
    wilcoxon_signed_rank,
)

N_EPISODES, N_TARGETS = 64, 4


def make_episodes(seed):
    rng = np.random.default_rng(seed)
    return {"target": rng.integers(0, N_TARGETS, size=N_EPISODES)}


def accuracy(inputs, outputs):
    return (outputs == inputs["target"]).astype(float)


def listener_system(inputs, seed, hook=None):
    """Sends the target and decodes it: performance depends on the message."""
    messages = inputs["target"][:, None, None].astype(np.int64)  # (B, R=1, L=1)
    received = apply_hook(hook, messages)
    return received[:, 0, 0]


def deaf_system(inputs, seed, hook=None):
    """Sends a message but always guesses 0: ablations cannot hurt it."""
    apply_hook(hook, inputs["target"][:, None, None].astype(np.int64))
    return np.zeros(N_EPISODES, dtype=np.int64)


def random_baseline(inputs, seed, hook=None):
    return np.random.default_rng(seed + 1000).integers(0, N_TARGETS, size=N_EPISODES)


def oracle_baseline(inputs, seed, hook=None):
    return inputs["target"].copy()


# One receiver, so the default ablations leave out swap_receivers
CHANNEL = MessageSpec.discrete(1, N_TARGETS, n_receivers=1)
SINGLE = list(default_ablations(1))


def make_task(**spec_kwargs):
    spec_kwargs.setdefault("message", CHANNEL)
    spec = TaskSpec("referential", n_seeds=8, **spec_kwargs)
    return Task(spec, make_episodes, accuracy)


def team_system(inputs, seed, hook=None):
    """Three receivers each get the target plus their index; all must decode it."""
    target = inputs["target"]
    sent = (target[:, None] + np.arange(3)[None, :])[:, :, None].astype(np.int64)
    received = apply_hook(hook, sent)[:, :, 0] - np.arange(3)[None, :]
    return np.where((received == target[:, None]).all(axis=1), target, -1)


@pytest.fixture
def registry():
    reg = BaselineRegistry()
    reg.register("referential", "random", random_baseline)
    return reg


class TestTask:
    """Task spec, inputs and scoring."""

    def test_spec_seeds(self):
        assert TaskSpec("t", n_seeds=3, seed_offset=100).seeds() == [100, 101, 102]
        assert TaskSpec("t").seeds(2) == [0, 1]
        with pytest.raises(ValueError):
            TaskSpec("")
        with pytest.raises(ValueError):
            TaskSpec("t", n_seeds=0)

    def test_fixed_inputs(self):
        task = Task(
            TaskSpec("fixed"), {"target": np.zeros(N_EPISODES, dtype=int)}, accuracy
        )
        assert task.score(deaf_system, seed=5) == (1.0, N_EPISODES)

    def test_scalar_checker(self):
        task = Task(TaskSpec("scalar"), make_episodes, lambda i, o: 0.5)
        assert task.score(deaf_system, 0) == (0.5, 1)

    def test_non_finite_scores_raise(self):
        task = Task(TaskSpec("bad"), make_episodes, lambda i, o: [1.0, np.nan])
        with pytest.raises(ValueError, match="non-finite"):
            task.score(deaf_system, 0)

    def test_evaluate(self):
        result = evaluate(make_task(), listener_system, [0, 1, 2], name="sys")
        assert result.per_seed == [1.0, 1.0, 1.0]
        assert result.mean == 1.0 and result.ci == 0.0 and result.iqm == 1.0
        assert result.n_episodes == 3 * N_EPISODES
        with pytest.raises(ValueError):
            evaluate(make_task(), listener_system, [])


class TestBaselineRegistry:
    """Mandatory baselines."""

    def test_register_and_required(self, registry):
        assert "referential" in registry
        assert registry.required("referential") == ["random"]
        registry.unregister("referential", "random")
        assert "referential" not in registry

    def test_decorator_registration(self):
        reg = BaselineRegistry()

        @register_baseline("task", "always_zero", registry=reg)
        def zero(inputs, seed, hook=None):
            return 0

        assert reg.baselines("task") == {"always_zero": zero}
        with pytest.raises(TypeError):
            reg.register("task", "bad", 3)

    def test_experiment_requires_baselines(self):
        with pytest.raises(ValueError, match="no mandatory baselines"):
            run_experiment(
                make_task(), {"sys": listener_system}, registry=BaselineRegistry()
            )

    def test_non_strict_report_is_incomplete(self):
        result = run_experiment(
            make_task(),
            {"sys": listener_system},
            registry=BaselineRegistry(),
            strict=False,
            ablations=None,
        )
        assert not result.complete
        assert "INCOMPLETE" in result.format_table()
        with pytest.raises(ValueError):
            result.require_complete()

    def test_task_baselines_merge_with_registry(self, registry):
        task = make_task()
        task.baselines["oracle"] = oracle_baseline
        result = run_experiment(
            task, {"sys": listener_system}, registry=registry, ablations=None
        )
        assert result.required_baselines == ["random", "oracle"]
        assert result.complete and result.require_complete() is result

    def test_missing_baseline_detected(self, registry):
        result = run_experiment(
            make_task(), {"sys": listener_system}, registry=registry, ablations=None
        )
        result.conditions = [c for c in result.conditions if c.name != "random"]
        assert result.missing_baselines == ["random"]
        assert not result.complete

    def test_name_clash(self, registry):
        with pytest.raises(ValueError, match="both"):
            run_experiment(make_task(), {"random": listener_system}, registry=registry)


class TestAblations:
    """Ablation deltas detect listening and only listening."""

    def test_listener_loses_performance(self):
        results = run_ablations(
            make_task(), listener_system, list(range(8)), name="sys"
        )
        assert [r.intervention for r in results] == SINGLE
        for r in results:
            assert r.listening, r.intervention
            assert r.intact_mean == 1.0
            assert r.delta_mean > 0.5  # Chance is 0.25
            assert r.delta_low > 0
            assert r.a12 == 1.0
            assert r.p_holm >= r.p
            assert r.ablated.role == "ablation"
            assert r.test == "wilcoxon"
            assert r.p == wilcoxon_signed_rank([1.0] * 8, r.ablated.per_seed)[1]

    def test_default_ablations_swap_receivers_when_several(self):
        assert "swap_receivers" in ABLATIONS
        assert "swap_receivers" not in default_ablations(1)
        assert default_ablations(3) == ABLATIONS == default_ablations(None)
        task = make_task(message=MessageSpec.discrete(1, N_TARGETS + 2, n_receivers=3))
        results = run_ablations(task, team_system, list(range(6)))
        assert [r.intervention for r in results] == list(ABLATIONS)
        assert all(r.listening for r in results)

    def test_swap_receivers_on_one_receiver_explains_itself(self):
        task = make_task(message=None)  # Receiver count unknown: all four are tried
        with pytest.raises(ValueError, match="n_receivers"):
            run_ablations(task, listener_system, [0, 1])

    def test_hooks_validate_against_the_task_channel(self):
        task = make_task(message=MessageSpec.discrete(2, N_TARGETS, n_receivers=1))
        with pytest.raises(ValueError, match="shape"):
            run_ablations(task, listener_system, [0, 1], ["zero"])

    def test_paired_test_sees_what_mann_whitney_misses(self):
        """A small, consistent per-seed drop under large between-seed spread."""
        levels = {seed: 0.1 * seed for seed in range(10)}

        def system(inputs, seed, hook=None):
            sent = np.zeros((4, 1, 1), dtype=np.int64)
            ablated = hook is not None and apply_hook(hook, sent + 1).max() == 0
            return np.full(4, levels[seed] - (0.02 + 0.001 * seed) * ablated)

        task = Task(
            TaskSpec("spread", n_seeds=10, message=CHANNEL),
            lambda seed: None,
            lambda i, o: o,
        )
        (zero,) = run_ablations(task, system, list(range(10)), ["zero"])
        intact = [levels[s] for s in range(10)]
        assert zero.p < 0.01 and zero.listening
        assert mann_whitney(intact, zero.ablated.per_seed)[1] > 0.5

    def test_deaf_system_unaffected(self):
        results = run_ablations(make_task(), deaf_system, list(range(8)))
        for r in results:
            assert not r.listening
            assert r.delta_mean == 0.0

    def test_ablations_are_deterministic(self):
        a = run_ablations(make_task(), listener_system, [0, 1, 2], ["shuffle_batch"])
        b = run_ablations(make_task(), listener_system, [0, 1, 2], ["shuffle_batch"])
        assert a[0].ablated.per_seed == b[0].ablated.per_seed

    def test_lower_is_better_orientation(self):
        task = Task(
            TaskSpec("errors", higher_is_better=False),
            make_episodes,
            lambda i, o: (o != i["target"]).astype(float),
        )
        results = run_ablations(task, listener_system, list(range(6)), ["zero"])
        assert results[0].delta_mean > 0 and results[0].listening

    def test_seed_mismatch(self):
        intact = evaluate(make_task(), listener_system, [0, 1])
        with pytest.raises(ValueError):
            run_ablations(make_task(), listener_system, [0, 2], intact=intact)


class TestExperiment:
    """End to end: run, compare, ablate, record messages, report."""

    @pytest.fixture
    def result(self, registry):
        result = run_experiment(
            make_task(),
            {"listener": listener_system, "deaf": deaf_system},
            registry=registry,
            ablate=["listener"],
            metadata={"note": "toy"},
            record=True,
        )
        task = make_task()
        listener = result.condition("listener")
        targets = [task.inputs(seed)["target"] for seed in result.seeds]
        messages = np.concatenate([rec.stacked() for rec in listener.recordings])
        result.add_communication(
            "listener",
            communication_metrics(
                messages, np.concatenate(targets), vocab_size=N_TARGETS, n_shuffles=50
            ),
        )
        return result

    def test_recordings_and_outputs_come_from_the_scored_runs(self, registry):
        result = run_experiment(
            make_task(),
            {"listener": listener_system},
            registry=registry,
            keep_outputs=True,
            record=True,
        )
        task = make_task()
        listener = result.condition("listener")
        assert len(listener.outputs) == len(listener.recordings) == 8
        for seed, out, rec in zip(result.seeds, listener.outputs, listener.recordings):
            target = task.inputs(seed)["target"]
            assert np.array_equal(out, target)
            assert np.array_equal(rec.stacked()[:, 0, 0], target)
        assert result.condition("random").recordings[0].records == []  # Ignores hooks
        assert all(a.ablated.outputs is None for a in result.ablations)
        assert "outputs" not in result.to_dict()["conditions"][0]
        plain = evaluate(make_task(), listener_system, [0])
        assert plain.outputs is None and plain.recordings is None

    def test_recording_wraps_the_intervention_hook(self):
        results = run_ablations(
            make_task(), listener_system, [0, 1], ["zero"], record=True
        )
        rec = results[0].ablated.recordings[0]
        assert rec.inner is not None and rec.stacked().any()  # Recorded as sent
        expected = [float(np.mean(make_episodes(s)["target"] == 0)) for s in (0, 1)]
        assert results[0].ablated.per_seed == expected

    def test_conditions_and_comparisons(self, result):
        assert [c.name for c in result.systems] == ["listener", "deaf"]
        assert [c.name for c in result.baselines] == ["random"]
        rows = {(r["a"], r["b"]): r for r in result.comparisons}
        assert rows[("listener", "random")]["a12"] == 1.0
        assert rows[("listener", "random")]["p_holm"] < 0.01
        assert rows[("listener", "random")]["mean_diff"] > 0.5
        assert len(result.ablations) == len(SINGLE)
        assert all(a.system == "listener" for a in result.ablations)
        assert result.communication["listener"]["mi_bits"] > 1.9

    def test_comparison_orientation_lower_is_better(self, registry):
        task = Task(
            TaskSpec("referential", n_seeds=6, higher_is_better=False),
            make_episodes,
            lambda i, o: (o != i["target"]).astype(float),
        )
        result = run_experiment(
            task, {"listener": listener_system}, registry=registry, ablations=None
        )
        row = result.comparisons[0]
        assert row["a12"] == 1.0  # Listener is better
        assert row["mean_diff"] < 0  # Raw error difference

    def test_unknown_ablate_target(self, registry):
        with pytest.raises(KeyError):
            run_experiment(
                make_task(),
                {"sys": listener_system},
                registry=registry,
                ablate=["other"],
            )

    def test_json_round_trip(self, result, tmp_path):
        path = tmp_path / "result.json"
        text = result.to_json(path)
        data = json.loads(text)
        assert data["complete"] is True and data["missing_baselines"] == []
        loaded = ExperimentResult.from_json(path)
        assert loaded.to_dict() == result.to_dict()
        assert ExperimentResult.from_json(text).to_dict() == result.to_dict()
        assert (
            loaded.ablations[0].ablated.per_seed == result.ablations[0].ablated.per_seed
        )

    def test_json_handles_nan(self, registry):
        result = run_experiment(
            make_task(),
            {"sys": listener_system},
            seeds=[0],
            registry=registry,
            ablations=None,
        )
        assert np.isnan(result.conditions[0].ci)
        data = json.loads(result.to_json())  # allow_nan=False would raise on NaN
        assert data["conditions"][0]["ci"] is None
        assert np.isnan(ExperimentResult.from_dict(data).conditions[0].ci)

    def test_csv(self, result, tmp_path):
        import csv

        path = result.to_csv(tmp_path / "result.csv")
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        sections = [r["section"] for r in rows]
        assert sections.count("system") == 2 and sections.count("baseline") == 1
        assert sections.count("ablation") == len(SINGLE)
        assert sections.count("comparison") == 2
        assert "communication" in sections
        listener = next(r for r in rows if r["name"] == "listener")
        assert float(listener["mean"]) == 1.0
        assert all(r["task"] == "referential" for r in rows)

    def test_text_table(self, result):
        text = result.format_table()
        assert text == str(result)
        for needle in (
            "Task: referential",
            "Baselines: random",
            "listener",
            "deaf",
            "shuffle_batch",
            "listener vs random",
            "mi_bits",
            "IQM",
        ):
            assert needle in text
        assert "INCOMPLETE" not in text
