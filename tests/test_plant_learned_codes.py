"""Tests for the learned plant codes: byte budgets, determinism, merging and the G2 verdict.

Synthetic and fast. Torch-dependent tests skip when torch is absent; the real-data check
skips without the Rieth cache.
"""

import importlib.util
import json

import numpy as np
import pytest

from genetic_coil_arena.envs.plant.learned_codes import (
    METHODS,
    Featuriser,
    kmeans,
    node_features,
    pack_bits,
    unpack_bits,
)
from genetic_coil_arena.envs.plant.replicates import Protocol, Replicate
from genetic_coil_arena.envs.plant.rieth import RiethData, synthetic_rieth
from genetic_coil_arena.envs.plant.tep import NODES
from genetic_coil_arena.experiments import plant_learned_codes as plc

FAULTS = (1, 2, 3)
SMALL = dict(  # noqa: C408
    faults=FAULTS,
    n_replicates=2,
    train_runs=4,
    val_runs=2,
    test_runs=2,
    train_normal_runs=4,
    val_normal_runs=2,
    test_normal_runs=2,
    n_available=12,
)
torch_missing = importlib.util.find_spec("torch") is None
needs_torch = pytest.mark.skipif(torch_missing, reason="torch is not installed")
FAST = plc.Scale(
    "smoke",
    {"classifier": "logreg", "l2": 1e-2, "max_iter": 30, "context": 2},
    {
        "e2e": (("bottleneck", ("ste", "gumbel")),),
        "vq": (("whiten", (False, True)),),
        "local": (("hidden", (8, 16)),),
    },
    {
        "e2e": {"epochs": 2, "hidden": 8, "head": 8},
        "vq": {"dim": 4},
        "local": {"hidden": 8, "max_iter": 20, "folds": 2},
    },
)


@pytest.fixture(scope="module")
def rep():
    data = synthetic_rieth(n_runs=12, n_faults=3, seed=0, strength=5.0)
    return Replicate(data, Protocol(**SMALL), 0)


class TestBits:
    def test_pack_unpack_round_trip_and_range(self):
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, size=(5, 4, 16)).astype(bool)
        msg = pack_bits(bits)
        assert msg.shape == (5, 4, 2) and msg.min() >= 0 and msg.max() < 256
        back = unpack_bits(msg) > 0
        assert (back == bits).all()

    def test_pack_needs_whole_bytes(self):
        with pytest.raises(ValueError, match="multiple of 8"):
            pack_bits(np.ones((2, 5), bool))

    def test_kmeans_handles_fewer_points_than_centres(self):
        x = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
        assert kmeans(x, 32, np.random.default_rng(1)).shape == (32, 3)


class TestFeatures:
    def test_padding_is_zero_and_standardised(self, rep):
        f = Featuriser(rep.layout).fit(rep.train["windows"])
        z = f(rep.train["windows"])
        assert z.shape[:2] == (len(rep.train["windows"]), len(NODES))
        assert (
            z.shape[2] == node_features(rep.train["windows"][:2], rep.layout).shape[2]
        )
        assert (z[:, ~f.mask] == 0).all()
        assert np.abs(z).max() <= 8.0


def _check_encoder(rep, method, budget, scale=FAST, seed=0):
    prep = plc.build_cell(rep, method, budget, {}, seed, scale)
    enc = prep.encoder
    msgs = enc.encode_windows(rep.val["windows"])
    assert msgs.shape == (len(rep.val["windows"]), len(NODES), budget)
    assert msgs.dtype == np.int64 and msgs.min() >= 0 and msgs.max() < 256
    assert enc.bytes_used == (budget,) * len(NODES)
    assert enc.decode(msgs).shape == (len(msgs), enc.n_features)
    return prep, msgs


class TestEncoders:
    @pytest.mark.parametrize("method", ["vq", "local"])
    @pytest.mark.parametrize("budget", [1, 2])
    def test_budget_and_determinism_numpy_methods(self, rep, method, budget):
        _, m1 = _check_encoder(rep, method, budget)
        _, m2 = _check_encoder(rep, method, budget)
        assert (m1 == m2).all()
        _, m3 = _check_encoder(rep, method, budget, seed=7)
        assert m3.shape == m1.shape

    @needs_torch
    @pytest.mark.parametrize("budget", [1, 2])
    @pytest.mark.parametrize("bottleneck", ["ste", "gumbel"])
    def test_end_to_end_budget_and_determinism(self, rep, budget, bottleneck):
        hp = {"bottleneck": bottleneck}
        a = plc.build_cell(rep, "e2e", budget, hp, 3, FAST).encoder
        b = plc.build_cell(rep, "e2e", budget, hp, 3, FAST).encoder
        x = rep.val["windows"]
        ma, mb = a.encode_windows(x), b.encode_windows(x)
        assert ma.shape[-1] == budget and ma.max() < 256 and ma.min() >= 0
        assert (ma == mb).all()
        assert a.decode(ma).shape == (len(x), len(NODES) * 8 * budget)

    def test_local_code_is_class_and_level(self, rep):
        _, msgs = _check_encoder(rep, "local", 1)
        assert (msgs >> 3).max() < rep.protocol.n_classes
        assert (msgs & 7).max() <= 7

    def test_unknown_method_and_zero_budget_are_refused(self, rep):
        with pytest.raises(ValueError, match="Unknown method"):
            plc.build_cell(rep, "nope", 1, {}, 0, FAST)
        with pytest.raises(ValueError, match="at least 1 byte"):
            plc.build_cell(rep, "vq", 0, {}, 0, FAST)


class TestHooks:
    def test_zero_node_and_shuffle(self):
        m = np.arange(24).reshape(6, 2, 2) + 1
        out = plc.zero_node(1)(m, 0)
        assert (out[:, 1] == 0).all() and (out[:, 0] == m[:, 0]).all()
        shuf = plc.shuffle_runs(0)
        s = shuf(m, 0)
        assert sorted(s[:, 0, 0].tolist()) == sorted(m[:, 0, 0].tolist())
        assert not (s == m).all(axis=(1, 2)).any()  # nobody keeps its own messages
        assert (shuf(m, 5) == s).all()  # same assignment at every step


class TestCell:
    @pytest.mark.parametrize("method", ["vq", "local"])
    def test_rows_with_ablations_and_budget_check(self, rep, method):
        rows = plc.run_cell(rep, method, 2, {}, 0, FAST, ablate=True)
        assert [r["variant"] for r in rows] == list(plc.VARIANTS)
        for r in rows:
            assert r["msg_bytes"] <= r["budget"] == 2 and r["max_symbol"] < 256
            assert r["method"] == method and r["replicate"] == 0
        json.dumps(rows)  # serialisable
        again = plc.run_cell(rep, method, 2, {}, 0, FAST, ablate=False)
        assert len(again) == 1
        assert (
            again[0]["macro_f1"] == rows[0]["macro_f1"]
        )  # deterministic given the seed

    def test_tune_picks_a_grid_value(self, rep):
        hp, trace = plc.tune_hp(rep, "vq", 1, FAST)
        assert hp["whiten"] in (False, True) and len(trace) == 2


def _row(method, budget, replicate, f1, variant="base", seed_index=0):
    return {
        "method": method,
        "budget": budget,
        "replicate": replicate,
        "seed_index": seed_index,
        "variant": variant,
        "macro_f1": f1,
        "macro_f1_matched": f1,
        "false_alarm_rate": 0.05,
        "detection_delay": 30.0,
        "msg_bytes": budget,
        "max_symbol": 200,
    }


def _baseline(score, enc="features", budget=4):
    return {
        "rows": [
            {
                "fault_set": "diagnosable",
                "variant": "tuned",
                "encoding": enc,
                "budget": budget,
                "replicate": r,
                "macro_f1": score + 0.001 * r,
                "macro_f1_matched": score + 0.001 * r,
                "false_alarm_rate": 0.05,
                "detection_delay": 36.0,
            }
            for r in range(10)
        ]
    }


def _partial(method, budget, gain, replicates=10, scale="full"):
    rng = np.random.default_rng(budget)
    rows = [
        _row(method, budget, r, 0.9 + 0.001 * r + gain + 0.002 * rng.normal())
        for r in range(replicates)
    ]
    return {
        "kind": "plant_learned_codes_partial",
        "method": method,
        "budget": budget,
        "scale": scale,
        "protocol": {"n_replicates": 10},
        "commit": "abc123def",
        "seconds": 60.0,
        "tuning": {
            "method": method,
            "budget": budget,
            "hp": {"x": 1},
            "trace": [{"val_f1": 0.8}],
        },
        "rows": rows,
    }


class TestMerge:
    def test_merges_dedupes_and_checks_protocol(self):
        a, b = _partial("vq", 4, 0.0), _partial("local", 4, 0.0)
        later = _partial("vq", 4, 0.1, replicates=3)
        merged = plc.merge_partials([a, b, later])
        keys = [(r["method"], r["replicate"]) for r in merged["rows"]]
        assert len(keys) == len(set(keys)) == 20
        vq0 = next(
            r for r in merged["rows"] if r["method"] == "vq" and r["replicate"] == 0
        )
        assert vq0["macro_f1"] > 0.95  # the later partial replaced the earlier one
        assert set(merged["tuning"]) == {"vq|4", "local|4"}
        assert merged["partials"] == 3
        other = {**b, "protocol": {"n_replicates": 5}}
        with pytest.raises(ValueError, match="different protocols"):
            plc.merge_partials([a, other])
        with pytest.raises(ValueError, match="Nothing"):
            plc.merge_partials([])

    def test_read_partials_from_folder_ignores_other_json(self, tmp_path):
        (tmp_path / "a.json").write_text(json.dumps(_partial("vq", 1, 0.0)))
        (tmp_path / "b.json").write_text(json.dumps({"kind": "other"}))
        (tmp_path / "c.json").write_text("{not json")
        got = plc.read_partials([tmp_path])
        assert len(got) == 1 and got[0]["method"] == "vq"


class TestVerdict:
    def _tests(self, gains, hand=0.90):
        rows = []
        for (method, budget), gain in gains.items():
            rows += _partial(method, budget, gain)["rows"]
        return plc.g2_tests(rows, plc.normalise_baseline(_baseline(hand)), budgets=(4,))

    def test_pass_needs_a_ci_above_zero(self):
        tests = self._tests({("vq", 4): 0.05, ("local", 4): 0.0, ("e2e", 4): -0.02})
        by = {t["method"]: t for t in tests}
        assert by["vq"]["win"] and by["vq"]["diff"] > 0.04
        assert not by["local"]["win"] and not by["e2e"]["win"]
        verdict, why = plc.g2_verdict(tests, budgets=(4,))
        assert verdict == "PASS" and "vq at 4 B" in why

    def test_kill_when_all_cells_tested_and_none_win(self):
        tests = self._tests({("vq", 4): -0.01, ("local", 4): 0.0, ("e2e", 4): -0.05})
        assert plc.g2_verdict(tests, budgets=(4,))[0] == "KILL"

    def test_incomplete_when_a_cell_is_missing_or_short(self):
        tests = self._tests({("vq", 4): -0.01, ("local", 4): 0.0})
        assert plc.g2_verdict(tests, budgets=(4,))[0] == "INCOMPLETE"
        short = [{**t, "n": 3} for t in self._tests({("vq", 4): 0.05})]
        assert plc.g2_verdict(short, budgets=(4,))[0] != "PASS"

    def test_picks_the_best_hand_design_per_budget(self):
        base = plc.normalise_baseline(
            {
                "rows": _baseline(0.8, "summary")["rows"]
                + _baseline(0.9, "features")["rows"]
                + _baseline(0.7, "alarms", 1)["rows"]
            }
        )
        enc, values = plc.best_hand(base, 4, "macro_f1")
        assert enc == "features" and len(values) == 10
        assert (
            plc.best_hand(base, 2, "macro_f1")[0] == "alarms"
        )  # the 1 B code counts anywhere

    def test_holm_adjusts_upwards(self):
        tests = self._tests({("vq", 4): 0.01, ("local", 4): 0.012, ("e2e", 4): 0.0})
        assert all(t["p_holm"] >= t["p"] for t in tests)


class TestReport:
    def test_report_renders_with_verdict_and_smoke_banner(self):
        parts = [_partial(m, b, 0.0, scale="smoke") for m in METHODS for b in (1, 2, 4)]
        merged = plc.merge_partials(parts)
        base = {
            "rows": sum(
                (
                    _baseline(0.9, e, b)["rows"]
                    for e, b in (("features", 1), ("features", 2), ("features", 4))
                ),
                [],
            )
        }
        text = plc.build_report(merged, base, "baseline.json")
        assert "SMOKE RUN" in text and "## Verdict (goal G2)" in text
        assert "INCOMPLETE" in text or "KILL" in text or "PASS" in text
        assert "rows with a message longer than their budget: 0" in text


class TestWorkers:
    def test_thread_cap(self):
        assert plc.torch_threads_for(8, 1) == (8, 1)
        assert plc.torch_threads_for(8, 4) == (2, 4)
        procs, threads = plc.torch_threads_for(3, 8)
        assert procs * threads <= 3

    def test_env_overrides_workers(self, monkeypatch):
        monkeypatch.setenv("GCA_WORKERS", "12")
        assert plc.env_workers(8) == 12
        monkeypatch.setenv("GCA_WORKERS", "junk")
        assert plc.env_workers(8) == 8
        monkeypatch.delenv("GCA_WORKERS")
        assert plc.env_workers(5) == 5


class TestRealData:
    def test_real_cache_budget_check(self):
        try:
            data = RiethData.from_cache()
        except Exception:  # noqa: BLE001
            pytest.skip("Rieth cache not available")
        small = Protocol(
            faults=(1, 4),
            n_replicates=10,
            train_runs=3,
            val_runs=2,
            test_runs=2,
            train_normal_runs=3,
            val_normal_runs=2,
            test_normal_runs=2,
        )
        rep = Replicate(data, small, 0)
        prep = plc.build_cell(rep, "vq", 2, {}, 0, FAST)
        msgs = prep.encoder.encode_windows(rep.val["windows"])
        assert msgs.shape[-1] == 2 and msgs.max() < 256
