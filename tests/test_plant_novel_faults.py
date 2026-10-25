"""Tests for the novel-fault experiment (R8 step 4): splits, freezing, metrics, the verdict.

Synthetic and fast. Torch-dependent tests skip when torch is absent; the real-data check
skips without the Rieth cache.
"""

import importlib.util
import json

import numpy as np
import pytest

from genetic_coil_arena.envs.plant.replicates import Protocol, Replicate
from genetic_coil_arena.envs.plant.rieth import RiethData, synthetic_rieth
from genetic_coil_arena.experiments import plant_learned_codes as plc
from genetic_coil_arena.experiments import plant_novel_faults as pnf

FAULTS = (1, 2, 3, 4, 5, 6)
UNSEEN = (2, 5)
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
needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is not installed"
)
FAST = plc.Scale(
    "smoke",
    {"classifier": "logreg", "l2": 1e-2, "max_iter": 30, "context": 2},
    {
        "e2e": (("bottleneck", ("ste",)),),
        "e2e_aux": (("bottleneck", ("ste",)),),
        "vq": (("whiten", (False, True)),),
        "local": (("hidden", (8, 16)),),
        "oracle": (("hidden", (8, 16)),),
    },
    {
        "e2e": {"epochs": 2, "hidden": 8, "head": 8},
        "e2e_aux": {"epochs": 2, "hidden": 8, "head": 8},
        "vq": {"dim": 4},
        "local": {"hidden": 8, "max_iter": 20, "folds": 2},
    },
)


@pytest.fixture(scope="module")
def rep():
    data = synthetic_rieth(n_runs=12, n_faults=6, seed=0, strength=5.0)
    return Replicate(data, Protocol(**SMALL), 0)


@pytest.fixture(scope="module")
def setting():
    return pnf.Setting(FAULTS, UNSEEN)


class TestSplits:
    def test_distinct_sorted_deterministic_and_sized(self):
        a = pnf.make_splits(n_splits=6)
        assert a == pnf.make_splits(n_splits=6)
        assert len(set(a)) == 6
        for s in a:
            assert len(s) == pnf.N_UNSEEN and list(s) == sorted(s)
            assert set(s) <= set(pnf.DIAGNOSABLE_FAULTS)
        assert a != pnf.make_splits(n_splits=6, seed=1)

    def test_impossible_requests_are_refused(self):
        with pytest.raises(ValueError):
            pnf.make_splits((1, 2, 3), n_splits=5, n_unseen=1)
        with pytest.raises(ValueError):
            pnf.make_splits((1, 2, 3), n_splits=1, n_unseen=3)

    def test_setting_class_maps(self, setting):
        assert setting.seen == (1, 3, 4, 6)
        assert list(setting.seen_classes) == [1, 3, 4, 6]
        assert list(setting.unseen_classes) == [2, 5]
        assert setting.n_seen_classes == 5
        assert list(setting.compact) == [0, 1, -1, 2, 3, -1, 4]
        with pytest.raises(ValueError):
            pnf.Setting(FAULTS, ())
        with pytest.raises(ValueError):
            pnf.Setting((1, 2), (1, 2))

    def test_seen_view_drops_whole_unseen_runs(self, rep, setting):
        keep, view = pnf.seen_view(rep.train, setting)
        rc = pnf.run_classes(rep.train)
        assert not np.isin(rc[view["run"]], setting.unseen_classes).any()
        assert view["labels"].min() >= 0 and view["labels"].max() == 4
        assert len(view["windows"]) == keep.sum() < len(keep)
        # a seen run keeps all its windows and the order
        r = np.flatnonzero(rc == 1)[0]
        assert (rep.train["run"] == r).sum() == (view["run"] == r).sum()


class TestMetrics:
    def test_nmi_extremes(self):
        y = np.repeat(np.arange(4), 25)
        assert pnf.nmi(y, y) == pytest.approx(1.0)
        assert pnf.nmi(y, (y + 1) % 4) == pytest.approx(1.0)  # relabelling
        assert pnf.nmi(y, np.zeros_like(y)) == 0.0
        rng = np.random.default_rng(0)
        assert pnf.nmi(y, rng.integers(0, 4, size=y.size)) < 0.1

    def test_separability_blobs_and_noise(self):
        rng = np.random.default_rng(0)
        classes = np.array([2, 5, 7])
        lab = np.repeat(classes, 120)
        run = np.repeat(np.arange(12), 30)
        x = rng.normal(size=(360, 6)) + 6.0 * np.eye(3)[
            np.searchsorted(classes, lab)
        ].repeat(2, axis=1)
        good = pnf.separability(x, lab, run, classes)
        assert good["probe_acc"] > 0.95 and good["nmi"] > 0.9
        assert good["chance"] == pytest.approx(1 / 3)
        bad = pnf.separability(rng.normal(size=(360, 6)), lab, run, classes)
        assert bad["probe_acc"] < 0.6 and bad["nmi"] < 0.2

    def test_run_detection(self):
        # two runs of 5 windows; labelled from window 2; run 0 alarms at window 3, run 1 never
        lab = np.tile([0, 0, 1, 1, 1], 2)
        run = np.repeat([0, 1], 5)
        n_seen = np.tile(np.arange(20, 120, 20), 2)
        alarm = np.zeros(10, bool)
        alarm[[3, 4]] = True
        out = pnf.run_detection(
            alarm, lab, run, n_seen, np.array([True, True]), onset=30
        )
        assert out["detect_rate"] == 0.5
        assert out["delay"] == pytest.approx(n_seen[3] - 30)
        assert out["alarm_rate"] == pytest.approx(2 / 6)

    def test_nearest_known_maps_to_the_modal_seen_class(self, setting):
        lab = np.array([2] * 10 + [5] * 10)
        alarm = np.ones(20, bool)
        fault_c = np.array([2] * 8 + [1] * 2 + [4] * 10)  # compact classes
        out = pnf.nearest_known(fault_c, alarm, lab, setting)
        assert out["2"][0] == 3.0 and out["2"][1] == pytest.approx(
            0.8
        )  # compact 2 = fault 3
        assert out["5"][0] == 6.0 and out["5"][1] == 1.0
        out = pnf.nearest_known(fault_c, np.zeros(20, bool), lab, setting)
        assert out["2"] == [-1.0, pytest.approx(np.nan, nan_ok=True), 0.0]

    def test_fingerprint_sees_array_changes(self):
        a = {"w": np.arange(5.0), "m": [np.ones(3)]}
        fp = pnf.fingerprint(a)
        assert fp == pnf.fingerprint({"w": np.arange(5.0), "m": [np.ones(3)]})
        a["m"][0][1] = 2.0
        assert pnf.fingerprint(a) != fp


class TestExtension:
    def test_extension_runs_are_nested_and_unseen_only(self, rep, setting):
        rc = pnf.run_classes(rep.train)
        r1 = pnf.extension_runs(rep.train, setting, 1, 0)
        r3 = pnf.extension_runs(rep.train, setting, 3, 0)
        assert len(r1) == 2 and len(r3) == 6
        assert set(r1) <= set(r3)
        assert set(rc[r3]) == set(setting.unseen_classes)
        assert (pnf.extension_runs(rep.train, setting, 3, 0) == r3).all()
        assert set(pnf.extension_runs(rep.train, setting, 3, 1)) != set(r3)

    @pytest.mark.parametrize("code", ["local", "vq", "features", "oracle"])
    def test_cell_runs_keeps_the_code_frozen_and_reports_everything(
        self, rep, setting, code
    ):
        budget = 0 if code == "oracle" else 2
        row = pnf.run_cell(rep, setting, code, budget, {}, 0, 0, FAST, shots=(1, 3))
        assert row["frozen_ok"] and row["unseen"] == list(UNSEEN)
        for k in ("0", "1", "3"):
            for name in ("f1_all", "f1_unseen", "f1_seen"):
                assert 0.0 <= row[f"{name}_k{k}"] <= 1.0
        assert (
            row["f1_unseen_k0"] == 0.0
        )  # a fixed vocabulary cannot name an unseen fault
        assert 0.0 <= row["probe_acc"] <= 1.0 and 0.0 <= row["nmi"] <= 1.0
        assert set(row["mapping"]) == {str(f) for f in UNSEEN}
        json.dumps(row)  # plain data

    @needs_torch
    @pytest.mark.parametrize("code", ["e2e", "e2e_aux"])
    def test_extending_the_centre_leaves_encoder_weights_unchanged(
        self, rep, setting, code
    ):
        s1 = pnf.fit_stage1(rep, setting, code, 1, {}, 0, FAST)
        enc = s1.coded.frozen
        weights = {k: v.copy() for k, v in enc.weights.items()}
        x_all = s1.coded.decode(rep.train["windows"])
        x_all[s1.keep_train] = s1.x_train
        for k in (1, 3):
            centre = pnf.extend_centre(rep, setting, s1, x_all, k, 0, 0)
            assert centre.n_classes == setting.n_classes
        for name, w in weights.items():
            assert (enc.weights[name] == w).all()
        row = pnf.run_cell(rep, setting, code, 1, {}, 0, 0, FAST, shots=(3,))
        assert row["frozen_ok"]

    @needs_torch
    def test_reconstruction_term_changes_the_code(self, rep, setting):
        _, train = pnf.seen_view(rep.train, setting)
        from genetic_coil_arena.envs.plant.learned_codes import (
            Featuriser,
            train_end_to_end,
        )

        feats = Featuriser(rep.layout).fit(train["windows"])(train["windows"])
        kw = dict(  # noqa: C408
            n_classes=setting.n_seen_classes,
            budget=1,
            seed=0,
            hidden=8,
            head=8,
            epochs=3,
        )
        plain, _ = train_end_to_end(feats, train["labels"], **kw)
        aux, _ = train_end_to_end(feats, train["labels"], recon=1.0, **kw)
        again, _ = train_end_to_end(feats, train["labels"], recon=1.0, **kw)
        assert not np.allclose(plain["w2"], aux["w2"])
        assert np.allclose(aux["w2"], again["w2"])

    def test_tuning_picks_a_grid_value(self, rep, setting):
        hp, trace = pnf.tune_hp(rep, setting, "local", 1, FAST)
        assert hp["hidden"] in (8, 16) and trace
        assert pnf.tune_hp(rep, setting, "features", 1, FAST) == ({}, [])
        hp, _ = pnf.tune_hp(rep, setting, "oracle", 0, FAST)
        assert hp["hidden"] in (8, 16)


# --------------------------------------------------------------------- verdict and report
def _row(code, budget, replicate, unseen_k3, seen=0.9):
    row = {
        "code": code,
        "budget": budget,
        "replicate": replicate,
        "split": replicate % 5,
        "unseen": [1, 2],
        "seen_f1": seen,
        "far": 0.05,
        "unseen_alarm_rate": 0.8,
        "unseen_detect_rate": 1.0,
        "unseen_delay": 30.0,
        "seen_alarm_rate": 0.9,
        "seen_delay": 25.0,
        "map_share": 0.7,
        "mapping": {"1": [3.0, 0.7, 0.8], "2": [4.0, 0.6, 0.8]},
        "probe_acc": 0.5,
        "nmi": 0.3,
        "chance": 0.2,
        "frozen_ok": True,
        "hp": {},
        "seconds": 1.0,
    }
    for k in (0, 1, 3, 10):
        row[f"f1_unseen_k{k}"] = 0.0 if k == 0 else unseen_k3
        row[f"f1_all_k{k}"] = 0.5
        row[f"f1_seen_k{k}"] = 0.9
    return row


def _rows(gain, n=10, codes=("e2e", "e2e_aux", "vq"), budgets=(1, 2, 4, 8)):
    rng = np.random.default_rng(0)
    rows = []
    for r in range(n):
        noise = 0.01 * rng.normal()
        for b in budgets:
            rows.append(_row("local", b, r, 0.6 + noise))
            for c in codes:
                g = gain.get(c, 0.0) if isinstance(gain, dict) else gain
                rows.append(_row(c, b, r, 0.6 + noise + g + 0.002 * rng.normal()))
        rows.append(_row("oracle", 0, r, 0.9, seen=0.97))
    return rows


class TestVerdict:
    def test_pass_needs_ci_above_zero_for_e2e_or_aux(self):
        v, why = pnf.g2b_verdict(pnf.g2b_tests(_rows({"e2e_aux": 0.05})))
        assert v == "PASS" and "e2e_aux" in why
        v, _ = pnf.g2b_verdict(pnf.g2b_tests(_rows({"vq": 0.05})))
        assert v == "INCONCLUSIVE"  # vq ahead, but vq cannot pass G2b

    def test_kill_when_designed_is_never_behind(self):
        v, _ = pnf.g2b_verdict(pnf.g2b_tests(_rows(-0.05)))
        assert v == "KILL"
        v, _ = pnf.g2b_verdict(pnf.g2b_tests(_rows(0.0)))
        assert v in ("KILL", "INCONCLUSIVE")

    def test_tiny_noisy_lead_is_inconclusive_and_missing_cells_incomplete(self):
        rng = np.random.default_rng(1)
        rows = []
        for r in range(10):
            for b in (1, 2, 4, 8):
                rows.append(_row("local", b, r, 0.6))
                for c in ("e2e", "e2e_aux", "vq"):
                    rows.append(_row(c, b, r, 0.6 + 0.01 + 0.2 * rng.normal()))
        assert pnf.g2b_verdict(pnf.g2b_tests(rows))[0] in ("INCONCLUSIVE", "PASS")
        v, _ = pnf.g2b_verdict(pnf.g2b_tests(_rows(0.05, n=4)))
        assert v == "INCOMPLETE"
        v, _ = pnf.g2b_verdict(pnf.g2b_tests(_rows(0.05, budgets=(1, 2, 4))))
        assert v == "PASS"  # a pass does not need the 8 B cells

    def test_oracle_check_flags_a_coded_cell_that_beats_it(self):
        rows = _rows(0.0)
        out = pnf.oracle_check(rows)
        assert out and all(o["dominates"] for o in out)
        rows += [
            dict(_row("e2e", 1, r, 0.6, seen=0.99), replicate=r) for r in range(10)
        ]
        rows = [
            r
            for r in rows
            if not (r["code"] == "e2e" and r["budget"] == 1 and r["seen_f1"] < 0.99)
        ]
        beaten = [o for o in pnf.oracle_check(rows) if o["beaten"]]
        assert [(o["code"], o["budget"]) for o in beaten] == [("e2e", 1)]


class TestMergeAndReport:
    def _partial(self, rows, code="e2e", budget=1, scale="full"):
        return {
            "kind": pnf.PARTIAL_KIND,
            "code": code,
            "budget": budget,
            "scale": scale,
            "protocol": default_protocol(),
            "splits": [[1, 2], [3, 4]],
            "commit": "abc1234567",
            "seconds": 60.0,
            "tuning": {"code": code, "budget": budget, "hp": {"x": 1}},
            "rows": rows,
        }

    def test_merge_dedupes_and_checks_consistency(self):
        a = self._partial([_row("e2e", 1, 0, 0.5), _row("e2e", 1, 1, 0.5)])
        b = self._partial([_row("e2e", 1, 1, 0.9)])
        merged = pnf.merge_partials([a, b])
        assert len(merged["rows"]) == 2 and merged["cpu_seconds"] == 120.0
        assert [r["f1_unseen_k3"] for r in merged["rows"]] == [0.5, 0.9]
        bad = self._partial([], scale="smoke")
        with pytest.raises(ValueError, match="scale"):
            pnf.merge_partials([a, bad])
        with pytest.raises(ValueError):
            pnf.merge_partials([])

    def test_read_partials_ignores_other_json(self, tmp_path):
        (tmp_path / "ok.json").write_text(json.dumps(self._partial([])))
        (tmp_path / "other.json").write_text(json.dumps({"kind": "x"}))
        (tmp_path / "broken.json").write_text("{")
        assert len(pnf.read_partials([tmp_path])) == 1

    def test_report_has_verdict_splits_and_smoke_banner(self):
        p = self._partial(_rows({"e2e": 0.05}))
        text = pnf.build_report(pnf.merge_partials([p]))
        assert "**PASS.**" in text and "split 0: 1, 2" in text
        assert "(tuning)" in text and "SMOKE" not in text
        smoke = self._partial(_rows(0.0, n=2), scale="smoke")
        assert "SMOKE RUN" in pnf.build_report(pnf.merge_partials([smoke]))


def default_protocol():
    return plc.default_protocol(10)


def test_real_cache_split_and_cell():
    try:
        data = RiethData.from_cache()
    except Exception:  # noqa: BLE001
        pytest.skip("Rieth cache not available")
    proto = Protocol(
        n_replicates=1,
        train_runs=4,
        val_runs=3,
        test_runs=3,
        train_normal_runs=6,
        val_normal_runs=3,
        test_normal_runs=3,
    )
    rep = Replicate(data, proto, 0)
    setting = pnf.Setting(proto.faults, pnf.make_splits(n_splits=1)[0])
    row = pnf.run_cell(
        rep, setting, "local", 1, {}, 0, 0, plc.SCALES["smoke"], shots=(1,)
    )
    assert row["frozen_ok"] and np.isfinite(row["seen_f1"])
