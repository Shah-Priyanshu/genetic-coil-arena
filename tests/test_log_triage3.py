"""R10 step 3: trained fault specialists, typed reports, fusion, the train/test split, the teams,
the report and the aggregation job. Synthetic data only (no GPU); the real-data checks skip
without the RCAEval cache."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from genetic_coil_arena.envs.logtriage import fault_features as ff
from genetic_coil_arena.envs.logtriage import team as tm
from genetic_coil_arena.envs.logtriage import trained as tr
from genetic_coil_arena.envs.logtriage.rcaeval import FAULT_TYPES
from genetic_coil_arena.envs.logtriage.synthetic import synthetic_incident
from genetic_coil_arena.envs.logtriage.views import ApproxCounter
from genetic_coil_arena.experiments import log_triage as lt
from genetic_coil_arena.experiments import log_triage2 as t2
from genetic_coil_arena.experiments import log_triage3 as t3
from genetic_coil_arena.experiments import log_triage3_report as rp3
from genetic_coil_arena.learning.classifiers import MLPClassifier
from genetic_coil_arena.protocol.goalspec import SpecValidationError

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "r10_step3.py"


def labelled(n, system, rep, base):
    out = []
    for i in range(n):
        inc = synthetic_incident(base + i, rep=rep)
        inc.system = system
        inc.id = f"{system}/syn{base + i}/{rep}"
        out.append(inc)
    return out


class TestClassifier:
    def make(self, n=240, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 3, n)
        x = rng.normal(size=(n, 5)) + 3.0 * np.eye(3, 5)[y]
        return x, y

    @pytest.mark.parametrize("hidden", [0, 8])
    def test_learns_and_round_trips(self, hidden):
        x, y = self.make()
        clf = MLPClassifier(3, hidden=hidden, n_models=2, epochs=200).fit(x, y)
        assert (clf.predict(x) == y).mean() > 0.9
        proba = clf.predict_proba(x[:7])
        assert proba.shape == (7, 3) and np.allclose(proba.sum(axis=1), 1)
        again = MLPClassifier.from_dict(json.loads(json.dumps(clf.to_dict())))
        assert np.allclose(again.predict_proba(x[:7]), proba)

    def test_balanced_weights_help_a_rare_class(self):
        rng = np.random.default_rng(1)
        x = np.concatenate([rng.normal(0, 1, (200, 2)), rng.normal(1.5, 1, (10, 2))])
        y = np.array([0] * 200 + [1] * 10)
        plain = MLPClassifier(2, balanced=False, epochs=200).fit(x, y)
        bal = MLPClassifier(2, balanced=True, epochs=200).fit(x, y)
        assert (bal.predict(x) == 1).sum() > (plain.predict(x) == 1).sum()

    def test_bad_input(self):
        with pytest.raises(ValueError):
            MLPClassifier(3).fit(np.zeros((4, 2)), np.array([0, 1, 2, 5]))
        with pytest.raises(RuntimeError):
            MLPClassifier(3).predict_proba(np.zeros((1, 2)))


class TestFeatures:
    def test_names_match_size_and_are_by_kind(self):
        assert len(ff.feature_names()) == ff.N_FEATURES
        vec = ff.row_features(
            {"cpu": [30.0, 1, 5.0, 2.0], "weird": [99.0, 1, 1.0, 1.0]}
        )
        assert vec.shape == (ff.N_FEATURES,)
        names = ff.feature_names()
        assert vec[names.index("cpu:present")] == 1
        assert vec[names.index("cpu:is_top")] == 1
        assert vec[names.index("mem:present")] == 0
        # a kind the model does not know is ignored, not an error
        assert np.allclose(vec, ff.row_features({"cpu": [30.0, 1, 5.0, 2.0]}))
        assert not ff.row_features({}).any()

    def test_direction_and_onset(self):
        up = ff.row_features({"mem": [10.0, 1, 0.0, 1.0]})
        down = ff.row_features({"mem": [10.0, -1, None, -0.5]})
        names = ff.feature_names()
        assert (
            up[names.index("mem:signed_logz")]
            > 0
            > down[names.index("mem:signed_logz")]
        )
        assert (
            up[names.index("mem:onset")] == 1.0
            and down[names.index("mem:onset")] == 0.0
        )

    def test_reads_only_its_service(self):
        inc = synthetic_incident(3)
        s = inc.root_service
        before = ff.service_features(inc, s)
        inc.metrics = {s: inc.metrics[s]}
        assert np.array_equal(ff.service_features(inc, s), before)
        assert not ff.service_features(inc, "nobody").any()


class TestTypedReports:
    def test_reports_validate(self):
        inc = synthetic_incident(1)
        base = tm.metric_message(inc, 0)
        msg = tr.trained_metric_message(base, [0.1, 0.5, 0.1, 0.1, 0.1, 0.1])
        assert tr.trained_metric_errors(msg) == [] and msg["fault"] == "mem"
        assert tr.trained_metric_errors(tr.trained_metric_message(base, None)) == []
        assert tr.trained_metric_errors(dict(msg, fault="banana"))
        assert tr.trained_metric_errors(dict(msg, fault_probs=[0.5]))
        log = tr.trained_log_message("svc", [0, 0, 0, 0.2, 0.3, 0.5])
        assert tr.trained_log_errors(log) == [] and log["fault"] == "socket"
        assert abs(log["confidence"] - 0.5) < 1e-6
        assert tr.trained_log_errors(tr.empty_trained_log_message("x")) == []
        # junk probabilities become an empty-ish uniform report, never an invalid one
        assert tr.trained_log_errors(tr.trained_log_message("x", [np.nan] * 6)) == []

    def test_goals_are_validated(self):
        reg = tr.trained_registry()
        reg.make(tr.SPEC_METRIC_FAULT, {"incident": "i", "service": "a"})
        with pytest.raises(SpecValidationError):
            reg.make(tr.SPEC_LOG_FAULT, {"incident": "i", "service": "a"})
        goal = reg.make(tr.SPEC_METRIC_FAULT, {"incident": "i", "service": "a"})
        assert reg.result_errors(goal, {"service": "a"})

    def test_fusion_rules(self):
        pm = np.array([0.7, 0.1, 0.05, 0.05, 0.05, 0.05])
        pl = np.array([0.05, 0.05, 0.05, 0.05, 0.1, 0.7])
        assert tr.fuse("metric", pm, pl).argmax() == 0
        assert tr.fuse("log", pm, pl).argmax() == 5
        assert tr.fuse("mean_w75", pm, pl).argmax() == 5
        assert tr.fuse("geo_w25", pm, pl).argmax() == 0
        assert np.isclose(tr.fuse("mean_w50", pm, pl).sum(), 1)
        assert tr.fuse("max_conf", pm, pl * 0 + [0, 0, 0, 0, 0, 1]).argmax() == 5
        assert tr.fuse("log", pm, None).argmax() == 0  # missing report: the other one
        assert tr.fuse("log", None, None) is None
        with pytest.raises(ValueError):
            tr.fuse("nope", pm, pl)
        with pytest.raises(ValueError):
            tr.fuse("stack", pm, pl)

    def test_stack_learns_to_trust_the_better_report(self):
        rng = np.random.default_rng(0)
        n = 300
        y = rng.integers(0, 6, n)
        good = np.eye(6)[y] * 0.8 + 0.04
        bad = rng.dirichlet(np.ones(6), n)
        stack = tr.fit_stack(bad, good, y)
        pred = [
            int(np.argmax(tr.fuse("stack", a, b, stack))) for a, b in zip(bad, good)
        ]
        assert np.mean(np.array(pred) == y) > 0.95

    def test_fault_falls_back_to_the_metric_kind_heuristic(self):
        mm = tr.trained_metric_message(
            {
                "service": "a",
                "score": 9.0,
                "onset_s": 1.0,
                "top": [{"metric": "cpu", "dir": "up", "z": 9.0}],
            },
            None,
        )
        assert tr.fault_of("metric", mm, None) == "cpu"
        lm = tr.trained_log_message("a", [0, 0, 0.9, 0.1, 0, 0])
        assert tr.fault_of("log", mm, lm) == "disk"
        assert tr.fault_of("log", mm, tr.empty_trained_log_message("a")) == "cpu"

    def test_rule_decide_ranks_by_metric_score(self):
        mms = [tr.empty_trained_metric_message(n) for n in "abc"]
        mms[1] = dict(mms[1], score=5.0)
        lms = [tr.trained_log_message(n, [0, 0, 0, 1, 0, 0]) for n in "abc"]
        ranking, fault = tr.rule_decide_trained("abc", mms, lms, "log")
        assert ranking[0] == "b" and fault == "delay"


class TestSplit:
    def test_roles(self):
        assert t3.role_of("TT", 1) == t3.role_of("TT", 3) == "train"
        assert t3.role_of("SS", 1) == t3.role_of("OB", 1) == "train"
        assert t3.role_of("SS", 2) == t3.role_of("OB", 3) == "test"
        assert t3.role_of("synthetic", 2) == "other"

    def files(self, tmp_path):
        out = []
        for system in ("TT", "SS", "OB"):
            for rep in (1, 2, 3):
                d = tmp_path / "incidents" / system
                d.mkdir(parents=True, exist_ok=True)
                f = d / f"svc_cpu__{rep}.npz"
                f.write_bytes(b"")
                out.append(f)
        return out

    def test_partition_has_no_test_incident_in_training(self, tmp_path):
        parts = t3.paths_by_role(self.files(tmp_path))
        train = [t3.file_key(f) for f in parts["train"]]
        test = [t3.file_key(f) for f in parts["test"]]
        t3.assert_no_test_leak(train, test)
        assert {k[0] for k in train} & {k[0] for k in test} == set()
        assert sorted(k[0] for k in test) == [
            "OB/svc_cpu/2",
            "OB/svc_cpu/3",
            "SS/svc_cpu/2",
            "SS/svc_cpu/3",
        ]
        assert all(k[1] == "TT" or k[2] == 1 for k in train)
        assert len(train) == 5 and len(test) == 4

    def test_leak_is_detected(self, tmp_path):
        parts = t3.paths_by_role(self.files(tmp_path))
        train = [t3.file_key(f) for f in parts["train"]]
        test = [t3.file_key(f) for f in parts["test"]]
        with pytest.raises(ValueError, match="both train and test"):
            t3.assert_no_test_leak(train + test[:1], test)
        with pytest.raises(ValueError, match="not train"):
            t3.assert_no_test_leak(train + [("SS/x/2", "SS", 2)], test)
        with pytest.raises(ValueError, match="not test"):
            t3.assert_no_test_leak(train, test + [("SS/x/1", "SS", 1)])

    def test_real_cache_split_if_present(self):
        from genetic_coil_arena.envs.logtriage import rcaeval

        files = rcaeval.list_incidents(None, t3.TRAIN_SYSTEMS)
        if not files:
            pytest.skip("RCAEval cache not present")
        parts = t3.paths_by_role(files)
        train = {t3.file_key(f)[0] for f in parts["train"]}
        test = {t3.file_key(f)[0] for f in parts["test"]}
        assert not train & test
        assert all(i.split("/")[0] == "TT" or i.endswith("/1") for i in train)
        assert all(
            i.split("/")[0] in ("SS", "OB") and not i.endswith("/1") for i in test
        )


class TestMetricSpecialist:
    def dataset(self):
        incs = (
            labelled(40, "TT", 1, 0)
            + labelled(30, "SS", 1, 100)
            + labelled(30, "OB", 1, 200)
        )
        return incs, t3.metric_dataset(incs)

    def test_dataset_is_root_service_only(self):
        incs, data = self.dataset()
        assert data.x.shape == (100, ff.N_FEATURES) and set(data.groups) == {
            "TT",
            "SS",
            "OB",
        }
        assert data.y[0] == FAULT_TYPES.index(incs[0].fault_type)
        assert np.array_equal(
            data.x[0], ff.service_features(incs[0], incs[0].root_service)
        )

    def test_loso_selection_and_messages(self):
        incs, data = self.dataset()
        grid = ({"hidden": 0, "l2": 0.1, "epochs": 150}, {"hidden": 8, "epochs": 100})
        sel = t3.select_metric_model(data, grid)
        assert sel["chosen"] in list(grid)
        assert sel["oof"].shape == (100, 6)
        # a held-out system is never in the fit: out-of-fold rows differ from in-sample ones
        model = t3._make(sel["chosen"], 0).fit(data.x, data.y)
        spec = t3.MetricFaultSpecialist(model)
        inc = labelled(1, "SS", 2, 500)[0]
        mm = spec.messages(inc)
        assert len(mm) == inc.n_services
        assert all(tr.trained_metric_errors(m) == [] for m in mm)
        root = mm[inc.services.index(inc.root_service)]
        assert root["fault"] == inc.fault_type  # the synthetic signature is easy
        assert (
            root["score"]
            == tm.metric_message(inc, inc.services.index(inc.root_service))["score"]
        )


class TestFusionChoice:
    def test_prefers_the_better_specialist(self):
        rng = np.random.default_rng(0)
        n = 180
        y = rng.integers(0, 6, n)
        groups = np.array(["TT", "SS", "OB"] * (n // 3))
        good = np.eye(6)[y] * 0.7 + 0.05
        bad = rng.dirichlet(np.ones(6), n)
        c = t3.select_fusion(bad, good, y, groups)
        assert c.rule in {
            "log",
            "mean_w75",
            "geo_w75",
            "max_conf",
            "stack",
            "mean_w50",
            "geo_w50",
        }
        assert c.table["log"]["mean"] > c.table["metric"]["mean"]
        assert set(c.table["log"]) == {"TT", "SS", "OB", "mean"}
        c2 = t3.select_fusion(good, bad, y, groups)
        assert c2.table["metric"]["mean"] > c2.table["log"]["mean"]


class TestLogSpecialist:
    def test_training_examples_are_root_service_views(self):
        inc = synthetic_incident(2)
        cfg = t3.LogSettings(view_tokens=400)
        ex = t3.log_training_examples(inc, ApproxCounter(), cfg)
        assert len(ex) == len(cfg.jitters)
        assert all(y == FAULT_TYPES.index(inc.fault_type) for _, y in ex)
        for prompt, _ in ex:
            assert f"service '{inc.root_service}'" in prompt
            assert "Answer with the letter only" in prompt
            for i, f in enumerate(FAULT_TYPES):
                assert f"{lt.FAULT_LETTERS[i]}. {f}:" in prompt
        # only its own service's lines
        other = next(s for s in inc.services if s != inc.root_service)
        assert f"Logs of service {other}" not in ex[0][0]

    def test_letter_probs(self):
        p = t3.letter_probs([[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], [-9.0] * 6])
        assert np.allclose(p.sum(axis=1), 1) and p[0].argmax() == 0
        assert np.allclose(p[1], 1 / 6)

    def test_score_prompts_uses_closed_set_letters(self):
        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [ord(text[0])]

            def apply_chat_template(self, msgs, **kw):
                return msgs[-1]["content"]

        class Backend:
            tokenizer = Tok()

            def __init__(self):
                self.calls = []

            def next_token_logprobs(self, prompts, ids, adapter, chunk_size=None):
                self.calls.append((list(prompts), list(ids), adapter))
                return [
                    [-float(abs(len(p) % 6 - j)) for j in range(6)] for p in prompts
                ]

        be = Backend()
        out = t3.score_prompts(be, "ad", ["a" * 3, "bb" * 8, "c" * 5], batch=2)
        assert out.shape == (3, 6) and np.allclose(out.sum(axis=1), 1)
        assert {c[2] for c in be.calls} == {"ad"}
        assert be.calls[0][1] == [ord(c) for c in lt.FAULT_LETTERS]
        assert sum(len(c[0]) for c in be.calls) == 3


class Setup:
    """Typed reports for synthetic test incidents, from a trained metric model."""

    def __init__(self, n=12):
        train = (
            labelled(30, "TT", 1, 0)
            + labelled(20, "SS", 1, 100)
            + labelled(20, "OB", 1, 200)
        )
        data = t3.metric_dataset(train)
        self.spec = t3.MetricFaultSpecialist(
            t3._make({"hidden": 0, "epochs": 150}, 0).fit(data.x, data.y)
        )
        self.test = labelled(n, "SS", 2, 1000)

    def inputs(self, inc, log_noise=0.1):
        mm = self.spec.messages(inc)
        rng = np.random.default_rng(abs(hash(inc.id)) % 2**32)
        lm = []
        for n in inc.services:
            p = rng.dirichlet(np.ones(6))
            if n == inc.root_service:
                p = (
                    np.eye(6)[FAULT_TYPES.index(inc.fault_type)] * (1 - log_noise)
                    + log_noise / 6
                )
            lm.append(tr.trained_log_message(n, p))
        return mm, lm


class TestTeams:
    def test_records_and_ablations(self):
        setup = Setup()
        per = {i.id: (i.services, *setup.inputs(i)) for i in setup.test}
        pool = t3.build_pool3(per)
        inc = setup.test[0]
        mm, lm = setup.inputs(inc)
        recs = t3.team3_records(inc, "team_trained_both", mm, lm, "log", None, pool)
        names = [r.condition for r in recs]
        assert names[0] == "team_trained_both"
        assert {
            "team_trained_both:drop_root",
            "team_trained_both:shuffle_root",
            "team_trained_both:drop_other",
            "team_trained_both:shuffle_other",
            "team_trained_both:drop_fault_root",
            "team_trained_both:shuffle_fault_root",
        } == set(names[1:])
        main = recs[0]
        assert (
            main.top1 == 1 and main.pred_fault == inc.fault_type and main.fault_ok == 1
        )
        by = {r.condition.split(":")[1]: r for r in recs[1:]}
        assert by["drop_root"].top1 == 0  # the root's score is gone
        # dropping only the fault report keeps the ranking (the scores are intact)
        assert by["drop_fault_root"].ranking == main.ranking
        assert main.extra["bytes"] > 0
        assert all(r.model == "trained" for r in recs)

    def test_log_only_team_does_not_read_the_metric_fault_report(self):
        setup = Setup()
        inc = setup.test[1]
        mm, lm = setup.inputs(inc)
        wrong = [dict(m, fault="cpu", fault_probs=[1, 0, 0, 0, 0, 0]) for m in mm]
        stripped = t3.strip_fault_reports(wrong)
        assert all(m["fault"] == "unknown" for m in stripped)
        rec = t3.team3_records(
            inc, "team_trained_log", stripped, lm, "log", None, None, False
        )[0]
        assert rec.pred_fault == inc.fault_type

    def test_invalid_reports_are_refused(self):
        setup = Setup(2)
        inc = setup.test[0]
        mm, lm = setup.inputs(inc)
        with pytest.raises(ValueError, match="invalid"):
            t3.team3_records(
                inc,
                "team_trained_both",
                [dict(mm[0], fault="x")] + mm[1:],
                lm,
                "log",
                None,
            )

    def test_no_ablation_option(self):
        setup = Setup(2)
        inc = setup.test[0]
        mm, lm = setup.inputs(inc)
        assert (
            len(
                t3.team3_records(
                    inc, "team_trained_metric", mm, None, "metric", None, ablate=False
                )
            )
            == 1
        )


def synthetic_records(n=14):
    setup = Setup(n)
    cfg = t2.Settings2()
    plain_pool = t2.build_pool(
        {i.id: (i.services, t2.metric_messages(i), None) for i in setup.test}
    )
    inputs = {i.id: setup.inputs(i) for i in setup.test}
    pool = t3.build_pool3({i.id: (i.services, *inputs[i.id]) for i in setup.test})
    recs = []
    for inc in setup.test:
        mm, lm = inputs[inc.id]
        recs.append(t2.run_baro(inc))
        recs += t2.team_records(
            inc,
            "team_metric_rule",
            t2.metric_messages(inc),
            None,
            "none",
            None,
            None,
            cfg,
            pool=plain_pool,
        )
        recs += t3.team3_records(
            inc, "team_trained_metric", mm, None, "metric", None, pool
        )
        recs += t3.team3_records(
            inc, "team_trained_log", t3.strip_fault_reports(mm), lm, "log", None, pool
        )
        recs += t3.team3_records(
            inc, "team_trained_both", mm, lm, "mean_w50", None, pool
        )
        recs += t3.team3_records(
            inc, "team_trained_metric_ttonly", mm, None, "metric", None, pool
        )
    return [r.to_dict() for r in recs]


class TestReport:
    def test_report_and_verdict_on_synthetic_records(self):
        extras = {
            "metric": {
                "n_train": 70,
                "oof_per_system": {"TT": 0.5, "SS": 0.4, "OB": 0.45},
                "heuristic_per_system": {"TT": 0.3, "SS": 0.3, "OB": 0.3},
                "chosen": {"hidden": 0},
                "per_fault": {
                    f: {"n": 5, "oof_accuracy": 0.5, "heuristic_accuracy": 0.3}
                    for f in FAULT_TYPES
                },
                "grid": [
                    {
                        "config": {"hidden": 0},
                        "per_system": {"TT": 0.5, "SS": 0.4},
                        "mean": 0.45,
                    }
                ],
            },
            "fusion": {
                "n_oof": 10,
                "chosen": "metric",
                "n_by_system": {"TT": 10},
                "table": {"metric": {"TT": 0.5, "mean": 0.5}},
            },
            "lora": {
                "final": {
                    "held_out": None,
                    "n_fit_incidents": 5,
                    "n_examples": 15,
                    "steps": 7,
                    "mean_last_loss": 0.5,
                    "train_s": 12.0,
                    "peak_mib": 900.0,
                }
            },
        }
        text, payload = rp3.build_report(
            synthetic_records(), extras, analysis="hand-written"
        )
        assert "G4c" in text and "hand-written" in text and "analysis:start" in text
        assert payload["n_test_incidents"] == 14
        teams = payload["verdict"]["teams"]
        assert set(teams) == set(rp3.MAIN)
        assert payload["verdict"]["overall"] in {"PASS", "KILL", "INCONCLUSIVE"}
        assert all(v == 1.0 for v in payload["same_top1_as_baro"].values())
        assert any(a["ablation"] == "drop_fault_root" for a in payload["ablations"])
        assert any(g["group"] == "fault disk" for g in payload["joint_by_group"])
        # the log specialist here knows the answer: the log team must beat BARO on fault type
        assert (
            teams["team_trained_log"]["team_joint"]
            >= teams["team_trained_log"]["baro_joint"]
        )

    def test_pass_needs_margin_ci_holm_and_top1(self):
        rows = []
        for inc in labelled(40, "SS", 2, 0):
            rank = [inc.root_service] + [
                s for s in inc.services if s != inc.root_service
            ]
            wrong = "mem" if inc.fault_type != "mem" else "cpu"
            rows.append(
                lt.score_record(inc, "none", "baro_metrics", rank, wrong).to_dict()
            )
            for cond in rp3.MAIN:
                rows.append(
                    lt.score_record(
                        inc, "trained", cond, rank, inc.fault_type
                    ).to_dict()
                )
        _, payload = rp3.build_report(rows)
        assert payload["verdict"]["overall"] == "PASS"
        # an equal team is a KILL
        rows2 = [dict(r, pred_fault="x", fault_ok=0) for r in rows]
        _, payload2 = rp3.build_report(rows2)
        assert payload2["verdict"]["overall"] == "KILL"

    def test_dev_repetition_never_enters(self):
        recs = synthetic_records(6)
        dev = labelled(3, "SS", 1, 5000)[0]
        recs.append(
            lt.score_record(dev, "none", "baro_metrics", dev.services, "cpu").to_dict()
        )
        _, payload = rp3.build_report(recs)
        assert payload["n_test_incidents"] == 6


class TestScript:
    def load(self, monkeypatch):
        spec = importlib.util.spec_from_file_location("r10_step3", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.syspath_prepend(str(SCRIPT.parent))
        spec.loader.exec_module(mod)
        return mod

    def test_teams_and_report_find_nested_artifacts(self, tmp_path, monkeypatch):
        mod = self.load(monkeypatch)
        recs = synthetic_records()
        half = len(recs) // 2
        runs = tmp_path / "runs"
        for name, part in (("r10s3-a", recs[:half]), ("r10s3-b", recs[half:])):
            d = runs / name / "artifacts" / "nested"
            d.mkdir(parents=True)
            (d / "results.jsonl").write_text(
                "\n".join(json.dumps(r) for r in part) + "\n", encoding="utf-8"
            )
        agg = runs / "r10s3-zz" / "artifacts"
        agg.mkdir(parents=True)
        monkeypatch.setenv("GCA_RUN_DIR", str(agg))
        out = tmp_path / "out" / "r10_step3.md"
        assert mod.main(["report", "--from", "runs:r10s3-[ab]", "--out", str(out)]) == 0
        payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        assert payload["n_test_incidents"] == 14
        assert "G4c" in out.read_text(encoding="utf-8")
        # a hand-written analysis survives a re-run
        text = out.read_text(encoding="utf-8").replace(
            "*Analysis pending the full run.*", "kept note"
        )
        out.write_text(text, encoding="utf-8")
        assert mod.main(["report", "--from", "runs:r10s3-[ab]", "--out", str(out)]) == 0
        assert "kept note" in out.read_text(encoding="utf-8")
        sys.modules.pop("r10_step3", None)

    def test_data_command_refuses_a_cache_without_tt(self, tmp_path, monkeypatch):
        mod = self.load(monkeypatch)
        for system in ("SS", "OB"):
            d = tmp_path / "incidents" / system
            d.mkdir(parents=True)
            for rep in (1, 2):
                (d / f"a_cpu__{rep}.npz").write_bytes(b"")
        with pytest.raises(SystemExit, match="Train Ticket"):
            mod.main(
                ["data", "--cache-dir", str(tmp_path), "--out", str(tmp_path / "o")]
            )
        sys.modules.pop("r10_step3", None)
