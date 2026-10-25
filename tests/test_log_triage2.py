"""R10 step 2: metric and log specialists, centres, ablations, the object-layer run, the report
and the aggregation job. Synthetic data and a scripted reasoner only (no GPU, no Ollama);
the real-data check skips without the RCAEval cache."""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from genetic_coil_arena.envs.logtriage import heuristics as hz
from genetic_coil_arena.envs.logtriage import team as tm
from genetic_coil_arena.envs.logtriage.synthetic import synthetic_incident
from genetic_coil_arena.envs.logtriage.views import ApproxCounter
from genetic_coil_arena.experiments import log_triage as lt
from genetic_coil_arena.experiments import log_triage2 as t2
from genetic_coil_arena.experiments import log_triage2_report as rp2
from genetic_coil_arena.experiments.log_triage2_runtime import RuntimeTeam
from genetic_coil_arena.protocol.goalspec import SpecValidationError

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "r10_step2.py"


class FakeReasoner(lt.Reasoner):
    """Log specialist: fault from keywords in its view. Centre: highest metric score."""

    name = "fake"
    style = "json"

    def __init__(self, log_reply=None):
        self.counter = ApproxCounter()
        self.log_reply = log_reply
        self.centre_bodies: list[str] = []
        self.writes = 0

    def write(self, prompts, max_new_tokens, typed, schema=None):
        out = []
        for p in prompts:
            self.writes += 1
            assert schema is tm.LOG_SCHEMA
            name = re.search(r"service '([\w-]+)'", p).group(1)
            text = self.log_reply
            if text is None:
                fault = hz.guess_fault(p.splitlines(), default="unknown")
                text = json.dumps(
                    {
                        "service": "ignored",
                        "fault": fault,
                        "confidence": 0.9,
                        "evidence": ["ERROR x"],
                    }
                )
            assert name in p
            out.append(lt.Written(text, n_prompt=len(p) // 3, n_new=len(text) // 3))
        return out

    def decide(self, bodies, services, ctx):
        out = []
        for body, names in zip(bodies, services):
            self.centre_bodies.append(body)
            score = dict.fromkeys(names, 0.0)
            fault = "cpu"
            for line in body.splitlines():
                m = re.match(r"- ([\w-]+): metrics=(\{.*?\})(?: logs=(\{.*\}))?$", line)
                if m and m.group(1) in score:
                    score[m.group(1)] = json.loads(m.group(2))["score"]
                    if m.group(3) and json.loads(m.group(3))["fault"] != "unknown":
                        fault = (
                            fault
                            if score[m.group(1)] < max(score.values())
                            else json.loads(m.group(3))["fault"]
                        )
            ranking = sorted(names, key=lambda n: (-score[n], n))
            out.append(lt.Decision(ranking, fault, n_prompt=len(body) // 3, n_new=3))
        return out


def incidents(n=8, rep=2, base=0):
    return [synthetic_incident(base + i, rep=rep) for i in range(n)]


class TestSpecialists:
    def test_metric_message_is_valid_and_reads_only_its_service(self):
        inc = synthetic_incident(1)
        for s in range(inc.n_services):
            msg = tm.metric_message(inc, s)
            assert tm.metric_errors(msg) == []
            assert msg["service"] == inc.services[s]
            assert tm.nbytes(msg) < 200
        root = inc.services.index(inc.root_service)
        top = tm.metric_message(inc, root)
        assert top["top"][0]["z"] > 20 and top["onset_s"] == 12.0
        # a service's message depends on its own metrics only
        before = tm.metric_message(inc, root)
        inc.metrics = {inc.root_service: inc.metrics[inc.root_service]}
        assert tm.metric_message(inc, root) == before

    def test_score_is_baro(self):
        for inc in incidents(6):
            ours = np.array(
                [tm.metric_message(inc, s)["score"] for s in range(inc.n_services)]
            )
            assert np.allclose(ours, hz.baro_metrics(inc), atol=0.006)

    def test_log_view_is_own_service_and_fits(self):
        inc = synthetic_incident(2)
        c = ApproxCounter()
        s = inc.services.index(inc.root_service)
        view = tm.log_view(inc, s, 12.0, 300, c)
        assert c.count(view) <= 300
        assert inc.root_service in view
        assert all(
            f"svc{k}" not in view.replace(inc.root_service, "")
            for k in range(inc.n_services)
            if f"svc{k}" != inc.root_service
        )
        assert tm.log_view(inc, s, None, 120, c)  # no onset: window around the alert
        tiny = tm.log_view(inc, s, 12.0, 5, c)
        assert (
            tiny.count("\n") == 1
        )  # the header plus the empty-window line only when nothing fits

    def test_clean_log_message(self):
        ok = tm.clean_log_message(
            {
                "service": "x",
                "fault": "mem",
                "confidence": 3,
                "evidence": ["a" * 300, "b", "c", "d"],
            },
            "svc",
        )
        assert ok["service"] == "svc" and ok["confidence"] == 1.0
        assert (
            len(ok["evidence"]) == 3 and len(ok["evidence"][0]) == tm.MAX_EVIDENCE_CHARS
        )
        assert tm.clean_log_message({"fault": "banana"}, "svc") is None
        assert tm.clean_log_message("nope", "svc") is None

    def test_goals_are_validated(self):
        reg = tm.team_registry()
        reg.make(tm.SPEC_METRIC, {"incident": "i", "service": "a"})
        reg.make(tm.SPEC_LOG, {"incident": "i", "service": "a", "onset_s": None})
        with pytest.raises(SpecValidationError):
            reg.make(tm.SPEC_LOG, {"incident": "i", "service": "a"})
        goal = reg.make(tm.SPEC_METRIC, {"incident": "i", "service": "a"})
        assert reg.result_errors(goal, {"service": "a"})  # incomplete result is refused


class TestCentres:
    def test_fault_policies(self):
        mm = {
            "service": "a",
            "score": 9.0,
            "onset_s": 1.0,
            "top": [{"metric": "cpu", "dir": "up", "z": 9.0}],
        }
        lm = {"service": "a", "fault": "mem", "confidence": 0.6, "evidence": []}
        assert tm.fault_for("metric", mm, lm) == "cpu"
        assert tm.fault_for("log", mm, lm) == "mem"
        assert tm.fault_for("log_c50", mm, lm) == "mem"
        assert tm.fault_for("log_c80", mm, lm) == "cpu"
        assert tm.fault_for("log", mm, dict(lm, fault="unknown")) == "cpu"
        assert tm.fault_for("log", dict(mm, top=[]), None) == tm.DEFAULT_FAULT
        with pytest.raises(ValueError):
            tm.fault_for("nope", mm, lm)

    def test_rule_ranking_is_stable_argmax(self):
        mms = [tm.empty_metric_message(n) for n in "abc"]
        mms[2]["score"] = 5.0
        assert tm.rule_ranking(mms) == [2, 0, 1]

    def test_llm_centre_body_fits_and_drops_evidence_first(self):
        inc = synthetic_incident(0)
        mm = t2.metric_messages(inc)
        lm = [
            dict(tm.empty_log_message(n), fault="cpu", evidence=["x" * 70] * 3)
            for n in inc.services
        ]
        c = ApproxCounter()
        full = t2.centre_body(inc, mm, lm, c, 10_000)
        assert "xxxxxxxx" in full
        small = t2.centre_body(inc, mm, lm, c, c.count(full) - 40)
        assert "xxxxxxxx" not in small and "logs=" in small
        smaller = t2.centre_body(inc, mm, lm, c, 200)
        assert "logs=" not in smaller


def batch_records(incs, reasoner, hybrid_model="fake"):
    cfg = t2.Settings2()
    recs = []
    pool_m = t2.build_pool(
        {i.id: (i.services, t2.metric_messages(i), None) for i in incs}
    )
    logs = {}
    for inc in incs:
        mm = t2.metric_messages(inc)
        lm, info = t2.log_messages(inc, mm, reasoner, cfg)
        logs[inc.id] = (lm, info)
    pool_h = t2.build_pool(
        {i.id: (i.services, t2.metric_messages(i), logs[i.id][0]) for i in incs}
    )
    for inc in incs:
        mm = t2.metric_messages(inc)
        lm, info = logs[inc.id]
        recs.append(t2.run_baro(inc))
        recs += t2.team_records(
            inc, "team_metric_rule", mm, None, "none", None, None, cfg, pool=pool_m
        )
        recs += t2.team_records(
            inc,
            "team_metric_llm",
            mm,
            None,
            reasoner.name,
            None,
            reasoner,
            cfg,
            pool=pool_m,
        )
        recs += t2.team_records(
            inc, "team_hybrid_rule", mm, lm, hybrid_model, info, None, cfg, pool=pool_h
        )
        recs += t2.team_records(
            inc,
            "team_hybrid_llm",
            mm,
            lm,
            hybrid_model,
            info,
            reasoner,
            cfg,
            pool=pool_h,
        )
    return [r.to_dict() for r in recs]


class TestConditions:
    def test_metric_only_rule_team_is_distributed_baro(self):
        incs = incidents(10)
        recs = batch_records(incs, FakeReasoner())
        by = {(r["condition"], r["incident"]): r for r in recs}
        for inc in incs:
            a, b = by[("baro_metrics", inc.id)], by[("team_metric_rule", inc.id)]
            assert a["ranking"][0] == b["ranking"][0]
            assert a["pred_fault"] == b["pred_fault"]
            assert b["extra"]["bytes_log"] == 0 and b["extra"]["bytes_metric"] > 0

    def test_hybrid_records_and_ablations(self):
        incs = incidents(6)
        recs = batch_records(incs, FakeReasoner())
        names = {r["condition"] for r in recs}
        assert {
            "team_hybrid_rule:drop_root",
            "team_hybrid_rule:shuffle_other",
            "team_hybrid_llm:drop_root",
        } <= names
        assert (
            "team_hybrid_llm:drop_other" not in names
        )  # LLM centres: root ablations only
        for inc in incs:
            intact = next(
                r
                for r in recs
                if r["condition"] == "team_hybrid_rule" and r["incident"] == inc.id
            )
            dropped = next(
                r
                for r in recs
                if r["condition"] == "team_hybrid_rule:drop_root"
                and r["incident"] == inc.id
            )
            assert intact["top1"] == 1 and dropped["top1"] == 0
            assert set(intact["extra"]["fault_policies"]) >= set(tm.FAULT_POLICIES)
            assert intact["extra"]["bytes_log"] > 0

    def test_llm_ablations_skip_dev(self):
        dev = incidents(2, rep=1, base=50)
        recs = batch_records(dev, FakeReasoner())
        assert not [r for r in recs if r["condition"].startswith("team_hybrid_llm:")]
        assert [r for r in recs if r["condition"].startswith("team_hybrid_rule:")]

    def test_invalid_log_reply_becomes_an_empty_report(self):
        inc = synthetic_incident(4)
        r = FakeReasoner(log_reply="not json")
        lm, info = t2.log_messages(inc, t2.metric_messages(inc), r, t2.Settings2())
        assert info["valid"] == 0 and all(m["fault"] == "unknown" for m in lm)
        assert all(tm.log_errors(m) == [] for m in lm)

    def test_baro_fault_comes_from_the_metric_kind(self):
        inc = synthetic_incident(5)
        rec = t2.run_baro(inc)
        assert rec.ranking[0] == inc.root_service
        assert rec.fault_ok == 1 or inc.fault_type in ("loss",)


class TestRuntimePlumbing:
    def test_object_layer_matches_the_batch_path(self, tmp_path):
        incs = incidents(3)
        trace = tmp_path / "t.jsonl"
        team = RuntimeTeam(incs, FakeReasoner(), t2.Settings2(), trace_path=trace)
        batch = FakeReasoner()
        for inc in incs:
            res = team.triage(inc)
            mm = t2.metric_messages(inc)
            lm, _ = t2.log_messages(inc, mm, batch, t2.Settings2())
            ranking, faults = tm.rule_decide(inc.services, mm, lm)
            assert res["ranking"] == ranking and res["fault_type"] == faults["chosen"]
            assert res["metric_reports"] == mm and res["log_reports"] == lm
            # 2 envelopes per goal (goal + result); S metric + S log + 1 decide goals
            assert res["n_envelopes"] == 2 * (2 * inc.n_services + 1)
            assert res["envelope_bytes"] > res["report_bytes"] > 0
        team.close()
        from genetic_coil_arena.runtime.trace import TraceLog

        loaded = TraceLog.load(trace)
        assert len(loaded.trace_ids()) == 3
        assert loaded.situation(incs[0].id)["incident"] == incs[0].id

    def test_llm_centre_through_the_runtime(self):
        inc = synthetic_incident(7)
        centre = FakeReasoner()
        team = RuntimeTeam(
            [inc], FakeReasoner(), t2.Settings2(), centre="llm", centre_reasoner=centre
        )
        res = team.triage(inc)
        assert res["ranking"][0] == inc.root_service and centre.centre_bodies
        with pytest.raises(ValueError):
            RuntimeTeam([inc], FakeReasoner(), centre="llm")


def synthetic_records():
    dev = incidents(6, rep=1, base=100)
    test = incidents(14, rep=2, base=0)
    r = FakeReasoner()
    recs = batch_records(dev + test, r)
    # the singles are scripted: a weaker, noisy reader of the same evidence
    for inc in dev + test:
        ranking = list(inc.services)
        if inc.id.endswith("1") or hash(inc.id) % 3:
            ranking = [inc.root_service] + [s for s in ranking if s != inc.root_service]
        recs.append(
            lt.score_record(inc, "fake", "single_selected", ranking, "cpu").to_dict()
        )
        recs.append(
            lt.score_record(
                inc, "fake", "oracle_selected", ranking, inc.fault_type
            ).to_dict()
        )
    return recs


class TestReport:
    def test_report_and_verdict_on_synthetic_records(self):
        text, payload = rp2.build_report(synthetic_records(), analysis="hand-written")
        assert "G4b" in text and "hand-written" in text
        assert payload["n_test_incidents"] == 14
        assert payload["policies"]["fake"] in tm.FAULT_POLICIES
        teams = payload["verdict"]["teams"]
        assert set(teams) == {"team_hybrid_rule[fake]", "team_hybrid_llm[fake]"}
        assert payload["verdict"]["overall"] in {"PASS", "KILL", "INCONCLUSIVE"}
        assert payload["distributed_baro_check"]["same_top1_service"] == 1.0
        assert any(a["ablation"] == "drop_root" for a in payload["ablations"])
        # dev incidents never enter the table
        assert all(
            t["n"] == 14 for t in payload["table"] if t["condition"] == "baro_metrics"
        )

    def test_pass_and_kill_rules(self):
        rows = []
        incs = incidents(30, rep=2)
        for inc in incs:
            rank = [inc.root_service] + [
                s for s in inc.services if s != inc.root_service
            ]
            wrong = "mem" if inc.fault_type != "mem" else "cpu"
            rows.append(
                lt.score_record(inc, "none", "baro_metrics", rank, wrong).to_dict()
            )
            # a team that has the same ranking and always the right fault
            r = lt.score_record(
                inc, "m", "team_hybrid_rule", rank, inc.fault_type
            ).to_dict()
            r["extra"] = {
                "fault_policies": dict.fromkeys(tm.FAULT_POLICIES, inc.fault_type)
            }
            rows.append(r)
        _, payload = rp2.build_report(rows)
        v = payload["verdict"]["teams"]["team_hybrid_rule[m]"]
        assert v["verdict"] == "PASS" and v["non_inferior_top1"]
        assert payload["verdict"]["overall"] == "PASS"
        # same fault accuracy: the single is within the CI
        rows2 = [dict(r, fault_ok=r["fault_ok"]) for r in rows]
        for r in rows2:
            if r["condition"] == "team_hybrid_rule":
                r["extra"] = {"fault_policies": dict.fromkeys(tm.FAULT_POLICIES, "mem")}
                r["pred_fault"] = "mem"
                r["fault_ok"] = int(r["fault"] == "mem")
        for r in rows2:
            if r["condition"] == "baro_metrics":
                r["pred_fault"] = "mem"
                r["fault_ok"] = int(r["fault"] == "mem")
        _, payload2 = rp2.build_report(rows2)
        assert payload2["verdict"]["overall"] == "KILL"


class TestAggregationJob:
    def test_report_finds_nested_artifacts(self, tmp_path, monkeypatch):
        runs = tmp_path / "runs"
        recs = synthetic_records()
        half = len(recs) // 2
        for name, part in (("r10s2-a", recs[:half]), ("r10s2-b", recs[half:])):
            d = runs / name / "artifacts" / "nested"
            d.mkdir(parents=True)
            (d / "results.jsonl").write_text(
                "\n".join(json.dumps(r) for r in part) + "\n", encoding="utf-8"
            )
        (runs / "r10s2-a" / "artifacts" / "runtime_summary.json").write_text(
            json.dumps(
                {
                    "model": "m",
                    "n_incidents": 2,
                    "top1": 1.0,
                    "fault_ok": 0.5,
                    "joint": 0.5,
                    "batch_ranking_equal": True,
                    "batch_fault_equal": True,
                    "envelopes_per_incident": 60,
                    "envelope_bytes_per_incident": 20000,
                    "report_bytes_per_incident": 1500,
                    "rows": [],
                }
            ),
            encoding="utf-8",
        )
        agg = runs / "r10s2-zz" / "artifacts"
        agg.mkdir(parents=True)
        monkeypatch.setenv("GCA_RUN_DIR", str(agg))
        spec = importlib.util.spec_from_file_location("r10_step2", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        import sys

        monkeypatch.syspath_prepend(str(SCRIPT.parent))
        spec.loader.exec_module(mod)
        out = tmp_path / "out" / "r10_step2.md"
        assert mod.main(["report", "--from", "runs:r10s2-[ab]", "--out", str(out)]) == 0
        payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        assert payload["n_test_incidents"] == 14
        assert payload["runtime"][0]["envelopes_per_incident"] == 60
        assert "Object layer check" in out.read_text(encoding="utf-8")
        sys.modules.pop("r10_step2", None)


def test_real_incidents_if_cached():
    from genetic_coil_arena.envs.logtriage import rcaeval

    files = rcaeval.list_incidents(None, ("SS",))
    if not files:
        pytest.skip("RCAEval cache not present")
    for path in files[:: max(len(files) // 4, 1)][:4]:
        inc = rcaeval.load_incident(path)
        mm = t2.metric_messages(inc)
        assert all(tm.metric_errors(m) == [] for m in mm)
        assert max(tm.nbytes(m) for m in mm) < 400
        rec = t2.run_baro(inc)
        team = t2.team_records(
            inc, "team_metric_rule", mm, None, "none", None, None, t2.Settings2()
        )[0]
        scores = dict(zip(inc.services, hz.baro_metrics(inc)))
        assert abs(scores[rec.ranking[0]] - scores[team.ranking[0]]) < 0.006
