"""R10 step 4 (G4d): the RE3 loader, the split and leak check, the new-template log baselines,
the typed log reports and teams, the report and the verdict rule. Synthetic data and fake
models only (no GPU); the real-data checks skip without the RE3 cache."""

import io
import json
import re

import numpy as np
import pytest

from genetic_coil_arena.envs.logtriage import logsignal as lg
from genetic_coil_arena.envs.logtriage import rcaeval
from genetic_coil_arena.envs.logtriage import team as tm
from genetic_coil_arena.envs.logtriage import team4 as t4
from genetic_coil_arena.envs.logtriage.synthetic import synthetic_incident
from genetic_coil_arena.envs.logtriage.views import ApproxCounter
from genetic_coil_arena.experiments import log_triage as lt
from genetic_coil_arena.experiments import log_triage4 as l4
from genetic_coil_arena.experiments import log_triage4_report as rp4
from genetic_coil_arena.experiments import log_triage4_teams as t4x


def re3(seed, rep=1, system="RE3-SS", **kw):
    inc = synthetic_incident(seed, rep=rep, **kw)
    inc.system = system
    inc.id = f"{system}/syn{seed}_f1/{rep}"
    inc.fault_type = "f1"
    return inc


# ---------------------------------------------------------------------------- loader


class TestRE3Loader:
    def test_case_names(self):
        assert rcaeval.parse_case("carts_f1") == ("carts", "f1")
        assert rcaeval.parse_case("ts-route-service_f3_1") == (
            "ts-route-service",
            "f3_1",
        )
        assert rcaeval.parse_case("front-end_f2") == ("front-end", "f2")
        # RE2 names are unchanged
        assert rcaeval.parse_case("payment_mem") == ("payment", "mem")
        assert rcaeval.parse_case("recommendation_service_cpu") == (
            "recommendation_service",
            "cpu",
        )
        with pytest.raises(ValueError):
            rcaeval.parse_case("nonsense")

    def test_archives_and_datasets(self):
        for system in rcaeval.RE3_SYSTEMS:
            name, size, md5 = rcaeval.ARCHIVES[system]
            assert name == system + ".zip" and size > 0 and len(md5) == 32
            assert rcaeval.dataset_of(system) == "RE3"
        assert rcaeval.dataset_of("SS") == "RE2"
        # RE2 defaults and consumers are untouched
        assert rcaeval.DEFAULT_SYSTEMS == ("SS", "OB")
        assert set(rcaeval.ARCHIVES) >= {"SS", "OB", "TT"}

    def test_convert_case_with_re3_fault_id(self):
        logs = (
            "time,timestamp,container_name,message,pod_name,node_name\n"
            "04:14,1000000000000,carts,hello,p,n\n"
            "04:14,1005000000000,carts,java.lang.NullPointerException boom,p,n\n"
            "04:14,1002000000000,front-end,ok,p,n\n"
        )
        inc = rcaeval.convert_case(
            io.StringIO(logs), 1001.0, "carts", "f1", 2, "RE3-SS", "carts_f1"
        )
        assert inc.id == "RE3-SS/carts_f1/2" and inc.fault_type == "f1"
        assert inc.services == ["carts", "front-end"] and inc.n_lines == 3


@pytest.fixture
def cache(tmp_path):
    for seed, rep, system in [
        (1, 1, "RE3-SS"),
        (2, 2, "RE3-SS"),
        (3, 3, "RE3-SS"),
        (4, 1, "RE3-OB"),
        (5, 2, "RE3-OB"),
        (6, 4, "RE3-TT"),
    ]:
        inc = re3(seed, rep, system)
        case = inc.id.split("/")[1]
        rcaeval.save_incident(inc, rcaeval._incident_path(tmp_path, system, case, rep))
    return tmp_path


class TestSplit:
    def test_roles(self):
        assert l4.role4("RE3-SS", 1) == "dev"
        assert l4.role4("RE3-TT", 4) == "test"
        assert l4.role4("SS", 1) == "other"  # RE2 is not part of this split

    def test_load_splits_by_repetition(self, cache):
        dev, test = l4.load_incidents(cache)
        assert {i.rep for i in dev} == {1} and len(dev) == 2
        assert {i.rep for i in test} == {2, 3, 4} and len(test) == 4
        assert not {i.id for i in dev} & {i.id for i in test}

    def test_leak_check(self):
        a, b = re3(1, 1), re3(2, 2)
        l4.assert_split([l4.key_of(a)], [l4.key_of(b)])
        with pytest.raises(ValueError, match="dev and test"):
            l4.assert_split([l4.key_of(a)], [l4.key_of(a)])
        with pytest.raises(ValueError, match="not dev"):
            l4.assert_split([l4.key_of(b)], [])
        with pytest.raises(ValueError, match="not test"):
            l4.assert_split([], [l4.key_of(a)])
        with pytest.raises(ValueError, match="not test"):
            l4.assert_split([], [("SS/x/2", "SS", 2)])  # RE2 data is not RE3 test data

    def test_shard_and_limit_are_stable_and_disjoint(self, cache):
        parts = [l4.load_incidents(cache, shard=(k, 3)) for k in range(3)]
        ids = [i.id for dev, test in parts for i in dev + test]
        assert len(ids) == len(set(ids)) == 6
        limited = l4.load_incidents(cache, limit=3)
        assert len(limited[0]) + len(limited[1]) == 3

    def test_describe(self, cache):
        dev, test = l4.load_incidents(cache)
        d = l4.describe(dev + test)
        assert d["RE3-SS"]["n"] == 3 and d["RE3-SS"]["reps"] == {1: 1, 2: 1, 3: 1}
        assert set(d) == {"RE3-SS", "RE3-OB", "RE3-TT"}


@pytest.mark.skipif(
    not rcaeval.list_incidents(None, rcaeval.RE3_SYSTEMS),
    reason="RE3 cache absent (scripts/fetch_rcaeval.py --systems RE3-SS,RE3-OB,RE3-TT)",
)
class TestRealRE3:
    def test_counts_and_split(self):
        dev, test = l4.load_incidents()
        assert len(dev) + len(test) == 90
        assert all(i.rep == 1 for i in dev) and all(i.rep != 1 for i in test)
        d = l4.describe(dev + test)
        assert {s: v["n"] for s, v in d.items()} == {
            "RE3-SS": 30,
            "RE3-OB": 30,
            "RE3-TT": 30,
        }
        assert all(i.root_service in i.services for i in dev + test)
        assert all(i.fault_type.startswith("f") for i in dev + test)


# ------------------------------------------------------------------- log baselines


class TestLogSignal:
    def test_new_error_scorers_find_the_root(self):
        inc = synthetic_incident(3)
        root = inc.services.index(inc.root_service)
        for mode in lg.LOG_SCORERS:
            scores = lg.new_template_scores(inc, mode)
            assert scores.shape == (inc.n_services,)
            assert scores[root] > 0, mode
        with pytest.raises(ValueError):
            lg.new_template_scores(inc, "nope")

    def test_no_new_templates_scores_zero(self):
        inc = synthetic_incident(3)
        inc.t = np.abs(inc.t) * -1.0 - 1.0  # nothing after the alert
        inc.t.sort()
        inc._stats = None
        assert not lg.new_template_scores(inc, "new_err").any()

    def test_lifecycle_templates_count_in_new_ext(self):
        inc = synthetic_incident(4)
        inc.templates[0] = "INFO Application started. Press Ctrl+C to shut down."
        inc._tpl_sev = None
        inc._stats = None
        base = lg.new_template_scores(inc, "new_err")
        ext = lg.new_template_scores(inc, "new_ext")
        assert (ext >= base).all()

    def test_reciprocal_ranks_share_ties(self):
        rr = lg.reciprocal_ranks(np.array([5.0, 9.0, 5.0, 1.0]))
        assert rr.tolist() == [1 / 2, 1.0, 1 / 2, 1 / 4]

    def test_fuse_gives_silent_sources_no_credit(self):
        f = lg.fuse_ranks(np.array([0.0, 4.0, 2.0]), np.array([0.0, 0.0, 1.0]), 1.0)
        assert f[0] == 0.0 and f[2] == pytest.approx(0.5 + 1.0)

    def test_gate_stats(self):
        inc = synthetic_incident(3)
        g = lg.gate_stats(inc)
        assert g["root_new_err"] > 0 and g["root_rank"] == 1
        assert g["root_strictly_first"] is True


class TestNonLlmSystems:
    def test_tie_ranking_is_deterministic_and_unbiased_by_index(self):
        inc = synthetic_incident(1)
        zeros = np.zeros(inc.n_services)
        a = l4.tie_ranking(zeros, inc, "x")
        assert a == l4.tie_ranking(zeros, inc, "x")
        assert sorted(a) == list(range(inc.n_services))
        firsts = {
            l4.tie_ranking(zeros, synthetic_incident(s), "x")[0] for s in range(20)
        }
        assert len(firsts) > 1  # not always service 0
        assert l4.tie_ranking([0, 3, 1], inc)[0] == 1

    def test_fusion_grid_and_pick(self):
        grid = l4.weight_grid()
        assert len(grid) == len(l4.FUSION_WEIGHTS) ** len(l4.FUSION_PARTS) - 1
        dev = [re3(s, 1) for s in range(5)]
        cache = l4.all_parts(dev)
        w = l4.pick_fusion(dev, cache)
        assert set(w) == set(l4.FUSION_PARTS) and any(w.values())
        hits = [
            l4.top1_of(l4.tie_ranking(l4.fused_scores(cache[i.id], w), i, "fuse"), i)
            for i in dev
        ]
        assert np.mean(hits) >= 0.8

    def test_records(self):
        incs = [re3(s, 1 + s % 2) for s in range(4)]
        cache = l4.all_parts(incs)
        fusion = {"baro_metrics": 1.0, "new_ext": 1.0}
        recs = l4.non_llm_records(incs, cache, fusion)
        conds = {r.condition for r in recs}
        assert "nonllm_fusion" in conds and "nonllm_new_ext" in conds
        assert len(recs) == 4 * len(conds)
        assert all(r.model == "none" and r.pred_fault == l4.NO_FAULT for r in recs)
        assert next(r for r in recs if r.condition == "nonllm_fusion").extra["fusion"]

    def test_gate_summary(self):
        incs = [re3(s, 1) for s in range(3)]
        s = l4.gate_summary(l4.gate_rows(incs))
        assert s["RE3-SS"]["n"] == 3 and s["RE3-SS"]["root_has_new_err"] == 1.0


# --------------------------------------------------------------------- typed reports


class TestTeam4:
    def test_schema_and_cleaning(self):
        ok = t4.clean_log4_message(
            {
                "service": "x",
                "suspicion": 1.7,
                "kind": "exception",
                "evidence": ["a"] * 5,
            },
            "carts",
        )
        assert ok["service"] == "carts" and ok["suspicion"] == 1.0
        assert len(ok["evidence"]) == t4.MAX_EVIDENCE
        assert t4.clean_log4_message({"suspicion": 0.5}, "carts") is None
        assert (
            t4.clean_log4_message(
                {"service": "x", "suspicion": 0.5, "kind": "weird", "evidence": []}, "x"
            )
            is None
        )
        assert t4.clean_log4_message(None, "x") is None
        assert not t4.log4_errors(t4.empty_log4_message("a"))

    def test_view_note_when_no_metric_onset(self):
        inc = synthetic_incident(2)
        counter = ApproxCounter()
        with_onset = t4.log4_view(inc, 0, 12.0, 800, counter)
        without = t4.log4_view(inc, 0, None, 800, counter)
        assert "left their normal range at about +12s" in with_onset
        assert "centred on the alert" in without
        # step 2/3 behaviour of the shared view is unchanged
        assert tm.log_view(inc, 0, None, 800, counter).count("normal range") == 1

    def test_prompt_has_view_and_service(self):
        p = t4.log4_prompt("carts", "VIEW-TEXT")
        assert "'carts'" in p and p.endswith("VIEW-TEXT") and "suspicion" in p

    def test_fused_scores(self):
        mm = [{"score": s} for s in (10.0, 5.0, 1.0)]
        lm = [{"suspicion": s} for s in (0.0, 0.9, 0.0)]
        assert t4.fused_scores(mm, lm, "log_only").tolist() == [0.0, 1.0, 0.0]
        assert int(np.argmax(t4.fused_scores(mm, lm, "w2"))) == 1
        assert int(np.argmax(t4.fused_scores(mm, lm, "w0.5"))) == 0
        lex = t4.fused_scores(mm, [{"suspicion": 0.5}] * 3, "lex")
        assert int(np.argmax(lex)) == 0  # equal suspicion: the metric breaks the tie
        # a metric below the threshold gets no credit
        assert t4.fused_scores(mm, lm, "w1")[2] == 0.0

    def test_centre_body_fits_budget_and_hides_silent_services(self):
        inc = synthetic_incident(5, n_services=12)
        mm = [tm.metric_message(inc, s) for s in range(12)]
        lm = [t4.empty_log4_message(n) for n in inc.services]
        lm[3] = {
            "service": inc.services[3],
            "suspicion": 0.9,
            "kind": "exception",
            "evidence": ["E" * 70] * 3,
        }
        counter = ApproxCounter()
        body = t4.centre_body(mm, lm, counter, 4000)
        assert f"- {inc.services[3]}:" in body and "suspicion" in body
        tight = t4.centre_body(mm, lm, counter, counter.count(body) - 20)
        assert counter.count(tight) < counter.count(body)
        tiny = t4.centre_body(mm, lm, counter, 200)
        assert inc.services[3] in tiny  # the strongest report is kept longest

    def test_parse_ranking(self):
        names = ["a", "b", "c"]
        assert t4.parse_ranking('{"ranking": ["b", "zzz", "b"]}', names) == (
            ["b", "a", "c"],
            True,
        )
        assert t4.parse_ranking("not json", names) == (names, False)
        assert t4.parse_ranking('{"ranking": []}', names)[1] is False
        schema = t4.centre_schema(names)
        assert schema["properties"]["ranking"]["items"]["enum"] == names


# ------------------------------------------------------------------------ LLM teams


class FakeOllamaBackend:
    base_id = "fake"

    def __init__(self, picks):
        self.picks = picks  # incident root name the "model" always answers
        self.calls = []

    def generate(
        self, prompts, system=None, schema=None, max_new_tokens=None, options=None
    ):
        from genetic_coil_arena.backends.base import GenerationResult

        out = []
        for p in prompts:
            self.calls.append(p)
            names = schema["properties"]["ranking"]["items"]["enum"]
            pick = next((n for n in names if f"- {n}:" in p and "0.9" in p), names[0])
            out.append(
                GenerationResult(
                    text=json.dumps({"ranking": [pick]}),
                    n_prompt_tokens=len(p) // 3,
                    n_new_tokens=5,
                )
            )
        return out


class FakeReasoner(lt.Reasoner):
    """Log monitor: suspicious when its own view holds a new ERROR template."""

    name = "fake"
    style = "json"

    def __init__(self):
        self.counter = ApproxCounter()
        self.backend = FakeOllamaBackend(None)
        self.writes = 0

    def write(self, prompts, max_new_tokens, typed, schema=None):
        out = []
        for p in prompts:
            self.writes += 1
            assert schema is t4.LOG4_SCHEMA
            hot = bool(re.search(r"^\s+\d+/0: .*(ERROR|WARN)", p, re.M))
            text = json.dumps(
                {
                    "service": "ignored",
                    "suspicion": 0.9 if hot else 0.0,
                    "kind": "exception" if hot else "none",
                    "evidence": ["ERROR x"] if hot else [],
                }
            )
            out.append(lt.Written(text, n_prompt=len(p) // 3, n_new=len(text) // 3))
        return out

    def wrapper_tokens(self):
        return 10


class TestTeams:
    def setup_method(self):
        self.cfg = t4x.Settings4(log_view_tokens=400, ctx_tokens=2500)
        self.inc = re3(7, rep=2)
        self.reasoner = FakeReasoner()

    def messages(self):
        mm = t4x.metric_messages(self.inc)
        lm, info = t4x.log_messages4(self.inc, mm, self.reasoner, self.cfg)
        return mm, lm, info

    def test_log_messages_are_valid_and_one_per_service(self):
        mm, lm, info = self.messages()
        assert len(lm) == self.inc.n_services == self.reasoner.writes
        assert info["valid"] == self.inc.n_services
        assert [m["service"] for m in lm] == self.inc.services
        assert all(not t4.log4_errors(m) for m in lm)
        root = self.inc.services.index(self.inc.root_service)
        assert lm[root]["suspicion"] > 0

    def test_rule_team_records_and_ablations(self):
        mm, lm, info = self.messages()
        pool = t4x.build_pool4(
            {
                self.inc.id: (self.inc.services, mm, lm),
                "other": (self.inc.services, mm, [dict(m, suspicion=0.1) for m in lm]),
            },
            {self.inc.id: self.inc.root_service, "other": "elsewhere"},
        )
        # only reports sent while the service was not the root are drawn
        assert [p[2] for p in pool[self.inc.root_service]] == [True, False]
        recs = t4x.team_records4(
            self.inc, "fake", mm, lm, info, None, self.cfg, pool=pool
        )
        by = {r.condition: r for r in recs}
        assert set(by) == {
            "team4_rule",
            "team4_rule:drop_root",
            "team4_rule:shuffle_root",
            "team4_rule:drop_other",
            "team4_rule:shuffle_other",
        }
        main = by["team4_rule"]
        assert set(main.extra["rank_by_rule"]) == set(t4.RULE_WEIGHTS)
        assert main.top1 == 1 and main.pred_fault == l4.NO_FAULT
        assert (
            by["team4_rule:drop_root"].extra["ablated_service"] == self.inc.root_service
        )

    def test_llm_centre_and_test_only_ablations(self):
        mm, lm, info = self.messages()
        recs = t4x.team_records4(
            self.inc, "fake", mm, lm, info, self.reasoner, self.cfg, rule=False
        )
        conds = [r.condition for r in recs]
        assert conds[0] == "team4_llm" and "team4_llm:drop_root" in conds
        assert recs[0].top1 == 1 and recs[0].valid == 1
        # the dev split gets no LLM-centre ablations
        dev = re3(7, rep=1)
        mm2 = t4x.metric_messages(dev)
        lm2, info2 = t4x.log_messages4(dev, mm2, self.reasoner, self.cfg)
        recs = t4x.team_records4(
            dev, "fake", mm2, lm2, info2, self.reasoner, self.cfg, rule=False
        )
        assert [r.condition for r in recs] == ["team4_llm"]

    def test_no_ablation_flag(self):
        mm, lm, info = self.messages()
        recs = t4x.team_records4(
            self.inc, "fake", mm, lm, info, None, self.cfg, ablate=False
        )
        assert [r.condition for r in recs] == ["team4_rule"]

    def test_single_conditions(self):
        for cond in t4x.SINGLE4:
            rec = t4x.run_single4(self.inc, cond, self.reasoner, self.cfg)
            assert rec.condition == cond and rec.model == "fake" and rec.valid == 1
        # logs-only view carries no metric text
        assert "metrics:" in self.reasoner.backend.calls[0]
        assert "metrics:" not in self.reasoner.backend.calls[1]
        with pytest.raises(ValueError):
            t4x.run_single4(self.inc, "single_x", self.reasoner, self.cfg)


# ------------------------------------------------------------------------- report


def row(model, cond, inc, top1, rep, system="RE3-SS", ranks=None):
    return {
        "model": model,
        "condition": cond,
        "incident": inc,
        "system": system,
        "rep": rep,
        "root": "r",
        "ranking": ["r"] if top1 else ["x", "r"],
        "top1": top1,
        "top3": 1,
        "tokens": 10,
        "extra": {"rank_by_rule": ranks} if ranks else {},
    }


def make_rows(team_good, ref_good, n=60):
    rows = []
    for i in range(n):
        rep = 1 if i < 12 else 2
        iid = f"RE3-SS/c{i}/{rep}"
        rows.append(row("none", "nonllm_fusion", iid, int(i < ref_good), rep))
        rows.append(row("none", "nonllm_baro_metrics", iid, int(i % 2 == 0), rep))
        good = i < team_good
        ranks = {r: (["r"] if good else ["x", "r"]) for r in t4.RULE_WEIGHTS}
        rows.append(row("ollama:m", "team4_rule", iid, int(good), rep, ranks=ranks))
    return rows


class TestReport:
    def test_pick_rules_uses_dev_only(self):
        rows = make_rows(60, 30)
        for r in rows:
            if r["condition"] == "team4_rule" and r["rep"] == 1:
                r["extra"]["rank_by_rule"]["w2"] = ["x", "r"]
                r["extra"]["rank_by_rule"]["log_only"] = ["x", "r"]
        chosen, table = rp4.pick_rules(rows)
        assert chosen["ollama:m"] not in ("w2", "log_only")
        assert table["ollama:m"]["n_dev"] == 12

    def test_pass_when_team_clearly_beats_the_reference(self):
        rows = make_rows(team_good=60, ref_good=36)
        report, payload = rp4.build_report(rows)
        assert payload["verdict"]["overall"] == "PASS"
        assert "G4d verdict: PASS" in report and "<!-- analysis:start -->" in report

    def test_kill_when_team_equals_the_reference(self):
        rows = make_rows(team_good=36, ref_good=36)
        _, payload = rp4.build_report(rows)
        assert payload["verdict"]["overall"] == "KILL"

    def test_inconclusive_when_significant_but_below_margin(self):
        rows = []
        for i in range(500):
            rep = 1 if i < 100 else 2
            iid = f"RE3-SS/c{i}/{rep}"
            ref = int(i % 2 == 0)
            team = int(ref or i % 25 == 1)  # +4 points, never worse
            rows.append(row("none", "nonllm_fusion", iid, ref, rep))
            ranks = {r: (["r"] if team else ["x", "r"]) for r in t4.RULE_WEIGHTS}
            rows.append(row("ollama:m", "team4_rule", iid, team, rep, ranks=ranks))
        _, payload = rp4.build_report(rows)
        (team,) = payload["verdict"]["teams"].values()
        assert 0 < team["vs_dev_best"]["diff"] < rp4.MARGIN
        assert team["vs_dev_best"]["low"] > 0 and team["verdict"] == "INCONCLUSIVE"
        assert payload["verdict"]["overall"] == "INCONCLUSIVE"

    def test_reference_is_chosen_on_dev(self):
        rows = make_rows(60, 36)
        sys_ = rp4.Systems(rows, {})
        ref_dev, ref_test = rp4.best_nonllm(sys_)
        assert ref_dev == ("none", "nonllm_fusion")
        assert ref_test is not None

    def test_ablation_rows_and_single_comparison(self):
        rows = make_rows(60, 36)
        for r in list(rows):
            if r["condition"] == "team4_rule":
                rows.append(
                    dict(
                        r,
                        condition="team4_rule:drop_root",
                        top1=0,
                        ranking=["x"],
                        extra={
                            "rank_by_rule": {k: ["x", "r"] for k in t4.RULE_WEIGHTS}
                        },
                    )
                )
                rows.append(
                    dict(
                        r,
                        model="ollama:m",
                        condition="single_all",
                        top1=int(r["incident"][-1] == "2" and False),
                        extra={},
                    )
                )
        _, payload = rp4.build_report(rows)
        ab = payload["ablations"]
        assert ab and ab[0]["ablation"] == "drop_root" and ab[0]["diff"] < 0
        assert payload["single_vs_team"][0]["diff"] > 0

    def test_gate_text_and_nonllm_table(self):
        inc = re3(1, 1)
        gate = {
            "data": l4.describe([inc]),
            "gate": l4.gate_summary(l4.gate_rows([inc])),
            "gate_test": l4.gate_summary(l4.gate_rows([inc])),
        }
        text = rp4.gate_text(gate)
        assert "RE3-SS" in text and "Gate" in text
        report, _ = rp4.build_report(make_rows(60, 36), gate=gate, note="note")
        assert "Non-LLM systems" in report and report.count("RE3-SS") >= 2

    def test_dedupe_keeps_the_last_row(self):
        a = row("m", "c", "i", 0, 2)
        b = dict(a, top1=1)
        assert rp4.dedupe([a, b, {"kind": "gate"}]) == [b]
