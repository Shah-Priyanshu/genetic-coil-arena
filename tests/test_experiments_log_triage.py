"""R10 conditions, report and verdict with a scripted reasoner (no model, no data)."""

import json
import re

import numpy as np
import pytest

from genetic_coil_arena.envs.logtriage import messages as msgs
from genetic_coil_arena.envs.logtriage.synthetic import synthetic_incident
from genetic_coil_arena.envs.logtriage.views import ApproxCounter
from genetic_coil_arena.evaluation.stats import paired_bootstrap_ci
from genetic_coil_arena.experiments import log_triage as lt
from genetic_coil_arena.experiments import log_triage_report as rp


class FakeReasoner(lt.Reasoner):
    """Reads the evidence like a rule: the service with the most 'anomaly' evidence wins."""

    name = "fake"
    style = "letters"

    def __init__(self):
        self.counter = ApproxCounter()
        self.calls = {"decide": 0, "write": 0}
        self.bodies: list[str] = []
        self.ctxs: list[int] = []
        self.break_json = False

    def decide(self, bodies, services, ctx):
        out = []
        for body, names in zip(bodies, services):
            self.calls["decide"] += 1
            self.bodies.append(body)
            self.ctxs.append(ctx)
            score = dict.fromkeys(names, 0)
            for line in body.splitlines():
                m = re.match(r"- ([\w-]+): (.*)", line)
                if m and m.group(1) in score:
                    if '"anomaly":true' in m.group(2):
                        score[m.group(1)] += 5
                    if "abnormal" in m.group(2):
                        score[m.group(1)] += 5
                elif "ERROR" in line or "z=" in line:
                    for n in names:
                        if f"[{n}]" in line or line.startswith(f"{n}:"):
                            score[n] += 1
            ranking = sorted(names, key=lambda n: (-score[n], n))
            out.append(lt.Decision(ranking, "cpu", n_prompt=len(body) // 3, n_new=2))
        return out

    def write(self, prompts, max_new_tokens, typed):
        out = []
        for p in prompts:
            self.calls["write"] += 1
            name = re.search(r"service '([\w-]+)'", p).group(1)
            if typed:
                text = (
                    "not json"
                    if self.break_json
                    else json.dumps(
                        {
                            "service": "wrong-name-is-overwritten",
                            "anomaly": "z=" in p,
                            "first_anomaly_s": 3.0,
                            "line_rate_ratio": 1.0,
                            "top_errors": [{"template": "ERROR x", "count": 2}],
                            "suspected_fault": "cpu",
                        }
                    )
                )
            else:
                text = (
                    f"{name} looks abnormal. " * 40 if "z=" in p else f"{name} is fine."
                )
            out.append(lt.Written(text, n_prompt=len(p) // 3, n_new=len(text) // 3))
        return out


CFG = lt.Settings()


def test_decision_prompt_lists_options_and_rejects_too_many_services():
    text = lt.decision_prompt("EVIDENCE", ["a", "b", "c"], "letters")
    assert "A. a" in text and "C. c" in text and "a. cpu" in text and "EVIDENCE" in text
    assert "JSON" in lt.decision_prompt("e", ["a", "b"], "json")
    with pytest.raises(ValueError):
        lt.decision_prompt("e", [str(i) for i in range(27)])


def test_single_conditions_fit_the_context():
    inc = synthetic_incident(1, lines_per_service=2000)
    r = FakeReasoner()
    for cond in lt.SINGLE_CONDITIONS:
        rec = lt.run_single(inc, cond, r, CFG)
        ctx = CFG.oracle_ctx_tokens if cond == "oracle_selected" else CFG.ctx_tokens
        full = r.counter.count(lt.decision_prompt(r.bodies[-1], inc.services))
        assert full + r.wrapper_tokens() <= ctx
        assert rec.condition == cond and rec.n_services == inc.n_services
        assert rec.top1 in (0, 1) and rec.top3 >= rec.top1
    # head/tail truncation sees almost nothing of the post-alert errors, selection sees them
    assert r.ctxs[:2] == [CFG.ctx_tokens] * 2


def test_selected_single_finds_the_root_when_truncation_is_blind():
    hits = dict.fromkeys(("single_head", "single_selected"), 0)
    r = FakeReasoner()
    for seed in range(8):
        inc = synthetic_incident(seed, lines_per_service=1500)
        for c in hits:
            hits[c] += lt.run_single(inc, c, r, CFG).top1
    assert hits["single_selected"] >= 7
    assert hits["single_head"] <= hits["single_selected"]


def test_team_typed_budgets_validity_and_centre_context():
    inc = synthetic_incident(2, lines_per_service=800)
    r = FakeReasoner()
    recs = lt.run_team(inc, "typed", r, CFG, pool={})
    base = recs[0]
    assert base.condition == "team_typed"
    assert base.extra["valid_messages"] == 1.0
    assert max(base.extra["message_tokens"]) <= CFG.message_tokens
    assert base.centre_tokens > 0 and base.prompt_tokens > base.centre_tokens
    assert r.calls["write"] == inc.n_services
    # every centre prompt is within the context
    for body in r.bodies:
        fixed = r.counter.count(lt.decision_prompt(body, inc.services))
        assert fixed + r.wrapper_tokens() <= CFG.ctx_tokens
    names = {x.condition for x in recs}
    assert names == {"team_typed"} | {f"team_typed:{a}" for a in lt.ABLATIONS}
    assert base.top1 == 1


def test_invalid_specialist_json_becomes_an_empty_message():
    inc = synthetic_incident(3)
    r = FakeReasoner()
    r.break_json = True
    rec, *_ = lt.run_team(inc, "typed", r, CFG, pool={}, ablate=False)
    assert rec.extra["valid_messages"] == 0.0


def test_free_text_messages_are_cut_to_the_budget():
    inc = synthetic_incident(4)
    r = FakeReasoner()
    messages, info = lt.specialist_messages(inc, "free", r, CFG)
    assert all(r.counter.count(m) <= CFG.message_tokens for m in messages)
    recs = lt.run_team(inc, "free", r, CFG, pool={}, precomputed=(messages, info))
    assert recs[0].condition == "team_free"


def test_dropping_the_roots_message_hurts_dropping_another_does_not():
    r = FakeReasoner()
    drop_root = drop_other = base = 0
    n = 10
    for seed in range(n):
        inc = synthetic_incident(seed, lines_per_service=600)
        recs = {x.condition: x for x in lt.run_team(inc, "typed", r, CFG, pool={})}
        base += recs["team_typed"].top1
        drop_root += recs["team_typed:drop_root"].top1
        drop_other += recs["team_typed:drop_other"].top1
    assert base == n and drop_other >= n - 1 and drop_root < base


def test_shuffle_uses_the_same_service_from_another_incident():
    r = FakeReasoner()
    incs = [synthetic_incident(s) for s in range(4)]
    pool: dict[str, list] = {}
    store = {}
    for inc in incs:
        m, info = lt.specialist_messages(inc, "typed", r, CFG)
        store[inc.id] = (m, info)
        for name, msg in zip(inc.services, m):
            pool.setdefault(f"typed|{name}", []).append(msg)
    inc = incs[0]
    recs = lt.run_team(inc, "typed", r, CFG, pool=pool, precomputed=store[inc.id])
    assert any(x.condition.endswith("shuffle_root") for x in recs)


def test_heuristic_and_typed_heur_conditions_run_without_a_model():
    inc = synthetic_incident(5)
    for c in lt.HEURISTIC_CONDITIONS:
        rec = lt.run_heuristic(inc, c)
        assert rec.model == "none"
        assert set(rec.ranking) == set(inc.services)
    r = FakeReasoner()
    recs = lt.run_team(inc, "typed_heur", r, CFG, ablate=False)
    assert len(recs) == 1 and recs[0].condition == "team_typed_heur"
    assert r.calls["write"] == 0 and recs[0].top1 == 1


def test_overflow_stats_counts_tokens():
    inc = synthetic_incident(6, lines_per_service=3000)
    o = lt.overflow_stats(inc, ApproxCounter(), 4096)
    assert o["full_tokens"] > 20 * 4096 and o["ratio_unique"] < o["ratio_full"]


def test_jsonl_roundtrip_skips_torn_lines(tmp_path):
    p = tmp_path / "r.jsonl"
    lt.append_jsonl(p, [{"a": 1}, {"a": 2}])
    with open(p, "a", encoding="utf-8") as f:
        f.write('{"a": 3')  # killed mid-write
    assert [x["a"] for x in lt.read_jsonl(p)] == [1, 2]
    assert lt.read_jsonl(tmp_path / "missing.jsonl") == []


# ----------------------------------------------------------------------------- report


def _rec(model, cond, inc, top1, **kw):
    base = {
        "model": model,
        "condition": cond,
        "incident": f"SS/c{inc}/2",
        "system": "SS",
        "rep": 2,
        "n_services": 10,
        "root": "a",
        "fault": "cpu",
        "ranking": ["a"],
        "pred_fault": "cpu",
        "top1": top1,
        "top3": top1,
        "fault_ok": top1,
        "prompt_tokens": 100,
        "new_tokens": 10,
        "centre_tokens": 80,
        "tokens": 110,
        "latency_s": 1.0,
        "valid": 1,
        "extra": {},
    }
    base.update(kw)
    return base


def _records(team_hits, single_hits, n=60):
    rows = []
    for i in range(n):
        rows.append(_rec("m", "team_typed", i, int(i < team_hits)))
        for c in ("single_head", "single_tail", "single_selected"):
            rows.append(_rec("m", c, i, int(i < single_hits)))
        rows.append(_rec("m", "team_typed:drop_root", i, int(i < team_hits // 2)))
        rows.append(_rec("none", "heur_baro_metrics", i, int(i < 5)))
    rows.append(
        {
            "kind": "overflow",
            "incident": "SS/c0/2",
            "system": "SS",
            "n_lines": 9,
            "n_services": 10,
            "n_templates": 3,
            "full_tokens": 800000,
            "unique_template_tokens": 9000,
            "ratio_full": 195.3,
            "ratio_unique": 2.2,
            "largest_service_lines": 5,
            "counter": "approx3",
            "ctx": 4096,
        }
    )
    return rows


def test_verdict_pass_kill_and_inconclusive():
    def verdict(team_hits, single_hits):
        recs = [
            r for r in rp.dedupe(_records(team_hits, single_hits)) if r["model"] == "m"
        ]
        team = rp.vectors(recs, "m", "team_typed")
        singles = {c: rp.vectors(recs, "m", c) for c in rp.LLM_SINGLES}
        return rp.g4_verdict(team, singles)

    assert verdict(45, 20)["verdict"] == "PASS"
    assert verdict(20, 20)["verdict"] == "KILL"
    assert verdict(10, 40)["verdict"] == "KILL"
    # a significant but sub-5-point gain is neither pass nor kill
    rows = [r for r in rp.dedupe(_records(0, 0, n=400)) if r["model"] == "m"]
    team = {r["incident"]: 1.0 for r in rows if r["condition"] == "team_typed"}
    single = {k: (1.0 if i % 40 else 0.0) for i, k in enumerate(team)}
    team_small = dict.fromkeys(team, 1.0)
    v = rp.g4_verdict(team_small, {"single_head": single})
    assert v["verdict"] == "INCONCLUSIVE" and v["diff"] < 0.05


def test_verdict_not_run_without_records():
    assert rp.g4_verdict({}, {})["verdict"] == "NOT RUN"


def test_best_single_is_chosen_on_the_same_incidents():
    team = {f"i{k}": float(k % 2) for k in range(40)}
    weak = {f"i{k}": 0.0 for k in range(40)}
    strong = {f"i{k}": float(k % 2) for k in range(40)}
    v = rp.g4_verdict(team, {"single_head": weak, "single_selected": strong})
    assert v["best_single"] == "single_selected" and v["verdict"] == "KILL"


def test_report_renders_tables_ablations_and_carries_analysis(tmp_path):
    text, payload = rp.build_report(_records(45, 20), analysis="Hand-written words.")
    assert "G4 verdict" in text and "PASS" in text
    assert "drop_root" in text and "Overflow" in text and "195x" in text
    assert "(c) BARO on service metrics" in text
    assert payload["verdicts"]["m"]["verdict"] == "PASS"
    out = tmp_path / "r.md"
    out.write_text(text, encoding="utf-8")
    assert rp.carried_analysis(out) == "Hand-written words."
    text2, _ = rp.build_report(_records(45, 20), analysis=rp.carried_analysis(out))
    assert "Hand-written words." in text2
    assert rp.carried_analysis(tmp_path / "none.md") is None


def test_dedupe_keeps_the_last_record():
    a = _rec("m", "team_typed", 1, 0)
    b = _rec("m", "team_typed", 1, 1)
    assert [r["top1"] for r in rp.dedupe([a, b])] == [1]


def test_paired_bootstrap_ci_is_paired_and_seeded():
    x = np.array([1.0, 1, 1, 0, 1, 1, 0, 1, 1, 1] * 5)
    y = np.array([0.0, 0, 1, 0, 0, 1, 0, 0, 1, 0] * 5)
    d, lo, hi = paired_bootstrap_ci(x, y, n_boot=2000, seed=3)
    assert d == pytest.approx(0.5) and 0 < lo < d < hi
    assert paired_bootstrap_ci(x, y, n_boot=2000, seed=3) == (d, lo, hi)
    d0, lo0, hi0 = paired_bootstrap_ci(x, x)
    assert d0 == 0 and lo0 == hi0 == 0
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [1.0, 2.0])
    assert np.isnan(paired_bootstrap_ci([1.0], [0.0])[1])


def test_triage_registry_only_registers_once():
    reg = msgs.triage_registry()
    assert msgs.TRIAGE_SPEC in reg.names()
