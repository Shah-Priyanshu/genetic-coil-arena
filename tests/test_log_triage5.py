"""R10 step 5 (G4e): evidence blocks, the closed-set reader, shuffle ablations, the threshold
variant and the verdict rule. Synthetic data and a stub backend only (no GPU, no Ollama).
"""

import json

import numpy as np
import pytest

from genetic_coil_arena.backends.base import GenerationResult
from genetic_coil_arena.envs.logtriage import rerank as rr
from genetic_coil_arena.envs.logtriage.synthetic import synthetic_incident
from genetic_coil_arena.envs.logtriage.views import ApproxCounter
from genetic_coil_arena.experiments import log_triage as lt
from genetic_coil_arena.experiments import log_triage4 as l4
from genetic_coil_arena.experiments import log_triage5 as l5
from genetic_coil_arena.experiments import log_triage5_report as rp5


def re3(seed, rep=2, system="RE3-SS"):
    inc = synthetic_incident(seed, rep=rep)
    inc.system = system
    inc.id = f"{system}/syn{seed}_f1/{rep}"
    return inc


class StubBackend:
    """Answers with the candidate chosen by ``pick(names, prompt)``."""

    base_id = "stub"

    def __init__(self, pick):
        self.pick = pick
        self.prompts = []

    def generate(
        self, prompts, system=None, schema=None, max_new_tokens=None, options=None
    ):
        out = []
        for p in prompts:
            self.prompts.append(p)
            names = schema["properties"]["root"]["enum"]
            text = self.pick(names, p)
            out.append(
                GenerationResult(text=text, n_prompt_tokens=len(p) // 3, n_new_tokens=5)
            )
        return out


class StubReasoner(lt.Reasoner):
    name = "stub"
    style = "json"

    def __init__(self, backend):
        self.counter = ApproxCounter()
        self.backend = backend


def setup(inc):
    cache = l4.all_parts([inc])
    fusion = {"baro_metrics": 1.0, "new_ext": 1.0}
    order, scores = l5.fusion_order(inc, cache[inc.id], fusion)
    return order, scores, l5.metric_messages(inc)


def reply(root, conf=0.9):
    return json.dumps({"root": root, "confidence": conf})


class TestEvidence:
    def test_block_budget_and_content(self):
        inc = re3(3)
        order, scores, mm = setup(inc)
        counter = ApproxCounter()
        body = rr.evidence_body(inc, order[:3], scores, mm, counter)
        assert counter.count(body) <= rr.TOTAL_TOKENS
        for s in order[:3]:
            assert f"== {inc.services[s]} (ranker position" in body
        assert "NEW after the alert" in body and "usual templates" in body
        tiny = rr.evidence_body(inc, order[:3], scores, mm, counter, total=150)
        assert counter.count(tiny) < counter.count(body)

    def test_blind_hides_rank(self):
        inc = re3(3)
        order, scores, mm = setup(inc)
        body = rr.evidence_body(
            inc, order[:3], scores, mm, ApproxCounter(), show_rank=False
        )
        assert "ranker" not in body

    def test_root_error_template_is_listed_as_new(self):
        inc = re3(3)
        order, scores, mm = setup(inc)
        root = inc.services.index(inc.root_service)
        block = rr.evidence_block(inc, root, mm[root], 1, 1.0, 500, ApproxCounter())
        assert "[error]" in block or "[warn]" in block or "[lifecycle]" in block

    def test_parse_choice(self):
        names = ["a", "b", "c"]
        assert rr.parse_choice(reply("b", 2.0), names) == ("b", 1.0)
        assert rr.parse_choice('{"root": "z", "confidence": 0.5}', names)[0] is None
        assert rr.parse_choice("not json", names)[0] is None
        assert rr.parse_choice('{"root": "a"}', names)[0] is None


class TestReader:
    def test_valid_choice_moves_to_front(self):
        inc = re3(4)
        order, scores, mm = setup(inc)
        third = inc.services[order[2]]
        rd = StubReasoner(StubBackend(lambda names, p: reply(third, 0.8)))
        rec = l5.read_shortlist(
            inc, "stub", "rerank3", order, scores, mm, rd, l5.Settings5()
        )
        assert rec.ranking[0] == third and len(rec.ranking) == inc.n_services
        assert rec.extra["overridden"] == 1 and rec.valid == 1
        assert rec.extra["fusion_ranking"][0] == inc.services[order[0]]
        assert set(rec.extra["shortlist"]) == {inc.services[s] for s in order[:3]}

    def test_invalid_reply_keeps_fusion_top1(self):
        inc = re3(4)
        order, scores, mm = setup(inc)
        rd = StubReasoner(StubBackend(lambda names, p: "garbage"))
        rec = l5.read_shortlist(
            inc, "stub", "rerank3", order, scores, mm, rd, l5.Settings5()
        )
        assert rec.valid == 0 and rec.ranking[0] == inc.services[order[0]]
        assert rec.extra["overridden"] == 0

    def test_shuffle_changes_order_but_not_candidates(self):
        inc = re3(5)
        order, scores, mm = setup(inc)
        top = order[:3]
        shown = l5.shown_order(inc, top, "rerank3:shuffle")
        assert shown != top and sorted(shown) == sorted(top)
        assert l5.shown_order(inc, top, "rerank3") == top
        assert l5.shown_order(inc, top, "rerank3:shuffle") == shown  # deterministic
        rd = StubReasoner(StubBackend(lambda names, p: reply(names[0])))
        rec = l5.read_shortlist(
            inc, "stub", "rerank3:shuffle_blind", order, scores, mm, rd, l5.Settings5()
        )
        assert "ranker" not in rd.backend.prompts[0]
        assert rec.extra["shown"][0] == rec.extra["choice"]
        with pytest.raises(ValueError):
            l5.read_shortlist(
                inc, "stub", "nope", order, scores, mm, rd, l5.Settings5()
            )


def fake_rows(good_fusion, good_reader, n=40, model="ollama:m", conf=0.9):
    rows = []
    for i in range(n):
        rep = 1 if i < 10 else 2
        iid = f"RE3-SS/c{i}/{rep}"
        base = {"incident": iid, "system": "RE3-SS", "rep": rep, "root": "r", "top3": 1}
        f_ok = i < good_fusion
        rows.append(
            dict(
                base,
                model="none",
                condition="nonllm_fusion",
                top1=int(f_ok),
                ranking=["r"] if f_ok else ["x", "r"],
                extra={},
            )
        )
        r_ok = i < good_reader
        fusion_rk = ["r", "x"] if f_ok else ["x", "r"]
        pick = "r" if r_ok else "x"
        rk = [pick] + [n_ for n_ in fusion_rk if n_ != pick]
        for cond in ("rerank3", "rerank3:shuffle"):
            rows.append(
                dict(
                    base,
                    model=model,
                    condition=cond,
                    top1=int(rk[0] == "r"),
                    ranking=rk,
                    valid=1,
                    extra={
                        "choice": pick,
                        "confidence": conf,
                        "fusion_top1": fusion_rk[0],
                        "shown": [fusion_rk[0], "z"],
                        "overridden": int(pick != fusion_rk[0]),
                        "fusion_ranking": fusion_rk,
                    },
                )
            )
    return rows


class TestReport:
    def test_pass_when_reader_clearly_better(self):
        rows = fake_rows(good_fusion=15, good_reader=40)
        report, payload = rp5.build_report(rows)
        assert payload["verdict"]["overall"] == "PASS"
        assert "G4e verdict: PASS" in report and "Overrides" in report
        o = [r for r in payload["overrides"] if r["condition"] == "rerank3"][0]
        assert o["right"] == o["overrides"] and o["wrong"] == 0

    def test_kill_when_reader_worse_or_equal(self):
        _, p = rp5.build_report(fake_rows(good_fusion=40, good_reader=20))
        assert p["verdict"]["overall"] == "KILL"
        _, p = rp5.build_report(fake_rows(good_fusion=30, good_reader=30))
        assert p["verdict"]["overall"] == "KILL"

    def test_threshold_uses_dev_and_blocks_low_confidence_overrides(self):
        rows = fake_rows(good_fusion=5, good_reader=40, conf=0.6)
        chosen, table = rp5.pick_thresholds(rows)
        assert table["ollama:m"]["n_dev"] == 10 and chosen["ollama:m"] <= 0.6
        r = [
            x for x in rows if x["condition"] == "rerank3" and x["extra"]["overridden"]
        ][0]
        kept = rp5.with_threshold(r, 0.5)
        blocked = rp5.with_threshold(r, 0.9)
        assert kept["ranking"][0] == r["extra"]["choice"]
        assert blocked["ranking"] == r["extra"]["fusion_ranking"]
        assert blocked["extra"]["overridden"] == 0

    def test_split_rows_use_repetition(self):
        rows = fake_rows(20, 20)
        _, p = rp5.build_report(rows)
        ref = p["verdict"]["reference"]
        assert ref["test_top1"] == pytest.approx(
            np.mean([i < 20 for i in range(10, 40)])
        )
