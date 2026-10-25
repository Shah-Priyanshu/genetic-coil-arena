"""R7a step 2: the refusal policy, training data, the leak check, report maths.

Everything runs with fake backends: no Ollama, no GPU.
"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from genetic_coil_arena.backends import AdapterTextBackend
from genetic_coil_arena.experiments import r7a_sealed
from genetic_coil_arena.experiments import r7a_train_data as td
from genetic_coil_arena.experiments.r7a_domains import (
    PLANT_DEFAULTS,
    SQUAD_DEFAULTS,
    SQUAD_MISSION,
    build_domains,
    build_registry,
)
from genetic_coil_arena.experiments.r7a_evalset import EvalItem, few_shot_texts
from genetic_coil_arena.experiments.r7a_policy import policy_packs, with_policy_packs
from genetic_coil_arena.experiments.r7a_report import (
    clarify_scores,
    evaluate_forward,
    forward_metrics,
    verdict,
)
from genetic_coil_arena.experiments.r7a_step2 import (
    find_file,
    lora_pack,
    reconstruct_rows,
    resolve_runs,
    select_best_run,
)
from genetic_coil_arena.experiments.r7a_step2_report import build_report, render
from genetic_coil_arena.translator import (
    CLARIFY,
    ERROR,
    INVALID,
    OK,
    REFUSED,
    Example,
    OverlapIndex,
    PolicyTranslator,
    containment,
    encode_reply,
    fill_defaults,
    format_prompt,
    frame_ngrams,
    frame_tokens,
    policy_reply_schema,
    relax_schema,
    retry_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
GOOD = {"eaters": ["red"], "caution": "high", "step_limit": 120, "food_target": 0}


@dataclass
class Gen:
    text: str
    latency_s: float = 0.1
    n_prompt_tokens: int = 10


class Fake:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, prompts, adapter_id=None, **kw):
        self.calls.append((list(prompts), kw))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return [Gen(reply if isinstance(reply, str) else json.dumps(reply))]


@pytest.fixture(scope="module")
def registry():
    return build_registry()


@pytest.fixture(scope="module")
def packs():
    return {d: policy_packs(d)[0] for d in ("squad", "plant")}


def translate(registry, packs, replies, domain="squad", **kw):
    backend = Fake(replies)
    tr = PolicyTranslator(backend, registry, **kw)
    return tr, backend, packs[domain]


# ---------------------------------------------------------------------- policy


class TestSchema:
    def test_relaxed_schema_keeps_structure_and_drops_constraints(self, registry):
        full = registry.get(SQUAD_MISSION).params_schema
        relaxed = relax_schema(full)
        assert "required" not in relaxed
        props = relaxed["properties"]
        assert props["step_limit"] == {"type": "integer"}
        assert props["caution"] == {"type": "string"}
        assert props["eaters"] == {"type": "array", "items": {"type": "string"}}
        assert full["properties"]["step_limit"]["maximum"] == 300  # original untouched

    def test_reply_schema_has_refusal_clarify_and_relaxed_params(self, registry):
        s = policy_reply_schema(registry, SQUAD_MISSION)
        assert set(s["properties"]) == {"refusal", "clarify", "params"}
        assert s["required"] == ["refusal", "clarify"]
        assert "enum" not in json.dumps(s["properties"]["params"])

    def test_fill_defaults_does_not_overwrite_and_drops_nulls(self):
        out = fill_defaults({"caution": "low", "step_limit": None}, SQUAD_DEFAULTS)
        assert out == {"caution": "low", "step_limit": 300, "food_target": 0}

    def test_example_reply_in_policy_mode_has_both_keys(self):
        e = Example("q", None, "", "which one?")
        assert json.loads(e.reply(policy=True)) == {
            "refusal": "",
            "clarify": "which one?",
        }
        assert json.loads(Example("x", {"a": 1}).reply(policy=True))["clarify"] == ""
        assert "clarify" not in json.loads(Example("x", {"a": 1}).reply())


class TestPolicy:
    def test_missing_optional_fields_take_documented_defaults(self, registry, packs):
        tr, backend, pack = translate(
            registry, packs, [{"params": {"eaters": ["blue"]}}]
        )
        res = tr.translate("blue forages", pack)
        assert res.status == OK and res.attempts == 1
        assert res.params == {"eaters": ["blue"], **SQUAD_DEFAULTS}

    def test_plant_with_nothing_stated_is_the_default_config(self, registry, packs):
        tr, _, pack = translate(registry, packs, [{"params": {}}], domain="plant")
        assert tr.translate("run it", pack).params == PLANT_DEFAULTS

    def test_mandatory_field_missing_becomes_a_question(self, registry, packs):
        tr, backend, pack = translate(
            registry, packs, [{"params": {"step_limit": 100}}]
        )
        res = tr.translate("play it safe for 100 steps", pack)
        assert res.status == CLARIFY and res.spec is None
        assert "Which snakes" in res.question and res.reason == res.question
        assert len(backend.calls) == 1  # a question is final, no retry

    def test_model_can_ask_a_question(self, registry, packs):
        tr, _, pack = translate(
            registry, packs, [{"refusal": "", "clarify": "How long?"}]
        )
        res = tr.translate("forage for a while", pack)
        assert (res.status, res.question) == (CLARIFY, "How long?")

    def test_refusal_is_final(self, registry, packs):
        tr, backend, pack = translate(registry, packs, [{"refusal": "not an order"}])
        res = tr.translate("make coffee", pack)
        assert res.status == REFUSED and res.reason == "not an order"
        assert len(backend.calls) == 1

    def test_forbidden_value_is_sent_back_once_then_the_model_may_refuse(
        self, registry, packs
    ):
        bad = {"params": {"eaters": ["red"], "step_limit": 1000}}
        tr, backend, pack = translate(
            registry, packs, [bad, {"refusal": "1000 steps is out of range"}]
        )
        res = tr.translate("red forages 1000 steps", pack)
        assert res.status == REFUSED and res.attempts == 2
        retry_text = backend.calls[1][0][0]
        assert "rejected" in retry_text and "1000" in retry_text
        assert "Instruction: red forages 1000 steps" in retry_text

    def test_value_still_forbidden_after_the_retry_is_refused_as_unsatisfiable(
        self, registry, packs
    ):
        bad = {"params": {"eaters": ["red"], "step_limit": 1000}}
        tr, _, pack = translate(registry, packs, [bad, bad])
        res = tr.translate("red forages 1000 steps", pack)
        assert res.status == REFUSED
        assert res.reason.startswith("unsatisfiable under the schema")
        assert res.spec is None and res.errors

    def test_retry_can_fix_a_slip(self, registry, packs):
        slip = {"params": {"eaters": ["red"], "target_far": 1}}
        tr, _, pack = translate(
            registry,
            packs,
            [{"params": {"byte_budget": 100}}, {"params": {"byte_budget": 8}}],
            domain="plant",
        )
        res = tr.translate("eight bytes", pack)
        assert res.status == OK and res.attempts == 2
        assert res.params["byte_budget"] == 8
        assert slip  # (kept for readability of the failing case above)

    def test_whole_floats_count_as_integers(self, registry, packs):
        tr, _, pack = translate(
            registry, packs, [{"params": {"eaters": ["red"], "step_limit": 120.0}}]
        )
        assert tr.translate("x", pack).params["step_limit"] == 120

    def test_non_json_is_retried_then_invalid(self, registry, packs):
        tr, backend, pack = translate(registry, packs, ["not json", "still not"])
        res = tr.translate("x", pack)
        assert res.status == INVALID and res.attempts == 2

    def test_missing_params_without_refusal_or_question_is_invalid(
        self, registry, packs
    ):
        tr, _, pack = translate(registry, packs, [{"refusal": ""}, {"refusal": ""}])
        assert tr.translate("x", pack).status == INVALID

    def test_backend_failure_is_a_typed_error(self, registry, packs):
        tr, _, pack = translate(registry, packs, [RuntimeError("boom")])
        res = tr.translate("x", pack)
        assert res.status == ERROR and "boom" in res.reason

    def test_unknown_key_is_rejected_by_the_registry(self, registry, packs):
        reply = {"params": {"eaters": ["red"], "speed": 3}}
        tr, _, pack = translate(registry, packs, [reply, reply])
        assert tr.translate("x", pack).spec is None

    def test_request_carries_the_policy_schema_and_examples(self, registry, packs):
        tr, backend, pack = translate(
            registry, packs, [{"params": {"eaters": ["red"]}}]
        )
        tr.translate("red forages", pack)
        kwargs = backend.calls[0][1]
        assert kwargs["schema"]["required"] == ["refusal", "clarify"]
        assert len(kwargs["examples"]) == len(pack.examples)
        assert all(json.loads(r)["clarify"] is not None for _, r in kwargs["examples"])

    def test_retries_can_be_disabled(self, registry, packs):
        bad = {"params": {"eaters": ["red"], "step_limit": 1000}}
        tr, backend, pack = translate(registry, packs, [bad], retries=0)
        assert tr.translate("x", pack).status == REFUSED
        assert len(backend.calls) == 1

    def test_never_returns_an_unvalidated_spec(self, registry, packs):
        for reply in (
            {"params": {"eaters": ["purple"]}},
            {"params": {"eaters": []}},
            {"params": {"eaters": ["red"], "caution": "extreme"}},
        ):
            tr, _, pack = translate(registry, packs, [reply, reply])
            res = tr.translate("x", pack)
            assert res.spec is None and res.status != OK


class TestPolicyPacks:
    def test_packs_have_defaults_questions_and_clarify_examples(self):
        for name in ("squad", "plant"):
            pack, report = policy_packs(name)
            assert pack.defaults and report.system.startswith(pack.system)
            assert any(e.clarify for e in pack.examples)
            assert any(e.refusal for e in pack.examples)
            assert len(report.examples) > len(pack.examples)
        assert policy_packs("squad")[0].questions["eaters"]

    def test_few_shot_texts_share_nothing_with_the_sealed_set(self):
        sealed = r7a_sealed.load_sealed()
        index = OverlapIndex([i.text for i in sealed], 4, td.MASKS)
        for name in ("squad", "plant"):
            pack, report = policy_packs(name)
            for e in report.examples:
                score, _ = index.max_containment(e.text)
                assert score < 0.5, (e.text, score)
        assert not {i.text for i in sealed} & few_shot_texts()

    def test_with_policy_packs_swaps_only_the_packs(self):
        domains = build_domains(None, 1, 1)
        swapped = with_policy_packs(domains)
        assert swapped["squad"].executor is domains["squad"].executor
        assert swapped["squad"].pack.defaults == SQUAD_DEFAULTS
        assert domains["squad"].pack.defaults == {}


# ---------------------------------------------------------------------- distill / overlap


class TestDistillFormat:
    def test_defaults_are_left_out_of_the_reply(self):
        reply = encode_reply(params=GOOD | {"food_target": 0}, defaults=SQUAD_DEFAULTS)
        assert json.loads(reply) == {
            "params": {"caution": "high", "eaters": ["red"], "step_limit": 120}
        }
        assert " " not in reply
        assert json.loads(
            encode_reply(
                params=SQUAD_DEFAULTS | {"eaters": ["red"]}, defaults=SQUAD_DEFAULTS
            )
        ) == {"params": {"eaters": ["red"]}}

    def test_refusal_and_clarify_replies(self):
        assert json.loads(encode_reply(refusal="no")) == {"refusal": "no"}
        assert json.loads(encode_reply(clarify="which?")) == {"clarify": "which?"}

    def test_compact_reply_round_trips_through_the_policy(self, registry, packs):
        for gold in (
            GOOD,
            {
                "eaters": ["red", "blue"],
                "caution": "normal",
                "step_limit": 300,
                "food_target": 9,
            },
        ):
            tr, _, pack = translate(registry, packs, [td.spec_reply("squad", gold)])
            assert tr.translate("x", pack).params == gold

    def test_prompt_format_is_stable(self):
        assert (
            format_prompt("squad.mission", "go")
            == "Task: squad.mission\nRequest: go\nReply:\n"
        )

    def test_adapter_backend_formats_prompts_and_ignores_the_rest(self):
        class HF:
            base_id = "base"

            def __init__(self):
                self.seen = None

            def generate(self, prompts, adapter_id, max_new_tokens):
                self.seen = (list(prompts), adapter_id, max_new_tokens)
                return [Gen("{}")]

        hf = HF()
        be = AdapterTextBackend(hf, "ad", max_new_tokens=50)
        be.generate(
            ["do it"],
            system="squad.mission",
            examples=[("a", "b")],
            schema={},
            keep_alive="1m",
        )
        assert hf.seen == ([format_prompt("squad.mission", "do it")], "ad", 50)

    def test_retry_prompt_names_the_errors_and_the_instruction(self):
        text = retry_prompt(
            "red eats", ["step_limit: 1000 is greater than the maximum of 300"]
        )
        assert (
            "rejected" in text
            and "1000" in text
            and text.endswith("Instruction: red eats")
        )


class TestOverlap:
    MASKS = td.MASKS

    def test_frames_ignore_numbers_colours_and_case(self):
        a = frame_tokens("Red forages for 120 steps.", self.MASKS)
        b = frame_tokens("blue forages for one hundred twenty steps.", self.MASKS)
        assert a == b == ["<C>", "forages", "for", "<N>", "steps", "."]

    def test_containment_is_directional(self):
        ref = frame_ngrams("alpha beta gamma delta epsilon", 3)
        cand = frame_ngrams("alpha beta gamma delta", 3)
        assert containment(ref, cand) == pytest.approx(2 / 3)
        assert containment(cand, ref) == 1.0

    def test_index_finds_near_copies_and_ignores_unrelated_text(self):
        refs = ["Send blue to eat 120 steps.", "Use 8 bytes per message."]
        index = OverlapIndex(refs, 4, self.MASKS)
        score, which = index.max_containment("Send red to eat 90 steps.")
        assert score == 1.0 and which == 0
        assert (
            index.max_containment("A completely different sentence about snakes.")[0]
            < 0.5
        )

    def test_short_texts_compare_as_a_whole(self):
        index = OverlapIndex(["Go now."], 4, self.MASKS)
        assert index.max_containment("go now.")[0] == 1.0
        assert index.max_containment("stop now.")[0] == 0.0


# ---------------------------------------------------------------------- training data


@pytest.fixture(scope="module")
def templates():
    return td.make_templates(150, seed=3)


class TestTemplates:
    def test_gold_is_valid_and_replies_decode_to_it(self, templates, registry):
        specs = [r for r in templates if r.kind == "spec"]
        assert len(specs) >= 300
        for rec in specs:
            registry.make(td.SPEC_NAMES[rec.domain], rec.gold)
            filled = fill_defaults(
                json.loads(rec.reply)["params"], td.DEFAULTS[rec.domain]
            )
            assert filled == rec.gold, rec.text

    def test_mix_has_refusals_clarifications_and_reports(self, templates):
        kinds = {
            k: sum(r.kind == k for r in templates)
            for k in ("spec", "refuse", "clarify")
        }
        assert kinds["refuse"] > 0.06 * len(templates)
        assert kinds["clarify"] > 0.03 * len(templates)
        assert any("report" in r.style for r in templates)
        for r in templates:
            if r.kind == "refuse":
                assert "refusal" in json.loads(r.reply)
            if r.kind == "clarify":
                assert json.loads(r.reply)["clarify"].endswith(("?", "."))

    def test_deterministic_and_unique_texts(self):
        a = td.make_templates(40, seed=5)
        b = td.make_templates(40, seed=5)
        assert [r.text for r in a] == [r.text for r in b]
        keyed = [(r.domain, r.text) for r in a if "report" not in r.style]
        assert len(set(keyed)) == len(keyed)

    def test_records_round_trip_json(self, templates):
        rec = templates[0]
        assert td.Record.from_json(rec.to_json()) == rec

    def test_typo_keeps_numbers(self):
        import random

        rng = random.Random(0)
        out = td.typo(rng, "Squad forages for 120 steps carefully")
        assert "120" in out and out != "Squad forages for 120 steps carefully"

    def test_far_forms_are_one_of_the_documented_spellings(self):
        import random

        rng = random.Random(1)
        seen = {td.far_forms(rng, 0.005) for _ in range(60)}
        assert {"0.5%", "0.005"} <= seen or any("1 in 200" in s for s in seen)


class TestLeakControl:
    def test_filter_drops_sealed_copies_and_keeps_the_rest(self, templates):
        sealed = [i.text for i in r7a_sealed.load_sealed()]
        copy = td.Record(
            "x", "squad", "spec", "paraphrase", "p", sealed[0].lower(), "{}"
        )
        kept, dropped = td.filter_overlap([*templates, copy], sealed)
        assert copy in [d[0] for d in dropped]
        assert len(kept) >= 0.9 * len(templates)

    def test_assert_no_leak_raises_on_a_frame_copy_with_other_numbers(self):
        sealed = [i.text for i in r7a_sealed.load_sealed()]
        text = sealed[0].replace("red", "green").replace("90", "120")
        rec = td.Record("x", "squad", "spec", "paraphrase", "p", text, "{}")
        with pytest.raises(ValueError, match="overlap"):
            td.assert_no_leak([rec], sealed)

    def test_templates_after_filtering_are_clean_against_every_held_out_set(
        self, templates
    ):
        refs = [
            i.text
            for i in (
                *r7a_sealed.load_sealed(),
                *r7a_sealed.load_dev(),
                *r7a_sealed.load_old_set(),
            )
        ]
        kept, _ = td.filter_overlap(templates, refs)
        td.assert_no_leak(kept, refs)

    def test_training_pool_never_contains_a_sealed_text_verbatim(self, templates):
        sealed = {i.text.lower() for i in r7a_sealed.load_sealed()}
        kept, _ = td.filter_overlap(
            templates, [i.text for i in r7a_sealed.load_sealed()]
        )
        assert not {r.text.lower() for r in kept} & sealed


class TestParaphrases:
    def test_prompt_states_meaning_and_style_but_no_sealed_text(self, templates):
        rec = next(r for r in templates if r.kind == "spec" and r.domain == "squad")
        prompt = td.paraphrase_prompt("squad", rec, "typos", 6)
        assert (
            "Foraging snakes" in prompt
            and "spelling mistakes" in prompt
            and "6" in prompt
        )
        for item in r7a_sealed.load_sealed():
            assert item.text not in prompt

    def test_generation_with_a_fake_backend_writes_rows_and_resumes(self, templates):
        class Gen2:
            def __init__(self):
                self.n = 0

            def generate(self, prompts, **kw):
                self.n += 1
                return [
                    Gen(
                        json.dumps(
                            {
                                "texts": [
                                    f"variant {self.n} a",
                                    f"variant {self.n} b",
                                    "",
                                ]
                            }
                        )
                    )
                ]

        bases = [r for r in templates if r.kind == "spec"][:4]
        sink = []
        backend = Gen2()
        rows = td.generate_paraphrases(
            backend, bases, ["typos", "casual_chat"], 3, 0, sink=sink.append
        )
        assert len(rows) == 8 and len(sink) == 8  # blank strings are dropped
        assert {r["base_id"] for r in rows} == {b.id for b in bases}
        more = td.generate_paraphrases(
            Gen2(), bases, ["typos"], 3, 0, done_ids={b.id for b in bases[:2]}
        )
        assert {r["base_id"] for r in more} == {b.id for b in bases[2:]}

    def test_a_failing_call_loses_one_base_not_the_run(self, templates):
        class Flaky:
            def __init__(self):
                self.n = 0

            def generate(self, prompts, **kw):
                self.n += 1
                if self.n == 1:
                    raise RuntimeError("timeout")
                return [Gen(json.dumps({"texts": ["ok one"]}))]

        bases = [r for r in templates if r.kind == "spec"][:3]
        sink = []
        rows = td.generate_paraphrases(
            Flaky(), bases, ["typos"], 3, 0, sink=sink.append
        )
        assert len(rows) == 2
        assert any(
            s["text"] == "" for s in sink
        )  # the failure is recorded so a rerun skips it

    def test_verification_keeps_only_paraphrases_that_recover_the_gold(
        self, registry, packs, templates
    ):
        spec = next(r for r in templates if r.kind == "spec" and r.domain == "squad")
        refuse = next(
            r for r in templates if r.kind == "refuse" and r.domain == "squad"
        )
        clar = next(r for r in templates if r.kind == "clarify" and r.domain == "squad")
        bases = {b.id: b for b in (spec, refuse, clar)}
        good = json.loads(spec.reply)
        replies = [
            good,  # spec: recovered
            {"params": {"eaters": ["red"], "step_limit": 77}},  # spec: a different spec
            {"refusal": "no"},  # refuse: refused
            {"clarify": "which?"},  # refuse expected but got a question
            {"clarify": "which?"},  # clarify: asked
            good,  # clarify expected but got a spec
        ]
        rows = [
            {"base_id": spec.id, "style": "s", "text": "a"},
            {"base_id": spec.id, "style": "s", "text": "b"},
            {"base_id": refuse.id, "style": "s", "text": "c"},
            {"base_id": refuse.id, "style": "s", "text": "d"},
            {"base_id": clar.id, "style": "s", "text": "e"},
            {"base_id": clar.id, "style": "s", "text": "f"},
        ]
        tr = PolicyTranslator(Fake([]), registry, retries=0)

        class Sequenced:
            def generate(self, prompts, **kw):
                r = replies.pop(0)
                return [Gen(json.dumps(r))]

        tr.backend = Sequenced()
        out = td.verify_paraphrases(tr, packs, bases, rows)
        assert [r["verified"] for r in out] == [True, False, True, False, True, False]
        recs = td.paraphrase_records(bases, out)
        assert [r.text for r in recs] == ["a", "c", "e"]
        assert recs[0].reply == spec.reply and recs[0].source == "paraphrase"

    def test_duplicate_paraphrases_are_collapsed(self, templates):
        base = next(r for r in templates if r.kind == "spec")
        rows = [
            {"base_id": base.id, "style": "s", "text": "Same words", "verified": True},
            {"base_id": base.id, "style": "s", "text": "same words", "verified": True},
        ]
        assert len(td.paraphrase_records({base.id: base}, rows)) == 1


class TestRetryAndSelection:
    def test_retry_examples_carry_real_validator_errors_and_the_correct_reply(
        self, templates, registry
    ):
        retries = td.retry_examples(registry, templates, seed=1, n=40)
        assert len(retries) > 20
        for r in retries:
            assert r.kind == "retry" and r.text.startswith(
                "Your previous reply was rejected"
            )
            assert "Instruction:" in r.text
            filled = fill_defaults(json.loads(r.reply)["params"], td.DEFAULTS[r.domain])
            assert filled == r.gold

    def test_selection_mixes_sources_and_adds_retries(self, templates, registry):
        paras = [
            td.Record(
                f"p{i}", "squad", "spec", "paraphrase", "s", f"para text {i}", "{}"
            )
            for i in range(60)
        ]
        retries = td.retry_examples(registry, templates, seed=1, n=40)
        pool = list(templates) + paras + retries
        chosen = td.select_training(pool, 100, seed=0, paraphrase_share=0.5)
        assert len(chosen) == 100
        assert sum(r.kind == "retry" for r in chosen) == 4
        assert sum(r.source == "paraphrase" for r in chosen) == 48
        assert chosen != td.select_training(pool, 100, seed=1)
        assert [r.id for r in chosen] == [
            r.id for r in td.select_training(pool, 100, seed=0)
        ]

    def test_selection_tops_up_with_templates_when_paraphrases_run_out(self, templates):
        paras = [td.Record("p0", "squad", "spec", "paraphrase", "s", "just one", "{}")]
        chosen = td.select_training(list(templates) + paras, 50, seed=0)
        assert len(chosen) == 50 and sum(r.source == "paraphrase" for r in chosen) == 1


# ---------------------------------------------------------------------- report maths


def fake_item(i, kind, gold, domain="squad"):
    return EvalItem(f"i{i}", domain, "sealed", "t", f"text {i}", gold, kind)


class TestStep2Metrics:
    def run(self, registry, outcomes):
        """Evaluate fake items: outcomes are the replies the policy gets in order."""
        domains = with_policy_packs(build_domains(None, 1, 1))
        items = [
            fake_item(0, "spec", GOOD),
            fake_item(1, "spec", GOOD),
            fake_item(2, "refuse", None),
            fake_item(3, "refuse", None),
            fake_item(4, "clarify", None),
            fake_item(5, "clarify", None),
        ]
        tr = PolicyTranslator(Fake(outcomes), registry, retries=0)
        return evaluate_forward(
            lambda it: tr.translate(it.text, domains[it.domain].pack),
            items, domains, registry, execute_tasks=False,
        )  # fmt: skip

    def test_clarify_is_neither_exact_nor_refusal(self, registry):
        good = {"params": {"eaters": ["red"], "caution": "high", "step_limit": 120}}
        records = self.run(
            registry,
            [
                good,
                {"clarify": "which?"},
                {"refusal": "no"},
                {"clarify": "?"},
                {"clarify": "q"},
                {"refusal": "r"},
            ],
        )
        by = {r["id"]: r for r in records}
        assert by["i0"]["exact"] and not by["i0"]["clarified"]
        assert (
            not by["i1"]["exact"] and by["i1"]["clarified"] and not by["i1"]["refused"]
        )
        assert by["i3"]["clarified"] and not by["i3"]["refused"]
        m = forward_metrics(records)
        assert m["exact"]["sealed"]["n"] == 2 and m["exact"]["sealed"]["exact"] == 0.5
        # refusal: gold refuse = i2, i3; predicted refuse = i2 (tp), i5 (a clarify item: fp)
        assert (m["refusal"]["tp"], m["refusal"]["fp"], m["refusal"]["fn"]) == (1, 1, 1)
        c = m["clarify"]
        assert (c["asked"], c["on_spec"], c["gold"], c["tp"]) == (3, 1, 2, 1)
        assert (
            c["on_spec_rate"] == 0.5
            and c["recall"] == 0.5
            and c["precision"] == pytest.approx(1 / 3)
        )

    def test_clarify_scores_handle_no_questions(self):
        s = clarify_scores([{"gold": None, "kind": "refuse", "clarified": False}])
        assert s["asked"] == 0 and s["rate"] == 0

    def test_old_records_without_kind_still_work(self):
        recs = [
            {"split": "heldout", "domain": "squad", "gold": GOOD, "exact": True, "fields": {"caution": True},
             "refused": False, "status": "ok", "attempts": 1, "latency_s": 0.1},
            {"split": "heldout", "domain": "squad", "gold": None, "refused": True, "status": "refused",
             "attempts": 1, "latency_s": 0.1},
        ]  # fmt: skip
        m = forward_metrics(recs)
        assert m["refusal"]["recall"] == 1.0 and m["clarify"]["asked"] == 0

    def test_verdict_reads_the_sealed_split(self):
        fwd = {
            "exact": {"sealed": {"n": 100, "exact": 0.96, "exact_ci": [0.9, 0.99]}},
            "task": {},
        }
        v = verdict(fwd, None, 0.8, 3.0, False, split="sealed")
        assert v["label"] == "PASS" and "sealed" in v["rows"][0]["criterion"]
        fwd["exact"]["sealed"]["exact"] = 0.8
        assert (
            verdict(fwd, None, 0.8, 3.0, False, split="sealed")["label"]
            == "KILL criterion hit"
        )

    def test_execution_cache_shares_runs_of_equal_params(self, registry):
        calls = []

        class Exec:
            def __call__(self, params):
                calls.append(1)
                return {
                    "score": 1.0,
                    "food": 1.0,
                    "survivors": 1.0,
                    "steps": 1.0,
                    "per_seed": [1.0, 2.0],
                }

        domains = with_policy_packs(build_domains(None, 1, 1))
        domains["squad"].executor = Exec()
        item = fake_item(0, "spec", GOOD)
        reply = {"params": {"eaters": ["red"], "caution": "high", "step_limit": 120}}
        tr = PolicyTranslator(Fake([reply]), registry)
        cache = {}
        recs = evaluate_forward(
            lambda it: tr.translate(it.text, domains[it.domain].pack),
            [item],
            domains,
            registry,
            cache=cache,
        )
        assert len(calls) == 1  # gold run reused for the identical NL spec
        assert recs[0]["nl_per_seed"] == recs[0]["gold_per_seed"] == [1.0, 2.0]


class TestStep2Glue:
    def test_lora_pack_is_just_the_task_name_plus_code_side_policy(self):
        for domains in (
            build_domains(None, 1, 1),
            with_policy_packs(build_domains(None, 1, 1)),
        ):
            pack = lora_pack(domains["squad"])
            assert pack.defaults == SQUAD_DEFAULTS and pack.questions["eaters"]
        pack = lora_pack(domains["squad"])
        assert pack.system == SQUAD_MISSION and pack.examples == ()
        assert pack.defaults == SQUAD_DEFAULTS and pack.questions["eaters"]

    def test_resolve_runs_finds_sibling_artifacts(self, tmp_path):
        for job in ("r7a2-train-a", "r7a2-train-b", "other"):
            (tmp_path / "runs" / job / "artifacts").mkdir(parents=True)
        found = resolve_runs(["runs:r7a2-train-*"], runs_root=tmp_path / "runs")
        assert [p.parent.name for p in found] == ["r7a2-train-a", "r7a2-train-b"]
        assert all(p.name == "artifacts" for p in found)

    def test_resolve_runs_from_the_runner_layout(self, tmp_path, monkeypatch):
        run_dir = tmp_path / "runs" / "me" / "artifacts"
        run_dir.mkdir(parents=True)
        (tmp_path / "runs" / "r7a2-x" / "artifacts").mkdir(parents=True)
        monkeypatch.setenv("GCA_RUN_DIR", str(run_dir))
        assert [p.parent.name for p in resolve_runs(["runs:r7a2-*"])] == ["r7a2-x"]

    def test_select_best_run_prefers_score_then_fewer_steps(self, tmp_path):
        def write(name, score, steps):
            d = tmp_path / name
            d.mkdir()
            (d / "dev.json").write_text(
                json.dumps(
                    {"name": name, "config": {"steps": steps}, "dev": {"score": score}}
                ),
                encoding="utf-8",
            )
            return d / "dev.json"

        files = [write("a", 0.9, 300), write("b", 0.95, 900), write("c", 0.95, 300)]
        assert select_best_run(files)["name"] == "c"
        with pytest.raises(FileNotFoundError):
            select_best_run([])
        assert find_file([tmp_path / "a"], "dev.json").name == "dev.json"
        with pytest.raises(FileNotFoundError):
            find_file([tmp_path / "a"], "nope.json")

    def test_round_trip_rows_compare_the_reconstruction_with_the_forward_spec(
        self, registry
    ):
        from genetic_coil_arena.experiments.r7a_pipeline import execute

        domains = with_policy_packs(build_domains(None, 1, 1))

        class Exec:
            def __call__(self, params):
                return {
                    "score": 3.0,
                    "food": 1.0,
                    "survivors": 1.0,
                    "steps": 1.0,
                    "per_seed": [3.0],
                }

        domains["squad"].executor = Exec()
        spec = registry.make(SQUAD_MISSION, GOOD)
        view = execute(domains["squad"], registry, spec, "t").view
        records = [{"id": "i0", "domain": "squad", "split": "sealed", "nl_view": view}]
        summaries = {
            "i0": {
                "summary": "The red snake was ordered to forage at caution level high with a step limit of 120 and no food target; the squad finished with mission score 3.0.",
                "latency_s": 0.5,
            }
        }
        tr = PolicyTranslator(
            Fake(
                [{"params": {"eaters": ["red"], "caution": "high", "step_limit": 120}}]
            ),
            registry,
        )
        rows = reconstruct_rows(tr.translate, records, summaries, domains, registry)
        assert (
            rows[0]["round_trip"]
            and rows[0]["fidelity"] == 1.0
            and rows[0]["all_facts"]
        )
        wrong = PolicyTranslator(Fake([{"params": {"eaters": ["blue"]}}]), registry)
        assert not reconstruct_rows(
            wrong.translate, records, summaries, domains, registry
        )[0]["round_trip"]
        missing = reconstruct_rows(tr.translate, records, {}, domains, registry)
        assert not missing[0]["round_trip"] and missing[0]["recon_status"] is None


# ---------------------------------------------------------------------- report and script


def _system(name, exact, **extra):
    n = 145
    fwd = {
        "n_items": 174,
        "exact": {
            "sealed": {
                "n": n,
                "exact": exact,
                "exact_ci": [exact - 0.05, min(1, exact + 0.03)],
                "field_accuracy": {"caution": exact},
            },
            "sealed/squad": {
                "n": 73,
                "exact": exact,
                "exact_ci": [0.8, 1.0],
                "field_accuracy": {},
            },
            "sealed/plant": {
                "n": 72,
                "exact": exact,
                "exact_ci": [0.8, 1.0],
                "field_accuracy": {},
            },
        },
        "task": {
            "all/squad": {
                "n_items": 73,
                "n_seeds": 8,
                "nl_mean": 6.0,
                "nl_ci": 0.5,
                "gold_mean": 6.1,
                "gold_ci": 0.5,
                "diff_mean": -0.1,
                "diff_ci": 0.2,
                "within_ci": True,
                "failed_translations": 1,
            }
        },
        "refusal": {
            "tp": 18,
            "fp": 2,
            "fn": 1,
            "tn": 0,
            "precision": 0.9,
            "recall": 0.95,
        },
        "clarify": {
            "asked": 9,
            "rate": 0.05,
            "on_spec": 1,
            "on_spec_rate": 0.01,
            "gold": 10,
            "tp": 8,
            "precision": 0.8,
            "recall": 0.8,
        },
        "status_counts": {"ok": 140},
    }
    s = {"name": name, "forward": fwd, "records": [
        {"id": "x0", "kind": "spec", "exact": False, "refused": True, "clarified": False, "status": "refused",
         "text": "t", "gold": GOOD, "pred": None, "reason": "r"}],
         "latency": {"cold_first_call_s": 1.0, "warm_item_p50_s": 0.5, "warm_item_p95_s": 0.9, "mean_attempts": 1.0},
         "vram": {"model_smi_mib": 3000.0, "weights_mib": 3300.0, "peak_allocated_mib": 3500.0},
         "reverse": {"all": {"n": 140, "round_trip": 0.95, "round_trip_ci": [0.9, 0.98], "field_fidelity": 1.0, "all_facts_rate": 1.0}}}  # fmt: skip
    s.update(extra)
    return s


class TestReportRendering:
    def test_verdict_and_render_for_a_passing_distilled_model(self):
        lora = _system("lora", 0.96, config={"rank": 8, "n": 1000}, dev={"score": 0.9})
        o14 = _system("o14", 0.80)
        report = build_report(
            lora, {"systems": {"o14": o14, "regex": _system("regex", 0.2)}}, [], None
        )
        assert report["systems"]["lora"]["verdict"]["label"] == "PASS"
        assert report["systems"]["o14"]["verdict"]["label"] == "KILL criterion hit"
        md = render(report)
        assert (
            "r7a_sealed_test" in md
            and "Qwen3-1.7B + LoRA" in md
            and "Clarify rate" in md
        )
        assert report["sealed_sha256"] in md

    def test_memory_over_four_gb_fails_the_distilled_target(self):
        lora = _system("lora", 0.97, config={"rank": 8, "n": 1000})
        lora["vram"]["peak_allocated_mib"] = 4500.0
        report = build_report(lora, None, [], None)
        v = report["systems"]["lora"]["verdict"]
        assert v["label"].startswith("NOT PASSED")
        assert any(
            "beside the specialists" in r["criterion"] and r["pass"] is False
            for r in v["rows"]
        )


@pytest.mark.slow
def test_script_pipeline_runs_end_to_end_with_scripted_models(tmp_path):
    env_out = tmp_path / "art"
    base = [
        sys.executable,
        str(ROOT / "scripts" / "r7a_step2.py"),
        "--out",
        str(env_out),
    ]
    env = {
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONUTF8": "1",
        "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", ""),
        "PATH": __import__("os").environ.get("PATH", ""),
    }
    subprocess.run(
        [*base, "para-gen", "--fake", "--bases", "12"],
        check=True,
        env=env,
        cwd=tmp_path,
    )
    assert (env_out / "para_raw.jsonl").exists()
    subprocess.run(
        [*base, "para-verify", "--fake", "--raw", str(env_out), "--templates", "60", "--retries", "10"],
        check=True, env=env, cwd=tmp_path,
    )  # fmt: skip
    stats = json.loads((env_out / "pool_stats.json").read_text(encoding="utf-8"))
    assert stats["n_pool"] > 100 and stats["kinds"]["retry"] > 0
    pool = [
        td.Record.from_json(x)
        for x in (env_out / "pool.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    td.assert_no_leak(pool, [i.text for i in r7a_sealed.load_sealed()])
