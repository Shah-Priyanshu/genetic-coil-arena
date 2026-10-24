"""The domain-agnostic translator: forward (with a fake backend), reverse and metrics."""

import json
import subprocess
import sys
from dataclasses import dataclass

import pytest

from genetic_coil_arena.protocol import (
    Envelope,
    SpecDefinition,
    SpecRegistry,
    SpecValidationError,
)
from genetic_coil_arena.translator import (
    ERROR,
    INVALID,
    OK,
    REFUSED,
    Example,
    Fact,
    ReverseTranslator,
    SummaryPack,
    TranslationPack,
    Translator,
    canonical_params,
    envelope_header,
    exact_match,
    fact_fidelity,
    field_matches,
    percentile,
    refusal_scores,
    reply_schema,
    trace_view,
    wilson,
)

PARAMS = {
    "type": "object",
    "required": ["speed", "colours"],
    "properties": {
        "speed": {"type": "integer", "minimum": 1, "maximum": 10},
        "colours": {
            "type": "array",
            "items": {"type": "string", "enum": ["red", "blue"]},
            "minItems": 1,
            "uniqueItems": True,
        },
    },
    "additionalProperties": False,
}


@dataclass
class Gen:
    text: str
    latency_s: float = 0.1
    n_prompt_tokens: int = 10


class FakeBackend:
    """Replies from a script; records every call's arguments."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, prompts, adapter_id=None, **kwargs):
        self.calls.append((list(prompts), kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return [Gen(reply if isinstance(reply, str) else json.dumps(reply))]


@pytest.fixture
def registry():
    reg = SpecRegistry()
    reg.register(SpecDefinition("toy.go", 1, PARAMS, "Go somewhere"))
    return reg


PACK = TranslationPack(
    "toy.go",
    "system prompt",
    (
        Example("go fast with red", {"speed": 9, "colours": ["red"]}),
        Example("bake a cake", None, "not about going"),
    ),
)


def make(registry, replies, **kw):
    backend = FakeBackend(replies)
    return Translator(backend, registry, **kw), backend


class TestForward:
    def test_valid_reply_becomes_a_validated_goalspec(self, registry):
        tr, backend = make(
            registry, [{"refusal": "", "params": {"speed": 5, "colours": ["blue"]}}]
        )
        res = tr.translate("blue at speed five", PACK)
        assert res.ok and res.status == OK and res.attempts == 1
        assert res.spec.params == {"speed": 5, "colours": ["blue"]}
        assert registry.is_valid(res.spec)
        assert res.latency_s == pytest.approx(0.1) and res.n_prompt_tokens == 10

    def test_request_carries_schema_system_and_examples(self, registry):
        tr, backend = make(
            registry, [{"refusal": "", "params": {"speed": 1, "colours": ["red"]}}]
        )
        tr.translate("slow red", PACK)
        prompts, kw = backend.calls[0]
        assert prompts == ["slow red"] and kw["system"] == "system prompt"
        assert kw["schema"] == reply_schema(registry, "toy.go")
        assert kw["schema"]["properties"]["params"] == PARAMS
        assert kw["examples"][0] == (
            "go fast with red",
            json.dumps({"refusal": "", "params": {"speed": 9, "colours": ["red"]}}),
        )
        assert json.loads(kw["examples"][1][1]) == {"refusal": "not about going"}

    def test_schema_is_a_copy_of_the_registry_schema(self, registry):
        schema = reply_schema(registry, "toy.go")
        schema["properties"]["params"]["required"].append("x")
        assert registry.get("toy.go").params_schema["required"] == ["speed", "colours"]

    def test_invalid_reply_is_retried_once_with_the_error(self, registry):
        bad = {"refusal": "", "params": {"speed": 99, "colours": ["red"]}}
        good = {"refusal": "", "params": {"speed": 10, "colours": ["red"]}}
        tr, backend = make(registry, [bad, good])
        res = tr.translate("max speed red", PACK)
        assert res.ok and res.attempts == 2
        assert len(res.call_latencies) == 2 and res.latency_s == pytest.approx(0.2)
        prompt, kw = backend.calls[1]
        assert "rejected" in prompt[0] and "max speed red" in prompt[0]
        assert "/params" in prompt[0]  # the validation error is quoted
        assert kw["examples"][-1] == ("max speed red", json.dumps(bad))

    def test_two_invalid_replies_are_a_typed_failure(self, registry):
        bad = {"refusal": "", "params": {"speed": 99, "colours": ["red"]}}
        tr, backend = make(registry, [bad, bad])
        res = tr.translate("x", PACK)
        assert res.status == INVALID and not res.ok and res.spec is None
        assert res.attempts == 2 and res.errors and len(backend.calls) == 2

    def test_retries_can_be_disabled(self, registry):
        tr, backend = make(registry, ["not json"], retries=0)
        res = tr.translate("x", PACK)
        assert res.status == INVALID and len(backend.calls) == 1

    def test_non_json_reply_is_retried(self, registry):
        tr, _ = make(
            registry,
            ["oops", {"refusal": "", "params": {"speed": 2, "colours": ["red"]}}],
        )
        assert tr.translate("x", PACK).ok

    def test_refusal_is_final_and_not_retried(self, registry):
        tr, backend = make(registry, [{"refusal": "no such colour"}])
        res = tr.translate("send the pink one", PACK)
        assert res.status == REFUSED and res.reason == "no such colour"
        assert res.spec is None and len(backend.calls) == 1

    def test_missing_params_without_refusal_is_invalid(self, registry):
        tr, _ = make(registry, [{"refusal": ""}, {"refusal": ""}])
        res = tr.translate("x", PACK)
        assert res.status == INVALID and "params" in res.reason

    def test_unknown_field_is_rejected_by_the_registry(self, registry):
        extra = {"refusal": "", "params": {"speed": 2, "colours": ["red"], "bonus": 1}}
        tr, _ = make(registry, [extra, extra])
        assert tr.translate("x", PACK).status == INVALID

    def test_backend_failure_is_a_typed_error(self, registry):
        tr, _ = make(registry, [RuntimeError("server down")])
        res = tr.translate("x", PACK)
        assert res.status == ERROR and "server down" in res.reason

    def test_priority_and_deadline_reach_the_spec(self, registry):
        tr, _ = make(
            registry, [{"refusal": "", "params": {"speed": 2, "colours": ["red"]}}]
        )
        res = tr.translate("x", PACK, priority=3, deadline=9)
        assert res.spec.priority == 3 and res.spec.deadline == 9

    def test_keep_alive_is_passed_on(self, registry):
        tr, backend = make(
            registry,
            [{"refusal": "", "params": {"speed": 2, "colours": ["red"]}}],
            keep_alive=0,
        )
        tr.translate("x", PACK)
        assert backend.calls[0][1]["keep_alive"] == 0

    def test_translator_never_returns_an_unvalidated_spec(self, registry):
        # A reply for a spec name the registry lacks cannot become a GoalSpec
        tr, _ = make(
            registry, [{"refusal": "", "params": {"speed": 2, "colours": ["red"]}}]
        )
        with pytest.raises(KeyError):
            tr.translate("x", TranslationPack("toy.missing", "s"))
        with pytest.raises(SpecValidationError):
            registry.make("toy.go", {"speed": 0, "colours": ["red"]})


class TestReverse:
    def view(self, registry):
        goal = registry.make("toy.go", {"speed": 4, "colours": ["red"]})
        env = Envelope(
            sender="centre",
            recipients=("worker",),
            payload_type="json",
            payload={"secret": "payload"},
            clock=3,
        )
        return goal, trace_view(goal, [env], {"state": "completed", "score": 1.5})

    def test_headers_have_no_payload(self, registry):
        goal, view = self.view(registry)
        assert view.headers[0] == {
            "sender": "centre",
            "recipients": ["worker"],
            "payload_type": "json",
            "codec": None,
            "clock": 3,
        }
        assert "secret" not in view.render()
        assert json.loads(view.render())["goal"]["params"] == goal.params
        assert envelope_header(
            Envelope(sender="a", recipients=("b",), payload_type="json", payload=1)
        )

    def test_summary_then_reconstruction_round_trip(self, registry):
        goal, view = self.view(registry)
        replies = [
            "Red went at speed 4; score 1.5.",
            {"refusal": "", "params": {"speed": 4, "colours": ["red"]}},
        ]
        tr, backend = make(registry, replies)
        trip = ReverseTranslator(tr).round_trip(
            view, SummaryPack("sum system", max_new_tokens=50), PACK
        )
        assert trip.summary.text == "Red went at speed 4; score 1.5."
        prompts, kw = backend.calls[0]
        assert prompts == [view.render()] and kw["system"] == "sum system"
        assert kw["max_new_tokens"] == 50 and "schema" not in kw
        assert backend.calls[1][0] == [
            "Red went at speed 4; score 1.5."
        ]  # the summary is re-translated
        schema = registry.get("toy.go").params_schema
        assert trip.matches(goal, schema)

    def test_reconstruction_that_differs_does_not_match(self, registry):
        goal, view = self.view(registry)
        tr, _ = make(
            registry,
            ["summary", {"refusal": "", "params": {"speed": 5, "colours": ["red"]}}],
        )
        trip = ReverseTranslator(tr).round_trip(view, SummaryPack("s"), PACK)
        assert not trip.matches(goal)

    def test_refused_reconstruction_does_not_match(self, registry):
        goal, view = self.view(registry)
        tr, _ = make(registry, ["summary", {"refusal": "not an order"}])
        trip = ReverseTranslator(tr).round_trip(view, SummaryPack("s"), PACK)
        assert trip.reconstructed.status == REFUSED and not trip.matches(goal)

    def test_summary_backend_failure_is_reported(self, registry):
        _, view = self.view(registry)
        tr, _ = make(registry, [RuntimeError("down")])
        trip = ReverseTranslator(tr).round_trip(view, SummaryPack("s"), PACK)
        assert trip.summary.error and trip.reconstructed is None


class TestMetrics:
    def test_canonical_params_sorts_only_unique_arrays(self):
        a = canonical_params(
            {"colours": ["red", "blue"], "order": [2, 1]},
            {"properties": {"colours": {"uniqueItems": True}, "order": {}}},
        )
        assert a == {"colours": ["blue", "red"], "order": [2, 1]}

    def test_exact_match_and_field_matches(self):
        schema = PARAMS
        gold = {"speed": 3, "colours": ["red", "blue"]}
        assert exact_match({"speed": 3, "colours": ["blue", "red"]}, gold, schema)
        assert not exact_match({"speed": 4, "colours": ["blue", "red"]}, gold, schema)
        assert not exact_match(None, gold, schema)
        assert field_matches(
            {"speed": 4, "colours": ["blue", "red"]}, gold, schema
        ) == {
            "speed": False,
            "colours": True,
        }
        assert field_matches(None, gold, schema) == {"speed": False, "colours": False}
        assert (
            field_matches({"colours": ["red", "blue"]}, gold, schema)["speed"] is False
        )

    def test_wilson_interval(self):
        p, lo, hi = wilson(9, 10)
        assert (
            p == 0.9
            and lo == pytest.approx(0.5958, abs=1e-3)
            and hi == pytest.approx(0.9821, abs=1e-3)
        )
        assert wilson(0, 10)[1] == 0.0 and wilson(10, 10)[2] == 1.0
        assert all(x != x for x in wilson(0, 0))

    def test_percentile(self):
        assert percentile([1, 2, 3, 4], 50) == 2.5
        assert percentile([5], 95) == 5
        assert percentile([0, 10], 95) == pytest.approx(9.5)
        assert percentile([], 50) != percentile([], 50)  # NaN

    def test_refusal_precision_and_recall(self):
        r = refusal_scores(
            [(True, True), (True, False), (False, True), (False, False), (False, False)]
        )
        assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 2)
        assert r["precision"] == 0.5 and r["recall"] == 0.5
        none = refusal_scores([(False, False)])
        assert (
            none["precision"] != none["precision"] and none["recall"] != none["recall"]
        )

    def test_fact_fidelity(self):
        facts = [
            Fact("speed", (r"\bspeed 4\b",)),
            Fact("colours", (r"(?s)^(?=.*\bred\b)",), (r"\bblue\b",)),
        ]
        ok = fact_fidelity("Red went at speed 4.", facts)
        assert ok["fidelity"] == 1.0 and ok["all"]
        bad = fact_fidelity("Red and blue went at speed 5.", facts)
        assert bad["facts"] == {"speed": False, "colours": False}
        assert bad["fidelity"] == 0.0 and not bad["all"]
        assert fact_fidelity("x", [])["all"] is False


def test_package_does_not_load_domain_code():
    code = (
        "import sys, genetic_coil_arena.translator; "
        "bad = [m for m in sys.modules if m.startswith(('genetic_coil_arena.envs', "
        "'genetic_coil_arena.snake_core', 'genetic_coil_arena.evo', "
        "'genetic_coil_arena.analysis', 'numba'))]; print(bad); assert not bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
