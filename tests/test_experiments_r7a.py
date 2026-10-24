"""R7a: evaluation-set integrity, domain executors, the command-centre path, report maths.

No Ollama and no GPU: the translator is exercised with scripted fake backends (the live
check is marked ``integration`` and skipped when the server is unreachable).
"""

import json
import re
from dataclasses import dataclass

import numpy as np
import pytest

from genetic_coil_arena.backends.ollama import OllamaBackend, OllamaError
from genetic_coil_arena.envs.plant.tep import NODES, synthetic_tep
from genetic_coil_arena.experiments.r7a_domains import (
    COLOURS,
    PLANT_DIAGNOSE,
    SQUAD_MISSION,
    PlantDiagnoseExecutor,
    SquadMissionExecutor,
    build_domains,
    build_registry,
    keyword_plant,
    keyword_squad,
    outcome_of,
    plant_facts,
    squad_facts,
)
from genetic_coil_arena.experiments.r7a_evalset import (
    SPLITS,
    build_evalset,
    few_shot_texts,
    words,
)
from genetic_coil_arena.experiments.r7a_pipeline import execute, keyword_translator
from genetic_coil_arena.experiments.r7a_report import (
    evaluate_forward,
    evaluate_reverse,
    forward_metrics,
    reverse_metrics,
    select_items,
    verdict,
)
from genetic_coil_arena.protocol import SpecValidationError
from genetic_coil_arena.translator import (
    OK,
    ReverseTranslator,
    TranslationResult,
    Translator,
    fact_fidelity,
)

SQUAD_GOAL = {
    "eaters": ["red", "green"],
    "caution": "high",
    "step_limit": 60,
    "food_target": 0,
}
PLANT_GOAL = {
    "byte_budget": 16,
    "target_far": 0.01,
    "active_nodes": ["reactor", "stripper"],
}


@pytest.fixture(scope="module")
def registry():
    return build_registry()


@pytest.fixture(scope="module")
def items():
    return build_evalset()


@pytest.fixture(scope="module")
def domains():
    return build_domains(synthetic_tep(0), squad_seeds=2, plant_seeds=2)


# ------------------------------------------------------------------ evaluation set
class TestEvalSet:
    def test_size_and_invalid_share(self, items):
        assert len(items) >= 150
        share = sum(i.invalid for i in items) / len(items)
        assert 0.08 <= share <= 0.12

    def test_every_gold_spec_validates(self, items, registry):
        for item in items:
            if item.gold is not None:
                assert registry.make(item.spec_name, item.gold).params == item.gold

    def test_ids_and_texts_are_unique(self, items):
        assert len({i.id for i in items}) == len(items)
        assert len({i.text for i in items}) == len(items)

    def test_every_domain_and_split_is_covered(self, items):
        for domain in ("squad", "plant"):
            for split in SPLITS:
                sub = [i for i in items if i.domain == domain and i.split == split]
                assert sum(not i.invalid for i in sub) >= 30
                assert sum(i.invalid for i in sub) >= 3

    def test_heldout_styles_are_not_the_indist_styles(self, items):
        indist = {i.style for i in items if i.split == "indist" and not i.invalid}
        heldout = {i.style for i in items if i.split == "heldout" and not i.invalid}
        assert not indist & heldout

    def test_few_shot_instructions_are_not_in_the_set(self, items):
        assert not few_shot_texts() & {i.text for i in items}

    def test_deterministic_for_a_seed(self):
        assert build_evalset(3) == build_evalset(3)
        assert build_evalset(3) != build_evalset(4)

    def test_invalid_requests_really_are_unsatisfiable(self, items, registry):
        # Whatever the keyword parser makes of them must not be a gold-equal valid spec
        for item in (i for i in items if i.invalid):
            assert item.gold is None

    def test_no_item_leaks_its_label_format(self, items):
        for item in items:
            assert not re.search(
                r"step_limit|food_target|byte_budget|target_far|active_nodes", item.text
            )

    def test_defaults_appear_when_a_field_is_not_mentioned(self, items):
        squad = [i for i in items if i.domain == "squad" and i.gold]
        assert any(i.gold["food_target"] == 0 for i in squad)
        assert any(i.gold["step_limit"] == 300 for i in squad)

    def test_words(self):
        assert [words(n) for n in (0, 8, 16, 51, 120, 250)] == [
            "zero", "eight", "sixteen", "fifty-one", "one hundred twenty", "two hundred fifty",
        ]  # fmt: skip

    def test_select_items_is_stratified(self, items):
        sub = select_items(items, 24, seed=1)
        assert 20 <= len(sub) <= 40
        assert {(i.domain, i.split, i.invalid) for i in sub} == {
            (i.domain, i.split, i.invalid) for i in items
        }
        assert select_items(items, 10_000) == list(items)


# ------------------------------------------------------------------ specs and parsers
class TestSpecs:
    def test_both_specs_registered_with_result_schemas(self, registry):
        assert registry.names() == [PLANT_DIAGNOSE, SQUAD_MISSION]
        assert (
            registry.get(SQUAD_MISSION).result_schema
            and registry.get(PLANT_DIAGNOSE).result_schema
        )

    @pytest.mark.parametrize(
        "name, bad",
        [
            (SQUAD_MISSION, {**SQUAD_GOAL, "step_limit": 10}),
            (SQUAD_MISSION, {**SQUAD_GOAL, "food_target": 500}),
            (SQUAD_MISSION, {**SQUAD_GOAL, "eaters": ["yellow"]}),
            (SQUAD_MISSION, {**SQUAD_GOAL, "eaters": []}),
            (SQUAD_MISSION, {**SQUAD_GOAL, "caution": "reckless"}),
            (PLANT_DIAGNOSE, {**PLANT_GOAL, "byte_budget": 500}),
            (PLANT_DIAGNOSE, {**PLANT_GOAL, "target_far": 0.1}),
            (PLANT_DIAGNOSE, {**PLANT_GOAL, "active_nodes": ["boiler"]}),
            (PLANT_DIAGNOSE, {**PLANT_GOAL, "active_nodes": ["reactor", "reactor"]}),
        ],
    )
    def test_out_of_range_params_do_not_validate(self, registry, name, bad):
        with pytest.raises(SpecValidationError):
            registry.make(name, bad)

    def test_keyword_squad(self):
        got = keyword_squad(
            "Red and blue forage, play it safe, 120 steps, target 15 food."
        )
        assert got == {
            "eaters": ["red", "blue"],
            "caution": "high",
            "step_limit": 120,
            "food_target": 15,
        }
        assert keyword_squad("everyone except green, be aggressive")["eaters"] == [
            "red",
            "blue",
        ]
        assert keyword_squad("Tell me a joke") is None
        assert keyword_squad("the whole squad eats")["eaters"] == list(COLOURS)

    def test_keyword_plant(self):
        got = keyword_plant("8 bytes, 2% false alarms, reactor and stripper only")
        assert got == {
            "byte_budget": 8,
            "target_far": 0.02,
            "active_nodes": ["reactor", "stripper"],
        }
        assert keyword_plant("all nodes but the reactor")["active_nodes"] == list(
            NODES[1:]
        )
        assert keyword_plant("what is the weather") is None

    def test_keyword_translator_goes_through_the_registry(self, registry, domains):
        @dataclass
        class It:
            domain: str
            text: str

        tr = keyword_translator(registry, domains)
        assert tr(It("squad", "red forages for 120 steps")).status == OK
        assert (
            tr(It("squad", "red forages for 10 steps")).status == "invalid"
        )  # below the range
        assert tr(It("squad", "order a pizza")).status == "refused"


# ------------------------------------------------------------------ executors
class TestExecutors:
    def test_squad_result_is_valid_and_cached(self, registry):
        ex = SquadMissionExecutor(n_seeds=3)
        first = ex(SQUAD_GOAL)
        assert registry.get(SQUAD_MISSION).result_schema and not registry.result_errors(
            registry.make(SQUAD_MISSION, SQUAD_GOAL), first
        )
        assert len(first["per_seed"]) == 3 and first["steps"] <= 60
        assert (
            ex({**SQUAD_GOAL, "eaters": ["green", "red"]}) == first
        )  # order-free cache key
        assert ex.runs == 1

    def test_squad_params_are_honoured(self):
        ex = SquadMissionExecutor(n_seeds=4)
        short = ex({**SQUAD_GOAL, "step_limit": 50})
        long = ex({**SQUAD_GOAL, "step_limit": 200})
        assert short["steps"] <= 50 < long["steps"]
        assert long["food"] >= short["food"]
        one = ex({**SQUAD_GOAL, "eaters": ["red"], "step_limit": 200})
        assert one["food"] < long["food"]  # fewer foragers eat less
        capped = ex({**SQUAD_GOAL, "step_limit": 200, "food_target": 2})
        assert capped["food"] <= 2 and capped["steps"] <= long["steps"]

    def test_squad_is_deterministic(self):
        a, b = SquadMissionExecutor(3), SquadMissionExecutor(3)
        assert a(SQUAD_GOAL) == b(SQUAD_GOAL)

    def test_plant_result_is_valid_and_honours_the_budget_and_nodes(self, registry):
        ex = PlantDiagnoseExecutor(synthetic_tep(0), n_seeds=2)
        res = ex(PLANT_GOAL)
        assert not registry.result_errors(
            registry.make(PLANT_DIAGNOSE, PLANT_GOAL), res
        )
        assert res["bytes_per_node"] == pytest.approx(8.0)  # 16 B from 2 of 4 nodes
        full = ex({**PLANT_GOAL, "active_nodes": list(NODES), "byte_budget": 8})
        assert full["bytes_per_node"] == pytest.approx(8.0)
        assert len(res["per_seed"]) == 2 and all(0 <= s <= 1 for s in res["per_seed"])
        assert ex({**PLANT_GOAL, "active_nodes": ["stripper", "reactor"]}) == res
        assert ex.runs == 2


# ------------------------------------------------------------------ command centre path
class TestPipeline:
    def test_squad_goal_runs_through_the_centre(self, registry, domains):
        spec = registry.make(SQUAD_MISSION, SQUAD_GOAL)
        run = execute(domains["squad"], registry, spec, "red and green forage")
        assert run.state == "completed" and run.n_records >= 4
        assert run.view.goal.params == SQUAD_GOAL
        senders = [h["sender"] for h in run.view.headers]
        assert "command-centre" in senders and "squad-lead" in senders
        assert run.view.outcome["state"] == "completed" and "score" in run.view.outcome
        assert "per_seed" not in run.view.outcome
        assert len(run.per_seed) == 2

    def test_plant_goal_runs_through_the_centre(self, registry, domains):
        spec = registry.make(PLANT_DIAGNOSE, PLANT_GOAL)
        run = execute(domains["plant"], registry, spec)
        assert run.state == "completed"
        assert {h["sender"] for h in run.view.headers} >= {
            "command-centre",
            "plant-gateway",
        }

    def test_same_goal_same_score(self, registry, domains):
        spec = registry.make(SQUAD_MISSION, SQUAD_GOAL)
        a = execute(domains["squad"], registry, spec)
        b = execute(
            domains["squad"], registry, registry.make(SQUAD_MISSION, SQUAD_GOAL)
        )
        assert a.per_seed == b.per_seed

    def test_a_failing_executor_fails_the_task(self, registry, domains):
        import copy

        broken = copy.copy(domains["squad"])
        broken.executor = lambda params: {"nope": 1}
        run = execute(broken, registry, registry.make(SQUAD_MISSION, SQUAD_GOAL))
        assert run.state != "completed" and run.view is None

    def test_outcome_rounding(self):
        assert outcome_of("squad", {"score": 7.625, "food": 7.125, "survivors": 1.0, "steps": 98.1, "per_seed": [1]}) == {
            "state": "completed", "score": 7.6, "food": 7.1, "survivors": 1.0, "steps": 98.1,
        }  # fmt: skip


# ------------------------------------------------------------------ facts
class TestFacts:
    def view(self, registry, domains, domain, params):
        spec = registry.make(domains[domain].spec_name, params)
        return execute(domains[domain], registry, spec).view

    def test_squad_summary_with_every_fact_passes(self, registry, domains):
        view = self.view(registry, domains, "squad", SQUAD_GOAL)
        score = view.outcome["score"]
        text = f"The red and green snakes were ordered to forage at caution level high with a step limit of 60 and no food target; mission score {score}."
        assert fact_fidelity(text, squad_facts(view))["all"]

    def test_squad_contradictions_and_omissions_fail(self, registry, domains):
        view = self.view(registry, domains, "squad", SQUAD_GOAL)
        facts = squad_facts(view)
        score = view.outcome["score"]
        wrong_colour = f"The red, green and blue snakes foraged, caution level high, step limit 60, no food target, score {score}."
        res = fact_fidelity(wrong_colour, facts)
        assert not res["facts"]["eaters"] and res["facts"]["step_limit"]
        missing = "The red and green snakes foraged with caution level low, step limit 61, food target 5."
        res = fact_fidelity(missing, facts)
        assert not any(
            res["facts"][k] for k in ("caution", "step_limit", "food_target", "score")
        )

    def test_plant_summary_facts(self, registry, domains):
        view = self.view(registry, domains, "plant", PLANT_GOAL)
        score = view.outcome["score"]
        good = f"Ran with 16 bytes per message and 1% false alarms on the reactor and stripper nodes, macro-F1 {score}."
        assert fact_fidelity(good, plant_facts(view))["all"]
        bad = f"Ran with 160 bytes per message and 0.1% false alarms on the reactor, separator and stripper, macro-F1 {score}."
        facts = fact_fidelity(bad, plant_facts(view))["facts"]
        assert (
            not facts["byte_budget"]
            and not facts["target_far"]
            and not facts["active_nodes"]
        )
        assert facts["score"]

    def test_compressor_alias(self, registry, domains):
        view = self.view(
            registry,
            domains,
            "plant",
            {**PLANT_GOAL, "active_nodes": ["compressor_utilities"]},
        )
        facts = plant_facts(view)
        for text in (
            "compressor_utilities",
            "compressor utilities",
            "compressor and utilities",
        ):
            assert next(f for f in facts if f.name == "active_nodes").holds(
                f"only {text}"
            )


# ------------------------------------------------------------------ scripted translator
class Scripted:
    """A fake backend that answers from the gold labels (or deliberately wrongly)."""

    def __init__(self, items, wrong=()):
        self.by_text = {i.text: i for i in items}
        self.wrong = set(wrong)
        self.n_calls = 0

    def generate(self, prompts, adapter_id=None, **kwargs):
        self.n_calls += 1
        text = prompts[0]
        if "schema" not in kwargs:  # summary request
            view = json.loads(text)
            return [_Gen(f"summary of {view['goal']['spec']}")]
        item = self.by_text.get(text)
        if item is None:  # a summary being re-translated: echo the last goal
            return [_Gen(json.dumps({"refusal": "not an order"}))]
        if item.gold is None:
            return [_Gen(json.dumps({"refusal": "cannot"}))]
        params = json.loads(json.dumps(item.gold))
        if item.id in self.wrong:
            if "step_limit" in params:
                params["step_limit"] = 51 if params["step_limit"] != 51 else 52
            else:
                params["byte_budget"] = 51 if params["byte_budget"] != 51 else 8
        return [_Gen(json.dumps({"refusal": "", "params": params}))]


@dataclass
class _Gen:
    text: str
    latency_s: float = 0.01
    n_prompt_tokens: int = 5


class TestReport:
    def test_perfect_translator_scores_perfectly(self, items, registry, domains):
        sub = select_items(items, 16, seed=2)
        tr = Translator(Scripted(sub), registry)
        recs = evaluate_forward(
            lambda it: tr.translate(it.text, domains[it.domain].pack),
            sub,
            domains,
            registry,
        )
        m = forward_metrics(recs)
        assert m["exact"]["all"]["exact"] == 1.0
        assert m["refusal"]["precision"] == 1.0 and m["refusal"]["recall"] == 1.0
        for block in m["task"].values():
            assert block["diff_mean"] == pytest.approx(0.0) and block["within_ci"]
        assert m["status_counts"].get("ok", 0) == sum(not i.invalid for i in sub)

    def test_wrong_specs_lower_exactness_and_task_score(self, items, registry, domains):
        sub = [i for i in items if i.domain == "squad" and not i.invalid][:6]
        tr = Translator(Scripted(sub, wrong={i.id for i in sub}), registry)
        recs = evaluate_forward(
            lambda it: tr.translate(it.text, domains[it.domain].pack),
            sub,
            domains,
            registry,
        )
        m = forward_metrics(recs)
        assert m["exact"]["all"]["exact"] < 0.5
        assert m["exact"]["all"]["field_accuracy"]["eaters"] == 1.0
        assert any(not r["exact"] for r in recs)

    def test_failed_translation_scores_zero_on_every_seed(
        self, items, registry, domains
    ):
        sub = [i for i in items if i.domain == "squad" and not i.invalid][:2]

        def refuse(item):
            return TranslationResult(item.text, "refused", reason="no")

        recs = evaluate_forward(refuse, sub, domains, registry)
        assert all(r["nl_per_seed"] == [0.0, 0.0] for r in recs)
        block = forward_metrics(recs)["task"]["all/squad"]
        assert (
            block["nl_mean"] == 0.0
            and block["diff_mean"] < 0
            and block["failed_translations"] == 2
        )

    def test_reverse_metrics_and_verdict(self, items, registry, domains):
        sub = select_items(items, 12, seed=5)
        backend = Scripted(sub)
        tr = Translator(backend, registry)
        recs = evaluate_forward(
            lambda it: tr.translate(it.text, domains[it.domain].pack),
            sub,
            domains,
            registry,
        )
        rows = evaluate_reverse(ReverseTranslator(tr), recs, domains, registry)
        assert rows and all(r["summary"].startswith("summary of") for r in rows)
        assert all(not r["round_trip"] for r in rows)  # the fake refuses every summary
        rm = reverse_metrics(rows)
        assert rm["all"]["round_trip"] == 0.0 and rm["all"]["field_fidelity"] < 1.0
        v = verdict(forward_metrics(recs), rm, 0.5, 9.0, smoke=False)
        assert v["label"] == "KILL criterion hit"
        assert verdict(forward_metrics(recs), rm, 0.5, 9.0, smoke=True)[
            "label"
        ].startswith("SMOKE")
        crit = {r["criterion"]: r for r in v["rows"]}
        assert (
            crit["G3 warm translator p95"]["pass"] and crit["G3 fits in 12 GB"]["pass"]
        )
        assert crit["G3 round trip (reverse summary re-translated)"]["kill"]


# ------------------------------------------------------------------ live (optional)
@pytest.mark.integration
def test_live_ollama_translates_one_instruction(registry, domains):
    backend = OllamaBackend("qwen2.5-coder:14b", keep_alive=0)
    if not backend.is_up():
        pytest.skip("Ollama is not running")
    try:
        have = [
            m["name"]
            for m in backend._transport("GET", "/api/tags", None, 10).get("models", [])
        ]
    except OllamaError:
        pytest.skip("Ollama is not answering")
    if backend.base_id not in have:
        pytest.skip(f"{backend.base_id} is not installed")
    res = Translator(backend, registry).translate(
        "Red and blue forage, play it safe, 120 steps.", domains["squad"].pack
    )
    assert res.ok and res.spec.params["eaters"] and registry.is_valid(res.spec)
    assert np.isfinite(res.latency_s)
