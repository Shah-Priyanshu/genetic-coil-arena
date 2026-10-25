"""R7a step 2: the refusal-policy packs for the two domains.

Step 1 asked the model to judge validity and refuse, and it refused valid terse or indirect
requests (refusal precision 59%). Here the model only *extracts* (and may say out of
domain / contradictory / ambiguous); code fills documented defaults, asks for a missing
mandatory choice, and validates (``translator.policy.PolicyTranslator``). These packs carry
the prompts, few-shots, defaults and questions. The few-shot texts are written to share no
phrasing with the step-1 set or the sealed test set (a test checks the texts and frames).
"""

from dataclasses import replace
from typing import Any

from ..translator import Example, TranslationPack
from .r7a_domains import (
    PLANT_DEFAULTS,
    PLANT_DIAGNOSE,
    PLANT_EXAMPLES,
    PLANT_REPORT_EXAMPLES,
    REPORT_NOTE,
    SQUAD_DEFAULTS,
    SQUAD_EXAMPLES,
    SQUAD_MISSION,
    SQUAD_REPORT_EXAMPLES,
    Domain,
)

SQUAD_QUESTIONS = {"eaters": "Which snakes (red, green or blue) should forage?"}

SQUAD_POLICY_SYSTEM = """You turn an order for a squad of three foraging snakes (red, green, blue) into JSON.
Reply with {"refusal": "", "clarify": "", "params": {...}}. Fill exactly one of the three: params for an order you can read, refusal, or clarify.

Your job is to EXTRACT what the order says, in whatever style it is written (terse, polite, chatty, typos, numbers in words, odd units, indirect hints, several sentences). Never refuse or clarify just because an order is short, informal or indirect. If the order corrects itself ("make that green", "actually 150"), use the final version.

params:
- eaters: the colours told to forage. Snakes not mentioned as foragers just wander. "Everyone", "all", "the whole squad" means all three; "everyone except X" means the other two.
- caution: "high" for careful, safe, prudent, cautious, "no risks", "nobody should get hurt"; "low" for reckless, bold, daring, aggressive, "losses are fine"; "normal" for normal or usual care, or when caution is not mentioned. Caution is the opposite of risk appetite.
- step_limit: integer, how many steps (moves, ticks, turns) the mission lasts. Leave it out when none is stated (default 300).
- food_target: integer, the team's food count that ends the mission. Leave it out when none is stated (default 0). A dozen is 12, two dozen 24.

Extract values exactly as stated. Do NOT clamp them and do NOT refuse because a value looks too big, too small or unknown (a fourth colour, 1000 steps): put it in params and the checker will reject it.
Allowed values, for reference: colours red, green, blue; step_limit 50 to 300; food_target 0 to 40.

refusal (short reason) only when the request is not an order about this squad at all (out of domain), or it contradicts itself (for example two different step limits at once, or everyone forages and nobody forages).
clarify (one short question) only when the order is about the squad but a stated quantity is vague ("for a while", "plenty", "a bit more") or a needed choice is unclear ("two of the snakes", "some of them", or no snake is named). Things the order does not mention are not unclear: they take their defaults."""

SQUAD_POLICY_EXAMPLES = (
    Example(
        "Let the green and the red snake gather food for a hundred and ten steps, and be careful.",
        {
            "eaters": ["red", "green"],
            "caution": "high",
            "step_limit": 110,
            "food_target": 0,
        },
    ),
    Example(
        "blue goes foraging. reckless. we are done at 18 food.",
        {"eaters": ["blue"], "caution": "low", "step_limit": 300, "food_target": 18},
    ),
    Example(
        "Everybody except blue should eat for 240 steps; make it two dozen food. Normal care.",
        {
            "eaters": ["red", "green"],
            "caution": "normal",
            "step_limit": 240,
            "food_target": 24,
        },
    ),
    Example(
        "The red snake should gather food for 1200 steps.",
        {"eaters": ["red"], "caution": "normal", "step_limit": 1200, "food_target": 0},
    ),
    Example(
        "Send the purple snake to eat, 90 steps.",
        {"eaters": ["purple"], "caution": "normal", "step_limit": 90, "food_target": 0},
    ),
    Example("Order me a coffee.", None, "this is not an order for the squad"),
    Example(
        "Green must sit the run out, yet green is the only snake that forages.",
        None,
        "green both forages and does not forage",
    ),
    Example(
        "Let a couple of the snakes eat for 120 steps.",
        None,
        "",
        "Which two snakes (red, green or blue) should forage?",
    ),
    Example(
        "Green should hunt for food for a good while.",
        None,
        "",
        "How many steps should the mission last?",
    ),
)

PLANT_QUESTIONS: dict[str, str] = {}

PLANT_POLICY_SYSTEM = """You turn a request about a plant-floor fault-diagnosis system into JSON.
Four sensor nodes (reactor, separator, stripper, compressor_utilities) send small byte messages to a gateway that diagnoses process faults.
Reply with {"refusal": "", "clarify": "", "params": {...}}. Fill exactly one of the three: params for a request you can read, refusal, or clarify.

Your job is to EXTRACT what the request says, in whatever style it is written (terse, polite, chatty, typos, numbers in words, odd units and notations, indirect hints, several sentences). Never refuse or clarify just because a request is short, informal or indirect. If the request corrects itself ("no wait, 51"), use the final version. Settings it does not mention take their defaults, so leave them out.

params:
- byte_budget: integer bytes per node message (also written octets or B). Default 16.
- target_far: the false-alarm rate as a fraction between 0 and 1. Convert: 2% or 2 percent is 0.02, 0.5% is 0.005, 1 in 200 is 0.005, "a hundredth" is 0.01. Default 0.01.
- active_nodes: the nodes that may transmit, from reactor, separator, stripper, compressor_utilities ("compressor", "utilities" and "compressor/utilities" all mean compressor_utilities). "Everything except X", "all but X" and "X off" mean the remaining nodes. Default: all four.

Extract values exactly as stated. Do NOT round them to the nearest allowed value and do NOT refuse because a value looks unusual or not allowed (32 bytes, 3%, a fifth node): put it in params and the checker will reject it.
Allowed values, for reference: byte_budget 8, 16 or 51; target_far 0.001, 0.005, 0.01, 0.02 or 0.05; nodes as listed.

refusal (short reason) only when the request is not about this system at all (out of domain), or it contradicts itself (for example two byte budgets at once, or all nodes active but only one allowed).
clarify (one short question) only when the request is about this system but a stated quantity is vague ("small", "low", "a bit more", "a few nodes") or a needed choice is unclear ("some of the nodes"). Things the request does not mention are not unclear: they take their defaults."""

PLANT_POLICY_EXAMPLES = (Example(
    "Use 8 B messages, a false alarm fraction of 0.02, and only the separator and the reactor.",
    {"byte_budget": 8, "target_far": 0.02, "active_nodes": ["reactor", "separator"]},
), Example(
    "All nodes except the stripper, fifty-one octets, 1 in 100 false alarms.",
    {
        "byte_budget": 51,
        "target_far": 0.01,
        "active_nodes": ["reactor", "separator", "compressor_utilities"],
    },
), Example(
    "Keep the usual message size; false alarms at a tenth of a percent.",
    {
        "byte_budget": 16,
        "target_far": 0.001,
        "active_nodes": [
            "reactor",
            "separator",
            "stripper",
            "compressor_utilities",
        ],
    },
    ),  # fmt: skip
    Example(
        "Give the nodes 100 bytes each.",
        {
            "byte_budget": 100,
            "target_far": 0.01,
            "active_nodes": [
                "reactor",
                "separator",
                "stripper",
                "compressor_utilities",
            ],
        },
    ),
    Example(
        "Use the cooling tower node for the diagnosis.",
        {"byte_budget": 16, "target_far": 0.01, "active_nodes": ["cooling_tower"]},
    ),
    Example(
        "Draw me a diagram of the stripper.",
        None,
        "this is not a request about the diagnosis system",
    ),
    Example(
        "Use 16 bytes with the reactor off, but every node must stay on.",
        None,
        "the reactor is both off and on",
    ),
    Example(
        "Make the false alarms somewhat rarer than usual.",
        None,
        "",
        "Which false-alarm rate should be used (0.1%, 0.5%, 1%, 2% or 5%)?",
    ),
)


def _report_only(
    report: tuple[Example, ...], base: tuple[Example, ...]
) -> tuple[Example, ...]:
    return tuple(e for e in report if e not in base)


def policy_packs(domain: str) -> tuple[TranslationPack, TranslationPack]:
    """``(pack, report_pack)`` of the refusal policy for ``squad`` or ``plant``.

    The report pack also reads a status report of an order that was already carried out
    (the round trip), as in step 1.
    """
    if domain == "squad":
        system, examples, defaults = (
            SQUAD_POLICY_SYSTEM,
            SQUAD_POLICY_EXAMPLES,
            SQUAD_DEFAULTS,
        )
        spec, questions = SQUAD_MISSION, SQUAD_QUESTIONS
        report = _report_only(SQUAD_REPORT_EXAMPLES, SQUAD_EXAMPLES)
    elif domain == "plant":
        system, examples, defaults = (
            PLANT_POLICY_SYSTEM,
            PLANT_POLICY_EXAMPLES,
            PLANT_DEFAULTS,
        )
        spec, questions = PLANT_DIAGNOSE, PLANT_QUESTIONS
        report = _report_only(PLANT_REPORT_EXAMPLES, PLANT_EXAMPLES)
    else:
        raise KeyError(domain)
    pack = TranslationPack(
        spec, system, examples, defaults=defaults, questions=questions
    )
    report_pack = TranslationPack(
        spec,
        system + REPORT_NOTE,
        examples + report,
        defaults=defaults,
        questions=questions,
    )
    return pack, report_pack


def with_policy_packs(domains: dict[str, Domain]) -> dict[str, Domain]:
    """The same domains with the refusal-policy packs in place of the step-1 packs."""
    out: dict[str, Any] = {}
    for name, dom in domains.items():
        pack, report_pack = policy_packs(name)
        out[name] = replace(dom, pack=pack, report_pack=report_pack)
    return out
