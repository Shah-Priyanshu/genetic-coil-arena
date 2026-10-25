"""R7a step 2: training data for the distilled translator.

Three sources, all labelled from specs (never by a model):

1. **Templates.** Gold params (or a refusal / clarify category) are drawn first, then rendered
   from phrase banks written for training. The banks are separate from the step-1 banks and
   share no frame with the sealed test set (checked by n-gram overlap).
2. **LLM paraphrases.** For a gold spec and a style card, a generator model (qwen3-coder:30b
   in the queued jobs) writes several phrasings. ``verify_paraphrases`` then asks a
   *different* model (qwen2.5-coder:14b) to translate each one with the refusal policy and
   keeps only the phrasings from which that model recovers the gold outcome. The label is
   the spec's, never the verifier's.
3. **Retry examples.** The second-attempt prompt of the policy translator (the previous
   reply was rejected, with the validation errors) paired with the corrected reply, built
   by corrupting a gold reply and asking the registry for the real error text.

Every record: ``{id, domain, kind, source, style, text, reply, gold}`` where ``reply`` is
the compact JSON the model is trained to write (``translator.distill.encode_reply``).
``filter_overlap`` drops anything whose frame overlaps the sealed set or the dev/step-1
sets; the training job refuses to start if any survives.
"""

import json
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from ..envs.plant.tep import NODES
from ..protocol.goalspec import SpecRegistry, SpecValidationError
from ..translator import OverlapIndex, encode_reply, retry_prompt
from ..translator.policy import PolicyTranslator
from .r7a_domains import (
    BYTE_BUDGETS,
    COLOURS,
    PLANT_DEFAULTS,
    PLANT_DIAGNOSE,
    SQUAD_DEFAULTS,
    SQUAD_MISSION,
    TARGET_FARS,
)
from .r7a_evalset import words

DEFAULTS = {"squad": SQUAD_DEFAULTS, "plant": PLANT_DEFAULTS}
SPEC_NAMES = {"squad": SQUAD_MISSION, "plant": PLANT_DIAGNOSE}
MASKS: dict[str, str] = {
    **dict.fromkeys(COLOURS, "<C>"),
    **dict.fromkeys(
        ("reactor", "separator", "stripper", "compressor", "utilities"), "<K>"
    ),
}

# ---------------------------------------------------------------------- helpers


def _join(names: Sequence[str], word: str = "and") -> str:
    names = list(names)
    return (
        names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" {word} " + names[-1]
    )


def _num(rng: random.Random, n: int, p_words: float = 0.25) -> str:
    return words(n) if n < 1000 and rng.random() < p_words else str(n)


def typo(rng: random.Random, text: str) -> str:
    """Corrupt one or two long alphabetic words (swap or drop a letter); numbers are safe."""
    toks = text.split(" ")
    cands = [i for i, t in enumerate(toks) if re.fullmatch(r"[A-Za-z]{5,}", t)]
    for i in rng.sample(cands, min(len(cands), rng.choice([1, 1, 2]))):
        t = toks[i]
        j = rng.randrange(1, len(t) - 1)
        toks[i] = (
            t[:j] + t[j + 1] + t[j] + t[j + 2 :]
            if rng.random() < 0.6
            else t[:j] + t[j + 1 :]
        )
    return " ".join(toks)


def _finish(rng: random.Random, text: str) -> tuple[str, list[str]]:
    tags = []
    if rng.random() < 0.12:
        text, tags = typo(rng, text), ["typo"]
    if rng.random() < 0.12:
        text = text.lower()
    if rng.random() < 0.08:
        text += rng.choice([" Thanks.", " Please.", " Go.", " Thank you."])
    return text, tags


# ---------------------------------------------------------------------- squad templates

_S_ONE = [
    "{c} is the forager", "have {c} do the foraging", "{c} is on food-gathering duty",
    "only {c} gathers food", "let {c} collect food", "{c} to forage",
    "put {c} in charge of finding food", "{c} goes out for food",
]  # fmt: skip
_S_TWO = [
    "{a} and {b} are the foragers", "{a} plus {b} gather food",
    "have both {a} and {b} collect food", "{a} and {b} on food-gathering duty",
    "the {a} and the {b} snakes go out for food", "{a}, {b} to forage",
]  # fmt: skip
_S_ALL = [
    "all three snakes gather food", "the entire squad collects food",
    "every snake is a forager", "red, green and blue all forage",
    "all snakes on food duty", "each of the three snakes forages",
]  # fmt: skip
_S_EXCL = [
    "every snake but {x} forages", "all of the snakes other than {x} gather food",
    "{x} sits out while the rest forage", "the rest forage, {x} does not",
]  # fmt: skip
_S_CAUTION = {
    "low": ["take chances", "be fearless", "ignore the danger", "push hard, losses are acceptable",
            "go in aggressively", "risky play is fine", "no need to be careful",
            "a big appetite for risk"],
    "normal": ["with ordinary caution", "caution at the standard level",
               "no special precautions", "a normal level of care"],
    "high": ["avoid danger", "stay careful", "keep everyone alive", "minimise risk",
             "play conservatively", "careful play only", "be cautious",
             "a small appetite for risk", "the lowest risk you can manage"],
}  # fmt: skip
_S_UNITS = ["steps", "moves", "ticks", "turns", "rounds", "cycles"]
_S_STEP = [
    "lasting {n} {u}", "{n} {u} in total", "run it {n} {u}", "end after {n} {u}",
    "a limit of {n} {u}", "time allowed: {n} {u}", "you get {n} {u}",
]  # fmt: skip
_S_NOUNS = ["food", "pellets", "apples", "pieces of food", "items", "bites"]
_S_TARGET = [
    "until {n} {noun} are collected", "goal: {n} {noun}", "collect {n} {noun} then stop",
    "the aim is {n} {noun}", "finish once the squad has {n} {noun}", "a target of {n} {noun}",
]  # fmt: skip
_S_STEPS = (
    50,
    60,
    70,
    75,
    80,
    90,
    100,
    110,
    120,
    130,
    140,
    150,
    160,
    175,
    180,
    200,
    220,
    240,
    250,
    275,
    300,
)
_S_TARGETS = (
    1,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    14,
    15,
    16,
    18,
    20,
    22,
    24,
    25,
    28,
    30,
    32,
    35,
    40,
)
_S_SETS = [
    ["red"], ["green"], ["blue"], ["red", "green"], ["red", "blue"],
    ["green", "blue"], ["red", "green", "blue"],
]  # fmt: skip


def _squad_eaters(rng: random.Random, eaters: list[str]) -> str:
    if len(eaters) == 1:
        return rng.choice(_S_ONE).format(c=eaters[0])
    if len(eaters) == 3:
        return rng.choice(_S_ALL)
    if rng.random() < 0.35:
        other = [c for c in COLOURS if c not in eaters][0]
        return rng.choice(_S_EXCL).format(x=other)
    a, b = eaters
    return rng.choice(_S_TWO).format(a=a, b=b)


def _squad_target_phrase(rng: random.Random, n: int) -> str:
    special = {6: "half a dozen", 12: "a dozen", 24: "two dozen"}
    if n in special and rng.random() < 0.5:
        return rng.choice(["collect {n} then stop", "the aim is {n} in all"]).format(
            n=special[n]
        )
    return rng.choice(_S_TARGET).format(n=_num(rng, n), noun=rng.choice(_S_NOUNS))


def draw_squad(rng: random.Random) -> tuple[dict[str, Any], dict[str, bool]]:
    """Gold squad params and which optional ones the text will state."""
    shown = {
        "caution": rng.random() < 0.6,
        "step_limit": rng.random() < 0.7,
        "food_target": rng.random() < 0.5,
    }
    if not any(shown.values()):
        shown["step_limit"] = True
    p = {
        "eaters": list(rng.choice(_S_SETS)),
        "caution": (
            rng.choice(["low", "normal", "high"]) if shown["caution"] else "normal"
        ),
        "step_limit": rng.choice(_S_STEPS) if shown["step_limit"] else 300,
        "food_target": rng.choice(_S_TARGETS) if shown["food_target"] else 0,
    }
    # now and then state a default explicitly
    if not shown["step_limit"] and rng.random() < 0.1:
        shown["step_limit"] = True
    if not shown["food_target"] and rng.random() < 0.08:
        shown["food_target"] = True
    return p, shown


def render_squad(
    rng: random.Random, p: dict[str, Any], shown: dict[str, bool]
) -> tuple[str, str]:
    """A training sentence for squad params; returns ``(text, style tags)``."""
    eat = _squad_eaters(rng, p["eaters"])
    parts = []
    if shown["caution"]:
        parts.append(rng.choice(_S_CAUTION[p["caution"]]))
    if shown["step_limit"]:
        parts.append(
            rng.choice(_S_STEP).format(
                n=_num(rng, p["step_limit"]), u=rng.choice(_S_UNITS)
            )
        )
    if shown["food_target"]:
        if p["food_target"] == 0:
            parts.append(rng.choice(["no food target", "no particular food goal"]))
        else:
            parts.append(_squad_target_phrase(rng, p["food_target"]))
    rng.shuffle(parts)
    rest = ", ".join(parts)
    frame = rng.randrange(8)
    cap = lambda s: s[:1].upper() + s[1:]  # noqa: E731
    if frame == 0:
        text = f"{cap(eat)}. " + " ".join(cap(x) + "." for x in parts)
    elif frame == 1:
        text = f"{cap(eat)}; " + "; ".join(parts) + "."
    elif frame == 2:
        text = f"We want this: {eat}, {rest}."
    elif frame == 3:
        text = f"Order: {eat} - " + " - ".join(parts)
    elif frame == 4:
        text = (cap(rest) + f", and {eat}.") if parts else f"{cap(eat)}."
    elif frame == 5:
        text = f"Squad task. {cap(eat)}. Details: {rest}."
    elif frame == 6:
        text = f"Please make sure that {eat}; " + " and ".join(parts) + "."
    else:
        text = f"{cap(eat)} ({rest})"
    text, tags = _finish(rng, text)
    return text, ",".join(["template", *tags])


# ---------------------------------------------------------------------- plant templates

_P_BUDGET = [
    "uplink size {n} bytes", "each message carries {n} bytes", "keep payloads to {n} B",
    "a {n}-byte budget", "{n} octets per node", "messages of {n} bytes",
]  # fmt: skip
_P_NODES_ALL = [
    "all four nodes report",
    "every sensor node is active",
    "no node is switched off",
]
_P_NODE_NAMES = {
    "reactor": ["reactor", "the reactor node"],
    "separator": ["separator", "the separator node"],
    "stripper": ["stripper", "the stripper node"],
    "compressor_utilities": ["compressor and utilities", "compressor/utilities", "the utilities node"],
}  # fmt: skip
_P_SETS = [[n] for n in NODES] + [
    ["reactor", "separator"], ["reactor", "stripper"], ["reactor", "compressor_utilities"],
    ["separator", "stripper"], ["separator", "compressor_utilities"],
    ["stripper", "compressor_utilities"], ["reactor", "separator", "stripper"],
    ["reactor", "separator", "compressor_utilities"], ["reactor", "stripper", "compressor_utilities"],
    ["separator", "stripper", "compressor_utilities"], list(NODES),
]  # fmt: skip


def far_forms(rng: random.Random, far: float) -> str:
    """One of several spellings of a false-alarm rate."""
    pct = far * 100
    pct_s = f"{pct:g}"
    forms = [f"{pct_s}%", f"{pct_s} percent", f"{pct_s} per cent", f"{far:g}"]
    ratio = 1 / far
    if abs(ratio - round(ratio)) < 1e-9:
        forms += [
            f"1 in {round(ratio)}",
            "one in " + ("a thousand" if ratio >= 1000 else words(round(ratio))),
        ]
    if far * 1000 == int(far * 1000):
        forms.append(f"{int(far * 1000)} per thousand")
    return rng.choice(forms)


_P_FAR = [
    "false alarm rate {f}", "allow false alarms at {f}", "false alarms capped at {f}",
    "FAR = {f}", "alarm rate on healthy data: {f}", "keep spurious alarms to {f}",
]  # fmt: skip


def draw_plant(rng: random.Random) -> tuple[dict[str, Any], dict[str, bool]]:
    """Gold plant params and which of them the text will state."""
    shown = {
        "byte_budget": rng.random() < 0.8,
        "target_far": rng.random() < 0.75,
        "active_nodes": rng.random() < 0.65,
    }
    if not any(shown.values()):
        shown["byte_budget"] = True
    nodes = list(rng.choice(_P_SETS)) if shown["active_nodes"] else list(NODES)
    p = {
        "byte_budget": rng.choice(BYTE_BUDGETS) if shown["byte_budget"] else 16,
        "target_far": rng.choice(TARGET_FARS) if shown["target_far"] else 0.01,
        "active_nodes": [n for n in NODES if n in nodes],
    }
    if not shown["byte_budget"] and rng.random() < 0.08:
        shown["byte_budget"] = True
    if not shown["target_far"] and rng.random() < 0.08:
        shown["target_far"] = True
    return p, shown


def _plant_nodes_phrase(rng: random.Random, nodes: list[str]) -> str:
    if len(nodes) == 4:
        return rng.choice(_P_NODES_ALL)
    if len(nodes) == 3 and rng.random() < 0.5:
        out = [n for n in NODES if n not in nodes][0]
        return rng.choice(
            [
                "everything apart from {x}",
                "switch {x} off, keep the others",
                "all nodes but {x}",
            ]
        ).format(x=rng.choice(_P_NODE_NAMES[out]))
    names = [rng.choice(_P_NODE_NAMES[n]) for n in nodes]
    return rng.choice(
        [
            "active nodes: {w}",
            "only {w} transmit",
            "restrict reporting to {w}",
            "{w} do the sensing",
        ]
    ).format(w=_join(names))


def render_plant(
    rng: random.Random, p: dict[str, Any], shown: dict[str, bool]
) -> tuple[str, str]:
    """A training sentence for plant params; returns ``(text, style tags)``."""
    parts = []
    if shown["byte_budget"]:
        parts.append(rng.choice(_P_BUDGET).format(n=_num(rng, p["byte_budget"], 0.2)))
    if shown["target_far"]:
        parts.append(rng.choice(_P_FAR).format(f=far_forms(rng, p["target_far"])))
    if shown["active_nodes"]:
        parts.append(_plant_nodes_phrase(rng, p["active_nodes"]))
    rng.shuffle(parts)
    rest = ", ".join(parts)
    cap = lambda s: s[:1].upper() + s[1:]  # noqa: E731
    frame = rng.randrange(7)
    if frame == 0:
        text = " ".join(cap(x) + "." for x in parts)
    elif frame == 1:
        text = "Configure the diagnosis: " + "; ".join(parts) + "."
    elif frame == 2:
        text = f"For the gateway, {rest}."
    elif frame == 3:
        text = "Settings - " + " - ".join(parts)
    elif frame == 4:
        text = f"We need {rest}."
    elif frame == 5:
        text = f"Diagnosis run ({rest})"
    else:
        text = cap(rest) + ", please."
    text, tags = _finish(rng, text)
    return text, ",".join(["template", *tags])


# ---------------------------------------------------------------------- status reports


def render_report(rng: random.Random, domain: str, p: dict[str, Any]) -> str:
    """A status report of a finished task, in the shape the summary writer produces.

    The round trip feeds such a report back to the translator, so the distilled model sees
    a few in training (the shape is fixed by the summary prompt, not by any test item).
    """
    if domain == "squad":
        names = _join(p["eaters"])
        noun, verb = ("snake", "was") if len(p["eaters"]) == 1 else ("snakes", "were")
        target = (
            "no food target"
            if p["food_target"] == 0
            else f"a food target of {p['food_target']}"
        )
        score = round(rng.uniform(0.5, 12.0), 1)
        glue = rng.choice([" ", ", "])
        return (
            f"The {names} {noun} {verb} ordered to forage at caution level {p['caution']}"
            f"{glue}with a step limit of {p['step_limit']} and {target}; "
            f"the squad finished with mission score {score}."
        )
    score = round(rng.uniform(0.2, 0.95), 2)
    pct = f"{p['target_far'] * 100:g}%"
    nodes = p["active_nodes"]
    if len(nodes) == 4 and rng.random() < 0.5:
        return (
            f"With {p['byte_budget']} bytes per message and {pct} false alarms, all four nodes "
            f"({', '.join(NODES)}) reached macro-F1 {score}."
        )
    return (
        f"The diagnosis ran with {p['byte_budget']} bytes per message and {pct} false alarms "
        f"on the {_join(nodes)} nodes, reaching macro-F1 {score}."
    )


# ---------------------------------------------------------------------- refuse / clarify

_OOD = [
    "Book me a flight to Denver.", "What is 17 times 23?", "Translate hello into Spanish.",
    "Who won the football match last night?", "Recommend a good book on gardening.",
    "Explain how photosynthesis works.", "Write a short poem about autumn.",
    "Remind me to call my mother tomorrow.", "How do I bake sourdough bread?",
    "Reboot the email server.", "Turn the office lights off.", "What is the capital of Peru?",
    "Draft a polite email to my landlord.", "Play some jazz music.", "Tell me about the French Revolution.",
    "How tall is Mount Everest?", "Please clean my desk.", "Sing me a lullaby.",
    "Order more printer paper.", "What does DNA stand for?",
]  # fmt: skip
_BAD_COLOURS = [
    "yellow",
    "purple",
    "orange",
    "pink",
    "black",
    "white",
    "brown",
    "gray",
    "violet",
    "cyan",
]
_BAD_NODES = [
    "condenser",
    "boiler",
    "furnace",
    "pump",
    "turbine",
    "cooling tower",
    "heat exchanger",
    "fifth node",
    "mixer",
]
_BAD_STEPS = (5, 10, 15, 20, 30, 40, 400, 500, 600, 800, 1000, 1500, 2000)
_BAD_TARGETS = (41, 50, 60, 75, 100, 150, 200, 500)
_BAD_BYTES = (1, 2, 4, 6, 10, 12, 20, 24, 32, 64, 100, 128, 256, 500, 1024)
_BAD_FARS = (0.0001, 0.002, 0.003, 0.03, 0.04, 0.1, 0.15, 0.2, 0.25, 0.5)


def draw_refusal(rng: random.Random, domain: str) -> tuple[str, str, str]:
    """``(text, reason, category)`` of a request that must be refused."""
    cat = rng.choice(["ood", "range", "bad_name", "contradiction", "ood"])
    if domain == "squad":
        if cat == "ood":
            return rng.choice(_OOD), "this is not an order for the squad", cat
        if cat == "bad_name":
            if rng.random() < 0.7:
                bad = rng.choice(_BAD_COLOURS)
                text = rng.choice(
                    [
                        "Send the {b} snake to forage for {n} steps.",
                        "{B} and red gather food, {n} moves.",
                        "Get the {b} one to eat, {n} ticks.",
                        "The {b} snake forages.",
                    ]
                ).format(b=bad, B=bad.capitalize(), n=rng.choice(_S_STEPS))
                return (
                    text,
                    f"there is no {bad} snake; the colours are red, green and blue",
                    cat,
                )
            n = rng.choice(["four", "five", "six", "all four", "seven"])
            return (
                f"Have {n} snakes forage for {rng.choice(_S_STEPS)} steps.",
                "the squad has only three snakes",
                cat,
            )
        if cat == "contradiction":
            c = rng.choice(COLOURS)
            a, b = rng.sample(list(_S_STEPS), 2)
            text = rng.choice(
                [
                    f"{c.capitalize()} forages for {a} steps and also for {b} steps.",
                    "All snakes forage, yet none of the snakes may forage.",
                    f"{c.capitalize()} should be careful and also reckless, {a} steps.",
                    f"Run the mission for {a} steps. No, wait: it must last exactly {b} steps and also {a}.",
                ]
            )
            return text, "the order contradicts itself", cat
        # range
        c = rng.choice(COLOURS)
        if rng.random() < 0.65:
            n = rng.choice(_BAD_STEPS)
            text = rng.choice(
                [
                    f"{c.capitalize()} forages for {n} steps.",
                    f"Let {c} gather food, lasting {_num(rng, n)} moves.",
                    f"{n} ticks of foraging for the {c} snake.",
                ]
            )
            return text, "step limit must be between 50 and 300", cat
        n = rng.choice(_BAD_TARGETS)
        return (
            f"{c.capitalize()} forages until {n} food is collected.",
            "food target must be between 0 and 40",
            cat,
        )
    # plant
    if cat == "ood":
        return rng.choice(_OOD), "this is not a request about the diagnosis system", cat
    if cat == "bad_name":
        bad = rng.choice(_BAD_NODES)
        text = rng.choice(
            [
                "Include the {b} in the diagnosis.",
                "Add {b} to the active nodes, 16 bytes.",
                "Use the {b} and the reactor.",
                "Only the {b} should report.",
            ]
        ).format(b=bad)
        return text, f"there is no {bad} node", cat
    if cat == "contradiction":
        a, b = rng.sample(list(BYTE_BUDGETS), 2)
        text = rng.choice(
            [
                f"Use {a} bytes per message and also {b} bytes per message.",
                "Every node must report, but only the reactor may report.",
                f"False alarms at 1% and also at 5%, {a} bytes.",
                "Switch the stripper off, and keep the stripper active.",
            ]
        )
        return text, "the request contradicts itself", cat
    if rng.random() < 0.5:
        n = rng.choice(_BAD_BYTES)
        text = rng.choice(
            [
                f"Use {n} bytes per message.",
                f"Give each node a {n}-byte budget.",
                f"{n} B payloads, all nodes.",
            ]
        )
        return text, "the byte budget must be 8, 16 or 51", cat
    f = rng.choice(_BAD_FARS)
    text = rng.choice(
        [
            f"Allow false alarms at {far_forms(rng, f)}.",
            f"False alarm rate {far_forms(rng, f)}, 16 bytes.",
            f"Target a {far_forms(rng, f)} false-alarm rate.",
        ]
    )
    return text, "the false-alarm rate must be 0.1%, 0.5%, 1%, 2% or 5%", cat


def draw_clarify(rng: random.Random, domain: str) -> tuple[str, str, str]:
    """``(text, question, category)`` of an in-domain but ambiguous request."""
    cat = rng.choice(["missing_choice", "vague", "relative", "vague"])
    if domain == "squad":
        if cat == "missing_choice":
            n = rng.choice(_S_STEPS)
            text = rng.choice(
                [
                    f"Play it safe for {n} steps.",
                    f"Careful play only, {n} moves, target {rng.choice(_S_TARGETS)} food.",
                    f"Forage for {n} ticks, aggressively.",
                    f"Go for {rng.choice(_S_TARGETS)} food within {n} steps.",
                ]
            )
            return text, "Which snakes (red, green or blue) should forage?", cat
        if cat == "vague":
            c = rng.choice(COLOURS)
            text = rng.choice(
                [
                    f"{c.capitalize()} forages for quite some time.",
                    f"Let {c} gather a lot of food.",
                    f"{c.capitalize()} should hunt for a short while, carefully.",
                    f"{c.capitalize()} eats until it is full.",
                ]
            )
            q = (
                "How many steps should the mission last?"
                if "time" in text or "while" in text
                else "How much food counts as the target?"
            )
            return text, q, cat
        if cat == "relative":
            text = rng.choice(
                [
                    "Be a little more careful than last time.",
                    "Same as before but longer.",
                    "Make the mission somewhat shorter.",
                    "A bit more food than yesterday.",
                ]
            )
            return (
                text,
                "Please give the actual values (snakes, caution, steps, food target).",
                cat,
            )
        a = rng.choice(["two", "a few", "some", "a couple of", "most of"])
        return (
            f"Send {a} of the snakes out to forage.",
            "Which snakes (red, green or blue) should forage?",
            cat,
        )
    if cat == "missing_choice":
        text = rng.choice(
            [
                "Only some of the nodes should be active.",
                "Use a few nodes, 8 bytes.",
                "Switch a couple of nodes off.",
            ]
        )
        return (
            text,
            "Which nodes (reactor, separator, stripper, compressor_utilities) should be active?",
            cat,
        )
    if cat == "relative":
        text = rng.choice(
            [
                "Make the messages somewhat larger.",
                "A bit fewer false alarms than now.",
                "Same as before but cheaper.",
            ]
        )
        return (
            text,
            "Please give the actual values (bytes, false-alarm rate, nodes).",
            cat,
        )
    text = rng.choice(
        [
            "Keep the uplink small.",
            "Use a generous payload.",
            "Allow only a handful of false alarms.",
            "False alarms must be rare.",
        ]
    )
    q = (
        "Which false-alarm rate should be used?"
        if "alarm" in text
        else "How many bytes per message (8, 16 or 51)?"
    )
    return text, q, cat


# ---------------------------------------------------------------------- records


@dataclass
class Record:
    """One training example."""

    id: str
    domain: str
    kind: str  # spec | refuse | clarify | retry
    source: str  # template | paraphrase | retry
    style: str
    text: str
    reply: str
    gold: Optional[dict[str, Any]] = None
    category: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_json(line: str) -> "Record":
        return Record(**json.loads(line))


def spec_reply(domain: str, params: Mapping[str, Any]) -> str:
    """The compact reply for gold params (defaults left out)."""
    return encode_reply(params=params, defaults=DEFAULTS[domain])


def make_templates(
    n_spec: int,
    seed: int = 0,
    refuse_share: float = 0.11,
    clarify_share: float = 0.07,
    report_share: float = 0.08,
) -> list[Record]:
    """``n_spec`` valid templated examples per domain, plus refusals and clarifications.

    Refusals and clarifications are added so that they make up ``refuse_share`` and
    ``clarify_share`` of the result; ``report_share * n_spec`` status-report examples
    (for the round trip) are added per domain.
    """
    rng = random.Random(seed)
    out: list[Record] = []
    seen: set[tuple[str, str]] = set()
    makers = {"squad": (draw_squad, render_squad), "plant": (draw_plant, render_plant)}
    n_rep = round(n_spec * report_share)
    n_ref = round(n_spec * refuse_share / (1 - refuse_share - clarify_share))
    n_cla = round(n_spec * clarify_share / (1 - refuse_share - clarify_share))
    for domain, (draw, render) in makers.items():
        k = 0
        while k < n_spec:
            params, shown = draw(rng)
            text, style = render(rng, params, shown)
            if (domain, text) in seen:
                continue
            seen.add((domain, text))
            out.append(
                Record(
                    f"tpl-{domain}-{k:05d}",
                    domain,
                    "spec",
                    "template",
                    style,
                    text,
                    spec_reply(domain, params),
                    params,
                )
            )
            k += 1
        for j in range(n_rep):
            params, _ = draw(rng)
            text = render_report(rng, domain, params)
            out.append(
                Record(
                    f"tpl-{domain}-rep{j:04d}",
                    domain,
                    "spec",
                    "template",
                    "template,report",
                    text,
                    spec_reply(domain, params),
                    params,
                )
            )
        for label, count, draw_fn in (
            ("refuse", n_ref, draw_refusal),
            ("clarify", n_cla, draw_clarify),
        ):
            j, tries = 0, 0
            while j < count and tries < count * 20:
                tries += 1
                text, msg, cat = draw_fn(rng, domain)
                if rng.random() < 0.1:
                    text = typo(rng, text)
                if (domain, text) in seen:
                    continue
                seen.add((domain, text))
                reply = (
                    encode_reply(refusal=msg)
                    if label == "refuse"
                    else encode_reply(clarify=msg)
                )
                out.append(
                    Record(
                        f"tpl-{domain}-{label[:3]}{j:04d}",
                        domain,
                        label,
                        "template",
                        "template",
                        text,
                        reply,
                        None,
                        cat,
                    )
                )
                j += 1
    return out


def retry_examples(
    registry: SpecRegistry, records: Sequence[Record], seed: int = 0, n: int = 200
) -> list[Record]:
    """Examples of the second attempt: a rejected first reply, the real errors, the fix.

    A spec record's gold reply is corrupted (an out-of-range value, a wrong enum, a unknown
    key); the registry's own error text goes into :func:`retry_prompt`, and the target is
    the original (correct) reply.
    """
    rng = random.Random(seed)
    specs = [r for r in records if r.kind == "spec" and r.gold]
    out: list[Record] = []
    for rec in rng.sample(specs, min(n, len(specs))):
        params = json.loads(json.dumps(rec.gold))
        key = rng.choice(sorted(params))
        value = params[key]
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            params[key] = value * 20 + 1
        elif isinstance(value, float):
            params[key] = round(value * 100, 6)
        elif isinstance(value, list):
            params[key] = value + ["nonexistent"]
        else:
            params[key] = "unknown"
        try:
            registry.make(SPEC_NAMES[rec.domain], params)
        except SpecValidationError as exc:
            errors = list(exc.errors) or [str(exc)]
        else:
            continue
        out.append(
            Record(
                rec.id.replace("tpl-", "retry-").replace("par-", "retry-"),
                rec.domain,
                "retry",
                "retry",
                "retry",
                retry_prompt(rec.text, errors),
                rec.reply,
                rec.gold,
            )
        )
    return out


# ---------------------------------------------------------------------- paraphrases

STYLE_CARDS: dict[str, str] = {
    "polite_question": "a polite question or a request phrased as 'could you / would it be possible'",
    "casual_chat": "casual chat, lower-case, contractions, a bit rambling",
    "terse_note": "an extremely terse note, fragments, symbols or separators like / | + =",
    "numbers_in_words": "every number written out in words (no digits)",
    "mixed_notation": "unusual number notations: fractions, 'per thousand', ratios, hexadecimal, scientific notation, 'a dozen', half a hundred",
    "indirect_hint": "an indirect hint or wish rather than a command ('it would be nice if', 'I was thinking')",
    "multi_sentence": "two or three short sentences, with the details in separate sentences",
    "self_correction": "the speaker corrects themselves mid-way ('wait', 'no, make that', 'actually'), the FINAL values being the specified ones",
    "formal_memo": "a formal memo or a ticket from an operations manual",
    "typos": "a hurried message with a few spelling mistakes and missing letters",
    "negation": "describing the groups by exclusion ('everyone except', 'all but', 'leave X out', 'no X')",
    "jargon": "operator jargon and abbreviations",
}


def describe(domain: str, rec: Record) -> str:
    """A plain description of what a paraphrase must convey (for the generator prompt)."""
    if rec.kind == "spec":
        p = rec.gold or {}
        if domain == "squad":
            tgt = (
                "none (not stated)" if p["food_target"] == 0 else str(p["food_target"])
            )
            return (
                f"Squad of snakes (red, green, blue). Foraging snakes: {', '.join(p['eaters'])}. "
                f"Caution: {p['caution']} ({'careful/safe' if p['caution'] == 'high' else 'reckless/risk-taking' if p['caution'] == 'low' else 'ordinary, unremarkable care'}). "
                f"Step limit: {p['step_limit']}{' (the default, so you may leave it unstated)' if p['step_limit'] == 300 else ''}. Food target: {tgt}."
            )
        nodes = ", ".join(p["active_nodes"])
        return (
            f"Plant fault-diagnosis system with sensor nodes reactor, separator, stripper, compressor_utilities. "
            f"Bytes per message: {p['byte_budget']}. False-alarm rate: {p['target_far']} (a fraction; {p['target_far'] * 100:g}%). Active nodes: {nodes}."
        )
    if rec.kind == "refuse":
        return f"A request that cannot be fulfilled because: {rec.category.replace('_', ' ')}. Same flavour as: \"{rec.text}\""
    return f"An ambiguous request ({rec.category.replace('_', ' ')}) that leaves a needed choice or quantity vague. Same flavour as: \"{rec.text}\""


PARAPHRASE_SCHEMA = {
    "type": "object",
    "properties": {"texts": {"type": "array", "items": {"type": "string"}}},
    "required": ["texts"],
}


def paraphrase_prompt(domain: str, rec: Record, style: str, k: int) -> str:
    """The prompt asking the generator for ``k`` phrasings of one gold meaning."""
    return (
        f"Write {k} different ways a person might type the following request to a control system.\n"
        f"Meaning to express:\n{describe(domain, rec)}\n\n"
        f"Style: {STYLE_CARDS[style]}.\n"
        "Rules: state every listed value and nothing that is not listed (values marked as default or 'not stated' may be left out or stated plainly); "
        "do not add other instructions; each phrasing must be different in wording; do not number them; keep each under 40 words.\n"
        'Answer as JSON: {"texts": ["...", "..."]}'
    )


def generate_paraphrases(
    backend: Any,
    bases: Sequence[Record],
    styles: Sequence[str],
    per_call: int = 8,
    seed: int = 0,
    sink: Optional[Callable[[dict[str, Any]], None]] = None,
    done_ids: Optional[set[str]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[dict[str, Any]]:
    """Ask ``backend`` for paraphrases of each base record in one rotating style.

    Returns raw rows ``{base_id, style, text}``; ``sink`` receives each call's rows as they
    arrive (so a job can append them to a file) and ``done_ids`` skips bases already done.
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for i, base in enumerate(bases):
        style = styles[(i + rng.randrange(len(styles))) % len(styles)]
        if done_ids and base.id in done_ids:
            continue
        prompt = paraphrase_prompt(base.domain, base, style, per_call)
        try:
            (res,) = backend.generate(
                [prompt],
                schema=PARAPHRASE_SCHEMA,
                options={"temperature": 0.9, "num_ctx": 4096},
            )
            texts = json.loads(res.text).get("texts", [])
        except Exception:  # noqa: BLE001 - a bad call loses one base, not the run
            texts = []
        new = [
            {"base_id": base.id, "style": style, "text": t.strip()}
            for t in texts
            if isinstance(t, str) and 3 <= len(t.strip()) <= 400
        ]
        rows.extend(new)
        if sink:
            for row in new:
                sink(row)
            if not new:
                sink({"base_id": base.id, "style": style, "text": ""})
        if progress:
            progress(i + 1, len(bases))
    return rows


def verify_paraphrases(
    translator: PolicyTranslator,
    packs: Mapping[str, Any],
    bases: Mapping[str, Record],
    rows: Sequence[Mapping[str, Any]],
    sink: Optional[Callable[[dict[str, Any]], None]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[dict[str, Any]]:
    """Keep the paraphrases from which a different model recovers the gold outcome.

    A spec paraphrase passes when the verifier's params equal the gold params (defaults
    filled); a refusal passes when the verifier refuses (or cannot produce a spec); a
    clarification passes when it asks a question.
    """
    from ..translator import CLARIFY, OK, exact_match

    out: list[dict[str, Any]] = []
    for k, row in enumerate(rows):
        base = bases[row["base_id"]]
        res = translator.translate(row["text"], packs[base.domain])
        schema = translator.registry.get(SPEC_NAMES[base.domain]).params_schema
        if base.kind == "spec":
            ok = res.status == OK and exact_match(res.params, base.gold or {}, schema)
        elif base.kind == "refuse":
            ok = res.status not in (OK, CLARIFY)
        else:
            ok = res.status == CLARIFY
        verdict = {**row, "verified": bool(ok), "verifier_status": res.status}
        out.append(verdict)
        if sink:
            sink(verdict)
        if progress:
            progress(k + 1, len(rows))
    return out


def paraphrase_records(
    bases: Mapping[str, Record], verified: Sequence[Mapping[str, Any]]
) -> list[Record]:
    """Training records for the verified paraphrases (labels copied from their bases)."""
    out = []
    counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for row in verified:
        if not row.get("verified") or not row["text"]:
            continue
        base = bases[row["base_id"]]
        key = (base.domain, row["text"].lower())
        if key in seen:
            continue
        seen.add(key)
        j = counts.get(base.id, 0)
        counts[base.id] = j + 1
        out.append(
            Record(
                f"par-{base.id}-{j}",
                base.domain,
                base.kind,
                "paraphrase",
                row["style"],
                row["text"],
                base.reply,
                base.gold,
                base.category,
            )
        )
    return out


# ---------------------------------------------------------------------- leak control


def filter_overlap(
    records: Sequence[Record],
    references: Sequence[str],
    threshold: float = 0.5,
    n: int = 4,
) -> tuple[list[Record], list[tuple[Record, float, str]]]:
    """Drop records whose frame shares >= ``threshold`` of some reference's n-grams.

    Returns ``(kept, dropped)`` where ``dropped`` holds ``(record, score, reference)``.
    The retry records carry the instruction inside a longer prompt, so they are checked
    on that prompt; a leak there would also show as a high containment.
    """
    index = OverlapIndex(references, n, MASKS)
    kept, dropped = [], []
    for rec in records:
        score, j = index.max_containment(rec.text)
        if score >= threshold:
            dropped.append((rec, score, index.references[j]))
        else:
            kept.append(rec)
    return kept, dropped


def assert_no_leak(
    records: Sequence[Record],
    references: Sequence[str],
    threshold: float = 0.5,
    n: int = 4,
) -> None:
    """Raise if any record's frame overlaps a reference text (the training job calls this)."""
    _, dropped = filter_overlap(records, references, threshold, n)
    if dropped:
        rec, score, ref = dropped[0]
        raise ValueError(
            f"{len(dropped)} training records overlap the held-out texts; first: "
            f"{rec.text!r} ~ {ref!r} ({score:.2f})"
        )


# ---------------------------------------------------------------------- selection


def select_training(
    pool: Sequence[Record],
    n: int,
    seed: int = 0,
    paraphrase_share: float = 0.5,
    retry_share: float = 0.04,
) -> list[Record]:
    """A training set of about ``n`` records from the pool.

    Takes ``paraphrase_share`` of the non-retry examples from verified paraphrases when
    there are enough (the rest from templates), keeps the pool's refuse / clarify mix, and
    adds retry examples for ``retry_share`` of ``n``.
    """
    rng = random.Random(seed)
    main = [r for r in pool if r.kind != "retry"]
    retry = [r for r in pool if r.kind == "retry"]
    n_retry = min(len(retry), round(n * retry_share))
    n_main = n - n_retry
    para = [r for r in main if r.source == "paraphrase"]
    tpl = [r for r in main if r.source == "template"]
    rng.shuffle(para)
    rng.shuffle(tpl)
    want_para = min(len(para), round(n_main * paraphrase_share))
    chosen = para[:want_para] + tpl[: n_main - want_para]
    if len(chosen) < n_main:  # templates ran out; top up from the rest
        chosen += para[want_para : want_para + n_main - len(chosen)]
    rng.shuffle(retry)
    chosen += retry[:n_retry]
    rng.shuffle(chosen)
    return chosen
