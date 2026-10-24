"""The R7a evaluation set: instructions with gold GoalSpecs, built without a model.

Gold params are drawn first, then rendered to text from phrase banks, so the label is
the generator's own and never a model's. Two splits per domain:

- ``indist``: plain templated phrasings close in style to the few-shot examples;
- ``heldout``: other styles (terse, verbose and polite, indirect, numbers and units
  written out, "everyone except ..." negations) whose phrasings appear in no few-shot
  or in-distribution template.

About 10% of the items are invalid or unsatisfiable requests (``gold`` is ``None``): the
expected outcome is a clean refusal or failure.
"""

import random
from dataclasses import dataclass
from typing import Any, Optional

from ..envs.plant.tep import NODES
from .r7a_domains import (
    BYTE_BUDGETS,
    COLOURS,
    PLANT_DEFAULTS,
    PLANT_DIAGNOSE,
    SQUAD_DEFAULTS,
    SQUAD_MISSION,
    TARGET_FARS,
)

SPLITS = ("indist", "heldout")


@dataclass(frozen=True)
class EvalItem:
    """One instruction with its gold label.

    Attributes:
        id: Stable id, e.g. ``squad-heldout-07``
        domain: ``squad`` or ``plant``
        split: ``indist`` or ``heldout``
        style: Phrasing style tag
        text: The instruction
        gold: Gold params, or None when the request must be refused
    """

    id: str
    domain: str
    split: str
    style: str
    text: str
    gold: Optional[dict[str, Any]]

    @property
    def spec_name(self) -> str:
        """Spec the item targets."""
        return SQUAD_MISSION if self.domain == "squad" else PLANT_DIAGNOSE

    @property
    def invalid(self) -> bool:
        """Whether the expected outcome is a refusal."""
        return self.gold is None


def _join(names: list[str], word: str = "and") -> str:
    return (
        names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" {word} " + names[-1]
    )


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_TEENS = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
          16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen"}  # fmt: skip
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}  # fmt: skip


def words(n: int) -> str:
    """English words for 0 <= n < 1000 ('one hundred twenty', 'fifty-one')."""
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + (f"-{_ONES[o]}" if o else "")
    h, rest = divmod(n, 100)
    return f"{_ONES[h]} hundred" + (f" {words(rest)}" if rest else "")


# ---------------------------------------------------------------------- squad

STEP_CHOICES = (50, 60, 75, 80, 90, 100, 120, 150, 200, 250, 300)
TARGET_CHOICES = (5, 8, 10, 12, 15, 20, 24, 25, 30)
EATER_SETS = [
    ["red"], ["green"], ["blue"], ["red", "green"], ["red", "blue"], ["green", "blue"],
    ["red", "green", "blue"],
]  # fmt: skip


def _squad_params(rng: random.Random) -> tuple[dict[str, Any], dict[str, bool]]:
    eaters = rng.choice(EATER_SETS)
    show = {
        "caution": rng.random() < 0.65,
        "step_limit": rng.random() < 0.7,
        "food_target": rng.random() < 0.55,
    }
    if not any(show.values()):
        show["step_limit"] = True
    params = {
        "eaters": list(eaters),
        "caution": (
            rng.choice(["low", "normal", "high"])
            if show["caution"]
            else SQUAD_DEFAULTS["caution"]
        ),
        "step_limit": (
            rng.choice(STEP_CHOICES)
            if show["step_limit"]
            else SQUAD_DEFAULTS["step_limit"]
        ),
        "food_target": (
            rng.choice(TARGET_CHOICES)
            if show["food_target"]
            else SQUAD_DEFAULTS["food_target"]
        ),
    }
    if show["caution"] and params["caution"] == "normal":
        pass  # "normal caution" is stated explicitly
    return params, show


def _squad_eaters_indist(rng: random.Random, eaters: list[str]) -> str:
    if len(eaters) == 3:
        return rng.choice(
            [
                "everyone forages",
                "the whole squad collects food",
                "all three snakes go eat",
            ]
        )
    who = _join(eaters)
    return rng.choice(
        [f"{who} forage", f"send {who} out to eat", f"{who} should collect food"]
    )


def _squad_eaters_heldout(rng: random.Random, eaters: list[str]) -> str:
    if len(eaters) == 3:
        return rng.choice(
            [
                "every last snake should go and find something to eat",
                "I want the full squad out foraging",
                "no snake sits idle, all of them feed",
            ]
        )
    if len(eaters) == 2 and rng.random() < 0.5:
        other = [c for c in COLOURS if c not in eaters][0]
        return rng.choice(
            [
                f"everyone but {other} goes foraging",
                f"all snakes except {other} should feed",
            ]
        )
    who = _join(eaters)
    return rng.choice(
        [
            f"it would be good if {who} could go and find some food",
            f"I'd like {who} to head out and graze",
            f"the job of {who} is to hunt for food",
            f"{who} on food duty",
        ]
    )


_CAUTION_A = {
    "low": ["be aggressive", "take risks", "play recklessly"],
    "normal": ["normal caution", "standard care"],
    "high": ["be careful", "play it safe", "stay cautious"],
}
_CAUTION_B = {
    "low": ["losses are fine so go all out", "don't hold back, risks are fine", "daring play is welcome", "bold moves please"],
    "normal": ["nothing special in terms of caution", "an ordinary level of care will do", "moderate care"],
    "high": ["nobody should get hurt", "losing a snake would be a disaster", "avoid any danger", "treat the walls with respect and avoid losses"],
}  # fmt: skip


def _squad_steps(rng: random.Random, n: int, style: str) -> str:
    if style == "indist":
        return rng.choice([f"for {n} steps", f"limit {n} steps", f"{n} steps max"])
    forms = [
        f"for {words(n)} steps",
        f"you have {words(n)} moves",
        f"stop after {n} ticks",
        f"within {words(n)} ticks",
        f"cap the run at {n} moves",
    ]
    return rng.choice(forms)


def _squad_target(rng: random.Random, n: int, style: str) -> str:
    if style == "indist":
        return rng.choice(
            [
                f"until the team has eaten {n} food",
                f"target {n} food",
                f"stop after {n} pieces of food",
            ]
        )
    forms = [
        f"we are done once {words(n)} items have been eaten",
        f"aim for {words(n)} pieces of food in total",
        f"the squad needs {n} bites between them",
    ]
    if n == 12:
        forms.append("a dozen in total")
    if n == 24:
        forms.append("two dozen items")
    return rng.choice(forms)


def _squad_text(
    rng: random.Random, params: dict[str, Any], show: dict[str, bool], split: str
) -> tuple[str, str]:
    caution_bank = _CAUTION_A if split == "indist" else _CAUTION_B
    eat = (_squad_eaters_indist if split == "indist" else _squad_eaters_heldout)(
        rng, params["eaters"]
    )
    parts = []
    if show["caution"]:
        parts.append(rng.choice(caution_bank[params["caution"]]))
    if show["step_limit"]:
        parts.append(_squad_steps(rng, params["step_limit"], split))
    if show["food_target"]:
        parts.append(_squad_target(rng, params["food_target"], split))
    rng.shuffle(parts)
    if split == "indist":
        style = rng.choice(["plain", "mission"])
        if style == "mission":
            return f"Mission: {eat}, " + ", ".join(parts) + ".", style
        return f"{eat.capitalize()}; " + "; ".join(parts) + ".", style
    style = rng.choice(["terse", "verbose", "indirect"])
    if style == "terse":
        short = ", ".join(parts)
        who = "+".join(params["eaters"]) if len(params["eaters"]) < 3 else "all"
        return f"{who}: {short}".strip(": ") + ".", style
    if style == "verbose":
        tail = (", and " + ", and ".join(parts)) if parts else ""
        return (
            f"Dear control, if it is not too much trouble, please arrange for the following: {eat}{tail}. Thank you very much.",
            style,
        )
    tail = (" Also, " + "; ".join(parts) + ".") if parts else ""
    return f"Here is what I am hoping for today: {eat}.{tail}", style


SQUAD_INVALID = {
    "indist": [
        "Send the yellow snake to eat for 100 steps.",
        "Have red eat 500 food within 20 steps.",
        "Write me a poem about snakes.",
        "Blue should forage, but also no snake may eat anything.",
    ],
    "heldout": [
        "Could the purple one and the red one grab some food? Cap the run at one thousand ticks.",
        "Please order a large pizza for the control room.",
        "I would like the squad to run for ten moves, tops.",
        "Green forages and green does not forage; make it so.",
    ],
}


# ---------------------------------------------------------------------- plant

NODE_SETS = [
    [n] for n in NODES
] + [
    ["reactor", "stripper"], ["separator", "compressor_utilities"], ["reactor", "separator"],
    ["stripper", "compressor_utilities"], ["reactor", "separator", "stripper"],
    ["separator", "stripper", "compressor_utilities"], list(NODES),
]  # fmt: skip
_NODE_NAMES_A = {
    "reactor": "reactor", "separator": "separator", "stripper": "stripper",
    "compressor_utilities": "compressor utilities",
}  # fmt: skip
_NODE_NAMES_B = {
    "reactor": "reactor", "separator": "separator", "stripper": "stripper",
    "compressor_utilities": "compressor and utilities node",
}  # fmt: skip


def _plant_params(rng: random.Random) -> tuple[dict[str, Any], dict[str, bool]]:
    show = {
        "byte_budget": rng.random() < 0.85,
        "target_far": rng.random() < 0.8,
        "active_nodes": rng.random() < 0.75,
    }
    if not any(show.values()):
        show["byte_budget"] = True
    nodes = (
        list(rng.choice(NODE_SETS))
        if show["active_nodes"]
        else list(PLANT_DEFAULTS["active_nodes"])
    )
    if show["active_nodes"]:
        nodes = [n for n in NODES if n in nodes]
    params = {
        "byte_budget": (
            rng.choice(BYTE_BUDGETS)
            if show["byte_budget"]
            else PLANT_DEFAULTS["byte_budget"]
        ),
        "target_far": (
            rng.choice(TARGET_FARS)
            if show["target_far"]
            else PLANT_DEFAULTS["target_far"]
        ),
        "active_nodes": nodes,
    }
    return params, show


_PCT = {0.001: "0.1", 0.005: "0.5", 0.01: "1", 0.02: "2", 0.05: "5"}
_FAR_WORDS = {
    0.001: ["a tenth of a percent", "one in a thousand"],
    0.005: ["half a percent", "one in two hundred", "zero point five percent"],
    0.01: ["one percent", "one in a hundred", "a hundredth"],
    0.02: ["two percent", "two in a hundred", "one in fifty"],
    0.05: ["five percent", "one in twenty", "five in a hundred"],
}


def _plant_budget(rng: random.Random, n: int, split: str) -> str:
    if split == "indist":
        return rng.choice(
            [f"{n} bytes per message", f"a budget of {n} bytes", f"{n}-byte uplinks"]
        )
    return rng.choice(
        [
            f"each node may send {words(n)} bytes",
            f"messages capped at {words(n)} octets",
            f"payloads of {words(n)} bytes at most",
            f"a {n} B payload",
        ]
    )


def _plant_far(rng: random.Random, far: float, split: str) -> str:
    if split == "indist":
        return rng.choice(
            [
                f"false alarm rate {_PCT[far]}%",
                f"target {_PCT[far]}% false alarms",
                f"keep false alarms at {_PCT[far]}%",
            ]
        )
    return rng.choice(
        [
            f"false alarms should stay near {rng.choice(_FAR_WORDS[far])}",
            f"tolerate roughly {rng.choice(_FAR_WORDS[far])} of normal readings raising an alarm",
            f"the alarm budget for normal operation is {rng.choice(_FAR_WORDS[far])}",
        ]
    )


def _plant_nodes(rng: random.Random, nodes: list[str], split: str) -> str:
    names = _NODE_NAMES_A if split == "indist" else _NODE_NAMES_B
    if len(nodes) == 4:
        if split == "indist":
            return rng.choice(["all four nodes", "use every node"])
        return rng.choice(
            ["the whole sensor network takes part", "no node is left out"]
        )
    if len(nodes) == 1:
        n = names[nodes[0]]
        if split == "indist":
            return rng.choice([f"only the {n}", f"{n} node alone"])
        return rng.choice(
            [f"rely on nothing but the {n}", f"everything from the {n} and nobody else"]
        )
    if len(nodes) == 3 and split == "heldout":
        out = [n for n in NODES if n not in nodes][0]
        return rng.choice(
            [f"all nodes bar the {names[out]}", f"everything except the {names[out]}"]
        )
    who = _join([names[n] for n in nodes])
    if split == "indist":
        return rng.choice([f"use the {who}", f"nodes: {who}"])
    return rng.choice(
        [f"the {who} should do the reporting", f"let the {who} carry the load"]
    )


def _plant_text(
    rng: random.Random, params: dict[str, Any], show: dict[str, bool], split: str
) -> tuple[str, str]:
    parts = []
    if show["byte_budget"]:
        parts.append(_plant_budget(rng, params["byte_budget"], split))
    if show["target_far"]:
        parts.append(_plant_far(rng, params["target_far"], split))
    if show["active_nodes"]:
        parts.append(_plant_nodes(rng, params["active_nodes"], split))
    rng.shuffle(parts)
    if split == "indist":
        style = rng.choice(["plain", "config"])
        if style == "config":
            return "Diagnosis setup: " + ", ".join(parts) + ".", style
        return (
            parts[0].capitalize()
            + (", " + ", ".join(parts[1:]) if parts[1:] else "")
            + "."
        ), style
    style = rng.choice(["terse", "verbose", "indirect"])
    if style == "terse":
        return " / ".join(parts) + ".", style
    if style == "verbose":
        return (
            "Good morning. For the fault diagnosis run, I would like you to configure things as follows: "
            + "; and ".join(parts)
            + ". Please confirm when done.",
            style,
        )
    return (
        "Thinking about the plant diagnosis, I suppose "
        + ", and also ".join(parts)
        + ", if that is possible.",
        style,
    )


PLANT_INVALID = {
    "indist": [
        "Use 500 bytes per message.",
        "Set the false alarm rate to 10%.",
        "Please add the furnace node to the diagnosis run.",
        "What is the weather on the plant floor tomorrow?",
    ],
    "heldout": [
        "Each node may send zero bytes, yet still report every fault.",
        "Tolerate about one in ten normal readings raising an alarm.",
        "Include the fifth node and the reactor.",
        "Please recommend a good restaurant near the plant.",
    ],
}


# ---------------------------------------------------------------------- assembly


def build_evalset(seed: int = 7, per_split: int = 36) -> list[EvalItem]:
    """The full evaluation set: per domain and split, ``per_split`` valid items plus 4 invalid.

    With the default ``per_split`` that is 160 items, 16 of them invalid (10%).
    """
    rng = random.Random(seed)
    items: list[EvalItem] = []
    makers = {
        "squad": (_squad_params, _squad_text, SQUAD_INVALID),
        "plant": (_plant_params, _plant_text, PLANT_INVALID),
    }
    for domain, (draw, render, invalid) in makers.items():
        for split in SPLITS:
            seen: set[str] = set()
            made = 0
            while made < per_split:
                params, show = draw(rng)
                text, style = render(rng, params, show, split)
                if text in seen:
                    continue
                seen.add(text)
                items.append(
                    EvalItem(
                        f"{domain}-{split}-{made:02d}",
                        domain,
                        split,
                        style,
                        text,
                        params,
                    )
                )
                made += 1
            for k, text in enumerate(invalid[split]):
                items.append(
                    EvalItem(
                        f"{domain}-{split}-x{k}", domain, split, "invalid", text, None
                    )
                )
    return items


def few_shot_texts() -> set[str]:
    """Instruction texts of the translation few-shots (they must not appear in the set)."""
    from .r7a_domains import PLANT_REPORT_EXAMPLES, SQUAD_REPORT_EXAMPLES

    return {e.text for e in SQUAD_REPORT_EXAMPLES + PLANT_REPORT_EXAMPLES}
