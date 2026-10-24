"""R7a domains: the squad and the plant floor as targets of one natural-language translator.

Each domain contributes data and glue, never translator code:

- a ``SpecDefinition`` (``squad.mission``, ``plant.diagnose``) registered in one
  ``SpecRegistry``;
- prompts and few-shot examples (``TranslationPack``, ``SummaryPack``);
- an *executor* that honours the GoalSpec by configuring or wrapping the public APIs of
  ``envs.squad`` and ``envs.plant`` (no edits to those packages) and returns a result
  that validates against the definition's result schema;
- the facts a summary of that task must state (``Fact``) for the automatic fidelity check;
- a keyword/regex parser, the non-LLM baseline.

Squad (``squad.mission``): the scripted ``NoCommPolicy`` is wrapped so that colours not
ordered to eat get their own-food window plane blanked (they wander); ``caution`` sets the
wander rate; ``step_limit`` is the episode length; ``food_target`` ends a snake's mission
once the eaters' team has eaten that much. Score: eater food (capped at the target) plus
0.5 per eater still alive at the stop.

Plant floor (``plant.diagnose``): the step-2 hand-designed ``features`` encoder and the
calibrated ``DecodedCentre`` are built at the ordered byte budget and target false-alarm
rate (``PlantContext.target_far``); nodes not ordered active are silenced by zeroing
their uplink messages. Score: macro-F1 over the fault classes. The silencing is applied at
test time only (the centre was trained with all four nodes), a deliberate simplification.
"""

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np

from ..envs.base import Observations
from ..envs.plant.baselines import PlantContext
from ..envs.plant.centres import DecodedCentre, block_subsample
from ..envs.plant.encoders import make_encoder
from ..envs.plant.runner import diagnosis_metrics
from ..envs.plant.tep import NODES, TepData
from ..envs.squad import NoCommPolicy, SquadConfig, SquadEnv
from ..protocol.goalspec import SpecDefinition, SpecRegistry
from ..translator import Example, Fact, SummaryPack, TraceView, TranslationPack

SQUAD_MISSION = "squad.mission"
PLANT_DIAGNOSE = "plant.diagnose"
COLOURS = ("red", "green", "blue")
BYTE_BUDGETS = (8, 16, 51)
TARGET_FARS = (0.001, 0.005, 0.01, 0.02, 0.05)
CAUTIONS = ("low", "normal", "high")
EXPLORE = {"low": 0.3, "normal": 0.1, "high": 0.0}
STEP_RANGE = (50, 300)
FOOD_TARGET_RANGE = (0, 40)
EVAL_SEED_BASE = 2_000_000_000  # Held-out seeds, as in the benchmarks

SQUAD_DEFAULTS = {"caution": "normal", "step_limit": 300, "food_target": 0}
PLANT_DEFAULTS = {
    "byte_budget": 16,
    "target_far": 0.01,
    "active_nodes": list(NODES),
}

_NUMBER = {"type": "number"}


# ---------------------------------------------------------------------- specs


def squad_definition() -> SpecDefinition:
    """``squad.mission`` v1: a parameterised cooperative-foraging mission."""
    params = {
        "type": "object",
        "required": ["eaters", "caution", "step_limit", "food_target"],
        "properties": {
            "eaters": {
                "type": "array",
                "items": {"type": "string", "enum": list(COLOURS)},
                "minItems": 1,
                "maxItems": len(COLOURS),
                "uniqueItems": True,
            },
            "caution": {"type": "string", "enum": list(CAUTIONS)},
            "step_limit": {
                "type": "integer",
                "minimum": STEP_RANGE[0],
                "maximum": STEP_RANGE[1],
            },
            "food_target": {
                "type": "integer",
                "minimum": FOOD_TARGET_RANGE[0],
                "maximum": FOOD_TARGET_RANGE[1],
            },
        },
        "additionalProperties": False,
    }
    result = {
        "type": "object",
        "required": ["score", "food", "survivors", "steps", "per_seed"],
        "properties": {
            "score": _NUMBER,
            "food": _NUMBER,
            "survivors": _NUMBER,
            "steps": _NUMBER,
            "per_seed": {"type": "array", "items": _NUMBER},
        },
    }
    return SpecDefinition(
        SQUAD_MISSION,
        1,
        params,
        "Order the foraging squad: which colours forage, how cautious, for how long, "
        "and a food target",
        result,
    )


def plant_definition() -> SpecDefinition:
    """``plant.diagnose`` v1: a parameterised fault-diagnosis configuration."""
    params = {
        "type": "object",
        "required": ["byte_budget", "target_far", "active_nodes"],
        "properties": {
            "byte_budget": {"type": "integer", "enum": list(BYTE_BUDGETS)},
            "target_far": {"type": "number", "enum": list(TARGET_FARS)},
            "active_nodes": {
                "type": "array",
                "items": {"type": "string", "enum": list(NODES)},
                "minItems": 1,
                "maxItems": len(NODES),
                "uniqueItems": True,
            },
        },
        "additionalProperties": False,
    }
    result = {
        "type": "object",
        "required": ["score", "false_alarm_rate", "detection_rate", "per_seed"],
        "properties": {
            "score": _NUMBER,
            "false_alarm_rate": _NUMBER,
            "detection_rate": _NUMBER,
            "bytes_per_node": _NUMBER,
            "per_seed": {"type": "array", "items": _NUMBER},
        },
    }
    return SpecDefinition(
        PLANT_DIAGNOSE,
        1,
        params,
        "Configure the plant-floor sensor squad: uplink bytes, false-alarm rate, nodes",
        result,
    )


def build_registry(backend: str = "auto") -> SpecRegistry:
    """A registry holding both R7a specs."""
    registry = SpecRegistry(backend=backend)
    registry.register(squad_definition())
    registry.register(plant_definition())
    return registry


# ---------------------------------------------------------------------- executors


def _json_float(x: float, digits: int = 4) -> float:
    return round(float(x), digits)


class SquadMissionExecutor:
    """Runs ``squad.mission`` params on the scripted squad (results cached by params).

    Args:
        n_seeds: Episodes per mission (seeds ``EVAL_SEED_BASE + k``)
        config: Base squad configuration (``max_steps`` is replaced by ``step_limit``)
    """

    def __init__(self, n_seeds: int = 8, config: Optional[SquadConfig] = None):
        self.seeds = [EVAL_SEED_BASE + k for k in range(n_seeds)]
        self.config = config or SquadConfig()
        self._cache: dict[str, dict[str, Any]] = {}
        self.runs = 0

    def __call__(self, params: Mapping[str, Any]) -> dict[str, Any]:
        key = json.dumps({**params, "eaters": sorted(params["eaters"])}, sort_keys=True)
        if key not in self._cache:
            self._cache[key] = self._run(params)
            self.runs += 1
        return json.loads(json.dumps(self._cache[key]))

    def _run(self, params: Mapping[str, Any]) -> dict[str, Any]:
        eaters = sorted(COLOURS.index(c) for c in params["eaters"])
        wander = [i for i in range(len(COLOURS)) if i not in eaters]
        target = int(params["food_target"])
        cfg = replace(self.config, max_steps=int(params["step_limit"]))
        env = SquadEnv(cfg)
        policy = NoCommPolicy(explore=EXPLORE[params["caution"]])
        obs = env.reset(self.seeds)
        policy.reset(self.seeds)
        b = len(self.seeds)
        stopped = np.zeros(b, dtype=bool)
        stop_food = np.zeros(b)
        stop_alive = np.zeros(b)
        stop_step = np.zeros(b)
        while not env.done.all():
            windows = obs.specialists["window"]
            if wander:
                windows = windows.copy()
                windows[:, wander, 2] = 0  # blind to own-colour food: they just wander
            seen = Observations(
                specialists={**obs.specialists, "window": windows}, centre=obs.centre
            )
            obs, _, done, _ = env.step(policy.act(seen))
            if target > 0:
                food = env.score[:, eaters].sum(axis=1)
                hit = (food >= target) & ~stopped
                if hit.any():
                    stop_food[hit] = food[hit]
                    stop_alive[hit] = env.alive[:, eaters].sum(axis=1)[hit]
                    stop_step[hit] = env.steps[hit]
                    stopped |= hit
        final_food = env.score[:, eaters].sum(axis=1).astype(float)
        final_alive = env.alive[:, eaters].sum(axis=1).astype(float)
        food = np.where(stopped, stop_food, final_food)
        alive = np.where(stopped, stop_alive, final_alive)
        steps = np.where(stopped, stop_step, env.steps.astype(float))
        if target > 0:
            food = np.minimum(food, target)
        score = food + 0.5 * alive
        return {
            "score": _json_float(score.mean()),
            "food": _json_float(food.mean()),
            "survivors": _json_float(alive.mean()),
            "steps": _json_float(steps.mean()),
            "per_seed": [_json_float(s) for s in score],
        }


class PlantDiagnoseExecutor:
    """Runs ``plant.diagnose`` params on the calibrated feature encoder and centre.

    Args:
        data: TEP runs (``TepData.from_cache()`` for the real data)
        n_seeds: Fit-and-replay repeats per configuration (each a different 80% block
            subsample of the training split)
        encoding: Hand-designed encoder family (``features`` by default)
    """

    def __init__(self, data: TepData, n_seeds: int = 5, encoding: str = "features"):
        self.data = data
        self.seeds = list(range(n_seeds))
        self.encoding = encoding
        self._contexts: dict[float, PlantContext] = {}
        self._fits: dict[
            tuple[float, int, tuple[int, ...], int], tuple[Any, Any, Any]
        ] = {}
        self._cache: dict[str, dict[str, Any]] = {}
        self.runs = 0

    def _context(self, far: float) -> PlantContext:
        if far not in self._contexts:
            self._contexts[far] = PlantContext(self.data, target_far=far)
        return self._contexts[far]

    def __call__(self, params: Mapping[str, Any]) -> dict[str, Any]:
        key = json.dumps(
            {**params, "active_nodes": sorted(params["active_nodes"])}, sort_keys=True
        )
        if key not in self._cache:
            self._cache[key] = self._run(params)
            self.runs += 1
        return json.loads(json.dumps(self._cache[key]))

    def _fit(
        self, far: float, budget: int, silent: tuple[int, ...], seed: int
    ) -> tuple[Any, Any, Any]:
        """Encoder and calibrated centre trained with the silent nodes' windows zeroed."""
        key = (far, budget, silent, seed)
        if key in self._fits:
            return self._fits[key]
        ctx = self._context(far)
        rng = np.random.default_rng(10_000 + seed)  # same subsample as PlantContext.fit
        keep = block_subsample(ctx.train["run"], ctx.fraction, rng, ctx.block)
        windows = ctx.train["windows"][keep].copy()
        if silent:
            windows[:, list(silent)] = 0.0
        labels, run = ctx.train["labels"][keep], ctx.train["run"][keep]
        encoder = make_encoder(self.encoding, budget, ctx.layout, seed).fit(
            windows[labels == 0]
        )
        centre = DecodedCentre(
            encoder, ctx.n_classes, ctx.classifier, seed,
            target_far=far, gap=ctx.gap, l2=ctx.l2,
        ).fit(windows, labels, run)  # fmt: skip
        self._fits[key] = (ctx.env(budget), _Silenced(encoder, silent), centre)
        return self._fits[key]

    def _run(self, params: Mapping[str, Any]) -> dict[str, Any]:
        far, budget = float(params["target_far"]), int(params["byte_budget"])
        silent = tuple(
            i for i, n in enumerate(NODES) if n not in params["active_nodes"]
        )
        ctx = self._context(far)
        scores, fars, dets, sent = [], [], [], []
        for seed in self.seeds:
            env, encoder, centre = self._fit(far, budget, silent, seed)
            res = ctx.rollout(env, encoder, centre)
            m = diagnosis_metrics(res, ctx.n_classes)
            scores.append(m["macro_f1"])
            fars.append(m["false_alarm_rate"])
            dets.append(m["detection_rate"])
            sent.append(
                float(res["bytes_per_node"]) * (len(NODES) - len(silent)) / len(NODES)
            )
        return {
            "score": _json_float(np.mean(scores)),
            "false_alarm_rate": _json_float(np.nanmean(fars)),
            "detection_rate": _json_float(np.nanmean(dets)),
            "bytes_per_node": _json_float(np.mean(sent), 2),
            "per_seed": [_json_float(s) for s in scores],
        }


class _Silenced:
    """An encoder behind silenced nodes: their windows read as zeros (the normal mean)."""

    def __init__(self, encoder: Any, silent: tuple[int, ...]):
        self.encoder, self.silent = encoder, silent

    def encode(self, specialist_obs: dict[str, np.ndarray]) -> np.ndarray:
        windows = specialist_obs["window"]
        if self.silent:
            windows = windows.copy()
            windows[:, list(self.silent)] = 0.0
        return self.encoder.encode({**specialist_obs, "window": windows})

    def __getattr__(self, name: str) -> Any:
        return getattr(self.encoder, name)


# ---------------------------------------------------------------------- packs

SQUAD_SYSTEM = """You turn an order for a squad of three foraging snakes (red, green, blue) into JSON.
Reply with {"refusal": "", "params": {...}}.
Params:
- eaters: the colours ordered to forage. Snakes not ordered to eat just wander. "Everyone", "the whole squad" or "all snakes" means all three.
- caution: "low" (take risks, aggressive, reckless), "normal" (not mentioned, or normal care), "high" (careful, safe, avoid losses).
- step_limit: integer 50..300, how many steps the mission lasts. 300 when no limit is mentioned.
- food_target: integer 0..40, the team food count that completes the mission. 0 when no target is mentioned.
Refuse (give a short reason in "refusal" and omit "params") when the order names a colour other than red, green or blue, asks for a number outside the ranges above, contradicts itself, or is not an order for the squad."""

SQUAD_EXAMPLES = (
    Example(
        "Red and green forage, play it safe, 120 steps.",
        {
            "eaters": ["red", "green"],
            "caution": "high",
            "step_limit": 120,
            "food_target": 0,
        },
    ),
    Example(
        "Send blue out to eat until the team has 15 food. Be aggressive.",
        {"eaters": ["blue"], "caution": "low", "step_limit": 300, "food_target": 15},
    ),
    Example(
        "The whole squad should collect food for 200 steps.",
        {
            "eaters": ["red", "green", "blue"],
            "caution": "normal",
            "step_limit": 200,
            "food_target": 0,
        },
    ),
    Example(
        "Green eats, target 8 food, careful, limit 90 steps.",
        {"eaters": ["green"], "caution": "high", "step_limit": 90, "food_target": 8},
    ),
    Example(
        "Send the orange snake to eat for 100 steps.",
        None,
        "there is no orange snake; the colours are red, green and blue",
    ),
    Example("Tell me a joke.", None, "this is not an order for the squad"),
)

PLANT_SYSTEM = """You turn a request about a plant-floor fault-diagnosis system into JSON.
Four sensor nodes (reactor, separator, stripper, compressor_utilities) send small byte messages to a gateway that diagnoses process faults.
Reply with {"refusal": "", "params": {...}}.
Params:
- byte_budget: bytes per node message; exactly 8, 16 or 51. 16 when not mentioned.
- target_far: the false-alarm rate as a fraction of normal readings, exactly one of 0.001, 0.005, 0.01, 0.02, 0.05. Convert percentages and ratios: 0.1% or one in a thousand is 0.001, 0.5% is 0.005, 1% or one in a hundred is 0.01, 2% is 0.02, 5% or one in twenty is 0.05. 0.01 when not mentioned.
- active_nodes: the nodes that may transmit. All four when not mentioned. "Compressor", "utilities" and "compressor utilities" all mean compressor_utilities.
Refuse (give a short reason in "refusal" and omit "params") when the request asks for a byte budget or false-alarm rate that is not one of the allowed values, names a node that does not exist, or is not about this system."""

PLANT_EXAMPLES = (
    Example(
        "Use 8 bytes per message with a 1% false alarm rate, reactor and stripper only.",
        {"byte_budget": 8, "target_far": 0.01, "active_nodes": ["reactor", "stripper"]},
    ),
    Example(
        "All four nodes, 51 bytes, keep false alarms to half a percent.",
        {"byte_budget": 51, "target_far": 0.005, "active_nodes": list(NODES)},
    ),
    Example(
        "Run the diagnosis from the separator node alone at 2% false alarms.",
        {"byte_budget": 16, "target_far": 0.02, "active_nodes": ["separator"]},
    ),
    Example(
        "Give every node a 300 byte budget.",
        None,
        "300 bytes is not an allowed budget (8, 16 or 51)",
    ),
    Example(
        "Add the boiler node to the diagnosis.",
        None,
        "there is no boiler node",
    ),
    Example(
        "What is the capital of France?",
        None,
        "this is not a request about the plant system",
    ),
)

SQUAD_SUMMARY_SYSTEM = """You write the status report of a finished squad mission for a human monitor, from a JSON trace.
Write one or two plain sentences. State: which colours were ordered to forage (name only those colours, no others), "caution level <low|normal|high>", "step limit <n>", the food target ("food target <n>" or "no food target"), and the outcome numbers exactly as given in the trace ("mission score <x>")."""

SQUAD_SUMMARY_EXAMPLES = (
    (
        json.dumps(
            {
                "goal": {
                    "spec": SQUAD_MISSION,
                    "params": {
                        "caution": "high",
                        "eaters": ["red", "blue"],
                        "food_target": 0,
                        "step_limit": 120,
                    },
                },
                "messages": [
                    {
                        "sender": "command-centre",
                        "recipients": ["squad-lead"],
                        "payload_type": "json",
                        "codec": "goalspec",
                        "clock": 1,
                    },
                    {
                        "sender": "squad-lead",
                        "recipients": ["command-centre"],
                        "payload_type": "json",
                        "codec": "result",
                        "clock": 2,
                    },
                ],
                "outcome": {
                    "state": "completed",
                    "score": 6.3,
                    "food": 5.4,
                    "survivors": 1.8,
                    "steps": 120.0,
                },
            },
            sort_keys=True,
        ),
        "The red and blue snakes were ordered to forage at caution level high with a step limit of 120 and no food target; the squad finished with mission score 6.3.",
    ),
    (
        json.dumps(
            {
                "goal": {
                    "spec": SQUAD_MISSION,
                    "params": {
                        "caution": "low",
                        "eaters": ["green"],
                        "food_target": 10,
                        "step_limit": 250,
                    },
                },
                "messages": [
                    {
                        "sender": "command-centre",
                        "recipients": ["squad-lead"],
                        "payload_type": "json",
                        "codec": "goalspec",
                        "clock": 1,
                    },
                    {
                        "sender": "squad-lead",
                        "recipients": ["command-centre"],
                        "payload_type": "json",
                        "codec": "result",
                        "clock": 2,
                    },
                ],
                "outcome": {
                    "state": "completed",
                    "score": 8.9,
                    "food": 8.1,
                    "survivors": 1.6,
                    "steps": 250.0,
                },
            },
            sort_keys=True,
        ),
        "The green snake was ordered to forage at caution level low, with a step limit of 250 and a food target of 10; the squad finished with mission score 8.9.",
    ),
)

PLANT_SUMMARY_SYSTEM = """You write the status report of a finished plant-floor diagnosis run for a human monitor, from a JSON trace.
Write one or two plain sentences. State: the byte budget ("<n> bytes per message"), the target false-alarm rate as a percentage ("<p>% false alarms"), the active nodes (name only the active nodes, spelled as in the trace), and the outcome numbers exactly as given ("macro-F1 <x>")."""

PLANT_SUMMARY_EXAMPLES = (
    (
        json.dumps(
            {
                "goal": {
                    "spec": PLANT_DIAGNOSE,
                    "params": {
                        "active_nodes": ["reactor", "stripper"],
                        "byte_budget": 8,
                        "target_far": 0.01,
                    },
                },
                "messages": [
                    {
                        "sender": "command-centre",
                        "recipients": ["plant-gateway"],
                        "payload_type": "json",
                        "codec": "goalspec",
                        "clock": 1,
                    },
                    {
                        "sender": "plant-gateway",
                        "recipients": ["command-centre"],
                        "payload_type": "json",
                        "codec": "result",
                        "clock": 2,
                    },
                ],
                "outcome": {
                    "state": "completed",
                    "score": 0.61,
                    "false_alarm_rate": 0.05,
                    "detection_rate": 0.8,
                    "bytes_per_node": 4.0,
                },
            },
            sort_keys=True,
        ),
        "The diagnosis ran with 8 bytes per message and 1% false alarms on the reactor and stripper nodes, reaching macro-F1 0.61.",
    ),
    (
        json.dumps(
            {
                "goal": {
                    "spec": PLANT_DIAGNOSE,
                    "params": {
                        "active_nodes": [
                            "separator",
                            "compressor_utilities",
                            "reactor",
                            "stripper",
                        ],
                        "byte_budget": 51,
                        "target_far": 0.005,
                    },
                },
                "messages": [
                    {
                        "sender": "command-centre",
                        "recipients": ["plant-gateway"],
                        "payload_type": "json",
                        "codec": "goalspec",
                        "clock": 1,
                    },
                    {
                        "sender": "plant-gateway",
                        "recipients": ["command-centre"],
                        "payload_type": "json",
                        "codec": "result",
                        "clock": 2,
                    },
                ],
                "outcome": {
                    "state": "completed",
                    "score": 0.83,
                    "false_alarm_rate": 0.03,
                    "detection_rate": 0.93,
                    "bytes_per_node": 51.0,
                },
            },
            sort_keys=True,
        ),
        "With 51 bytes per message and 0.5% false alarms, all four nodes (reactor, separator, stripper, compressor_utilities) reached macro-F1 0.83.",
    ),
)


REPORT_NOTE = """
The input may also be a status report of an order that was already carried out. Then give the parameters of that order, and ignore result numbers such as scores."""

SQUAD_REPORT_EXAMPLES = SQUAD_EXAMPLES + (
    Example(
        "The green and blue snakes were ordered to forage at caution level normal with a step limit of 150 and no food target; the squad finished with mission score 5.2.",
        {
            "eaters": ["green", "blue"],
            "caution": "normal",
            "step_limit": 150,
            "food_target": 0,
        },
    ),
    Example(
        "The red snake was ordered to forage at caution level low, with a step limit of 80 and a food target of 6; the squad finished with mission score 3.4.",
        {"eaters": ["red"], "caution": "low", "step_limit": 80, "food_target": 6},
    ),
)

PLANT_REPORT_EXAMPLES = PLANT_EXAMPLES + (
    Example(
        "The diagnosis used 16 bytes per message and 2% false alarms on the separator and stripper nodes, achieving macro-F1 0.55.",
        {
            "byte_budget": 16,
            "target_far": 0.02,
            "active_nodes": ["separator", "stripper"],
        },
    ),
    Example(
        "With 8 bytes per message and 0.1% false alarms, the reactor, separator, stripper, compressor_utilities nodes reached macro-F1 0.40.",
        {"byte_budget": 8, "target_far": 0.001, "active_nodes": list(NODES)},
    ),
)


# ---------------------------------------------------------------------- facts


def _num(x: float) -> str:
    """Regex for a number that may be written with or without trailing zeros."""
    s = f"{x:.10g}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    pat = re.escape(s)
    if "." not in s:
        pat += r"(?:\.0+)?"
    return rf"(?<![\d.]){pat}(?![\d])"


def squad_facts(view: TraceView) -> list[Fact]:
    """What a summary of a squad mission must state."""
    p, out = view.goal.params, view.outcome
    eaters = set(p["eaters"])
    lookahead = "".join(rf"(?=.*\b{c}\b)" for c in sorted(eaters))
    level = p["caution"]
    facts = [
        Fact(
            "eaters",
            ("(?s)^" + lookahead,),
            tuple(rf"\b{c}\b" for c in COLOURS if c not in eaters),
        ),
        Fact(
            "caution",
            (
                rf"caution[^.,;]{{0,20}}\b{level}\b",
                rf"\b{level}\b[^.,;]{{0,12}}caution",
            ),
        ),
        Fact("step_limit", (_num(p["step_limit"]),)),
    ]
    if p["food_target"] > 0:
        facts.append(Fact("food_target", (_num(p["food_target"]),)))
    else:
        facts.append(
            Fact(
                "food_target",
                (r"\bno\b[^.;]{0,15}target", r"target[^.,;]{0,12}\b(none|0|zero)\b"),
            )
        )
    facts.append(Fact("score", (_num(out["score"]),)))
    return facts


def _node_pattern(node: str) -> str:
    return (
        r"compressor[ _/-]?(and )?utilities"
        if node == "compressor_utilities"
        else rf"\b{node}\b"
    )


def plant_facts(view: TraceView) -> list[Fact]:
    """What a summary of a plant diagnosis run must state."""
    p, out = view.goal.params, view.outcome
    active = list(p["active_nodes"])
    pct = p["target_far"] * 100
    lookahead = "".join(f"(?=.*{_node_pattern(n)})" for n in active)
    inactive = tuple(_node_pattern(n) for n in NODES if n not in active)
    return [
        Fact("byte_budget", (rf"(?<!\d){p['byte_budget']}[- ]?bytes?\b",)),
        Fact(
            "target_far",
            (rf"{_num(pct)}\s?(%|percent)", rf"{_num(p['target_far'])}"),
        ),
        Fact("active_nodes", ("(?s)^" + lookahead,), inactive),
        Fact("score", (_num(out["score"]),)),
    ]


# ---------------------------------------------------------------------- keyword baselines

_WORD_NUMBERS = {
    "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "hundred": 100, "eight": 8, "sixteen": 16,
}  # fmt: skip
_SQUAD_WORDS = re.compile(
    r"\b(snake|snakes|squad|red|green|blue|everyone|everybody|food|forage|steps?)\b",
    re.I,
)
_PLANT_WORDS = re.compile(
    r"\b(node|nodes|plant|alarm|alarms|byte|bytes|sensor|fault|faults|uplink|reactor|separator|stripper|compressor|utilities)\b",
    re.I,
)


def _int_before(text: str, units: str) -> Optional[int]:
    m = re.search(rf"(\d+)\s*-?\s*(?:{units})\b", text, re.I)
    return int(m.group(1)) if m else None


def keyword_squad(text: str) -> Optional[dict[str, Any]]:
    """Regex parser for ``squad.mission`` (the non-LLM baseline); None means refuse."""
    low = text.lower()
    if not _SQUAD_WORDS.search(low):
        return None
    named = [c for c in COLOURS if re.search(rf"\b{c}\b", low)]
    excluded = [
        c
        for c in named
        if re.search(
            rf"\b(except|but|not|without|besides|bar|other than)\s+(?:the\s+)?{c}\b",
            low,
        )
    ]
    chosen = [c for c in named if c not in excluded]
    if excluded and not chosen:
        chosen = [c for c in COLOURS if c not in excluded]
    if excluded and re.search(r"\b(everyone|everybody|all|squad)\b", low):
        chosen = [c for c in COLOURS if c not in excluded]
    if not chosen:
        chosen = (
            list(COLOURS)
            if re.search(r"\b(everyone|everybody|all|squad|whole|every)\b", low)
            else []
        )
    caution = "normal"
    if re.search(
        r"\b(careful|carefully|cautious|safe|safely|conservative|no risk)\b", low
    ):
        caution = "high"
    elif re.search(
        r"\b(aggressive|aggressively|reckless|recklessly|risk|risks|bold|daring)\b", low
    ):
        caution = "low"
    steps = _int_before(low, "steps?|ticks?|moves?|turns?")
    food = _int_before(low, "foods?|pieces?|items?|apples?|points?")
    return {
        "eaters": chosen,
        "caution": caution,
        "step_limit": steps if steps is not None else 300,
        "food_target": food if food is not None else 0,
    }


def keyword_plant(text: str) -> Optional[dict[str, Any]]:
    """Regex parser for ``plant.diagnose`` (the non-LLM baseline); None means refuse."""
    low = text.lower()
    if not _PLANT_WORDS.search(low):
        return None
    budget = _int_before(low, "bytes?|b")
    pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", low)
    far = float(pct.group(1)) / 100 if pct else 0.01
    aliases = {
        "reactor": r"reactor",
        "separator": r"separator",
        "stripper": r"stripper",
        "compressor_utilities": r"compressor|utilities",
    }
    named = [n for n, pat in aliases.items() if re.search(pat, low)]
    excluded = [
        n
        for n in named
        if re.search(
            rf"\b(except|but|not|without|besides|bar|other than)\s+(?:the\s+)?(?:{aliases[n]})",
            low,
        )
    ]
    chosen = [n for n in named if n not in excluded]
    if excluded:
        chosen = [n for n in NODES if n not in excluded]
    if not chosen:
        chosen = list(NODES)
    return {
        "byte_budget": budget if budget is not None else 16,
        "target_far": round(far, 6),
        "active_nodes": chosen,
    }


# ---------------------------------------------------------------------- domains


@dataclass
class Domain:
    """Everything one domain adds to the shared translator.

    Attributes:
        name: ``squad`` or ``plant``
        spec_name: Spec definition the domain's instructions target
        pack: Forward-translation prompts and few-shots
        summary_pack: Reverse-translation (summary) prompts and few-shots
        report_pack: Forward prompts that also accept a status report (the round trip)
        executor: ``params -> result`` that honours the GoalSpec
        facts: Facts a summary of a finished task must state
        keyword: The regex baseline parser
        specialist: Bus name of the specialist that executes the domain's goals
    """

    name: str
    spec_name: str
    pack: TranslationPack
    summary_pack: SummaryPack
    report_pack: TranslationPack
    executor: Callable[[Mapping[str, Any]], dict[str, Any]]
    facts: Callable[[TraceView], list[Fact]]
    keyword: Callable[[str], Optional[dict[str, Any]]]
    specialist: str
    defaults: Mapping[str, Any] = field(default_factory=dict)


def build_domains(
    plant_data: TepData, squad_seeds: int = 8, plant_seeds: int = 5
) -> dict[str, Domain]:
    """The two R7a domains, keyed by name."""
    return {
        "squad": Domain(
            "squad",
            SQUAD_MISSION,
            TranslationPack(SQUAD_MISSION, SQUAD_SYSTEM, SQUAD_EXAMPLES),
            SummaryPack(SQUAD_SUMMARY_SYSTEM, SQUAD_SUMMARY_EXAMPLES),
            TranslationPack(
                SQUAD_MISSION, SQUAD_SYSTEM + REPORT_NOTE, SQUAD_REPORT_EXAMPLES
            ),
            SquadMissionExecutor(squad_seeds),
            squad_facts,
            keyword_squad,
            "squad-lead",
            SQUAD_DEFAULTS,
        ),
        "plant": Domain(
            "plant",
            PLANT_DIAGNOSE,
            TranslationPack(PLANT_DIAGNOSE, PLANT_SYSTEM, PLANT_EXAMPLES),
            SummaryPack(PLANT_SUMMARY_SYSTEM, PLANT_SUMMARY_EXAMPLES),
            TranslationPack(
                PLANT_DIAGNOSE, PLANT_SYSTEM + REPORT_NOTE, PLANT_REPORT_EXAMPLES
            ),
            PlantDiagnoseExecutor(plant_data, plant_seeds),
            plant_facts,
            keyword_plant,
            "plant-gateway",
            PLANT_DEFAULTS,
        ),
    }


def outcome_of(domain: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """The outcome summary the reverse translator sees (rounded, no per-seed list)."""
    if domain == "squad":
        keys, digits = ("score", "food", "survivors", "steps"), 1
    else:
        keys, digits = (
            "score",
            "false_alarm_rate",
            "detection_rate",
            "bytes_per_node",
        ), 2
    out: dict[str, Any] = {"state": "completed"}
    for k in keys:
        if k in result and not math.isnan(result[k]):
            out[k] = round(float(result[k]), digits)
    return out
