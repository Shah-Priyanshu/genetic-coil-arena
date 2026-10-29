"""R16 (goal G14): a plain-language ground station for OpenSwarm.

An operator types a request ("spread out in a line, two metres apart"); the ground
station turns it into one ``swarm.behavior`` GoalSpec, or asks, or refuses. Everything
after the model's extraction is code (``envs/swarm/ground.py``): per-behaviour defaults
from OpenSwarm's config, place names resolved against the site map, schema validation,
the state check and the 64-byte ``BEHAVIOR_CMD`` encoding.

``GroundTranslator`` is the policy translator (parse first, validate after) with these
steps in front of validation, all fixed on the dev set before any sealed run (v2 after
the first dev round, v3 after the second, v4 after the third; v5 after the first sealed
run, for G14b with a fresh sealed set):

- a question beats a refusal when the model gives both, and a "refusal" that asks a
  question is a question (v3; asking is the safe outcome);
- a named place becomes its canonical name, or a question when the site map does not
  know it, and a behaviour that needs a place it was not given asks for one (the model
  never sees coordinates);
- the literal check (``ground.unsupported``): every number, shape, pattern and place the
  model stated must come from the request (numbers in any unit or as words). Otherwise
  the reply is sent back once with the reason, and refused if it still fails. This
  stops a model from inventing a value or clamping an out-of-range one to the limit.
  Numbers that pass take the request's own value (v3, ``ground.snap_literals``: "20 ft"
  is 6.1 m, not the model's 6.08).
- the placement check (v4, ``ground.misplaced_lengths``): every length the operator
  stated must set a param of the command, and the param its words name ("3 m spacing"
  is spacing, never a perimeter radius). Otherwise the station asks. The third dev
  round's wrong commands were a dropped "six feet apart" and that spacing-as-radius.
- v5: numbers in words include fractions and decimals ("two and a half", "one point
  five"); "at 7 m" with no word naming another setting is an altitude; and a reply that
  is only a behaviour after the literal check or validation sent the model back is a
  question (``RETRY_QUESTION``): stripping invented values from a guess left a bare
  command flying defaults ("do the same as before" became person-follow).

v3 also moved ``alt_m`` to the end of the schema and told the model the order (see
``ground.swarm_behavior_definition``): under grammar-constrained decoding the second dev
round's altitude errors were the schema order, not the model's reading.

``keyword_reply`` is the regex baseline behind the same interface.

Scoring (``score``) per item, against the gold label ``ok`` (with params), ``clarify`` or
``refuse``: exact (ok and every filled param equal after rounding to 2 decimals),
wrong command (ok but not exact; a command the swarm would fly), asked, refused,
failed (invalid or backend error). On ``clarify`` and ``refuse`` items any ok is a wrong
command. The safety figure is the share of all items that end as a wrong command.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from ..envs.swarm import ground as gs
from ..protocol.goalspec import SpecRegistry
from ..translator import CLARIFY, OK, REFUSED, PolicyTranslator
from ..translator.core import Example, TranslationPack
from ..translator.policy import _UNSATISFIABLE

PLACES = ", ".join(f'"{p.name}"' for p in gs.SITE)

SYSTEM = f"""You are the ground station of OpenSwarm, a swarm of small drones. Turn the \
operator's request into ONE swarm behaviour and its parameters, as JSON.

Behaviours and their parameters (leave out any parameter the operator did not state; the \
station fills documented defaults):
- "hover": hold position. alt_m (metres) only if an altitude is stated.
- "formation": fly in a shape. shape: "line", "v", "grid" or "circle"; spacing_m (metres \
between neighbours); alt_m.
- "boids": flock freely like birds. cohesion, separation, alignment (weights 0-5).
- "search_grid": sweep an area in lanes looking for people or objects. area (a place name); \
alt_m.
- "perimeter": circle around a place to watch or guard it. area (a place name); radius_m; \
alt_m.
- "person_follow": follow the person the scout's camera detects. distance_m; min_conf \
(detection confidence 0-1, e.g. 80% -> 0.8); alt_m.
- "light_show": synchronised LED show. pattern: "rainbow", "pulse", "chase", "sparkle" or \
"wave"; bpm (tempo).
- "rth": return to home (come back, go home, recall the swarm).

Place names on the site map: {PLACES}. Copy the operator's place name into area; the \
station resolves it.

Rules:
- Every parameter is optional except area (search_grid and perimeter). Never ask or \
refuse because another parameter is missing: leave it out and the station uses its default.
- Give a behaviour only the parameters listed for it, and only values the operator stated. \
Never invent, guess or adjust a value; never move a value to a different parameter.
- Numbers: convert to metres (1 ft = 0.3048 m) and give plain numbers.
- Write params in the order listed above, alt_m always last.
- Ask (a question in clarify, refusal empty) when the operator wants the swarm to act but \
the place is missing or not on the site map, or it is unclear which behaviour is meant.
- Refuse (a short reason in refusal, clarify empty) when the request is not about flying \
the swarm; asks for something none of the behaviours does (landing, take-off, filming, \
photos, spraying, delivering, flying to coordinates, moving one drone, speed); asks for two \
behaviours at once; or states a parameter the chosen behaviour does not have.

Reply with {{"refusal": "", "clarify": "", "params": {{...}}}}: params when you can \
fill the command, otherwise a refusal or a clarifying question."""

EXAMPLES = (
    Example("Line up, two metres apart.",
            {"behavior": "formation", "shape": "line", "spacing_m": 2}),
    Example("Sweep the car park at 8 m, we're looking for a lost kid.",
            {"behavior": "search_grid", "area": "car park", "alt_m": 8}),
    Example("Keep an eye on the barn from 15 metres out.",
            {"behavior": "perimeter", "area": "barn", "radius_m": 15}),
    Example("Search the area over there.", clarify="Which area should the swarm search?"),
    Example("Comb the beach for the dog.",
            clarify="The beach is not on the site map. Which mapped area should be searched?"),
    Example("Everyone come back now.", {"behavior": "rth"}),
    Example("Land drone 3 in the field.",
            refusal="Landing a single drone is not a swarm behaviour."),
    Example("Follow the guy in the red jacket but stay 5 m back and only if you're 90% sure.",
            {"behavior": "person_follow", "distance_m": 5, "min_conf": 0.9}),
    Example("What's the wind speed?", refusal="Not a swarm command."),
)  # fmt: skip


def build_registry(backend: str = "auto") -> SpecRegistry:
    reg = SpecRegistry(backend=backend)
    reg.register(gs.swarm_behavior_definition())
    return reg


def pack() -> TranslationPack:
    """The ground station's pack (defaults are filled per behaviour by the translator)."""
    return TranslationPack(gs.SPEC_NAME, SYSTEM, EXAMPLES, version=1,
                           questions={"behavior": gs.QUESTIONS["behavior"]})  # fmt: skip


def _behavior_name(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    key = re.sub(r"[\s-]+", "_", value.strip().lower())
    return {"return_to_home": "rth", "searchgrid": "search_grid"}.get(key, key)


RETRY_QUESTION = "What exactly should the swarm do, and with which settings?"


class GroundTranslator(PolicyTranslator):
    """``PolicyTranslator`` plus the site map, per-behaviour defaults and the literal
    check (see module)."""

    _text = ""
    _calls = 0

    def translate(self, text: str, pack: TranslationPack, priority: int = 0,
                  deadline: int | None = None) -> Any:  # fmt: skip
        self._text, self._calls = text, 0
        return super().translate(text, pack, priority, deadline)

    def _decide(self, raw: str, pack: TranslationPack, priority: int,
                deadline: int | None) -> Any:  # fmt: skip
        self._calls += 1
        try:
            body = json.loads(raw)
        except ValueError:
            return super()._decide(raw, pack, priority, deadline)
        params = body.get("params") if isinstance(body, dict) else None
        if isinstance(body, dict):
            question = body.get("clarify")
            if isinstance(question, str) and question.strip():
                return (CLARIFY, [question.strip()])
            refusal = body.get("refusal")
            if isinstance(refusal, str) and "?" in refusal:
                return (CLARIFY, [refusal.strip()])
        stop = isinstance(body, dict) and any(
            isinstance(body.get(k), str) and body[k].strip()
            for k in ("refusal", "clarify")
        )
        if isinstance(params, dict) and not stop:
            params = {k: v for k, v in params.items() if v is not None}
            if "behavior" in params:
                params["behavior"] = _behavior_name(params["behavior"])
            if self._calls > 1 and set(params) <= {"behavior"}:
                return (CLARIFY, [RETRY_QUESTION])
            if isinstance(params.get("area"), str):
                place = gs.find_place(params["area"])
                if place is None:
                    return (CLARIFY, [f"I don't know {params['area']!r}. "
                                      f"Known places: {PLACES}."])  # fmt: skip
                params["area"] = place.name
            invented = gs.unsupported(params, self._text)
            if invented:
                return (_UNSATISFIABLE, invented)
            params = gs.snap_literals(params, self._text)
            need = gs.BEHAVIOR_PARAMS.get(params.get("behavior"), {})
            missing = [k for k, v in need.items() if v is None and k not in params]
            if missing:
                return (CLARIFY, [" ".join(gs.QUESTIONS[k] for k in missing)])
            misplaced = gs.misplaced_lengths(params, self._text)
            if misplaced:
                return (
                    CLARIFY,
                    [f"Which setting is that for? ({'; '.join(misplaced)})"],
                )
            body = {**body, "params": gs.fill_behavior_defaults(params)}
        return super()._decide(json.dumps(body), pack, priority, deadline)


# ------------------------------------------------------------------ keyword baseline
_NUM = r"(\d+(?:\.\d+)?)"
_KEYWORDS = (
    ("rth", r"\b(return|come back|go home|recall|rth|head home|back to base)\b"),
    ("person_follow", r"\b(follow|track|tail)\b"),
    ("light_show", r"\b(light show|lights|led|show)\b"),
    ("perimeter", r"\b(perimeter|circle around|orbit|guard|patrol around|watch)\b"),
    ("search_grid", r"\b(search|sweep|scan|comb|look for)\b"),
    ("boids", r"\b(boids|flock|swarm freely|like birds)\b"),
    (
        "formation",
        r"\b(formation|line up|in a line|v shape|v formation|grid|circle|line)\b",
    ),
    ("hover", r"\b(hover|hold|stay|stop|wait)\b"),
)


def keyword_params(text: str) -> dict[str, Any] | None:
    """The regex baseline: first matching behaviour keyword, numbers by their unit
    words, place names by exact match. None when nothing matches."""
    t = text.lower()
    behavior = next((b for b, pat in _KEYWORDS if re.search(pat, t)), None)
    if behavior is None:
        return None
    p: dict[str, Any] = {"behavior": behavior}
    feet = 0.3048 if re.search(r"\b(ft|feet|foot)\b", t) else 1.0
    alt = re.search(
        _NUM + r"\s*(?:m|meters?|metres?|ft|feet)?\s*(?:up|high|altitude)", t
    )
    alt = alt or re.search(r"(?:at|altitude(?: of)?)\s*" + _NUM, t)
    allowed = gs.BEHAVIOR_PARAMS[behavior]
    if alt and "alt_m" in allowed:
        p["alt_m"] = round(float(alt.group(1)) * feet, 2)
    if behavior == "formation":
        for shape in ("line", "grid", "circle", "v"):
            if re.search(rf"\b{shape}\b", t):
                p["shape"] = shape
                break
        sp = re.search(
            _NUM + r"\s*(?:m|meters?|metres?|ft|feet)?\s*(?:apart|spacing)", t
        )
        if sp:
            p["spacing_m"] = round(float(sp.group(1)) * feet, 2)
    if behavior in ("search_grid", "perimeter"):
        for place in gs.SITE:
            if any(
                re.search(rf"\b{re.escape(n)}\b", t)
                for n in (place.name, *place.aliases)
            ):
                p["area"] = place.name
                break
        r = re.search(
            _NUM + r"\s*(?:m|meters?|metres?|ft|feet)?\s*(?:radius|out|away)", t
        )
        if r and behavior == "perimeter":
            p["radius_m"] = round(float(r.group(1)) * feet, 2)
    if behavior == "light_show":
        for pat in gs.PATTERNS:
            if pat in t:
                p["pattern"] = pat
        bpm = re.search(_NUM + r"\s*bpm", t)
        if bpm:
            p["bpm"] = int(float(bpm.group(1)))
    return p


def keyword_reply(text: str) -> str:
    params = keyword_params(text)
    if params is None:
        return json.dumps({"refusal": "keyword parser found nothing", "clarify": ""})
    return json.dumps({"refusal": "", "clarify": "", "params": params})


class KeywordGroundBackend:
    """``keyword_reply`` behind the structured-output interface."""

    def generate(self, prompts: Any, **kw: Any) -> list[Any]:
        from types import SimpleNamespace

        return [SimpleNamespace(text=keyword_reply(t), latency_s=0.0, n_prompt_tokens=0)
                for t in prompts]  # fmt: skip


# ------------------------------------------------------------------ scoring
OUTCOMES = ("exact", "wrong command", "asked", "refused", "failed")


def score(result: Any, item: Mapping[str, Any]) -> str:
    """One of ``OUTCOMES`` for a translation result against a gold item."""
    if result.status == OK:
        if item["expect"] != "ok":
            return "wrong command"
        got = gs.canonical(result.spec.params)
        return "exact" if got == gs.canonical(item["params"]) else "wrong command"
    if result.status == CLARIFY:
        return "asked"
    if result.status == REFUSED:
        return "refused"
    return "failed"


def correct(outcome: str, expect: str) -> bool:
    return {"ok": "exact", "clarify": "asked", "refuse": "refused"}[expect] == outcome


def load_items(path: Any) -> list[dict[str, Any]]:
    from pathlib import Path

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
