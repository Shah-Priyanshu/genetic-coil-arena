"""R16 (goal G14): a plain-language ground station for OpenSwarm.

An operator types a request ("spread out in a line, two metres apart"); the ground
station turns it into one ``swarm.behavior`` GoalSpec, or asks, or refuses. Everything
after the model's extraction is code (``envs/swarm/ground.py``): per-behaviour defaults
from OpenSwarm's config, place names resolved against the site map, schema validation,
the state check and the 64-byte ``BEHAVIOR_CMD`` encoding.

``GroundTranslator`` is the policy translator (parse first, validate after) with two
steps in front of validation: a named place becomes its canonical name, or a question
when the site map does not know it, and a behaviour that needs a place it was not given
asks for one. The model never sees coordinates.

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
- "person_follow": follow the person the scout's camera detects. distance_m; alt_m; \
min_conf (detection confidence 0-1, e.g. 80% -> 0.8).
- "light_show": synchronised LED show. pattern: "rainbow", "pulse", "chase", "sparkle" or \
"wave"; bpm (tempo).
- "rth": return to home (come back, go home, recall the swarm).

Place names on the site map: {PLACES}. Copy the operator's place name into area; the \
station resolves it.

Rules:
- Numbers: convert to metres (1 ft = 0.3048 m) and give plain numbers.
- If the request needs a place the operator did not name (e.g. "search over there"), ask \
in clarify which area.
- If the request is not about flying the swarm, or asks for something none of the \
behaviours does (landing, filming, spraying, delivering, single-drone moves), or asks for \
two behaviours at once, refuse with a short reason.
- Never invent a value the operator did not give.

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


class GroundTranslator(PolicyTranslator):
    """``PolicyTranslator`` plus the site map and per-behaviour defaults (see module)."""

    def _decide(self, raw: str, pack: TranslationPack, priority: int,
                deadline: int | None) -> Any:  # fmt: skip
        try:
            body = json.loads(raw)
        except ValueError:
            return super()._decide(raw, pack, priority, deadline)
        params = body.get("params") if isinstance(body, dict) else None
        stop = isinstance(body, dict) and any(
            isinstance(body.get(k), str) and body[k].strip()
            for k in ("refusal", "clarify")
        )
        if isinstance(params, dict) and not stop:
            params = {k: v for k, v in params.items() if v is not None}
            if "behavior" in params:
                params["behavior"] = _behavior_name(params["behavior"])
            if isinstance(params.get("area"), str):
                place = gs.find_place(params["area"])
                if place is None:
                    return (CLARIFY, [f"I don't know {params['area']!r}. "
                                      f"Known places: {PLACES}."])  # fmt: skip
                params["area"] = place.name
            need = gs.BEHAVIOR_PARAMS.get(params.get("behavior"), {})
            missing = [k for k, v in need.items() if v is None and k not in params]
            if missing:
                return (CLARIFY, [" ".join(gs.QUESTIONS[k] for k in missing)])
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
