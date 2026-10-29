"""OpenSwarm's ground station: behaviours, places, safety checks and BEHAVIOR_CMD packets.

Mirrors the user's drone platform (``D:\\Projects\\drone``, read-only for this repo):

- ``firmware/src/swarm/behaviors.h``: ``BehaviorID`` (hover 0 ... rth 7) and
  ``FormationShape`` (line 0, v 1, grid 2, circle 3);
- ``SWARM_PROTOCOL.md``, message 0x03 ``BEHAVIOR_CMD``: ``behavior_id``, ``param_count``
  and ``params[64]``, behaviour-specific JSON;
- ``configs/swarm_config.json``: the defaults (formation V at 1.5 m, boids weights 1.0 /
  1.5 / 1.0, person-follow 3 m behind at 2 m with confidence 0.7), the minimum separation
  (0.8 m), the battery thresholds (20% low, 10% critical) and the optical-flow ceiling
  for indoor flight (3 m).

What the ground station adds, all of it deterministic code:

- ``swarm_behavior_definition``: the ``swarm.behavior`` v1 GoalSpec. One behaviour per
  command; each behaviour accepts only its own params (the schema forbids the others).
- ``SITE``: the place book. Operators name places ("the north field"); drones get local
  coordinates in metres from home (x east, y north). Places marked ``no_fly`` exist so
  the operator can name them and be refused with the reason.
- ``check``: the command against the swarm's state (who is flying, batteries, vision
  modules, indoor or outdoor). Returns the reasons to block, and warnings.
- ``encode`` / ``decode``: the GoalSpec as a ``BEHAVIOR_CMD`` packet with compact JSON
  params that always fit the 64 bytes.

Values the firmware config does not give are ground-station choices and marked so:
search and perimeter altitude 5 m, perimeter radius 10 m, light show "rainbow" at 120
bpm, altitude limit 30 m outdoors.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ...protocol.goalspec import SpecDefinition

SPEC_NAME = "swarm.behavior"
BEHAVIORS = {
    "hover": 0,
    "formation": 1,
    "boids": 2,
    "search_grid": 3,
    "perimeter": 4,
    "person_follow": 5,
    "light_show": 6,
    "rth": 7,
}
SHAPES = {"line": 0, "v": 1, "grid": 2, "circle": 3}
MIN_DRONES = {"line": 2, "v": 3, "grid": 4, "circle": 3}
PATTERNS = {
    "rainbow": 0,
    "pulse": 1,
    "chase": 2,
    "sparkle": 3,
    "wave": 4,
}  # ground station
PARAMS_MAX = 64

ALT_MAX_M = 30.0  # ground-station choice (outdoor ceiling)
INDOOR_ALT_MAX_M = 3.0  # swarm_config: optical_flow_altitude_max_m
SEPARATION_MIN_M = 0.8  # swarm_config: boids.separation_min_m
LOW_BATTERY_PCT, CRITICAL_BATTERY_PCT = 20, 10  # swarm_config: flight

# Params each behaviour accepts, with the defaults the ground station fills in when the
# operator does not say (None = mandatory, ask for it).
BEHAVIOR_PARAMS: dict[str, dict[str, Any]] = {
    "hover": {"alt_m": "hold"},  # "hold" = keep the current altitude (param omitted)
    "formation": {"shape": "v", "spacing_m": 1.5, "alt_m": "hold"},
    "boids": {"cohesion": 1.0, "separation": 1.5, "alignment": 1.0},
    "search_grid": {"area": None, "alt_m": 5.0},
    "perimeter": {"area": None, "radius_m": 10.0, "alt_m": 5.0},
    "person_follow": {"distance_m": 3.0, "alt_m": 2.0, "min_conf": 0.7},
    "light_show": {"pattern": "rainbow", "bpm": 120},
    "rth": {},
}
QUESTIONS = {
    "area": "Which area? Name a place on the site map.",
    "behavior": "What should the swarm do?",
}


@dataclass(frozen=True)
class Place:
    """A named area: a box in metres from home (x east, y north)."""

    name: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1
    aliases: tuple[str, ...] = ()
    no_fly: str = ""  # the reason, when flying there is forbidden

    @property
    def centre(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return ((x0 + x1) / 2, (y0 + y1) / 2)


SITE: tuple[Place, ...] = (
    Place("home", (-10, -10, 10, 10), ("base", "launch point", "launch pad", "here")),
    Place("north field", (-40, 30, 40, 90), ("the field to the north",)),
    Place("car park", (50, -20, 90, 20), ("parking lot", "parking")),
    Place("river bank", (-100, -50, -60, 50), ("riverbank", "the river")),
    Place("barn", (20, -40, 40, -25), ("the old barn",)),
    Place("east woods", (100, 0, 160, 80), ("the woods", "forest")),
    Place("orchard", (-60, -90, -10, -50), ("apple orchard",)),
    Place("school", (180, -60, 240, -10), ("schoolyard",), "over a school"),
    Place("highway", (-200, -200, 200, -150), ("the road", "motorway"), "over a road"),
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def find_place(name: str, site: Sequence[Place] = SITE) -> Place | None:
    """The place a name or alias refers to (case and punctuation ignored; a leading
    "the" is optional), or None."""
    key = _norm(name)
    key = key[4:] if key.startswith("the ") else key
    for p in site:
        names = [p.name, *p.aliases]
        if key in {
            _norm(n)[4:] if _norm(n).startswith("the ") else _norm(n) for n in names
        }:
            return p
    return None


# ------------------------------------------------------------------ the GoalSpec
def swarm_behavior_definition(site: Sequence[Place] = SITE) -> SpecDefinition:
    """``swarm.behavior`` v1: one behaviour and its own params.

    Property order matters: grammar-constrained decoding (Ollama structured outputs)
    writes optional properties in schema order, so once a model has written a later
    one it cannot go back. The behaviours' own params come in groups, and ``alt_m``,
    which five behaviours share, comes last (with it second, "search the car park at
    10 m" could only put the 10 after ``area``, into ``distance_m``)."""
    num = {"type": "number"}
    props: dict[str, Any] = {
        "behavior": {
            "type": "string",
            "enum": sorted(BEHAVIORS),
            "description": "The one behaviour the whole swarm switches to.",
        },
        "shape": {"type": "string", "enum": sorted(SHAPES),
                  "description": "Formation shape."},  # fmt: skip
        "spacing_m": {**num, "minimum": SEPARATION_MIN_M, "maximum": 10,
                      "description": "Distance between neighbours in metres."},  # fmt: skip
        "cohesion": {**num, "minimum": 0, "maximum": 5},
        "separation": {**num, "minimum": 0, "maximum": 5},
        "alignment": {**num, "minimum": 0, "maximum": 5},
        "area": {"type": "string", "enum": [p.name for p in site],
                 "description": "A place on the site map."},  # fmt: skip
        "radius_m": {**num, "minimum": 2, "maximum": 100,
                     "description": "Perimeter radius in metres."},  # fmt: skip
        "distance_m": {**num, "minimum": 1, "maximum": 10,
                       "description": "Follow distance in metres."},  # fmt: skip
        "min_conf": {**num, "minimum": 0.3, "maximum": 0.99,
                     "description": "Detection confidence needed to follow."},  # fmt: skip
        "pattern": {"type": "string", "enum": sorted(PATTERNS),
                    "description": "Light-show pattern."},  # fmt: skip
        "bpm": {"type": "integer", "minimum": 40, "maximum": 200,
                "description": "Light-show tempo in beats per minute."},  # fmt: skip
        "alt_m": {**num, "minimum": 0.5, "maximum": ALT_MAX_M,
                  "description": "Flight altitude in metres."},  # fmt: skip
    }
    rules = []
    for b, params in BEHAVIOR_PARAMS.items():
        allowed = {"behavior", *params}
        required = [k for k, v in params.items() if v is None]
        then: dict[str, Any] = {
            "not": {"anyOf": [{"required": [k]} for k in props if k not in allowed]}
        }
        if required:
            then["required"] = required
        rules.append({"if": {"properties": {"behavior": {"const": b}}}, "then": then})
    schema = {
        "type": "object",
        "properties": props,
        "required": ["behavior"],
        "additionalProperties": False,
        "allOf": rules,
    }
    return SpecDefinition(
        SPEC_NAME, 1, schema,
        description="Switch the OpenSwarm swarm to one behaviour (OpenSwarm's "
        "BEHAVIOR_CMD): hover, formation, boids, search_grid, perimeter, "
        "person_follow, light_show or rth (return to home).",
    )  # fmt: skip


def fill_behavior_defaults(params: Mapping[str, Any]) -> dict[str, Any]:
    """Params with ``None`` dropped and the behaviour's documented defaults filled in
    (hover's "hold" altitude stays omitted). Unknown behaviours pass through."""
    out = {k: v for k, v in params.items() if v is not None}
    b = out.get("behavior")
    if isinstance(b, str) and b in BEHAVIOR_PARAMS:
        for k, v in BEHAVIOR_PARAMS[b].items():
            if k not in out and v is not None and v != "hold":
                out[k] = v
    return out


def canonical(params: Mapping[str, Any]) -> dict[str, Any]:
    """Filled params with numbers rounded to 2 decimals (for exact-match scoring)."""
    out = {}
    for k, v in fill_behavior_defaults(params).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            out[k] = v
        else:
            out[k] = round(float(v), 2)
    return out


# ------------------------------------------------------------------ swarm state, checks
@dataclass(frozen=True)
class Drone:
    id: int
    battery_pct: float = 90.0
    has_vision: bool = False  # a K210 module (can be a SCOUT)


@dataclass(frozen=True)
class SwarmState:
    drones: tuple[Drone, ...] = field(default_factory=tuple)
    indoor: bool = False  # GPS unusable: optical flow only, low ceiling


@dataclass(frozen=True)
class Check:
    ok: bool
    blocks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def check(params: Mapping[str, Any], state: SwarmState,
          site: Sequence[Place] = SITE) -> Check:  # fmt: skip
    """Whether the swarm may run this command now; every reason to block it, and
    warnings that do not block."""
    b = params["behavior"]
    blocks, warns = [], []
    n = len(state.drones)
    if n == 0:
        blocks.append("no drones are connected")
    low = [d.id for d in state.drones if d.battery_pct < LOW_BATTERY_PCT]
    critical = [d.id for d in state.drones if d.battery_pct < CRITICAL_BATTERY_PCT]
    if critical and b not in ("rth", "hover"):
        blocks.append(
            f"drone(s) {', '.join(map(str, critical))} below {CRITICAL_BATTERY_PCT}% "
            "battery: only return-home or hover is allowed"
        )
    elif low and b not in ("rth", "hover"):
        warns.append(
            f"drone(s) {', '.join(map(str, low))} below {LOW_BATTERY_PCT}% battery"
        )
    alt = params.get("alt_m")
    if state.indoor and alt is not None and alt > INDOOR_ALT_MAX_M:
        blocks.append(
            f"indoors the swarm flies on optical flow, limited to {INDOOR_ALT_MAX_M:g} m"
        )
    if b == "person_follow" and not any(d.has_vision for d in state.drones):
        blocks.append(
            "person-follow needs a scout with a vision module; none is flying"
        )
    if b == "formation":
        need = MIN_DRONES[params.get("shape", BEHAVIOR_PARAMS["formation"]["shape"])]
        if 0 < n < need:
            blocks.append(f"a {params.get('shape', 'v')} formation needs at least "
                          f"{need} drones; {n} flying")  # fmt: skip
    if "area" in params:
        place = find_place(params["area"], site)
        if place is None:
            blocks.append(f"unknown place {params['area']!r}")
        elif place.no_fly:
            blocks.append(f"{place.name} is a no-fly area ({place.no_fly})")
        elif state.indoor:
            blocks.append("mapped outdoor areas cannot be flown indoors")
    return Check(not blocks, tuple(blocks), tuple(warns))


# ------------------------------------------------------------------ BEHAVIOR_CMD
def _r(x: float) -> Any:
    v = round(float(x), 1)
    return int(v) if v == int(v) else v


def encode(params: Mapping[str, Any], site: Sequence[Place] = SITE) -> bytes:
    """``BEHAVIOR_CMD`` bytes: behavior_id, param_count, then compact JSON params
    (places resolved to metres from home). Raises if the params exceed 64 bytes."""
    b = params["behavior"]
    p = fill_behavior_defaults(params)
    body: dict[str, Any] = {}
    if "alt_m" in p:
        body["a"] = _r(p["alt_m"])
    if b == "formation":
        body.update(s=SHAPES[p["shape"]], m=_r(p["spacing_m"]))
    elif b == "boids":
        body.update(c=_r(p["cohesion"]), s=_r(p["separation"]), l=_r(p["alignment"]))
    elif b in ("search_grid", "perimeter"):
        place = find_place(p["area"], site)
        if place is None:
            raise ValueError(f"unknown place {p['area']!r}")
        if b == "search_grid":
            body["b"] = [_r(v) for v in place.box]
        else:
            body.update(c=[_r(v) for v in place.centre], r=_r(p["radius_m"]))
    elif b == "person_follow":
        body.update(d=_r(p["distance_m"]), k=_r(p["min_conf"] * 100))
    elif b == "light_show":
        body.update(p=PATTERNS[p["pattern"]], t=int(p["bpm"]))
    raw = json.dumps(body, separators=(",", ":")).encode() if body else b""
    if len(raw) > PARAMS_MAX:
        raise ValueError(
            f"params need {len(raw)} bytes, BEHAVIOR_CMD holds {PARAMS_MAX}"
        )
    return bytes([BEHAVIORS[b], len(body)]) + raw.ljust(PARAMS_MAX, b"\0")


def decode(packet: bytes) -> dict[str, Any]:
    """The drone's view of a ``BEHAVIOR_CMD``: behaviour name and raw params."""
    names = {v: k for k, v in BEHAVIORS.items()}
    raw = packet[2:].rstrip(b"\0")
    return {"behavior": names[packet[0]], "count": packet[1],
            "params": json.loads(raw) if raw else {}}  # fmt: skip


# ------------------------------------------------------------------ literal check
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split()
)}  # fmt: skip
_TENS = {w: 10 * (i + 2) for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split()
)}  # fmt: skip
SHAPE_WORDS = {
    "line": ("line", "row", "single file", "column"),
    "v": ("v", "vee", "chevron"),
    "grid": ("grid", "square", "block"),
    "circle": ("circle", "ring"),
}


def spoken_numbers(text: str) -> list[float]:
    """Numbers written in digits ("2.5", "80%") or words ("one hundred and forty")."""
    t = text.lower().replace("-", " ")
    out = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", t)]
    total, cur, seen = 0, 0, False
    for word in re.findall(r"[a-z]+", t) + ["."]:
        if word in _UNITS:
            cur += _UNITS[word]
            seen = True
        elif word in _TENS:
            cur += _TENS[word]
            seen = True
        elif word == "hundred" and seen:
            cur = max(cur, 1) * 100
        elif word in ("a", "and") and seen:
            continue
        elif word == "hundred":  # "a hundred"
            cur, seen = 100, True
        else:
            if seen:
                out.append(float(total + cur))
            total, cur, seen = 0, 0, False
    return out


def literal_values(text: str) -> list[float]:
    """Every value a stated number can become: as said, from feet to metres, and from
    a percentage to a fraction."""
    out = []
    for n in spoken_numbers(text):
        out += [n, round(n * 0.3048, 2), round(n / 100, 2)]
    return out


def snap_literals(params: Mapping[str, Any], text: str) -> dict[str, Any]:
    """Params with every number replaced by the request's own value it matches (within
    ``unsupported``'s tolerance): "20 ft" is 6.1 m even when a model writes 6.08."""
    values = literal_values(text)
    out = dict(params)
    for k, v in params.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and values:
            c = min(values, key=lambda c: abs(v - c))
            if abs(v - c) <= _tolerance(c):
                out[k] = c
    return out


def _tolerance(c: float) -> float:
    return max(0.011, 0.005 * abs(c))


def _said(phrase: str, text: str) -> bool:
    return (
        re.search(rf"(?<![a-z0-9]){re.escape(_norm(phrase))}(?![a-z0-9])", text)
        is not None
    )


def unsupported(params: Mapping[str, Any], text: str,
                site: Sequence[Place] = SITE) -> list[str]:  # fmt: skip
    """The params a model stated that the request does not support: a number the
    operator never said (in any unit), a shape or pattern never named, a place none of
    whose names appear. Empty when every stated value comes from the request."""
    norm = _norm(text)
    values = literal_values(text)
    out = []
    for k, v in params.items():
        if k == "behavior" or v is None:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if not any(abs(v - c) <= _tolerance(c) for c in values):
                out.append(f"{k}={v:g} is not a number the operator said; leave "
                           "unstated parameters out")  # fmt: skip
        elif k == "shape" and isinstance(v, str):
            if not any(_said(w, norm) for w in SHAPE_WORDS.get(v, (v,))):
                out.append(f"shape={v} was not named; leave unstated parameters out")
        elif k == "pattern" and isinstance(v, str):
            if not _said(v, norm):
                out.append(f"pattern={v} was not named; leave unstated parameters out")
        elif k == "area" and isinstance(v, str):
            place = find_place(v, site)
            names = (place.name, *place.aliases) if place else (v,)
            if not any(_said(n, norm) for n in names):
                out.append(f"area={v} was not named by the operator")
    return out
