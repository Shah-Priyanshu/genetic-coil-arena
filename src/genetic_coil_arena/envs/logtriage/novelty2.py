"""R12b (goal G9b): a newcomer that reports what its errors are *about*, and the centre that fuses it.

G9's newcomer counted the new warning/error templates each service logged. On packet loss the
*callers* of the faulty service log the errors, so the count pointed one hop upstream. Here every
new template is attributed: a template logged by service S whose text names another service T of
the same system counts for T ("about T"), otherwise for S ("own"). The typed report
:data:`NOVELTY2_SCHEMA` (``rca.log_novelty`` v2) carries per service ``own``, ``about``,
``n_attr = own + about``, ``logged`` (G9's count: every new template the service logged, kept so
the G9 rule can be re-run as a reference) and up to :data:`MAX_TOP` example templates.

**Name matcher** (:class:`NameMatcher`), built from the system's service list only, no
per-incident rules. A template names service T when, case-insensitively and with ``-`` and ``_``
interchangeable:

* *bare forms* (as a whole token, not part of a longer dotted or hyphenated name): T itself when
  it is distinctive (it contains ``-``, ``_`` or a digit, or ends in ``service``/``svc``), and
  the variants ``<core>service``, ``<core>-service``, ``ts-<core>-service`` of its core name;
* *host forms*, for T itself and for its core name even when that is an ordinary word
  (``payment``, ``user``, ``orders``): ``<form>:<port>`` (a port is digits or ``<n>``),
  ``//<form>`` (URL), ``<form>.<dns label>`` followed by ``.svc``, ``:<port>`` or
  ``.cluster``, ``unknownhostexception: <form>`` and ``lookup <form>``.

The core name drops a leading ``ts-`` and a trailing ``-service``/``service``/``-svc``; a core
shared by two services (and a database service's core) is not a form. A path such as
``POST /orders``, a package such as ``com.rabbitmq.client`` or an e-mail address is not a match.
"""

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ...protocol.goalspec import SpecDefinition, compile_schema
from .incident import Incident
from .logsignal import is_lifecycle
from .messages import DECISION_SCHEMA
from .novelty import (
    MAX_CHARS,
    MAX_TOP,
    OLD_RULE,
    SPEC_DECIDE_JOINED,
    SPEC_LOG_NOVELTY,
    Rule,
    joined_scores,
)
from .trained import TRAINED_METRIC_SCHEMA, fault_of

NOVELTY2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["service", "own", "about", "n_attr", "logged", "top"],
    "additionalProperties": False,
    "properties": {
        "service": {"type": "string", "minLength": 1},
        "own": {"type": "integer", "minimum": 0},
        "about": {"type": "integer", "minimum": 0},
        "n_attr": {"type": "integer", "minimum": 0},
        "logged": {"type": "integer", "minimum": 0},
        "top": {
            "type": "array",
            "maxItems": MAX_TOP,
            "items": {"type": "string", "maxLength": MAX_CHARS},
        },
    },
}
_check_novelty2 = compile_schema(NOVELTY2_SCHEMA)


def novelty2_errors(message: Any) -> list[str]:
    return _check_novelty2(message)


# ------------------------------------------------------------------ the name matcher

_DB_SUFFIX = ("-db", "-mongo", "-mysql")
_PORT = r"(?:\d+|<n>)"
_PRE = r"(?<![\w.\-])"  # whole token: not inside a longer dotted or hyphenated name
_POST = r"(?![\w\-])"


def _flex(form: str) -> str:
    """Regex for ``form`` with ``-`` and ``_`` interchangeable."""
    return "".join("[-_]" if c in "-_" else re.escape(c) for c in form)


def core_name(name: str) -> str:
    n = name.lower()
    if n.startswith("ts-"):
        n = n[3:]
    for suf in ("-service", "service", "-svc"):
        if n.endswith(suf) and len(n) > len(suf):
            return n[: -len(suf)]
    return n


def _distinctive(form: str) -> bool:
    return (
        "-" in form
        or "_" in form
        or any(c.isdigit() for c in form)
        or form.endswith(("service", "svc"))
    )


class NameMatcher:
    """Which services of one system a text names (see the module docstring)."""

    def __init__(self, services: Sequence[str]):
        self.services = list(services)
        lowered = [s.lower() for s in services]
        is_db = [n.endswith(_DB_SUFFIX) for n in lowered]
        cores: dict[str, list[int]] = defaultdict(list)
        for i, n in enumerate(lowered):
            if not is_db[i]:
                cores[core_name(n)].append(i)
        bare: dict[int, set[str]] = defaultdict(set)
        host: dict[int, set[str]] = defaultdict(set)
        for i, n in enumerate(lowered):
            host[i].add(n)
            if _distinctive(n):
                bare[i].add(n)
            core = core_name(n)
            if is_db[i] or len(cores[core]) != 1 or not core:
                continue
            host[i].add(core)
            for form in (f"{core}service", f"{core}-service", f"ts-{core}-service"):
                bare[i].add(form)
        self._res: list[tuple[int, re.Pattern[str]]] = []
        for i in range(len(lowered)):
            parts = [
                (
                    _PRE
                    + "(?:"
                    + "|".join(_flex(f) for f in sorted(bare[i]))
                    + ")"
                    + _POST
                    if bare[i]
                    else None
                )
            ]
            hosts = "|".join(_flex(f) for f in sorted(host[i]))
            h = f"(?:{hosts})"
            parts += [
                _PRE + h + ":" + _PORT,
                "//" + h + _POST,
                _PRE + h + r"\.[a-z][a-z0-9\-]*(?:\.svc|:" + _PORT + r"|\.cluster)",
                r"unknownhostexception:\s*" + h + _POST,
                r"\blookup\s+" + h + _POST,
            ]
            self._res.append(
                (i, re.compile("|".join(p for p in parts if p), re.IGNORECASE))
            )

    def mentions(self, text: str) -> list[int]:
        """Indices of the services ``text`` names."""
        return [i for i, rx in self._res if rx.search(text)]

    def forms(self, index: int) -> dict[str, list[str]]:
        """The accepted forms of one service (for the pre-registration)."""
        name = self.services[index].lower()
        core = core_name(name)
        out: dict[str, list[str]] = {"bare": [], "host": [name]}
        if _distinctive(name):
            out["bare"].append(name)
        if not name.endswith(_DB_SUFFIX):
            out["host"].append(core)
            out["bare"] += [f"{core}service", f"{core}-service", f"ts-{core}-service"]
        return {k: list(dict.fromkeys(v)) for k, v in out.items()}


@lru_cache(maxsize=16)
def _matcher(services: tuple[str, ...]) -> NameMatcher:
    return NameMatcher(services)


# ------------------------------------------------------------------ the newcomer v2

Row = tuple[int, int, str, int]  # severity, post count, text, logging service index


@dataclass
class Attribution:
    """Per service index: templates credited to it as ``own`` / ``about``, and ``logged``."""

    own: dict[int, list[Row]]
    about: dict[int, list[Row]]
    logged: dict[int, int]


def attribution(inc: Incident) -> Attribution:
    """Attribute the new warning/error/lifecycle templates (cached on the incident)."""
    cached = getattr(inc, "_attribution", None)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    st = inc.stats()
    matcher = _matcher(tuple(inc.services))
    own: dict[int, list[Row]] = defaultdict(list)
    about: dict[int, list[Row]] = defaultdict(list)
    logged: dict[int, int] = defaultdict(int)
    for k in np.flatnonzero((st.pre == 0) & (st.post > 0)):
        text = inc.templates[int(st.tpl[k])]
        sev = int(st.sev[k])
        if not (sev >= 1 or is_lifecycle(text)):
            continue
        s = int(st.svc[k])
        row = (sev, int(st.post[k]), text, s)
        logged[s] += 1
        others = [t for t in matcher.mentions(text) if t != s]
        if others:
            for t in others:
                about[t].append(row)
        else:
            own[s].append(row)
    out = Attribution(dict(own), dict(about), dict(logged))
    inc._attribution = out  # type: ignore[attr-defined]
    return out


def novelty2_message(inc: Incident, service: int) -> dict[str, Any]:
    """The newcomer v2's report for ``services[service]``."""
    a = attribution(inc)
    own = a.own.get(service, [])
    about = a.about.get(service, [])
    ranked = sorted(own + about, key=lambda r: (-r[0], -r[1], r[2]))
    top = []
    for _sev, _, text, logger in ranked[:MAX_TOP]:
        top.append(
            (text if logger == service else f"{inc.services[logger]}: {text}")[
                :MAX_CHARS
            ]
        )
    return {
        "service": inc.services[service],
        "own": len(own),
        "about": len(about),
        "n_attr": len(own) + len(about),
        "logged": a.logged.get(service, 0),
        "top": top,
    }


def attribution_moved(inc: Incident) -> dict[str, int]:
    """How much attribution changed the counts: services whose n_attr differs from logged."""
    a = attribution(inc)
    moved = 0
    for s in range(len(inc.services)):
        n = len(a.own.get(s, [])) + len(a.about.get(s, []))
        if n != a.logged.get(s, 0):
            moved += 1
    return {
        "services_moved": moved,
        "templates_redirected": sum(len(v) for v in a.about.values()),
        "templates": sum(a.logged.values()),
    }


# ------------------------------------------------------------------ the joined centre v2

# The G9 chosen rule on the v1 count (``logged``): the reference row, never selectable.
REF_RULE = "g9_gate45_cap50"
ALONE_RULE = "attr_alone"
RULES2: dict[str, Rule] = {
    "a_rr_w1": Rule(1.0),
    "a_gate40": Rule(1.0, gate=40.0),
    "a_gate45": Rule(1.0, gate=45.0),
    "a_gate50": Rule(1.0, gate=50.0),
    "a_cap50": Rule(1.0, cap=50.0),
    "a_gate40_cap50": Rule(1.0, gate=40.0, cap=50.0),
    "a_gate45_cap50": Rule(1.0, gate=45.0, cap=50.0),
    "a_gate45_cap100": Rule(1.0, gate=45.0, cap=100.0),
}
_EXTRA: dict[str, Rule] = {
    REF_RULE: Rule(1.0, gate=45.0, cap=50.0),
    ALONE_RULE: Rule(1.0, z_weight=0.0),
}
_ALL: dict[str, Rule] = {**RULES2, **_EXTRA}
ALL_RULES2 = (OLD_RULE, REF_RULE, ALONE_RULE, *RULES2)


def count_key(rule: str) -> str:
    """The report field a rule counts: the G9 reference uses ``logged``, all others ``n_attr``."""
    return "logged" if rule == REF_RULE else "n_attr"


def joined2_ranking(
    rule: str,
    metric_msgs: Sequence[Mapping[str, Any]],
    novelty_msgs: Sequence[Mapping[str, Any]],
) -> tuple[list[int], bool]:
    """``(service indices best first, whether the log part took part)``."""
    z = np.array([float(m["score"]) for m in metric_msgs])
    n = np.array([float(m[count_key(rule)]) for m in novelty_msgs])
    score, used = joined_scores(rule, z, n, _ALL)
    order = np.lexsort((np.arange(len(z)), -z, -score))
    return [int(i) for i in order], used


def rule_decide_joined2(
    services: Sequence[str],
    metric_msgs: Sequence[Mapping[str, Any]],
    novelty_msgs: Sequence[Mapping[str, Any]],
    rule: str,
) -> tuple[list[str], str, bool]:
    order, used = joined2_ranking(rule, metric_msgs, novelty_msgs)
    fault = fault_of("metric", metric_msgs[order[0]], None)
    return [services[i] for i in order], fault, used


def joined2_definitions() -> list[SpecDefinition]:
    """``rca.log_novelty`` v2 and ``rca.decide_joined`` v2."""
    return [
        SpecDefinition(
            name=SPEC_LOG_NOVELTY,
            version=2,
            description="Report the new log templates credited to one service after the alert: "
            "those it logged about itself and those other services logged about it.",
            params_schema={
                "type": "object",
                "required": ["incident", "service"],
                "additionalProperties": False,
                "properties": {
                    "incident": {"type": "string", "minLength": 1},
                    "service": {"type": "string", "minLength": 1},
                },
            },
            result_schema=NOVELTY2_SCHEMA,
        ),
        SpecDefinition(
            name=SPEC_DECIDE_JOINED,
            version=2,
            description="Decide root-cause service and fault type from metric and attributed "
            "novelty reports.",
            params_schema={
                "type": "object",
                "required": [
                    "incident",
                    "candidates",
                    "metric_reports",
                    "novelty_reports",
                    "rule",
                ],
                "additionalProperties": False,
                "properties": {
                    "incident": {"type": "string", "minLength": 1},
                    "candidates": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    },
                    "metric_reports": {"type": "array", "items": TRAINED_METRIC_SCHEMA},
                    "novelty_reports": {"type": "array", "items": NOVELTY2_SCHEMA},
                    "rule": {"enum": list(ALL_RULES2)},
                },
            },
            result_schema=DECISION_SCHEMA,
        ),
    ]
