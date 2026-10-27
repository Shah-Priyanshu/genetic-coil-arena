"""R12 (goal G9): a log-novelty specialist joins the incident team, and the centre that fuses it.

The G4c team (``trained.py``) ranks services by the metric specialists' anomaly score. RE3's
code-level faults barely move the metrics but make the root service log lines it never logged
before (``logsignal.py``, the ``new_ext`` signal). The newcomer reports that signal:

* **Newcomer** (code, no training, one per service, sees only that service's own logs): a
  :data:`NOVELTY_SCHEMA` report ``{service, new_ext, top}``, ``new_ext`` the number of
  templates that first appear after the alert and are warning/error-like or lifecycle lines,
  ``top`` at most :data:`MAX_TOP` of them (:data:`MAX_CHARS` characters each, so the report
  stays within about 60 tokens).
* **Joined centre**: :data:`RULES` fuses the metric score z and the novelty count n of every
  service. A rule has a weight ``w`` for the log part, an optional ``gate`` (the log part is
  dropped when the best metric score reaches it: the metric evidence is decisive) and an
  optional ``cap`` (a service whose n exceeds it is treated as silent: a template storm on
  one service is a restart storm of a victim, not evidence). The score of a service is
  ``rr(z) + w * rr(n')`` with ``rr`` the reciprocal rank (ties share the best rank, no credit
  where the value is not positive); ties go to the higher z, then the service order. The rule
  ``metric`` (z only) is the old team's rule and reproduces its ranking exactly.

The centre sees only the reports; it is never told which dataset an incident comes from. The
fault type is still the trained metric specialist's report for the top service.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ...protocol.goalspec import SpecDefinition, compile_schema
from .incident import Incident
from .logsignal import is_lifecycle, reciprocal_ranks
from .messages import DECISION_SCHEMA
from .trained import TRAINED_METRIC_SCHEMA, fault_of

MAX_TOP = 3
MAX_CHARS = 80

NOVELTY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["service", "new_ext", "top"],
    "additionalProperties": False,
    "properties": {
        "service": {"type": "string", "minLength": 1},
        "new_ext": {"type": "integer", "minimum": 0},
        "top": {
            "type": "array",
            "maxItems": MAX_TOP,
            "items": {"type": "string", "maxLength": MAX_CHARS},
        },
    },
}
_check_novelty = compile_schema(NOVELTY_SCHEMA)

SPEC_LOG_NOVELTY = "rca.log_novelty"
SPEC_DECIDE_JOINED = "rca.decide_joined"


@dataclass(frozen=True)
class Rule:
    """A fusion rule of the joined centre (see the module docstring)."""

    w: float
    gate: Optional[float] = None
    cap: Optional[float] = None


OLD_RULE = "metric"
RULES: dict[str, Rule] = {
    "rr_w0.5": Rule(0.5),
    "rr_w1": Rule(1.0),
    "rr_w2": Rule(2.0),
    "gate40": Rule(1.0, gate=40.0),
    "gate45": Rule(1.0, gate=45.0),
    "cap20": Rule(1.0, cap=20.0),
    "cap50": Rule(1.0, cap=50.0),
    "gate45_cap50": Rule(1.0, gate=45.0, cap=50.0),
    "gate40_cap50": Rule(1.0, gate=40.0, cap=50.0),
}
ALL_RULES = (OLD_RULE, *RULES)


def novelty_errors(message: Any) -> list[str]:
    return _check_novelty(message)


# ------------------------------------------------------------------ the newcomer


def _new_rows(inc: Incident) -> dict[int, list[tuple[int, int, str]]]:
    """Per service, ``(severity, post count, text)`` of its new warning/error/lifecycle
    templates (cached on the incident: the line arrays may be dropped afterwards)."""
    cached = getattr(inc, "_novelty_rows", None)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    st = inc.stats()
    rows: dict[int, list[tuple[int, int, str]]] = {}
    for k in np.flatnonzero((st.pre == 0) & (st.post > 0)):
        text = inc.templates[int(st.tpl[k])]
        sev = int(st.sev[k])
        if sev >= 1 or is_lifecycle(text):
            rows.setdefault(int(st.svc[k]), []).append((sev, int(st.post[k]), text))
    inc._novelty_rows = rows  # type: ignore[attr-defined]
    return rows


def novelty_message(inc: Incident, service: int) -> dict[str, Any]:
    """The newcomer's report for ``services[service]``; reads only that service's logs."""
    found = _new_rows(inc).get(service, [])
    ranked = sorted(found, key=lambda r: (-r[0], -r[1], r[2]))
    return {
        "service": inc.services[service],
        "new_ext": len(found),
        "top": [r[2][:MAX_CHARS] for r in ranked[:MAX_TOP]],
    }


def empty_novelty_message(service: str) -> dict[str, Any]:
    return {"service": service, "new_ext": 0, "top": []}


# ------------------------------------------------------------------ the joined centre


def _rr(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, reciprocal_ranks(x), 0.0)


def joined_scores(rule: str, z: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, bool]:
    """``(score per service, whether the log part took part)`` under ``rule``."""
    z = np.asarray(z, dtype=np.float64)
    if rule == OLD_RULE:
        return z.copy(), False
    spec = RULES[rule]
    n = np.asarray(n, dtype=np.float64)
    if spec.gate is not None and z.max(initial=0.0) >= spec.gate:
        return _rr(z), False
    if spec.cap is not None:
        n = np.where(n > spec.cap, 0.0, n)
    return _rr(z) + spec.w * _rr(n), bool((n > 0).any())


def joined_ranking(
    rule: str,
    metric_msgs: Sequence[Mapping[str, Any]],
    novelty_msgs: Sequence[Mapping[str, Any]],
) -> tuple[list[int], bool]:
    """``(service indices best first, whether the log part took part)``.

    Ties go to the higher metric score, then the lower index; for ``metric`` this is the old
    team's stable ranking.
    """
    z = np.array([float(m["score"]) for m in metric_msgs])
    n = np.array([float(m["new_ext"]) for m in novelty_msgs])
    score, used = joined_scores(rule, z, n)
    order = np.lexsort((np.arange(len(z)), -z, -score))
    return [int(i) for i in order], used


def rule_decide_joined(
    services: Sequence[str],
    metric_msgs: Sequence[Mapping[str, Any]],
    novelty_msgs: Sequence[Mapping[str, Any]],
    rule: str,
) -> tuple[list[str], str, bool]:
    """``(ranking of service names, fault of the top service, log part took part)``."""
    order, used = joined_ranking(rule, metric_msgs, novelty_msgs)
    fault = fault_of("metric", metric_msgs[order[0]], None)
    return [services[i] for i in order], fault, used


# ------------------------------------------------------------------ specs


def joined_definitions() -> list[SpecDefinition]:
    """The two goals the newcomer and the joined centre add to the team's registry."""
    return [
        SpecDefinition(
            name=SPEC_LOG_NOVELTY,
            version=1,
            description="Report how many log templates are new at one service after the alert.",
            params_schema={
                "type": "object",
                "required": ["incident", "service"],
                "additionalProperties": False,
                "properties": {
                    "incident": {"type": "string", "minLength": 1},
                    "service": {"type": "string", "minLength": 1},
                },
            },
            result_schema=NOVELTY_SCHEMA,
        ),
        SpecDefinition(
            name=SPEC_DECIDE_JOINED,
            version=1,
            description="Decide root-cause service and fault type from metric and novelty reports.",
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
                    "metric_reports": {
                        "type": "array",
                        "items": TRAINED_METRIC_SCHEMA,
                    },
                    "novelty_reports": {"type": "array", "items": NOVELTY_SCHEMA},
                    "rule": {"enum": list(ALL_RULES)},
                },
            },
            result_schema=DECISION_SCHEMA,
        ),
    ]
