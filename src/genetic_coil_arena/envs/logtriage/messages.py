"""The specialists' messages and the triage task spec.

A typed message is a JSON object that must satisfy :data:`MESSAGE_SCHEMA` (anomaly flag,
first-anomaly time, line-rate ratio, top error templates with counts, suspected fault
type). A free-text message is a short paragraph. Both are cut to the same per-specialist
token budget, so the comparison is at matched bandwidth.

The task itself is a validated ``logs.triage`` GoalSpec in the shared protocol registry
(``triage_registry``); the centre's decision is checked against its result schema.
"""

import json
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

from ...protocol.goalspec import GoalSpec, SpecDefinition, SpecRegistry, compile_schema
from . import heuristics as hz
from .heuristics import guess_fault
from .incident import Incident
from .rcaeval import FAULT_TYPES
from .views import TokenCounter

FAULT_CHOICES = (*FAULT_TYPES, "none", "unknown")
MAX_ERRORS = 3
MAX_TEMPLATE_CHARS_IN_MESSAGE = 100

MESSAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "service",
        "anomaly",
        "first_anomaly_s",
        "line_rate_ratio",
        "top_errors",
        "suspected_fault",
    ],
    "additionalProperties": False,
    "properties": {
        "service": {"type": "string", "minLength": 1},
        "anomaly": {"type": "boolean"},
        "first_anomaly_s": {"type": ["number", "null"]},
        "line_rate_ratio": {"type": ["number", "null"]},
        "top_errors": {
            "type": "array",
            "maxItems": MAX_ERRORS,
            "items": {
                "type": "object",
                "required": ["template", "count"],
                "additionalProperties": False,
                "properties": {
                    "template": {
                        "type": "string",
                        "maxLength": MAX_TEMPLATE_CHARS_IN_MESSAGE,
                    },
                    "count": {"type": "integer", "minimum": 0},
                },
            },
        },
        "suspected_fault": {"enum": list(FAULT_CHOICES)},
    },
}

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["ranking", "fault_type"],
    "additionalProperties": False,
    "properties": {
        "ranking": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "uniqueItems": True,
        },
        "fault_type": {"enum": list(FAULT_TYPES)},
    },
}

TRIAGE_SPEC = "logs.triage"
_check_message = compile_schema(MESSAGE_SCHEMA)


def triage_registry() -> SpecRegistry:
    """A registry holding the ``logs.triage`` spec."""
    registry = SpecRegistry()
    registry.register(
        SpecDefinition(
            name=TRIAGE_SPEC,
            version=1,
            description=(
                "Find the root-cause service and fault type of an incident from "
                "per-service logs, at a fixed centre context and message budget."
            ),
            params_schema={
                "type": "object",
                "required": ["incident", "candidates", "context_tokens"],
                "additionalProperties": False,
                "properties": {
                    "incident": {"type": "string", "minLength": 1},
                    "candidates": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    },
                    "context_tokens": {"type": "integer", "minimum": 256},
                    "message_tokens": {"type": "integer", "minimum": 16},
                },
            },
            result_schema=DECISION_SCHEMA,
        )
    )
    return registry


def make_goal(
    registry: SpecRegistry,
    inc: Incident,
    context_tokens: int,
    message_tokens: Optional[int] = None,
) -> GoalSpec:
    params: dict[str, Any] = {
        "incident": inc.id,
        "candidates": list(inc.services),
        "context_tokens": int(context_tokens),
    }
    if message_tokens is not None:
        params["message_tokens"] = int(message_tokens)
    return registry.make(TRIAGE_SPEC, params)


def message_errors(message: Any) -> list[str]:
    """Schema problems of a typed message (empty if valid)."""
    return _check_message(message)


def message_json(message: Mapping[str, Any]) -> str:
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False)


def parse_json_object(text: str) -> Optional[dict[str, Any]]:
    """The first balanced ``{...}`` in ``text`` that parses as a JSON object, or None."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : i + 1])
                    except ValueError:
                        break
                    return value if isinstance(value, dict) else None
        start = text.find("{", start + 1)
    return None


def fit_message(
    message: Mapping[str, Any], counter: TokenCounter, budget: int
) -> dict[str, Any]:
    """Shrink a valid typed message to ``budget`` tokens: shorten, then drop, error templates."""
    msg = json.loads(json.dumps(message))
    errors = list(msg.get("top_errors", []))
    for limit in (MAX_TEMPLATE_CHARS_IN_MESSAGE, 70, 45, 25):
        for e in errors:
            e["template"] = e["template"][:limit]
        msg["top_errors"] = errors
        if counter.count(message_json(msg)) <= budget:
            return msg
    while errors and counter.count(message_json(msg)) > budget:
        errors.pop()
        msg["top_errors"] = errors
    return msg


def truncate_text(text: str, counter: TokenCounter, budget: int) -> str:
    """Cut ``text`` (at a character boundary) so that it fits ``budget`` tokens."""
    text = text.strip()
    if counter.count(text) <= budget:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip()


def empty_message(service: str) -> dict[str, Any]:
    """What the centre sees from a specialist that sent nothing usable."""
    return {
        "service": service,
        "anomaly": False,
        "first_anomaly_s": None,
        "line_rate_ratio": None,
        "top_errors": [],
        "suspected_fault": "unknown",
    }


def heuristic_message(
    inc: Incident, service: int, use_metrics: bool = True
) -> dict[str, Any]:
    """A typed message computed by code from one service's own logs (and metrics)."""
    name = inc.services[service]
    pre_n, post_n = inc.service_counts()
    pre, post = int(pre_n[service]), int(post_n[service])
    st = inc.stats()
    mine = np.flatnonzero(st.svc == service)
    score = st.surprise()[mine]
    order = mine[np.argsort(-score, kind="stable")]
    rising = [j for j in order if st.post[j] > st.pre[j]]
    severe = [j for j in rising if st.sev[j] >= 1]
    pool = severe if severe else rising
    top = pool[:MAX_ERRORS]
    err_pre = int(st.pre[mine][st.sev[mine] >= 2].sum())
    err_post = int(st.post[mine][st.sev[mine] >= 2].sum())
    ratio = round(post / pre, 2) if pre > 0 else None
    anomaly = bool(
        (ratio is not None and (ratio < 0.6 or ratio > 1.6))
        or err_post > max(3, 2 * err_pre)
        or any(st.sev[j] >= 2 and st.pre[j] == 0 for j in rising)
    )
    firsts = [float(st.first_post[j]) for j in top if np.isfinite(st.first_post[j])]
    texts = [inc.templates[int(st.tpl[j])] for j in top]
    if ratio is not None and ratio < 0.6:
        texts = texts or ["(service went quiet)"]
    fault = (
        guess_fault(
            texts, default="unknown", weights=[float(score_of(st, j)) for j in top]
        )
        if anomaly
        else "none"
    )
    top_errors = (
        [
            {
                "template": inc.templates[int(st.tpl[j])][
                    :MAX_TEMPLATE_CHARS_IN_MESSAGE
                ],
                "count": int(st.post[j]),
            }
            for j in top
        ]
        if anomaly
        else []
    )
    mrows = inc.metric_rows(name, min_z=5.0, top=2) if use_metrics else []
    if mrows:
        anomaly = True
        firsts += [float(row[1][2]) for row in mrows if row[1][2] is not None]
        top_errors = [
            {
                "template": f"metric {kind} {'up' if d > 0 else 'down'}",
                "count": int(min(z, 999)),
            }
            for kind, (z, d, _f, _r) in mrows
        ] + top_errors
        top_errors = top_errors[:MAX_ERRORS]
        fault = hz.fault_from_metrics(inc, name) or fault
    return {
        "service": name,
        "anomaly": anomaly,
        "first_anomaly_s": round(min(firsts), 1) if (firsts and anomaly) else None,
        "line_rate_ratio": ratio,
        "top_errors": top_errors,
        "suspected_fault": fault,
    }


def score_of(st: Any, j: int) -> float:
    return float(max(st.surprise()[j], 0.0)) + 1e-6
