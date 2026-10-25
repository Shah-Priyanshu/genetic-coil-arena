"""R10 step 2: a heterogeneous team with typed messages that carry the decisive signal.

* **Metric specialist** (code, one per service, sees only that service's metrics): sends a
  :data:`METRIC_SCHEMA` message with its anomaly score (BARO: the largest robust deviation
  of its metrics, in IQR units), the onset (earliest time a listed metric left its normal
  range) and its top deviating metrics with direction. Tens of bytes.
* **Log specialist** (a model, one per service, sees only that service's logs around the
  onset the metric specialist reported): sends a :data:`LOG_SCHEMA` message with a fault-type
  hypothesis, a confidence and up to three evidence templates.
* **Rule centre**: root cause = argmax score; fault type from the top metric kind or from
  the log hypothesis of that service (several fixed policies, :data:`FAULT_POLICIES`).

Both message kinds and the three goals that carry them (``rca.metric_report``,
``rca.log_report``, ``rca.decide``) are registered in :func:`team_registry`.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from ...protocol.goalspec import SpecDefinition, SpecRegistry, compile_schema
from .heuristics import METRIC_FAULT
from .incident import Incident
from .messages import DECISION_SCHEMA, FAULT_CHOICES
from .rcaeval import FAULT_TYPES
from .views import TokenCounter

METRIC_MIN_Z = 3.0
MAX_METRICS = 3
MAX_EVIDENCE = 3
MAX_EVIDENCE_CHARS = 80
DEFAULT_FAULT = "delay"

METRIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["service", "score", "onset_s", "top"],
    "additionalProperties": False,
    "properties": {
        "service": {"type": "string", "minLength": 1},
        "score": {"type": "number", "minimum": 0},
        "onset_s": {"type": ["number", "null"]},
        "top": {
            "type": "array",
            "maxItems": MAX_METRICS,
            "items": {
                "type": "object",
                "required": ["metric", "dir", "z"],
                "additionalProperties": False,
                "properties": {
                    "metric": {"type": "string", "minLength": 1, "maxLength": 24},
                    "dir": {"enum": ["up", "down"]},
                    "z": {"type": "number", "minimum": 0},
                },
            },
        },
    },
}

LOG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["service", "fault", "confidence", "evidence"],
    "additionalProperties": False,
    "properties": {
        "service": {"type": "string", "minLength": 1},
        "fault": {"enum": list(FAULT_CHOICES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {
            "type": "array",
            "maxItems": MAX_EVIDENCE,
            "items": {"type": "string", "maxLength": MAX_EVIDENCE_CHARS},
        },
    },
}

_check_metric = compile_schema(METRIC_SCHEMA)
_check_log = compile_schema(LOG_SCHEMA)

SPEC_METRIC = "rca.metric_report"
SPEC_LOG = "rca.log_report"
SPEC_DECIDE = "rca.decide"

FAULT_POLICIES = ("metric", "log", "log_c50", "log_c80")
DEFAULT_POLICY = "log_c50"


def metric_errors(message: Any) -> list[str]:
    return _check_metric(message)


def log_errors(message: Any) -> list[str]:
    return _check_log(message)


def compact(message: Mapping[str, Any]) -> str:
    """The wire form of a message (compact JSON); its length is the byte count."""
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False)


def nbytes(message: Mapping[str, Any]) -> int:
    return len(compact(message).encode("utf-8"))


# ---------------------------------------------------------------- metric specialist


def metric_message(inc: Incident, service: int) -> dict[str, Any]:
    """The metric specialist's message; reads only ``service``'s own metrics."""
    name = inc.services[service]
    rows = inc.metrics.get(name, {})
    score = max((float(v[0]) for v in rows.values()), default=0.0)
    listed = inc.metric_rows(name, min_z=METRIC_MIN_Z, top=MAX_METRICS)
    onsets = [float(v[2]) for _, v in listed if v[2] is not None]
    return {
        "service": name,
        "score": round(min(score, 1e6), 2),
        "onset_s": round(min(onsets), 1) if onsets else None,
        "top": [
            {
                "metric": kind,
                "dir": "up" if v[1] > 0 else "down",
                "z": round(min(float(v[0]), 9999.0), 1),
            }
            for kind, v in listed
        ],
    }


def empty_metric_message(service: str) -> dict[str, Any]:
    """What the centre sees from a metric specialist that sent nothing."""
    return {"service": service, "score": 0.0, "onset_s": None, "top": []}


def empty_log_message(service: str) -> dict[str, Any]:
    """What the centre sees from a log specialist that sent nothing usable."""
    return {"service": service, "fault": "unknown", "confidence": 0.0, "evidence": []}


def clean_log_message(
    parsed: Optional[Mapping[str, Any]], service: str
) -> Optional[dict[str, Any]]:
    """A model's reply as a valid log message (service forced, evidence cut), or None."""
    if not isinstance(parsed, Mapping):
        return None
    msg = dict(parsed)
    msg["service"] = service
    conf = msg.get("confidence")
    if isinstance(conf, (int, float)) and not isinstance(conf, bool):
        msg["confidence"] = round(min(max(float(conf), 0.0), 1.0), 2)
    ev = msg.get("evidence")
    if isinstance(ev, list):
        msg["evidence"] = [
            str(e)[:MAX_EVIDENCE_CHARS] for e in ev[:MAX_EVIDENCE] if isinstance(e, str)
        ]
    return None if log_errors(msg) else msg


# ----------------------------------------------------------------- log specialist


def log_view(
    inc: Incident,
    service: int,
    onset: Optional[float],
    budget: int,
    counter: TokenCounter,
    before_s: float = 120.0,
    after_s: float = 300.0,
) -> str:
    """The service's own log templates in a window around ``onset`` against its baseline.

    The window is ``[onset - before_s, onset + after_s]`` (seconds relative to the alert;
    onset defaults to 0); the baseline is everything earlier than the window and the
    alert. Templates are ranked by a rate log-ratio weighted by volume and severity, and
    packed into ``budget`` tokens.
    """
    name = inc.services[service]
    on = 0.0 if onset is None else float(onset)
    w_lo, w_hi = on - before_s, on + after_s
    b_hi = min(0.0, w_lo)
    mine = inc.svc == service
    t, tpl = inc.t[mine], inc.tpl[mine]
    in_win = (t >= w_lo) & (t <= w_hi)
    in_base = t < b_hi
    lo = float(inc.window[0])
    base_len = max(b_hi - lo, 1.0)
    win_len = max(w_hi - w_lo, 1.0)
    n_tpl = max(len(inc.templates), 1)
    win = np.bincount(tpl[in_win], minlength=n_tpl).astype(np.float64)
    base = np.bincount(tpl[in_base], minlength=n_tpl).astype(np.float64)
    eps = 0.5 / max(base_len, win_len)
    lift = np.log((win / win_len + eps) / (base / base_len + eps))
    sev = inc.template_severity.astype(np.float64)
    mass = np.log1p(np.maximum(win, base * win_len / base_len))
    score = np.where(lift > 0, lift, -0.5 * lift) * mass * (1.0 + sev)
    seen = np.flatnonzero((win > 0) | (base > 0))
    order = seen[np.argsort(-score[seen], kind="stable")]
    head = (
        f"Logs of service {name}. Window {w_lo:+.0f}s to {w_hi:+.0f}s (alert at 0s, "
        f"metrics left their normal range at about {on:+.0f}s); baseline = "
        f"{lo:+.0f}s to {b_hi:+.0f}s. {int(in_win.sum())} lines in the window, "
        f"{int(in_base.sum())} in the baseline. Templates, most changed first "
        "(count in window / count in baseline):"
    )
    lines = [head]
    used = counter.count(head)
    for j in order:
        row = f"  {int(win[j])}/{int(base[j])}: {inc.templates[int(j)][:200]}"
        cost = counter.count(row) + 1
        if used + cost > budget:
            continue
        lines.append(row)
        used += cost
    if len(lines) == 1:
        lines.append("  (no log lines in the window or the baseline)")
    return "\n".join(lines)


def log_prompt(service: str, view: str) -> str:
    return (
        f"You are the log monitor of service '{service}'. A separate metric monitor "
        "found that this service's metrics changed at the time given below; you see "
        "only this service's own logs (counts in a window around that time against a "
        "baseline). You cannot see any other service. Say which fault was injected "
        "into THIS service, if the logs tell you.\n"
        "Most lines are normal background and many services log harmless warnings all "
        "the time; a fault is only visible as templates that are new or much more "
        "frequent in the window than in the baseline (or lines that vanished). Do not "
        "report a fault because of a warning that is equally frequent in both. Many "
        "faults leave no trace in the logs at all: then answer unknown with "
        "a low confidence. Never answer none: the metrics of this service did "
        "change, even when its logs look normal.\n"
        "Fault types and what they tend to show: cpu (CPU hog: slow requests, "
        "throttling), mem (memory exhaustion: out-of-memory, GC, killed), disk (disk "
        "I/O stress: I/O errors, no space, slow writes), delay (network delay: "
        "timeouts, deadline exceeded, slow upstream calls), loss (packet loss: "
        "connection reset, refused, unavailable, retries), socket (socket exhaustion: "
        "too many open files, cannot bind, connection errors).\n"
        "Reply with one JSON object with exactly these keys: service; fault (cpu, mem, "
        "disk, delay, loss, socket or unknown); confidence (a number from 0 to "
        "1: how sure these lines make you, 0.8 or more only for a clear change); evidence (at most 3 short log templates "
        "copied from below that support your answer, empty if none). "
        f"Reply with the JSON only.\n\n{view}"
    )


# ------------------------------------------------------------------- rule centre


def metric_fault(msg: Mapping[str, Any]) -> str:
    """Fault type implied by a metric message's strongest metric kind."""
    top = msg.get("top") or []
    return METRIC_FAULT.get(top[0]["metric"], DEFAULT_FAULT) if top else DEFAULT_FAULT


def fault_for(
    policy: str, metric_msg: Mapping[str, Any], log_msg: Optional[Mapping[str, Any]]
) -> str:
    """Fault type of the chosen service under one policy (see :data:`FAULT_POLICIES`)."""
    if policy not in FAULT_POLICIES:
        raise ValueError(f"unknown fault policy {policy!r}")
    fallback = metric_fault(metric_msg)
    if policy == "metric" or not log_msg:
        return fallback
    fault = log_msg.get("fault")
    if fault not in FAULT_TYPES:
        return fallback
    need = {"log": 0.0, "log_c50": 0.5, "log_c80": 0.8}[policy]
    return str(fault) if float(log_msg.get("confidence", 0.0)) >= need else fallback


def rule_ranking(metric_msgs: Sequence[Mapping[str, Any]]) -> list[int]:
    """Service indices, highest anomaly score first (ties by index)."""
    scores = np.array([float(m["score"]) for m in metric_msgs])
    return [int(i) for i in np.argsort(-scores, kind="stable")]


def rule_decide(
    services: Sequence[str],
    metric_msgs: Sequence[Mapping[str, Any]],
    log_msgs: Optional[Sequence[Mapping[str, Any]]],
    policy: str = DEFAULT_POLICY,
) -> tuple[list[str], dict[str, str]]:
    """(ranking of service names, fault type of every policy for the top service)."""
    order = rule_ranking(metric_msgs)
    top = order[0]
    lm = log_msgs[top] if log_msgs else None
    faults = {p: fault_for(p, metric_msgs[top], lm) for p in FAULT_POLICIES}
    faults["chosen"] = faults[policy]
    return [services[i] for i in order], faults


# ------------------------------------------------------------------- registry


def team_registry() -> SpecRegistry:
    """A registry with the three goals of the heterogeneous team."""
    registry = SpecRegistry()
    registry.register(
        SpecDefinition(
            name=SPEC_METRIC,
            version=1,
            description="Report one service's metric anomaly (score, onset, top metrics).",
            params_schema={
                "type": "object",
                "required": ["incident", "service"],
                "additionalProperties": False,
                "properties": {
                    "incident": {"type": "string", "minLength": 1},
                    "service": {"type": "string", "minLength": 1},
                },
            },
            result_schema=METRIC_SCHEMA,
        )
    )
    registry.register(
        SpecDefinition(
            name=SPEC_LOG,
            version=1,
            description="Report a fault hypothesis from one service's logs around an onset.",
            params_schema={
                "type": "object",
                "required": ["incident", "service", "onset_s"],
                "additionalProperties": False,
                "properties": {
                    "incident": {"type": "string", "minLength": 1},
                    "service": {"type": "string", "minLength": 1},
                    "onset_s": {"type": ["number", "null"]},
                },
            },
            result_schema=LOG_SCHEMA,
        )
    )
    registry.register(
        SpecDefinition(
            name=SPEC_DECIDE,
            version=1,
            description="Decide root-cause service and fault type from the typed reports.",
            params_schema={
                "type": "object",
                "required": ["incident", "candidates", "metric_reports"],
                "additionalProperties": False,
                "properties": {
                    "incident": {"type": "string", "minLength": 1},
                    "candidates": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "string"},
                        "uniqueItems": True,
                    },
                    "metric_reports": {"type": "array", "items": METRIC_SCHEMA},
                    "log_reports": {"type": "array", "items": LOG_SCHEMA},
                },
            },
            result_schema=DECISION_SCHEMA,
        )
    )
    return registry


def centre_line(
    metric_msg: Mapping[str, Any], log_msg: Optional[Mapping[str, Any]]
) -> str:
    """One service's reports as the LLM centre reads them (typed JSON, no repeated name)."""
    parts = [
        "metrics=" + compact({k: v for k, v in metric_msg.items() if k != "service"})
    ]
    if log_msg is not None:
        parts.append(
            "logs=" + compact({k: v for k, v in log_msg.items() if k != "service"})
        )
    return " ".join(parts)
