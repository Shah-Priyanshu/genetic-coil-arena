"""R10 step 4 (goal G4d): typed log reports for code-level faults (RCAEval RE3).

RE3's fault ids have no cross-service meaning, so the log specialist does not name a fault
type. It answers the question its own logs can settle: *is this service itself misbehaving,
and how?* The report (:data:`LOG4_SCHEMA`) carries a suspicion in [0, 1], a coarse kind of
evidence and up to three evidence templates. A metric specialist (:mod:`.team`) still sends
its anomaly score; the centre fuses the two, either by a rule (:func:`fused_scores`, weights
chosen on the dev split) or with an LLM that reads one line of typed reports per service.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from ...protocol.goalspec import compile_schema
from . import team as tm
from .incident import Incident
from .logsignal import reciprocal_ranks
from .views import TokenCounter

KINDS = ("exception", "crash_restart", "error_response", "new_behaviour", "none")
MAX_EVIDENCE = 3
MAX_EVIDENCE_CHARS = 80
ONSET_NOTE = "the window is centred on the alert"

LOG4_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["service", "suspicion", "kind", "evidence"],
    "additionalProperties": False,
    "properties": {
        "service": {"type": "string", "minLength": 1},
        "suspicion": {"type": "number", "minimum": 0, "maximum": 1},
        "kind": {"enum": list(KINDS)},
        "evidence": {
            "type": "array",
            "maxItems": MAX_EVIDENCE,
            "items": {"type": "string", "maxLength": MAX_EVIDENCE_CHARS},
        },
    },
}
_check_log4 = compile_schema(LOG4_SCHEMA)

CENTRE_SCHEMA_NOTE = "ranking: service names, best first, at most three"

# (weight of the metric part, weight of the LLM log part) of the rule centre
RULE_WEIGHTS: dict[str, tuple[float, float]] = {
    "log_only": (0.0, 1.0),
    "w0.5": (1.0, 0.5),
    "w1": (1.0, 1.0),
    "w2": (1.0, 2.0),
    "lex": (1.0, 100.0),
}
DEFAULT_RULE = "w1"


def log4_errors(message: Any) -> list[str]:
    return _check_log4(message)


def empty_log4_message(service: str) -> dict[str, Any]:
    """What the centre sees from a log specialist that sent nothing usable."""
    return {"service": service, "suspicion": 0.0, "kind": "none", "evidence": []}


def clean_log4_message(
    parsed: Optional[Mapping[str, Any]], service: str
) -> Optional[dict[str, Any]]:
    """A model's reply as a valid report (service forced, evidence cut), or None."""
    if not isinstance(parsed, Mapping):
        return None
    msg = dict(parsed)
    msg["service"] = service
    s = msg.get("suspicion")
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        msg["suspicion"] = round(min(max(float(s), 0.0), 1.0), 2)
    ev = msg.get("evidence")
    if isinstance(ev, list):
        msg["evidence"] = [
            str(e)[:MAX_EVIDENCE_CHARS] for e in ev[:MAX_EVIDENCE] if isinstance(e, str)
        ]
    return None if log4_errors(msg) else msg


def log4_view(
    inc: Incident,
    service: int,
    onset: Optional[float],
    budget: int,
    counter: TokenCounter,
    before_s: float = 120.0,
    after_s: float = 600.0,
) -> str:
    """The service's own log templates around the onset (see :func:`team.log_view`)."""
    note = None if onset is not None else ONSET_NOTE
    return tm.log_view(
        inc, service, onset, budget, counter, before_s, after_s, onset_note=note
    )


def log4_prompt(service: str, view: str) -> str:
    return (
        f"You are the log monitor of service '{service}'. An incident started at the "
        "alert time (0s): a software fault (a code change such as a wrong parameter, a "
        "missing call or an unhandled exception) was injected into exactly one service. "
        "You see only this service's own logs (counts in a window around the alert "
        "against a baseline before it) and cannot see any other service. Decide how "
        "likely it is that THIS service is the one with the fault.\n"
        "Most lines are normal background, and many services log harmless warnings all "
        "the time. A faulty service shows lines that are new or much more frequent "
        "after the alert: exceptions and stack traces, error or failure responses, "
        "repeated start-up lines (the service was restarted or is crash-looping), or "
        "normal requests that now behave differently. A service whose lines are "
        "unchanged, or only change because other services changed, is not the fault.\n"
        "Reply with one JSON object with exactly these keys: service; suspicion (a "
        "number from 0 to 1: 0 if nothing in these logs changed meaningfully, 0.8 or "
        "more only for a clear new error, exception or restart); kind (exception, "
        "crash_restart, error_response, new_behaviour or none); evidence (at most 3 "
        "short log templates copied from below that support your answer, empty if "
        f"none). Reply with the JSON only.\n\n{view}"
    )


# ------------------------------------------------------------------- rule centre


def fused_scores(
    metric_msgs: Sequence[Mapping[str, Any]],
    log_msgs: Sequence[Mapping[str, Any]],
    rule: str = DEFAULT_RULE,
) -> np.ndarray:
    """Per-service score of the rule centre: weighted reciprocal ranks of the metric
    score and the log suspicion (a source that is silent for a service gives it no credit).
    """
    wm, wl = RULE_WEIGHTS[rule]
    ms = np.array([float(m["score"]) for m in metric_msgs])
    ls = np.array([float(m["suspicion"]) for m in log_msgs])
    m = np.where(ms >= tm.METRIC_MIN_Z, reciprocal_ranks(ms), 0.0)
    g = np.where(ls > 0, reciprocal_ranks(ls), 0.0)
    return wm * m + wl * g


# -------------------------------------------------------------------- LLM centre


def centre_lines(
    metric_msgs: Sequence[Mapping[str, Any]],
    log_msgs: Sequence[Mapping[str, Any]],
    with_evidence: bool = True,
) -> tuple[list[str], int]:
    """One line of typed reports per *reporting* service, alphabetical (no order cue).

    A service is listed if its metric score reaches the metric threshold or its log
    suspicion is above 0.1; the count of silent services is returned for the header.
    """
    rows = []
    silent = 0
    for mm, lm in zip(metric_msgs, log_msgs):
        loud_m = float(mm["score"]) >= tm.METRIC_MIN_Z
        loud_l = float(lm["suspicion"]) > 0.1
        if not (loud_m or loud_l):
            silent += 1
            continue
        lm2 = lm if with_evidence else dict(lm, evidence=[])
        rows.append((mm["service"], tm.centre_line(mm, lm2)))
    rows.sort()
    return [f"- {name}: {line}" for name, line in rows], silent


def centre_body(
    metric_msgs: Sequence[Mapping[str, Any]],
    log_msgs: Sequence[Mapping[str, Any]],
    counter: TokenCounter,
    budget: int,
) -> str:
    """The LLM centre's evidence; log evidence strings are dropped first if it is too long,
    then the weakest reports."""
    head = (
        "An alert fired at 0s; a software fault was injected into exactly one service. "
        "You cannot read metrics or logs yourself. Each service has monitors that sent "
        "typed JSON reports. metrics: score = the largest deviation of any of the "
        "service's metrics from its normal range (IQR units; resource faults move it, "
        "code faults often do not), onset_s = when it first left the normal range, top "
        "= its most deviating metrics. logs: suspicion = how likely its log monitor "
        "thinks the service itself has the fault (0 to 1), kind = what it saw, evidence "
        "= log templates. Services not listed sent nothing notable ({silent} of them)."
    )
    for with_ev in (True, False):
        lines, silent = centre_lines(metric_msgs, log_msgs, with_ev)
        body = head.replace("{silent}", str(silent)) + "\n" + "\n".join(lines)
        if counter.count(body) <= budget:
            return body
    # still too long: keep the reports with the highest suspicion, then metric score
    strength = {
        mm["service"]: (float(lm["suspicion"]), float(mm["score"]))
        for mm, lm in zip(metric_msgs, log_msgs)
    }
    keep = sorted(strength, key=lambda k: strength[k], reverse=True)
    while keep:
        keep = keep[: max(1, int(len(keep) * 0.8))]
        sel = set(keep)
        m2 = [m for m in metric_msgs if m["service"] in sel]
        l2 = [m for m in log_msgs if m["service"] in sel]
        lines, _ = centre_lines(m2, l2, False)
        body = head.replace("{silent}", "many") + "\n" + "\n".join(lines)
        if counter.count(body) <= budget or len(keep) == 1:
            return body
    return head.replace("{silent}", "all")


def centre_prompt(body: str) -> str:
    return (
        f"{body}\n\nWhich service is the root cause (the one the fault was injected "
        "into)? Weigh both kinds of report: a service whose logs show a clear new error "
        "or restart is a strong candidate even when its metrics look normal, and a "
        "service whose metrics deviate strongly but whose logs are unchanged may only "
        "be a victim. Reply as JSON: "
        '{"ranking": [the names of your three most likely root-cause services, best first]}.'
    )


def centre_schema(names: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["ranking"],
        "properties": {
            "ranking": {
                "type": "array",
                "items": {"enum": list(names)},
                "minItems": 1,
                "maxItems": 3,
            }
        },
    }


def parse_ranking(text: str, names: Sequence[str]) -> tuple[list[str], bool]:
    """(ranking completed with the unnamed services in order, whether the reply was usable)."""
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = {}
    picked: list[str] = []
    for n in parsed.get("ranking", []) if isinstance(parsed, dict) else []:
        if n in names and n not in picked:
            picked.append(n)
    return picked + [n for n in names if n not in picked], bool(picked)


def single_prompt(body: str, names: Sequence[str]) -> str:
    return (
        f"{body}\n\nExactly one service had a software fault (a code-level change) "
        "injected at the alert time 0s. Which service is the root cause? Reply as JSON: "
        '{"ranking": [the names of your three most likely root-cause services, best '
        "first]}."
    )
