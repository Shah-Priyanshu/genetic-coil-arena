"""R10 step 3: typed fault reports from *trained* specialists, and the rule centre that fuses them.

Step 2's rule centre ranks services by the metric specialists' anomaly score and takes the fault
type from the strongest metric kind. Here the fault type comes from a trained specialist's typed
report for the chosen service:

* the **metric fault specialist** (numpy classifier on its own service's metric features) sends a
  :data:`TRAINED_METRIC_SCHEMA` report: step 2's fields plus ``fault`` and ``fault_probs``;
* the **log fault specialist** (a LoRA adapter on Qwen3-1.7B reading its own service's log view)
  sends a :data:`TRAINED_LOG_SCHEMA` report: ``fault``, ``confidence`` and ``fault_probs``.

``fault_probs`` follow :data:`FAULT_TYPES` order. The centre fuses the two reports with one of
:data:`FUSIONS`, chosen by cross-validation on training data only.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from ...protocol.goalspec import SpecDefinition, SpecRegistry, compile_schema
from .messages import DECISION_SCHEMA
from .rcaeval import FAULT_TYPES
from .team import (
    LOG_SCHEMA,
    METRIC_SCHEMA,
    empty_metric_message,
    metric_fault,
    rule_ranking,
)

N_FAULTS = len(FAULT_TYPES)
FAULT_ENUM = [*FAULT_TYPES, "unknown"]
UNIFORM = [round(1.0 / N_FAULTS, 4)] * N_FAULTS

_PROBS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "minItems": N_FAULTS,
    "maxItems": N_FAULTS,
    "items": {"type": "number", "minimum": 0, "maximum": 1},
}

TRAINED_METRIC_SCHEMA: dict[str, Any] = {
    **METRIC_SCHEMA,
    "required": [*METRIC_SCHEMA["required"], "fault", "fault_probs"],
    "properties": {
        **METRIC_SCHEMA["properties"],
        "fault": {"enum": FAULT_ENUM},
        "fault_probs": _PROBS_SCHEMA,
    },
}

TRAINED_LOG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["service", "fault", "confidence", "fault_probs"],
    "additionalProperties": False,
    "properties": {
        "service": LOG_SCHEMA["properties"]["service"],
        "fault": {"enum": FAULT_ENUM},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "fault_probs": _PROBS_SCHEMA,
    },
}

_check_metric = compile_schema(TRAINED_METRIC_SCHEMA)
_check_log = compile_schema(TRAINED_LOG_SCHEMA)

SPEC_METRIC_FAULT = "rca.metric_fault_report"
SPEC_LOG_FAULT = "rca.log_fault_report"
SPEC_DECIDE_TRAINED = "rca.decide_trained"

FUSIONS = (
    "metric",
    "log",
    "mean_w25",
    "mean_w50",
    "mean_w75",
    "geo_w25",
    "geo_w50",
    "geo_w75",
    "max_conf",
    "stack",
)


def trained_metric_errors(message: Any) -> list[str]:
    return _check_metric(message)


def trained_log_errors(message: Any) -> list[str]:
    return _check_log(message)


# ------------------------------------------------------------------------- messages


def _clean_probs(probs: Sequence[float]) -> list[float]:
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    if p.shape != (N_FAULTS,) or not np.isfinite(p).all() or p.sum() <= 0:
        return list(UNIFORM)
    return [round(float(v), 4) for v in p / p.sum()]


def trained_metric_message(
    base: Mapping[str, Any], probs: Optional[Sequence[float]]
) -> dict[str, Any]:
    """Step 2's metric report plus the trained fault report (``None`` = nothing to say)."""
    msg = dict(base)
    if probs is None:
        msg.update(fault="unknown", fault_probs=list(UNIFORM))
    else:
        p = _clean_probs(probs)
        msg.update(fault=FAULT_TYPES[int(np.argmax(p))], fault_probs=p)
    return msg


def trained_log_message(
    service: str, probs: Optional[Sequence[float]]
) -> dict[str, Any]:
    """The log fault specialist's report; ``None`` probabilities give an empty report."""
    if probs is None:
        return empty_trained_log_message(service)
    p = _clean_probs(probs)
    return {
        "service": service,
        "fault": FAULT_TYPES[int(np.argmax(p))],
        "confidence": round(max(p), 4),
        "fault_probs": p,
    }


def empty_trained_log_message(service: str) -> dict[str, Any]:
    return {
        "service": service,
        "fault": "unknown",
        "confidence": 0.0,
        "fault_probs": list(UNIFORM),
    }


def empty_trained_metric_message(service: str) -> dict[str, Any]:
    return trained_metric_message(empty_metric_message(service), None)


def report_probs(msg: Optional[Mapping[str, Any]]) -> Optional[np.ndarray]:
    """The fault distribution of a report, or None for an empty ("unknown") report."""
    if not msg or msg.get("fault") not in FAULT_TYPES:
        return None
    return np.asarray(msg["fault_probs"], dtype=float)


# ------------------------------------------------------------------------- fusion


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def stack_features(pm: np.ndarray, pl: np.ndarray) -> np.ndarray:
    return np.concatenate([np.log(pm + 1e-4), np.log(pl + 1e-4)])


def fuse(
    rule: str,
    pm: Optional[np.ndarray],
    pl: Optional[np.ndarray],
    stack: Optional[Mapping[str, Any]] = None,
) -> Optional[np.ndarray]:
    """Fused fault distribution, or None if neither report says anything.

    ``stack`` holds ``{"w": (2*N_FAULTS, N_FAULTS), "b": (N_FAULTS,)}``: a logistic regression
    on both reports' log-probabilities (fitted on cross-validated predictions).
    """
    if rule not in FUSIONS:
        raise ValueError(f"unknown fusion rule {rule!r}")
    if pm is None or pl is None:
        return pm if pl is None else pl
    if rule == "metric":
        return pm
    if rule == "log":
        return pl
    if rule == "max_conf":
        return pm if pm.max() >= pl.max() else pl
    if rule == "stack":
        if stack is None:
            raise ValueError("the 'stack' rule needs fitted stack parameters")
        z = stack_features(pm, pl) @ np.asarray(stack["w"]) + np.asarray(stack["b"])
        return _softmax(z)
    kind, w = rule.split("_w")
    w_log = int(w) / 100.0
    if kind == "mean":
        return (1 - w_log) * pm + w_log * pl
    return _softmax((1 - w_log) * np.log(pm + 1e-4) + w_log * np.log(pl + 1e-4))


def fit_stack(
    pm: np.ndarray, pl: np.ndarray, y: np.ndarray, l2: float = 1.0, steps: int = 400
) -> dict[str, Any]:
    """Logistic regression on ``stack_features`` (rows: reports for root services)."""
    x = np.stack([stack_features(a, b) for a, b in zip(pm, pl)])
    onehot = np.eye(N_FAULTS)[np.asarray(y, dtype=int)]
    w = np.zeros((x.shape[1], N_FAULTS))
    b = np.zeros(N_FAULTS)
    n = len(x)
    for _ in range(steps):
        z = x @ w + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - onehot) / n
        w -= 0.5 * (x.T @ g + l2 * w / n)
        b -= 0.5 * g.sum(axis=0)
    return {"w": w.tolist(), "b": b.tolist()}


def fault_of(
    rule: str,
    metric_msg: Mapping[str, Any],
    log_msg: Optional[Mapping[str, Any]],
    stack: Optional[Mapping[str, Any]] = None,
) -> str:
    """The centre's fault type for the chosen service under fusion ``rule``.

    With no usable report the centre falls back to step 2's metric-kind heuristic.
    """
    p = fuse(rule, report_probs(metric_msg), report_probs(log_msg), stack)
    if p is None:
        return metric_fault(metric_msg)
    return FAULT_TYPES[int(np.argmax(p))]


def rule_decide_trained(
    services: Sequence[str],
    metric_msgs: Sequence[Mapping[str, Any]],
    log_msgs: Optional[Sequence[Mapping[str, Any]]],
    rule: str,
    stack: Optional[Mapping[str, Any]] = None,
) -> tuple[list[str], str]:
    """(ranking of service names by metric score, fault of the top service)."""
    order = rule_ranking(metric_msgs)
    top = order[0]
    lm = log_msgs[top] if log_msgs else None
    return [services[i] for i in order], fault_of(rule, metric_msgs[top], lm, stack)


# ----------------------------------------------------------------------- registry


def trained_registry() -> SpecRegistry:
    """A registry with the trained team's three goals."""
    registry = SpecRegistry()
    registry.register(
        SpecDefinition(
            name=SPEC_METRIC_FAULT,
            version=1,
            description="Report one service's metric anomaly and a trained fault-type distribution.",
            params_schema={
                "type": "object",
                "required": ["incident", "service"],
                "additionalProperties": False,
                "properties": {
                    "incident": {"type": "string", "minLength": 1},
                    "service": {"type": "string", "minLength": 1},
                },
            },
            result_schema=TRAINED_METRIC_SCHEMA,
        )
    )
    registry.register(
        SpecDefinition(
            name=SPEC_LOG_FAULT,
            version=1,
            description="Report a trained fault-type distribution from one service's logs.",
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
            result_schema=TRAINED_LOG_SCHEMA,
        )
    )
    registry.register(
        SpecDefinition(
            name=SPEC_DECIDE_TRAINED,
            version=1,
            description="Decide root-cause service and fault type from trained typed reports.",
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
                    "metric_reports": {
                        "type": "array",
                        "items": TRAINED_METRIC_SCHEMA,
                    },
                    "log_reports": {"type": "array", "items": TRAINED_LOG_SCHEMA},
                },
            },
            result_schema=DECISION_SCHEMA,
        )
    )
    return registry
