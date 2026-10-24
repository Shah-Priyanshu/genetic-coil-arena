"""Domain-agnostic metrics for translator evaluation (standard library only).

Exact match and per-field accuracy compare GoalSpec ``params`` after canonicalisation
(arrays whose schema says ``uniqueItems`` are order-free). Summary fidelity checks that
the facts a summary must state are present, via per-fact regular expressions that the
domain supplies.
"""

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional


def canonical_params(params: Any, schema: Optional[Mapping[str, Any]] = None) -> Any:
    """Params with order-free arrays sorted, so equal specs compare equal.

    Arrays are sorted when ``schema`` marks them ``uniqueItems`` (sets in disguise).
    """
    schema = schema or {}
    if isinstance(params, Mapping):
        props = schema.get("properties", {})
        return {k: canonical_params(v, props.get(k)) for k, v in sorted(params.items())}
    if isinstance(params, (list, tuple)):
        items = [canonical_params(v, schema.get("items")) for v in params]
        if schema.get("uniqueItems"):
            items.sort(key=repr)
        return items
    return params


def exact_match(
    pred: Optional[Mapping[str, Any]],
    gold: Mapping[str, Any],
    schema: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Whether ``pred`` params equal ``gold`` params (``None`` never matches)."""
    if pred is None:
        return False
    return canonical_params(pred, schema) == canonical_params(gold, schema)


def field_matches(
    pred: Optional[Mapping[str, Any]],
    gold: Mapping[str, Any],
    schema: Optional[Mapping[str, Any]] = None,
) -> dict[str, bool]:
    """Per gold field: whether ``pred`` has the same value (all False if no ``pred``)."""
    props = (schema or {}).get("properties", {})
    out = {}
    for key, value in gold.items():
        out[key] = bool(
            pred is not None
            and key in pred
            and canonical_params(pred[key], props.get(key))
            == canonical_params(value, props.get(key))
        )
    return out


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion and Wilson score interval ``(p, low, high)``; NaNs when ``total`` is 0."""
    if total <= 0:
        return (math.nan, math.nan, math.nan)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: Iterable[float], q: float) -> float:
    """Linear-interpolated percentile ``q`` in [0, 100]; NaN for an empty input."""
    xs = sorted(values)
    if not xs:
        return math.nan
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def refusal_scores(rows: Iterable[tuple[bool, bool]]) -> dict[str, float]:
    """Refusal precision and recall from ``(gold_refuse, predicted_refuse)`` pairs.

    Precision is NaN when nothing was refused; recall is NaN when nothing should be.
    """
    tp = fp = fn = tn = 0
    for gold, pred in rows:
        if gold and pred:
            tp += 1
        elif pred:
            fp += 1
        elif gold:
            fn += 1
        else:
            tn += 1
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else math.nan,
        "recall": tp / (tp + fn) if tp + fn else math.nan,
    }


@dataclass(frozen=True)
class Fact:
    """Something a summary must state.

    Attributes:
        name: Field or outcome the fact belongs to
        patterns: Regular expressions (case-insensitive); the fact holds if any matches
        forbidden: Regular expressions that must not match (a contradiction)
    """

    name: str
    patterns: tuple[str, ...]
    forbidden: tuple[str, ...] = ()

    def holds(self, text: str) -> bool:
        """Whether the text states the fact and contradicts nothing."""
        if any(re.search(p, text, re.IGNORECASE) for p in self.forbidden):
            return False
        return any(re.search(p, text, re.IGNORECASE) for p in self.patterns)


def fact_fidelity(summary: str, facts: Sequence[Fact]) -> dict[str, Any]:
    """Share of facts the summary states, per fact and whether all hold."""
    held = {f.name: f.holds(summary) for f in facts}
    return {
        "facts": held,
        "fidelity": sum(held.values()) / len(held) if held else math.nan,
        "all": all(held.values()) if held else False,
    }
