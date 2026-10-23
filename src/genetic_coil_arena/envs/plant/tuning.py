"""Choose the centre's hyperparameters on validation runs only (R8 step 2).

A staged coordinate search: one hyperparameter at a time, in the order of ``STAGES``, each
tried at the values in its list with every other setting at its current best; the value
with the best validation macro-F1 is kept. The score of a candidate is the macro-F1 over
the tuning stage's validation windows at the candidate's own calibrated threshold (5% of
the validation normal windows alarm), so a candidate cannot win by alarming more. Values are
listed cheapest first and, among candidates within ``tol`` of the best score, the cheapest
wins, which keeps the search from chasing noise. The test runs of any replicate are never
seen: the search runs on the tuning block of ``rieth.RunSplits`` (its own training and
validation runs).
"""

import time
from typing import Any, Optional

from .replicates import DEFAULT_CENTRE, Prepared, Replicate

# Hyperparameter -> candidate values, cheapest first
STAGES: tuple[tuple[str, tuple[Any, ...]], ...] = (
    ("context", (1, 2, 3, 4, 6)),
    ("hidden", (32, 64, 128)),
    ("l2", (1e-4, 1e-3, 1e-2, 1e-1)),
    ("max_iter", (60, 100, 200)),
)
TOLERANCE = 0.002


def tune_centre(
    replicate: Replicate,
    prep: Prepared,
    start: Optional[dict[str, Any]] = None,
    stages: tuple[tuple[str, tuple[Any, ...]], ...] = STAGES,
    tol: float = TOLERANCE,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the staged search for one cell on the tuning replicate.

    Args:
        replicate: The tuning stage (``Replicate(..., index=None)``)
        prep: The cell's encoder and features (``Replicate.prepare``)
        start: Starting hyperparameters (default ``DEFAULT_CENTRE``)
        stages: ``(name, candidate values)`` in search order
        tol: Candidates within this macro-F1 of the best count as tied; the earliest
            (cheapest) tied candidate wins

    Returns:
        ``(best params, trace)``; the trace lists every fit with its stage, value,
        validation macro-F1 and seconds
    """
    best = dict(start or DEFAULT_CENTRE)
    trace: list[dict[str, Any]] = []
    current: Optional[float] = None
    for name, values in stages:
        scores: list[tuple[Any, float]] = []
        for value in values:
            params = {**best, name: value}
            if current is not None and value == best[name]:
                scores.append((value, current))  # already measured
                continue
            t0 = time.perf_counter()
            centre = replicate.fit_centre(prep, params)
            score = replicate.validation_f1(centre, prep)
            trace.append(
                {
                    "stage": name,
                    "value": value,
                    "params": params,
                    "val_f1": float(score),
                    "seconds": time.perf_counter() - t0,
                }
            )
            scores.append((value, score))
            if current is None and value == best[name]:
                current = score
        top = max(score for _, score in scores)
        pick = next(value for value, score in scores if score >= top - tol)
        best[name] = pick
        current = dict(scores)[pick]
    return best, trace
