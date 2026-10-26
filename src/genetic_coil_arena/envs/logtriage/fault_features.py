"""Features of one service's metrics for a trained fault-type specialist (R10 step 3).

The inputs are the per-kind rows the converter stores (``[z, direction, first_t, rel]``,
see :func:`rcaeval.metric_features`). Features are indexed by metric *kind*, never by metric
name, so a model trained on Train Ticket applies to Sock Shop and Online Boutique. A
specialist reads only its own service's rows.

Per kind: present, log(1+z), signed log(1+z), signed log(1+|relative change|), whether and
how early the metric left its normal range. Across kinds: the strongest kind's log z, each
kind's z as a share of the strongest, a one-hot of the strongest kind, and the number of
kinds with z >= 5 and >= 20.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from .incident import Incident

KINDS = (
    "cpu",
    "mem",
    "socket",
    "diskio",
    "workload",
    "latency-50",
    "latency-90",
    "error",
)
PER_KIND = ("present", "logz", "signed_logz", "signed_logrel", "onset")
N_PER_KIND = len(PER_KIND)
N_FEATURES = len(KINDS) * N_PER_KIND + 1 + 2 * len(KINDS) + 2


def feature_names() -> list[str]:
    names = [f"{k}:{f}" for k in KINDS for f in PER_KIND]
    names += ["max_logz"]
    names += [f"{k}:share" for k in KINDS]
    names += [f"{k}:is_top" for k in KINDS]
    names += ["n_z>=5", "n_z>=20"]
    return names


def row_features(rows: Mapping[str, Any]) -> np.ndarray:
    """Features of one service from its ``{kind: [z, dir, first_t, rel]}`` rows."""
    out = np.zeros(N_FEATURES)
    logz = np.zeros(len(KINDS))
    for i, kind in enumerate(KINDS):
        row = rows.get(kind)
        if row is None:
            continue
        z = max(float(row[0]), 0.0)
        direction = 1.0 if float(row[1]) >= 0 else -1.0
        rel = float(row[3]) if len(row) > 3 and row[3] is not None else 0.0
        first = row[2]
        base = i * N_PER_KIND
        lz = float(np.log1p(min(z, 1e4)))
        logz[i] = lz
        out[base] = 1.0
        out[base + 1] = lz
        out[base + 2] = direction * lz
        out[base + 3] = float(np.sign(rel) * np.log1p(min(abs(rel), 100.0)))
        # earlier onset = larger value; a metric that never left its range scores 0
        out[base + 4] = (
            0.0 if first is None else 1.0 / (1.0 + max(float(first), 0.0) / 30)
        )
    k = len(KINDS) * N_PER_KIND
    top = float(logz.max())
    out[k] = top
    if top > 0:
        out[k + 1 : k + 1 + len(KINDS)] = logz / top
        out[k + 1 + len(KINDS) + int(logz.argmax())] = 1.0
    zs = np.expm1(logz)
    out[-2] = float((zs >= 5).sum())
    out[-1] = float((zs >= 20).sum())
    return out


def service_features(inc: Incident, service: str) -> np.ndarray:
    """Feature vector of ``service`` in ``inc`` (zeros when it has no metrics)."""
    return row_features(inc.metrics.get(service, {}))
