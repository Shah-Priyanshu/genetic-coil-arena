"""Non-LLM root-cause baselines on log features.

* :func:`baro_logs`: an adaptation of BARO (Pham et al., FSE 2024, the strongest simple
  method in RCAEval) to log counts. BARO scales each metric with the median and
  interquartile range of the pre-fault data and ranks services by the largest scaled
  deviation after the alert. Here the "metrics" of a service are binned counts of its
  lines, its error-like lines, its warning-like lines and its never-before-seen templates.
  This is an adaptation to log features, not the published metric-based numbers.
* :func:`surprise_sum`: a service's total positive template surprise.
* :func:`guess_fault`: keyword rules from the top templates to a fault type.
"""

import re
from collections.abc import Sequence
from typing import Optional

import numpy as np

from .incident import Incident

BIN_SECONDS = 10.0
FAULT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "mem",
        re.compile(r"memory|\boom\b|heap|gc overhead|killed|cannot allocate", re.I),
    ),
    (
        "disk",
        re.compile(r"disk|no space|i/o|\bio error|read-only|filesystem|\bfsync", re.I),
    ),
    ("socket", re.compile(r"socket|too many open|address already|bind|eof", re.I)),
    (
        "loss",
        re.compile(
            r"refused|reset|unreachable|packet|connect error|unavailable|"
            r"delayed connect|transport failure|HTTP50[234]|broken pipe",
            re.I,
        ),
    ),
    (
        "delay",
        re.compile(r"timeout|timed out|slow|took|latency|deadline|HTTP504", re.I),
    ),
    ("cpu", re.compile(r"cpu|throttl|busy|load average", re.I)),
)


def _robust_scale(pre: np.ndarray, post: np.ndarray) -> float:
    """Largest robust-scaled absolute deviation of ``post`` from the ``pre`` median."""
    if pre.size == 0 or post.size == 0:
        return 0.0
    med = float(np.median(pre))
    q1, q3 = np.percentile(pre, [25, 75])
    scale = max(float(q3 - q1), 0.1 * abs(med), 1.0)
    return float(np.max(np.abs(post - med)) / scale)


def _binned(t: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(lo, hi + BIN_SECONDS, BIN_SECONDS)
    return np.histogram(t, bins=edges)[0].astype(np.float64), edges


def baro_logs(inc: Incident) -> np.ndarray:
    """Anomaly score of every service (higher = more likely the root cause), shape (S,)."""
    lo, hi = inc.window
    lo = np.floor(lo / BIN_SECONDS) * BIN_SECONDS
    hi = np.ceil(hi / BIN_SECONDS) * BIN_SECONDS
    sev = inc.template_severity[inc.tpl] if inc.n_lines else np.zeros(0, dtype=np.int8)
    pre_lines = inc.t < 0
    key = inc.svc.astype(np.int64) * max(len(inc.templates), 1) + inc.tpl
    pre_keys = set(np.unique(key[pre_lines]).tolist())
    new_flag = np.array([k not in pre_keys for k in key.tolist()], dtype=bool)
    scores = np.zeros(inc.n_services)
    for s in range(inc.n_services):
        m = inc.svc == s
        features = (
            inc.t[m],
            inc.t[m & (sev >= 2)],
            inc.t[m & (sev >= 1)],
            inc.t[m & new_flag & (inc.t >= 0)],
        )
        best = 0.0
        for times in features:
            series, edges = _binned(times, lo, hi)
            centres = (edges[:-1] + edges[1:]) / 2
            best = max(best, _robust_scale(series[centres < 0], series[centres >= 0]))
        scores[s] = best
    return scores


def baro_metrics(inc: Incident) -> np.ndarray:
    """BARO on the service metrics: a service's largest robust deviation after the alert."""
    return np.array(
        [
            max((v[0] for v in inc.metrics.get(name, {}).values()), default=0.0)
            for name in inc.services
        ]
    )


METRIC_FAULT = {
    "cpu": "cpu",
    "mem": "mem",
    "diskio": "disk",
    "socket": "socket",
    "latency-50": "delay",
    "latency-90": "delay",
    "error": "loss",
    "workload": "loss",
}


def fault_from_metrics(inc: Incident, service: str) -> Optional[str]:
    """Fault type implied by the service's strongest deviating metric, if any."""
    rows = inc.metric_rows(service, min_z=5.0, top=1)
    return METRIC_FAULT.get(rows[0][0]) if rows else None


def surprise_sum(inc: Incident) -> np.ndarray:
    """Sum of positive template surprise per service, shape (S,)."""
    st = inc.stats()
    score = np.maximum(st.surprise(), 0.0)
    out = np.zeros(inc.n_services)
    np.add.at(out, st.svc, score)
    return out


def ranking(scores: np.ndarray) -> list[int]:
    """Service indices, best first; ties broken by index (stable)."""
    return [int(i) for i in np.argsort(-np.asarray(scores), kind="stable")]


def guess_fault(
    templates: Sequence[str],
    default: str = "delay",
    weights: Optional[Sequence[float]] = None,
) -> str:
    """Fault type from keyword rules over ``templates`` (weighted by ``weights``)."""
    votes: dict[str, float] = {}
    for i, text in enumerate(templates):
        w = float(weights[i]) if weights is not None and i < len(weights) else 1.0
        for fault, pattern in FAULT_RULES:
            if pattern.search(text):
                votes[fault] = votes.get(fault, 0.0) + w
                break
    if not votes:
        return default
    return max(votes, key=lambda k: votes[k])
