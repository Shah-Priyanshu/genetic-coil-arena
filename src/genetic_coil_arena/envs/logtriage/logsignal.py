"""Non-LLM log baselines built on *new* templates, and the RE3 log-signal gate statistics.

RE2's resource faults leave no trace in templated logs, but RE3 injects code-level faults
(wrong parameters, missing calls, exceptions) that make the root service log lines it has
never logged before. A strong simple baseline therefore scores each service by the
warning/error-like templates that first appear after the alert:

* ``new_err``: how many such templates;
* ``new_err_mass``: the same, weighted by severity and by log(1 + occurrences);
* ``new_err_idf``: ``new_err_mass`` with each template discounted by the number of services
  in which the same text is new (an error that fans out to callers is not evidence for any one);
* ``new_any_mass``: all new templates, whatever their severity;
* ``new_ext``: new warning/error-like or lifecycle (start-up, shutdown, traceback) templates.

:func:`fuse_ranks` combines a metric and a log score by a weighted sum of reciprocal ranks,
the strongest non-LLM way to use both sources.
"""

import re
from collections import Counter
from typing import Any

import numpy as np

from .incident import Incident

LOG_SCORERS = (
    "new_err",
    "new_err_mass",
    "new_err_idf",
    "new_any_mass",
    "new_ext",
)
# crash and restart traces that the plain severity rules miss (a crash-looping service logs
# its start-up lines again and a Python stack trace has no ERROR level)
_LIFECYCLE = re.compile(
    r"traceback|shutting down|shut down|starting|started|not implemented",
    re.IGNORECASE,
)


def _new_mask(inc: Incident) -> np.ndarray:
    st = inc.stats()
    return (st.pre == 0) & (st.post > 0)


def new_template_scores(inc: Incident, mode: str = "new_err_idf") -> np.ndarray:
    """Score of every service (higher = more likely the root cause), shape (S,)."""
    if mode not in LOG_SCORERS:
        raise ValueError(f"unknown log scorer {mode!r}; choose from {LOG_SCORERS}")
    st = inc.stats()
    new = _new_mask(inc)
    out = np.zeros(inc.n_services)
    if not new.any():
        return out
    sev = st.sev.astype(np.float64)
    mass = np.log1p(st.post.astype(np.float64))
    if mode == "new_err":
        w = (new & (sev >= 1)).astype(np.float64)
    elif mode == "new_err_mass":
        w = np.where(new & (sev >= 1), sev * mass, 0.0)
    elif mode == "new_ext":
        life = np.array([bool(_LIFECYCLE.search(inc.templates[t])) for t in st.tpl])
        w = np.where(new & ((sev >= 1) | life), 1.0, 0.0)
    elif mode == "new_any_mass":
        w = np.where(new, (1.0 + sev) * mass, 0.0)
    else:
        texts = [inc.templates[t] for t in st.tpl]
        spread = Counter(texts[k] for k in np.flatnonzero(new & (sev >= 1)))
        w = np.zeros(len(st))
        for k in np.flatnonzero(new & (sev >= 1)):
            w[k] = sev[k] * mass[k] / spread[texts[k]]
    np.add.at(out, st.svc, w)
    return out


def reciprocal_ranks(scores: np.ndarray) -> np.ndarray:
    """1 / rank of every service (rank 1 = highest score; ties share the best rank)."""
    s = np.asarray(scores, dtype=np.float64)
    order = -s
    rank = np.empty(s.size)
    uniq = np.sort(np.unique(order))
    pos = np.searchsorted(uniq, order)
    first = np.zeros(uniq.size)
    counts = np.bincount(pos, minlength=uniq.size)
    first[1:] = np.cumsum(counts)[:-1]
    rank[:] = first[pos] + 1
    return 1.0 / rank


def fuse_ranks(
    metric_scores: np.ndarray, log_scores: np.ndarray, log_weight: float
) -> np.ndarray:
    """Weighted sum of reciprocal ranks; a service scoring zero in a source gets no credit."""
    m = np.where(np.asarray(metric_scores) > 0, reciprocal_ranks(metric_scores), 0.0)
    g = np.where(np.asarray(log_scores) > 0, reciprocal_ranks(log_scores), 0.0)
    return m + log_weight * g


def gate_stats(inc: Incident) -> dict[str, Any]:
    """Per-incident evidence that the logs carry root-cause signal.

    ``root_new_err`` counts the warning/error-like templates new to the root service after
    the alert; ``others_mean`` is the same for the other services; ``root_rank`` is the
    root's rank by that count (ties share the best rank) and ``root_strictly_first`` whether
    it beats every other service outright.
    """
    st = inc.stats()
    new_err = _new_mask(inc) & (st.sev >= 1)
    root = inc.services.index(inc.root_service)
    per = np.bincount(st.svc[new_err], minlength=inc.n_services)
    others = np.delete(per, root)
    return {
        "id": inc.id,
        "system": inc.system,
        "root_new_err": int(per[root]),
        "others_mean": float(others.mean()) if others.size else 0.0,
        "root_rank": int(1 + (others > per[root]).sum()),
        "root_strictly_first": bool(others.size and per[root] > others.max()),
        "root_logs_lines": int((inc.svc == root).sum()),
    }
