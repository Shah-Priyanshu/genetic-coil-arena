"""What a model is shown: token counting and the ways to fit an incident into a budget.

* :func:`naive_view`: the templated lines in time order, cut at the budget (head or tail).
* :func:`selected_view`: a per-service summary line (line counts and error-like counts
  before and after the alert) plus the templates ranked by surprise (post against pre
  count, weighted by severity), packed to the budget with a guaranteed few per service.
  This is the strong single-model heuristic, and also what a specialist reads from its own
  stream.

Budgets count tokens with a :class:`TokenCounter`; the experiments pass the Qwen tokenizer,
tests and the CPU fallback use :class:`ApproxCounter`.
"""

import math
from collections.abc import Sequence
from typing import Any, Optional, Protocol

import numpy as np

from .incident import Incident


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int:  # pragma: no cover - protocol
        ...


class ApproxCounter:
    """Characters over a constant; deliberately on the high side for log text."""

    def __init__(self, chars_per_token: float = 3.0):
        self.chars_per_token = float(chars_per_token)
        self.name = f"approx{chars_per_token:g}"

    def count(self, text: str) -> int:
        return int(math.ceil(len(text) / self.chars_per_token))


class HFCounter:
    """Token counts from a Hugging Face tokenizer (no special tokens added)."""

    def __init__(self, tokenizer: Any, name: str = "hf"):
        self.tokenizer = tokenizer
        self.name = name

    def count(self, text: str) -> int:
        if len(text) < 20_000:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Long texts line by line in batches: one multi-megabyte string is very slow to
        # tokenise, and a newline is one token in these vocabularies
        lines = text.split("\n")
        total = len(lines) - 1
        for lo in range(0, len(lines), 4096):
            batch = self.tokenizer(lines[lo : lo + 4096], add_special_tokens=False)
            total += sum(len(ids) for ids in batch["input_ids"])
        return total


def alert_header(inc: Incident) -> str:
    lo, hi = inc.window
    return (
        f"Incident window {lo:+.0f}s to {hi:+.0f}s around an alert at 0s. "
        f"Services with logs: {', '.join(inc.services)}."
    )


def metric_text(inc: Incident, service: str) -> str:
    """The service's metrics that left their normal range, strongest first."""
    rows = inc.metric_rows(service)
    if not rows:
        return "no metric left its normal range"
    parts = []
    for kind, (z, direction, first, _rel) in rows:
        when = "" if first is None else f" from {first:+.0f}s"
        parts.append(f"{kind} {'up' if direction > 0 else 'down'} z={z:.0f}{when}")
    return "; ".join(parts)


def service_summary(inc: Incident, use_metrics: bool = True) -> list[str]:
    """One line per service: log volume and error-like lines before and after the alert,
    and (with ``use_metrics``) which of its metrics left their normal range."""
    pre, post = inc.service_counts()
    st = inc.stats()
    err_pre = np.zeros(inc.n_services, dtype=np.int64)
    err_post = np.zeros(inc.n_services, dtype=np.int64)
    sev = st.sev >= 2
    np.add.at(err_pre, st.svc[sev], st.pre[sev])
    np.add.at(err_post, st.svc[sev], st.post[sev])
    return [
        f"{name}: lines before/after {pre[i]}/{post[i]}, error-like before/after "
        f"{err_pre[i]}/{err_post[i]}"
        + (
            f"; metrics: {metric_text(inc, name)}"
            if use_metrics and inc.metrics
            else ""
        )
        for i, name in enumerate(inc.services)
    ]


def render_row(
    inc: Incident, svc: int, tpl: int, pre: int, post: int, first: float
) -> str:
    when = "gone" if not np.isfinite(first) else f"{first:+.0f}s"
    return (
        f"  [{inc.services[svc]}] first {when}, before {pre}, after {post}: "
        f"{inc.templates[tpl]}"
    )


def naive_view(
    inc: Incident, budget: int, counter: TokenCounter, tail: bool = False
) -> str:
    """Templated lines in time order, cut at ``budget`` tokens (head, or the last lines)."""
    taken: list[str] = []
    used = 0
    order: Sequence[int] = (
        range(inc.n_lines - 1, -1, -1) if tail else range(inc.n_lines)
    )
    for i in order:
        line = inc.line_text(i)
        cost = counter.count(line) + 1
        if used + cost > budget:
            break
        taken.append(line)
        used += cost
    if tail:
        taken.reverse()
    return "\n".join(taken)


def full_text(inc: Incident) -> str:
    """Every templated line, in time order (what a model with unlimited context would get)."""
    return "\n".join(inc.line_text(i) for i in range(inc.n_lines))


def _rows(inc: Incident, services: Optional[Sequence[int]]) -> tuple[np.ndarray, Any]:
    st = inc.stats()
    score = st.surprise()
    keep = np.ones(len(st), dtype=bool)
    if services is not None:
        keep = np.isin(st.svc, np.asarray(list(services), dtype=np.int32))
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(-score[idx], kind="stable")]
    return idx, st


def selected_view(
    inc: Incident,
    budget: int,
    counter: TokenCounter,
    services: Optional[Sequence[int]] = None,
    per_service_min: int = 2,
    max_row_chars: int = 200,
    use_metrics: bool = True,
) -> str:
    """The strong heuristic view of ``services`` (default: all) within ``budget`` tokens.

    Layout: a header, then per service its summary line and its chosen template rows. The
    summary lines come first and always fit (if they do not, the later services drop out);
    rows are added in two passes, ``per_service_min`` best rows of every service, then the
    best remaining rows over all services, until the budget is used.
    """
    chosen = list(range(inc.n_services)) if services is None else list(services)
    summaries = service_summary(inc, use_metrics)
    idx, st = _rows(inc, chosen)
    used = counter.count(alert_header(inc)) + 2
    kept_services: list[int] = []
    for s in chosen:
        cost = counter.count(summaries[s]) + 1
        if used + cost > budget:
            break
        used += cost
        kept_services.append(s)
    allowed = set(kept_services)

    picked: list[int] = []
    per_count: dict[int, int] = {}
    # pass 1: the best rows of every service
    for j in idx:
        s = int(st.svc[j])
        if s in allowed and per_count.get(s, 0) < per_service_min:
            picked.append(int(j))
            per_count[s] = per_count.get(s, 0) + 1
    # pass 2: everything else, best first
    firsts = set(picked)
    picked += [
        int(j) for j in idx if int(j) not in firsts and int(st.svc[j]) in allowed
    ]

    rows_by_service: dict[int, list[tuple[int, str]]] = {s: [] for s in kept_services}
    for rank, j in enumerate(picked):
        s = int(st.svc[j])
        text = render_row(
            inc,
            s,
            int(st.tpl[j]),
            int(st.pre[j]),
            int(st.post[j]),
            float(st.first_post[j]),
        )[: max_row_chars + 40]
        cost = counter.count(text) + 1
        if used + cost > budget:
            continue  # this row does not fit; a shorter one further down may
        used += cost
        rows_by_service[s].append((rank, text))

    def best(s: int) -> float:
        rows = rows_by_service[s]
        top_z = max(
            (v[0] for v in inc.metrics.get(inc.services[s], {}).values()), default=0.0
        )
        if use_metrics and top_z >= 3:
            return -top_z
        return min((r for r, _ in rows), default=10**9)

    lines = [alert_header(inc), ""]
    for s in sorted(kept_services, key=best):
        lines.append(summaries[s])
        lines += [text for _, text in sorted(rows_by_service[s])]
    return "\n".join(lines)


def source_view(
    inc: Incident,
    service: int,
    budget: int,
    counter: TokenCounter,
    use_metrics: bool = True,
) -> str:
    """What the specialist of ``service`` reads: only its own logs and metrics, selected
    to fit."""
    return selected_view(
        inc,
        budget,
        counter,
        services=[service],
        per_service_min=3,
        use_metrics=use_metrics,
    )
