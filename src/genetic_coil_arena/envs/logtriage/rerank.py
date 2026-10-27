"""R10 step 5 (goal G4e): evidence blocks and the closed-set prompt of an LLM final reader.

The non-LLM fusion ranks the services; an LLM then reads a *shortlist* (its top three,
unchanged) and names the root cause. Each candidate gets a compact block, packed into a
fixed token budget:

* the fusion's own rank and score (optional, hidden in the blind ablation);
* the metric specialist's anomaly score, onset and top metrics (as in the typed reports);
* the warning/error/lifecycle log templates that are *new after the alert* (absent before it)
  with their counts, then a few of the service's usual templates (pre-alert) for contrast.

Everything here is domain code for incident triage; the reply schema is a closed set.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from . import logsignal as lg
from . import team as tm
from .incident import Incident
from .views import TokenCounter

SHORTLIST = 3
TOTAL_TOKENS = 1500
NEW_CHARS = 150
USUAL_CHARS = 100
MAX_USUAL = 3
CONFIDENCE_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
_SEV = {0: "info", 1: "warn", 2: "error"}


def choice_schema(names: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["root", "confidence"],
        "properties": {
            "root": {"enum": list(names)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }


def parse_choice(text: str, names: Sequence[str]) -> tuple[Optional[str], float]:
    """(chosen service or None if the reply is unusable, confidence clipped to [0, 1])."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return None, 0.0
    if not isinstance(parsed, dict) or parsed.get("root") not in names:
        return None, 0.0
    conf = parsed.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        return None, 0.0
    return str(parsed["root"]), float(min(max(conf, 0.0), 1.0))


def _fit(
    rows: Sequence[str], budget: int, counter: TokenCounter, used: int
) -> list[str]:
    out = []
    for row in rows:
        cost = counter.count(row) + 1
        if used + cost > budget:
            continue
        out.append(row)
        used += cost
    return out


def metric_line(msg: Mapping[str, Any]) -> str:
    if float(msg["score"]) < tm.METRIC_MIN_Z:
        return "metrics: no deviation outside the normal range"
    onset = msg.get("onset_s")
    top = "; ".join(f"{t['metric']} {t['dir']} z={t['z']}" for t in msg.get("top", []))
    when = "?" if onset is None else f"{onset:+.0f}s"
    return (
        f"metrics: anomaly score {msg['score']} (IQR units), first left the normal "
        f"range at {when}" + (f"; top: {top}" if top else "")
    )


def evidence_block(
    inc: Incident,
    service: int,
    metric_msg: Mapping[str, Any],
    rank: Optional[int],
    score: Optional[float],
    budget: int,
    counter: TokenCounter,
) -> str:
    """One candidate's block (``rank``/``score`` None hides the ranker's opinion)."""
    name = inc.services[service]
    head = f"== {name}"
    if rank is not None and score is not None:
        head += f" (ranker position {rank}, ranker score {score:.2f})"
    head_lines = [head, metric_line(metric_msg)]
    st = inc.stats()
    mine = st.svc == service
    new = mine & (st.pre == 0) & (st.post > 0)
    life = np.array(
        [lg.is_lifecycle(inc.templates[int(t)]) for t in st.tpl], dtype=bool
    )
    sev = st.sev
    key = new & ((sev >= 1) | life)
    idx = np.flatnonzero(key)
    weight = np.maximum(sev[idx], 1) * np.log1p(st.post[idx])
    idx = idx[np.argsort(-weight, kind="stable")]
    n_other_new = int((new & ~key).sum())
    used = counter.count("\n".join(head_lines))
    # contrast rows are capped at a quarter of the block so new templates keep the room
    usual_idx = np.flatnonzero(mine & (st.pre > 0))
    usual_idx = usual_idx[np.argsort(-st.pre[usual_idx], kind="stable")][:MAX_USUAL]
    usual_rows = [
        f"  {int(st.pre[k])} -> {int(st.post[k])}: "
        f"{inc.templates[int(st.tpl[k])][:USUAL_CHARS]}"
        for k in usual_idx
    ]
    usual_rows = _fit(usual_rows, used + budget // 4, counter, used)
    new_rows = []
    for k in idx:
        label = "lifecycle" if (life[k] and sev[k] < 1) else _SEV[int(sev[k])]
        text = inc.templates[int(st.tpl[k])][:NEW_CHARS]
        new_rows.append(f"  x{int(st.post[k])} [{label}]: {text}")
    new_head = (
        "log templates NEW after the alert (never logged before it): "
        f"{len(idx)} warning/error/lifecycle"
        + (f", plus {n_other_new} plain" if n_other_new else "")
    )
    used += counter.count(new_head) + sum(counter.count(r) + 1 for r in usual_rows)
    shown = _fit(new_rows, budget, counter, used)
    lines = head_lines + [new_head] + shown
    if len(shown) < len(idx):
        lines.append(f"  (+{len(idx) - len(shown)} more not shown)")
    if usual_rows:
        lines.append("usual templates (count before alert -> after):")
        lines += usual_rows
    return "\n".join(lines)


def evidence_body(
    inc: Incident,
    order: Sequence[int],
    scores: Sequence[float],
    metric_msgs: Sequence[Mapping[str, Any]],
    counter: TokenCounter,
    show_rank: bool = True,
    total: int = TOTAL_TOKENS,
    ranks: Optional[Sequence[int]] = None,
) -> str:
    """The candidate blocks in presentation order. ``ranks`` are the ranker positions
    (1-based) of ``order``'s services; default 1, 2, 3 ... in that order."""
    per = total // max(len(order), 1)
    pos = list(ranks) if ranks else list(range(1, len(order) + 1))
    blocks = [
        evidence_block(
            inc,
            s,
            metric_msgs[s],
            pos[j] if show_rank else None,
            float(scores[s]) if show_rank else None,
            per,
            counter,
        )
        for j, s in enumerate(order)
    ]
    return "\n\n".join(blocks)


def choice_prompt(body: str, names: Sequence[str], ordered: bool) -> str:
    how = (
        "They are listed best first by an automatic ranker that combines metric "
        "anomaly scores with counts of new log templates; its rank is a strong prior but "
        "it is wrong in some incidents."
        if ordered
        else "They are listed in arbitrary order."
    )
    return (
        "An alert fired at 0s. A software fault (a code-level change such as a wrong "
        "parameter, a missing call or an unhandled exception) was injected into exactly "
        f"one service. {len(names)} candidate services remain. {how}\n"
        "Evidence per candidate: metric deviation, log templates that are new after the "
        "alert, and a few of the service's usual templates. The faulty service itself "
        "usually logs new errors, exceptions or restarts of its own; a healthy caller "
        "also logs new errors when a dependency fails (timeouts, refused connections, "
        "bad responses naming another service), so look at what the new lines say.\n\n"
        f"{body}\n\nWhich candidate is the root cause (the service the fault was injected "
        f'into)? Reply as JSON: {{"root": one of {", ".join(names)}, "confidence": '
        "a number from 0 to 1}."
    )
