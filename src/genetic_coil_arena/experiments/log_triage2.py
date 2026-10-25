"""R10 step 2: a heterogeneous team (metric specialists + log specialists + a centre).

Conditions (the model field of a record is the log-specialist model for hybrid teams, the
centre model for an LLM centre without log specialists, ``none`` when no model is used):

* ``baro_metrics``: BARO on all service metrics plus the metric-kind fault heuristic.
* ``single_selected`` / ``oracle_selected``: step 1's single model (:mod:`.log_triage`).
* ``team_metric_rule``: metric specialists only, rule centre (distributed BARO).
* ``team_metric_llm``: metric specialists only, LLM centre reading the typed messages.
* ``team_hybrid_rule``: metric + log specialists, rule centre.
* ``team_hybrid_llm``: metric + log specialists, LLM centre.
* ablations ``<team>:drop_root|shuffle_root|drop_other|shuffle_other``: the messages of
  one service (both kinds) are emptied or replaced by another incident's.

The rule centre ranks by the metric score, so a rule-centre team's top-1 equals BARO's;
what the log specialists can change is the fault type. Every fault policy of the rule centre
is stored in ``extra["fault_policies"]``; the report picks one on the dev split.
"""

import random
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from ..envs.logtriage import heuristics as hz
from ..envs.logtriage import messages as msgs
from ..envs.logtriage import team as tm
from ..envs.logtriage.incident import Incident
from ..envs.logtriage.rcaeval import FAULT_TYPES
from ..envs.logtriage.views import TokenCounter, alert_header
from . import log_triage as lt

CONDITIONS2 = (
    "baro_metrics",
    "single_selected",
    "oracle_selected",
    "team_metric_rule",
    "team_metric_llm",
    "team_hybrid_rule",
    "team_hybrid_llm",
)
TEAM2 = ("team_metric_rule", "team_metric_llm", "team_hybrid_rule", "team_hybrid_llm")
ABLATIONS2 = ("drop_root", "shuffle_root", "drop_other", "shuffle_other")
LLM_ABLATIONS = ("drop_root", "shuffle_root")

Pool = dict[str, list[tuple[dict[str, Any], Optional[dict[str, Any]]]]]


@dataclass
class Settings2:
    """Budgets of a step-2 run."""

    log_view_tokens: int = 1500
    log_new_tokens: int = 120
    log_batch: int = 4
    before_s: float = 120.0
    after_s: float = 300.0
    policy: str = tm.DEFAULT_POLICY
    ctx_tokens: int = 4096
    margin: int = 48
    llm_ablate_dev: bool = False  # LLM-centre ablations on the dev split too


# ------------------------------------------------------------------- specialists


def metric_messages(inc: Incident) -> list[dict[str, Any]]:
    """One metric message per service (code; each reads only its own service)."""
    return [tm.metric_message(inc, s) for s in range(inc.n_services)]


def log_messages(
    inc: Incident,
    metric_msgs: Sequence[dict[str, Any]],
    reasoner: lt.Reasoner,
    cfg: Settings2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One log message per service from ``reasoner``, each reading only its own logs
    around the onset its metric specialist reported (the centre relays the onset)."""
    counter = reasoner.counter
    prompts = []
    for s in range(inc.n_services):
        view = tm.log_view(
            inc,
            s,
            metric_msgs[s]["onset_s"],
            cfg.log_view_tokens,
            counter,
            cfg.before_s,
            cfg.after_s,
        )
        prompts.append(tm.log_prompt(inc.services[s], view))
    info: dict[str, Any] = {"prompt": 0, "new": 0, "valid": 0, "latency": 0.0}
    written: list[lt.Written] = []
    start = time.perf_counter()
    for lo in range(0, len(prompts), cfg.log_batch):
        written += reasoner.write(
            prompts[lo : lo + cfg.log_batch],
            cfg.log_new_tokens,
            True,
            schema=tm.LOG_SCHEMA,
        )
    info["latency"] = time.perf_counter() - start
    out = []
    for s, w in enumerate(written):
        info["prompt"] += w.n_prompt
        info["new"] += w.n_new
        clean = tm.clean_log_message(msgs.parse_json_object(w.text), inc.services[s])
        if clean is None:
            out.append(tm.empty_log_message(inc.services[s]))
        else:
            out.append(clean)
            info["valid"] += 1
    return out, info


# ----------------------------------------------------------------------- centres


def centre_body(
    inc: Incident,
    metric_msgs: Sequence[dict[str, Any]],
    log_msgs: Optional[Sequence[dict[str, Any]]],
    counter: TokenCounter,
    budget: int,
) -> str:
    """The LLM centre's evidence: one line of typed reports per service.

    If it exceeds ``budget``, log evidence is dropped first, then log reports.
    """
    head = (
        alert_header(inc).split(" Services with logs")[0]
        + " You cannot read the metrics or logs yourself. Each service has "
        "monitors that sent typed JSON reports. metrics: score = the largest deviation "
        "of any of the service's metrics from its normal range (IQR units, higher is "
        "more anomalous); onset_s = when its metrics first left the normal range "
        "(seconds after the alert); top = its most deviating metrics (direction, z)."
        + (
            " logs: the fault type the service's log monitor suspects, its "
            "confidence and evidence (unknown means its logs told it nothing)."
            if log_msgs is not None
            else ""
        )
        + " A fault spreads: services that depend on the root cause deviate too, but "
        "later and weaker, so pick the root-cause service from the metric reports "
        "(highest score, earliest onset)."
        + (
            " Use the log reports only for the fault type, and trust them only when "
            "their confidence is high; otherwise infer the fault type from the "
            "metric kinds (cpu, mem, diskio, socket, latency, error)."
            if log_msgs is not None
            else " Infer the fault type from the metric kinds (cpu, mem, diskio, "
            "socket, latency, error)."
        )
        + "\n"
    )
    variants: list[Optional[list[dict[str, Any]]]] = [log_msgs]
    if log_msgs is not None:
        variants.append([dict(m, evidence=[]) for m in log_msgs])
        variants.append(None)
    body = head
    for logs in variants:
        lines = [
            f"- {name}: {tm.centre_line(m, logs[i] if logs is not None else None)}"
            for i, (name, m) in enumerate(zip(inc.services, metric_msgs))
        ]
        body = head + "\n".join(lines)
        if counter.count(body) <= budget:
            break
    return body


def llm_decide(
    inc: Incident,
    metric_msgs: Sequence[dict[str, Any]],
    log_msgs: Optional[Sequence[dict[str, Any]]],
    reasoner: lt.Reasoner,
    cfg: Settings2,
) -> lt.Decision:
    budget = lt.body_budget(reasoner, inc.services, cfg.ctx_tokens, cfg.margin)
    body = centre_body(inc, metric_msgs, log_msgs, reasoner.counter, budget)
    (dec,) = reasoner.decide([body], [inc.services], cfg.ctx_tokens)
    return dec


# -------------------------------------------------------------------- conditions


def run_baro(inc: Incident) -> lt.Record:
    """BARO on all service metrics with the metric-kind fault heuristic."""
    order = hz.ranking(hz.baro_metrics(inc))
    top = inc.services[order[0]]
    rows = inc.metric_rows(top, min_z=0.0, top=1)
    fault = (
        hz.METRIC_FAULT.get(rows[0][0], tm.DEFAULT_FAULT) if rows else tm.DEFAULT_FAULT
    )
    return lt.score_record(
        inc, "none", "baro_metrics", [inc.services[i] for i in order], fault
    )


def message_bytes(
    metric_msgs: Sequence[dict[str, Any]], log_msgs: Optional[Sequence[dict[str, Any]]]
) -> dict[str, int]:
    mb = sum(tm.nbytes(m) for m in metric_msgs)
    lb = sum(tm.nbytes(m) for m in log_msgs) if log_msgs is not None else 0
    return {"bytes_metric": mb, "bytes_log": lb, "bytes": mb + lb}


def _decide(
    inc: Incident,
    centre: str,
    mm: Sequence[dict[str, Any]],
    lm: Optional[Sequence[dict[str, Any]]],
    reasoner: Optional[lt.Reasoner],
    cfg: Settings2,
) -> tuple[list[str], str, dict[str, Any], lt.Decision]:
    """(ranking, fault, extra, decision) of one centre on the given messages."""
    if centre == "rule":
        ranking, faults = tm.rule_decide(inc.services, mm, lm, cfg.policy)
        dec = lt.Decision(ranking, faults["chosen"], n_prompt=0, n_new=0)
        return ranking, faults["chosen"], {"fault_policies": faults}, dec
    assert reasoner is not None
    dec = llm_decide(inc, mm, lm, reasoner, cfg)
    return dec.ranking, dec.fault, {}, dec


def team_records(
    inc: Incident,
    team: str,
    metric_msgs: Sequence[dict[str, Any]],
    log_msgs: Optional[Sequence[dict[str, Any]]],
    spec_model: str,
    spec_info: Optional[dict[str, Any]],
    centre_reasoner: Optional[lt.Reasoner],
    cfg: Settings2,
    pool: Optional[Pool] = None,
    ablate: bool = True,
) -> list[lt.Record]:
    """The team condition and its message ablations.

    ``team`` is one of :data:`TEAM2`; ``log_msgs`` is None for the metric-only teams.
    ``spec_model`` is the record's model field. The LLM centre's ablations are limited to
    :data:`LLM_ABLATIONS` and to the test split (unless ``cfg.llm_ablate_dev``).
    """
    centre = "rule" if team.endswith("_rule") else "llm"
    hybrid = team.startswith("team_hybrid")
    if hybrid != (log_msgs is not None):
        raise ValueError(
            f"{team}: log messages {'missing' if hybrid else 'not wanted'}"
        )
    spec_info = spec_info or {"prompt": 0, "new": 0, "valid": 0, "latency": 0.0}
    t0 = time.perf_counter()
    ranking, fault, extra, dec = _decide(
        inc, centre, metric_msgs, log_msgs, centre_reasoner, cfg
    )
    centre_time = time.perf_counter() - t0
    extra = dict(
        extra,
        **message_bytes(metric_msgs, log_msgs),
        centre_model=centre_reasoner.name if centre_reasoner else "rule",
        valid_log_messages=(
            round(spec_info.get("valid", 0) / max(inc.n_services, 1), 3)
            if hybrid
            else None
        ),
    )
    records = [
        lt.score_record(
            inc,
            spec_model,
            team,
            ranking,
            fault,
            prompt_tokens=spec_info["prompt"] + dec.n_prompt,
            new_tokens=spec_info["new"] + dec.n_new,
            centre_tokens=dec.n_prompt,
            latency_s=spec_info["latency"] + centre_time,
            valid=int(dec.valid),
            extra=extra,
        )
    ]
    if not ablate:
        return records
    modes = ABLATIONS2
    if centre == "llm":
        if inc.rep == 1 and not cfg.llm_ablate_dev:
            return records
        modes = LLM_ABLATIONS
    rng = random.Random(zlib.crc32(inc.id.encode()))
    root = inc.services.index(inc.root_service)
    others = [i for i in range(inc.n_services) if i != root]
    picks = {"root": root, "other": rng.choice(others)}
    for label, idx in picks.items():
        for mode in ("drop", "shuffle"):
            if f"{mode}_{label}" not in modes:
                continue
            mm = list(metric_msgs)
            lm = list(log_msgs) if log_msgs is not None else None
            name = inc.services[idx]
            if mode == "drop":
                mm[idx] = tm.empty_metric_message(name)
                if lm is not None:
                    lm[idx] = tm.empty_log_message(name)
            else:
                alt = [
                    p
                    for p in (pool or {}).get(name, [])
                    if p[0] != metric_msgs[idx]
                    or p[1] != (log_msgs[idx] if log_msgs else None)
                ]
                if alt:
                    a_m, a_l = rng.choice(alt)
                    mm[idx] = dict(a_m, service=name)
                    if lm is not None and a_l is not None:
                        lm[idx] = dict(a_l, service=name)
            r, f, ex, d = _decide(inc, centre, mm, lm, centre_reasoner, cfg)
            records.append(
                lt.score_record(
                    inc,
                    spec_model,
                    f"{team}:{mode}_{label}",
                    r,
                    f,
                    prompt_tokens=d.n_prompt,
                    centre_tokens=d.n_prompt,
                    valid=int(d.valid),
                    extra=dict(ex, ablated_service=name),
                )
            )
    return records


def build_pool(
    per_incident: dict[
        str, tuple[Sequence[str], list[dict[str, Any]], Optional[list[dict[str, Any]]]]
    ],
) -> Pool:
    """service -> [(metric message, log message or None)] over incidents, for shuffles."""
    pool: Pool = {}
    for services, mm, lm in per_incident.values():
        for i, name in enumerate(services):
            pool.setdefault(name, []).append((mm[i], lm[i] if lm else None))
    return pool


FAULT_SET = set(FAULT_TYPES)
