"""R10 step 4: the LLM conditions (log specialists, centres, single model) on RE3 incidents.

Conditions (the ``model`` field of a record is the log-specialist model; ``none`` for non-LLM
systems, see :mod:`.log_triage4`):

* ``team4_rule``: metric specialists (code) + LLM log specialists, rule centre. Every
  weighting of :data:`..envs.logtriage.team4.RULE_WEIGHTS` is stored in
  ``extra["rank_by_rule"]``; the report picks one on the dev split.
* ``team4_llm``: the same reports read by an LLM centre (Ollama).
* ``single_all`` / ``single_logs``: one LLM reads a selected view of all services at the
  same context budget (with / without the service metrics).
* ablations ``team4_rule:<mode>`` / ``team4_llm:<mode>`` with mode ``drop_root``,
  ``shuffle_root`` (and, for the rule centre, ``drop_other``, ``shuffle_other``): one
  service's log report is emptied or replaced by the same service's report from an
  incident in which it was not the root cause.
"""

import random
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

from ..envs.logtriage import messages as msgs
from ..envs.logtriage import rcaeval, views
from ..envs.logtriage import team as tm
from ..envs.logtriage import team4 as t4
from ..envs.logtriage.incident import Incident
from . import log_triage as lt
from .log_triage4 import NO_FAULT, tie_ranking

TEAM4 = ("team4_rule", "team4_llm")
SINGLE4 = ("single_all", "single_logs")
ABLATIONS4 = ("drop_root", "shuffle_root", "drop_other", "shuffle_other")
LLM_ABLATIONS4 = ("drop_root", "shuffle_root")

Pool4 = dict[str, list[tuple[dict[str, Any], dict[str, Any], bool]]]


@dataclass
class Settings4:
    """Budgets of a step-4 run (token counts use the run's :class:`TokenCounter`)."""

    log_view_tokens: int = 1500
    log_new_tokens: int = 120
    log_batch: int = 4
    before_s: float = 120.0
    after_s: float = 600.0
    ctx_tokens: int = 6144
    margin: int = 48
    centre_new_tokens: int = 60


def metric_messages(inc: Incident) -> list[dict[str, Any]]:
    """One metric message per service (code; each reads only its own metrics)."""
    return [tm.metric_message(inc, s) for s in range(inc.n_services)]


def log_messages4(
    inc: Incident,
    metric_msgs: Sequence[dict[str, Any]],
    reasoner: lt.Reasoner,
    cfg: Settings4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One typed log report per service from ``reasoner``; each reads only its own logs."""
    counter = reasoner.counter
    prompts = []
    for s in range(inc.n_services):
        view = t4.log4_view(
            inc,
            s,
            metric_msgs[s]["onset_s"],
            cfg.log_view_tokens,
            counter,
            cfg.before_s,
            cfg.after_s,
        )
        prompts.append(t4.log4_prompt(inc.services[s], view))
    info: dict[str, Any] = {"prompt": 0, "new": 0, "valid": 0, "latency": 0.0}
    written: list[lt.Written] = []
    start = time.perf_counter()
    for lo in range(0, len(prompts), cfg.log_batch):
        written += reasoner.write(
            prompts[lo : lo + cfg.log_batch],
            cfg.log_new_tokens,
            True,
            schema=t4.LOG4_SCHEMA,
        )
    info["latency"] = time.perf_counter() - start
    out = []
    for s, w in enumerate(written):
        info["prompt"] += w.n_prompt
        info["new"] += w.n_new
        clean = t4.clean_log4_message(msgs.parse_json_object(w.text), inc.services[s])
        if clean is None:
            out.append(t4.empty_log4_message(inc.services[s]))
        else:
            out.append(clean)
            info["valid"] += 1
    return out, info


def rule_rankings(
    inc: Incident, mm: Sequence[dict[str, Any]], lm: Sequence[dict[str, Any]]
) -> dict[str, list[str]]:
    """Service ranking of every rule-centre weighting (ties hashed by incident id)."""
    out = {}
    for rule in t4.RULE_WEIGHTS:
        order = tie_ranking(t4.fused_scores(mm, lm, rule), inc, rule)
        out[rule] = [inc.services[i] for i in order]
    return out


def llm_centre(
    inc: Incident,
    mm: Sequence[dict[str, Any]],
    lm: Sequence[dict[str, Any]],
    reasoner: lt.Reasoner,
    cfg: Settings4,
) -> tuple[list[str], bool, int, int]:
    """(ranking, usable, prompt tokens, new tokens) of an Ollama centre on the typed reports."""
    fixed = (
        reasoner.counter.count(t4.centre_prompt(""))
        + reasoner.wrapper_tokens()
        + cfg.margin
    )
    body = t4.centre_body(mm, lm, reasoner.counter, cfg.ctx_tokens - fixed)
    (res,) = reasoner.backend.generate(
        [t4.centre_prompt(body)],
        system=lt.SYSTEM_PROMPT,
        schema=t4.centre_schema(inc.services),
        max_new_tokens=cfg.centre_new_tokens,
        options={"num_ctx": int(cfg.ctx_tokens + 1024)},
    )
    ranking, ok = t4.parse_ranking(res.text, inc.services)
    return ranking, ok, res.n_prompt_tokens, res.n_new_tokens


def team_records4(
    inc: Incident,
    model: str,
    mm: Sequence[dict[str, Any]],
    lm: Sequence[dict[str, Any]],
    spec_info: dict[str, Any],
    centre_reasoner: Optional[lt.Reasoner],
    cfg: Settings4,
    pool: Optional[Pool4] = None,
    ablate: bool = True,
    rule: bool = True,
) -> list[lt.Record]:
    """The rule team (every weighting stored), the LLM-centre team and their ablations.

    ``model`` is the log specialist's name. The LLM centre only runs when
    ``centre_reasoner`` is given; its ablations are limited to the test split.
    """
    nbytes = sum(tm.nbytes(m) for m in mm) + sum(tm.nbytes(m) for m in lm)
    base = {
        "bytes": nbytes,
        "valid_log_messages": round(
            spec_info.get("valid", 0) / max(inc.n_services, 1), 3
        ),
    }
    records: list[lt.Record] = []

    def rule_record(cond: str, mm_: Any, lm_: Any, extra: dict[str, Any]) -> lt.Record:
        ranks = rule_rankings(inc, mm_, lm_)
        return lt.score_record(
            inc,
            model,
            cond,
            ranks[t4.DEFAULT_RULE],
            NO_FAULT,
            prompt_tokens=spec_info["prompt"],
            new_tokens=spec_info["new"],
            latency_s=spec_info["latency"],
            extra=dict(base, rank_by_rule=ranks, **extra),
        )

    if rule:
        records.append(rule_record("team4_rule", mm, lm, {}))
    if centre_reasoner is not None:
        t0 = time.perf_counter()
        ranking, ok, n_p, n_n = llm_centre(inc, mm, lm, centre_reasoner, cfg)
        records.append(
            lt.score_record(
                inc,
                model,
                "team4_llm",
                ranking,
                NO_FAULT,
                prompt_tokens=spec_info["prompt"] + n_p,
                new_tokens=spec_info["new"] + n_n,
                centre_tokens=n_p,
                latency_s=spec_info["latency"] + time.perf_counter() - t0,
                valid=int(ok),
                extra=dict(base, centre_model=centre_reasoner.name),
            )
        )
    if not ablate:
        return records
    rng = random.Random(zlib.crc32(inc.id.encode()))
    root = inc.services.index(inc.root_service)
    others = [i for i in range(inc.n_services) if i != root]
    picks = {"root": root, "other": rng.choice(others)}
    for label, idx in picks.items():
        for mode in ("drop", "shuffle"):
            name = inc.services[idx]
            lm2 = list(lm)
            if mode == "drop":
                lm2[idx] = t4.empty_log4_message(name)
            else:
                # a report this service sent in an incident where it was *not* the root
                alt = [
                    p
                    for p in (pool or {}).get(name, [])
                    if not p[2] and p[1] != lm[idx]
                ]
                if not alt:
                    continue
                lm2[idx] = dict(rng.choice(alt)[1], service=name)
            tag = f"{mode}_{label}"
            if rule and tag in ABLATIONS4:
                records.append(
                    rule_record(f"team4_rule:{tag}", mm, lm2, {"ablated_service": name})
                )
            if (
                centre_reasoner is not None
                and tag in LLM_ABLATIONS4
                and inc.rep not in rcaeval.DEV_REPS
            ):
                ranking, ok, n_p, _ = llm_centre(inc, mm, lm2, centre_reasoner, cfg)
                records.append(
                    lt.score_record(
                        inc,
                        model,
                        f"team4_llm:{tag}",
                        ranking,
                        NO_FAULT,
                        centre_tokens=n_p,
                        prompt_tokens=n_p,
                        valid=int(ok),
                        extra={"ablated_service": name},
                    )
                )
    return records


def build_pool4(
    per_incident: dict[
        str, tuple[Sequence[str], list[dict[str, Any]], list[dict[str, Any]]]
    ],
    roots: dict[str, str],
) -> Pool4:
    """service -> [(metric message, log message, was root)] over incidents.

    ``roots`` maps an incident id to its root-cause service. The shuffle ablations only
    draw reports sent in incidents where the service was not the root (a report from
    another incident of the same faulty service would restore the signal).
    """
    pool: Pool4 = {}
    for iid, (services, mm, lm) in per_incident.items():
        for i, name in enumerate(services):
            pool.setdefault(name, []).append((mm[i], lm[i], roots.get(iid) == name))
    return pool


def run_single4(
    inc: Incident, condition: str, reasoner: lt.Reasoner, cfg: Settings4
) -> lt.Record:
    """One model reads the selected view of all services (``single_logs``: logs only)."""
    if condition not in SINGLE4:
        raise ValueError(f"unknown single condition {condition!r}")
    counter = reasoner.counter
    fixed = (
        counter.count(t4.single_prompt("", inc.services)) + reasoner.wrapper_tokens()
    )
    budget = cfg.ctx_tokens - fixed - cfg.margin
    body = views.selected_view(
        inc, budget, counter, use_metrics=(condition == "single_all")
    )
    t0 = time.perf_counter()
    (res,) = reasoner.backend.generate(
        [t4.single_prompt(body, inc.services)],
        system=lt.SYSTEM_PROMPT,
        schema=t4.centre_schema(inc.services),
        max_new_tokens=cfg.centre_new_tokens,
        options={"num_ctx": int(cfg.ctx_tokens + 1024)},
    )
    ranking, ok = t4.parse_ranking(res.text, inc.services)
    return lt.score_record(
        inc,
        reasoner.name,
        condition,
        ranking,
        NO_FAULT,
        prompt_tokens=res.n_prompt_tokens,
        new_tokens=res.n_new_tokens,
        centre_tokens=res.n_prompt_tokens,
        latency_s=time.perf_counter() - t0,
        valid=int(ok),
        extra={"ctx": cfg.ctx_tokens},
    )
