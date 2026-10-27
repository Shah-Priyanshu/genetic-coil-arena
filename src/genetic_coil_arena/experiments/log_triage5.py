"""R10 step 5 (goal G4e): an LLM as the *final reader* of the non-LLM fusion's top-3 shortlist.

Step 4 showed that per-service LLM specialists cannot rank (G4d kill); a plain count of new
templates, fused with the metric scores, does (``nonllm_fusion``). Step 5 asks whether an LLM
adds anything at the very end: it gets the fusion's top three services, unchanged, each with a
compact evidence block (:mod:`..envs.logtriage.rerank`), and names the root cause from that
closed set. The rest of the fusion's ranking stands, so top-3 equals the fusion's by
construction and only top-1 can move.

Conditions (``model`` is the reader's name):

* ``rerank3``: shortlist in the fusion's order, with the ranker position and score shown;
* ``rerank3:shuffle``: same blocks, candidate order shuffled (positions still shown);
* ``rerank3:shuffle_blind``: shuffled and the ranker position/score hidden. If a reader only
  copies the first block or the stated rank it collapses here.

A reply that is not valid keeps the fusion's top-1. The confidence threshold variant
(``rerank3_thr``: override the fusion's top-1 only at or above a confidence) is derived by
the report from the stored ``confidence``; the threshold is picked on dev.
"""

import random
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..envs.logtriage import rerank as rr
from ..envs.logtriage import team as tm
from ..envs.logtriage.incident import Incident
from . import log_triage as lt
from . import log_triage4 as l4

CONDITIONS5 = ("rerank3", "rerank3:shuffle", "rerank3:shuffle_blind")
FUSION = "nonllm_fusion"


@dataclass
class Settings5:
    evidence_tokens: int = rr.TOTAL_TOKENS
    ctx_tokens: int = 4096
    new_tokens: int = 40


def fusion_order(
    inc: Incident, parts: dict[str, np.ndarray], weights: dict[str, float]
) -> tuple[list[int], np.ndarray]:
    """(service indices best first, fused scores); same tie rule as ``non_llm_records``."""
    scores = l4.fused_scores(parts, weights)
    return l4.tie_ranking(scores, inc, "fusion"), scores


def shown_order(inc: Incident, top: Sequence[int], condition: str) -> list[int]:
    """The order in which the shortlist is shown (shuffled, never left as it was)."""
    top = list(top)
    if condition == "rerank3":
        return top
    rng = random.Random(zlib.crc32((inc.id + "|shuffle").encode()))
    perm = top[:]
    rng.shuffle(perm)
    if perm == top and len(top) > 1:
        perm = top[1:] + top[:1]
    return perm


def read_shortlist(
    inc: Incident,
    model: str,
    condition: str,
    order: Sequence[int],
    scores: np.ndarray,
    mm: Sequence[dict[str, Any]],
    reasoner: lt.Reasoner,
    cfg: Settings5,
) -> lt.Record:
    """One closed-set call; the record ranks the choice first, then the fusion's order."""
    if condition not in CONDITIONS5:
        raise ValueError(f"unknown condition {condition!r}")
    top = list(order[: rr.SHORTLIST])
    shown = shown_order(inc, top, condition)
    names = [inc.services[s] for s in shown]
    pos = [top.index(s) + 1 for s in shown]
    show_rank = condition != "rerank3:shuffle_blind"
    body = rr.evidence_body(
        inc,
        shown,
        scores,
        mm,
        reasoner.counter,
        show_rank=show_rank,
        total=cfg.evidence_tokens,
        ranks=pos,
    )
    prompt = rr.choice_prompt(body, names, ordered=(condition == "rerank3"))
    t0 = time.perf_counter()
    (res,) = reasoner.backend.generate(
        [prompt],
        system=lt.SYSTEM_PROMPT,
        schema=rr.choice_schema(names),
        max_new_tokens=cfg.new_tokens,
        options={"num_ctx": int(cfg.ctx_tokens)},
    )
    choice, conf = rr.parse_choice(res.text, names)
    full = [inc.services[s] for s in order]
    top1 = full[0]
    pick = choice if choice is not None else top1
    ranking = [pick] + [n for n in full if n != pick]
    return lt.score_record(
        inc,
        model,
        condition,
        ranking,
        l4.NO_FAULT,
        prompt_tokens=res.n_prompt_tokens,
        new_tokens=res.n_new_tokens,
        centre_tokens=res.n_prompt_tokens,
        latency_s=time.perf_counter() - t0,
        valid=int(choice is not None),
        extra={
            "shortlist": [inc.services[s] for s in top],
            "shown": names,
            "choice": choice,
            "confidence": conf,
            "fusion_top1": top1,
            "overridden": int(pick != top1),
            "fusion_ranking": full,
            "evidence_tokens": reasoner.counter.count(body),
        },
    )


def metric_messages(inc: Incident) -> list[dict[str, Any]]:
    return [tm.metric_message(inc, s) for s in range(inc.n_services)]
