"""R10 step 1: log triage with forced specialisation (goal G4).

Each incident has S services; their templated logs together exceed the centre's context
``ctx_tokens`` by orders of magnitude. Conditions, all with a centre prompt of at most
``ctx_tokens`` tokens (instructions included):

* ``heur_baro``, ``heur_surprise``: non-LLM rankings (:mod:`envs.logtriage.heuristics`).
* ``single_head`` / ``single_tail``: one model reads the first / last templated lines.
* ``single_selected``: one model reads the strong heuristic selection
  (:func:`envs.logtriage.views.selected_view`).
* ``oracle_selected``: the same selection at a long context (``oracle_ctx_tokens``).
* ``team_typed``: one specialist per service reads its own selected stream
  (``spec_ctx_tokens``) and sends a schema-validated JSON message of at most
  ``message_tokens``; the centre reads the messages only.
* ``team_free``: the same with free-text messages at the same token budget.
* ``team_typed_heur``: typed messages computed by code instead of a model.
* ablations of the two team conditions: the root-cause service's message (or a random
  other one) is dropped or replaced by another incident's message from the same service.

Models are reached through a :class:`Reasoner`: :class:`HFReasoner` (Qwen3-1.7B on
``HFPeftBackend``; answers are ranked by next-token log-probabilities over option letters,
so scoring never depends on the model formatting a reply) and :class:`OllamaReasoner`
(structured JSON output). Torch and transformers are imported lazily by the backend.
"""

import json
import random
import string
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..envs.logtriage import heuristics as hz
from ..envs.logtriage import messages as msgs
from ..envs.logtriage import views
from ..envs.logtriage.incident import Incident
from ..envs.logtriage.rcaeval import FAULT_TYPES
from ..envs.logtriage.views import ApproxCounter, TokenCounter

SERVICE_LETTERS = string.ascii_uppercase
FAULT_LETTERS = string.ascii_lowercase[: len(FAULT_TYPES)]
FAULT_GLOSS = {
    "cpu": "CPU hog on the service",
    "mem": "memory exhaustion or leak",
    "disk": "disk I/O stress or a full disk",
    "delay": "network delay added to the service",
    "loss": "packet loss on the service's network",
    "socket": "socket or connection exhaustion",
}

SINGLE_CONDITIONS = ("single_head", "single_tail", "single_selected", "oracle_selected")
TEAM_CONDITIONS = ("team_typed", "team_free", "team_typed_heur")
HEURISTIC_CONDITIONS = ("heur_baro_logs", "heur_surprise_logs", "heur_baro_metrics")
ALL_CONDITIONS = (*HEURISTIC_CONDITIONS, *SINGLE_CONDITIONS, *TEAM_CONDITIONS)
ABLATIONS = ("drop_root", "drop_other", "shuffle_root", "shuffle_other")

SYSTEM_PROMPT = (
    "You are an expert site reliability engineer doing root-cause analysis of a "
    "microservice incident. Exactly one service had a fault injected at the alert time."
)


@dataclass
class Settings:
    """Budgets of one run (token counts use the run's :class:`TokenCounter`)."""

    ctx_tokens: int = 4096
    spec_ctx_tokens: int = 3072
    message_tokens: int = 128
    oracle_ctx_tokens: int = 32768
    spec_new_tokens: int = 200
    spec_batch: int = 4
    decide_batch: int = 4
    margin: int = 48  # kept free below the context for the chat wrapper
    use_metrics: bool = True  # sources are logs + service metrics; False = logs only


# --------------------------------------------------------------------------- prompts


def _options(services: Sequence[str]) -> str:
    return "\n".join(f"{SERVICE_LETTERS[i]}. {s}" for i, s in enumerate(services))


def _faults() -> str:
    return "\n".join(
        f"{FAULT_LETTERS[i]}. {k}: {FAULT_GLOSS[k]}" for i, k in enumerate(FAULT_TYPES)
    )


def decision_prompt(body: str, services: Sequence[str], style: str = "letters") -> str:
    """The user message for a decision: evidence, options, and how to answer.

    ``style`` is ``letters`` (two lines: service letter, fault letter; scored by
    log-probabilities) or ``json`` (a JSON reply with a ranked list, for Ollama).
    """
    if len(services) > len(SERVICE_LETTERS):
        raise ValueError(f"at most {len(SERVICE_LETTERS)} services are supported")
    if style == "letters":
        reply = (
            "Reply with exactly two lines. Line 1: the letter of the root-cause "
            "service. Line 2: the letter of the fault type."
        )
    else:
        reply = (
            'Reply as JSON: {"ranking": [letters of your three most likely '
            'root-cause services, best first], "fault": letter of the fault type}.'
        )
    return (
        f"{body}\n\nWhich service is the root cause?\n{_options(services)}\n\n"
        f"Which fault type was injected?\n{_faults()}\n\n{reply}"
    )


def typed_specialist_prompt(service: str, view: str) -> str:
    return (
        f"You are the monitor of service '{service}'. Below is a summary of its own "
        "logs and metrics around an alert at 0s (counts are before and after the alert). You cannot "
        "see any other service. Report to the incident commander as one JSON object "
        "with exactly these keys: service; anomaly (true if the logs changed "
        "meaningfully after the alert); first_anomaly_s (seconds after the alert of the "
        "first new or changed log pattern, or null); line_rate_ratio (log lines after "
        "divided by before, from the summary, or null); top_errors (at most 3 of the "
        'most unusual log templates or metrics, each {"template": short text, "count": '
        "integer}); suspected_fault (one of cpu, mem, disk, delay, loss, socket, none, "
        f"unknown). Reply with the JSON only.\n\n{view}"
    )


def free_specialist_prompt(service: str, view: str, words: int) -> str:
    return (
        f"You are the monitor of service '{service}'. Below is a summary of its own "
        "logs and metrics around an alert at 0s (counts are before and after the alert). You cannot "
        "see any other service. Write a plain-text report of at most "
        f"{words} words for the incident commander: is this service behaving "
        "abnormally, what changed after the alert and when, and which fault type you "
        "suspect (cpu, mem, disk, delay, loss, socket). Reply with the report only."
        f"\n\n{view}"
    )


# ------------------------------------------------------------------------ reasoners


@dataclass
class Decision:
    ranking: list[str]
    fault: str
    n_prompt: int = 0
    n_new: int = 0
    valid: bool = True


@dataclass
class Written:
    text: str
    n_prompt: int = 0
    n_new: int = 0


class Reasoner:
    """What the conditions need from a model (see the subclasses)."""

    name = "reasoner"
    style = "letters"
    counter: TokenCounter

    def decide(
        self, bodies: Sequence[str], services: Sequence[Sequence[str]], ctx: int
    ) -> list[Decision]:
        raise NotImplementedError

    def write(
        self, prompts: Sequence[str], max_new_tokens: int, typed: bool
    ) -> list[Written]:
        raise NotImplementedError

    def wrapper_tokens(self) -> int:
        """Tokens the chat wrapper and system prompt add to a user message."""
        return self.counter.count(SYSTEM_PROMPT) + 24


def _rank_from_scores(scores: Sequence[float], names: Sequence[str]) -> list[str]:
    order = sorted(range(len(names)), key=lambda i: (-scores[i], i))
    return [names[i] for i in order]


class HFReasoner(Reasoner):
    """Qwen3 (or any chat model) on :class:`HFPeftBackend`, scored by log-probabilities."""

    def __init__(self, backend: Any, name: str = "qwen3-1.7b"):
        self.backend = backend
        self.name = name
        self.style = "letters"
        from ..envs.logtriage.views import HFCounter

        self.counter = HFCounter(backend.tokenizer, name=name)
        tok = backend.tokenizer
        self._stops = [
            i
            for i in (
                tok.convert_tokens_to_ids("<|im_end|>"),
                tok.eos_token_id,
            )
            if isinstance(i, int) and i >= 0
        ]
        self._service_ids = [self._single_token(c) for c in SERVICE_LETTERS]
        self._fault_ids = [self._single_token(c) for c in FAULT_LETTERS]

    def _single_token(self, text: str) -> int:
        ids = self.backend.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{text!r} is not a single token: {ids}")
        return int(ids[0])

    def chat(self, user: str) -> str:
        return self.backend.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def decide(
        self, bodies: Sequence[str], services: Sequence[Sequence[str]], ctx: int
    ) -> list[Decision]:
        out: list[Decision] = []
        users = [decision_prompt(b, s, "letters") for b, s in zip(bodies, services)]
        chats = [self.chat(u) for u in users]
        for lo in range(0, len(chats), self._batch(ctx)):
            hi = lo + self._batch(ctx)
            part = chats[lo:hi]
            names = services[lo:hi]
            k = max(len(n) for n in names)
            lp1 = self.backend.next_token_logprobs(
                part, self._service_ids[:k], chunk_size=2048
            )
            tops = []
            rankings = []
            for lp, nm in zip(lp1, names):
                ranked = _rank_from_scores(lp[: len(nm)], nm)
                rankings.append(ranked)
                tops.append(SERVICE_LETTERS[list(nm).index(ranked[0])])
            second = [p + f"{t}\n" for p, t in zip(part, tops)]
            lp2 = self.backend.next_token_logprobs(
                second, self._fault_ids, chunk_size=2048
            )
            for j, (rk, f) in enumerate(zip(rankings, lp2)):
                fault = FAULT_TYPES[int(np.argmax(f))]
                n_prompt = len(
                    self.backend.tokenizer.encode(part[j], add_special_tokens=False)
                )
                out.append(Decision(rk, fault, n_prompt=n_prompt, n_new=0))
        return out

    def _batch(self, ctx: int) -> int:
        return 1 if ctx > 8192 else 4

    def write(
        self, prompts: Sequence[str], max_new_tokens: int, typed: bool
    ) -> list[Written]:
        chats = [self.chat(p) for p in prompts]
        results = self.backend.generate(
            chats, max_new_tokens=max_new_tokens, stop_ids=self._stops or None
        )
        return [
            Written(r.text, n_prompt=r.n_prompt_tokens, n_new=r.n_new_tokens)
            for r in results
        ]


class OllamaReasoner(Reasoner):
    """A model behind Ollama, answering with schema-constrained JSON."""

    def __init__(self, backend: Any, counter: Optional[TokenCounter] = None):
        self.backend = backend
        self.name = backend.base_id
        self.style = "json"
        self.counter = counter or ApproxCounter()

    def decide(
        self, bodies: Sequence[str], services: Sequence[Sequence[str]], ctx: int
    ) -> list[Decision]:
        out: list[Decision] = []
        for body, names in zip(bodies, services):
            letters = list(SERVICE_LETTERS[: len(names)])
            schema = {
                "type": "object",
                "required": ["ranking", "fault"],
                "properties": {
                    "ranking": {
                        "type": "array",
                        "items": {"enum": letters},
                        "minItems": 1,
                        "maxItems": 3,
                    },
                    "fault": {"enum": list(FAULT_LETTERS)},
                },
            }
            (res,) = self.backend.generate(
                [decision_prompt(body, names, "json")],
                system=SYSTEM_PROMPT,
                schema=schema,
                max_new_tokens=80,
                options={"num_ctx": int(ctx + 1024)},
            )
            parsed = msgs.parse_json_object(res.text) or {}
            picked: list[str] = []
            for letter in parsed.get("ranking", []) if isinstance(parsed, dict) else []:
                if letter in letters and names[letters.index(letter)] not in picked:
                    picked.append(names[letters.index(letter)])
            valid = bool(picked)
            ranking = picked + [n for n in names if n not in picked]
            fault_letter = parsed.get("fault") if isinstance(parsed, dict) else None
            fault = (
                FAULT_TYPES[FAULT_LETTERS.index(fault_letter)]
                if fault_letter in FAULT_LETTERS
                else "unknown"
            )
            out.append(
                Decision(ranking, fault, res.n_prompt_tokens, res.n_new_tokens, valid)
            )
        return out

    def write(
        self, prompts: Sequence[str], max_new_tokens: int, typed: bool
    ) -> list[Written]:
        out = []
        for prompt in prompts:
            schema = msgs.MESSAGE_SCHEMA if typed else None
            (res,) = self.backend.generate(
                [prompt],
                system=SYSTEM_PROMPT,
                schema=schema,
                max_new_tokens=max_new_tokens,
                options={"num_ctx": 4096},
            )
            out.append(Written(res.text, res.n_prompt_tokens, res.n_new_tokens))
        return out


# ---------------------------------------------------------------------- conditions


@dataclass
class Record:
    """One (model, condition, incident) outcome."""

    model: str
    condition: str
    incident: str
    system: str
    rep: int
    n_services: int
    root: str
    fault: str
    ranking: list[str]
    pred_fault: str
    top1: int
    top3: int
    fault_ok: int
    prompt_tokens: int = 0
    new_tokens: int = 0
    centre_tokens: int = 0
    latency_s: float = 0.0
    valid: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.model}|{self.condition}|{self.incident}"

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["tokens"] = self.prompt_tokens + self.new_tokens
        return d


def score_record(
    inc: Incident,
    model: str,
    condition: str,
    ranking: Sequence[str],
    pred_fault: str,
    **kw: Any,
) -> Record:
    ranking = list(ranking)
    return Record(
        model=model,
        condition=condition,
        incident=inc.id,
        system=inc.system,
        rep=inc.rep,
        n_services=inc.n_services,
        root=inc.root_service,
        fault=inc.fault_type,
        ranking=ranking,
        pred_fault=pred_fault,
        top1=int(bool(ranking) and ranking[0] == inc.root_service),
        top3=int(inc.root_service in ranking[:3]),
        fault_ok=int(pred_fault == inc.fault_type),
        **kw,
    )


def body_budget(
    reasoner: Reasoner, services: Sequence[str], ctx: int, margin: int
) -> int:
    """Tokens left for evidence once instructions, options and the wrapper are counted."""
    fixed = reasoner.counter.count(decision_prompt("", services, reasoner.style))
    return ctx - fixed - reasoner.wrapper_tokens() - margin


def run_heuristic(inc: Incident, condition: str) -> Record:
    scores = {
        "heur_baro_logs": hz.baro_logs,
        "heur_surprise_logs": hz.surprise_sum,
        "heur_baro_metrics": hz.baro_metrics,
    }[condition](inc)
    order = hz.ranking(scores)
    ranking = [inc.services[i] for i in order]
    st = inc.stats()
    top = order[0]
    mine = np.flatnonzero((st.svc == top) & (st.post > st.pre))
    sur = st.surprise()[mine]
    best = mine[np.argsort(-sur, kind="stable")][:3]
    fault = hz.guess_fault(
        [inc.templates[int(st.tpl[j])] for j in best], default="delay"
    )
    if condition == "heur_baro_metrics":
        fault = hz.fault_from_metrics(inc, inc.services[top]) or fault
    return score_record(inc, "none", condition, ranking, fault)


def _single_body(
    inc: Incident, condition: str, reasoner: Reasoner, cfg: Settings
) -> tuple[str, int]:
    ctx = cfg.oracle_ctx_tokens if condition == "oracle_selected" else cfg.ctx_tokens
    budget = body_budget(reasoner, inc.services, ctx, cfg.margin)
    counter = reasoner.counter
    if condition == "single_head":
        text = views.alert_header(inc) + "\nLog lines (time, service, template):\n"
        text += views.naive_view(inc, budget - counter.count(text), counter)
    elif condition == "single_tail":
        text = views.alert_header(inc) + "\nLog lines (time, service, template):\n"
        text += views.naive_view(inc, budget - counter.count(text), counter, tail=True)
    else:
        text = views.selected_view(inc, budget, counter, use_metrics=cfg.use_metrics)
    return text, ctx


def run_single(
    inc: Incident, condition: str, reasoner: Reasoner, cfg: Settings
) -> Record:
    body, ctx = _single_body(inc, condition, reasoner, cfg)
    start = time.perf_counter()
    (dec,) = reasoner.decide([body], [inc.services], ctx)
    elapsed = time.perf_counter() - start
    return score_record(
        inc,
        reasoner.name,
        condition,
        dec.ranking,
        dec.fault,
        prompt_tokens=dec.n_prompt,
        new_tokens=dec.n_new,
        centre_tokens=dec.n_prompt,
        latency_s=elapsed,
        valid=int(dec.valid),
        extra={"ctx": ctx},
    )


def specialist_messages(
    inc: Incident, kind: str, reasoner: Optional[Reasoner], cfg: Settings
) -> tuple[list[Any], dict[str, Any]]:
    """One message per service: dicts (typed) or strings (free text), plus accounting."""
    counter = reasoner.counter if reasoner else ApproxCounter()
    n = inc.n_services
    info: dict[str, Any] = {"prompt": 0, "new": 0, "valid": 0, "latency": 0.0}
    if kind == "typed_heur":
        out = [
            msgs.fit_message(
                msgs.heuristic_message(inc, s, cfg.use_metrics),
                counter,
                cfg.message_tokens,
            )
            for s in range(n)
        ]
        info["valid"] = n
        return out, info
    assert reasoner is not None
    typed = kind == "typed"
    overhead = (
        counter.count(typed_specialist_prompt("x" * 12, "")) + reasoner.wrapper_tokens()
    )
    view_budget = cfg.spec_ctx_tokens - overhead - cfg.margin
    prompts = []
    for s in range(n):
        view = views.source_view(inc, s, view_budget, counter, cfg.use_metrics)
        name = inc.services[s]
        prompts.append(
            typed_specialist_prompt(name, view)
            if typed
            else free_specialist_prompt(
                name, view, max(20, int(cfg.message_tokens / 1.6))
            )
        )
    written: list[Written] = []
    start = time.perf_counter()
    for lo in range(0, n, cfg.spec_batch):
        written += reasoner.write(
            prompts[lo : lo + cfg.spec_batch], cfg.spec_new_tokens, typed
        )
    info["latency"] = time.perf_counter() - start
    out = []
    for s, w in enumerate(written):
        info["prompt"] += w.n_prompt
        info["new"] += w.n_new
        name = inc.services[s]
        if typed:
            parsed = msgs.parse_json_object(w.text)
            if parsed is not None:
                parsed["service"] = name
            if parsed is not None and not msgs.message_errors(parsed):
                out.append(msgs.fit_message(parsed, counter, cfg.message_tokens))
                info["valid"] += 1
            else:
                out.append(msgs.empty_message(name))
        else:
            text = msgs.truncate_text(w.text, counter, cfg.message_tokens)
            out.append(text)
            info["valid"] += int(bool(text))
    return out, info


def centre_body(
    inc: Incident, messages: Sequence[Any], counter: TokenCounter, budget: int
) -> str:
    """The centre's evidence: one line per service, in the order the services are listed.

    If the lines exceed ``budget``, every message is cut to a smaller share until they fit.
    """
    head = (
        views.alert_header(inc).split(" Services with logs")[0]
        + " You cannot read the logs yourself; each service's monitor sent this report "
        "from that service's own logs:\n"
    )
    share = max(16, (budget - counter.count(head)) // max(len(messages), 1))
    while True:
        lines = []
        for name, m in zip(inc.services, messages):
            if isinstance(m, dict):
                if share < 128:
                    m = msgs.fit_message(m, counter, max(share - 8, 12))
                text = msgs.message_json(m)
            else:
                text = msgs.truncate_text(m, counter, max(share - 8, 8))
            lines.append(f"- {name}: {text}")
        body = head + "\n".join(lines)
        if counter.count(body) <= budget or share <= 16:
            return body
        share = int(share * 0.85)


def _decide_messages(
    inc: Incident,
    messages: Sequence[Any],
    reasoner: Reasoner,
    cfg: Settings,
) -> Decision:
    budget = body_budget(reasoner, inc.services, cfg.ctx_tokens, cfg.margin)
    body = centre_body(inc, messages, reasoner.counter, budget)
    (dec,) = reasoner.decide([body], [inc.services], cfg.ctx_tokens)
    return dec


def run_team(
    inc: Incident,
    kind: str,
    reasoner: Reasoner,
    cfg: Settings,
    pool: Optional[dict[str, list[Any]]] = None,
    ablate: bool = True,
    precomputed: Optional[tuple[list[Any], dict[str, Any]]] = None,
) -> list[Record]:
    """The team condition plus its message ablations (records for each).

    ``precomputed`` is the ``(messages, info)`` of :func:`specialist_messages` when the
    caller generated the messages in a first pass (the shuffle ablation draws from the
    messages of the other incidents, so it needs all of them first).
    """
    condition = {
        "typed": "team_typed",
        "free": "team_free",
        "typed_heur": "team_typed_heur",
    }[kind]
    messages, info = precomputed or specialist_messages(inc, kind, reasoner, cfg)
    start = time.perf_counter()
    dec = _decide_messages(inc, messages, reasoner, cfg)
    centre_time = time.perf_counter() - start
    valid = info["valid"] / inc.n_services
    base = score_record(
        inc,
        reasoner.name,
        condition,
        dec.ranking,
        dec.fault,
        prompt_tokens=info["prompt"] + dec.n_prompt,
        new_tokens=info["new"] + dec.n_new,
        centre_tokens=dec.n_prompt,
        latency_s=info["latency"] + centre_time,
        valid=int(dec.valid),
        extra={
            "valid_messages": round(valid, 3),
            "message_tokens": [
                reasoner.counter.count(
                    msgs.message_json(m) if isinstance(m, dict) else str(m)
                )
                for m in messages
            ],
        },
    )
    records = [base]
    if not ablate or kind == "typed_heur":
        return records
    rng = random.Random(zlib.crc32(inc.id.encode()))
    root = inc.services.index(inc.root_service)
    others = [i for i in range(inc.n_services) if i != root]
    picks = {"root": root, "other": rng.choice(others)}
    for label, idx in picks.items():
        for mode in ("drop", "shuffle"):
            altered = list(messages)
            if mode == "drop":
                altered[idx] = (
                    msgs.empty_message(inc.services[idx])
                    if kind != "free"
                    else "(no report received)"
                )
            else:
                alt = (pool or {}).get(f"{kind}|{inc.services[idx]}", [])
                alt = [a for a in alt if a is not None and a != messages[idx]]
                altered[idx] = rng.choice(alt) if alt else messages[idx]
                if isinstance(altered[idx], dict):
                    altered[idx] = dict(altered[idx], service=inc.services[idx])
            d = _decide_messages(inc, altered, reasoner, cfg)
            records.append(
                score_record(
                    inc,
                    reasoner.name,
                    f"{condition}:{mode}_{label}",
                    d.ranking,
                    d.fault,
                    prompt_tokens=d.n_prompt,
                    centre_tokens=d.n_prompt,
                    valid=int(d.valid),
                    extra={"ablated_service": inc.services[idx]},
                )
            )
    return records


def overflow_stats(inc: Incident, counter: TokenCounter, ctx: int) -> dict[str, Any]:
    """How far the incident's templated logs are over one context budget."""
    full = views.full_text(inc)
    full_tokens = counter.count(full)
    uniq = sorted(
        {
            f"{inc.services[s]} {inc.templates[t]}"
            for s, t in zip(inc.svc.tolist(), inc.tpl.tolist())
        }
    )
    uniq_tokens = counter.count("\n".join(uniq))
    per_service = np.bincount(inc.svc, minlength=inc.n_services)
    return {
        "kind": "overflow",
        "incident": inc.id,
        "system": inc.system,
        "n_lines": inc.n_lines,
        "n_services": inc.n_services,
        "n_templates": len(inc.templates),
        "full_tokens": full_tokens,
        "unique_template_tokens": uniq_tokens,
        "ratio_full": full_tokens / ctx,
        "ratio_unique": uniq_tokens / ctx,
        "largest_service_lines": int(per_service.max()),
        "counter": counter.name,
        "ctx": ctx,
    }


def append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue  # a torn last line from a killed run
    return out
