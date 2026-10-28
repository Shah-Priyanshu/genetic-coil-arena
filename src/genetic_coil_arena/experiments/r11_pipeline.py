"""R11: the desk. One shared translator over three domains, then the whole chain per request.

``Desk.handle`` is the front door: an LLM call routes the request to a domain (squad, plant,
incident, or none), the domain's policy pack extracts the params (``PolicyTranslator``:
parse first, validate after), and for the incident domain the catalogue resolves the
reference to exactly one incident or returns a typed error that becomes ``clarify`` or
``refused``. ``process_incident_item`` then takes a request through the rest of the chain,
twice for the task score: the natural-language spec and the gold spec each run through
``r7a_pipeline.execute`` (command centre, signed envelopes, ``TraceLog``, the G4c team), the
trace view is summarised by the reverse translator, the summary is checked against its facts
and routed and translated again for the round trip.

``KeywordDesk`` is the regex baseline behind the same interface.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from ..envs.logtriage.catalogue import CatalogueEntry, IncidentCatalogue, ResolveError
from ..protocol.goalspec import GoalSpec, SpecRegistry, SpecValidationError
from ..translator import (
    CLARIFY,
    ERROR,
    INVALID,
    OK,
    REFUSED,
    PolicyTranslator,
    ReverseTranslator,
    TranslationResult,
    exact_match,
    fact_fidelity,
    field_matches,
)
from . import r11_incident as inc
from .r7a_domains import Domain
from .r7a_pipeline import execute


@dataclass
class DeskResult:
    """What the desk made of one request.

    Attributes:
        text: The request
        domain: The routed domain (``squad``, ``plant``, ``incident`` or ``none``)
        status: ``ok`` (a GoalSpec, and for incidents a resolved incident), ``clarify`` or
            ``refused``
        reason: The question or the reason when not ``ok``
        translation: The translator's result (None when routed to ``none``)
        entry: The resolved catalogue entry (incident requests only)
        route_s: Time of the routing call
        route_raw: The router's raw reply
    """

    text: str
    domain: str
    status: str
    reason: str = ""
    translation: Optional[TranslationResult] = None
    entry: Optional[CatalogueEntry] = None
    route_s: float = 0.0
    route_raw: str = ""

    @property
    def spec(self) -> Optional[GoalSpec]:
        return self.translation.spec if self.translation else None

    @property
    def translate_s(self) -> float:
        """Routing plus translation time."""
        return self.route_s + (self.translation.latency_s if self.translation else 0.0)


class _Backend(Protocol):
    def generate(self, prompts: Any, **kw: Any) -> list[Any]: ...


class KeywordBackend:
    """A model-free stand-in for tests and smoke runs: the keyword router and parser behind
    the structured-output interface (plain-text calls, i.e. summaries, get a fixed line).

    Args:
        parser: ``text -> params or None`` (default: the v1 regex parser)
    """

    def __init__(self, parser: Optional[Any] = None):
        self.parser = parser or inc.keyword_incident

    def generate(self, prompts: Any, **kw: Any) -> list[Any]:
        from types import SimpleNamespace

        schema = kw.get("schema") or {}
        out = []
        for text in prompts:
            if "domain" in schema.get("properties", {}):
                body: Any = {"domain": inc.keyword_route(text)}
            elif schema:
                params = self.parser(text)
                body = (
                    {"refusal": "keyword parser found nothing", "clarify": ""}
                    if params is None
                    else {"refusal": "", "clarify": "", "params": params}
                )
            else:
                body = None
            out.append(
                SimpleNamespace(
                    text="Keyword stub summary." if body is None else json.dumps(body),
                    latency_s=0.0,
                    n_prompt_tokens=0,
                )
            )
        return out


def _final(
    text: str,
    domain: str,
    tr: TranslationResult,
    catalogue: Optional[IncidentCatalogue],
    route_s: float,
    raw: str,
) -> DeskResult:
    """Map the translator's result (and, for incidents, the catalogue's) to the desk's."""
    if tr.status == OK:
        entry = None
        if domain == "incident" and catalogue is not None and tr.spec is not None:
            try:
                entry = catalogue.resolve(tr.spec.params)
            except ResolveError as exc:
                status = CLARIFY if exc.kind == "clarify" else REFUSED
                return DeskResult(
                    text, domain, status, exc.message, tr, None, route_s, raw
                )
        return DeskResult(text, domain, OK, "", tr, entry, route_s, raw)
    status = CLARIFY if tr.status == CLARIFY else REFUSED
    return DeskResult(text, domain, status, tr.reason, tr, None, route_s, raw)


class Desk:
    """The shared translator in front of the three domains.

    Args:
        backend: Structured-output backend (``OllamaBackend`` or a stub)
        registry: Registry with all three specs (``build_registry_r11``)
        domains: ``name -> Domain`` with policy packs (squad, plant, incident)
        catalogue: Incident catalogue for the resolution step
        keep_alive: Passed to the backend on every call when not None
        route_system: The router prompt (default: the R11 one)
    """

    def __init__(
        self,
        backend: _Backend,
        registry: SpecRegistry,
        domains: dict[str, Domain],
        catalogue: IncidentCatalogue,
        keep_alive: Any = None,
        route_system: Optional[str] = None,
    ):
        self.route_system = route_system or inc.ROUTE_SYSTEM
        self.backend = backend
        self.registry = registry
        self.domains = domains
        self.catalogue = catalogue
        self.keep_alive = keep_alive
        self.translator = PolicyTranslator(backend, registry, keep_alive=keep_alive)

    def route(self, text: str) -> tuple[str, float, str]:
        """``(domain, seconds, raw reply)``; an unreadable reply routes to ``none``."""
        kwargs: dict[str, Any] = {
            "system": self.route_system,
            "examples": [(t, json.dumps({"domain": d})) for t, d in inc.ROUTE_EXAMPLES],
            "schema": inc.route_schema(),
        }
        if self.keep_alive is not None:
            kwargs["keep_alive"] = self.keep_alive
        try:
            (res,) = self.backend.generate([text], **kwargs)
            raw = res.text
            body = json.loads(raw)
            domain = body.get("domain") if isinstance(body, dict) else None
            latency = float(getattr(res, "latency_s", 0.0))
        except Exception as exc:  # noqa: BLE001 - a backend failure routes to none
            return "none", 0.0, f"{type(exc).__name__}: {exc}"
        return (domain if domain in inc.DOMAINS else "none"), latency, raw

    def handle(self, text: str, report: bool = False) -> DeskResult:
        """Route, translate and (for incidents) resolve one request or status report."""
        domain, route_s, raw = self.route(text)
        if domain == "none":
            return DeskResult(
                text,
                "none",
                REFUSED,
                "no desk handles this request",
                None,
                None,
                route_s,
                raw,
            )
        dom = self.domains[domain]
        pack = dom.report_pack if report else dom.pack
        tr = self.translator.translate(text, pack)
        return _final(text, domain, tr, self.catalogue, route_s, raw)


class KeywordDesk:
    """The regex baseline: keyword router, keyword parsers, the same validation gate."""

    def __init__(
        self,
        registry: SpecRegistry,
        domains: dict[str, Domain],
        catalogue: IncidentCatalogue,
    ):
        self.registry = registry
        self.domains = domains
        self.catalogue = catalogue

    def handle(self, text: str, report: bool = False) -> DeskResult:
        domain = inc.keyword_route(text)
        if domain == "none":
            return DeskResult(text, "none", REFUSED, "no keywords matched")
        dom = self.domains[domain]
        params = dom.keyword(text)
        if params is None:
            tr = TranslationResult(
                text, REFUSED, reason="no keywords matched", attempts=1
            )
        else:
            try:
                spec = self.registry.make(dom.spec_name, params)
                tr = TranslationResult(text, OK, spec=spec, attempts=1)
            except SpecValidationError as exc:
                tr = TranslationResult(
                    text, INVALID, reason=str(exc), errors=tuple(exc.errors), attempts=1
                )
        return _final(text, domain, tr, self.catalogue, 0.0, "")


# ---------------------------------------------------------------------- one request


@dataclass
class ItemRun:
    """A processed request: the JSON-safe record and the (non-JSON) trace view."""

    record: dict[str, Any]
    view: Any = field(default=None, repr=False)


def _exec_timed(
    domain: Domain, registry: SpecRegistry, spec: GoalSpec, text: str
) -> tuple[Any, float]:
    """``(Execution, seconds the team took)``; a cached run reports its original time."""
    runs = getattr(domain.executor, "runs", None)
    start = time.perf_counter()
    run = execute(domain, registry, spec, text)
    elapsed = time.perf_counter() - start
    if run.result is None:
        return run, elapsed
    cached = runs is not None and domain.executor.runs == runs
    return run, float(run.result.get("wall_s", elapsed)) if cached else elapsed


def process_incident_item(
    desk: Any,
    item: Any,
    summariser: Optional[ReverseTranslator] = None,
    execute_tasks: bool = True,
) -> ItemRun:
    """One incident request through the whole chain (see the module docstring).

    ``desk`` is a :class:`Desk` or :class:`KeywordDesk`; with a ``summariser`` an executed
    natural-language run is summarised, fact-checked and round-tripped.
    """
    registry: SpecRegistry = desk.registry
    domain = desk.domains["incident"]
    schema = registry.get(inc.INVESTIGATE).params_schema
    res: DeskResult = desk.handle(item.text)
    pred = res.translation.params if res.translation else None
    rec: dict[str, Any] = {
        "id": item.id,
        "kind": item.kind,
        "category": item.category,
        "style": item.style,
        "text": item.text,
        "gold": item.gold,
        "routed": res.domain,
        "route_ok": res.domain == "incident",
        "status": res.status,
        "reason": res.reason,
        "decision_ok": (res.status == OK) == (item.kind == "spec")
        and (res.status == CLARIFY) == (item.kind == "clarify"),
        "pred": pred,
        "translate_s": res.translate_s,
        "route_s": res.route_s,
        "attempts": res.translation.attempts if res.translation else 0,
        "n_prompt_tokens": res.translation.n_prompt_tokens if res.translation else 0,
        "ticket": res.entry.ticket if res.entry else None,
    }
    view = None
    if item.gold is not None:
        rec["exact"] = res.status == OK and exact_match(res.spec.params, item.gold, schema)  # type: ignore[union-attr]
        rec["fields"] = field_matches(
            pred if res.status == OK else None, item.gold, schema
        )
        if execute_tasks:
            nl_score, nl_top1 = 0.0, 0.0
            team_s = summary_s = 0.0
            if res.status == OK and res.spec is not None:
                run, team_s = _exec_timed(domain, registry, res.spec, item.text)
                rec["nl_state"] = run.state
                if run.result is not None:
                    nl_score = float(run.result["score"])
                    nl_top1 = float(run.result["top1"])
                    view = run.view
                    rec["n_envelopes"] = run.result.get("n_envelopes")
            gold_run = execute(
                domain, registry, registry.make(inc.INVESTIGATE, item.gold), item.text
            )
            rec["gold_state"] = gold_run.state
            rec["gold_ticket"] = (gold_run.result or {}).get("public", {}).get("ticket")
            rec.update(
                nl_score=nl_score,
                nl_top1=nl_top1,
                gold_score=float((gold_run.result or {}).get("score", 0.0)),
                gold_top1=float((gold_run.result or {}).get("top1", 0.0)),
                team_s=team_s,
            )
            if view is not None and summariser is not None:
                rec.update(_reverse(summariser, desk, domain, view, schema))
                summary_s = rec["summary_s"]
            rec["total_s"] = rec["translate_s"] + team_s + summary_s
    return ItemRun(rec, view)


def _reverse(
    summariser: ReverseTranslator,
    desk: Any,
    domain: Domain,
    view: Any,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Summary, fact check and round trip of one executed trace."""
    summ = summariser.summarise(view, domain.summary_pack)
    text = summ.text.strip()
    out: dict[str, Any] = {
        "summary": text,
        "summary_s": summ.latency_s,
        "summary_error": summ.error,
    }
    if not text:
        return {
            **out,
            "facts": {},
            "fidelity": 0.0,
            "all_facts": False,
            "round_trip": False,
        }
    fid = fact_fidelity(text, domain.facts(view))
    recon = desk.handle(text, report=True)
    match = bool(
        recon.status == OK
        and recon.domain == "incident"
        and recon.spec is not None
        and recon.spec.spec == view.goal.spec
        and exact_match(recon.spec.params, view.goal.params, schema)
    )
    return {
        **out,
        "facts": fid["facts"],
        "fidelity": fid["fidelity"],
        "all_facts": bool(fid["all"]),
        "recon_domain": recon.domain,
        "recon_status": recon.status,
        "recon_params": recon.translation.params if recon.translation else None,
        "recon_s": recon.translate_s,
        "round_trip": match,
    }


def route_only(desk: Any, item: Any) -> dict[str, Any]:
    """Routing record of an R7a item (squad or plant): did the desk pick its domain?"""
    domain, route_s, raw = (
        desk.route(item.text)
        if hasattr(desk, "route")
        else (inc.keyword_route(item.text), 0.0, "")
    )
    return {
        "id": item.id,
        "kind": item.expected,
        "domain": item.domain,
        "routed": domain,
        "route_ok": domain == item.domain,
        "route_s": route_s,
        "text": item.text,
    }


__all__ = [
    "Desk",
    "DeskResult",
    "ERROR",
    "ItemRun",
    "KeywordDesk",
    "process_incident_item",
    "route_only",
]
