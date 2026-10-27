"""R11: the incident domain, a third target of the one shared translator.

``rca.investigate`` asks the desk to investigate one past RCAEval RE2 incident of the Sock Shop
or the Online Boutique and report the suspected root-cause services and the fault type. Like the
R7a domains this module contributes data and glue, never translator code:

* the ``SpecDefinition`` (at most six params: ``system``, ``ticket``, ``at``, ``top_k``,
  ``want_fault_type``, ``exclude_services``) and a registry that holds it beside the R7a specs;
* the policy pack (system prompt, few-shots, defaults, the question for a missing system),
  the summary pack and the report pack of the round trip;
* the facts a summary must state (system, ticket, suspects in order, fault type and its
  probability, onset, exclusions, and the requested time) for the G6 fidelity check;
* the keyword/regex baseline, and the domain router (an LLM call that picks the domain, and
  a keyword router as the baseline);
* ``incident_domain``: the :class:`~.r7a_domains.Domain` that plugs the incident executor
  into ``r7a_pipeline.execute``.

The desk clock is fixed: it is Thursday 2024-02-22, 09:00 UTC (:data:`DESK_NOW`), so that
"yesterday" and "two days ago" name the 20-21 February incidents; every time is UTC.
"""

import json
import re
from collections.abc import Mapping
from typing import Any, Optional

from ..envs.logtriage.catalogue import (
    AT_PATTERN,
    SYSTEM_NAMES,
    TICKET_PATTERN,
    CatalogueEntry,
)
from ..protocol.goalspec import SpecDefinition, SpecRegistry
from ..translator import Example, Fact, SummaryPack, TraceView, TranslationPack
from .r7a_domains import Domain, _num, build_registry

INVESTIGATE = "rca.investigate"
SYSTEMS = tuple(SYSTEM_NAMES.values())
TOP_K_RANGE = (1, 5)
DESK_NOW = "Thursday 2024-02-22, 09:00 UTC"
INCIDENT_SPECIALIST = "incident-desk"
FAULTS = ("cpu", "mem", "disk", "delay", "loss", "socket")
SOCK_SHOP_SERVICES = (
    "carts", "carts-db", "catalogue", "catalogue-db", "front-end", "orders",
    "orders-db", "payment", "queue-master", "rabbitmq", "rabbitmq-exporter",
    "session-db", "shipping", "user", "user-db",
)  # fmt: skip
ONLINE_BOUTIQUE_SERVICES = (
    "adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice",
    "frontend", "frontend-external", "paymentservice", "productcatalogservice",
    "recommendationservice", "redis", "shippingservice",
)  # fmt: skip
INCIDENT_DEFAULTS = {"top_k": 3, "want_fault_type": True, "exclude_services": []}
INCIDENT_QUESTIONS = {
    "system": "Which system is this about, sock-shop or online-boutique?"
}
DOMAINS = ("squad", "plant", "incident")


# ---------------------------------------------------------------------- spec


def investigate_definition() -> SpecDefinition:
    """``rca.investigate`` v1: investigate one catalogued incident."""
    params = {
        "type": "object",
        "required": ["system", "top_k", "want_fault_type", "exclude_services"],
        "properties": {
            "system": {"type": "string", "enum": list(SYSTEMS)},
            "ticket": {"type": "string", "pattern": TICKET_PATTERN},
            "at": {"type": "string", "pattern": AT_PATTERN},
            "top_k": {
                "type": "integer",
                "minimum": TOP_K_RANGE[0],
                "maximum": TOP_K_RANGE[1],
            },
            "want_fault_type": {"type": "boolean"},
            "exclude_services": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
                "maxItems": 6,
            },
        },
        "additionalProperties": False,
    }
    number = {"type": "number"}
    result = {
        "type": "object",
        "required": ["score", "top1", "per_seed", "public"],
        "properties": {
            "score": number,
            "top1": number,
            "per_seed": {"type": "array", "items": number},
            "wall_s": number,
            "n_envelopes": number,
            "public": {
                "type": "object",
                "required": [
                    "ticket", "system", "suspects", "onset_utc", "excluded_services",
                ],  # fmt: skip
                "properties": {
                    "ticket": {"type": "string"},
                    "system": {"type": "string", "enum": list(SYSTEMS)},
                    "suspects": {"type": "array", "items": {"type": "string"}},
                    "onset_utc": {"type": ["string", "null"]},
                    "excluded_services": {"type": "array", "items": {"type": "string"}},
                    "fault_type": {"type": "string", "enum": list(FAULTS)},
                    "fault_probability": number,
                },
            },
        },
    }
    return SpecDefinition(
        INVESTIGATE,
        1,
        params,
        "Investigate one past incident of the Sock Shop or Online Boutique: name it by "
        "ticket or time; get the top suspect services and the fault type",
        result,
    )


def build_registry_r11(backend: str = "auto") -> SpecRegistry:
    """The R7a registry (squad.mission, plant.diagnose) plus ``rca.investigate``."""
    registry = build_registry(backend)
    registry.register(investigate_definition())
    return registry


def incident_outcome(result: Mapping[str, Any]) -> dict[str, Any]:
    """What the reverse translator sees of a result: the public part, never the label score."""
    return {"state": "completed", **json.loads(json.dumps(result["public"]))}


# ---------------------------------------------------------------------- packs

INCIDENT_POLICY_SYSTEM = f"""You turn a request to the incident desk into JSON. The desk investigates one past incident of a web shop and reports the suspected root-cause services and the fault type.
Systems: "sock-shop" (the shop, the sock shop, the socks shop) and "online-boutique" (the boutique, the online boutique). Their services: sock-shop has {", ".join(SOCK_SHOP_SERVICES)}; online-boutique has {", ".join(ONLINE_BOUTIQUE_SERVICES)}.
The desk clock: it is {DESK_NOW}. All times are UTC; use 24-hour times.
Reply with {{"refusal": "", "clarify": "", "params": {{...}}}}. Fill exactly one of the three: params for a request you can read, refusal, or clarify.

Your job is to EXTRACT what the request says, in whatever style it is written (terse, polite, chatty, typos, numbers in words, several sentences). Never refuse or clarify just because a request is short, informal or indirect. If the request corrects itself, use the final version.

params:
- system: "sock-shop" or "online-boutique". Leave it out when the request names no system (a ticket number alone does not tell the system).
- ticket: the ticket as "INC-" and its four-digit number, e.g. "ticket 1042" or "inc1042" is "INC-1042". Leave it out when no ticket is named.
- at: when the incident happened. "YYYY-MM-DD" when only a day is given, "YYYY-MM-DDTHH:MM" when a time of day is given ("2:30 pm on 20 January" is "2024-01-20T14:30"). Work out "yesterday", "two days ago" or "last Friday" from the desk clock. A part of the day without a clock time ("in the afternoon") gives the day only. Leave it out when no time is given.
- top_k: integer, how many suspect services to list. Leave it out when not stated (default 3).
- want_fault_type: false only when the request says the fault type is not needed. Leave it out otherwise (default true).
- exclude_services: services to ignore, written exactly as in the lists above. Leave it out when none is named (default empty).

Extract values exactly as stated. Do NOT refuse because a ticket looks unknown, a number is large or a service is not in the lists: put it in params and the checker will reject it. Allowed values, for reference: top_k 1 to 5.

refusal (short reason) only when the request is not about investigating an incident of these two systems (out of domain, including requests to change, restart or delete something), or it contradicts itself (two different tickets, two systems, or a service both excluded and required).
clarify (one short question) only when the request is about an incident but names it too vaguely to look up ("recently", "last week", "a while ago", "that outage") and gives no ticket or time. Things the request does not mention take their defaults; a missing system is asked for by the checker, so just leave it out."""

INCIDENT_POLICY_EXAMPLES = (
    Example(
        "Could you walk me through ticket INC-1064 from the boutique?",
        {"system": "online-boutique", "ticket": "INC-1064"},
    ),
    Example(
        "The online boutique stalled at 9:30 pm on Friday 19 January, tell me what was at fault.",
        {"system": "online-boutique", "at": "2024-01-19T21:30"},
    ),
    Example(
        "Shop ticket INC-1102: I'd like the 4 likeliest services, and leave carts out of it.",
        {
            "system": "sock-shop",
            "ticket": "INC-1102",
            "top_k": 4,
            "exclude_services": ["carts"],
        },
    ),
    Example(
        "No interest in fault kinds, only suspects, for INC-1057 on the boutique.",
        {"system": "online-boutique", "ticket": "INC-1057", "want_fault_type": False},
    ),
    Example(
        "Look at the online boutique on 21 February.",
        {"system": "online-boutique", "at": "2024-02-21"},
    ),
    Example(
        "Pull INC-1066 from the queue and have a go.",
        {"ticket": "INC-1066"},
    ),
    Example(
        "Please pull up incident INC-3141 for the online boutique.",
        {"system": "online-boutique", "ticket": "INC-3141"},
    ),
    Example(
        "How about twelve suspects for the boutique's ticket INC-1059, please?",
        {"system": "online-boutique", "ticket": "INC-1059", "top_k": 12},
    ),
    Example(
        "Run the analysis of INC-1019 on the boutique, disregarding the loyalty-engine.",
        {
            "system": "online-boutique",
            "ticket": "INC-1019",
            "exclude_services": ["loyalty-engine"],
        },
    ),
    Example(
        "Please order a pizza for the on-call team.",
        None,
        "this is not a request to investigate an incident",
    ),
    Example(
        "Investigate the sock shop and the online boutique together in one go.",
        None,
        "the request names two systems at once",
    ),
    Example(
        "Can you check on that outage we had earlier on the shop?",
        None,
        "",
        "When did the outage happen? Give a ticket or a date and time.",
    ),
)

INCIDENT_SUMMARY_SYSTEM = """You write the status report of a finished incident investigation for an on-call engineer, from a JSON trace.
Write two to four plain sentences. State, using only the trace:
- the system, and how the incident was requested: "requested by ticket INC-nnnn" when the goal has a ticket, "requested by time <the goal's at value, copied exactly>" when it has an at value (then add "(ticket INC-nnnn)" from the outcome); both when the goal has both;
- the suspected services in order, exactly as in outcome.suspects, most likely first;
- when outcome has a fault type, "fault type <name> (probability <p>)"; when it has none, say that no fault type was requested;
- the onset as "onset <HH:MM> UTC" from outcome.onset_utc, or "onset unknown" when it is null;
- "excluded: <names>" for outcome.excluded_services, or "no services excluded" when empty."""

INCIDENT_SUMMARY_EXAMPLES = (
    (
        json.dumps(
            {
                "goal": {
                    "spec": INVESTIGATE,
                    "params": {
                        "exclude_services": ["user"],
                        "system": "sock-shop",
                        "ticket": "INC-1073",
                        "top_k": 2,
                        "want_fault_type": True,
                    },
                },
                "messages": [
                    {
                        "sender": "command-centre",
                        "recipients": [INCIDENT_SPECIALIST],
                        "payload_type": "json",
                        "codec": "goalspec",
                        "clock": 1,
                    },
                    {
                        "sender": INCIDENT_SPECIALIST,
                        "recipients": ["command-centre"],
                        "payload_type": "json",
                        "codec": "result",
                        "clock": 2,
                    },
                ],
                "outcome": {
                    "state": "completed",
                    "ticket": "INC-1073",
                    "system": "sock-shop",
                    "suspects": ["orders", "carts"],
                    "fault_type": "cpu",
                    "fault_probability": 0.71,
                    "onset_utc": "2024-01-20T06:12",
                    "excluded_services": ["user"],
                },
            },
            sort_keys=True,
        ),
        "Investigation of the sock-shop incident, requested by ticket INC-1073. The suspected services, most likely first, are orders, carts; fault type cpu (probability 0.71); onset 06:12 UTC. Excluded: user.",
    ),
    (
        json.dumps(
            {
                "goal": {
                    "spec": INVESTIGATE,
                    "params": {
                        "at": "2024-01-17T22:05",
                        "exclude_services": [],
                        "system": "online-boutique",
                        "top_k": 3,
                        "want_fault_type": False,
                    },
                },
                "messages": [
                    {
                        "sender": "command-centre",
                        "recipients": [INCIDENT_SPECIALIST],
                        "payload_type": "json",
                        "codec": "goalspec",
                        "clock": 1,
                    },
                    {
                        "sender": INCIDENT_SPECIALIST,
                        "recipients": ["command-centre"],
                        "payload_type": "json",
                        "codec": "result",
                        "clock": 2,
                    },
                ],
                "outcome": {
                    "state": "completed",
                    "ticket": "INC-1009",
                    "system": "online-boutique",
                    "suspects": ["redis", "cartservice", "frontend"],
                    "onset_utc": None,
                    "excluded_services": [],
                },
            },
            sort_keys=True,
        ),
        "Investigation of the online-boutique incident, requested by time 2024-01-17T22:05 (ticket INC-1009). The suspected services, most likely first, are redis, cartservice, frontend; no fault type was requested; onset unknown; no services excluded.",
    ),
)

INCIDENT_REPORT_NOTE = """
The input may also be a status report of an investigation that was already carried out. Then give the parameters of that request, and ignore everything the investigation found (fault type, probability, onset). In a report: give ticket only when it says the request was by ticket; give at only when it says it was requested by time; a ticket in brackets after a time is a result, not part of the request; top_k is the number of suspects listed; want_fault_type is false when no fault type was requested, otherwise true; exclude_services are the services reported as excluded."""

INCIDENT_REPORT_EXAMPLES = (
    Example(
        "Investigation of the sock-shop incident, requested by ticket INC-1034; the suspected services, most likely first, are carts-db, carts; fault type mem (probability 0.55); onset 07:40 UTC. no services excluded.",
        {"system": "sock-shop", "ticket": "INC-1034", "top_k": 2},
    ),
    Example(
        "Investigation of the online-boutique incident, requested by time 2024-02-21T04:20 (ticket INC-1120). The suspected services, most likely first, are checkoutservice, redis, frontend; no fault type was requested; onset 04:13 UTC; excluded: adservice.",
        {
            "system": "online-boutique",
            "at": "2024-02-21T04:20",
            "want_fault_type": False,
            "exclude_services": ["adservice"],
        },
    ),
)

INCIDENT_PACK = TranslationPack(
    INVESTIGATE,
    INCIDENT_POLICY_SYSTEM,
    INCIDENT_POLICY_EXAMPLES,
    defaults=INCIDENT_DEFAULTS,
    questions=INCIDENT_QUESTIONS,
)
INCIDENT_REPORT_PACK = TranslationPack(
    INVESTIGATE,
    INCIDENT_POLICY_SYSTEM + INCIDENT_REPORT_NOTE,
    INCIDENT_POLICY_EXAMPLES + INCIDENT_REPORT_EXAMPLES,
    defaults=INCIDENT_DEFAULTS,
    questions=INCIDENT_QUESTIONS,
)
INCIDENT_SUMMARY_PACK = SummaryPack(
    INCIDENT_SUMMARY_SYSTEM, INCIDENT_SUMMARY_EXAMPLES, max_new_tokens=220
)

# ---------------------------------------------------------------------- facts

_FAULT_WORDS = {
    "cpu": r"cpu",
    "mem": r"mem(ory)?",
    "disk": r"disk|storage",
    "delay": r"delay|latency",
    "loss": r"(packet[- ])?loss",
    "socket": r"socket",
}


def _service_pattern(name: str) -> str:
    """The service as a whole word, allowing a space or underscore for its hyphens."""
    parts = [re.escape(p) for p in re.split(r"[-_]", name)]
    return r"(?<![\w-])" + r"[-_ ]?".join(parts) + r"(?![\w-])"


def _system_pattern(system: str) -> str:
    first, second = system.split("-")
    return rf"{first}[- ]?{second}"


def incident_facts(view: TraceView) -> list[Fact]:
    """What a summary of an investigation must state."""
    p, out = view.goal.params, view.outcome
    suspects = list(out["suspects"])
    facts = [
        Fact("system", (_system_pattern(out["system"]),)),
        Fact("ticket", (re.escape(out["ticket"]),)),
        Fact(
            "suspects",
            ("(?s)" + r"[^a-z]*?".join(_service_pattern(s) for s in suspects),),
        ),
    ]
    if "at" in p:
        at = str(p["at"])
        date, _, hhmm = at.partition("T")
        if hhmm:
            facts.append(
                Fact(
                    "at",
                    (
                        rf"{re.escape(date)}[T ,]*(at )?{re.escape(hhmm)}",
                        rf"(?s)(?=.*{re.escape(date)})(?=.*(?<!\d){re.escape(hhmm)}(?!\d))",
                    ),
                )
            )
        else:
            facts.append(Fact("at", (re.escape(date),)))
    if "fault_type" in out:
        words = _FAULT_WORDS[out["fault_type"]]
        facts.append(Fact("fault_type", (rf"(?<![a-z])({words})(?![a-z])",)))
        prob = float(out["fault_probability"])
        facts.append(
            Fact(
                "fault_probability",
                (_num(prob), rf"{_num(round(prob * 100, 0))}\s?(%|percent)"),
            )
        )
    else:
        facts.append(
            Fact(
                "fault_type",
                (
                    r"no fault type",
                    r"fault type[^.;]{0,20}(not|n't)[^.;]{0,15}(requested|needed|asked|wanted)",
                    r"without (a |the )?fault type",
                    r"(only|just) the suspects",
                ),
                tuple(rf"fault type[: ]+({w})\b" for w in _FAULT_WORDS.values()),
            )
        )
    onset = out.get("onset_utc")
    if onset:
        facts.append(Fact("onset", (rf"(?<!\d){re.escape(onset[-5:])}(?!\d)",)))
    else:
        facts.append(
            Fact(
                "onset",
                (
                    r"onset[^.;]{0,15}(unknown|not (known|available|clear))",
                    r"unknown onset",
                ),
            )
        )
    excluded = list(out["excluded_services"])
    if excluded:
        lookahead = "".join(f"(?=.*{_service_pattern(s)})" for s in excluded)
        facts.append(
            Fact("exclusions", (r"(?s)(?=.*exclu|.*ignor|.*left out)" + lookahead,))
        )
    else:
        facts.append(
            Fact(
                "exclusions",
                (
                    r"no (services?|components?)[^.;]{0,15}(were )?(excluded|ignored|left out)",
                    r"(nothing|none)[^.;]{0,15}(was |were )?(excluded|ignored)",
                    r"without (any )?exclusions?",
                ),
            )
        )
    return facts


# ---------------------------------------------------------------------- keyword baselines

_INCIDENT_WORDS = re.compile(
    r"\b(inc-?\d+|ticket|incident|outage|boutique|sock[- ]?shop|socks shop|shop|suspects?|culprits?|root cause)\b",
    re.I,
)
_MONTHS = {
    m: i + 1
    for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())
}
_WORD_K = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "single": 1}


def keyword_incident(text: str) -> Optional[dict[str, Any]]:
    """Regex parser for ``rca.investigate`` (the non-LLM baseline); None means refuse."""
    low = text.lower()
    if not _INCIDENT_WORDS.search(low):
        return None
    params: dict[str, Any] = {}
    if re.search(r"boutique|\bob\b", low):
        params["system"] = "online-boutique"
    elif re.search(r"sock|\bshop\b", low):
        params["system"] = "sock-shop"
    m = re.search(r"(?:inc|ticket|#)[- #]*(\d{4})", low)
    if m:
        params["ticket"] = f"INC-{m.group(1)}"
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})(?:[ t]+(\d{1,2}):(\d{2}))?", low)
    if m:
        y, mo, d, hh, mm = m.groups()
        params["at"] = f"{y}-{mo}-{d}" + (f"T{int(hh):02d}:{mm}" if hh else "")
    else:
        dm = re.search(
            r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(jan|feb|mar)[a-z]*|(jan|feb|mar)[a-z]*\s+(\d{1,2})",
            low,
        )
        if dm:
            day = int(dm.group(1) or dm.group(4))
            mon = _MONTHS[(dm.group(2) or dm.group(3))[:3]]
            tm = re.search(r"(?<![\d-])(\d{1,2}):(\d{2})", low)
            at = f"2024-{mon:02d}-{day:02d}"
            params["at"] = at + (f"T{int(tm.group(1)):02d}:{tm.group(2)}" if tm else "")
    k = re.search(
        r"(?:top[- ]?|\b)(\d|one|two|three|four|five)\s*(?:-\s*)?(?:suspects?|candidates?|services|culprits?)",
        low,
    )
    if k:
        raw = k.group(1)
        params["top_k"] = int(raw) if raw.isdigit() else _WORD_K[raw]
    elif re.search(r"\btop[- ]?(\d)\b", low):
        params["top_k"] = int(re.search(r"\btop[- ]?(\d)\b", low).group(1))  # type: ignore[union-attr]
    if re.search(
        r"(no|skip|without|not need|don'?t (care|need)|suspects only)[^.]{0,30}fault",
        low,
    ):
        params["want_fault_type"] = False
    names = [*SOCK_SHOP_SERVICES, *ONLINE_BOUTIQUE_SERVICES]
    out = [
        s
        for s in sorted(set(names), key=len, reverse=True)
        if re.search(
            rf"(ignor\w*|exclud\w*|skip|without|leave out|leaving out|minus|not|disregard)\s+(?:the\s+)?[^.;]{{0,40}}(?<![\w-]){re.escape(s)}(?![\w-])",
            low,
        )
    ]
    if out:
        params["exclude_services"] = sorted(set(out))
    return {**INCIDENT_DEFAULTS, **params}


# ---------------------------------------------------------------------- the domain


def incident_domain(executor: Any) -> Domain:
    """The incident :class:`Domain` around an :class:`~.r11_team.IncidentExecutor`."""
    return Domain(
        "incident",
        INVESTIGATE,
        INCIDENT_PACK,
        INCIDENT_SUMMARY_PACK,
        INCIDENT_REPORT_PACK,
        executor,
        incident_facts,
        keyword_incident,
        INCIDENT_SPECIALIST,
        INCIDENT_DEFAULTS,
        incident_outcome,
    )


def entry_dict(entry: CatalogueEntry) -> dict[str, Any]:
    return entry.to_dict()


# ---------------------------------------------------------------------- the router

ROUTE_SYSTEM = """You route a request to one of three desks and answer {"domain": "<desk>"}.
- squad: orders for a squad of three foraging snakes (red, green, blue): who forages, how cautious, how many steps, a food target.
- plant: configuring the plant-floor fault-diagnosis system with four sensor nodes (reactor, separator, stripper, compressor_utilities): byte budget per message, false-alarm rate, which nodes transmit.
- incident: investigating a past incident of the sock shop or the online boutique web shops: a ticket (INC-nnnn) or a time, the suspect services, the fault type.
- none: anything else (chit-chat, general questions, requests to restart, change or delete things).
Choose the desk by topic, even when the request is malformed, asks for something impossible or is only half specified."""

ROUTE_EXAMPLES = (
    ("Send the red snake out to gather food for 90 steps.", "squad"),
    ("Let blue and green forage but keep it safe, and stop at 12 food.", "squad"),
    ("Use 16 byte messages with 2% false alarms on every node but the stripper.", "plant"),
    ("Set the diagnosis to 51 bytes and only let the reactor transmit.", "plant"),
    ("The boutique crashed, so what is behind ticket INC-1030 this time?", "incident"),
    ("Which services are to blame for the shop incident around 15:00 on 19 January?", "incident"),
    ("Do you know any good pasta recipes?", "none"),
    ("Please shut down the database cluster.", "none"),
)  # fmt: skip

_ROUTE_WORDS = {
    "squad": re.compile(
        r"\b(snake|snakes|squad|red|green|blue|forage|foraging|food|caution|cautious|careful)\b",
        re.I,
    ),
    "plant": re.compile(
        r"\b(node|nodes|plant|alarm|alarms|byte|bytes|sensor|uplink|reactor|separator|stripper|compressor|utilities|octets?)\b",
        re.I,
    ),
    "incident": _INCIDENT_WORDS,
}


def route_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["domain"],
        "properties": {"domain": {"type": "string", "enum": [*DOMAINS, "none"]}},
        "additionalProperties": False,
    }


def keyword_route(text: str) -> str:
    """The keyword baseline's router: the desk whose vocabulary matches most; else none."""
    scores = {
        d: len({w.lower() for w in rx.findall(text)}) for d, rx in _ROUTE_WORDS.items()
    }
    best = max(scores, key=lambda d: (scores[d], d == "incident"))
    return best if scores[best] > 0 else "none"
