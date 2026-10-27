"""R11 v2 (goal G8b): ``rca.investigate`` v2, the v2 prompts and the v2 domain.

G8 was killed on one criterion: four sealed requests gave a day without a month ("on the 17th at
10:20") and the translator, asked to know the catalogue, wrote the desk clock's month. v2 moves
that knowledge to the desk. Relative to :mod:`.r11_incident` (v1, untouched and reproducible):

* **Spec v2.** ``at`` carries exactly what the user gave, in ISO-8601 truncated forms: a full date
  (``2024-01-20``), a month and day (``--01-20``) or a day of the month (``---20``), each with an
  optional ``THH:MM``. The catalogue resolves it (policy in
  :mod:`..envs.logtriage.catalogue`): one matching date and one incident resolves, several dates
  or incidents ask, none refuses.
* **Summary.** It states the requested top-k and how the incident was named (ticket, or the time
  reference exactly as in the goal), so the re-translation recovers the same GoalSpec (all five v1
  round-trip failures lost ``top_k``).
* **Policy edges.** An out-of-range count ("top 9 suspects") must be passed on as stated so the
  checker refuses it (v1 clamped it to 5); a request with no incident reference asks.

Prompts and few-shots here were tuned on ``data/r11b_dev.jsonl`` and the v1 dev failures only;
``data/r11b_sealed_test.jsonl`` was fixed before any of this was written.
"""

import dataclasses
import json
import re
from typing import Any, Optional

from ..envs.logtriage.catalogue import AT_PATTERN_V2, TICKET_PATTERN
from ..protocol.goalspec import SpecDefinition, SpecRegistry
from ..translator import Example, SummaryPack, TranslationPack
from . import r11_incident as v1
from .r7a_domains import Domain, build_registry

SPEC_VERSION = 2


def investigate_definition_v2() -> SpecDefinition:
    """``rca.investigate`` v2: ``at`` may be a full, month-day or day-only date."""
    base = v1.investigate_definition()
    params = json.loads(json.dumps(base.params_schema))
    params["properties"]["at"] = {"type": "string", "pattern": AT_PATTERN_V2}
    params["properties"]["ticket"] = {"type": "string", "pattern": TICKET_PATTERN}
    return SpecDefinition(
        v1.INVESTIGATE,
        SPEC_VERSION,
        params,
        "Investigate one past incident of the Sock Shop or Online Boutique: name it by "
        "ticket or by when it happened (as much of the date as the user gave, plus an "
        "optional time); get the top suspect services and the fault type",
        base.result_schema,
    )


def build_registry_r11b(backend: str = "auto") -> SpecRegistry:
    """The R7a registry plus ``rca.investigate`` v2 (v1 is not registered here)."""
    registry = build_registry(backend)
    registry.register(investigate_definition_v2())
    return registry


# ---------------------------------------------------------------------- packs

_V1_AT = (
    '- at: when the incident happened. "YYYY-MM-DD" when only a day is given, '
    '"YYYY-MM-DDTHH:MM" when a time of day is given ("2:30 pm on 20 January" is "2024-01-20T14:30"). '
    'Work out "yesterday", "two days ago" or "last Friday" from the desk clock. A part of the day '
    'without a clock time ("in the afternoon") gives the day only. Leave it out when no time is given.'
)
_V2_AT = (
    "- at: when the incident happened, written with exactly as much of the date as the request says. Never complete it: "
    "do not add a month or a year the request does not say, even if the desk clock suggests one.\n"
    '  - A day of the month alone ("on the 17th", "the 20th") is "---17", "---20".\n'
    '  - A month and day with no year ("Jan 19", "19 January", "the 20th of February") is "--01-19", "--02-20".\n'
    '  - A date with a year ("19 January 2024", "2024-01-19") is "2024-01-19".\n'
    '  - "yesterday", "two days ago" or "the day before yesterday" are worked out from the desk clock to a full date.\n'
    '  - Add "THH:MM" (24-hour) when a clock time is given: "---17T10:20", "--01-19T21:30" for "9:30 pm", '
    '"---20T00:47" for "twelve forty-seven at night", "---19T18:45" for "a quarter to seven in the evening", "---20T23:50" for "11:50 pm".\n'
    '  - A part of the day without a clock time ("in the afternoon") gives the date only. Leave "at" out when no date or time is given.'
)
_V1_TOPK = "- top_k: integer, how many suspect services to list. Leave it out when not stated (default 3)."
_V2_TOPK = (
    "- top_k: integer, how many suspect services to list. Leave it out when not stated (default 3). "
    "Copy a stated number exactly, also when it is above 5 or below 1 (9, 12, 15, 0): never lower or raise it "
    "to fit; the checker refuses it."
)
_V1_CLARIFY = (
    'clarify (one short question) only when the request is about an incident but names it too vaguely to look up ("recently", '
    '"last week", "a while ago", "that outage") and gives no ticket or time. Things the request does not mention take their defaults; '
    "a missing system is asked for by the checker, so just leave it out."
)
_V2_CLARIFY = (
    "clarify (one short question) when the request is about investigating an incident but gives no ticket and no date or time "
    '("investigate an incident", "something odd is going on with the shop", "that outage last week", "a while ago"). '
    "That is a question to the requester, never a refusal. Things the request does not mention take their defaults; "
    "a missing system is asked for by the checker, so just leave it out."
)


def _policy_system_v2() -> str:
    text = v1.INCIDENT_POLICY_SYSTEM
    for old, new in (
        (_V1_AT, _V2_AT),
        (_V1_TOPK, _V2_TOPK),
        (_V1_CLARIFY, _V2_CLARIFY),
    ):
        if old not in text:
            raise ValueError(f"v1 policy text changed: {old[:40]!r}")
        text = text.replace(old, new)
    return text.replace(
        "Do NOT refuse because a ticket looks unknown, a number is large or a service is not in the lists: put it in params and the checker will reject it. Allowed values, for reference: top_k 1 to 5.",
        "Do NOT refuse because a ticket looks unknown, a number is large or a service is not in the lists: put it in params and the checker will reject it. Do not change a number or a date to make it fit. Allowed values, for reference: top_k 1 to 5.",
    )


INCIDENT_POLICY_SYSTEM_V2 = _policy_system_v2()

INCIDENT_POLICY_EXAMPLES_V2 = (
    Example(
        "Could you walk me through ticket INC-1064 from the boutique?",
        {"system": "online-boutique", "ticket": "INC-1064"},
    ),
    Example(
        "The online boutique stalled at 9:30 pm on Friday 19 January, tell me what was at fault.",
        {"system": "online-boutique", "at": "--01-19T21:30"},
    ),
    Example(
        "The shop fell over on the 18th at about a quarter past two in the afternoon.",
        {"system": "sock-shop", "at": "---18T14:15"},
    ),
    Example(
        "Online boutique, half past midnight on the 20th.",
        {"system": "online-boutique", "at": "---20T00:30"},
    ),
    Example(
        "Boutique problem on 3 March 2024 at 11:15, what was behind it?",
        {"system": "online-boutique", "at": "2024-03-03T11:15"},
    ),
    Example(
        "Have a go at what hit the sock shop on 3 February.",
        {"system": "sock-shop", "at": "--02-03"},
    ),
    Example(
        "Yesterday at 6 am the shop had trouble. Can you tell why?",
        {"system": "sock-shop", "at": "2024-02-21T06:00"},
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
        "Pull INC-1066 from the queue and have a go.",
        {"ticket": "INC-1066"},
    ),
    Example(
        "Boutique ticket INC-1031 - no, sorry, I meant the sock shop one, INC-1037. Use that.",
        {"system": "sock-shop", "ticket": "INC-1037"},
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
        "Give me a list of 9 suspects for the shop's INC-1075.",
        {"system": "sock-shop", "ticket": "INC-1075", "top_k": 9},
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
    Example(
        "Please investigate something on the boutique.",
        None,
        "",
        "Which incident? Give a ticket or a date and time.",
    ),
)

INCIDENT_SUMMARY_SYSTEM_V2 = """You write the status report of a finished incident investigation for an on-call engineer, from a JSON trace.
Write two to four plain sentences. State, using only the trace:
- the system, and how the incident was requested: "requested by ticket INC-nnnn" when the goal has a ticket, "requested by time <the goal's at value, copied exactly>" when it has an at value (then add "(ticket INC-nnnn)" from the outcome); both when the goal has both;
- the requested count, always, as "top <goal.top_k> suspects requested" (copy the number from the goal's top_k, which is also the length of outcome.suspects);
- the suspected services in order, exactly as in outcome.suspects, most likely first;
- when outcome has a fault type, "fault type <name> (probability <p>)"; when it has none, say that no fault type was requested;
- the onset as "onset <HH:MM> UTC" from outcome.onset_utc, or "onset unknown" when it is null;
- "excluded: <names>" for outcome.excluded_services, or "no services excluded" when empty."""


def _example(
    params: dict[str, Any], outcome: dict[str, Any], summary: str
) -> tuple[str, str]:
    trace = {
        "goal": {"spec": v1.INVESTIGATE, "params": params},
        "messages": [
            {
                "sender": "command-centre",
                "recipients": [v1.INCIDENT_SPECIALIST],
                "payload_type": "json",
                "codec": "goalspec",
                "clock": 1,
            },
            {
                "sender": v1.INCIDENT_SPECIALIST,
                "recipients": ["command-centre"],
                "payload_type": "json",
                "codec": "result",
                "clock": 2,
            },
        ],
        "outcome": {"state": "completed", **outcome},
    }
    return json.dumps(trace, sort_keys=True), summary


INCIDENT_SUMMARY_EXAMPLES_V2 = (
    _example(
        {
            "exclude_services": ["user"],
            "system": "sock-shop",
            "ticket": "INC-1073",
            "top_k": 2,
            "want_fault_type": True,
        },
        {
            "ticket": "INC-1073",
            "system": "sock-shop",
            "suspects": ["orders", "carts"],
            "fault_type": "cpu",
            "fault_probability": 0.71,
            "onset_utc": "2024-01-20T06:12",
            "excluded_services": ["user"],
        },
        "Investigation of the sock-shop incident, requested by ticket INC-1073, top 2 suspects requested. The suspected services, most likely first, are orders, carts; fault type cpu (probability 0.71); onset 06:12 UTC. Excluded: user.",
    ),
    _example(
        {
            "at": "---17T22:05",
            "exclude_services": [],
            "system": "online-boutique",
            "top_k": 3,
            "want_fault_type": False,
        },
        {
            "ticket": "INC-1009",
            "system": "online-boutique",
            "suspects": ["redis", "cartservice", "frontend"],
            "onset_utc": None,
            "excluded_services": [],
        },
        "Investigation of the online-boutique incident, requested by time ---17T22:05 (ticket INC-1009), top 3 suspects requested. The suspected services, most likely first, are redis, cartservice, frontend; no fault type was requested; onset unknown; no services excluded.",
    ),
)

INCIDENT_REPORT_NOTE_V2 = """
The input may also be a status report of an investigation that was already carried out. Then give the parameters of that request, and ignore everything the investigation found (fault type, probability, onset). In a report: give ticket only when it says the request was by ticket; give at only when it says it was requested by time, and copy that time exactly as written (such as ---17T22:05 or --01-19T21:30), never completing or correcting it; a ticket in brackets after a time is a result, not part of the request; top_k is the number in "top N suspects requested" (otherwise the number of suspects listed); want_fault_type is false when no fault type was requested, otherwise true; exclude_services are the services reported as excluded."""

INCIDENT_REPORT_EXAMPLES_V2 = (
    Example(
        "Investigation of the sock-shop incident, requested by ticket INC-1034, top 2 suspects requested; the suspected services, most likely first, are carts-db, carts; fault type mem (probability 0.55); onset 07:40 UTC. no services excluded.",
        {"system": "sock-shop", "ticket": "INC-1034", "top_k": 2},
    ),
    Example(
        "Investigation of the online-boutique incident, requested by time --01-18T01:55 (ticket INC-1018), top 1 suspects requested. The suspected services, most likely first, are checkoutservice; no fault type was requested; onset 01:48 UTC; excluded: adservice.",
        {
            "system": "online-boutique",
            "at": "--01-18T01:55",
            "top_k": 1,
            "want_fault_type": False,
            "exclude_services": ["adservice"],
        },
    ),
)

INCIDENT_PACK_V2 = TranslationPack(
    v1.INVESTIGATE,
    INCIDENT_POLICY_SYSTEM_V2,
    INCIDENT_POLICY_EXAMPLES_V2,
    version=SPEC_VERSION,
    defaults=v1.INCIDENT_DEFAULTS,
    questions=v1.INCIDENT_QUESTIONS,
)
INCIDENT_REPORT_PACK_V2 = TranslationPack(
    v1.INVESTIGATE,
    INCIDENT_POLICY_SYSTEM_V2 + INCIDENT_REPORT_NOTE_V2,
    INCIDENT_POLICY_EXAMPLES_V2 + INCIDENT_REPORT_EXAMPLES_V2,
    version=SPEC_VERSION,
    defaults=v1.INCIDENT_DEFAULTS,
    questions=v1.INCIDENT_QUESTIONS,
)
INCIDENT_SUMMARY_PACK_V2 = SummaryPack(
    INCIDENT_SUMMARY_SYSTEM_V2, INCIDENT_SUMMARY_EXAMPLES_V2, max_new_tokens=240
)

# ---------------------------------------------------------------------- keyword baseline

_MONTHS = v1._MONTHS
_DAY_FIRST = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(jan|feb|mar)[a-z]*(?:\s+(\d{4}))?"
)
_MONTH_FIRST = re.compile(
    r"(jan|feb|mar)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?"
)
_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_DAY_ONLY = re.compile(r"(?<![\w:-])(\d{1,2})(?:st|nd|rd|th)\b")
_CLOCK = re.compile(r"(?<![\d-])(\d{1,2}):(\d{2})\s*(am|pm)?")


def _keyword_at(low: str) -> Optional[str]:
    """The regex parser's v2 ``at``: only what the text says, no completion."""
    if m := _ISO.search(low):
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    elif "day before yesterday" in low or "two days ago" in low or "2 days ago" in low:
        date = "2024-02-20"
    elif "yesterday" in low:
        date = "2024-02-21"
    elif m := _DAY_FIRST.search(low):
        mon, day, year = _MONTHS[m.group(2)], int(m.group(1)), m.group(3)
        date = f"{year}-{mon:02d}-{day:02d}" if year else f"--{mon:02d}-{day:02d}"
    elif m := _MONTH_FIRST.search(low):
        mon, day, year = _MONTHS[m.group(1)], int(m.group(2)), m.group(3)
        date = f"{year}-{mon:02d}-{day:02d}" if year else f"--{mon:02d}-{day:02d}"
    elif m := _DAY_ONLY.search(low):
        date = f"---{int(m.group(1)):02d}"
    else:
        return None
    tm = _CLOCK.search(re.sub(r"\d{4}-\d{2}-\d{2}", " ", low))
    if tm:
        hh = int(tm.group(1))
        if tm.group(3) == "pm" and hh < 12:
            hh += 12
        elif tm.group(3) == "am" and hh == 12:
            hh = 0
        return f"{date}T{hh:02d}:{tm.group(2)}"
    return date


def keyword_incident_v2(text: str) -> Optional[dict[str, Any]]:
    """Regex parser for ``rca.investigate`` v2 (the non-LLM baseline); None means refuse."""
    params = v1.keyword_incident(text)
    if params is None:
        return None
    params.pop("at", None)
    at = _keyword_at(text.lower())
    if at is not None:
        params["at"] = at
    return params


# ---------------------------------------------------------------------- the domain


def incident_domain_v2(executor: Any) -> Domain:
    """The v2 incident :class:`Domain` (same executor, facts and specialist as v1)."""
    return dataclasses.replace(
        v1.incident_domain(executor),
        pack=INCIDENT_PACK_V2,
        summary_pack=INCIDENT_SUMMARY_PACK_V2,
        report_pack=INCIDENT_REPORT_PACK_V2,
        keyword=keyword_incident_v2,
    )
