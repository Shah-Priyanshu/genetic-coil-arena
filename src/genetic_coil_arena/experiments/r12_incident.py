"""R12 (goal G10): ``rca.investigate`` v3, the same desk grown to a new system and new incidents.

The desk of R11b learns about Train Ticket and the 62 RE3 test incidents without a prompt
edit. Relative to :mod:`.r11b_incident` (v2, untouched and reproducible):

* **Spec v3.** ``system`` gains ``train-ticket``; nothing else changes (``at`` keeps the v2
  forms, ``ticket`` the ``INC-nnnn`` pattern).
* **Prompts.** The v3 pack is the v2 pack with exactly two kinds of substitution, both
  generated, not written: the desk clock line takes the R12 clock (:data:`DESK_NOW_TEXT`; the one
  few-shot whose gold date is "yesterday" takes the date that clock gives), and the line that
  lists the allowed ``system`` values is generated from the v3 enum. :func:`prompt_diff` and
  ``tests/test_r12.py`` assert that nothing else differs; the system aliases paragraph, the
  few-shots, the summary prompt and the router prompt are the v2 ones, byte for byte.
* **Keyword baseline.** :func:`keyword_incident_v3` is the v2 regex parser extended to the new
  system ("train ticket" and its ``ts-*`` services), to all twelve months and to the R12 clock
  for "yesterday". It was written after the sealed texts and before any run on them, from the
  documented names of the system only.
"""

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Optional

from ..envs.logtriage import rcaeval
from ..envs.logtriage.catalogue import DESK_NOW_R12, R12_TICKET_BASE, IncidentCatalogue
from ..protocol.goalspec import SpecDefinition, SpecRegistry
from ..translator import Example, TranslationPack
from . import r11_incident as v1
from . import r11_team
from . import r11b_incident as v2
from .r7a_domains import Domain, build_registry

SPEC_VERSION = 3
SYSTEMS_V3 = (*v1.SYSTEMS, "train-ticket")
DESK_NOW_TEXT = "Wednesday 2024-12-11, 09:00 UTC"
YESTERDAY = "2024-12-10"
DAY_BEFORE = "2024-12-09"
TRAIN_TICKET_SERVICES = (
    "ts-admin-basic-info-service", "ts-admin-order-service", "ts-admin-route-service",
    "ts-admin-travel-service", "ts-admin-user-service", "ts-assurance-mongo",
    "ts-assurance-service", "ts-auth-mongo", "ts-auth-service", "ts-avatar-service",
    "ts-basic-service", "ts-cancel-service", "ts-config-mongo", "ts-config-service",
    "ts-consign-mongo", "ts-consign-price-mongo", "ts-consign-price-service",
    "ts-consign-service", "ts-contacts-mongo", "ts-contacts-service", "ts-execute-service",
    "ts-food-map-mongo", "ts-food-map-service", "ts-food-mongo", "ts-food-mysql",
    "ts-food-service", "ts-inside-payment-mongo", "ts-inside-payment-service",
    "ts-news-service", "ts-notification-mongo", "ts-notification-service",
    "ts-order-mongo", "ts-order-other-mongo", "ts-order-other-service", "ts-order-service",
    "ts-payment-mongo", "ts-payment-service", "ts-preserve-mongo", "ts-preserve-other-mongo",
    "ts-preserve-other-service", "ts-preserve-service", "ts-price-mongo", "ts-price-service",
    "ts-rebook-service", "ts-route-mongo", "ts-route-plan-service", "ts-route-service",
    "ts-seat-service", "ts-security-mongo", "ts-security-service", "ts-station-mongo",
    "ts-station-service", "ts-ticket-office-mongo", "ts-ticket-office-service",
    "ts-ticketinfo-service", "ts-train-mongo", "ts-train-service", "ts-travel-mongo",
    "ts-travel-plan-service", "ts-travel-service", "ts-travel2-mongo", "ts-travel2-service",
    "ts-ui-dashboard", "ts-user-mongo", "ts-user-service", "ts-verification-code-service",
    "ts-voucher-mysql", "ts-voucher-service",
)  # fmt: skip
INCIDENT_QUESTIONS_V3 = {
    "system": "Which system is this about, sock-shop, online-boutique or train-ticket?"
}


# ---------------------------------------------------------------------- catalogue


def load_catalogue_r12(cache_dir: Optional[Path] = None) -> IncidentCatalogue:
    """The grown catalogue: the 120 RE2 test incidents (``INC-1001``..) and the 62 RE3 test
    incidents (``INC-2001``.., numbered by alert time), read at the R12 desk clock.

    RE3 repetition 1 is the dev split and is not catalogued. RE3 was recorded in November and
    December 2024, after the R11 clock (2024-02-22), so the R12 clock is
    :data:`~..envs.logtriage.catalogue.DESK_NOW_R12`; the RE2 tickets keep their numbers.
    """
    re2 = r11_team.load_catalogue(cache_dir)
    files = [
        f
        for f in rcaeval.list_incidents(cache_dir, rcaeval.RE3_SYSTEMS)
        if int(f.stem.rsplit("__", 1)[1]) not in rcaeval.DEV_REPS
    ]
    if not files:
        raise FileNotFoundError(
            "no converted RE3 incidents: run `python scripts/fetch_rcaeval.py --systems "
            "RE3-SS,RE3-OB,RE3-TT`"
        )
    re3 = IncidentCatalogue.from_files(files, R12_TICKET_BASE, DESK_NOW_R12)
    return IncidentCatalogue.merged([re2, re3], DESK_NOW_R12)


# ---------------------------------------------------------------------- spec


def investigate_definition_v3() -> SpecDefinition:
    """``rca.investigate`` v3: v2 with ``train-ticket`` among the systems."""
    base = v2.investigate_definition_v2()
    params = json.loads(json.dumps(base.params_schema))
    params["properties"]["system"]["enum"] = list(SYSTEMS_V3)
    result = json.loads(json.dumps(base.result_schema))
    result["properties"]["public"]["properties"]["system"]["enum"] = list(SYSTEMS_V3)
    return SpecDefinition(
        v1.INVESTIGATE,
        SPEC_VERSION,
        params,
        "Investigate one past incident of the Sock Shop, the Online Boutique or Train "
        "Ticket: name it by ticket or by when it happened (as much of the date as the user "
        "gave, plus an optional time); get the top suspect services and the fault type",
        result,
    )


def build_registry_r12(backend: str = "auto") -> SpecRegistry:
    """The R7a registry plus ``rca.investigate`` v3 (v1 and v2 are not registered here)."""
    registry = build_registry(backend)
    registry.register(investigate_definition_v3())
    return registry


# ---------------------------------------------------------------------- packs


def system_choices(systems: tuple[str, ...] = SYSTEMS_V3) -> str:
    """The allowed system values as the prompt lists them: ``"a", "b" or "c"``."""
    quoted = [f'"{s}"' for s in systems]
    return ", ".join(quoted[:-1]) + f" or {quoted[-1]}"


_V2_CHOICES = system_choices(v1.SYSTEMS)


def _policy_system_v3() -> str:
    text = v2.INCIDENT_POLICY_SYSTEM_V2
    for old, new in (
        (v1.DESK_NOW, DESK_NOW_TEXT),
        (f"- system: {_V2_CHOICES}.", f"- system: {system_choices()}."),
    ):
        if text.count(old) != 1:
            raise ValueError(f"v2 policy text changed: {old[:40]!r}")
        text = text.replace(old, new)
    return text


def _shift_clock(example: Example) -> Example:
    """The v2 "yesterday" few-shot with the date the R12 clock gives."""
    params = example.params
    if params and params.get("at") == "2024-02-21T06:00":
        return dataclasses.replace(
            example, params={**params, "at": f"{YESTERDAY}T06:00"}
        )
    return example


INCIDENT_POLICY_SYSTEM_V3 = _policy_system_v3()
INCIDENT_POLICY_EXAMPLES_V3 = tuple(
    _shift_clock(e) for e in v2.INCIDENT_POLICY_EXAMPLES_V2
)
INCIDENT_REPORT_EXAMPLES_V3 = v2.INCIDENT_REPORT_EXAMPLES_V2
INCIDENT_PACK_V3 = TranslationPack(
    v1.INVESTIGATE,
    INCIDENT_POLICY_SYSTEM_V3,
    INCIDENT_POLICY_EXAMPLES_V3,
    version=SPEC_VERSION,
    defaults=v1.INCIDENT_DEFAULTS,
    questions=INCIDENT_QUESTIONS_V3,
)
INCIDENT_REPORT_PACK_V3 = TranslationPack(
    v1.INVESTIGATE,
    INCIDENT_POLICY_SYSTEM_V3 + v2.INCIDENT_REPORT_NOTE_V2,
    INCIDENT_POLICY_EXAMPLES_V3 + INCIDENT_REPORT_EXAMPLES_V3,
    version=SPEC_VERSION,
    defaults=v1.INCIDENT_DEFAULTS,
    questions=INCIDENT_QUESTIONS_V3,
)
INCIDENT_SUMMARY_PACK_V3 = v2.INCIDENT_SUMMARY_PACK_V2


def prompt_diff() -> dict[str, Any]:
    """Every difference between the v2 and v3 prompt material (the G10 audit).

    ``system`` lists the changed lines of the system prompt (v2 line, v3 line); ``examples``
    the few-shots whose params differ; the summary pack, the report note and the router
    prompt are checked for identity.
    """
    old, new = (
        v2.INCIDENT_POLICY_SYSTEM_V2.splitlines(),
        INCIDENT_POLICY_SYSTEM_V3.splitlines(),
    )
    changed = [(a, b) for a, b in zip(old, new) if a != b]
    shots = [
        (a.text, a.params, b.params)
        for a, b in zip(v2.INCIDENT_POLICY_EXAMPLES_V2, INCIDENT_POLICY_EXAMPLES_V3)
        if a != b
    ]
    return {
        "system_lines_changed": changed,
        "same_line_count": len(old) == len(new),
        "examples_changed": shots,
        "n_examples": len(INCIDENT_POLICY_EXAMPLES_V3),
        "summary_pack_same": INCIDENT_SUMMARY_PACK_V3 is v2.INCIDENT_SUMMARY_PACK_V2,
        "report_note_same": True,
        "router_same": True,
    }


# ---------------------------------------------------------------------- keyword baseline

_TT = re.compile(r"train[- ]?ticket|trian[- ]?ticket|\btt\b", re.I)
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_DAY_FIRST = re.compile(
    rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{_MON}(?:,?\s+(\d{{4}}))?"
)
_MONTH_FIRST = re.compile(
    rf"{_MON}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?"
)
_ISO = v2._ISO
_DAY_ONLY = v2._DAY_ONLY
_CLOCK = v2._CLOCK


def _keyword_at_v3(low: str) -> Optional[str]:
    """The regex parser's v3 ``at``: only what the text says, all months, the R12 clock."""
    if m := _ISO.search(low):
        date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    elif "day before yesterday" in low or "two days ago" in low or "2 days ago" in low:
        date = DAY_BEFORE
    elif "yesterday" in low:
        date = YESTERDAY
    elif m := _DAY_FIRST.search(low):
        mon, day, year = v1._MONTHS[m.group(2)], int(m.group(1)), m.group(3)
        date = f"{year}-{mon:02d}-{day:02d}" if year else f"--{mon:02d}-{day:02d}"
    elif m := _MONTH_FIRST.search(low):
        mon, day, year = v1._MONTHS[m.group(1)], int(m.group(2)), m.group(3)
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


def keyword_incident_v3(text: str) -> Optional[dict[str, Any]]:
    """Regex parser for ``rca.investigate`` v3 (the non-LLM baseline); None means refuse."""
    named = _TT.search(text) is not None
    # "train ticket" is a system name, not a ticket reference: the v1 ticket regex would
    # otherwise read "train ticket 2024-12-08" as INC-2024
    scrub = _TT.sub(" incident ", text) if named else text
    params = v1.keyword_incident(scrub)
    if params is None:
        return None
    params.pop("at", None)
    at = _keyword_at_v3(scrub.lower())
    if at is not None:
        params["at"] = at
    if named:
        params["system"] = "train-ticket"
        low = scrub.lower()
        found = [
            s
            for s in sorted(TRAIN_TICKET_SERVICES, key=len, reverse=True)
            if re.search(
                rf"(ignor\w*|exclud\w*|skip|without|leave out|leaving out|minus|not|disregard)\s+(?:the\s+)?[^.;]{{0,60}}(?<![\w-]){re.escape(s)}(?![\w-])",
                low,
            )
        ]
        if found:
            params["exclude_services"] = sorted(set(found))
    return params


# ---------------------------------------------------------------------- the domain


def incident_domain_v3(executor: Any) -> Domain:
    """The v3 incident :class:`Domain` (same executor, facts and specialist as v1 and v2)."""
    return dataclasses.replace(
        v1.incident_domain(executor),
        pack=INCIDENT_PACK_V3,
        summary_pack=INCIDENT_SUMMARY_PACK_V3,
        report_pack=INCIDENT_REPORT_PACK_V3,
        keyword=keyword_incident_v3,
    )
