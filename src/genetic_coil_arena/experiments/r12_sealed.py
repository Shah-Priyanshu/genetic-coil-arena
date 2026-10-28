"""R12 (goal G10): the sealed incident-request set of the grown desk and its dev set.

``data/r12_sealed_test.jsonl`` holds 107 hand-written requests to the incident desk after it
grew by the 62 RE3 test incidents (Sock Shop, Online Boutique and the new Train Ticket system):
91 with a gold GoalSpec (66 name an RE3 incident, 32 of them on Train Ticket; 25 name an RE2
incident as a regression check), 6 that must be answered with a question and 10 that must be
refused. It was written after the R11b results and before any R12 translator run, from the
catalogue of tickets and times and the documented desk policy (never from model outputs). The
keyword parser's extension to Train Ticket was written after the texts and before any run of it
on them, from the documented names of the system only, and checked on the dev set alone.

Gold convention (``rca.investigate`` v3, same as v2): the gold ``at`` is the transcription of what
the user said, never completed from the catalogue ("on the 8th at 10:26" is ``---08T10:26``), and
"yesterday" is worked out from the R12 desk clock (Wednesday 2024-12-11 09:00 UTC, so
``2024-12-10``). Each spec item has exactly one natural reading and a ``resolves_to`` ticket: the
one incident the desk policy must resolve the gold to against the R12 catalogue (RE2 tickets
``INC-1001``..``INC-1120``, RE3 test tickets ``INC-2001``..``INC-2062``). RE3 incidents come in
bursts (up to 18 a day per system), so a date without a time is a ``clarify`` item.

Its SHA-256 (over the LF-normalised bytes) is pinned here and asserted by a test and by every
script that loads it. **Never tune on it.** ``data/r12_dev.jsonl`` (41 items) is the dev set;
G10 forbids prompt changes, so it is only a check before the sealed runs.
"""

import json
from pathlib import Path
from typing import Optional

from .r11_sealed import DOMAIN, IncidentItem, _load, file_sha256

DATA = Path(__file__).parent / "data"
SEALED_PATH = DATA / "r12_sealed_test.jsonl"
DEV_PATH = DATA / "r12_dev.jsonl"
SEALED_SHA256 = "a0285faf8d99d69e1e732ad02a0bca8ca205c2ef7f51606bd11e322dc614c1ea"

__all__ = [
    "DOMAIN",
    "IncidentItem",
    "SEALED_SHA256",
    "file_sha256",
    "group_of",
    "load_dev",
    "load_resolves_to",
    "load_sealed",
]


def load_sealed(verify: bool = True, path: Path = SEALED_PATH) -> list[IncidentItem]:
    """The R12 sealed items; raises if the file does not match the pinned hash."""
    if verify and file_sha256(path) != SEALED_SHA256:
        raise ValueError(
            f"{path.name} does not match the pinned SHA-256 {SEALED_SHA256}; "
            "the sealed test set must not change"
        )
    return _load(path, "sealed")


def load_dev(path: Path = DEV_PATH) -> list[IncidentItem]:
    """The R12 dev set (a check before the sealed run; never the verdict)."""
    return _load(path, "dev")


def load_resolves_to(path: Path = SEALED_PATH) -> dict[str, Optional[str]]:
    """``item id -> ticket`` the gold must resolve to (None for clarify / refuse items)."""
    out: dict[str, Optional[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row.get("resolves_to")
    return out


def group_of(ticket: Optional[str]) -> str:
    """``re3`` for the RE3 tickets (``INC-2xxx``), ``re2`` for ``INC-1xxx``, else ``none``."""
    if not ticket:
        return "none"
    return "re3" if ticket.startswith("INC-2") else "re2"
