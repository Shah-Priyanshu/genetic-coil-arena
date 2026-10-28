"""R12c (goal G10b): the sealed incident-request set of the literal-check desk and its dev set.

``data/r12c_sealed_test.jsonl`` holds 106 hand-written requests to the R12 incident desk (spec
v3, desk clock Wednesday 2024-12-11 09:00 UTC) over the same catalogue as R12: 84 with a gold
GoalSpec, 8 that must be answered with a question and 14 that must be refused. It was written
after the R12 results and before any R12c translator run, from the catalogue of tickets and
times and the documented desk policy (never from model outputs), and it is weighted toward the
failure classes of the G10 sealed run: dates without a year (month and day, day only), dates
with a year, relative days, top-k in digits and in words, beside tickets, exclusions, fault-type
requests and combinations.

Gold convention as in R12: the gold ``at`` transcribes what the user said (never completed
from the catalogue), "yesterday" and "N days ago" are worked out from the desk clock, and each
spec item has one natural reading and a ``resolves_to`` ticket.

**Guess traps** (``"trap": true``, 13 items) are the requests where a literal the model made
up could turn a question or a refusal into a silent answer:

* five ``clarify`` items name the online boutique's 20th: it has incidents on both 2024-01-20
  and 2024-02-20, so the day alone is ambiguous, but a guessed month resolves to one incident
  (the catalogue has no other system-and-day that exists in two months);
* six ``refuse`` items ask for more suspects than the schema allows (7 to 12), which a model
  that clamps to 5 would answer;
* two spec items name that same 20th with a time that fits one of the two dates only.

Its SHA-256 (over the LF-normalised bytes) is pinned here and asserted by a test and by every
script that loads it. **Never tune on it.** ``data/r12c_dev.jsonl`` (31 items) is the dev set
for smoke checks.
"""

import json
from pathlib import Path
from typing import Optional

from .r11_sealed import DOMAIN, IncidentItem, _load, file_sha256
from .r12_sealed import group_of

DATA = Path(__file__).parent / "data"
SEALED_PATH = DATA / "r12c_sealed_test.jsonl"
DEV_PATH = DATA / "r12c_dev.jsonl"
SEALED_SHA256 = "e20d15a588c6b0f35be26d668e9b9817ce1ea7147fd436ac55fd4af00726968f"

__all__ = [
    "DOMAIN",
    "IncidentItem",
    "SEALED_SHA256",
    "file_sha256",
    "group_of",
    "load_dev",
    "load_resolves_to",
    "load_sealed",
    "load_traps",
]


def load_sealed(verify: bool = True, path: Path = SEALED_PATH) -> list[IncidentItem]:
    """The R12c sealed items; raises if the file does not match the pinned hash."""
    if verify and file_sha256(path) != SEALED_SHA256:
        raise ValueError(
            f"{path.name} does not match the pinned SHA-256 {SEALED_SHA256}; "
            "the sealed test set must not change"
        )
    return _load(path, "sealed")


def load_dev(path: Path = DEV_PATH) -> list[IncidentItem]:
    """The R12c dev set (a smoke check before the sealed run; never the verdict)."""
    return _load(path, "dev")


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_resolves_to(path: Path = SEALED_PATH) -> dict[str, Optional[str]]:
    """``item id -> ticket`` the gold must resolve to (None for clarify / refuse items)."""
    return {r["id"]: r.get("resolves_to") for r in _rows(path)}


def load_traps(path: Path = SEALED_PATH) -> set[str]:
    """Ids of the guess-trap items."""
    return {r["id"] for r in _rows(path) if r.get("trap")}
