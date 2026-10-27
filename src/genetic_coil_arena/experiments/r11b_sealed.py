"""R11 v2 (goal G8b): the fresh sealed incident-request set and its dev set.

``data/r11b_sealed_test.jsonl`` holds 142 hand-written requests to the incident desk (120 with a
gold GoalSpec, 8 that must be answered with a question, 14 that must be refused). It was written
after the G8 kill and before any v2 prompt, few-shot or desk work, from the catalogue of tickets
and times and the documented desk policy (never from model outputs). It shares no text with the
v1 sealed or dev sets, the v2 dev set or the R7a sets (frame n-gram check in ``tests/test_r11b.py``).

Gold convention (``spec`` v2): the gold ``at`` is the transcription of what the user said, never
completed from the catalogue. "On the 17th at 10:20" is ``---17T10:20``, "Jan 19 at 18:47" is
``--01-19T18:47``, a full date stays full, and "yesterday" is worked out from the desk clock
(``2024-02-21``). Each spec item has exactly one natural reading and a ``resolves_to`` ticket:
the one incident the desk policy (see :mod:`..envs.logtriage.catalogue`) must resolve the gold to.
Ambiguous or unsatisfiable phrasings are ``clarify`` / ``refuse`` items only.

Its SHA-256 (over the LF-normalised bytes) is pinned here and asserted by a test and by every
script that loads it. **Never tune on it**: prompts and few-shots use ``data/r11b_dev.jsonl``,
the v1 dev failures and the R7a dev set.
"""

import json
from pathlib import Path
from typing import Optional

from .r11_sealed import DOMAIN, IncidentItem, _load, file_sha256

DATA = Path(__file__).parent / "data"
SEALED_PATH = DATA / "r11b_sealed_test.jsonl"
DEV_PATH = DATA / "r11b_dev.jsonl"
SEALED_SHA256 = "78e48e2755bd6718683d64d61c262edc1c3c0c99dc7d06dce40fc82a7af8639c"

__all__ = [
    "DOMAIN",
    "IncidentItem",
    "SEALED_SHA256",
    "file_sha256",
    "load_dev",
    "load_resolves_to",
    "load_sealed",
]


def load_sealed(verify: bool = True, path: Path = SEALED_PATH) -> list[IncidentItem]:
    """The v2 sealed items; raises if the file does not match the pinned hash."""
    if verify and file_sha256(path) != SEALED_SHA256:
        raise ValueError(
            f"{path.name} does not match the pinned SHA-256 {SEALED_SHA256}; "
            "the sealed test set must not change"
        )
    return _load(path, "sealed")


def load_dev(path: Path = DEV_PATH) -> list[IncidentItem]:
    """The v2 dev set (prompt and few-shot work; never the verdict)."""
    return _load(path, "dev")


def load_resolves_to(path: Path = SEALED_PATH) -> dict[str, Optional[str]]:
    """``item id -> ticket`` the gold must resolve to (None for clarify / refuse items)."""
    out: dict[str, Optional[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row.get("resolves_to")
    return out
