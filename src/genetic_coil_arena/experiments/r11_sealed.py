"""R11: the sealed incident-request set and the dev set.

``data/r11_sealed_test.jsonl`` holds 130 hand-written requests to the incident desk (110 with
a gold GoalSpec, 8 that must be answered with a question, 12 that must be refused). It was
written before any prompt, few-shot or model work, from the catalogue of tickets and times
(never from model outputs); the gold labels were typed with the texts. Its SHA-256 (over the
LF-normalised bytes) is pinned here and asserted by a test and by every script that loads it.
**Never tune on it**: prompts and few-shots use ``data/r11_dev.jsonl`` and the R7a dev set.

A ``clarify`` or ``refuse`` item may be answered by the translator (vague or out-of-domain
request) or by the catalogue (a date that names several incidents, an unknown ticket); the
desk's final decision is what counts.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DATA = Path(__file__).parent / "data"
SEALED_PATH = DATA / "r11_sealed_test.jsonl"
DEV_PATH = DATA / "r11_dev.jsonl"
SEALED_SHA256 = "8aeaeb2b899d86eff2453537d1af6ce7f17949c8711fb9d1f46bbdc6ba3e8fad"
DOMAIN = "incident"


@dataclass(frozen=True)
class IncidentItem:
    """One request with its gold label.

    Attributes:
        id: Stable id
        domain: Always ``incident`` (the R7a items keep their own ``squad``/``plant``)
        split: ``sealed`` or ``dev``
        style: Phrasing style (plain, terse, chatty, formal, typo, words, question, keyvalue)
        category: What the request exercises (ticket, time, relative, topk, exclude, fault,
            combo, clarify, refuse)
        text: The request
        gold: Gold params for a ``spec`` item, else None
        kind: ``spec``, ``clarify`` or ``refuse``
    """

    id: str
    domain: str
    split: str
    style: str
    category: str
    text: str
    gold: Optional[dict[str, Any]]
    kind: str

    @property
    def expected(self) -> str:
        return self.kind


def file_sha256(path: Path = SEALED_PATH) -> str:
    """SHA-256 of the file with CRLF turned into LF (so git's autocrlf cannot change it)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _load(path: Path, split: str) -> list[IncidentItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        items.append(
            IncidentItem(
                row["id"],
                row["domain"],
                split,
                row["style"],
                row["category"],
                row["text"],
                row["gold"],
                row["kind"],
            )
        )
    return items


def load_sealed(verify: bool = True, path: Path = SEALED_PATH) -> list[IncidentItem]:
    """The sealed items; raises if the file does not match the pinned hash."""
    if verify and file_sha256(path) != SEALED_SHA256:
        raise ValueError(
            f"{path.name} does not match the pinned SHA-256 {SEALED_SHA256}; "
            "the sealed test set must not change"
        )
    return _load(path, "sealed")


def load_dev(path: Path = DEV_PATH) -> list[IncidentItem]:
    """The dev set (prompt and few-shot work; never the verdict)."""
    return _load(path, "dev")
