"""R7a step 2: the sealed test set and the dev set.

The sealed set (``data/r7a_sealed_test.jsonl``) was written by hand from the specs, before
any step-2 prompt, policy or model work, in phrasings that appear neither in any prompt's
few-shots nor in the step-1 set. Gold labels were typed with the texts; no model labelled
anything. Its SHA-256 (over the LF-normalised bytes) is pinned here and asserted by a test
and by every script that loads it. **Never tune on it**: prompts, training data and
checkpoint selection use the step-1 set and the dev set only.

``kind`` of an item is ``spec`` (gold params), ``refuse`` (out of domain, contradictory or
unsatisfiable under the schema) or ``clarify`` (in domain but ambiguous: the right answer
is a question).

The dev set is the step-1 generator under another seed, with every text that is already in
the step-1 set removed. It is for model selection and prompt checks.
"""

import hashlib
import json
from pathlib import Path

from .r7a_evalset import EvalItem, build_evalset

SEALED_PATH = Path(__file__).parent / "data" / "r7a_sealed_test.jsonl"
SEALED_SHA256 = "ca6ff07b5fb66ef21c288ba9a82d5e7763992fe2142b1a582baf597e542b5e01"
DEV_SEED = 101


def sealed_sha256(path: Path = SEALED_PATH) -> str:
    """SHA-256 of the file with CRLF turned into LF (so git's autocrlf cannot change it)."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_sealed(verify: bool = True, path: Path = SEALED_PATH) -> list[EvalItem]:
    """The sealed items (split ``sealed``); raises if the file does not match the pinned hash."""
    if verify and sealed_sha256(path) != SEALED_SHA256:
        raise ValueError(
            f"{path.name} does not match the pinned SHA-256 {SEALED_SHA256}; "
            "the sealed test set must not change"
        )
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        items.append(
            EvalItem(
                row["id"],
                row["domain"],
                "sealed",
                row["style"],
                row["text"],
                row["gold"],
                row["kind"],
            )
        )
    return items


def load_old_set() -> list[EvalItem]:
    """The step-1 evaluation set (160 items; used for prompt tuning in step 1)."""
    return build_evalset(7, 36)


def load_dev() -> list[EvalItem]:
    """The dev set: a second draw of the step-1 generator with no text from the step-1 set."""
    old = {i.text for i in load_old_set()}
    out = []
    for item in build_evalset(DEV_SEED, 36):
        if item.text in old:
            continue
        out.append(
            EvalItem(
                item.id.replace("-indist-", "-dev-in-").replace("-heldout-", "-dev-ho-"),
                item.domain,
                "dev",
                item.style,
                item.text,
                item.gold,
                item.kind,
            )
        )
    return out
