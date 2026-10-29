"""R16 (goal G14): the sealed operator-request set of the OpenSwarm ground station.

``data/r16_sealed_test.jsonl`` holds 150 operator requests: 90 that should become a
command (every behaviour, 10 to 13 each), 30 that should be asked about and 30 that should
be refused. A separate agent (Sonnet) wrote it, and the 60-item dev set
``data/r16_dev.jsonl``, from ``data/r16_label_guide.md`` alone. It never saw the
translator's prompt, its few-shot examples or any model output. The prompt and the
translator's checks were revised once, on the dev set only (2026-10-29, after the first
dev runs), before any model saw the sealed set; the sealed texts were never read while
developing. Every gold command passes the ``swarm.behavior`` v1 schema (checked by a
test).

Its SHA-256 (over the LF-normalised bytes) is pinned here and asserted by a test and by
``load_sealed``. **Never tune on it**; use the dev set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .r11_sealed import file_sha256
from .r16_ground_station import load_items

DATA = Path(__file__).parent / "data"
SEALED_PATH = DATA / "r16_sealed_test.jsonl"
DEV_PATH = DATA / "r16_dev.jsonl"
SEALED_SHA256 = "db87f382be7456d7eca22edf163e9a10b37c489e6fbbb75a57b20a64afaa69f8"


def load_sealed() -> list[dict[str, Any]]:
    """The sealed items, after checking the pinned hash."""
    got = file_sha256(SEALED_PATH)
    if got != SEALED_SHA256:
        raise RuntimeError(f"r16 sealed set changed: {got} != {SEALED_SHA256}")
    return load_items(SEALED_PATH)


def load_dev() -> list[dict[str, Any]]:
    return load_items(DEV_PATH)
