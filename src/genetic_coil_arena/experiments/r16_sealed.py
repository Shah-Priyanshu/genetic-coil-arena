"""R16 (goal G14): the sealed operator-request set of the OpenSwarm ground station.

``data/r16_sealed_test.jsonl`` holds 150 operator requests: 90 that should become a
command (every behaviour, 10 to 13 each), 30 that should be asked about and 30 that should
be refused. A separate agent (Sonnet) wrote it, and the 60-item dev set
``data/r16_dev.jsonl``, from ``data/r16_label_guide.md`` alone. It never saw the
translator's prompt, its few-shot examples or any model output. The prompt, the schema's
property order and the translator's checks were revised three times, on the dev set only
(2026-10-29: v2 after the first dev runs, v3 after the second, v4 after the third),
before any model saw the sealed set; the sealed texts were never read while developing.
Every gold command passes the ``swarm.behavior`` v1 schema (checked by a test).

Its SHA-256 (over the LF-normalised bytes) is pinned here and asserted by a test and by
``load_sealed``. **Never tune on it**; use the dev set.

**G14b** (``data/r16b_sealed_test.jsonl``, split ``sealed-b``): after the first sealed
run (G14, v4) missed the wrong-command bar, the station was revised to v5 using what
those misses showed. That set is spent, so a fresh one of 150 requests (90 / 30 / 30)
was written by a separate agent (Sonnet) from ``data/r16b_label_guide.md`` alone (the
v1 guide plus the v5 conventions: fractions in words, "at N m" is an altitude, two
behaviours in sequence are refused). Ten of its texts happened to match dev or old
sealed texts exactly; the agent rewrote those ten before the set was pinned. Pinned
below, never read while developing.
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
SEALED_B_PATH = DATA / "r16b_sealed_test.jsonl"
SEALED_B_SHA256 = "196a839ff5dc93b60231a060bbe39c54c91052841c8bd63c664f5bcb518d43ed"


def load_sealed(b: bool = False) -> list[dict[str, Any]]:
    """The sealed items (G14's, or G14b's with ``b``), after checking the pinned hash."""
    path, want = (SEALED_B_PATH, SEALED_B_SHA256) if b else (SEALED_PATH, SEALED_SHA256)
    got = file_sha256(path)
    if got != want:
        raise RuntimeError(f"r16 sealed set {path.name} changed: {got} != {want}")
    return load_items(path)


def load_dev() -> list[dict[str, Any]]:
    return load_items(DEV_PATH)
