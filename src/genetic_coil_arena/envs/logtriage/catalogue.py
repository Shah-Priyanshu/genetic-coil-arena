"""The incident catalogue: how a person names an RCAEval incident.

The R11 desk lets a user refer to an incident the way an on-call engineer would: by ticket
(``INC-1042``) or by when it happened. The catalogue turns the converted RE2 test incidents
into that vocabulary, deterministically and without reading any label.

* **Ticket.** ``INC-`` plus a four-digit number from ``TICKET_BASE``: incidents sorted by
  (injection time, incident id) are numbered in that order, so the same data always gives the
  same tickets.
* **Time.** RCAEval's real injection timestamp (``inject_time``, Unix seconds, read as UTC) is
  the incident's alert time. The log window of the incident is ``alert + window``.
* **Resolution.** :meth:`IncidentCatalogue.resolve` maps a request's reference (``system``,
  optional ``ticket`` and ``at``) to exactly one entry, or raises a typed :class:`ResolveError`
  whose ``kind`` is ``clarify`` (a question for the requester) or ``refuse`` (cannot be
  satisfied). A date-only ``at`` (``2024-01-20``) names the system's only incident that UTC day;
  a date and time (``2024-01-20T14:30``) names the system's only incident whose alert is within
  :data:`TOLERANCE_S` seconds of it.
* **Partial dates (``rca.investigate`` v2).** The translator transcribes what the user said and the
  desk resolves it, so ``at`` may also be ``--MM-DD`` (month and day, no year) or ``---DD`` (day of
  the month only), each with an optional ``THH:MM`` (ISO-8601 truncated forms). The policy: the
  candidate dates are the calendar dates at or before the desk clock (:data:`DESK_NOW`,
  Thursday 2024-02-22 09:00 UTC) that match the partial date and have an incident of that system
  matching the time rule (within :data:`TOLERANCE_S` of the time, or on that UTC day without one;
  incidents after the desk clock do not exist yet). Exactly one candidate date with exactly one
  incident resolves; several candidate dates (the 20th is in January and in February) or several
  incidents on the one date make the desk ask (``clarify``); none refuses. A ticket given with a
  partial date must be an incident whose date matches it. Full dates keep the v1 behaviour.

Only file metadata (ids, systems, services, timestamps) is read; labels never enter the
catalogue, so a request cannot leak them.
"""

import datetime as dt
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

SYSTEM_NAMES = {"SS": "sock-shop", "OB": "online-boutique"}
SYSTEM_CODES = {v: k for k, v in SYSTEM_NAMES.items()}
# every converted-file system code the catalogue can hold (R12 adds RE3 and Train Ticket); the
# RE3 code-level-fault incidents belong to the same three systems as RE2
CODE_SYSTEM = {
    **SYSTEM_NAMES,
    "TT": "train-ticket",
    "RE3-SS": "sock-shop",
    "RE3-OB": "online-boutique",
    "RE3-TT": "train-ticket",
}
KNOWN_SYSTEMS = frozenset(CODE_SYSTEM.values())
TICKET_PREFIX = "INC-"
TICKET_BASE = 1001
TICKET_PATTERN = r"^INC-\d{4}$"
AT_PATTERN = r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2})?$"
AT_PATTERN_V2 = r"^(\d{4}-\d{2}-\d{2}|--\d{2}-\d{2}|---\d{2})(T\d{2}:\d{2})?$"
TOLERANCE_S = 600.0  # a time names the incident whose alert is within this many seconds

_AT = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2}))?$")
_AT2 = re.compile(
    r"^(?:(\d{4})-(\d{2})-(\d{2})|--(\d{2})-(\d{2})|---(\d{2}))(?:T(\d{2}):(\d{2}))?$"
)
UTC = dt.timezone.utc
DESK_NOW = dt.datetime(2024, 2, 22, 9, 0, tzinfo=UTC)  # Thursday; the desk clock of R11
# R12 (G10): RCAEval RE3 was recorded in November and December 2024, after the R11 clock, so the
# R12 desk reads the clock on Wednesday 2024-12-11, the day after the last RE3 test incident.
DESK_NOW_R12 = dt.datetime(2024, 12, 11, 9, 0, tzinfo=UTC)
R12_TICKET_BASE = 2001


class ResolveError(ValueError):
    """A reference that names no single incident.

    Attributes:
        kind: ``clarify`` (ask the requester) or ``refuse`` (cannot be satisfied)
        message: The question, or the reason
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


def utc_text(seconds: float, with_date: bool = True) -> str:
    """``2024-01-20T14:31`` (or ``14:31``) for a Unix time, in UTC."""
    stamp = dt.datetime.fromtimestamp(seconds, UTC)
    return stamp.strftime("%Y-%m-%dT%H:%M" if with_date else "%H:%M")


def parse_at(text: str) -> tuple[Optional[float], str]:
    """``(Unix time or None for a date-only reference, UTC date)``; raises ``ValueError``."""
    m = _AT.match(text)
    if not m:
        raise ValueError(f"{text!r} is not YYYY-MM-DD or YYYY-MM-DDTHH:MM")
    y, mo, d, hh, mm = m.groups()
    date = f"{y}-{mo}-{d}"
    day = dt.datetime(
        int(y), int(mo), int(d), tzinfo=UTC
    )  # raises on an impossible date
    if hh is None:
        return None, date
    return (
        day.replace(hour=int(hh), minute=int(mm)).timestamp(),
        date,
    )


@dataclass(frozen=True)
class TimeRef:
    """A parsed ``at`` of ``rca.investigate`` v2.

    Attributes:
        year: Year, or None when the user gave none
        month: Month, or None when the user gave only a day of the month
        day: Day of the month
        hhmm: ``(hour, minute)`` or None for a date-only reference
    """

    year: Optional[int]
    month: Optional[int]
    day: int
    hhmm: Optional[tuple[int, int]] = None

    @property
    def partial(self) -> bool:
        return self.year is None


def parse_ref(text: str) -> TimeRef:
    """Parse ``YYYY-MM-DD``, ``--MM-DD`` or ``---DD`` with an optional ``THH:MM``.

    Raises:
        ValueError: Not one of those forms, or an impossible month, day or time
    """
    m = _AT2.match(text)
    if not m:
        raise ValueError(
            f"{text!r} is not YYYY-MM-DD, --MM-DD or ---DD (each with an optional THH:MM)"
        )
    y, mo, d, mo2, d2, d3, hh, mm = m.groups()
    month = int(mo or mo2) if (mo or mo2) else None
    day = int(d or d2 or d3)
    hhmm = None if hh is None else (int(hh), int(mm))
    if (month is not None and not 1 <= month <= 12) or not 1 <= day <= 31:
        raise ValueError(f"{text!r} is not a possible date")
    if hhmm is not None and not (hhmm[0] < 24 and hhmm[1] < 60):
        raise ValueError(f"{text!r} is not a possible time")
    if y is not None:
        dt.datetime(int(y), month or 1, day, tzinfo=UTC)  # raises on an impossible date
    return TimeRef(None if y is None else int(y), month, day, hhmm)


@dataclass(frozen=True)
class CatalogueEntry:
    """One catalogued incident.

    Attributes:
        ticket: ``INC-1001`` style id
        incident_id: The RCAEval incident id, e.g. ``SS/carts_cpu/2``
        code: System code (``SS`` or ``OB``; ``RE3-SS``, ``RE3-OB``, ``RE3-TT`` in R12)
        inject_time: Alert time, Unix seconds
        services: The system's services (candidates of the root cause)
        window: Log window, seconds relative to the alert
    """

    ticket: str
    incident_id: str
    code: str
    inject_time: float
    services: tuple[str, ...]
    window: tuple[float, float] = (0.0, 0.0)

    @property
    def system(self) -> str:
        return CODE_SYSTEM[self.code]

    @property
    def alert_utc(self) -> str:
        return utc_text(self.inject_time)

    @property
    def date_utc(self) -> str:
        return utc_text(self.inject_time)[:10]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket": self.ticket,
            "incident": self.incident_id,
            "system": self.system,
            "alert_utc": self.alert_utc,
            "window_utc": [
                utc_text(self.inject_time + self.window[0]),
                utc_text(self.inject_time + self.window[1]),
            ],
            "n_services": len(self.services),
        }


def read_meta(path: Path) -> dict[str, Any]:
    """The JSON metadata of a converted incident file (the line arrays stay on disk)."""
    with np.load(path) as data:
        return dict(json.loads(bytes(data["meta"]).decode("utf-8")))


class IncidentCatalogue:
    """Tickets and wall-clock times of a fixed set of incidents.

    Args:
        entries: Catalogue entries (their tickets must be unique)
        files: Optional ``ticket -> converted incident file`` (to load an incident later)
        now: The desk clock (partial dates resolve at or before it; default :data:`DESK_NOW`)
    """

    def __init__(
        self,
        entries: Iterable[CatalogueEntry],
        files: Optional[Mapping[str, Path]] = None,
        now: dt.datetime = DESK_NOW,
    ):
        self.now = now
        self.entries = sorted(entries, key=lambda e: e.ticket)
        self.by_ticket = {e.ticket: e for e in self.entries}
        if len(self.by_ticket) != len(self.entries):
            raise ValueError("duplicate tickets")
        self.by_incident = {e.incident_id: e for e in self.entries}
        self.files = dict(files or {})

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def from_files(
        cls,
        paths: Sequence[Path],
        ticket_base: int = TICKET_BASE,
        now: dt.datetime = DESK_NOW,
    ) -> "IncidentCatalogue":
        """Number the incidents of ``paths`` by (alert time, id) and keep their file paths."""
        rows = []
        for path in paths:
            meta = read_meta(Path(path))
            code = meta["system"]
            if code not in CODE_SYSTEM:
                raise ValueError(f"{meta['id']}: system {code!r} is not catalogued")
            rows.append((float(meta["meta"]["inject_time"]), meta["id"], meta, path))
        rows.sort(key=lambda r: (r[0], r[1]))
        entries, files = [], {}
        for k, (when, iid, meta, path) in enumerate(rows):
            ticket = f"{TICKET_PREFIX}{ticket_base + k:04d}"
            entries.append(
                CatalogueEntry(
                    ticket,
                    iid,
                    meta["system"],
                    when,
                    tuple(meta["services"]),
                    (float(meta["window"][0]), float(meta["window"][1])),
                )
            )
            files[ticket] = Path(path)
        return cls(entries, files, now)

    @classmethod
    def merged(
        cls, parts: Sequence["IncidentCatalogue"], now: dt.datetime
    ) -> "IncidentCatalogue":
        """One catalogue over several (tickets must not collide), read at the clock ``now``."""
        entries = [e for part in parts for e in part.entries]
        files = {t: p for part in parts for t, p in part.files.items()}
        return cls(entries, files, now)

    def fingerprint(self) -> str:
        """SHA-256 over ticket, incident id and alert time of every entry."""
        text = "\n".join(
            f"{e.ticket}|{e.incident_id}|{e.inject_time:.0f}" for e in self.entries
        )
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def of_system(self, system: str) -> list[CatalogueEntry]:
        return [e for e in self.entries if e.system == system]

    def services_of(self, system: str) -> list[str]:
        """The service names of a system (every incident of a system shares them)."""
        found = self.of_system(system)
        return list(found[0].services) if found else []

    def resolve(self, params: Mapping[str, Any]) -> CatalogueEntry:
        """The one incident a request's reference names (see the module docstring).

        Raises:
            ResolveError: ``clarify`` when the reference is missing or matches several
                incidents; ``refuse`` for an unknown ticket, a ticket of another system, a
                ticket and a time that disagree, a time with no incident, or an excluded
                service that the system does not have.
        """
        system = params.get("system")
        if system not in KNOWN_SYSTEMS:
            raise ResolveError(
                "clarify", "Which system is this about, sock-shop or online-boutique?"
            )
        ticket, at = params.get("ticket"), params.get("at")
        entry: Optional[CatalogueEntry] = None
        if ticket is not None:
            entry = self.by_ticket.get(ticket)
            if entry is None:
                raise ResolveError("refuse", f"there is no ticket {ticket}")
            if entry.system != system:
                raise ResolveError(
                    "refuse", f"{ticket} is a {entry.system} incident, not {system}"
                )
        if at is not None and str(at).startswith(
            "--"
        ):  # v2: month and day, or day only
            try:
                ref = parse_ref(str(at))
            except ValueError as exc:
                raise ResolveError("refuse", str(exc)) from exc
            if entry is not None:
                if not self._matches(ref, dt.date.fromisoformat(entry.date_utc)):
                    raise ResolveError(
                        "refuse", f"{ticket} happened on {entry.date_utc}, not {at}"
                    )
            else:
                entry = self._by_partial(system, ref, str(at))
        elif at is not None:
            try:
                when, date = parse_at(str(at))
            except ValueError as exc:
                raise ResolveError("refuse", str(exc)) from exc
            if entry is not None:
                if entry.date_utc != date:
                    raise ResolveError(
                        "refuse", f"{ticket} happened on {entry.date_utc}, not {date}"
                    )
            else:
                entry = self._by_time(system, when, date)
        if entry is None:
            raise ResolveError(
                "clarify",
                "Which incident? Give a ticket (INC-nnnn) or a date and time.",
            )
        unknown = [
            s for s in params.get("exclude_services") or [] if s not in entry.services
        ]
        if unknown:
            raise ResolveError(
                "refuse", f"{system} has no service named {', '.join(unknown)}"
            )
        return entry

    def _by_time(self, system: str, when: Optional[float], date: str) -> CatalogueEntry:
        pool = self.of_system(system)
        if when is None:
            hits = [e for e in pool if e.date_utc == date]
            label = f"on {date}"
        else:
            hits = [e for e in pool if abs(e.inject_time - when) <= TOLERANCE_S]
            label = f"within {int(TOLERANCE_S // 60)} minutes of {utc_text(when)}"
        if not hits:
            raise ResolveError("refuse", f"no {system} incident {label}")
        if len(hits) > 1:
            names = ", ".join(f"{e.ticket} ({e.alert_utc})" for e in hits[:4])
            more = "" if len(hits) <= 4 else f" and {len(hits) - 4} more"
            raise ResolveError(
                "clarify",
                f"{len(hits)} {system} incidents {label}: {names}{more}. Which one?",
            )
        return hits[0]

    @staticmethod
    def _matches(ref: TimeRef, day: dt.date) -> bool:
        """Whether a calendar day is one the (partial) reference can mean."""
        return (
            day.day == ref.day
            and (ref.month is None or day.month == ref.month)
            and (ref.year is None or day.year == ref.year)
        )

    def _by_partial(self, system: str, ref: TimeRef, text: str) -> CatalogueEntry:
        """Resolve a partial date by the policy in the module docstring."""
        pool = [
            e for e in self.of_system(system) if e.inject_time <= self.now.timestamp()
        ]
        if pool:
            first = min(
                dt.datetime.fromtimestamp(e.inject_time, UTC).date() for e in pool
            )
            last = self.now.date()
            days = [
                first - dt.timedelta(days=1) + dt.timedelta(days=k)
                for k in range((last - first).days + 2)
            ]
        else:
            days = []
        found: dict[dt.date, list[CatalogueEntry]] = {}
        for day in days:
            if day > self.now.date() or not self._matches(ref, day):
                continue
            if ref.hhmm is None:
                hits = [e for e in pool if e.date_utc == day.isoformat()]
            else:
                when = dt.datetime.combine(day, dt.time(*ref.hhmm), UTC).timestamp()
                hits = [e for e in pool if abs(e.inject_time - when) <= TOLERANCE_S]
            if hits:
                found[day] = hits
        label = f"matching {text} at or before {utc_text(self.now.timestamp())}"
        if not found:
            raise ResolveError("refuse", f"no {system} incident {label}")
        if len(found) > 1:
            dates = ", ".join(d.isoformat() for d in sorted(found))
            raise ResolveError(
                "clarify",
                f"{text} matches {system} incidents on {len(found)} dates ({dates}). "
                "Which date do you mean?",
            )
        hits = next(iter(found.values()))
        if len(hits) > 1:
            names = ", ".join(f"{e.ticket} ({e.alert_utc})" for e in hits[:4])
            more = "" if len(hits) <= 4 else f" and {len(hits) - 4} more"
            raise ResolveError(
                "clarify",
                f"{len(hits)} {system} incidents {label}: {names}{more}. Which one?",
            )
        return hits[0]
