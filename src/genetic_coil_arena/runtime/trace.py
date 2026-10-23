"""Append-only trace of every envelope, with optional situation snapshots.

A trace is a sequence of records, each one of

- ``envelope``: an envelope that was sent (its full dict form, manifest included);
- ``situation``: a JSON snapshot of the world when a request was made, keyed by
  ``trace_id``;
- ``event``: a runtime note (a rejected signature, a task state change, ...).

Saved as JSONL: a header line ``{"format": "gca-trace", "version": 1}`` and then one
record per line. With a ``path``, records are appended to the file as they happen,
so a crash loses nothing. :meth:`TraceLog.replay` re-emits envelopes in order, and
:meth:`TraceLog.triples` yields (situation, spec, messages) per trace: the training
data for the reverse translator.
"""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    Optional,
    Union,
)

from ..protocol.envelope import Envelope, canonical_json
from ..protocol.goalspec import GoalSpec
from ..protocol.messages import goal_from_envelope, is_goal

TRACE_FORMAT = "gca-trace"
TRACE_VERSION = 1
RECORD_KINDS = ("envelope", "situation", "event")

PathLike = Union[str, Path]


@dataclass(frozen=True)
class TraceRecord:
    """One line of a trace.

    Attributes:
        seq: Position in the trace, from 0
        kind: ``envelope``, ``situation`` or ``event``
        trace_id: Trace the record belongs to (None for global events)
        clock: Logical time of the record
        data: Envelope dict, situation snapshot, or event details
    """

    seq: int
    kind: str
    trace_id: Optional[str]
    clock: int
    data: Any

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict."""
        return {
            "seq": self.seq,
            "kind": self.kind,
            "trace_id": self.trace_id,
            "clock": self.clock,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceRecord":
        """Inverse of :meth:`to_dict`."""
        if data.get("kind") not in RECORD_KINDS:
            raise ValueError(f"Unknown trace record kind {data.get('kind')!r}")
        return cls(
            seq=int(data["seq"]),
            kind=data["kind"],
            trace_id=data.get("trace_id"),
            clock=int(data.get("clock", 0)),
            data=data["data"],
        )

    def envelope(self) -> Envelope:
        """The envelope of an ``envelope`` record."""
        if self.kind != "envelope":
            raise ValueError(f"Record {self.seq} is a {self.kind}, not an envelope")
        return Envelope.from_dict(self.data)


@dataclass(frozen=True)
class TrainingTriple:
    """(situation, spec, messages) for one trace: reverse-translator training data.

    Attributes:
        trace_id: The trace
        situation: Last situation snapshot of the trace (None if none was recorded)
        spec: The first GoalSpec sent in the trace (None if there was none)
        messages: Every envelope of the trace, in order
    """

    trace_id: str
    situation: Any
    spec: Optional[GoalSpec]
    messages: tuple[Envelope, ...]


class TraceLog:
    """Append-only, optionally file-backed log of envelopes and situations.

    Args:
        path: If given, every record is appended to this JSONL file immediately
            (a new file gets a header; an existing trace file is continued)
    """

    def __init__(self, path: Optional[PathLike] = None):
        self._records: list[TraceRecord] = []
        self.path = Path(path) if path is not None else None
        self._file: Optional[IO[str]] = None
        if self.path is not None:
            if self.path.exists() and self.path.stat().st_size > 0:
                self._records = list(_read_records(self.path))
                self._file = open(self.path, "a", encoding="utf-8")
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = open(self.path, "w", encoding="utf-8")
                self._write_line(_header())

    # -- appending

    def record(self, envelope: Envelope) -> TraceRecord:
        """Append an envelope (usable directly as a bus tap)."""
        return self._append(
            "envelope", envelope.trace_id, envelope.clock, envelope.to_dict()
        )

    def snapshot(self, trace_id: str, situation: Any, clock: int = 0) -> TraceRecord:
        """Append a JSON situation snapshot for ``trace_id``."""
        return self._append("situation", trace_id, clock, situation)

    def event(
        self,
        name: str,
        trace_id: Optional[str] = None,
        clock: int = 0,
        **details: Any,
    ) -> TraceRecord:
        """Append a runtime event, e.g. ``event("signature_rejected", reason=...)``."""
        return self._append("event", trace_id, clock, {"event": name, **details})

    def _append(
        self, kind: str, trace_id: Optional[str], clock: int, data: Any
    ) -> TraceRecord:
        canonical_json(data)  # Fail early on non-JSON data, before anything is written
        record = TraceRecord(len(self._records), kind, trace_id, int(clock), data)
        self._records.append(record)
        if self._file is not None:
            self._write_line(record.to_dict())
        return record

    def _write_line(self, data: Mapping[str, Any]) -> None:
        assert self._file is not None
        self._file.write(canonical_json(data) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the backing file (the in-memory log stays readable)."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "TraceLog":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Avoid ResourceWarning when a file-backed log is dropped without close()
        file = getattr(self, "_file", None)
        if file is not None:
            file.close()

    # -- reading

    @property
    def records(self) -> tuple[TraceRecord, ...]:
        """All records, in order."""
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[TraceRecord]:
        return iter(tuple(self._records))

    def trace_ids(self) -> list[str]:
        """Distinct trace ids, in order of first appearance."""
        seen: dict[str, None] = {}
        for record in self._records:
            if record.trace_id is not None:
                seen.setdefault(record.trace_id, None)
        return list(seen)

    def envelopes(self, trace_id: Optional[str] = None) -> list[Envelope]:
        """Logged envelopes, optionally of one trace."""
        return [
            record.envelope()
            for record in self._records
            if record.kind == "envelope"
            and (trace_id is None or record.trace_id == trace_id)
        ]

    def situations(self, trace_id: str) -> list[Any]:
        """All situation snapshots of ``trace_id``, in order."""
        return [
            r.data
            for r in self._records
            if r.kind == "situation" and r.trace_id == trace_id
        ]

    def situation(self, trace_id: str) -> Any:
        """Latest situation snapshot of ``trace_id`` (None if none)."""
        snapshots = self.situations(trace_id)
        return snapshots[-1] if snapshots else None

    def events(self, name: Optional[str] = None) -> list[TraceRecord]:
        """Event records, optionally only those called ``name``."""
        return [
            r
            for r in self._records
            if r.kind == "event" and (name is None or r.data.get("event") == name)
        ]

    def replay(
        self,
        sink: Optional[Callable[[Envelope], Any]] = None,
        trace_id: Optional[str] = None,
    ) -> list[Envelope]:
        """Rebuild logged envelopes in order, feeding each to ``sink`` if given.

        Replaying into a bus's ``publish`` re-runs the message flow; replaying into a
        specialist's ``handle`` re-tests it against recorded inputs.
        """
        envelopes = self.envelopes(trace_id)
        if sink is not None:
            for envelope in envelopes:
                sink(envelope)
        return envelopes

    def triples(self) -> list[TrainingTriple]:
        """One (situation, spec, messages) triple per trace id."""
        triples = []
        for trace_id in self.trace_ids():
            messages = tuple(self.envelopes(trace_id))
            spec = next((goal_from_envelope(e) for e in messages if is_goal(e)), None)
            triples.append(
                TrainingTriple(trace_id, self.situation(trace_id), spec, messages)
            )
        return triples

    # -- persistence

    def save(self, path: PathLike) -> Path:
        """Write the whole log as a JSONL trace file; returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(_header()) + "\n")
            for record in self._records:
                handle.write(canonical_json(record.to_dict()) + "\n")
        return path

    @classmethod
    def load(cls, path: PathLike) -> "TraceLog":
        """Read a JSONL trace file into an in-memory log (not file-backed)."""
        log = cls()
        log._records = list(_read_records(Path(path)))
        return log


def _header() -> dict[str, Any]:
    return {"format": TRACE_FORMAT, "version": TRACE_VERSION}


def _read_records(path: Path) -> Iterator[TraceRecord]:
    with open(path, encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty, not a trace file")
    header = json.loads(lines[0])
    if header.get("format") != TRACE_FORMAT:
        raise ValueError(f"{path} is not a {TRACE_FORMAT} file")
    if header.get("version") != TRACE_VERSION:
        raise ValueError(f"Unsupported trace version {header.get('version')!r}")
    for number, line in enumerate(lines[1:], start=2):
        try:
            yield TraceRecord.from_dict(json.loads(line))
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{path}:{number}: bad trace record ({exc})") from exc
