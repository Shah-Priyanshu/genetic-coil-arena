"""Message envelope exchanged between specialists and the command centre.

An envelope pairs a fixed, readable header (who sends it, to whom, for which task,
when) with a payload whose ``payload_type`` says how to read it:

- ``json``: any JSON value, e.g. a GoalSpec, a result or an error
- ``codes``: a tuple of discrete symbols (a learned or hand-designed protocol);
  see :mod:`genetic_coil_arena.protocol.codecs`
- ``latent_ref``: a reference to a tensor held in a TensorStore. Tensors are never
  inlined, so envelopes stay small, loggable and signable.

Header semantics follow the A2A message/task model: ``task_id`` groups the messages
of one task, ``trace_id`` correlates everything caused by one request, and
``parent_id`` points at the message this one answers.
"""

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Union

import numpy as np

PROTOCOL_VERSION = 1
PAYLOAD_TYPES = ("json", "codes", "latent_ref")
LATENT_REF_KEYS = ("store", "key", "shape", "dtype", "sha256")

# Serialized field names, in a stable order (manifest is only present when signed)
ENVELOPE_FIELDS = (
    "v",
    "msg_id",
    "trace_id",
    "parent_id",
    "sender",
    "recipients",
    "task_id",
    "clock",
    "payload_type",
    "payload_encoding",
    "payload",
)


class EnvelopeError(ValueError):
    """Raised for malformed envelopes."""


def new_id() -> str:
    """Return a fresh random identifier (32 hex characters)."""
    return uuid.uuid4().hex


def canonical_json(value: Any) -> str:
    """Serialize to deterministic JSON: sorted keys, no whitespace, no NaN/Infinity.

    Used wherever bytes must be reproducible, e.g. for signatures and digests.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def as_recipients(value: Any) -> tuple[str, ...]:
    """Normalize a single name or a sequence of names to a tuple."""
    return (value,) if isinstance(value, str) else tuple(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


@dataclass(frozen=True)
class Envelope:
    """One message between parties of the ecosystem.

    Construct with keywords. ``recipients`` accepts a single name or a sequence and is
    stored as a tuple; a ``codes`` payload is stored as a tuple of ints.

    Attributes:
        sender: Name of the sending party
        recipients: Names of the receiving parties
        payload_type: One of PAYLOAD_TYPES
        payload: JSON value, tuple of symbol ids, or a latent reference dict
        payload_encoding: How to interpret the payload (codec name, version, sizes)
        task_id: Task this message belongs to, if any
        trace_id: Correlates all messages caused by one request
        parent_id: msg_id of the message this one responds to
        clock: Logical time (simulator tick or Lamport-style counter)
        msg_id: Unique message id
        v: Envelope format version
        manifest: HMAC signature manifest added by protocol.signing.sign
    """

    sender: str
    recipients: tuple[str, ...]
    payload_type: str
    payload: Any
    payload_encoding: dict[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    trace_id: str = field(default_factory=new_id)
    parent_id: Optional[str] = None
    clock: int = 0
    msg_id: str = field(default_factory=new_id)
    v: int = PROTOCOL_VERSION
    manifest: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        recipients = as_recipients(self.recipients)
        object.__setattr__(self, "recipients", recipients)

        if not _is_int(self.v) or not 1 <= int(self.v) <= PROTOCOL_VERSION:
            raise EnvelopeError(
                f"Unsupported envelope version {self.v!r}; "
                f"this code understands 1..{PROTOCOL_VERSION}"
            )
        for name in ("sender", "msg_id", "trace_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise EnvelopeError(f"{name} must be a non-empty string, got {value!r}")
        for name in ("task_id", "parent_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise EnvelopeError(
                    f"{name} must be None or a non-empty string, got {value!r}"
                )
        if not recipients or not all(isinstance(r, str) and r for r in recipients):
            raise EnvelopeError(
                f"recipients must be non-empty strings, got {recipients!r}"
            )
        if not _is_int(self.clock) or self.clock < 0:
            raise EnvelopeError(f"clock must be a non-negative int, got {self.clock!r}")
        object.__setattr__(self, "clock", int(self.clock))
        if self.payload_type not in PAYLOAD_TYPES:
            raise EnvelopeError(
                f"payload_type must be one of {PAYLOAD_TYPES}, got {self.payload_type!r}"
            )
        if not isinstance(self.payload_encoding, dict):
            raise EnvelopeError("payload_encoding must be a dict")
        if self.manifest is not None and not isinstance(self.manifest, dict):
            raise EnvelopeError("manifest must be None or a dict")
        try:
            canonical_json(self.payload_encoding)
        except (TypeError, ValueError) as exc:
            raise EnvelopeError(
                f"payload_encoding is not JSON-serializable: {exc}"
            ) from exc

        self._check_payload()

    def _check_payload(self) -> None:
        payload = self.payload
        if isinstance(payload, np.ndarray):
            raise EnvelopeError(
                "Arrays are never inlined in envelopes; use protocol.codecs "
                "(codes) or a TensorStore (latent_ref)"
            )

        if self.payload_type == "json":
            try:
                canonical_json(payload)
            except (TypeError, ValueError) as exc:
                raise EnvelopeError(
                    f"json payload is not JSON-serializable: {exc}"
                ) from exc

        elif self.payload_type == "codes":
            if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
                raise EnvelopeError("codes payload must be a sequence of ints")
            if not all(_is_int(symbol) for symbol in payload):
                raise EnvelopeError("codes payload must contain only ints")
            symbols = tuple(int(symbol) for symbol in payload)
            for size_key in ("V", "L"):
                size = self.payload_encoding.get(size_key)
                if size is not None and (not _is_int(size) or size < 0):
                    raise EnvelopeError(
                        f"payload_encoding[{size_key!r}] must be a non-negative int"
                    )
            vocab_size = self.payload_encoding.get("V")
            length = self.payload_encoding.get("L")
            if vocab_size is not None and any(not 0 <= s < vocab_size for s in symbols):
                raise EnvelopeError(
                    f"codes payload has symbols outside [0, {vocab_size})"
                )
            if length is not None and len(symbols) != length:
                raise EnvelopeError(
                    f"codes payload has {len(symbols)} symbols, encoding says L={length}"
                )
            object.__setattr__(self, "payload", symbols)

        else:  # latent_ref
            if not isinstance(payload, Mapping) or any(
                key not in payload for key in LATENT_REF_KEYS
            ):
                raise EnvelopeError(
                    f"latent_ref payload must be a mapping with keys {LATENT_REF_KEYS}"
                )
            ref = dict(payload)
            try:
                canonical_json(ref)
            except (TypeError, ValueError) as exc:
                raise EnvelopeError(
                    f"latent_ref payload is not JSON-serializable: {exc}"
                ) from exc
            object.__setattr__(self, "payload", ref)

    def __hash__(self) -> int:
        # Payloads and encodings are dicts, so hash on the (unique) message id
        return hash(self.msg_id)

    # ------------------------------------------------------------------ serialization

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict (``manifest`` is included only when present)."""
        data: dict[str, Any] = {
            "v": self.v,
            "msg_id": self.msg_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "sender": self.sender,
            "recipients": list(self.recipients),
            "task_id": self.task_id,
            "clock": self.clock,
            "payload_type": self.payload_type,
            "payload_encoding": self.payload_encoding,
            "payload": (
                list(self.payload) if self.payload_type == "codes" else self.payload
            ),
        }
        if self.manifest is not None:
            data["manifest"] = self.manifest
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Envelope":
        """Rebuild an envelope from :meth:`to_dict` output (unknown keys are rejected)."""
        missing = [name for name in ENVELOPE_FIELDS if name not in data]
        if missing:
            raise EnvelopeError(f"Envelope dict is missing fields {missing}")
        unknown = sorted(set(data) - set(ENVELOPE_FIELDS) - {"manifest"})
        if unknown:
            raise EnvelopeError(f"Envelope dict has unknown fields {unknown}")
        return cls(
            sender=data["sender"],
            recipients=data["recipients"],
            payload_type=data["payload_type"],
            payload=data["payload"],
            payload_encoding=dict(data["payload_encoding"]),
            task_id=data["task_id"],
            trace_id=data["trace_id"],
            parent_id=data["parent_id"],
            clock=data["clock"],
            msg_id=data["msg_id"],
            v=data["v"],
            manifest=(
                dict(data["manifest"]) if data.get("manifest") is not None else None
            ),
        )

    def to_json(self) -> str:
        """Canonical JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "Envelope":
        """Parse :meth:`to_json` output."""
        return cls.from_dict(json.loads(text))

    # ------------------------------------------------------------------ helpers

    @property
    def is_signed(self) -> bool:
        """Whether a signature manifest is attached (it may still be invalid)."""
        return self.manifest is not None

    def with_manifest(self, manifest: Optional[dict[str, Any]]) -> "Envelope":
        """Copy of this envelope with the given (or no) signature manifest."""
        return replace(self, manifest=manifest)

    def reply(
        self,
        sender: str,
        payload: Any = None,
        payload_type: str = "json",
        recipients: Optional[Union[str, Sequence[str]]] = None,
        payload_encoding: Optional[dict[str, Any]] = None,
        clock: Optional[int] = None,
    ) -> "Envelope":
        """New envelope answering this one.

        Keeps ``task_id`` and ``trace_id``, sets ``parent_id`` to this message, and by
        default addresses the original sender at the same clock.
        """
        return Envelope(
            sender=sender,
            recipients=as_recipients(self.sender if recipients is None else recipients),
            payload_type=payload_type,
            payload=payload,
            payload_encoding=dict(payload_encoding or {}),
            task_id=self.task_id,
            trace_id=self.trace_id,
            parent_id=self.msg_id,
            clock=self.clock if clock is None else clock,
        )
