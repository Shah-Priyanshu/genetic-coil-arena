"""The specialist interface and the clock adapter that drives specialists.

A specialist is message-driven: it advertises what it can do through
:meth:`Specialist.capabilities` (an A2A-style Agent Card) and reacts to each envelope
with :meth:`Specialist.handle`, returning the envelopes it wants sent.

:class:`ClockAdapter` wraps any specialist so the same object runs

- synchronously in a simulator: :meth:`ClockAdapter.tick` delivers the envelopes
  received since the previous tick (observe), then lets a clock-aware specialist act
  on its own (``on_tick``) and returns everything to send (act); and
- asynchronously in production: :meth:`ClockAdapter.serve` consumes an
  ``asyncio.Queue`` and forwards outputs as they are produced.

The adapter is also the specialist's security sidecar: it can drop inbound envelopes
whose HMAC manifest does not verify and sign every outbound envelope.
"""

import asyncio
import fnmatch
import inspect
from collections import deque
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from ..protocol.envelope import Envelope
from ..protocol.goalspec import GOALSPEC_CODEC, GoalSpec
from ..protocol.messages import (
    complete,
    fail,
    goal_from_envelope,
    is_goal,
)
from ..protocol.signing import Keys, check_signature, sign
from ..protocol.tensor_store import TensorStore

AGENT_CARD_VERSION = "1"


@dataclass(frozen=True)
class Capabilities:
    """What a specialist accepts; maps onto an A2A Agent Card.

    Attributes:
        name: Unique specialist name (also its bus address)
        version: Specialist implementation version
        accepts: Accepted GoalSpecs as ``"name"`` (any version) or ``"name@version"``;
            the name part may be an fnmatch pattern such as ``"arith.*"``
        payload_encodings: Codecs it can read (``goalspec``, ``codes``, ``vector``, ...)
        description: Human-readable summary
        tags: Free-form labels for discovery
    """

    name: str
    version: str = "0"
    accepts: tuple[str, ...] = ()
    payload_encodings: tuple[str, ...] = (GOALSPEC_CODEC,)
    description: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Capabilities.name must be a non-empty string")
        for attr in ("accepts", "payload_encodings", "tags"):
            value = getattr(self, attr)
            object.__setattr__(
                self, attr, (value,) if isinstance(value, str) else tuple(value)
            )
        for entry in self.accepts:
            _parse_accept(entry)

    def accepts_spec(self, spec: str, version: int) -> bool:
        """Whether a GoalSpec (name, version) is accepted."""
        for entry in self.accepts:
            pattern, wanted = _parse_accept(entry)
            if fnmatch.fnmatchcase(spec, pattern) and (
                wanted is None or wanted == version
            ):
                return True
        return False

    def accepts_goal(self, goal: GoalSpec) -> bool:
        """Whether ``goal`` is accepted (by name and version)."""
        return self.accepts_spec(goal.spec, goal.version)

    def to_agent_card(self) -> dict[str, Any]:
        """A2A-style Agent Card: one skill per accepted spec entry."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "protocolVersion": AGENT_CARD_VERSION,
            "defaultInputModes": list(self.payload_encodings),
            "defaultOutputModes": ["task_result"],
            "skills": [
                {"id": entry, "name": entry, "tags": list(self.tags)}
                for entry in self.accepts
            ],
        }

    @classmethod
    def from_agent_card(cls, card: Mapping[str, Any]) -> "Capabilities":
        """Inverse of :meth:`to_agent_card`."""
        skills = card.get("skills", [])
        tags = skills[0].get("tags", []) if skills else []
        return cls(
            name=card["name"],
            version=str(card.get("version", "0")),
            accepts=tuple(skill["id"] for skill in skills),
            payload_encodings=tuple(card.get("defaultInputModes", (GOALSPEC_CODEC,))),
            description=card.get("description", ""),
            tags=tuple(tags),
        )


def _parse_accept(entry: str) -> tuple[str, Optional[int]]:
    if not isinstance(entry, str) or not entry:
        raise ValueError(f"accepts entries must be non-empty strings, got {entry!r}")
    name, sep, version = entry.partition("@")
    if not sep:
        return name, None
    if not version.isdigit() or int(version) < 1:
        raise ValueError(f"Bad version in accepts entry {entry!r}")
    return name, int(version)


@runtime_checkable
class Specialist(Protocol):
    """A message-driven agent.

    ``handle`` must not block on other specialists; to delegate, return envelopes
    (for example a GoalSpec addressed to the command centre) and react to the
    answers when they arrive.
    """

    def capabilities(self) -> Capabilities:
        """Describe what this specialist accepts."""
        ...

    def handle(self, envelope: Envelope) -> list[Envelope]:
        """React to one envelope; return envelopes to send."""
        ...


@runtime_checkable
class ClockAware(Protocol):
    """Optional extension: act on every tick, not only when messages arrive."""

    def on_tick(self, clock: int) -> list[Envelope]:
        """Called once per tick after this tick's envelopes were handled."""
        ...


# ---------------------------------------------------------------------- adapter


@dataclass
class Rejection:
    """An inbound envelope the adapter refused, and why."""

    envelope: Envelope
    reason: str


class ClockAdapter:
    """Drives one specialist in a tick-based or asyncio loop.

    Args:
        specialist: The specialist to drive
        signing_key: If set, every outbound envelope is HMAC-signed with it
        key_id: Key name written into outbound manifests (default: specialist name)
        verify_keys: If set, inbound envelopes must carry a valid manifest under
            these keys, or they are dropped and recorded in :attr:`rejected`
        store: Tensor store used to re-hash ``latent_ref`` payloads while verifying
    """

    def __init__(
        self,
        specialist: Specialist,
        signing_key: Optional[bytes] = None,
        key_id: Optional[str] = None,
        verify_keys: Optional[Keys] = None,
        store: Optional[TensorStore] = None,
    ):
        if not isinstance(specialist, Specialist):
            raise TypeError("specialist must implement capabilities() and handle()")
        self.specialist = specialist
        self.name = specialist.capabilities().name
        self.signing_key = signing_key
        self.key_id = key_id or self.name
        self.verify_keys = verify_keys
        self.store = store
        self.clock = 0
        self.rejected: list[Rejection] = []
        self._inbox: deque[Envelope] = deque()

    def receive(self, envelope: Envelope) -> None:
        """Queue an envelope for the next tick (use as a bus subscription handler)."""
        self._inbox.append(envelope)

    @property
    def pending(self) -> int:
        """Envelopes waiting for the next tick."""
        return len(self._inbox)

    def tick(self, clock: Optional[int] = None) -> list[Envelope]:
        """Run one observe/act step.

        Args:
            clock: Current simulator tick (default: previous tick + 1)

        Returns:
            Envelopes to send, in production order
        """
        self.clock = self.clock + 1 if clock is None else int(clock)
        inbox, self._inbox = self._inbox, deque()
        outputs: list[Envelope] = []
        for envelope in inbox:
            outputs.extend(self.process(envelope))
        if isinstance(self.specialist, ClockAware):
            outputs.extend(self._outbound(self.specialist.on_tick(self.clock)))
        return outputs

    def step(
        self, envelopes: Sequence[Envelope], clock: Optional[int] = None
    ) -> list[Envelope]:
        """Deliver ``envelopes`` and tick once (simulator convenience)."""
        for envelope in envelopes:
            self.receive(envelope)
        return self.tick(clock)

    def process(self, envelope: Envelope) -> list[Envelope]:
        """Verify, handle and sign one envelope immediately (no queueing)."""
        if self.verify_keys is not None:
            reason = check_signature(envelope, self.verify_keys, self.store)
            if reason is not None:
                self.rejected.append(Rejection(envelope, reason))
                return []
        return self._outbound(self.specialist.handle(envelope))

    def _outbound(self, envelopes: Optional[Sequence[Envelope]]) -> list[Envelope]:
        outputs = list(envelopes or [])
        for envelope in outputs:
            if not isinstance(envelope, Envelope):
                raise TypeError(
                    f"{self.name} returned {type(envelope).__name__}, not Envelope"
                )
        if self.signing_key is not None:
            outputs = [
                sign(envelope, self.signing_key, self.key_id) for envelope in outputs
            ]
        return outputs

    async def serve(
        self,
        inbox: "asyncio.Queue[Optional[Envelope]]",
        send: Callable[[Envelope], Union[None, Awaitable[None]]],
    ) -> int:
        """Production loop: handle envelopes from ``inbox`` until ``None`` arrives.

        Args:
            inbox: Queue of inbound envelopes; put ``None`` to stop
            send: Called with each outbound envelope (may be a coroutine function)

        Returns:
            Number of inbound envelopes processed
        """
        processed = 0
        while True:
            envelope = await inbox.get()
            try:
                if envelope is None:
                    return processed
                for output in self.process(envelope):
                    sent = send(output)
                    if inspect.isawaitable(sent):
                        await sent
                processed += 1
            finally:
                inbox.task_done()


# ---------------------------------------------------------------------- helpers


GoalHandler = Callable[[dict[str, Any]], Any]


@dataclass
class FunctionSpecialist:
    """A specialist that answers goals by calling plain functions.

    Each accepted spec name maps to ``fn(params) -> result``. An exception becomes a
    ``failed`` result. Non-goal envelopes (results, control messages) are ignored.

    Attributes:
        name: Specialist name
        handlers: Spec name to function
        version: Implementation version
        description: Card description
    """

    name: str
    handlers: dict[str, GoalHandler]
    version: str = "1"
    description: str = ""
    handled: int = field(default=0, init=False)

    def capabilities(self) -> Capabilities:
        """Accept every spec name in ``handlers`` (any version)."""
        return Capabilities(
            name=self.name,
            version=self.version,
            accepts=tuple(sorted(self.handlers)),
            description=self.description,
        )

    def handle(self, envelope: Envelope) -> list[Envelope]:
        """Answer a goal envelope with a completed or failed result."""
        if not is_goal(envelope):
            return []
        self.handled += 1
        try:
            goal = goal_from_envelope(envelope)
            handler = self.handlers.get(goal.spec)
            if handler is None:
                return [
                    fail(
                        envelope,
                        self.name,
                        f"{self.name} does not handle {goal.spec!r}",
                    )
                ]
            return [complete(envelope, self.name, handler(goal.params))]
        except Exception as exc:  # noqa: BLE001 - reported to the requester
            return [fail(envelope, self.name, f"{type(exc).__name__}: {exc}")]
