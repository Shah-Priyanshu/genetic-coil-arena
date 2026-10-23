"""In-process message bus.

Envelopes are published into a FIFO queue and handed to subscribers in rounds:
:meth:`InProcessBus.deliver` delivers exactly the envelopes that were queued when it
was called, so anything published while delivering waits for the next round. That
gives every message a one-round latency and makes runs deterministic regardless of
subscriber order.

The :class:`MessageBus` protocol is the seam for a networked bus later.
"""

from collections import deque
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from ..protocol.envelope import Envelope

Handler = Callable[[Envelope], None]
Tap = Callable[[Envelope], object]


@dataclass(frozen=True)
class DeadLetter:
    """An envelope that could not be delivered to one recipient."""

    envelope: Envelope
    recipient: str
    reason: str


@runtime_checkable
class MessageBus(Protocol):
    """Minimal bus interface used by the command centre."""

    def subscribe(self, name: str, handler: Handler) -> None:
        """Deliver envelopes addressed to ``name`` to ``handler``."""
        ...

    def unsubscribe(self, name: str) -> None:
        """Stop delivering to ``name``."""
        ...

    def publish(self, envelope: Envelope) -> None:
        """Queue an envelope for delivery."""
        ...

    def deliver(self) -> int:
        """Deliver one round; return the number of (envelope, recipient) deliveries."""
        ...

    @property
    def pending(self) -> int:
        """Envelopes waiting for delivery."""
        ...


class InProcessBus:
    """Single-process bus with taps (observers) and a dead-letter list."""

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}
        self._queue: deque[Envelope] = deque()
        self._taps: list[Tap] = []
        self.dead_letters: list[DeadLetter] = []
        self.rounds = 0

    def subscribe(self, name: str, handler: Handler) -> None:
        """Register ``handler`` for ``name`` (one handler per name)."""
        if name in self._handlers:
            raise ValueError(f"{name!r} is already subscribed")
        self._handlers[name] = handler

    def unsubscribe(self, name: str) -> None:
        """Remove ``name``'s handler (no error if absent)."""
        self._handlers.pop(name, None)

    def is_subscribed(self, name: str) -> bool:
        """Whether ``name`` has a handler."""
        return name in self._handlers

    def add_tap(self, tap: Tap) -> None:
        """Call ``tap(envelope)`` on every publish (e.g. ``TraceLog.record``)."""
        self._taps.append(tap)

    def publish(self, envelope: Envelope) -> None:
        """Queue an envelope and show it to every tap."""
        if not isinstance(envelope, Envelope):
            raise TypeError(f"Can only publish Envelope, got {type(envelope).__name__}")
        for tap in self._taps:
            tap(envelope)
        self._queue.append(envelope)

    @property
    def pending(self) -> int:
        """Envelopes waiting for the next round."""
        return len(self._queue)

    def deliver(self) -> int:
        """Deliver the envelopes queued before this call.

        Unknown recipients produce a :class:`DeadLetter` instead of an error.

        Returns:
            Number of (envelope, recipient) deliveries made
        """
        batch = list(self._queue)
        self._queue.clear()
        delivered = 0
        for envelope in batch:
            for recipient in envelope.recipients:
                handler = self._handlers.get(recipient)
                if handler is None:
                    self.dead_letters.append(
                        DeadLetter(envelope, recipient, "no subscriber")
                    )
                    continue
                handler(envelope)
                delivered += 1
        self.rounds += 1
        return delivered
