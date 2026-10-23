"""Registry of specialists: capabilities, version, evaluation scores and runner."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Optional, Union

from ..protocol.goalspec import GoalSpec
from .specialist import Capabilities, ClockAdapter, Specialist


@dataclass
class SpecialistRecord:
    """A registered specialist.

    Attributes:
        name: Unique name (bus address)
        version: Implementation version from its capabilities
        capabilities: Its agent card
        adapter: The ClockAdapter that drives it
        scores: Evaluation scores by metric (e.g. ``elo``, ``success_rate``); routers
            read these
        order: Registration order, for deterministic tie-breaking
    """

    name: str
    version: str
    capabilities: Capabilities
    adapter: ClockAdapter
    scores: dict[str, float] = field(default_factory=dict)
    order: int = 0

    @property
    def specialist(self) -> Specialist:
        """The wrapped specialist."""
        return self.adapter.specialist


class SpecialistRegistry:
    """Specialists by name, with capability lookup."""

    def __init__(self) -> None:
        self._records: dict[str, SpecialistRecord] = {}
        self._counter = 0

    def register(
        self,
        specialist: Union[Specialist, ClockAdapter],
        scores: Optional[Mapping[str, float]] = None,
        replace: bool = False,
    ) -> SpecialistRecord:
        """Add a specialist (wrapped in a plain ClockAdapter unless one is given).

        Args:
            specialist: A Specialist, or a ClockAdapter configured with keys
            scores: Initial evaluation scores
            replace: Allow replacing a specialist of the same name (e.g. a new version)

        Raises:
            ValueError: The name is taken and ``replace`` is False
        """
        adapter = (
            specialist
            if isinstance(specialist, ClockAdapter)
            else ClockAdapter(specialist)
        )
        capabilities = adapter.specialist.capabilities()
        if capabilities.name in self._records and not replace:
            raise ValueError(
                f"A specialist named {capabilities.name!r} is already registered"
            )
        record = SpecialistRecord(
            name=capabilities.name,
            version=capabilities.version,
            capabilities=capabilities,
            adapter=adapter,
            scores={k: float(v) for k, v in (scores or {}).items()},
            order=self._counter,
        )
        self._counter += 1
        self._records[record.name] = record
        return record

    def unregister(self, name: str) -> SpecialistRecord:
        """Remove and return a specialist (KeyError if unknown)."""
        try:
            return self._records.pop(name)
        except KeyError:
            raise KeyError(f"No specialist named {name!r}") from None

    def lookup(self, name: str) -> SpecialistRecord:
        """Record by name (KeyError if unknown)."""
        try:
            return self._records[name]
        except KeyError:
            raise KeyError(f"No specialist named {name!r}") from None

    def get(self, name: str) -> Optional[SpecialistRecord]:
        """Record by name, or None."""
        return self._records.get(name)

    def find(
        self, goal: GoalSpec, encoding: Optional[str] = None
    ) -> list[SpecialistRecord]:
        """Specialists whose capabilities accept ``goal`` (and read ``encoding``).

        Returned in registration order.
        """
        return [
            record
            for record in self.records()
            if record.capabilities.accepts_goal(goal)
            and (encoding is None or encoding in record.capabilities.payload_encodings)
        ]

    def set_score(self, name: str, metric: str, value: float) -> None:
        """Record an evaluation score for a specialist."""
        self.lookup(name).scores[metric] = float(value)

    def records(self) -> list[SpecialistRecord]:
        """All records in registration order."""
        return sorted(self._records.values(), key=lambda r: r.order)

    def names(self) -> list[str]:
        """Registered names, sorted."""
        return sorted(self._records)

    def agent_cards(self) -> list[dict[str, object]]:
        """A2A-style agent cards of all specialists, with version and scores."""
        cards = []
        for record in self.records():
            card: dict[str, object] = dict(record.capabilities.to_agent_card())
            card["scores"] = dict(record.scores)
            cards.append(card)
        return cards

    def __contains__(self, name: object) -> bool:
        return name in self._records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[SpecialistRecord]:
        return iter(self.records())
