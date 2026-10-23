"""Routers: pick which specialist gets a goal.

:class:`Router` is the seam for Elo-based and learned routers. The default
:class:`RuleBasedRouter` tries explicit rules first and otherwise picks the
best-scoring specialist that accepts the goal.
"""

import fnmatch
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

from ..protocol.goalspec import GoalSpec
from .registry import SpecialistRegistry


@runtime_checkable
class Router(Protocol):
    """Chooses a specialist for a goal."""

    def route(self, goal: GoalSpec, registry: SpecialistRegistry) -> Optional[str]:
        """Name of the specialist to assign, or None if nobody should get it."""
        ...


@dataclass(frozen=True)
class RoutingRule:
    """Send goals whose spec name matches ``spec`` (fnmatch) to ``specialist``.

    Attributes:
        spec: Spec name pattern, e.g. ``"arith.*"``
        specialist: Target specialist name
        when: Optional extra predicate on the goal
    """

    spec: str
    specialist: str
    when: Optional[Callable[[GoalSpec], bool]] = None

    def matches(self, goal: GoalSpec) -> bool:
        """Whether this rule applies to ``goal``."""
        if not fnmatch.fnmatchcase(goal.spec, self.spec):
            return False
        return self.when is None or bool(self.when(goal))


class RuleBasedRouter:
    """Rules first, then the highest-scoring capable specialist.

    A rule only fires if its target is registered and accepts the goal; otherwise the
    next rule is tried. The fallback sorts capable specialists by
    ``scores[score_key]`` (missing scores count as ``default_score``), then by
    registration order.

    Args:
        rules: Rules, tried in order
        score_key: Score metric used by the fallback
        default_score: Score assumed for specialists without one
    """

    def __init__(
        self,
        rules: Sequence[RoutingRule] = (),
        score_key: str = "score",
        default_score: float = 0.0,
    ):
        self.rules = list(rules)
        self.score_key = score_key
        self.default_score = default_score

    def add_rule(self, rule: RoutingRule) -> None:
        """Append a rule (lowest precedence)."""
        self.rules.append(rule)

    def route(self, goal: GoalSpec, registry: SpecialistRegistry) -> Optional[str]:
        """See :class:`Router`."""
        for rule in self.rules:
            if not rule.matches(goal):
                continue
            record = registry.get(rule.specialist)
            if record is not None and record.capabilities.accepts_goal(goal):
                return record.name
        candidates = registry.find(goal)
        if not candidates:
            return None
        best = min(
            candidates,
            key=lambda r: (-r.scores.get(self.score_key, self.default_score), r.order),
        )
        return best.name
