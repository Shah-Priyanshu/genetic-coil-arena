"""The R7a vertical slice, object layer: GoalSpec to command centre to trace view.

``execute`` runs one validated GoalSpec through a fresh ``CommandCentre`` (signed
envelopes, in-process bus, its own ``TraceLog``): the centre routes the goal to the
domain's specialist, which honours it through the domain executor and answers with the
result. The returned :class:`Execution` carries the task outcome and the
:class:`~genetic_coil_arena.translator.TraceView` the reverse translator reads.

``keyword_translator`` wraps a domain's regex parser as a translator, so the baseline
goes through the same ``SpecRegistry.make`` gate as the model.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..protocol.goalspec import GoalSpec, SpecRegistry, SpecValidationError
from ..runtime import ClockAdapter, CommandCentre, FunctionSpecialist, TraceLog
from ..translator import (
    INVALID,
    OK,
    REFUSED,
    TraceView,
    TranslationResult,
    trace_view,
)
from .r7a_domains import Domain, outcome_of

CENTRE = "command-centre"
# Demo-only secrets; real deployments load keys from a secret store
KEYS = {
    CENTRE: b"r7a-centre-key",
    "squad-lead": b"r7a-squad-key",
    "plant-gateway": b"r7a-plant-key",
}


@dataclass
class Execution:
    """One task run through the command centre.

    Attributes:
        state: Final task state (``completed`` when it worked)
        result: The specialist's result (None unless completed)
        error: Failure message otherwise
        view: Trace view of the task (goal, envelope headers, outcome)
        n_records: Records in the task's trace
    """

    state: str
    result: Optional[dict[str, Any]]
    error: Optional[str]
    view: Optional[TraceView]
    n_records: int = 0

    @property
    def per_seed(self) -> list[float]:
        """Per-seed task scores (empty if the task did not complete)."""
        return list(self.result["per_seed"]) if self.result else []


def execute(
    domain: Domain,
    registry: SpecRegistry,
    spec: GoalSpec,
    instruction: Optional[str] = None,
) -> Execution:
    """Submit ``spec`` to a fresh command centre with the domain's specialist."""
    trace = TraceLog()
    # a domain added later (R11's incident desk) brings its own specialist name
    KEYS.setdefault(domain.specialist, f"r7a-{domain.specialist}-key".encode())
    centre = CommandCentre(
        registry, trace=trace, name=CENTRE, signing_key=KEYS[CENTRE], verify_keys=KEYS
    )
    specialist = FunctionSpecialist(
        domain.specialist,
        {domain.spec_name: domain.executor},
        description=f"R7a {domain.name} executor",
    )
    centre.register(
        ClockAdapter(
            specialist,
            signing_key=KEYS[domain.specialist],
            verify_keys={CENTRE: KEYS[CENTRE]},
        )
    )
    situation = {"instruction": instruction} if instruction is not None else None
    task_id = centre.submit(spec, situation=situation)
    centre.run_until_idle(max_steps=50)
    status = centre.status(task_id)
    if status.state != "completed":
        return Execution(status.state, None, status.error, None, len(trace))
    envelopes = trace.envelopes(status.trace_id)
    outcome = (
        domain.outcome(status.result)
        if domain.outcome is not None
        else outcome_of(domain.name, status.result)
    )
    view = trace_view(spec, envelopes, outcome)
    return Execution(status.state, status.result, None, view, len(trace))


def keyword_translator(
    registry: SpecRegistry, domains: dict[str, Domain]
) -> Callable[[Any], TranslationResult]:
    """The regex baseline as a translator ``item -> TranslationResult``.

    The parser's params still go through ``SpecRegistry.make``; a parser that returns
    ``None`` is a refusal and params the registry rejects are ``invalid``.
    """

    def translate(item: Any) -> TranslationResult:
        domain = domains[item.domain]
        params = domain.keyword(item.text)
        if params is None:
            return TranslationResult(
                item.text, REFUSED, reason="no keywords matched", attempts=1
            )
        try:
            spec = registry.make(domain.spec_name, params)
        except SpecValidationError as exc:
            return TranslationResult(
                item.text,
                INVALID,
                reason=str(exc),
                errors=tuple(exc.errors),
                attempts=1,
            )
        return TranslationResult(item.text, OK, spec=spec, attempts=1)

    return translate
