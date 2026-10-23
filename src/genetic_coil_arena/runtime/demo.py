"""Tiny domain-agnostic demo: arithmetic specialists coordinated by the centre.

Three specialists:

- ``adder`` answers ``arith.add`` and ``arith.sum``;
- ``multiplier`` answers ``arith.mul``;
- ``planner`` answers ``arith.dot`` (dot product) by delegating: it asks the centre
  for one ``arith.mul`` per pair, then one ``arith.sum`` of the products.

Every envelope is HMAC-signed by its sender and verified by its receiver, every
message lands in a trace, and the demo finally shows that a tampered result is
caught. Run with ``python -m genetic_coil_arena.runtime.demo [trace.jsonl]``.
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..protocol.envelope import Envelope
from ..protocol.goalspec import GoalSpec, SpecDefinition, SpecRegistry
from ..protocol.messages import (
    COMPLETED,
    complete,
    fail,
    goal_envelope,
    goal_from_envelope,
    is_control,
    is_goal,
    is_result,
)
from .command_centre import CommandCentre
from .specialist import Capabilities, ClockAdapter, FunctionSpecialist
from .trace import TraceLog

CENTRE = "command-centre"

_NUMBER = {"type": "number"}
_VECTOR = {"type": "array", "items": _NUMBER, "minItems": 1}


def demo_specs(backend: str = "auto") -> SpecRegistry:
    """Spec registry with the four arithmetic GoalSpecs."""
    specs = SpecRegistry(backend=backend)
    pair = {
        "type": "object",
        "required": ["a", "b"],
        "properties": {"a": _NUMBER, "b": _NUMBER},
        "additionalProperties": False,
    }
    specs.register(SpecDefinition("arith.add", 1, pair, "a + b", _NUMBER))
    specs.register(SpecDefinition("arith.mul", 1, pair, "a * b", _NUMBER))
    specs.register(
        SpecDefinition(
            "arith.sum",
            1,
            {
                "type": "object",
                "required": ["values"],
                "properties": {"values": {"type": "array", "items": _NUMBER}},
                "additionalProperties": False,
            },
            "Sum of a list of numbers",
            _NUMBER,
        )
    )
    specs.register(
        SpecDefinition(
            "arith.dot",
            1,
            {
                "type": "object",
                "required": ["x", "y"],
                "properties": {"x": _VECTOR, "y": _VECTOR},
                "additionalProperties": False,
            },
            "Dot product of two equal-length vectors",
            _NUMBER,
        )
    )
    return specs


def adder() -> FunctionSpecialist:
    """Specialist for ``arith.add`` and ``arith.sum``."""
    return FunctionSpecialist(
        "adder",
        {
            "arith.add": lambda p: p["a"] + p["b"],
            "arith.sum": lambda p: sum(p["values"]),
        },
        description="Adds numbers",
    )


def multiplier() -> FunctionSpecialist:
    """Specialist for ``arith.mul``."""
    return FunctionSpecialist(
        "multiplier", {"arith.mul": lambda p: p["a"] * p["b"]}, description="Multiplies"
    )


@dataclass
class _Job:
    request: Envelope
    pending: dict[str, int] = field(default_factory=dict)  # sub-request msg_id -> index
    products: list[Optional[float]] = field(default_factory=list)
    sum_request: Optional[str] = None


class DotPlanner:
    """Computes ``arith.dot`` by delegating multiplications and a sum to the centre.

    Shows a stateful, purely message-driven specialist: it never calls another
    specialist directly and never blocks.
    """

    def __init__(self, name: str = "planner", specs: Optional[SpecRegistry] = None):
        self.name = name
        self.specs = specs or demo_specs()
        self._jobs: dict[str, _Job] = {}  # goal msg_id -> job
        self._owner: dict[str, str] = {}  # sub-request msg_id -> goal msg_id

    def capabilities(self) -> Capabilities:
        """Accepts ``arith.dot`` version 1."""
        return Capabilities(
            name=self.name,
            version="1",
            accepts=("arith.dot@1",),
            description="Dot products via delegation",
            tags=("planner",),
        )

    def handle(self, envelope: Envelope) -> list[Envelope]:
        """Start a job on a goal, advance it on a sub-result, drop it on cancel."""
        if is_goal(envelope):
            return self._start(envelope)
        if is_result(envelope):
            return self._advance(envelope)
        if is_control(envelope, "cancel"):
            for key, job in list(self._jobs.items()):
                if job.request.task_id == envelope.task_id:
                    self._drop(key)
        return []

    def _ask(self, job_key: str, job: _Job, goal: GoalSpec) -> Envelope:
        request = goal_envelope(
            goal,
            sender=self.name,
            recipients=CENTRE,
            task_id=job.request.task_id,
            trace_id=job.request.trace_id,
            parent_id=job.request.msg_id,
            clock=job.request.clock,
        )
        self._owner[request.msg_id] = job_key
        return request

    def _start(self, envelope: Envelope) -> list[Envelope]:
        params = goal_from_envelope(envelope).params
        x, y = params["x"], params["y"]
        if len(x) != len(y):
            return [fail(envelope, self.name, f"length mismatch: {len(x)} vs {len(y)}")]
        job = _Job(request=envelope, products=[None] * len(x))
        self._jobs[envelope.msg_id] = job
        out = []
        for i, (a, b) in enumerate(zip(x, y)):
            sub = self._ask(
                envelope.msg_id, job, self.specs.make("arith.mul", {"a": a, "b": b})
            )
            job.pending[sub.msg_id] = i
            out.append(sub)
        return out

    def _advance(self, envelope: Envelope) -> list[Envelope]:
        parent_id = envelope.parent_id
        job_key = self._owner.pop(parent_id, None) if parent_id else None
        job = self._jobs.get(job_key) if job_key else None
        if parent_id is None or job_key is None or job is None:
            return []
        if envelope.payload["state"] != COMPLETED:
            self._drop(job_key)
            return [
                fail(
                    job.request,
                    self.name,
                    f"sub-goal failed: {envelope.payload.get('error')}",
                )
            ]
        if parent_id == job.sum_request:
            self._drop(job_key)
            return [complete(job.request, self.name, envelope.payload["result"])]
        job.products[job.pending.pop(parent_id)] = envelope.payload["result"]
        if job.pending:
            return []
        total = self.specs.make("arith.sum", {"values": job.products})
        sub = self._ask(job_key, job, total)
        job.sum_request = sub.msg_id
        return [sub]

    def _drop(self, job_key: Optional[str]) -> None:
        self._jobs.pop(job_key or "", None)
        for msg_id in [m for m, k in self._owner.items() if k == job_key]:
            del self._owner[msg_id]


class TamperingAdapter(ClockAdapter):
    """A compromised sidecar: signs results, then rewrites the payload in transit."""

    def _outbound(self, envelopes: Optional[Sequence[Envelope]]) -> list[Envelope]:
        out = []
        for envelope in super()._outbound(envelopes):
            if is_result(envelope) and envelope.payload.get("state") == COMPLETED:
                forged = dict(envelope.payload, result=-1)
                envelope = Envelope.from_dict({**envelope.to_dict(), "payload": forged})
            out.append(envelope)
        return out


# Demo-only secrets; real deployments load keys from a secret store
DEMO_KEYS = {
    CENTRE: b"centre-key",
    "adder": b"adder-key",
    "multiplier": b"mul-key",
    "planner": b"planner-key",
}


def build_centre(
    trace: Optional[TraceLog] = None, backend: str = "auto"
) -> CommandCentre:
    """Centre plus the three demo specialists, everyone signing and verifying."""
    keys = DEMO_KEYS
    specs = demo_specs(backend)
    centre = CommandCentre(
        specs, trace=trace, name=CENTRE, signing_key=keys[CENTRE], verify_keys=keys
    )
    for specialist in (adder(), multiplier(), DotPlanner(specs=specs)):
        name = specialist.capabilities().name
        centre.register(
            ClockAdapter(
                specialist, signing_key=keys[name], verify_keys={CENTRE: keys[CENTRE]}
            )
        )
    return centre


def run_demo(trace_path: Optional[Path] = None, verbose: bool = True) -> dict[str, Any]:
    """Run the demo; returns the task states, results, and trace statistics."""
    say = print if verbose else (lambda *args, **kwargs: None)
    trace = TraceLog(trace_path)
    centre = build_centre(trace)
    specs = centre.specs

    dot = centre.submit(
        specs.make("arith.dot", {"x": [1, 2, 3], "y": [4, 5, 6]}),
        situation={"note": "user asked for the dot product of (1,2,3) and (4,5,6)"},
    )
    add = centre.submit(specs.make("arith.add", {"a": 2, "b": 40}, priority=1))
    steps = centre.run_until_idle()
    say(f"arith.dot -> {centre.status(dot).state}: {centre.status(dot).result}")
    say(f"arith.add -> {centre.status(add).state}: {centre.status(add).result}")
    say(
        f"{steps} ticks, {len(centre.status(dot).subtasks)} sub-goals, {len(trace)} trace records"
    )

    # A compromised channel rewrites the adder's signed answer: the centre notices
    centre.register(
        TamperingAdapter(
            adder(),
            signing_key=DEMO_KEYS["adder"],
            verify_keys={CENTRE: DEMO_KEYS[CENTRE]},
        ),
        replace=True,
    )
    tampered = centre.submit(specs.make("arith.add", {"a": 1, "b": 1}))
    centre.run_until_idle()
    status = centre.status(tampered)
    say(f"tampered arith.add -> {status.state}: {status.error}")

    triples = trace.triples()
    trace.close()
    return {
        "dot": centre.status(dot),
        "add": centre.status(add),
        "tampered": status,
        "steps": steps,
        "trace_records": len(trace),
        "triples": len(triples),
        "signature_rejections": len(trace.events("signature_rejected")),
    }


def main(argv: Sequence[str] = ()) -> int:
    """Entry point: optional first argument is a JSONL trace path."""
    path = Path(argv[0]) if argv else None
    summary = run_demo(path)
    ok = (
        summary["dot"].result == 32
        and summary["add"].result == 42
        and summary["tampered"].state == "failed"
    )
    if path is not None:
        print(f"trace written to {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
