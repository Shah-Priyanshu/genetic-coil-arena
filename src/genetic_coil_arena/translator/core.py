"""Natural language to a validated GoalSpec, and back.

The forward translator asks a structured-output backend (``OllamaBackend`` or anything
with the same ``generate`` signature) for a JSON object whose schema is taken from the
``SpecRegistry``, then validates the result through the registry. The model never
builds a GoalSpec: ``SpecRegistry.make`` does, or the translation fails with a typed
status. One retry is made with the validation errors; after that the failure is final.

The reply schema wraps the spec's own params schema::

    {"refusal": "<empty, or why the request cannot be turned into a goal>",
     "params": <the params schema of the spec definition>}

so a request that cannot be satisfied can be refused cleanly instead of being forced
into a valid-looking goal.

The reverse translator turns a trace (the GoalSpec, envelope headers and an outcome
summary) into a short human-readable summary, and reconstructs a GoalSpec from that
summary by running it through the forward translator: the round trip.

Prompts and few-shot examples are data (:class:`TranslationPack`, :class:`SummaryPack`),
supplied per domain by the caller. This package must not import ``envs``, ``snake_core``,
``evo`` or ``analysis``.
"""

import copy
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from ..protocol.envelope import Envelope
from ..protocol.goalspec import GoalSpec, SpecRegistry, SpecValidationError

OK = "ok"
REFUSED = "refused"  # the model declined: the request is out of scope or unsatisfiable
INVALID = "invalid"  # the reply never validated (after the retry)
ERROR = "error"  # the backend failed
STATUSES = (OK, REFUSED, INVALID, ERROR)


class StructuredBackend(Protocol):
    """What the translator needs from a backend (``OllamaBackend`` fits)."""

    def generate(
        self, prompts: Sequence[str], adapter_id: Optional[str] = None, **kw: Any
    ) -> list[Any]:
        """One completion per prompt; results have ``text``, ``latency_s``, ``n_prompt_tokens``."""
        ...


# ---------------------------------------------------------------------- data


@dataclass(frozen=True)
class Example:
    """One few-shot pair: an instruction and the answer it should get.

    Attributes:
        text: The instruction
        params: Params of the expected goal (``None`` for a refusal)
        refusal: Why the request is refused (empty when ``params`` is given)
    """

    text: str
    params: Optional[Mapping[str, Any]] = None
    refusal: str = ""

    def reply(self) -> str:
        """The JSON reply the model should produce."""
        body: dict[str, Any] = {"refusal": self.refusal}
        if self.params is not None:
            body["params"] = dict(self.params)
        return json.dumps(body, sort_keys=False)


@dataclass(frozen=True)
class TranslationPack:
    """Per-domain data for the forward translator.

    Attributes:
        spec_name: The spec definition the instructions target
        system: System prompt (the domain's conventions and defaults)
        examples: Few-shot examples
        version: Spec version (latest if None)
    """

    spec_name: str
    system: str
    examples: tuple[Example, ...] = ()
    version: Optional[int] = None


@dataclass(frozen=True)
class SummaryPack:
    """Per-domain data for the reverse translator's summary.

    Attributes:
        system: System prompt
        examples: Few-shot ``(rendered trace view, summary)`` pairs
        max_new_tokens: Summary length cap
    """

    system: str
    examples: tuple[tuple[str, str], ...] = ()
    max_new_tokens: int = 160


@dataclass(frozen=True)
class TranslationResult:
    """Outcome of one forward translation.

    Attributes:
        text: The instruction
        status: ``ok``, ``refused``, ``invalid`` or ``error``
        spec: The validated GoalSpec when ``status`` is ``ok``
        reason: The model's refusal, or the last problem found
        errors: Validation errors of the last attempt
        attempts: Model calls made (1 or 2)
        latency_s: Total model time
        call_latencies: Time of each call
        n_prompt_tokens: Prompt tokens of the first call (0 if unknown)
        raw: Last raw reply
    """

    text: str
    status: str
    spec: Optional[GoalSpec] = None
    reason: str = ""
    errors: tuple[str, ...] = ()
    attempts: int = 0
    latency_s: float = 0.0
    call_latencies: tuple[float, ...] = ()
    n_prompt_tokens: int = 0
    raw: str = ""

    @property
    def ok(self) -> bool:
        """Whether a validated GoalSpec came out."""
        return self.status == OK

    @property
    def params(self) -> Optional[dict[str, Any]]:
        """Params of the spec, or None."""
        return None if self.spec is None else copy.deepcopy(self.spec.params)


# ---------------------------------------------------------------------- forward


def reply_schema(
    registry: SpecRegistry, spec_name: str, version: Optional[int] = None
) -> dict[str, Any]:
    """The structured-output schema for a spec: refusal text plus the params schema."""
    definition = registry.get(spec_name, version)
    return {
        "type": "object",
        "properties": {
            "refusal": {
                "type": "string",
                "description": "Empty when the request can be turned into a goal; "
                "otherwise a short reason why it cannot.",
            },
            "params": copy.deepcopy(definition.params_schema),
        },
        "required": ["refusal"],
        "additionalProperties": False,
    }


class Translator:
    """Natural language to a validated GoalSpec.

    Args:
        backend: Structured-output backend
        registry: The spec registry every result is validated against
        retries: Extra attempts after a failed validation (default 1)
        keep_alive: Passed to the backend's ``generate`` when not None
    """

    def __init__(
        self,
        backend: StructuredBackend,
        registry: SpecRegistry,
        retries: int = 1,
        keep_alive: Any = None,
    ):
        self.backend = backend
        self.registry = registry
        self.retries = retries
        self.keep_alive = keep_alive

    def translate(
        self,
        text: str,
        pack: TranslationPack,
        priority: int = 0,
        deadline: Optional[int] = None,
    ) -> TranslationResult:
        """Translate one instruction.

        On a reply that is not JSON or does not validate, retry once with the problem
        stated; after that return ``status="invalid"``. A non-empty ``refusal`` ends
        the translation at once with ``status="refused"``.
        """
        schema = reply_schema(self.registry, pack.spec_name, pack.version)
        shots = [(e.text, e.reply()) for e in pack.examples]
        prompt = text
        latencies: list[float] = []
        n_prompt = 0
        raw = ""
        errors: list[str] = []
        for attempt in range(1, self.retries + 2):
            kwargs: dict[str, Any] = {
                "system": pack.system,
                "examples": shots,
                "schema": schema,
            }
            if self.keep_alive is not None:
                kwargs["keep_alive"] = self.keep_alive
            try:
                (result,) = self.backend.generate([prompt], **kwargs)
            except (
                Exception
            ) as exc:  # noqa: BLE001 - a backend failure is a typed result
                return TranslationResult(
                    text,
                    ERROR,
                    reason=f"{type(exc).__name__}: {exc}",
                    attempts=attempt,
                    latency_s=sum(latencies),
                    call_latencies=tuple(latencies),
                    n_prompt_tokens=n_prompt,
                    raw=raw,
                )
            latencies.append(float(getattr(result, "latency_s", 0.0)))
            if attempt == 1:
                n_prompt = int(getattr(result, "n_prompt_tokens", 0) or 0)
            raw = result.text
            outcome = self._check(raw, pack, priority, deadline)
            if isinstance(outcome, GoalSpec):
                return TranslationResult(
                    text,
                    OK,
                    spec=outcome,
                    attempts=attempt,
                    latency_s=sum(latencies),
                    call_latencies=tuple(latencies),
                    n_prompt_tokens=n_prompt,
                    raw=raw,
                )
            kind, errors = outcome
            if kind == REFUSED:
                return TranslationResult(
                    text,
                    REFUSED,
                    reason=errors[0],
                    attempts=attempt,
                    latency_s=sum(latencies),
                    call_latencies=tuple(latencies),
                    n_prompt_tokens=n_prompt,
                    raw=raw,
                )
            # Retry: show the model its own reply and what was wrong with it
            shots = shots + [(prompt, raw)]
            prompt = (
                "Your previous reply was rejected: "
                + "; ".join(errors)
                + f"\nAnswer the original instruction again, with a corrected JSON reply.\nInstruction: {text}"
            )
        return TranslationResult(
            text,
            INVALID,
            reason=errors[0] if errors else "invalid reply",
            errors=tuple(errors),
            attempts=self.retries + 1,
            latency_s=sum(latencies),
            call_latencies=tuple(latencies),
            n_prompt_tokens=n_prompt,
            raw=raw,
        )

    def _check(
        self, raw: str, pack: TranslationPack, priority: int, deadline: Optional[int]
    ) -> Any:
        """A validated GoalSpec, or ``(kind, errors)`` where kind is REFUSED or INVALID."""
        try:
            body = json.loads(raw)
        except ValueError:
            return (INVALID, [f"the reply is not JSON: {raw[:80]!r}"])
        if not isinstance(body, dict):
            return (INVALID, ["the reply must be a JSON object"])
        refusal = body.get("refusal", "")
        if isinstance(refusal, str) and refusal.strip():
            return (REFUSED, [refusal.strip()])
        params = body.get("params")
        if not isinstance(params, dict):
            return (INVALID, ["'params' is missing although 'refusal' is empty"])
        try:
            return self.registry.make(
                pack.spec_name,
                params,
                version=pack.version,
                priority=priority,
                deadline=deadline,
            )
        except SpecValidationError as exc:
            return (INVALID, list(exc.errors) or [str(exc)])


# ---------------------------------------------------------------------- reverse


@dataclass(frozen=True)
class TraceView:
    """What the reverse translator sees of one task.

    Attributes:
        goal: The GoalSpec that was executed
        headers: Envelope headers (no payloads) in trace order
        outcome: Outcome summary: the task's state and result, as plain JSON
    """

    goal: GoalSpec
    headers: tuple[dict[str, Any], ...]
    outcome: dict[str, Any]

    def render(self) -> str:
        """The trace as compact JSON text, the reverse translator's input."""
        return json.dumps(
            {
                "goal": {"spec": self.goal.spec, "params": self.goal.params},
                "messages": list(self.headers),
                "outcome": self.outcome,
            },
            sort_keys=True,
        )


def envelope_header(envelope: Envelope) -> dict[str, Any]:
    """The readable header of an envelope: who, what kind, when (no payload)."""
    return {
        "sender": envelope.sender,
        "recipients": list(envelope.recipients),
        "payload_type": envelope.payload_type,
        "codec": envelope.payload_encoding.get("codec"),
        "clock": envelope.clock,
    }


def trace_view(
    goal: GoalSpec, envelopes: Sequence[Envelope], outcome: Mapping[str, Any]
) -> TraceView:
    """Build a :class:`TraceView` from a task's goal, its envelopes and its outcome."""
    return TraceView(
        goal=goal,
        headers=tuple(envelope_header(e) for e in envelopes),
        outcome=copy.deepcopy(dict(outcome)),
    )


@dataclass(frozen=True)
class SummaryResult:
    """A generated summary and what it cost."""

    text: str
    latency_s: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class RoundTrip:
    """Summary of a trace and the GoalSpec reconstructed from it.

    Attributes:
        summary: The human-readable summary
        reconstructed: Forward translation of the summary
        summary_latency_s: Time to write the summary
    """

    summary: SummaryResult
    reconstructed: Optional[TranslationResult]

    def matches(
        self, original: GoalSpec, schema: Optional[Mapping[str, Any]] = None
    ) -> bool:
        """Whether the reconstruction equals ``original`` (spec name and params)."""
        from .metrics import exact_match

        rec = self.reconstructed
        if rec is None or rec.spec is None or rec.spec.spec != original.spec:
            return False
        return exact_match(rec.spec.params, original.params, schema)


class ReverseTranslator:
    """Traces to summaries, and summaries back to GoalSpecs.

    Args:
        translator: The forward translator used for reconstruction (it also owns the
            backend that writes the summaries)
        keep_alive: Passed to the backend when not None
    """

    def __init__(self, translator: Translator, keep_alive: Any = None):
        self.translator = translator
        self.keep_alive = keep_alive

    def summarise(self, view: TraceView, pack: SummaryPack) -> SummaryResult:
        """A short natural-language summary of a trace, for a human monitor."""
        kwargs: dict[str, Any] = {
            "system": pack.system,
            "examples": list(pack.examples),
            "max_new_tokens": pack.max_new_tokens,
        }
        if self.keep_alive is not None:
            kwargs["keep_alive"] = self.keep_alive
        try:
            (result,) = self.translator.backend.generate([view.render()], **kwargs)
        except Exception as exc:  # noqa: BLE001
            return SummaryResult("", 0.0, f"{type(exc).__name__}: {exc}")
        return SummaryResult(
            result.text.strip(), float(getattr(result, "latency_s", 0.0))
        )

    def reconstruct(self, summary: str, pack: TranslationPack) -> TranslationResult:
        """Rebuild a GoalSpec from a summary (forward translation of the summary)."""
        return self.translator.translate(summary, pack)

    def round_trip(
        self, view: TraceView, summary_pack: SummaryPack, pack: TranslationPack
    ) -> RoundTrip:
        """Summarise a trace, then reconstruct the GoalSpec from the summary."""
        summary = self.summarise(view, summary_pack)
        if not summary.text:
            return RoundTrip(summary, None)
        return RoundTrip(summary, self.reconstruct(summary.text, pack))


def timed(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """``(fn(*args), seconds)``."""
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start
