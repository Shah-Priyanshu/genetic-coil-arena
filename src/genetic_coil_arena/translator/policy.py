"""The refusal policy: parse first, then validate.

:class:`PolicyTranslator` is the step-2 forward translator. It differs from
:class:`~genetic_coil_arena.translator.core.Translator` in what a model may decide:

1. The model *extracts*: it fills a relaxed params object (types only; no ranges or
   enums, so a value outside the schema is still parsed instead of being refused on
   sight). It may also say that the request is out of domain, or contradicts itself
   (``refusal``), or is in domain but ambiguous (``clarify``, a question).
2. Code then *decides*: params the request did not mention take their documented
   defaults (``TranslationPack.defaults``); a mandatory param with no default that is
   missing becomes a clarifying question; the registry validates the rest. A value the
   schema forbids (``0.2%`` false alarms, a 1000-step mission) is sent back once with
   the validation errors, and if it is still invalid the request is unsatisfiable.

Outcomes: ``ok`` (a validated GoalSpec), ``refused`` (out of domain, contradictory, or
unsatisfiable under the schema), ``clarify`` (a question instead of a spec), ``invalid``
(the reply never became valid) and ``error`` (the backend failed). A clarification is
neither an exact match nor a refusal; reports count it separately.

The reply is a JSON object ``{"refusal": "", "clarify": "", "params": {...}}`` in which
empty strings and ``params`` may be omitted. This package must not import ``envs``,
``snake_core``, ``evo`` or ``analysis``.
"""

import copy
import json
from collections.abc import Mapping
from typing import Any, Optional

from ..protocol.goalspec import GoalSpec, SpecRegistry, SpecValidationError
from .core import (
    CLARIFY,
    ERROR,
    INVALID,
    OK,
    REFUSED,
    StructuredBackend,
    TranslationPack,
    TranslationResult,
)

_UNSATISFIABLE = (
    "unsatisfiable"  # internal: the params parse but the schema forbids them
)
_KEEP_KEYS = ("type", "items", "properties", "additionalProperties", "description")


def relax_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """A schema with only the structure left: types, properties and items.

    Enums, ranges, ``required``, ``minItems``/``maxItems`` and ``uniqueItems`` are
    dropped, so a model can state any value the request contains and code decides
    afterwards whether it is allowed.
    """
    out: dict[str, Any] = {}
    for key in _KEEP_KEYS:
        if key not in schema:
            continue
        value = schema[key]
        if key == "properties":
            out[key] = {k: relax_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, Mapping):
            out[key] = relax_schema(value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def policy_reply_schema(
    registry: SpecRegistry, spec_name: str, version: Optional[int] = None
) -> dict[str, Any]:
    """Structured-output schema of the policy reply: refusal, clarify, relaxed params."""
    definition = registry.get(spec_name, version)
    return {
        "type": "object",
        "properties": {
            "refusal": {
                "type": "string",
                "description": "Empty unless the request is out of domain or contradicts "
                "itself; then a short reason.",
            },
            "clarify": {
                "type": "string",
                "description": "Empty unless a stated quantity is vague or a needed "
                "choice is missing; then a short question for the requester.",
            },
            "params": relax_schema(definition.params_schema),
        },
        "required": ["refusal", "clarify"],
        "additionalProperties": False,
    }


def retry_prompt(text: str, errors: list[str]) -> str:
    """The second-attempt prompt: what was wrong, then the original instruction.

    Training data for a distilled model uses this same function so the format matches.
    """
    return (
        "Your previous reply was rejected: "
        + "; ".join(errors)
        + "\nAnswer the original instruction again with a corrected JSON reply. "
        "If the request really asks for a value the schema does not allow, "
        'reply with a "refusal" that says so.' + f"\nInstruction: {text}"
    )


def fill_defaults(
    params: Mapping[str, Any], defaults: Mapping[str, Any]
) -> dict[str, Any]:
    """Params with ``None`` entries dropped and absent ones set from ``defaults``."""
    out = {k: copy.deepcopy(v) for k, v in params.items() if v is not None}
    for key, value in defaults.items():
        if key not in out:
            out[key] = copy.deepcopy(value)
    return out


def coerce_integers(params: Any, schema: Mapping[str, Any]) -> Any:
    """Whole floats become ints where the schema says integer (``120.0`` -> ``120``)."""
    kind = schema.get("type")
    if kind == "integer" and isinstance(params, float) and params.is_integer():
        return int(params)
    if isinstance(params, dict):
        props = schema.get("properties", {})
        return {k: coerce_integers(v, props.get(k, {})) for k, v in params.items()}
    if isinstance(params, list):
        return [coerce_integers(v, schema.get("items", {})) for v in params]
    return params


class PolicyTranslator:
    """Natural language to a validated GoalSpec, with the parse-then-validate policy.

    Same call shape as :class:`~genetic_coil_arena.translator.core.Translator`
    (``translate(text, pack)``, a ``backend`` attribute), so it can sit under a
    :class:`~genetic_coil_arena.translator.core.ReverseTranslator`.

    Args:
        backend: Structured-output backend (``generate`` as in ``StructuredBackend``)
        registry: The registry every result is validated against
        retries: Extra attempts after a malformed or invalid reply (default 1)
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
        """Translate one instruction (see the module docstring for the policy)."""
        schema = policy_reply_schema(self.registry, pack.spec_name, pack.version)
        shots = [(e.text, e.reply(policy=True)) for e in pack.examples]
        prompt = text
        latencies: list[float] = []
        n_prompt = 0
        raw = ""
        errors: list[str] = []

        def done(status: str, **kw: Any) -> TranslationResult:
            return TranslationResult(
                text,
                status,
                attempts=len(latencies),
                latency_s=sum(latencies),
                call_latencies=tuple(latencies),
                n_prompt_tokens=n_prompt,
                raw=raw,
                **kw,
            )

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
                if not latencies:
                    latencies.append(0.0)
                return done(ERROR, reason=f"{type(exc).__name__}: {exc}")
            latencies.append(float(getattr(result, "latency_s", 0.0)))
            if attempt == 1:
                n_prompt = int(getattr(result, "n_prompt_tokens", 0) or 0)
            raw = result.text
            outcome = self._decide(raw, pack, priority, deadline)
            if isinstance(outcome, GoalSpec):
                return done(OK, spec=outcome)
            kind, errors = outcome
            if kind in (REFUSED, CLARIFY):
                return done(kind, reason=errors[0])
            if kind == _UNSATISFIABLE and attempt == self.retries + 1:
                return done(
                    REFUSED,
                    reason="unsatisfiable under the schema: " + "; ".join(errors),
                    errors=tuple(errors),
                )
            shots = shots + [(prompt, raw)]
            prompt = retry_prompt(text, errors)
        return done(
            INVALID,
            reason=errors[0] if errors else "invalid reply",
            errors=tuple(errors),
        )

    def _decide(
        self, raw: str, pack: TranslationPack, priority: int, deadline: Optional[int]
    ) -> Any:
        """A GoalSpec, or ``(kind, messages)`` with kind REFUSED, CLARIFY, INVALID or unsatisfiable."""
        try:
            body = json.loads(raw)
        except ValueError:
            return (INVALID, [f"the reply is not JSON: {raw[:80]!r}"])
        if not isinstance(body, dict):
            return (INVALID, ["the reply must be a JSON object"])
        refusal = body.get("refusal", "")
        if isinstance(refusal, str) and refusal.strip():
            return (REFUSED, [refusal.strip()])
        question = body.get("clarify", "")
        if isinstance(question, str) and question.strip():
            return (CLARIFY, [question.strip()])
        params = body.get("params")
        if not isinstance(params, dict):
            return (
                INVALID,
                ["'params' is missing although 'refusal' and 'clarify' are empty"],
            )
        definition = self.registry.get(pack.spec_name, pack.version)
        schema = definition.params_schema
        filled = coerce_integers(fill_defaults(params, pack.defaults), schema)
        missing = [k for k in schema.get("required", ()) if k not in filled]
        if missing:
            asks = [
                pack.questions.get(k, f"Please specify {k.replace('_', ' ')}.")
                for k in missing
            ]
            return (CLARIFY, [" ".join(asks)])
        try:
            return self.registry.make(
                pack.spec_name,
                filled,
                version=pack.version,
                priority=priority,
                deadline=deadline,
            )
        except SpecValidationError as exc:
            return (_UNSATISFIABLE, list(exc.errors) or [str(exc)])
