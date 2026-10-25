"""Natural language to validated GoalSpecs, and traces back to summaries.

Domain-agnostic: prompts and few-shot examples are data supplied per domain (see
``experiments.r7a_domains``). ``Translator`` is the step-1 forward translator;
``PolicyTranslator`` (parse first, validate after, with a ``clarify`` outcome) is the
step-2 one, and ``distill`` holds what a distilled model and its leak checks share. Must not import ``snake_core``, ``evo``, ``envs`` or
``analysis``.
"""

from .core import (
    CLARIFY,
    ERROR,
    INVALID,
    OK,
    REFUSED,
    STATUSES,
    Example,
    ReverseTranslator,
    RoundTrip,
    SummaryPack,
    SummaryResult,
    TraceView,
    TranslationPack,
    TranslationResult,
    Translator,
    envelope_header,
    reply_schema,
    trace_view,
)
from .distill import (
    OverlapIndex,
    containment,
    encode_reply,
    format_prompt,
    frame_ngrams,
    frame_tokens,
    max_containment,
)
from .metrics import (
    Fact,
    canonical_params,
    exact_match,
    fact_fidelity,
    field_matches,
    percentile,
    refusal_scores,
    wilson,
)
from .policy import (
    PolicyTranslator,
    coerce_integers,
    fill_defaults,
    policy_reply_schema,
    relax_schema,
    retry_prompt,
)

__all__ = [
    "CLARIFY",
    "ERROR",
    "INVALID",
    "OK",
    "REFUSED",
    "STATUSES",
    "Example",
    "Fact",
    "ReverseTranslator",
    "RoundTrip",
    "SummaryPack",
    "SummaryResult",
    "TraceView",
    "TranslationPack",
    "TranslationResult",
    "Translator",
    "OverlapIndex",
    "PolicyTranslator",
    "canonical_params",
    "coerce_integers",
    "containment",
    "encode_reply",
    "fill_defaults",
    "format_prompt",
    "frame_ngrams",
    "frame_tokens",
    "max_containment",
    "policy_reply_schema",
    "relax_schema",
    "retry_prompt",
    "envelope_header",
    "exact_match",
    "fact_fidelity",
    "field_matches",
    "percentile",
    "refusal_scores",
    "reply_schema",
    "trace_view",
    "wilson",
]
