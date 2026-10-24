"""Natural language to validated GoalSpecs, and traces back to summaries.

Domain-agnostic: prompts and few-shot examples are data supplied per domain (see
``experiments.r7a_domains``). Must not import ``snake_core``, ``evo``, ``envs`` or
``analysis``.
"""

from .core import (
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

__all__ = [
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
    "canonical_params",
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
