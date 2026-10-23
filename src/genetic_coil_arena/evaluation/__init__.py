"""CommArena evaluation harness: does communication between specialists matter?

Domain-agnostic: works on numpy arrays and callables and never imports an
environment, the protocol or the runtime.

- ``runner``: tasks, mandatory baselines, multi-seed runs and message ablations
- ``interventions``: message interventions and the hook environment loops apply
- ``metrics``: communication metrics (entropy, MI, CIC, topsim, posdis, bosdis, ...)
- ``stats``: mean/t-CI, IQM with stratified bootstrap, Mann-Whitney + Holm, A12
- ``report``: the standard result object (JSON, CSV, plain-text table)
"""

from .interventions import (
    ABLATIONS,
    CONTINUOUS,
    DISCRETE,
    INTERVENTIONS,
    InterventionHook,
    MessageHook,
    MessageSpec,
    RecordingHook,
    apply_hook,
    get_intervention,
    random_matched,
    shuffle_batch,
    step_rng,
    swap_receivers,
    zero_messages,
)
from .metrics import (
    bag_of_symbols_disentanglement,
    causal_influence,
    channel_usage,
    communication_metrics,
    entropy,
    intervention_influence,
    learnability,
    listening_summary,
    message_entropy,
    mi_null_test,
    mutual_information,
    positional_disentanglement,
    quantize_continuous,
    steps_to_threshold,
    symbol_entropy,
    topographic_similarity,
)
from .report import AblationResult, ConditionResult, ExperimentResult, format_table
from .runner import (
    BASELINES,
    BaselineRegistry,
    Task,
    TaskSpec,
    evaluate,
    register_baseline,
    required_baselines,
    run_ablations,
    run_experiment,
)
from .stats import (
    compare_conditions,
    holm_correction,
    iqm,
    mann_whitney,
    mean_ci,
    paired_difference,
    stratified_bootstrap_ci,
    vargha_delaney_a12,
)

__all__ = [
    "ABLATIONS",
    "BASELINES",
    "CONTINUOUS",
    "DISCRETE",
    "INTERVENTIONS",
    "AblationResult",
    "BaselineRegistry",
    "ConditionResult",
    "ExperimentResult",
    "InterventionHook",
    "MessageHook",
    "MessageSpec",
    "RecordingHook",
    "Task",
    "TaskSpec",
    "apply_hook",
    "bag_of_symbols_disentanglement",
    "causal_influence",
    "channel_usage",
    "communication_metrics",
    "compare_conditions",
    "entropy",
    "evaluate",
    "format_table",
    "get_intervention",
    "holm_correction",
    "intervention_influence",
    "iqm",
    "learnability",
    "listening_summary",
    "mann_whitney",
    "mean_ci",
    "message_entropy",
    "mi_null_test",
    "mutual_information",
    "paired_difference",
    "positional_disentanglement",
    "quantize_continuous",
    "random_matched",
    "register_baseline",
    "required_baselines",
    "run_ablations",
    "run_experiment",
    "shuffle_batch",
    "step_rng",
    "stratified_bootstrap_ci",
    "steps_to_threshold",
    "swap_receivers",
    "symbol_entropy",
    "topographic_similarity",
    "vargha_delaney_a12",
    "zero_messages",
]
