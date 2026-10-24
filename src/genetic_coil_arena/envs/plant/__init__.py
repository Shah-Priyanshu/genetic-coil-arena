"""Plant-floor sensor squad: edge nodes uplink bandwidth-limited messages to a gateway.

The second lab domain, deliberately unlike Snake: four sensor nodes on a Tennessee Eastman
process, a byte-budgeted LoRaWAN-style uplink, and a gateway centre that diagnoses faults
from the messages alone. It implements the shared ``envs.base`` API (see ``env``).
"""

from .budget import (
    BUDGETS,
    MAX_BUDGET,
    VOCAB_SIZE,
    lorawan_airtime_s,
    message_spec,
    min_uplink_period_s,
)
from .centres import (
    DEFAULT_TARGET_FAR,
    DecodedCentre,
    OracleCentre,
    SilentCentre,
    block_subsample,
    blocked_folds,
    raw_features,
    stack_context,
)
from .classifiers import MLPClassifier, SoftmaxRegression, make_classifier
from .encoders import (
    ENCODERS,
    AlarmEncoder,
    FeatureEncoder,
    NoiseEncoder,
    SilentEncoder,
    SummaryEncoder,
    UplinkEncoder,
    make_encoder,
)
from .env import N_ACTIONS, PlantConfig, PlantEnv
from .replicates import Protocol, Replicate
from .rieth import (
    RiethData,
    RunSelection,
    RunSplits,
    rieth_cache_dir,
    synthetic_rieth,
    to_runs,
)
from .runner import diagnosis_metrics, macro_f1, run_episodes
from .tep import (
    FAULT_IDS,
    NODE_VARIABLES,
    NODES,
    ONSET_TEST,
    UNOBSERVABLE_FAULTS,
    VARIABLES,
    Normaliser,
    TepData,
    default_cache_dir,
    synthetic_tep,
)
from .tuning import tune_centre
from .windows import NodeLayout, make_layout

__all__ = [
    "BUDGETS",
    "DEFAULT_TARGET_FAR",
    "ENCODERS",
    "FAULT_IDS",
    "MAX_BUDGET",
    "N_ACTIONS",
    "NODES",
    "NODE_VARIABLES",
    "ONSET_TEST",
    "UNOBSERVABLE_FAULTS",
    "VARIABLES",
    "VOCAB_SIZE",
    "AlarmEncoder",
    "DecodedCentre",
    "FeatureEncoder",
    "MLPClassifier",
    "NodeLayout",
    "NoiseEncoder",
    "Normaliser",
    "OracleCentre",
    "PlantConfig",
    "PlantEnv",
    "Protocol",
    "Replicate",
    "RiethData",
    "RunSelection",
    "RunSplits",
    "SilentCentre",
    "SilentEncoder",
    "SoftmaxRegression",
    "SummaryEncoder",
    "TepData",
    "UplinkEncoder",
    "block_subsample",
    "blocked_folds",
    "default_cache_dir",
    "diagnosis_metrics",
    "lorawan_airtime_s",
    "macro_f1",
    "make_classifier",
    "make_encoder",
    "make_layout",
    "message_spec",
    "min_uplink_period_s",
    "raw_features",
    "rieth_cache_dir",
    "run_episodes",
    "stack_context",
    "synthetic_rieth",
    "synthetic_tep",
    "to_runs",
    "tune_centre",
]
