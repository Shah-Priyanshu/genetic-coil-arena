"""Model backends: the real-model layer behind the specialist interface.

- ``base``: the ``ModelBackend`` / ``LatentBackend`` protocols and result types
- ``hf_peft``: ``HFPeftBackend``, LoRA specialists on one shared base model, with
  hidden-state capture, soft-prompt injection, KV-prefix forwarding and selective-logits
  training
- ``ollama``: ``OllamaBackend``, text-only generation with JSON-schema output
- ``adapter_text``: ``AdapterTextBackend``, one LoRA adapter as a distilled-translator backend

Importing this package needs neither torch, transformers nor peft: they load on the
first call that needs them.
"""

from .adapter_text import AdapterTextBackend
from .base import (
    CAP_GENERATE,
    CAP_HIDDEN,
    CAP_KV,
    CAP_SOFT,
    CAP_STRUCTURED,
    CAP_TRAIN,
    DEFAULT_BASE_ID,
    GenerationResult,
    HiddenStates,
    KVPrefix,
    LatentBackend,
    ModelBackend,
)
from .hf_peft import (
    DEFAULT_TARGET_MODULES,
    FitStats,
    HFPeftBackend,
    ModelInputs,
    kv_bytes_per_token,
)
from .ollama import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    OllamaBackend,
    OllamaError,
    http_transport,
)

__all__ = [
    "AdapterTextBackend",
    "CAP_GENERATE",
    "CAP_HIDDEN",
    "CAP_KV",
    "CAP_SOFT",
    "CAP_STRUCTURED",
    "CAP_TRAIN",
    "DEFAULT_BASE_ID",
    "DEFAULT_HOST",
    "DEFAULT_MODEL",
    "DEFAULT_TARGET_MODULES",
    "FitStats",
    "GenerationResult",
    "HFPeftBackend",
    "HiddenStates",
    "KVPrefix",
    "LatentBackend",
    "ModelBackend",
    "ModelInputs",
    "OllamaBackend",
    "OllamaError",
    "http_transport",
    "kv_bytes_per_token",
]
