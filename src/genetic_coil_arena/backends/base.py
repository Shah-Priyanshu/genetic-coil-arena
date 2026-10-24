"""The model-backend interface: what a specialist needs from a language model.

A backend is identified by ``(base_id, adapter_id)``: one base model, plus an optional
named adapter (a LoRA specialist) selected per call. ``ModelBackend`` is the minimum, a
text-in, text-out ``generate``. ``LatentBackend`` adds what latent communication needs
between specialists that share a base: hidden-state capture and access to the KV cache.

Types of tensors are ``Any`` here so this module imports no ML framework.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

DEFAULT_BASE_ID = "Qwen/Qwen3-1.7B"

CAP_GENERATE = "generate"
CAP_STRUCTURED = "structured_output"
CAP_HIDDEN = "hidden_states"
CAP_KV = "kv_cache"
CAP_SOFT = "soft_prompt"
CAP_TRAIN = "train"


@dataclass(frozen=True)
class GenerationResult:
    """One generated completion.

    Attributes:
        text: Generated text, without the stop token
        token_ids: Generated token ids, including the stop token if one ended the text
        n_prompt_tokens: Length of the prompt, in tokens (soft and cached tokens included
            for local backends)
        n_new_tokens: Number of tokens generated
        finish_reason: ``"stop"`` (a stop token) or ``"length"`` (the budget ran out)
        latency_s: Wall time of the call that produced it (shared by a whole batch)
        meta: Backend-specific extras, e.g. Ollama's timing counters
    """

    text: str
    token_ids: tuple[int, ...] = ()
    n_prompt_tokens: int = 0
    n_new_tokens: int = 0
    finish_reason: str = "stop"
    latency_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class HiddenStates:
    """Hidden states of a batch of prompts at one layer.

    Prompts are left-padded, so ``states[:, -k:]`` are each prompt's last ``k`` real
    positions.

    Attributes:
        states: Tensor ``[B, L, D]``
        mask: Tensor ``[B, L]``, 1 for real positions and 0 for padding
        layer: Index into the hidden-state tuple (0 = embeddings, N = last layer)
    """

    states: Any
    mask: Any
    layer: int


@dataclass
class KVPrefix:
    """The KV cache of a batch of prompts, ready to be a receiver's prefix.

    Attributes:
        cache: The cache object (a transformers ``DynamicCache``); extending it mutates
            it, so :meth:`HFPeftBackend.build_inputs` clones it by default
        mask: Tensor ``[B, Lp]``, 1 for real cached positions and 0 for padding
        lengths: Tensor ``[B]`` of real cached lengths
    """

    cache: Any
    mask: Any
    lengths: Any


@runtime_checkable
class ModelBackend(Protocol):
    """A text generator addressed by ``(base_id, adapter_id)``.

    Attributes:
        base_id: Identifier of the base model (a hub id or an Ollama tag)
        capabilities: What the backend can do, a subset of the ``CAP_*`` names
    """

    base_id: str
    capabilities: frozenset[str]

    def generate(
        self,
        prompts: Sequence[str],
        adapter_id: Optional[str] = None,
        **options: Any,
    ) -> list[GenerationResult]:
        """Complete each prompt; ``adapter_id=None`` runs the base model."""
        ...


@runtime_checkable
class LatentBackend(ModelBackend, Protocol):
    """A backend whose internal state specialists can exchange (shared base model)."""

    def hidden_states(
        self,
        prompts: Sequence[str],
        layer: int,
        adapter_id: Optional[str] = None,
    ) -> HiddenStates:
        """Hidden states of ``prompts`` at ``layer`` under ``adapter_id``."""
        ...

    def prefill(
        self, prompts: Sequence[str], adapter_id: Optional[str] = None
    ) -> KVPrefix:
        """Run ``prompts`` under ``adapter_id`` and keep their KV cache."""
        ...
