"""A distilled-translator view of a LoRA adapter on the HF backend.

The translators (``Translator``, ``PolicyTranslator``) call ``backend.generate(prompts,
system=..., examples=..., schema=...)``. A distilled model has the task and the reply
format in its weights, so this wrapper ignores the few-shot examples and the schema and
sends ``translator.distill.format_prompt(system, prompt)`` to one adapter of a
:class:`~genetic_coil_arena.backends.hf_peft.HFPeftBackend`. ``system`` carries the task
name (the spec name), nothing else.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from ..translator.distill import format_prompt
from .base import CAP_GENERATE, GenerationResult


class AdapterTextBackend:
    """``generate`` over one LoRA adapter, with the distilled prompt format.

    Args:
        hf: The HF backend (model loaded, adapter present)
        adapter_id: Adapter to run
        max_new_tokens: Generation budget per reply
    """

    capabilities = frozenset({CAP_GENERATE})

    def __init__(self, hf: Any, adapter_id: str, max_new_tokens: int = 96):
        self.hf = hf
        self.adapter_id = adapter_id
        self.base_id = f"{hf.base_id}+{adapter_id}"
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        prompts: Sequence[str],
        adapter_id: Optional[str] = None,
        *,
        system: Optional[str] = None,
        examples: Sequence[tuple[str, str]] = (),
        schema: Optional[Mapping[str, Any]] = None,
        max_new_tokens: Optional[int] = None,
        keep_alive: Any = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> list[GenerationResult]:
        """Greedy replies, one per prompt (extra keywords are accepted and ignored)."""
        task = system or ""
        formatted = [format_prompt(task, p) for p in prompts]
        return self.hf.generate(
            formatted,
            adapter_id or self.adapter_id,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
        )
