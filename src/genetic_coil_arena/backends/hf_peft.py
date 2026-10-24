"""Hugging Face + PEFT backend: LoRA specialists on one shared base model.

One base model stays resident and several named LoRA adapters ride on it; switching
costs about 4 ms (``set_adapter``). Because every specialist shares the base, they share
a hidden space, which is what makes the latent channels possible:

- **hidden-state capture** at a chosen layer (``hidden_states``);
- **soft-prompt injection**: tokens made by a learned projector are prepended to the
  receiver's input embeddings (``build_inputs(soft=...)``, ``generate(soft=...)``);
- **KV-prefix forwarding**: the receiver attends to the sender's KV cache of the
  message (``prefill`` then ``build_inputs(prefix=...)``);
- **selective-logits training**: ``lm_head`` runs only on supervised positions. Full
  vocabulary logits dominate training memory; skipping the rest lets a 16 x 128 batch
  train without gradient checkpointing on a 12 GB card (R4).

Raw residual-stream states are 76-100x larger than token embeddings, so they must never
be injected as they are; put a learned projector (``channels.latent``) in between.

torch, transformers and peft are imported lazily, on first use, so importing this
module needs none of them. On Windows the GPU driver silently spills to system RAM
instead of raising out-of-memory (a run once hung for over 10 minutes that way), so
``load`` caps the process at ``memory_fraction`` of the card (0.85).
"""

import contextlib
import copy
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .base import (
    CAP_GENERATE,
    CAP_HIDDEN,
    CAP_KV,
    CAP_SOFT,
    CAP_TRAIN,
    DEFAULT_BASE_ID,
    GenerationResult,
    HiddenStates,
    KVPrefix,
)

DEFAULT_TARGET_MODULES = ("q_proj", "v_proj")


@dataclass
class ModelInputs:
    """Everything one forward pass needs, built by :meth:`HFPeftBackend.build_inputs`.

    Attributes:
        inputs_embeds: ``[B, T, D]`` (soft tokens first, then the text)
        attention_mask: ``[B, P + T]``; ``P`` is the cached prefix length (0 if none)
        position_ids: ``[B, T]``, continuing after the cached prefix
        labels: ``[B, T]`` token ids to predict, ``-100`` where there is no target
        past_key_values: Cache to extend (a clone of the prefix cache), or ``None``
        n_prompt_tokens: ``[B]`` real tokens of the prompt, soft tokens included
    """

    inputs_embeds: Any
    attention_mask: Any
    position_ids: Any
    labels: Any
    past_key_values: Any
    n_prompt_tokens: Any


@dataclass
class FitStats:
    """What a training run did."""

    steps: int
    wall_s: float
    final_loss: float
    mean_last_loss: float
    losses: list[float] = field(default_factory=list)
    peak_memory_mib: float = 0.0
    n_trainable: int = 0


def kv_bytes_per_token(
    num_layers: int, num_kv_heads: int, head_dim: int, itemsize: int = 2
) -> int:
    """Bytes of KV cache one token occupies (keys and values, every layer).

    For Qwen3-1.7B in bf16 (28 layers, 8 KV heads, head size 128) this is 114,688,
    i.e. 112 KiB.
    """
    return 2 * num_layers * num_kv_heads * head_dim * itemsize


class HFPeftBackend:
    """A causal LM with named LoRA adapters; see the module docstring.

    Nothing heavy happens until the first call that needs the model (or an explicit
    :meth:`load`). Pass ``model`` and ``tokenizer`` to wrap objects you built yourself
    (tests do, with a tiny random model).

    ``adapter_id=None`` in :meth:`generate`, :meth:`hidden_states` and :meth:`prefill`
    means the plain base model (adapters disabled).

    Args:
        base_id: Hub id or local path of the base model
        device: ``"cuda"``, ``"cpu"`` or ``None`` for CUDA when available
        dtype: ``"bfloat16"``, ``"float16"`` or ``"float32"``
        attn_implementation: Passed to ``from_pretrained``
        memory_fraction: Cap on the share of GPU memory this process may use; ``None``
            disables the cap (don't, on Windows)
        num_threads: Cap on torch CPU threads (the CPU is shared; ``None`` leaves it)
        local_files_only: Never touch the network
        model, tokenizer: Pre-built objects to wrap instead of loading ``base_id``
    """

    capabilities = frozenset({CAP_GENERATE, CAP_HIDDEN, CAP_KV, CAP_SOFT, CAP_TRAIN})

    def __init__(
        self,
        base_id: str = DEFAULT_BASE_ID,
        *,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        memory_fraction: Optional[float] = 0.85,
        num_threads: Optional[int] = 4,
        local_files_only: bool = False,
        model: Any = None,
        tokenizer: Any = None,
    ):
        self.base_id = base_id
        self.adapter_id: Optional[str] = None
        self.device_name = device
        self.dtype_name = dtype
        self.attn_implementation = attn_implementation
        self.memory_fraction = memory_fraction
        self.num_threads = num_threads
        self.local_files_only = local_files_only
        self._model = model
        self._tokenizer = tokenizer
        self._peft: Any = None
        self._adapter_cfg: dict[str, dict[str, Any]] = {}
        self._training_adapters: frozenset[str] = frozenset()
        self._torch: Any = None

    # ------------------------------------------------------------------- loading

    def load(self) -> "HFPeftBackend":
        """Import the ML stack and load the model and tokenizer (idempotent)."""
        if self._torch is not None:
            return self
        import torch

        self._torch = torch
        if self.num_threads:
            torch.set_num_threads(int(self.num_threads))
        device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = getattr(torch, self.dtype_name)
        if self.device.type == "cuda" and self.memory_fraction:
            torch.cuda.set_per_process_memory_fraction(
                float(self.memory_fraction), self.device
            )
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.base_id, local_files_only=self.local_files_only
            )
        if self._model is None:
            from transformers import AutoModelForCausalLM

            kwargs: dict[str, Any] = {
                "dtype": self.dtype,
                "attn_implementation": self.attn_implementation,
                "local_files_only": self.local_files_only,
            }
            if self.device.type == "cuda":
                kwargs["device_map"] = str(self.device)
            self._model = AutoModelForCausalLM.from_pretrained(self.base_id, **kwargs)
            if self.device.type != "cuda":
                self._model.to(self.device)
        else:
            self._model.to(self.device)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()
        return self

    @property
    def tokenizer(self) -> Any:
        return self.load()._tokenizer

    @property
    def model(self) -> Any:
        """The underlying ``AutoModelForCausalLM`` (LoRA layers injected in place)."""
        return self.load()._model

    @property
    def torch(self) -> Any:
        """The ``torch`` module (importing it and loading the model on first use)."""
        return self.load()._torch

    @property
    def config(self) -> Any:
        return self.model.config

    @property
    def num_layers(self) -> int:
        return int(self.config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    @property
    def stop_ids(self) -> tuple[int, ...]:
        eos = self.tokenizer.eos_token_id
        return () if eos is None else (int(eos),)

    @property
    def kv_bytes_per_token(self) -> int:
        """Bytes of KV cache per token at the model's dtype."""
        cfg = self.config
        heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = (
            getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        )
        itemsize = self.torch.empty((), dtype=self.dtype).element_size()
        return kv_bytes_per_token(self.num_layers, heads, head_dim, itemsize)

    def _lm(self) -> Any:
        """The causal LM that LoRA layers live in (``model.model`` is its transformer)."""
        self.load()
        return self._peft.base_model.model if self._peft is not None else self._model

    # ------------------------------------------------------------------ adapters

    @property
    def adapters(self) -> list[str]:
        """Names of the loaded adapters."""
        return list(self._adapter_cfg)

    def add_adapter(
        self,
        adapter_id: str,
        *,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
        seed: Optional[int] = None,
        activate: bool = True,
    ) -> str:
        """Add a fresh LoRA adapter (zero-initialised B, so it starts as the base).

        Args:
            adapter_id: Name of the specialist
            r, alpha, dropout, target_modules: LoRA hyper-parameters
            seed: Seeds the random init of the LoRA A matrices
            activate: Make it the active adapter
        """
        if not adapter_id or "." in adapter_id:
            raise ValueError(
                f"adapter id must be non-empty and have no dots: {adapter_id!r}"
            )
        if adapter_id in self._adapter_cfg:
            raise ValueError(f"adapter {adapter_id!r} already exists")
        self.load()
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            task_type="CAUSAL_LM",
        )
        if seed is not None:
            self._torch.manual_seed(seed)
        if self._peft is None:
            self._peft = get_peft_model(self._model, cfg, adapter_name=adapter_id)
            self._peft.eval()
        else:
            self._peft.add_adapter(adapter_id, cfg)
        self._adapter_cfg[adapter_id] = {
            "r": r,
            "alpha": alpha,
            "dropout": dropout,
            "target_modules": list(target_modules),
        }
        for p in self.adapter_parameters(adapter_id):
            p.requires_grad_(False)
        if activate or self.adapter_id is None:
            self.set_adapter(adapter_id)
        return adapter_id

    def set_adapter(
        self, adapter_id: Optional[str], *, trainable: Optional[bool] = None
    ) -> None:
        """Select the active adapter; ``None`` runs the base model.

        PEFT freezes every other adapter when one is selected, so a trainable adapter
        would lose its gradients whenever a frozen sender runs in between. The backend
        therefore remembers the adapters that :meth:`fit` is training and re-enables their
        gradients after every switch.

        Args:
            adapter_id: Adapter name or ``None``
            trainable: Whether the adapter's parameters require gradients; by default
                only the adapter currently being trained
        """
        self.load()
        if adapter_id is None:
            if self._peft is not None:
                self._peft.base_model.disable_adapter_layers()
            self.adapter_id = None
            return
        if adapter_id not in self._adapter_cfg:
            raise KeyError(f"unknown adapter {adapter_id!r}; have {self.adapters}")
        self._peft.base_model.enable_adapter_layers()
        self._peft.set_adapter(adapter_id)
        self.adapter_id = adapter_id
        for name in self._training_adapters:  # PEFT froze them; they stay trainable
            for p in self.adapter_parameters(name):
                p.requires_grad_(True)
        flag = adapter_id in self._training_adapters if trainable is None else trainable
        for p in self.adapter_parameters(adapter_id):
            p.requires_grad_(flag)

    @contextlib.contextmanager
    def use_adapter(self, adapter_id: Optional[str]) -> Iterator[None]:
        """Temporarily select an adapter (``None`` for the base model)."""
        previous = self.adapter_id
        if previous == adapter_id:
            yield
            return
        self.set_adapter(adapter_id)
        try:
            yield
        finally:
            self.set_adapter(previous)

    def adapter_parameters(self, adapter_id: str) -> list[Any]:
        """The LoRA parameters of one adapter."""
        if self._peft is None:
            return []
        marker = f".{adapter_id}."
        return [
            p
            for name, p in self._peft.named_parameters()
            if "lora_" in name and marker in name
        ]

    def adapter_state(self, adapter_id: str) -> dict[str, Any]:
        """A CPU copy of an adapter's weights (keys without the adapter name)."""
        from peft import get_peft_model_state_dict

        state = get_peft_model_state_dict(self._peft, adapter_name=adapter_id)
        return {k: v.detach().cpu().clone() for k, v in state.items()}

    def load_adapter_state(self, adapter_id: str, state: Mapping[str, Any]) -> None:
        """Overwrite an adapter's weights with ``state`` (from :meth:`adapter_state`)."""
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(self._peft, dict(state), adapter_name=adapter_id)

    def save_adapter(self, adapter_id: str, path: Union[str, Path]) -> None:
        """Save an adapter's weights and hyper-parameters to one file."""
        self._torch.save(
            {
                "config": self._adapter_cfg[adapter_id],
                "state": self.adapter_state(adapter_id),
            },
            path,
        )

    def load_adapter(self, adapter_id: str, path: Union[str, Path]) -> str:
        """Create adapter ``adapter_id`` from a file written by :meth:`save_adapter`."""
        self.load()
        blob = self._torch.load(path, map_location="cpu")
        self.add_adapter(adapter_id, activate=False, **blob["config"])
        self.load_adapter_state(adapter_id, blob["state"])
        return adapter_id

    def delete_adapter(self, adapter_id: str) -> None:
        """Remove an adapter and free its weights."""
        if adapter_id not in self._adapter_cfg:
            raise KeyError(f"unknown adapter {adapter_id!r}")
        if self.adapter_id == adapter_id:
            others = [a for a in self._adapter_cfg if a != adapter_id]
            if others:
                self.set_adapter(others[0])
            else:
                self.set_adapter(None)
        self._peft.delete_adapter(adapter_id)
        del self._adapter_cfg[adapter_id]

    # ------------------------------------------------------------------- tokens

    def embed(self, input_ids: Any) -> Any:
        """Token embeddings ``[B, L, D]`` of ``input_ids``."""
        return self._lm().get_input_embeddings()(input_ids)

    def embedding_rms(self) -> float:
        """Root-mean-square element of the token embedding matrix.

        A learned projector should output soft tokens of about this scale (about 0.033
        for Qwen3-1.7B); raw hidden states are 76-100x larger.
        """
        weight = self._lm().get_input_embeddings().weight
        return float(weight.detach().float().pow(2).mean().sqrt())

    def build_inputs(
        self,
        prompts: Sequence[str],
        targets: Optional[Sequence[str]] = None,
        *,
        soft: Any = None,
        soft_head: str = "",
        prefix: Optional[KVPrefix] = None,
        padding_side: Optional[str] = None,
        clone_prefix: bool = True,
    ) -> ModelInputs:
        """Tokenize and assemble a batch.

        The layout is ``[soft_head][soft tokens][prompt][target][stop token]``,
        optionally behind a cached prefix. Prompts and targets are tokenized separately,
        so training and generation see the same prompt tokens. Labels cover the target
        and its stop token only.

        ``soft_head`` is a fixed text (the same for every row) placed before the soft
        tokens. Without it the first soft token sits at position 0, where decoder LLMs
        keep their attention sink and overwrite the token's content.

        Args:
            prompts: Text of each row
            targets: Text the model should produce after the prompt (training); ``None``
                for generation
            soft: Tensor ``[B, k, D]`` placed before the text (a projector's output)
            soft_head: Text placed before the soft tokens (needs ``soft``)
            prefix: KV prefix the text continues from
            padding_side: ``"right"`` by default when training, else ``"left"``
            clone_prefix: Copy the prefix cache (extending it mutates it)
        """
        torch = self.torch
        tok = self.tokenizer
        rows = len(prompts)
        if targets is not None and len(targets) != rows:
            raise ValueError(f"{len(targets)} targets for {rows} prompts")
        prompt_ids = tok(list(prompts), add_special_tokens=False)["input_ids"]
        if targets is not None:
            stop = list(self.stop_ids[:1])
            target_ids = [
                list(ids) + stop
                for ids in tok(list(targets), add_special_tokens=False)["input_ids"]
            ]
        else:
            target_ids = [[] for _ in range(rows)]
        side = padding_side or ("right" if targets is not None else "left")
        if side not in ("left", "right"):
            raise ValueError(f"padding_side must be 'left' or 'right', got {side!r}")

        width = max(len(p) + len(t) for p, t in zip(prompt_ids, target_ids))
        ids = torch.full((rows, width), int(tok.pad_token_id), dtype=torch.long)
        labels = torch.full((rows, width), -100, dtype=torch.long)
        mask = torch.zeros((rows, width), dtype=torch.long)
        for i, (p, t) in enumerate(zip(prompt_ids, target_ids)):
            n = len(p) + len(t)
            lo = width - n if side == "left" else 0
            ids[i, lo : lo + n] = torch.tensor(p + t, dtype=torch.long)
            if t:
                labels[i, lo + len(p) : lo + n] = torch.tensor(t, dtype=torch.long)
            mask[i, lo : lo + n] = 1
        ids, labels, mask = (x.to(self.device) for x in (ids, labels, mask))
        n_prompt = torch.tensor([len(p) for p in prompt_ids], device=self.device)

        embeds = self.embed(ids)
        if soft is not None:
            if soft.shape[0] != rows:
                raise ValueError(f"soft has batch {soft.shape[0]}, expected {rows}")
            head = [
                int(i) for i in tok(soft_head, add_special_tokens=False)["input_ids"]
            ]
            pieces = [soft.to(embeds.dtype), embeds]
            if head:
                head_ids = torch.tensor(head, dtype=torch.long, device=self.device)
                pieces.insert(0, self.embed(head_ids)[None].expand(rows, -1, -1))
            lead = len(head) + soft.shape[1]
            embeds = torch.cat(pieces, dim=1)
            ones = torch.ones((rows, lead), dtype=torch.long, device=self.device)
            mask = torch.cat([ones, mask], dim=1)
            labels = torch.cat([ones * -100, labels], dim=1)
            n_prompt = n_prompt + lead
        elif soft_head:
            raise ValueError("soft_head needs soft tokens")

        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
        attention_mask, past = mask, None
        if prefix is not None:
            past = copy.deepcopy(prefix.cache) if clone_prefix else prefix.cache
            attention_mask = torch.cat([prefix.mask.to(mask.dtype), mask], dim=1)
            position_ids = position_ids + prefix.lengths.to(position_ids.dtype)[:, None]
        return ModelInputs(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            past_key_values=past,
            n_prompt_tokens=n_prompt,
        )

    # ------------------------------------------------------------------- forward

    def forward_hidden(
        self,
        inputs: ModelInputs,
        *,
        output_hidden_states: bool = False,
        use_cache: Optional[bool] = None,
    ) -> Any:
        """Run the transformer (no ``lm_head``) on assembled inputs.

        ``use_cache`` defaults to whether a cache was passed in.
        """
        if use_cache is None:
            use_cache = inputs.past_key_values is not None
        return self._lm().model(
            inputs_embeds=inputs.inputs_embeds,
            attention_mask=inputs.attention_mask,
            position_ids=inputs.position_ids,
            past_key_values=inputs.past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

    def lm_loss(self, hidden: Any, labels: Any, chunk_size: Optional[int] = 256) -> Any:
        """Cross-entropy on the supervised positions only (selective logits).

        ``lm_head`` is applied to the hidden states that predict a labelled token and
        nothing else, so the ``[B, L, vocab]`` logits tensor is never built. Long
        targets still make an ``[N, vocab]`` float32 tensor (0.6 MB per supervised token
        for a 152k vocabulary); ``chunk_size`` computes it ``chunk_size`` rows at a time
        and recomputes each chunk in the backward pass instead of storing it.

        Args:
            hidden: ``[B, T, D]`` last hidden states
            labels: ``[B, T]`` with ``-100`` where there is no target; position ``t``
                predicts ``labels[:, t + 1]``
            chunk_size: Rows per chunk, or ``None`` for one piece
        """
        torch = self.torch
        target = labels[:, 1:]
        selected = target != -100
        rows, goal = hidden[:, :-1][selected], target[selected]
        head = self._lm().lm_head

        def chunk_loss(h: Any, t: Any) -> Any:
            logits = head(h).float()
            return torch.nn.functional.cross_entropy(logits, t, reduction="sum")

        if not chunk_size or rows.shape[0] <= chunk_size:
            return chunk_loss(rows, goal) / rows.shape[0]
        total = rows.new_zeros((), dtype=torch.float32)
        for lo in range(0, rows.shape[0], chunk_size):
            piece = (rows[lo : lo + chunk_size], goal[lo : lo + chunk_size])
            if torch.is_grad_enabled() and rows.requires_grad:
                total = total + torch.utils.checkpoint.checkpoint(
                    chunk_loss, *piece, use_reentrant=False
                )
            else:
                total = total + chunk_loss(*piece)
        return total / rows.shape[0]

    def training_loss(self, inputs: ModelInputs) -> Any:
        """Forward pass and selective-logits loss for assembled inputs."""
        out = self.forward_hidden(inputs)
        return self.lm_loss(out.last_hidden_state, inputs.labels)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Trade about 30% more time for much lower activation memory (training only)."""
        self.load()
        target = self._peft if self._peft is not None else self._model
        if enabled:
            target.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            target.gradient_checkpointing_disable()

    # ---------------------------------------------------------------- generation

    def greedy_decode(
        self,
        inputs: ModelInputs,
        *,
        max_new_tokens: int,
        stop_ids: Optional[Sequence[int]] = None,
    ) -> tuple[Any, Any, Any]:
        """Greedy decoding from assembled inputs, with KV caching.

        Returns:
            ``(tokens, n_new, stopped)``: tokens ``[B, T]`` (padded after a row's stop
            token), the number of tokens each row produced (stop token included), and
            whether each row ended on a stop token
        """
        torch = self.torch
        lm = self._lm()
        stops = torch.tensor(
            list(self.stop_ids if stop_ids is None else stop_ids),
            dtype=torch.long,
            device=self.device,
        )
        pad = int(self.tokenizer.pad_token_id)
        with torch.no_grad():
            out = self.forward_hidden(inputs, use_cache=True)
            cache = out.past_key_values
            hidden = out.last_hidden_state[:, -1]
            mask = inputs.attention_mask
            position = inputs.position_ids[:, -1:] + 1
            rows = mask.shape[0]
            finished = torch.zeros(rows, dtype=torch.bool, device=self.device)
            n_new = torch.zeros(rows, dtype=torch.long, device=self.device)
            produced = []
            for step in range(max_new_tokens):
                token = lm.lm_head(hidden).argmax(-1)
                token = torch.where(finished, torch.full_like(token, pad), token)
                produced.append(token)
                n_new += (~finished).long()
                if stops.numel():
                    finished = finished | torch.isin(token, stops)
                if step == max_new_tokens - 1 or bool(finished.all()):
                    break
                mask = torch.cat([mask, mask.new_ones((rows, 1))], dim=1)
                out = lm.model(
                    input_ids=token[:, None],
                    attention_mask=mask,
                    position_ids=position,
                    past_key_values=cache,
                    use_cache=True,
                )
                hidden = out.last_hidden_state[:, -1]
                position = position + 1
        tokens = torch.stack(produced, dim=1)
        return tokens, n_new, finished

    def generate(
        self,
        prompts: Sequence[str],
        adapter_id: Optional[str] = None,
        *,
        max_new_tokens: int = 128,
        soft: Any = None,
        soft_head: str = "",
        prefix: Optional[KVPrefix] = None,
        stop_ids: Optional[Sequence[int]] = None,
        clone_prefix: bool = True,
    ) -> list[GenerationResult]:
        """Greedy completion of each prompt under ``adapter_id`` (``None`` = base model).

        Args:
            prompts: Text of each row
            adapter_id: Adapter to run
            max_new_tokens: Generation budget
            soft: Soft tokens ``[B, k, D]`` to put before each prompt (latent message)
            soft_head: Text to put before the soft tokens
            prefix: KV prefix each prompt continues from (KV message)
            stop_ids: Token ids that end a row (default: the tokenizer's EOS)
            clone_prefix: Copy the prefix cache instead of extending it in place
        """
        torch = self.torch
        start = time.perf_counter()
        with self.use_adapter(adapter_id):
            inputs = self.build_inputs(
                prompts,
                soft=soft,
                soft_head=soft_head,
                prefix=prefix,
                clone_prefix=clone_prefix,
            )
            tokens, n_new, stopped = self.greedy_decode(
                inputs, max_new_tokens=max_new_tokens, stop_ids=stop_ids
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        results = []
        tokens_cpu, n_cpu = tokens.cpu(), n_new.cpu()
        stopped_cpu, n_prompt = stopped.cpu(), inputs.n_prompt_tokens.cpu()
        stop_set = set(self.stop_ids if stop_ids is None else stop_ids)
        for i in range(len(prompts)):
            ids = tokens_cpu[i, : int(n_cpu[i])].tolist()
            text_ids = [t for t in ids if t not in stop_set]
            results.append(
                GenerationResult(
                    text=self.tokenizer.decode(text_ids, skip_special_tokens=True),
                    token_ids=tuple(ids),
                    n_prompt_tokens=int(n_prompt[i]),
                    n_new_tokens=int(n_cpu[i]),
                    finish_reason="stop" if bool(stopped_cpu[i]) else "length",
                    latency_s=elapsed,
                    meta={"batch_size": len(prompts)},
                )
            )
        return results

    # ---------------------------------------------------------------- latent state

    def hidden_states(
        self,
        prompts: Sequence[str],
        layer: int,
        adapter_id: Optional[str] = None,
        *,
        grad: bool = False,
    ) -> HiddenStates:
        """Hidden states of ``prompts`` at ``layer`` (no gradients unless ``grad``).

        With ``grad`` the states stay attached to the graph, so a loss downstream of
        them trains the sender's adapter too (pass it to :meth:`fit` as an extra
        adapter).

        Index 0 is the embedding output and index ``num_layers`` the last layer (after
        the final norm); negative indices count from the end. Prompts are left-padded,
        so the last positions line up across the batch.
        """
        torch = self.torch
        with contextlib.ExitStack() as stack:
            if not grad:
                stack.enter_context(torch.no_grad())
            stack.enter_context(self.use_adapter(adapter_id))
            inputs = self.build_inputs(prompts)
            out = self.forward_hidden(inputs, output_hidden_states=True)
        index = layer if layer >= 0 else self.num_layers + 1 + layer
        if not 0 <= index <= self.num_layers:
            raise ValueError(f"layer {layer} is outside [0, {self.num_layers}]")
        return HiddenStates(
            states=out.hidden_states[index],
            mask=inputs.attention_mask,
            layer=index,
        )

    def next_token_logprobs(
        self,
        prompts: Sequence[str],
        token_ids: Sequence[int],
        adapter_id: Optional[str] = None,
        *,
        chunk_size: Optional[int] = None,
    ) -> list[list[float]]:
        """Log-probability of each of ``token_ids`` as the token right after each prompt.

        One forward pass, no cache kept, and the vocabulary projection runs on the last
        position only. The softmax is over the whole vocabulary, so the values are true
        log-probabilities, not renormalised over ``token_ids``. Use it to rank a closed
        set of answers (e.g. option letters) without generating text.

        Args:
            prompts: Text of each row
            token_ids: Candidate next tokens
            adapter_id: Adapter to run (``None`` = base model)
            chunk_size: If set and a prompt is longer, read it in chunks of this many
                tokens through a KV cache. Attention memory then grows with the chunk,
                not with the square of the prompt, which is what lets a 16k-token prompt
                fit in 12 GB (the model's attention otherwise materialises
                ``[heads, T, T]``)

        Returns:
            ``[len(prompts)][len(token_ids)]`` nested lists
        """
        torch = self.torch
        ids = torch.tensor(list(token_ids), dtype=torch.long, device=self.device)
        with torch.no_grad(), self.use_adapter(adapter_id):
            inputs = self.build_inputs(prompts)
            length = inputs.inputs_embeds.shape[1]
            if chunk_size and length > chunk_size:
                lm = self._lm()
                cache = self._new_cache()
                for lo in range(0, length, chunk_size):
                    hi = min(lo + chunk_size, length)
                    out = lm.model(
                        inputs_embeds=inputs.inputs_embeds[:, lo:hi],
                        attention_mask=inputs.attention_mask[:, :hi],
                        position_ids=inputs.position_ids[:, lo:hi],
                        past_key_values=cache,
                        use_cache=True,
                    )
            else:
                out = self.forward_hidden(inputs, use_cache=False)
            logits = self._lm().lm_head(out.last_hidden_state[:, -1]).float()
            logp = torch.log_softmax(logits, dim=-1)[:, ids]
        return logp.cpu().tolist()

    def prefill(
        self, prompts: Sequence[str], adapter_id: Optional[str] = None
    ) -> KVPrefix:
        """Run ``prompts`` under ``adapter_id`` and keep the KV cache (no gradients)."""
        torch = self.torch
        with torch.no_grad(), self.use_adapter(adapter_id):
            inputs = self.build_inputs(prompts)
            out = self.forward_hidden(inputs, use_cache=True)
        mask = inputs.attention_mask
        return KVPrefix(cache=out.past_key_values, mask=mask, lengths=mask.sum(-1))

    # ----------------------------------------------------------------- KV caches

    def _new_cache(self) -> Any:
        from transformers import DynamicCache

        return DynamicCache()

    @staticmethod
    def kv_pairs(cache: Any) -> list[tuple[Any, Any]]:
        """``(keys, values)`` tensors of every layer, whatever the cache version."""
        if hasattr(cache, "layers"):
            return [(layer.keys, layer.values) for layer in cache.layers]
        if hasattr(cache, "key_cache"):
            return list(zip(cache.key_cache, cache.value_cache))
        return [(k, v) for k, v in cache]

    def cache_from_pairs(self, pairs: Sequence[tuple[Any, Any]]) -> Any:
        """A fresh cache holding the given ``(keys, values)`` per layer."""
        cache = self._new_cache()
        for index, (keys, values) in enumerate(pairs):
            cache.update(keys, values, index)
        return cache

    def map_kv(self, prefix: KVPrefix, fn: Callable[[int, str, Any], Any]) -> KVPrefix:
        """A copy of ``prefix`` with ``fn(layer, "k" | "v", tensor)`` applied to each tensor.

        The building block of KV-channel interventions (zero, shuffle, noise).
        """
        pairs = [
            (fn(i, "k", k), fn(i, "v", v))
            for i, (k, v) in enumerate(self.kv_pairs(prefix.cache))
        ]
        return KVPrefix(
            cache=self.cache_from_pairs(pairs), mask=prefix.mask, lengths=prefix.lengths
        )

    @classmethod
    def kv_nbytes(cls, cache: Any) -> int:
        """Total bytes held by a cache (padding included)."""
        return sum(
            k.numel() * k.element_size() + v.numel() * v.element_size()
            for k, v in cls.kv_pairs(cache)
        )

    # ------------------------------------------------------------------ training

    def fit(
        self,
        adapter_id: str,
        step_fn: Callable[[int], Any],
        *,
        steps: int,
        lr: float = 5e-4,
        extra_parameters: Sequence[Any] = (),
        extra_lr: Optional[float] = None,
        extra_adapters: Sequence[str] = (),
        warmup: int = 20,
        min_lr_frac: float = 0.1,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = 1.0,
        log_every: int = 50,
        logger: Optional[Callable[[str], None]] = None,
        label: str = "",
    ) -> FitStats:
        """Train an adapter (and optionally extra modules such as a projector).

        AdamW with linear warm-up and a cosine decay to ``min_lr_frac * lr``. Every
        other adapter and the base stay frozen.

        Args:
            adapter_id: Adapter to train
            step_fn: ``step_fn(step) -> loss`` builds a batch and returns the scalar loss
                (use :meth:`build_inputs` and :meth:`training_loss`)
            steps: Number of optimiser steps
            lr: LoRA learning rate
            extra_parameters: Parameters of modules trained alongside (e.g. a projector)
            extra_lr: Learning rate of ``extra_parameters`` (default ``lr``)
            extra_adapters: Further adapters to train in the same run (a sender that
                learns end to end with its receiver); same learning rate as ``adapter_id``
            warmup, min_lr_frac: Schedule shape
            weight_decay: AdamW weight decay
            grad_clip: Global gradient-norm clip, or ``None``
            log_every: Log a line this often (and at the end)
            logger: Callable taking a line; ``None`` is silent
            label: Prefix of the log lines

        Raises:
            RuntimeError: The loss became non-finite
        """
        torch = self.torch
        self._training_adapters = frozenset([adapter_id, *extra_adapters])
        self.set_adapter(adapter_id)
        lora = self.adapter_parameters(adapter_id)
        for other in extra_adapters:
            lora += self.adapter_parameters(other)
        extra = list(extra_parameters)
        groups = [{"params": lora, "lr": lr}]
        if extra:
            groups.append({"params": extra, "lr": lr if extra_lr is None else extra_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)

        def scale(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, steps - warmup)
            return min_lr_frac + (1 - min_lr_frac) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
        trainable = lora + extra
        self._set_train_mode(True)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        losses: list[float] = []
        try:
            for step in range(steps):
                loss = step_fn(step)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"{label} loss is {float(loss)} at step {step}")
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
                if logger and ((step + 1) % log_every == 0 or step + 1 == steps):
                    recent = sum(losses[-log_every:]) / len(losses[-log_every:])
                    logger(
                        f"{label} step {step + 1}/{steps} loss {recent:.4f} "
                        f"({time.perf_counter() - start:.0f}s)"
                    )
        finally:
            self._set_train_mode(False)
            self._training_adapters = frozenset()
            self.set_adapter(adapter_id)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        tail = losses[-max(1, min(20, len(losses))) :]
        return FitStats(
            steps=steps,
            wall_s=time.perf_counter() - start,
            final_loss=losses[-1] if losses else float("nan"),
            mean_last_loss=sum(tail) / len(tail) if tail else float("nan"),
            losses=losses,
            peak_memory_mib=self.peak_memory_mib(),
            n_trainable=sum(p.numel() for p in trainable),
        )

    def _set_train_mode(self, training: bool) -> None:
        target = self._peft if self._peft is not None else self._model
        target.train(training)

    # -------------------------------------------------------------------- memory

    def peak_memory_mib(self) -> float:
        """Peak GPU memory allocated since the last reset, in MiB (0 on CPU)."""
        if self._torch is None or self.device.type != "cuda":
            return 0.0
        return float(self._torch.cuda.max_memory_allocated(self.device)) / 2**20

    def reset_peak_memory(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def synchronize(self) -> None:
        """Wait for queued GPU work (for timing)."""
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
