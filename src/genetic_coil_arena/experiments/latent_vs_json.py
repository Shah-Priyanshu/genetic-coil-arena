"""R9: do latent messages beat JSON between two LoRA specialists on one base model?

A two-stage records pipeline (``experiments.records``) run four ways, all on the same
base model with LoRA specialists:

- ``json``: the extractor A reads the raw snippet and writes raw-field JSON; the
  normalizer B reads that text and writes canonical JSON. B never sees the snippet.
- ``latent``: A's frozen hidden states (layer ``L``, last ``k`` positions or ``k``
  attention-pooled vectors) go through a learned projector into ``k`` soft tokens in
  front of B's input. Trained: B's LoRA and the projector. A is the extractor of ``json``.
- ``kv``: B attends to A's KV cache of the snippet (a high-bandwidth upper reference;
  possible because A and B share a base). Trained: B's LoRA.
- ``single``: one LoRA reads the snippet and writes canonical JSON (the
  non-communicating reference).

Each pipeline has ``respond(records)`` (end to end, with per-stage timings) so that
accuracy, message size, latency and message ablations are measured on the very same
code path. ``Experiment`` trains, evaluates and caches one result file per
``(seed, condition)`` so an interrupted run resumes.

The model layer is ``backends.HFPeftBackend``; the latent wire format and projector are
in ``channels.latent``.
"""

import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..backends import FitStats, HFPeftBackend, KVPrefix
from ..channels import (
    envelopes_to_latents,
    latents_to_envelopes,
    through_wire,
    wire_nbytes,
)
from ..evaluation.interventions import get_intervention
from ..protocol import InMemoryTensorStore
from .records import (
    CANON_FIELDS,
    RAW_FIELDS,
    Record,
    canonical_text,
    make_records,
    raw_json,
    score_fields,
    summarize_scores,
)

A_PROMPT = "Invoice:\n{text}\nExtract the fields as JSON:\n"
JSON_B_PROMPT = "Message:\n{message}\nNormalize to canonical JSON:\n"
RECEIVER_TAIL = "Normalize to canonical JSON:\n"  # after soft tokens or a KV prefix
LATENT_TAIL = "\nNormalize to canonical JSON:\n"  # follows the soft tokens like JSON_B_PROMPT follows its message
SOFT_HEAD = (
    "Message:\n"  # before the soft tokens, so none sits at position 0 (the sink)
)
SINGLE_PROMPT = "Invoice:\n{text}\nNormalize to canonical JSON:\n"

TEST_SEED = 1_000  # the test sets are shared by every run seed
ABLATIONS = ("zero", "shuffle", "noise")
SPLITS = ("id_test", "heldout_test")


@dataclass(frozen=True)
class Settings:
    """Hyper-parameters of a run (see ``scripts/r9_latent_vs_json.py`` for the CLI)."""

    steps: int = 200  # optimiser steps of the extractor, JSON, KV and single adapters
    latent_steps: int = 500  # latent receivers learn a new channel and get more
    batch_size: int = 16
    lr: float = 1e-3
    projector_lr: float = 1e-3
    soft_head: str = SOFT_HEAD
    soft_scale: float = 1.0  # initial soft-token scale, in units of the embedding scale
    warm_start: bool = True  # latent receivers start from the JSON receiver's LoRA
    n_calibration: int = 96  # training records used to measure the sender's statistics
    lora_r: int = 8
    lora_alpha: int = 16
    layer: int = 14
    n_test: int = 300  # records per test split
    n_ablation: int = 192  # records per split for the message ablations
    eval_batch: int = 64
    n_latency: int = 12  # records per split timed one at a time
    max_new_tokens: int = 80
    checkpoint_long: bool = (
        True  # gradient checkpointing for sequences over ~130 tokens
    )
    log_every: int = 50


@dataclass(frozen=True)
class Condition:
    """One way of running the pipeline.

    Attributes:
        kind: ``json``, ``latent``, ``kv`` or ``single``
        k: Soft tokens (latent only)
        layer: Hidden-state layer the latent message is taken from (latent only);
            ``None`` means ``Settings.layer``
        pooling: ``attn`` (``k`` learned queries over every position) or ``last`` (the
            last ``k`` positions)
        train_sender: Also train a copy of A's LoRA through the channel (end to end)
    """

    kind: str
    k: int = 0
    layer: Optional[int] = None
    pooling: str = "attn"
    train_sender: bool = False
    tag: str = ""

    @property
    def name(self) -> str:
        if self.kind == "latent":
            return f"latent_k{self.k}" + (f"_{self.tag}" if self.tag else "")
        return {"json": "json", "kv": "kv_prefix", "single": "single"}[self.kind]


# ------------------------------------------------------------------------------- helpers


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of ``range(n)`` with no fixed point (``n >= 2``)."""
    if n < 2:
        raise ValueError("a derangement needs at least 2 items")
    perm = rng.permutation(n)
    out = np.empty(n, dtype=np.int64)
    out[perm] = np.roll(perm, 1)
    return out


def _sync_time(backend: HFPeftBackend) -> float:
    backend.synchronize()
    return time.perf_counter()


def _batches(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


@dataclass
class Response:
    """What a pipeline returns for a batch of records.

    Attributes:
        outputs: Canonical-JSON text per record
        timings: Seconds spent in ``sender``, ``handoff`` and ``receiver``
        message_bytes: Wire bytes of the message per record
        message_tokens: Tokens per record (JSON text tokens; soft tokens; cached tokens)
        sender_texts: A's raw-field JSON (``json`` pipeline only)
    """

    outputs: list[str]
    timings: dict[str, float] = field(default_factory=dict)
    message_bytes: list[int] = field(default_factory=list)
    message_tokens: list[int] = field(default_factory=list)
    sender_texts: Optional[list[str]] = None


# ----------------------------------------------------------------------------- pipelines


class Pipeline:
    """A trained condition, runnable end to end on records."""

    def __init__(self, backend: HFPeftBackend, settings: Settings):
        self.be = backend
        self.settings = settings

    def respond(
        self,
        records: Sequence[Record],
        *,
        ablation: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        wire_dtype: str = "bfloat16",
        through_protocol: bool = False,
    ) -> Response:
        raise NotImplementedError

    def nll(
        self,
        records: Sequence[Record],
        *,
        ablation: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Teacher-forced negative log-likelihood of the gold canonical JSON, per token.

        A smooth measure of how much the receiver gets from the message: it moves even
        when exact match is still zero. No generation, so it is cheap.
        """
        raise NotImplementedError

    def _loss(self, adapter: str, inputs: Any) -> float:
        with self.be.torch.no_grad(), self.be.use_adapter(adapter):
            return float(self.be.training_loss(inputs))

    def _gen(self, prompts: Sequence[str], adapter: str, **kwargs: Any) -> list[Any]:
        return self.be.generate(
            prompts, adapter, max_new_tokens=self.settings.max_new_tokens, **kwargs
        )


class JsonPipeline(Pipeline):
    """A writes raw-field JSON text; B reads that text."""

    kind = "json"

    def __init__(self, backend: HFPeftBackend, settings: Settings, a: str, b: str):
        super().__init__(backend, settings)
        self.a, self.b = a, b

    def nll(
        self,
        records: Sequence[Record],
        *,
        ablation: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """NLL given the gold raw-field JSON as the message (no generation by A)."""
        texts = [raw_json(r.raw) for r in records]
        if ablation == "shuffle":
            order = derangement(len(texts), rng or np.random.default_rng(0))
            texts = [texts[i] for i in order]
        elif ablation is not None:
            raise ValueError("a text message can only be shuffled")
        inputs = self.be.build_inputs(
            [JSON_B_PROMPT.format(message=t) for t in texts],
            [canonical_text(r.gold) for r in records],
        )
        return self._loss(self.b, inputs)

    def respond(
        self,
        records: Sequence[Record],
        *,
        ablation: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        wire_dtype: str = "bfloat16",
        through_protocol: bool = False,
    ) -> Response:
        t0 = _sync_time(self.be)
        sent = self._gen([A_PROMPT.format(text=r.text) for r in records], self.a)
        t1 = _sync_time(self.be)
        texts = [g.text for g in sent]
        if ablation == "shuffle":
            texts = [
                texts[i]
                for i in derangement(len(texts), rng or np.random.default_rng(0))
            ]
        elif ablation is not None:
            raise ValueError("a text message can only be shuffled")
        t2 = _sync_time(self.be)
        got = self._gen([JSON_B_PROMPT.format(message=t) for t in texts], self.b)
        t3 = _sync_time(self.be)
        return Response(
            outputs=[g.text for g in got],
            timings={"sender": t1 - t0, "handoff": t2 - t1, "receiver": t3 - t2},
            message_bytes=[len(g.text.encode("utf-8")) for g in sent],
            message_tokens=[
                g.n_new_tokens - (1 if g.finish_reason == "stop" else 0) for g in sent
            ],
            sender_texts=[g.text for g in sent],
        )


class LatentPipeline(Pipeline):
    """A's hidden states, through a learned projector, as soft tokens in front of B."""

    kind = "latent"

    def __init__(
        self,
        backend: HFPeftBackend,
        settings: Settings,
        a: str,
        b: str,
        projector: Any,
        layer: int,
    ):
        super().__init__(backend, settings)
        self.a, self.b, self.proj, self.layer = a, b, projector, layer
        self.store = InMemoryTensorStore("r9")

    def message(self, texts: Sequence[str]) -> Any:
        """A's message ``[B, k, D]`` (float32) for the given snippets."""
        torch = self.be.torch
        prompts = [A_PROMPT.format(text=t) for t in texts]
        hs = self.be.hidden_states(prompts, self.layer, self.a)
        with torch.no_grad():
            return self.proj.encode(hs.states, hs.mask)

    def _receive(
        self,
        message: np.ndarray,
        ablation: Optional[str],
        rng: Optional[np.random.Generator],
        wire_dtype: str,
        through_protocol: bool,
    ) -> Any:
        """The wire and the receiver's projector: message array to soft tokens."""
        torch = self.be.torch
        if through_protocol:
            envelopes = latents_to_envelopes(
                message,
                self.store,
                "extractor",
                "normalizer",
                wire_dtype=wire_dtype,
                layer=self.layer,
                pooling=self.proj.pooling,
            )
            message = envelopes_to_latents(envelopes, self.store, "normalizer")
            for env in envelopes:
                self.store.delete(env.payload)
        else:
            message = through_wire(message, wire_dtype)
        if ablation is not None:
            name = {
                "zero": "zero",
                "shuffle": "shuffle_batch",
                "noise": "random_matched",
            }[ablation]
            message = get_intervention(name)(message, rng or np.random.default_rng(0))
        with torch.no_grad():
            return self.proj.decode(torch.from_numpy(message).to(self.be.device))

    def nll(
        self,
        records: Sequence[Record],
        *,
        ablation: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        message = self.message([r.text for r in records]).cpu().numpy()
        soft = self._receive(message, ablation, rng, "bfloat16", False)
        inputs = self.be.build_inputs(
            [LATENT_TAIL] * len(records),
            [canonical_text(r.gold) for r in records],
            soft=soft,
            soft_head=self.settings.soft_head,
        )
        return self._loss(self.b, inputs)

    def respond(
        self,
        records: Sequence[Record],
        *,
        ablation: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        wire_dtype: str = "bfloat16",
        through_protocol: bool = False,
    ) -> Response:
        t0 = _sync_time(self.be)
        message = self.message([r.text for r in records]).cpu().numpy()
        t1 = _sync_time(self.be)
        k, dim = message.shape[1:]
        soft = self._receive(message, ablation, rng, wire_dtype, through_protocol)
        t2 = _sync_time(self.be)
        got = self._gen(
            [LATENT_TAIL] * len(records),
            self.b,
            soft=soft,
            soft_head=self.settings.soft_head,
        )
        t3 = _sync_time(self.be)
        return Response(
            outputs=[g.text for g in got],
            timings={"sender": t1 - t0, "handoff": t2 - t1, "receiver": t3 - t2},
            message_bytes=[wire_nbytes(k, dim, wire_dtype)] * len(records),
            message_tokens=[k] * len(records),
        )


class KVPipeline(Pipeline):
    """B attends to A's KV cache of the snippet."""

    kind = "kv"

    def __init__(self, backend: HFPeftBackend, settings: Settings, a: str, b: str):
        super().__init__(backend, settings)
        self.a, self.b = a, b

    def ablate(
        self, prefix: KVPrefix, ablation: Optional[str], rng: np.random.Generator
    ) -> KVPrefix:
        """Zero, shuffle across the batch, or replace by matched Gaussian noise."""
        torch = self.be.torch
        if ablation is None:
            return prefix
        if ablation == "zero":
            return self.be.map_kv(prefix, lambda layer, kind, t: torch.zeros_like(t))
        if ablation == "shuffle":
            idx = torch.from_numpy(derangement(prefix.mask.shape[0], rng)).to(
                self.be.device
            )
            pairs = [(k[idx], v[idx]) for k, v in self.be.kv_pairs(prefix.cache)]
            return KVPrefix(
                cache=self.be.cache_from_pairs(pairs),
                mask=prefix.mask[idx],
                lengths=prefix.lengths[idx],
            )
        if ablation == "noise":
            gen = torch.Generator(device=self.be.device)
            gen.manual_seed(int(rng.integers(2**31)))

            def noise(layer: int, kind: str, t: Any) -> Any:
                z = torch.randn(
                    t.shape, generator=gen, device=t.device, dtype=torch.float32
                )
                return (t.float().mean() + t.float().std() * z).to(t.dtype)

            return self.be.map_kv(prefix, noise)
        raise ValueError(f"unknown ablation {ablation!r}")

    def nll(
        self,
        records: Sequence[Record],
        *,
        ablation: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        prefix = self.be.prefill(
            [A_PROMPT.format(text=r.text) for r in records], self.a
        )
        prefix = self.ablate(prefix, ablation, rng or np.random.default_rng(0))
        inputs = self.be.build_inputs(
            [RECEIVER_TAIL] * len(records),
            [canonical_text(r.gold) for r in records],
            prefix=prefix,
            clone_prefix=False,
        )
        return self._loss(self.b, inputs)

    def respond(
        self,
        records: Sequence[Record],
        *,
        ablation: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        wire_dtype: str = "bfloat16",
        through_protocol: bool = False,
    ) -> Response:
        t0 = _sync_time(self.be)
        prefix = self.be.prefill(
            [A_PROMPT.format(text=r.text) for r in records], self.a
        )
        t1 = _sync_time(self.be)
        lengths = prefix.lengths.cpu().tolist()
        prefix = self.ablate(prefix, ablation, rng or np.random.default_rng(0))
        t2 = _sync_time(self.be)
        got = self._gen(
            [RECEIVER_TAIL] * len(records), self.b, prefix=prefix, clone_prefix=False
        )
        t3 = _sync_time(self.be)
        return Response(
            outputs=[g.text for g in got],
            timings={"sender": t1 - t0, "handoff": t2 - t1, "receiver": t3 - t2},
            message_bytes=[n * self.be.kv_bytes_per_token for n in lengths],
            message_tokens=lengths,
        )


class SinglePipeline(Pipeline):
    """One LoRA reads the snippet and writes canonical JSON."""

    kind = "single"

    def __init__(self, backend: HFPeftBackend, settings: Settings, adapter: str):
        super().__init__(backend, settings)
        self.adapter = adapter

    def respond(
        self,
        records: Sequence[Record],
        *,
        ablation: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        wire_dtype: str = "bfloat16",
        through_protocol: bool = False,
    ) -> Response:
        if ablation is not None:
            raise ValueError("the single model has no message to ablate")
        t0 = _sync_time(self.be)
        got = self._gen(
            [SINGLE_PROMPT.format(text=r.text) for r in records], self.adapter
        )
        t1 = _sync_time(self.be)
        return Response(
            outputs=[g.text for g in got],
            timings={"sender": 0.0, "handoff": 0.0, "receiver": t1 - t0},
            message_bytes=[0] * len(records),
            message_tokens=[0] * len(records),
        )


# -------------------------------------------------------------------------- measurement


def evaluate(
    pipeline: Pipeline,
    splits: dict[str, Sequence[Record]],
    *,
    batch_size: int = 64,
    ablation: Optional[str] = None,
    wire_dtype: str = "bfloat16",
    seed: int = 0,
    keep_examples: int = 0,
) -> dict[str, Any]:
    """Accuracy and message sizes of a pipeline on each split.

    Returns:
        ``{split: {n, exact, valid, fields, message_bytes, message_tokens,
        stage_a?, examples?}}``. ``stage_a`` (JSON pipeline) scores the extractor's raw
        fields against the snippet's gold raw fields.
    """
    result: dict[str, Any] = {}
    for split, records in splits.items():
        scores, a_scores, nbytes, ntokens, examples = [], [], [], [], []
        rng = np.random.default_rng([seed, 17])
        for batch in _batches(list(records), batch_size):
            if ablation == "shuffle" and len(batch) < 2:
                continue
            resp = pipeline.respond(
                batch, ablation=ablation, rng=rng, wire_dtype=wire_dtype
            )
            for i, (rec, out) in enumerate(zip(batch, resp.outputs)):
                scores.append(score_fields(out, rec.gold, CANON_FIELDS))
                if resp.sender_texts is not None:
                    a_scores.append(
                        score_fields(resp.sender_texts[i], rec.raw, RAW_FIELDS)
                    )
                if len(examples) < keep_examples:
                    examples.append(
                        {
                            "text": rec.text,
                            "gold": canonical_text(rec.gold),
                            "output": out,
                            "message": (
                                None
                                if resp.sender_texts is None
                                else resp.sender_texts[i]
                            ),
                        }
                    )
            nbytes.extend(resp.message_bytes)
            ntokens.extend(resp.message_tokens)
        summary = summarize_scores(scores, CANON_FIELDS)
        summary["message_bytes"] = float(np.mean(nbytes))
        summary["message_tokens"] = float(np.mean(ntokens))
        if a_scores:
            summary["stage_a"] = summarize_scores(a_scores, RAW_FIELDS)
        if keep_examples:
            summary["examples"] = examples
        result[split] = summary
    return result


def evaluate_nll(
    pipeline: Pipeline,
    splits: dict[str, Sequence[Record]],
    *,
    batch_size: int = 100,
    ablation: Optional[str] = None,
    seed: int = 0,
) -> dict[str, float]:
    """Mean teacher-forced NLL per target token on each split (see ``Pipeline.nll``)."""
    out = {}
    for split, records in splits.items():
        rng = np.random.default_rng([seed, 29])
        weighted, total = 0.0, 0
        for batch in _batches(list(records), batch_size):
            if ablation == "shuffle" and len(batch) < 2:
                continue
            weighted += pipeline.nll(batch, ablation=ablation, rng=rng) * len(batch)
            total += len(batch)
        out[split] = weighted / total
    return out


def measure_latency(
    pipeline: Pipeline,
    records: Sequence[Record],
    *,
    warmup: int = 3,
    wire_dtype: str = "bfloat16",
) -> dict[str, Any]:
    """End-to-end latency per record, one record at a time (batch size 1).

    The latent pipeline runs through the protocol (envelope plus ``TensorStore``) here,
    so the handoff includes what a real exchange would cost. Times are wall clock with a
    CUDA sync after each stage.
    """
    for rec in list(records)[:warmup]:
        pipeline.respond([rec], wire_dtype=wire_dtype, through_protocol=True)
    totals, stages = [], {"sender": [], "handoff": [], "receiver": []}
    for rec in records:
        resp = pipeline.respond([rec], wire_dtype=wire_dtype, through_protocol=True)
        totals.append(sum(resp.timings.values()))
        for name in stages:
            stages[name].append(resp.timings[name])
    return {
        "n": len(totals),
        "total_mean_s": statistics.fmean(totals),
        "total_median_s": statistics.median(totals),
        "stage_mean_s": {name: statistics.fmean(v) for name, v in stages.items()},
    }


# ------------------------------------------------------------------------------ the run


class Experiment:
    """Trains and evaluates the R9 conditions for one or more run seeds.

    Results are written to ``out_dir/results/seed{S}_{condition}.json`` and the trained
    adapters and projectors to ``out_dir/ckpt``; an existing result file is reused, so
    a killed run resumes where it stopped.
    """

    def __init__(
        self,
        backend: HFPeftBackend,
        settings: Settings,
        out_dir: Path,
        log: Callable[[str], None] = print,
    ):
        self.be = backend
        self.s = settings
        self.out = Path(out_dir)
        self.log = log
        (self.out / "results").mkdir(parents=True, exist_ok=True)
        (self.out / "ckpt").mkdir(parents=True, exist_ok=True)
        self._tests: Optional[dict[str, list[Record]]] = None

    # ------------------------------------------------------------------------- data

    @property
    def tests(self) -> dict[str, list[Record]]:
        if self._tests is None:
            self._tests = {
                split: make_records(self.s.n_test, split, TEST_SEED) for split in SPLITS
            }
        return self._tests

    def train_records(self, seed: int) -> list[Record]:
        n_steps = max(self.s.steps, self.s.latent_steps)
        return make_records(n_steps * self.s.batch_size, "train", seed)

    def _batch(self, records: Sequence[Record], step: int) -> Sequence[Record]:
        size = self.s.batch_size
        return records[step * size : (step + 1) * size]

    # --------------------------------------------------------------------- training

    def _new_adapter(self, name: str, seed: int) -> str:
        return self.be.add_adapter(
            name,
            r=self.s.lora_r,
            alpha=self.s.lora_alpha,
            seed=seed,
            activate=False,
        )

    def _fit(
        self,
        adapter: str,
        step_fn: Callable[[int], Any],
        label: str,
        extra: Sequence[Any] = (),
        long_sequences: bool = False,
        extra_adapters: Sequence[str] = (),
        steps: Optional[int] = None,
    ) -> FitStats:
        checkpoint = long_sequences and self.s.checkpoint_long
        self.be.set_gradient_checkpointing(checkpoint)
        try:
            return self.be.fit(
                adapter,
                step_fn,
                steps=self.s.steps if steps is None else steps,
                lr=self.s.lr,
                extra_parameters=extra,
                extra_lr=self.s.projector_lr,
                extra_adapters=extra_adapters,
                log_every=self.s.log_every,
                logger=self.log,
                label=label,
            )
        finally:
            self.be.set_gradient_checkpointing(False)

    def train_extractor(
        self, seed: int, records: Sequence[Record]
    ) -> tuple[str, FitStats]:
        name = f"A_s{seed}"
        self._new_adapter(name, seed * 1000 + 1)

        def step(i: int) -> Any:
            batch = self._batch(records, i)
            inputs = self.be.build_inputs(
                [A_PROMPT.format(text=r.text) for r in batch],
                [raw_json(r.raw) for r in batch],
            )
            return self.be.training_loss(inputs)

        return name, self._fit(name, step, f"[s{seed} extractor]", long_sequences=True)

    def train_json_receiver(
        self, seed: int, records: Sequence[Record]
    ) -> tuple[str, FitStats]:
        name = f"Bjson_s{seed}"
        self._new_adapter(name, seed * 1000 + 2)

        def step(i: int) -> Any:
            batch = self._batch(records, i)
            inputs = self.be.build_inputs(
                [JSON_B_PROMPT.format(message=raw_json(r.raw)) for r in batch],
                [canonical_text(r.gold) for r in batch],
            )
            return self.be.training_loss(inputs)

        return name, self._fit(name, step, f"[s{seed} json receiver]")

    def make_projector(self, cond: Condition) -> Any:
        from ..channels import LatentProjector

        return LatentProjector.for_backend(
            self.be,
            cond.k,
            pooling=cond.pooling,
            init_rms=self.be.embedding_rms() * self.s.soft_scale,
        ).to(self.be.device)

    def train_latent_receiver(
        self,
        seed: int,
        a: str,
        cond: Condition,
        records: Sequence[Record],
        init_from: Optional[str] = None,
    ) -> tuple[str, str, Any, FitStats]:
        torch = self.be.torch
        layer = self.s.layer if cond.layer is None else cond.layer
        name = f"B_{cond.name}_s{seed}"
        self._new_adapter(name, seed * 1000 + 3)
        if init_from is not None and self._ckpt(init_from).exists():
            blob = torch.load(self._ckpt(init_from), map_location="cpu")
            self.be.load_adapter_state(name, blob["state"])
        sender = a
        if cond.train_sender:  # a trainable copy of the extractor
            sender = f"Alat_{cond.name}_s{seed}"
            self._new_adapter(sender, seed * 1000 + 7)
            self.be.load_adapter_state(sender, self.be.adapter_state(a))
        torch.manual_seed(seed * 1000 + 4)
        proj = self.make_projector(cond)
        calibration = self.be.hidden_states(
            [A_PROMPT.format(text=r.text) for r in records[: self.s.n_calibration]],
            layer,
            sender,
        )
        proj.calibrate(calibration.states, calibration.mask)
        del calibration

        def step(i: int) -> Any:
            batch = self._batch(records, i)
            prompts = [A_PROMPT.format(text=r.text) for r in batch]
            hs = self.be.hidden_states(prompts, layer, sender, grad=cond.train_sender)
            message = proj.encode(hs.states, hs.mask)
            message = message.to(torch.bfloat16).float()  # the wire is bfloat16
            inputs = self.be.build_inputs(
                [LATENT_TAIL] * len(batch),
                [canonical_text(r.gold) for r in batch],
                soft=proj.decode(message),
                soft_head=self.s.soft_head,
            )
            return self.be.training_loss(inputs)

        stats = self._fit(
            name,
            step,
            f"[s{seed} {cond.name}]",
            list(proj.parameters()),
            long_sequences=cond.train_sender,
            extra_adapters=[sender] if cond.train_sender else (),
            steps=self.s.latent_steps,
        )
        proj.eval()
        return name, sender, proj, stats

    def train_kv_receiver(
        self, seed: int, a: str, records: Sequence[Record]
    ) -> tuple[str, FitStats]:
        name = f"Bkv_s{seed}"
        self._new_adapter(name, seed * 1000 + 5)

        def step(i: int) -> Any:
            batch = self._batch(records, i)
            prefix = self.be.prefill([A_PROMPT.format(text=r.text) for r in batch], a)
            inputs = self.be.build_inputs(
                [RECEIVER_TAIL] * len(batch),
                [canonical_text(r.gold) for r in batch],
                prefix=prefix,
                clone_prefix=False,
            )
            return self.be.training_loss(inputs)

        return name, self._fit(name, step, f"[s{seed} kv prefix]")

    def train_single(
        self, seed: int, records: Sequence[Record]
    ) -> tuple[str, FitStats]:
        name = f"S_s{seed}"
        self._new_adapter(name, seed * 1000 + 6)

        def step(i: int) -> Any:
            batch = self._batch(records, i)
            inputs = self.be.build_inputs(
                [SINGLE_PROMPT.format(text=r.text) for r in batch],
                [canonical_text(r.gold) for r in batch],
            )
            return self.be.training_loss(inputs)

        return name, self._fit(name, step, f"[s{seed} single]", long_sequences=True)

    # ------------------------------------------------------------------ evaluation

    def _result_path(self, seed: int, name: str) -> Path:
        return self.out / "results" / f"seed{seed}_{name}.json"

    def _save(self, path: Path, result: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(result, indent=1, default=_json_default), encoding="utf-8"
        )

    def _train_info(
        self, stats: FitStats, extra: Optional[dict[str, Any]] = None
    ) -> dict:
        info = {
            "steps": stats.steps,
            "batch_size": self.s.batch_size,
            "examples_seen": stats.steps * self.s.batch_size,
            "wall_s": stats.wall_s,
            "final_loss": stats.final_loss,
            "mean_last_loss": stats.mean_last_loss,
            "peak_memory_mib": stats.peak_memory_mib,
            "n_trainable": stats.n_trainable,
        }
        info.update(extra or {})
        return info

    def full_evaluation(
        self,
        pipeline: Pipeline,
        seed: int,
        *,
        ablations: Sequence[str] = (),
        int8: bool = False,
    ) -> dict[str, Any]:
        """Accuracy, optional ablations and int8 wire, and latency of one pipeline."""
        s = self.s
        out: dict[str, Any] = {
            "eval": evaluate(
                pipeline,
                self.tests,
                batch_size=s.eval_batch,
                seed=seed,
                keep_examples=6,
            )
        }
        small = {k: v[: s.n_ablation] for k, v in self.tests.items()}
        if ablations:
            # "intact" is the same subset without an intervention: the fair reference
            out["ablations"] = {
                name: evaluate(
                    pipeline,
                    small,
                    batch_size=s.eval_batch,
                    ablation=None if name == "intact" else name,
                    seed=seed,
                )
                for name in ("intact", *ablations)
            }
            out["nll"] = {
                name: evaluate_nll(
                    pipeline,
                    small,
                    batch_size=s.eval_batch,
                    ablation=None if name == "intact" else name,
                    seed=seed,
                )
                for name in ("intact", *ablations)
            }
        if int8:
            out["eval_int8"] = evaluate(
                pipeline,
                self.tests,
                batch_size=s.eval_batch,
                wire_dtype="int8",
                seed=seed,
            )
        if isinstance(pipeline, LatentPipeline):
            # The envelope + TensorStore path must deliver exactly the direct message
            probe = list(self.tests["heldout_test"][:8])
            direct = pipeline.respond(probe).outputs
            via_protocol = pipeline.respond(probe, through_protocol=True).outputs
            out["protocol_roundtrip_identical"] = direct == via_protocol
        timed = [r for split in SPLITS for r in self.tests[split][-s.n_latency :]]
        out["latency"] = measure_latency(pipeline, timed)
        return out

    # ------------------------------------------------------------------- one seed

    def _ckpt(self, name: str) -> Path:
        return self.out / "ckpt" / f"{name}.pt"

    def run_seed(
        self,
        seed: int,
        conditions: Sequence[Condition],
    ) -> list[dict[str, Any]]:
        """Run the given conditions for one seed (training what is not cached)."""
        s = self.s
        torch = self.be.torch
        records = self.train_records(seed)
        results: list[dict[str, Any]] = []
        todo = [c for c in conditions if not self._result_path(seed, c.name).exists()]
        for c in conditions:
            if c not in todo:
                results.append(
                    json.loads(self._result_path(seed, c.name).read_text("utf-8"))
                )
        if not todo:
            return results

        # The extractor A is shared by every condition that reads its state
        needs_a = any(c.kind in ("json", "latent", "kv") for c in todo)
        a_name, a_stats = f"A_s{seed}", None
        a_info_path = self.out / "results" / f"seed{seed}_extractor_train.json"
        if needs_a:
            if self._ckpt(a_name).exists():
                self.be.load_adapter(a_name, self._ckpt(a_name))
                a_stats = json.loads(a_info_path.read_text("utf-8"))
                self.log(f"[s{seed}] loaded extractor from checkpoint")
            else:
                a_name, stats = self.train_extractor(seed, records)
                self.be.save_adapter(a_name, self._ckpt(a_name))
                a_stats = self._train_info(stats)
                self._save(a_info_path, a_stats)

        for cond in todo:
            self.log(f"[s{seed}] === condition {cond.name} ===")
            torch.manual_seed(seed * 1000 + 9)
            extra_adapters = []
            if cond.kind == "json":
                b_name, stats = self.train_json_receiver(seed, records)
                pipe: Pipeline = JsonPipeline(self.be, s, a_name, b_name)
                ev = self.full_evaluation(pipe, seed, ablations=("shuffle",))
                train = {"receiver": self._train_info(stats), "extractor": a_stats}
                extras: dict[str, Any] = {}
                extra_adapters = [b_name]
            elif cond.kind == "latent":
                layer = s.layer if cond.layer is None else cond.layer
                b_name, sender, proj, stats = self.train_latent_receiver(
                    seed,
                    a_name,
                    cond,
                    records,
                    init_from=f"Bjson_s{seed}" if s.warm_start else None,
                )
                pipe = LatentPipeline(self.be, s, sender, b_name, proj, layer)
                ev = self.full_evaluation(pipe, seed, ablations=ABLATIONS, int8=True)
                train = {"receiver": self._train_info(stats), "extractor": a_stats}
                extras = {
                    "layer": layer,
                    "pooling": cond.pooling,
                    "k": cond.k,
                    "projector_params": sum(p.numel() for p in proj.parameters()),
                }
                extra_adapters = [b_name]
                if sender != a_name:
                    self.be.save_adapter(sender, self._ckpt(sender))
                    extra_adapters.append(sender)
                torch.save(proj.state_dict(), self._ckpt(f"proj_{cond.name}_s{seed}"))
            elif cond.kind == "kv":
                b_name, stats = self.train_kv_receiver(seed, a_name, records)
                pipe = KVPipeline(self.be, s, a_name, b_name)
                ev = self.full_evaluation(pipe, seed, ablations=ABLATIONS)
                train = {"receiver": self._train_info(stats), "extractor": a_stats}
                extras = {"kv_bytes_per_token": self.be.kv_bytes_per_token}
                extra_adapters = [b_name]
            elif cond.kind == "single":
                b_name, stats = self.train_single(seed, records)
                pipe = SinglePipeline(self.be, s, b_name)
                ev = self.full_evaluation(pipe, seed)
                train = {"receiver": self._train_info(stats)}
                extras = {}
                extra_adapters = [b_name]
            else:
                raise ValueError(f"unknown condition kind {cond.kind!r}")
            if cond.kind != "single":
                self.be.save_adapter(b_name, self._ckpt(f"{b_name}"))
            result = {
                "seed": seed,
                "condition": cond.name,
                "kind": cond.kind,
                "settings": asdict(s),
                "train": train,
                **extras,
                **ev,
            }
            self._save(self._result_path(seed, cond.name), result)
            results.append(result)
            self._log_result(result)
            for name in extra_adapters:
                self.be.delete_adapter(name)
        if needs_a and a_name in self.be.adapters:
            self.be.delete_adapter(a_name)
        return results

    def _log_result(self, result: dict[str, Any]) -> None:
        ev = result["eval"]
        self.log(
            f"[s{result['seed']}] {result['condition']}: "
            f"ID exact {ev['id_test']['exact']:.3f}  "
            f"held-out exact {ev['heldout_test']['exact']:.3f}  "
            f"latency {result['latency']['total_mean_s'] * 1000:.0f} ms  "
            f"train {result['train']['receiver']['wall_s']:.0f}s"
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialise {type(value).__name__}")
