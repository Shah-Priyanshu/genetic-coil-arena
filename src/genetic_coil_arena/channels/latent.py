"""Latent channel: hidden-state messages between specialists that share a base model.

Two halves:

- **Wire helpers** (numpy only). A latent message is a ``(k, D)`` block of ``k`` vectors
  of size ``D`` (hidden states, or pooled summaries of them). It travels as a
  ``latent_ref`` envelope: the tensor sits in a protocol ``TensorStore`` and the
  envelope carries the reference plus a ``latent_tokens`` payload encoding (``k``,
  ``D``, the wire dtype, the byte size). ``wire_dtype`` is what the message would cost
  on a real wire (``float32``, ``bfloat16`` or ``int8`` with a float16 scale per
  vector); :func:`through_wire` applies that precision, so the receiver sees exactly
  what the wire would deliver. ``latents_to_envelopes`` / ``envelopes_to_latents`` are
  inverses.
- **The projector** (torch, built on first use). Raw residual-stream states are 76-100x
  larger than token embeddings and shift the receiver's next-token distribution
  violently, so a message is never injected as it is. They are also dominated by a few
  "massive" dimensions (hundreds of times the typical coordinate): after a plain RMS
  norm those drown the content, and a receiver trained on them learns nothing in
  hundreds of steps. ``LatentProjector`` therefore first standardizes each coordinate
  with statistics measured on the sender (``calibrate``), then turns the message into
  ``k`` soft tokens of embedding scale: ``MLP -> + position -> RMSNorm`` with a learned
  gain that starts at the embedding scale. ``pooling="last"`` takes the sender's last
  ``k`` positions as the message (no sender-side parameters beyond the calibration);
  ``pooling="attn"`` lets ``k`` learned queries attend over every position (a
  Perceiver-style bottleneck) and sends the ``k`` pooled vectors. The standardized
  message is what travels, so int8 quantization has a sane range.

torch is imported only when ``LatentProjector`` or ``QueryPool`` is first touched, so
the wire helpers work without it.
"""

import functools
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from ..protocol import CODEC_VERSION, Envelope, EnvelopeError, TensorStore

LATENT_CODEC = "latent_tokens"
WIRE_DTYPES = ("float32", "bfloat16", "int8")
POOLING_MODES = ("last", "attn")


# ------------------------------------------------------------------------ wire formats


def _check_wire_dtype(wire_dtype: str) -> None:
    if wire_dtype not in WIRE_DTYPES:
        raise ValueError(f"wire_dtype must be one of {WIRE_DTYPES}, got {wire_dtype!r}")


def wire_nbytes(k: int, dim: int, wire_dtype: str = "bfloat16") -> int:
    """Bytes of a ``(k, dim)`` latent message on the wire.

    ``bfloat16`` costs 2 bytes per element, ``float32`` 4, and ``int8`` 1 plus one
    float16 scale per vector (2 bytes).
    """
    _check_wire_dtype(wire_dtype)
    if wire_dtype == "float32":
        return 4 * k * dim
    if wire_dtype == "bfloat16":
        return 2 * k * dim
    return k * dim + 2 * k


def round_to_bfloat16(values: np.ndarray) -> np.ndarray:
    """Round float values to the nearest bfloat16 (ties to even), returned as float32."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounding = ((bits >> 16) & 1) + np.uint32(0x7FFF)
    return ((bits + rounding) & np.uint32(0xFFFF0000)).view(np.float32)


def quantize_int8(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 quantization with one scale per vector (last axis).

    Returns:
        ``(codes, scales)``: int8 array of the input's shape, and float16 scales of
        shape ``values.shape[:-1]``. An all-zero vector gets scale 1.
    """
    arr = np.asarray(values, dtype=np.float32)
    peak = np.abs(arr).max(axis=-1)
    scales = np.where(peak > 0, peak / 127.0, 1.0).astype(np.float16)
    codes = np.rint(arr / scales.astype(np.float32)[..., None])
    return np.clip(codes, -127, 127).astype(np.int8), scales


def dequantize_int8(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize_int8` (up to quantization error), as float32."""
    return codes.astype(np.float32) * scales.astype(np.float32)[..., None]


def through_wire(values: np.ndarray, wire_dtype: str = "bfloat16") -> np.ndarray:
    """What the receiver sees after sending ``values`` at ``wire_dtype`` (as float32)."""
    _check_wire_dtype(wire_dtype)
    arr = np.asarray(values, dtype=np.float32)
    if wire_dtype == "float32":
        return arr
    if wire_dtype == "bfloat16":
        return round_to_bfloat16(arr)
    return dequantize_int8(*quantize_int8(arr))


# ----------------------------------------------------------------------------- envelopes


def latent_encoding(
    k: int,
    dim: int,
    wire_dtype: str = "bfloat16",
    layer: Optional[int] = None,
    pooling: Optional[str] = None,
) -> dict[str, Any]:
    """``payload_encoding`` of a latent message of ``k`` vectors of size ``dim``.

    ``dtype`` is how the tensor sits in the store (always float32); ``wire_dtype`` and
    ``wire_bytes`` say what it costs on a real wire. ``layer`` and ``pooling`` record
    where in the sender the message was taken.
    """
    _check_wire_dtype(wire_dtype)
    encoding: dict[str, Any] = {
        "codec": LATENT_CODEC,
        "codec_version": CODEC_VERSION,
        "k": int(k),
        "D": int(dim),
        "dtype": "float32",
        "wire_dtype": wire_dtype,
        "wire_bytes": wire_nbytes(k, dim, wire_dtype),
    }
    if layer is not None:
        encoding["layer"] = int(layer)
    if pooling is not None:
        encoding["pooling"] = pooling
    return encoding


def latents_to_envelopes(
    messages: np.ndarray,
    store: TensorStore,
    sender: str,
    recipient: str,
    *,
    wire_dtype: str = "bfloat16",
    layer: Optional[int] = None,
    pooling: Optional[str] = None,
    task_ids: Optional[Sequence[Optional[str]]] = None,
    trace_ids: Optional[Sequence[str]] = None,
    parent_ids: Optional[Sequence[Optional[str]]] = None,
    clock: int = 0,
) -> list[Envelope]:
    """Store each ``(k, D)`` message and wrap its reference in a ``latent_ref`` envelope.

    The stored tensor is the message after :func:`through_wire`, so what the receiver
    reads is what the wire would carry.

    Args:
        messages: Float array ``(B, k, D)``, one message per episode
        store: Tensor store that keeps the messages
        sender: Sending party
        recipient: The single receiving party
        wire_dtype: ``float32``, ``bfloat16`` or ``int8``
        layer, pooling: Provenance recorded in the encoding
        task_ids, trace_ids, parent_ids: Optional per-episode ids (default: none, none,
            and a fresh trace id)
        clock: Logical time stamped on every envelope

    Returns:
        One envelope per episode
    """
    arr = np.asarray(messages)
    if arr.ndim != 3:
        raise ValueError(f"messages must have shape (B, k, D), got {arr.shape}")
    if arr.size and not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"messages must be floating point, got dtype {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("messages must be finite")
    if not isinstance(recipient, str):
        raise TypeError("recipient must be a single name")
    batch, k, dim = arr.shape
    for name, ids in (
        ("task_ids", task_ids),
        ("trace_ids", trace_ids),
        ("parent_ids", parent_ids),
    ):
        if ids is not None and len(ids) != batch:
            raise ValueError(f"{name} has {len(ids)} entries, expected {batch}")

    wire = through_wire(arr, wire_dtype)
    encoding = latent_encoding(k, dim, wire_dtype, layer, pooling)
    envelopes = []
    for b in range(batch):
        extra: dict[str, Any] = {}
        if trace_ids is not None:
            extra["trace_id"] = trace_ids[b]
        envelopes.append(
            Envelope(
                sender=sender,
                recipients=(recipient,),
                payload_type="latent_ref",
                payload=store.put(wire[b]),
                payload_encoding=dict(encoding),
                task_id=None if task_ids is None else task_ids[b],
                parent_id=None if parent_ids is None else parent_ids[b],
                clock=clock,
                **extra,
            )
        )
    return envelopes


def _check_latent_envelope(envelope: Envelope) -> dict[str, Any]:
    if envelope.payload_type != "latent_ref":
        raise EnvelopeError(
            f"Envelope {envelope.msg_id} has payload_type {envelope.payload_type!r}, "
            "expected 'latent_ref'"
        )
    encoding = envelope.payload_encoding
    if encoding.get("codec") != LATENT_CODEC:
        raise EnvelopeError(
            f"Envelope {envelope.msg_id} uses codec {encoding.get('codec')!r}, "
            f"expected {LATENT_CODEC!r}"
        )
    if encoding.get("codec_version") != CODEC_VERSION:
        raise EnvelopeError(
            f"Unsupported {LATENT_CODEC} codec version {encoding.get('codec_version')!r}"
        )
    return encoding


def envelopes_to_latents(
    envelopes: Sequence[Envelope],
    store: TensorStore,
    recipient: Optional[str] = None,
    verify: bool = False,
) -> np.ndarray:
    """Inverse of :func:`latents_to_envelopes`.

    Args:
        envelopes: One ``latent_ref`` envelope per episode
        store: Tensor store holding the referenced messages
        recipient: If given, check every envelope is addressed to it alone
        verify: Re-hash each tensor and compare it to its reference digest

    Returns:
        float32 array ``(B, k, D)``

    Raises:
        EnvelopeError: Wrong payload type or codec, mixed shapes, or wrong addressing
    """
    if not envelopes:
        raise ValueError("need at least one envelope")
    first = _check_latent_envelope(envelopes[0])
    k, dim = first["k"], first["D"]
    out = np.zeros((len(envelopes), k, dim), dtype=np.float32)
    for b, envelope in enumerate(envelopes):
        encoding = _check_latent_envelope(envelope)
        if (encoding["k"], encoding["D"]) != (k, dim):
            raise EnvelopeError("all envelopes in a batch must share k and D")
        if recipient is not None and envelope.recipients != (recipient,):
            raise EnvelopeError(
                f"Envelope {envelope.msg_id} is addressed to {envelope.recipients}, "
                f"expected ({recipient!r},)"
            )
        tensor = store.get(envelope.payload, verify=verify)
        if tensor.shape != (k, dim):
            raise EnvelopeError(
                f"Referenced tensor has shape {tensor.shape}, expected {(k, dim)}"
            )
        out[b] = tensor
    return out


def message_wire_nbytes(envelope: Envelope) -> int:
    """Bytes a latent envelope's tensor costs on the wire (the header is extra)."""
    return int(_check_latent_envelope(envelope)["wire_bytes"])


# --------------------------------------------------------------------------- projector


@functools.cache
def _torch_classes() -> dict[str, Any]:
    """Define the torch modules (on first use, so importing this file needs no torch)."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class QueryPool(nn.Module):
        """Sender side: ``k`` learned queries attend over a sequence of hidden states.

        Multi-head cross-attention with learned queries; padded positions are masked.
        The ``k`` pooled vectors have the sender's width, so a message still costs
        ``k * d_in`` numbers on the wire.

        Args:
            d_in: Width of the hidden states
            k: Number of queries (message vectors)
            n_heads: Attention heads
            d_attn: Attention width (a multiple of ``n_heads``)
        """

        def __init__(self, d_in: int, k: int, n_heads: int = 8, d_attn: int = 1024):
            super().__init__()
            if d_attn % n_heads:
                raise ValueError("d_attn must be a multiple of n_heads")
            self.k, self.n_heads, self.d_attn = k, n_heads, d_attn
            self.norm = nn.RMSNorm(d_in, eps=1e-6)
            self.queries = nn.Parameter(torch.randn(k, d_attn) * d_attn**-0.5)
            self.key = nn.Linear(d_in, d_attn, bias=False)
            self.value = nn.Linear(d_in, d_attn, bias=False)
            self.out = nn.Linear(d_attn, d_in, bias=False)

        def forward(self, states: Any, mask: Optional[Any] = None) -> Any:
            batch, length, _ = states.shape
            heads, head_dim = self.n_heads, self.d_attn // self.n_heads
            x = self.norm(states.float())
            keys = self.key(x).view(batch, length, heads, head_dim).transpose(1, 2)
            values = self.value(x).view(batch, length, heads, head_dim).transpose(1, 2)
            queries = (
                self.queries.view(1, self.k, heads, head_dim)
                .transpose(1, 2)
                .expand(batch, -1, -1, -1)
            )
            attn_mask = None if mask is None else mask[:, None, None, :].bool()
            pooled = F.scaled_dot_product_attention(
                queries, keys, values, attn_mask=attn_mask
            )
            return self.out(pooled.transpose(1, 2).reshape(batch, self.k, self.d_attn))

    class LatentProjector(nn.Module):
        """Turns a latent message into ``k`` soft tokens of embedding scale.

        ``encode`` is the sender's half (choose the message from hidden states),
        ``decode`` the receiver's (message to soft tokens); ``forward`` is both.

        Args:
            d_in: Width of the sender's hidden states
            d_out: Width of the receiver's embeddings
            k: Number of message vectors / soft tokens
            pooling: ``"last"`` (the sender's last ``k`` positions) or ``"attn"``
                (``k`` learned queries over all positions)
            d_hidden: MLP width (default ``d_out``)
            init_rms: Initial root-mean-square of a soft-token element; use the
                backend's ``embedding_rms()`` (about 0.033 for Qwen3-1.7B)
            n_heads, d_attn: Shape of the attention pool (``pooling="attn"``)
            clip: Standardized coordinates are clamped to ``[-clip, clip]``
        """

        def __init__(
            self,
            d_in: int,
            d_out: int,
            k: int,
            *,
            pooling: str = "last",
            d_hidden: Optional[int] = None,
            init_rms: float = 0.033,
            n_heads: int = 8,
            d_attn: int = 1024,
            clip: float = 8.0,
        ):
            super().__init__()
            if pooling not in POOLING_MODES:
                raise ValueError(
                    f"pooling must be one of {POOLING_MODES}, got {pooling!r}"
                )
            if k < 1:
                raise ValueError("k must be at least 1")
            self.d_in, self.d_out, self.k, self.pooling = d_in, d_out, k, pooling
            self.clip = float(clip)
            self.register_buffer("norm_mean", torch.zeros(d_in))
            self.register_buffer("norm_scale", torch.ones(d_in))
            self.pool = (
                QueryPool(d_in, k, n_heads, d_attn) if pooling == "attn" else None
            )
            hidden = d_hidden or d_out
            self.norm = nn.RMSNorm(d_in, eps=1e-6)
            self.mlp = nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out)
            )
            self.position = nn.Parameter(torch.randn(k, d_out) * 0.02)
            self.gain = nn.Parameter(torch.full((d_out,), float(init_rms)))

        @classmethod
        def for_backend(cls, backend: Any, k: int, **kwargs: Any) -> "LatentProjector":
            """A projector sized for ``backend`` (same width in and out, embedding scale)."""
            kwargs.setdefault("init_rms", backend.embedding_rms())
            return cls(backend.hidden_size, backend.hidden_size, k, **kwargs)

        def standardize(self, states: Any) -> Any:
            """Per-coordinate z-score with the calibrated statistics, clamped."""
            z = (states.float() - self.norm_mean) * self.norm_scale
            return z.clamp(-self.clip, self.clip)

        @torch.no_grad()
        def calibrate(self, states: Any, mask: Optional[Any] = None) -> None:
            """Measure the sender's per-coordinate mean and spread.

            With ``pooling="last"`` only the last ``k`` positions are used (the ones that
            are sent); with ``"attn"`` every real position except each row's first,
            which is the attention sink with its huge activations.

            Args:
                states: Hidden states ``[B, L, d_in]``, left-padded
                mask: ``[B, L]``, 1 for real positions
            """
            x = states.float()
            if self.pool is None:
                x = x[:, -self.k :].reshape(-1, x.shape[-1])
            else:
                real = (
                    torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
                    if mask is None
                    else mask.bool().clone()
                )
                first = real.long().argmax(-1)
                real[torch.arange(x.shape[0], device=x.device), first] = False
                x = x[real]
            self.norm_mean.copy_(x.mean(0))
            self.norm_scale.copy_(1.0 / (x.std(0) + 1e-3))

        def encode(self, states: Any, mask: Optional[Any] = None) -> Any:
            """The message ``[B, k, d_in]`` chosen from hidden states ``[B, L, d_in]``.

            The message is standardized (see :meth:`calibrate`). ``states`` must be
            left-padded, so the last positions are real.
            """
            if self.pool is not None:
                return self.pool(self.standardize(states), mask)
            if states.shape[1] < self.k:
                raise ValueError(
                    f"need at least k={self.k} positions, got {states.shape[1]}"
                )
            return self.standardize(states[:, -self.k :])

        def decode(self, message: Any) -> Any:
            """Soft tokens ``[B, k, d_out]`` (float32) from a message ``[B, k, d_in]``."""
            hidden = self.mlp(self.norm(message.float())) + self.position
            return F.rms_norm(hidden, (self.d_out,), weight=self.gain, eps=1e-6)

        def forward(self, states: Any, mask: Optional[Any] = None) -> Any:
            return self.decode(self.encode(states, mask))

    for cls in (QueryPool, LatentProjector):
        # so pickle / torch.save can find them again through the module's __getattr__
        cls.__module__ = __name__
        cls.__qualname__ = cls.__name__
    return {"QueryPool": QueryPool, "LatentProjector": LatentProjector}


def __getattr__(name: str) -> Any:
    if name in ("QueryPool", "LatentProjector"):
        return _torch_classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
