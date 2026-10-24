"""Channels: how specialists that share a base model exchange internal state.

- ``latent``: hidden-state messages. Wire helpers (numpy) wrap them as protocol
  ``latent_ref`` envelopes in a ``TensorStore``; the projector (torch, built on first
  use) turns them into soft tokens for the receiver.

Importing this package needs no torch.
"""

from typing import Any

from .latent import (
    LATENT_CODEC,
    POOLING_MODES,
    WIRE_DTYPES,
    dequantize_int8,
    envelopes_to_latents,
    latent_encoding,
    latents_to_envelopes,
    message_wire_nbytes,
    quantize_int8,
    round_to_bfloat16,
    through_wire,
    wire_nbytes,
)

__all__ = [
    "LATENT_CODEC",
    "POOLING_MODES",
    "WIRE_DTYPES",
    "LatentProjector",
    "QueryPool",
    "dequantize_int8",
    "envelopes_to_latents",
    "latent_encoding",
    "latents_to_envelopes",
    "message_wire_nbytes",
    "quantize_int8",
    "round_to_bfloat16",
    "through_wire",
    "wire_nbytes",
]


def __getattr__(name: str) -> Any:
    # LatentProjector and QueryPool need torch: resolve them on first access
    if name in ("LatentProjector", "QueryPool"):
        from . import latent

        return getattr(latent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
