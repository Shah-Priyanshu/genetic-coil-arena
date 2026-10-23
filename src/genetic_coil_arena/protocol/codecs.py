"""Converters between the batched array fast path and envelopes.

Training loops move messages as arrays; orchestration, logging and real models use
envelopes. These functions are the explicit bridge. Array conventions (shared by all
packages that produce or consume messages):

- discrete messages: int64 array of shape (B, R, L) with values in [0, V)
- continuous messages: float32 array of shape (B, R, D)

where B = parallel episodes, R = receivers, L = symbols per message, D = vector size.

Envelope batches are nested lists ``[B][R]``: one envelope per (episode, receiver),
addressed to that receiver alone. Every envelope of episode ``b`` shares its
``task_id``/``trace_id``/``parent_id`` entry.
"""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

from .envelope import Envelope, new_id
from .tensor_store import TensorStore

CODEC_VERSION = 1
CODES_CODEC = "codes"
VECTOR_CODEC = "vector"

EnvelopeBatch = list[list[Envelope]]


def codes_encoding(vocab_size: int, length: int) -> dict[str, Any]:
    """payload_encoding for discrete messages of ``length`` symbols from a vocabulary of ``vocab_size``."""
    return {
        "codec": CODES_CODEC,
        "codec_version": CODEC_VERSION,
        "V": int(vocab_size),
        "L": int(length),
    }


def vector_encoding(dim: int) -> dict[str, Any]:
    """payload_encoding for continuous messages of dimension ``dim``."""
    return {
        "codec": VECTOR_CODEC,
        "codec_version": CODEC_VERSION,
        "D": int(dim),
        "dtype": "float32",
    }


def _per_episode(
    values: Optional[Sequence[Optional[str]]], batch_size: int, name: str, default: Any
) -> list[Optional[str]]:
    if values is None:
        return [default() if callable(default) else default for _ in range(batch_size)]
    values = list(values)
    if len(values) != batch_size:
        raise ValueError(
            f"{name} has {len(values)} entries, expected one per episode ({batch_size})"
        )
    return values


def _check_recipients(recipients: Sequence[str], num_receivers: int) -> list[str]:
    recipients = list(recipients)
    if len(recipients) != num_receivers:
        raise ValueError(
            f"Got {len(recipients)} recipients for {num_receivers} receivers"
        )
    if len(set(recipients)) != len(recipients):
        raise ValueError("recipients must be unique")
    return recipients


def codes_to_envelopes(
    codes: np.ndarray,
    vocab_size: int,
    sender: str,
    recipients: Sequence[str],
    task_ids: Optional[Sequence[Optional[str]]] = None,
    trace_ids: Optional[Sequence[str]] = None,
    parent_ids: Optional[Sequence[Optional[str]]] = None,
    clock: int = 0,
) -> EnvelopeBatch:
    """Turn a (B, R, L) array of symbols into a [B][R] batch of ``codes`` envelopes.

    Args:
        codes: Integer array of shape (B, R, L) with values in [0, vocab_size)
        vocab_size: Vocabulary size V
        sender: Sending party
        recipients: Receiver names, one per R
        task_ids: Optional task id per episode
        trace_ids: Optional trace id per episode (default: a new id per episode)
        parent_ids: Optional parent message id per episode
        clock: Logical time stamped on every envelope

    Returns:
        Nested list of envelopes, indexed [episode][receiver]
    """
    arr = np.asarray(codes)
    if arr.ndim != 3:
        raise ValueError(f"codes must have shape (B, R, L), got {arr.shape}")
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"codes must be integers, got dtype {arr.dtype}")
    if vocab_size < 1:
        raise ValueError("vocab_size must be at least 1")
    if arr.size and (arr.min() < 0 or arr.max() >= vocab_size):
        raise ValueError(f"codes must lie in [0, {vocab_size})")

    batch_size, num_receivers, length = arr.shape
    names = _check_recipients(recipients, num_receivers)
    tasks = _per_episode(task_ids, batch_size, "task_ids", None)
    traces = [
        t or new_id() for t in _per_episode(trace_ids, batch_size, "trace_ids", None)
    ]
    parents = _per_episode(parent_ids, batch_size, "parent_ids", None)
    encoding = codes_encoding(vocab_size, length)
    rows = arr.astype(np.int64).tolist()

    return [
        [
            Envelope(
                sender=sender,
                recipients=(names[r],),
                payload_type="codes",
                payload=tuple(rows[b][r]),
                payload_encoding=dict(encoding),
                task_id=tasks[b],
                trace_id=traces[b],
                parent_id=parents[b],
                clock=clock,
            )
            for r in range(num_receivers)
        ]
        for b in range(batch_size)
    ]


def _check_batch(
    envelopes: Sequence[Sequence[Envelope]], recipients: Optional[Sequence[str]]
) -> int:
    if not envelopes or not envelopes[0]:
        raise ValueError(
            "envelope batch must have at least one episode and one receiver"
        )
    num_receivers = len(envelopes[0])
    if any(len(row) != num_receivers for row in envelopes):
        raise ValueError("every episode must have the same number of receivers")
    if recipients is not None:
        recipients = _check_recipients(recipients, num_receivers)
        for row in envelopes:
            for r, envelope in enumerate(row):
                if envelope.recipients != (recipients[r],):
                    raise ValueError(
                        f"Envelope {envelope.msg_id} is addressed to "
                        f"{envelope.recipients}, expected ({recipients[r]!r},)"
                    )
    return num_receivers


def _check_encoding(envelope: Envelope, payload_type: str, codec: str) -> None:
    if envelope.payload_type != payload_type:
        raise ValueError(
            f"Envelope {envelope.msg_id} has payload_type {envelope.payload_type!r}, "
            f"expected {payload_type!r}"
        )
    encoding = envelope.payload_encoding
    if encoding.get("codec") != codec:
        raise ValueError(
            f"Envelope {envelope.msg_id} uses codec {encoding.get('codec')!r}, expected {codec!r}"
        )
    if encoding.get("codec_version") != CODEC_VERSION:
        raise ValueError(
            f"Unsupported {codec} codec version {encoding.get('codec_version')!r}"
        )


def envelopes_to_codes(
    envelopes: Sequence[Sequence[Envelope]], recipients: Optional[Sequence[str]] = None
) -> np.ndarray:
    """Inverse of :func:`codes_to_envelopes`.

    Args:
        envelopes: Batch indexed [episode][receiver]
        recipients: If given, check that envelope ``[b][r]`` is addressed to ``recipients[r]``

    Returns:
        int64 array of shape (B, R, L)
    """
    num_receivers = _check_batch(envelopes, recipients)
    first = envelopes[0][0]
    _check_encoding(first, "codes", CODES_CODEC)
    vocab_size, length = first.payload_encoding["V"], first.payload_encoding["L"]

    codes = np.zeros((len(envelopes), num_receivers, length), dtype=np.int64)
    for b, row in enumerate(envelopes):
        for r, envelope in enumerate(row):
            _check_encoding(envelope, "codes", CODES_CODEC)
            if (
                envelope.payload_encoding["V"] != vocab_size
                or envelope.payload_encoding["L"] != length
            ):
                raise ValueError("all envelopes in a batch must share V and L")
            codes[b, r] = envelope.payload
    return codes


def vectors_to_envelopes(
    vectors: np.ndarray,
    store: TensorStore,
    sender: str,
    recipients: Sequence[str],
    task_ids: Optional[Sequence[Optional[str]]] = None,
    trace_ids: Optional[Sequence[str]] = None,
    parent_ids: Optional[Sequence[Optional[str]]] = None,
    clock: int = 0,
) -> EnvelopeBatch:
    """Store each (B, R, D) message vector and wrap its reference in a ``latent_ref`` envelope.

    Args:
        vectors: Float array of shape (B, R, D); stored as float32
        store: Tensor store that keeps the vectors
        sender, recipients, task_ids, trace_ids, parent_ids, clock: As in codes_to_envelopes

    Returns:
        Nested list of envelopes, indexed [episode][receiver]
    """
    arr = np.asarray(vectors)
    if arr.ndim != 3:
        raise ValueError(f"vectors must have shape (B, R, D), got {arr.shape}")
    if arr.size and not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"vectors must be floating point, got dtype {arr.dtype}")
    arr = arr.astype(np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError("vectors must be finite")

    batch_size, num_receivers, dim = arr.shape
    names = _check_recipients(recipients, num_receivers)
    tasks = _per_episode(task_ids, batch_size, "task_ids", None)
    traces = [
        t or new_id() for t in _per_episode(trace_ids, batch_size, "trace_ids", None)
    ]
    parents = _per_episode(parent_ids, batch_size, "parent_ids", None)
    encoding = vector_encoding(dim)

    return [
        [
            Envelope(
                sender=sender,
                recipients=(names[r],),
                payload_type="latent_ref",
                payload=store.put(arr[b, r]),
                payload_encoding=dict(encoding),
                task_id=tasks[b],
                trace_id=traces[b],
                parent_id=parents[b],
                clock=clock,
            )
            for r in range(num_receivers)
        ]
        for b in range(batch_size)
    ]


def envelopes_to_vectors(
    envelopes: Sequence[Sequence[Envelope]],
    store: TensorStore,
    recipients: Optional[Sequence[str]] = None,
    verify: bool = False,
) -> np.ndarray:
    """Inverse of :func:`vectors_to_envelopes`.

    Args:
        envelopes: Batch indexed [episode][receiver]
        store: Tensor store holding the referenced vectors
        recipients: If given, check addressing as in envelopes_to_codes
        verify: Re-hash each tensor and compare it to its reference digest

    Returns:
        float32 array of shape (B, R, D)
    """
    num_receivers = _check_batch(envelopes, recipients)
    first = envelopes[0][0]
    _check_encoding(first, "latent_ref", VECTOR_CODEC)
    dim = first.payload_encoding["D"]

    vectors = np.zeros((len(envelopes), num_receivers, dim), dtype=np.float32)
    for b, row in enumerate(envelopes):
        for r, envelope in enumerate(row):
            _check_encoding(envelope, "latent_ref", VECTOR_CODEC)
            if envelope.payload_encoding["D"] != dim:
                raise ValueError("all envelopes in a batch must share D")
            tensor = store.get(envelope.payload, verify=verify)
            if tensor.shape != (dim,):
                raise ValueError(
                    f"Referenced tensor has shape {tensor.shape}, expected ({dim},)"
                )
            vectors[b, r] = tensor
    return vectors
