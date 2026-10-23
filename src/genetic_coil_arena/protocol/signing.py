"""HMAC signature manifests for envelopes.

A learned or latent payload is unreadable to humans, so a message could say one
thing in its readable header while carrying something else in its payload. The
manifest binds the header and a digest of the payload under an HMAC key: changing
either one invalidates the signature. For ``latent_ref`` payloads the reference
carries the tensor's digest, and :func:`verify` can re-hash the tensor in its store
to catch a tensor swapped after sending.

Manifest format::

    {"alg": "HMAC-SHA256", "key_id": "...", "payload_sha256": "...", "mac": "..."}
"""

import hashlib
import hmac
from typing import Any, Dict, Mapping, Optional, Union

from .envelope import Envelope, canonical_json
from .tensor_store import TensorStore, tensor_digest

MAC_ALGORITHM = "HMAC-SHA256"

Keys = Union[bytes, Mapping[str, bytes]]


class SignatureError(ValueError):
    """Raised by :func:`require_valid` when an envelope fails verification."""


def payload_digest(envelope: Envelope) -> str:
    """SHA-256 of the canonical JSON form of the payload."""
    payload = envelope.to_dict()["payload"]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _signed_bytes(envelope: Envelope, key_id: str, digest: str) -> bytes:
    header = envelope.to_dict()
    header.pop("payload")
    header.pop("manifest", None)
    body = {"alg": MAC_ALGORITHM, "key_id": key_id, "header": header, "payload_sha256": digest}
    return canonical_json(body).encode("utf-8")


def sign(envelope: Envelope, key: bytes, key_id: str = "default") -> Envelope:
    """Return a copy of ``envelope`` carrying a signature manifest.

    Args:
        envelope: Envelope to sign (an existing manifest is replaced)
        key: Secret HMAC key
        key_id: Name of the key, so verifiers can pick it from a keyring
    """
    if not key:
        raise ValueError("signing key must be non-empty")
    digest = payload_digest(envelope)
    mac = hmac.new(key, _signed_bytes(envelope, key_id, digest), hashlib.sha256).hexdigest()
    manifest: Dict[str, Any] = {"alg": MAC_ALGORITHM, "key_id": key_id,
                                "payload_sha256": digest, "mac": mac}
    return envelope.with_manifest(manifest)


def check_signature(envelope: Envelope, keys: Keys,
                    store: Optional[TensorStore] = None) -> Optional[str]:
    """Verify an envelope's manifest.

    Args:
        envelope: Envelope to check
        keys: A single key, or a keyring mapping key_id to key
        store: If given, latent_ref tensors are re-hashed and compared to their reference

    Returns:
        None if valid, otherwise a short reason
    """
    manifest = envelope.manifest
    if not manifest:
        return "unsigned"
    if manifest.get("alg") != MAC_ALGORITHM:
        return f"unsupported algorithm {manifest.get('alg')!r}"
    key_id = manifest.get("key_id")
    if isinstance(keys, (bytes, bytearray)):
        key: Optional[bytes] = bytes(keys)
    else:
        key = keys.get(key_id) if isinstance(key_id, str) else None
    if not key:
        return f"unknown key id {key_id!r}"

    digest = payload_digest(envelope)
    if not hmac.compare_digest(digest, str(manifest.get("payload_sha256", ""))):
        return "payload digest mismatch"
    expected = hmac.new(key, _signed_bytes(envelope, key_id, digest), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, str(manifest.get("mac", ""))):
        return "bad signature"

    if store is not None and envelope.payload_type == "latent_ref":
        try:
            tensor = store.get(envelope.payload)
        except KeyError:
            return "referenced tensor not found"
        if tensor_digest(tensor) != envelope.payload["sha256"]:
            return "tensor digest mismatch"
    return None


def verify(envelope: Envelope, keys: Keys, store: Optional[TensorStore] = None) -> bool:
    """True if the envelope's signature (and referenced tensor, if checked) is valid."""
    return check_signature(envelope, keys, store) is None


def require_valid(envelope: Envelope, keys: Keys, store: Optional[TensorStore] = None) -> Envelope:
    """Return the envelope if valid, else raise :class:`SignatureError`."""
    reason = check_signature(envelope, keys, store)
    if reason is not None:
        raise SignatureError(f"Envelope {envelope.msg_id} from {envelope.sender!r}: {reason}")
    return envelope
