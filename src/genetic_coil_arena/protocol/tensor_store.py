"""Tensor stores: where latent payloads live, so envelopes only carry references.

A reference is a JSON dict ``{"store", "key", "shape", "dtype", "sha256"}``. The
digest lets a receiver (or :func:`protocol.signing.verify`) detect a tensor that was
swapped or modified after the reference was sent.
"""

import hashlib
from typing import Any, Dict, Mapping, Protocol, runtime_checkable

import numpy as np

from .envelope import canonical_json, new_id


def tensor_digest(array: np.ndarray) -> str:
    """SHA-256 over dtype, shape and raw bytes of an array."""
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(canonical_json({"dtype": arr.dtype.str, "shape": list(arr.shape)}).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


class TensorDigestError(ValueError):
    """Raised when a stored tensor no longer matches its reference digest."""


@runtime_checkable
class TensorStore(Protocol):
    """Storage for tensors passed by reference between specialists."""

    name: str

    def put(self, array: np.ndarray) -> Dict[str, Any]:
        """Store a copy of ``array`` and return its reference."""
        ...

    def get(self, ref: Mapping[str, Any], verify: bool = False) -> np.ndarray:
        """Return the tensor for ``ref``; with ``verify`` check its digest."""
        ...

    def delete(self, ref: Mapping[str, Any]) -> None:
        """Forget the tensor for ``ref`` (no error if already gone)."""
        ...


class InMemoryTensorStore:
    """Process-local tensor store; stored tensors are read-only copies."""

    def __init__(self, name: str = "memory"):
        if not name:
            raise ValueError("store name must be non-empty")
        self.name = name
        self._tensors: Dict[str, np.ndarray] = {}

    def put(self, array: np.ndarray) -> Dict[str, Any]:
        """Store a read-only copy of ``array`` and return its reference."""
        arr = np.array(array, copy=True)
        if arr.dtype == object:
            raise TypeError("object arrays cannot be stored")
        arr.setflags(write=False)
        key = new_id()
        self._tensors[key] = arr
        return {"store": self.name, "key": key, "shape": list(arr.shape),
                "dtype": arr.dtype.str, "sha256": tensor_digest(arr)}

    def get(self, ref: Mapping[str, Any], verify: bool = False) -> np.ndarray:
        """Return the stored tensor.

        Raises:
            KeyError: Unknown store name or key
            TensorDigestError: ``verify`` is set and the tensor does not match ``ref``
        """
        if ref.get("store") != self.name:
            raise KeyError(f"Reference belongs to store {ref.get('store')!r}, not {self.name!r}")
        arr = self._tensors[ref["key"]]
        if verify and tensor_digest(arr) != ref.get("sha256"):
            raise TensorDigestError(f"Tensor {ref['key']} does not match its reference digest")
        return arr

    def delete(self, ref: Mapping[str, Any]) -> None:
        """Forget the tensor for ``ref``."""
        self._tensors.pop(ref.get("key"), None)

    def __contains__(self, ref: object) -> bool:
        return isinstance(ref, Mapping) and ref.get("store") == self.name and ref.get("key") in self._tensors

    def __len__(self) -> int:
        return len(self._tensors)
