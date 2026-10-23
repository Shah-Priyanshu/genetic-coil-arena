"""A minimal, dependency-free reader for R's ``.RData`` files holding numeric data frames.

The Rieth et al. Tennessee Eastman data is published as R ``save()`` files. To convert it
once without installing R (or ``pyreadr``), this module parses just the part of R's
serialization format that such a file uses: a compressed (gzip, bzip2 or xz) XDR stream
holding a pairlist of named objects, whose values are data frames (lists of numeric
columns with ``names``/``class``/``row.names`` attributes). Anything else (environments,
closures, byte code, complex numbers) raises ``ValueError`` rather than guessing.

Numeric columns are returned as numpy arrays; doubles can be narrowed on the way in
(``float_dtype``) so a 4 GB frame never has to exist at full precision.
"""

import bz2
import gzip
import lzma
import struct
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

import numpy as np

# SEXP type codes (R serialize.c)
_NIL, _SYM, _LIST, _CHAR = 254, 1, 2, 9
_LGL, _INT, _REAL, _STR, _VEC, _EXPR, _RAW = 10, 13, 14, 16, 19, 20, 24
_REF, _ALTREP = 255, 238
_ENV_LIKE = {241, 242, 247, 251, 252, 253}  # base/empty/global env, missing arg, ...
_PAIRLIST_LIKE = {_LIST, 6}  # pairlist and language objects share the cons layout
NA_INTEGER = -(2**31)


class RSymbol(str):
    """An R symbol (a name); a ``str`` so tags compare directly with column names."""


class RList(list):
    """A generic vector (``VECSXP``) with its attributes (``names``, ``class``, ...)."""

    attrs: dict[str, Any]

    def __init__(self, values: Any = (), attrs: Optional[dict[str, Any]] = None):
        super().__init__(values)
        self.attrs = attrs or {}

    @property
    def names(self) -> list[str]:
        """The element names (empty strings when the list is unnamed)."""
        names = self.attrs.get("names")
        return list(names) if names is not None else [""] * len(self)


def _open(path: Path) -> BinaryIO:
    """Open a possibly compressed file by sniffing its magic bytes."""
    with open(path, "rb") as handle:
        magic = handle.read(6)
    if magic[:2] == b"\x1f\x8b":
        return gzip.open(path, "rb")  # type: ignore[return-value]
    if magic[:3] == b"BZh":
        return bz2.open(path, "rb")  # type: ignore[return-value]
    if magic[:6] == b"\xfd7zXZ\x00":
        return lzma.open(path, "rb")  # type: ignore[return-value]
    return open(path, "rb")


class _Reader:
    """Recursive-descent reader over an XDR serialization stream."""

    def __init__(self, stream: BinaryIO, float_dtype: Optional[Any] = None):
        self.s = stream
        self.float_dtype = float_dtype
        self.refs: list[Any] = []

    # ------------------------------------------------------------------ primitives
    def _read(self, n: int) -> bytes:
        data = self.s.read(n)
        if len(data) != n:
            raise ValueError("Unexpected end of the R serialization stream")
        return data

    def _int(self) -> int:
        return int(struct.unpack(">i", self._read(4))[0])

    def _length(self) -> int:
        n = self._int()
        if n == -1:  # long vector: two 32-bit halves
            hi, lo = self._int(), self._int()
            return (hi << 32) + lo
        return n

    def _numeric(self, n: int, dtype: str, native: Any) -> np.ndarray:
        size = np.dtype(dtype).itemsize
        raw = np.frombuffer(self._read(n * size), dtype=dtype)
        if native is np.float64 and self.float_dtype is not None:
            return raw.astype(self.float_dtype)
        return raw.astype(native)  # to native byte order

    # ------------------------------------------------------------------- header
    def header(self) -> None:
        if self._read(2) != b"X\n":
            raise ValueError(
                "Only the XDR (binary) R serialization format is supported"
            )
        version = self._int()
        self._int()  # writer R version
        self._int()  # minimum reader R version
        if version == 3:
            self._read(self._int())  # native encoding name
        elif version != 2:
            raise ValueError(f"Unsupported R serialization version {version}")

    # -------------------------------------------------------------------- items
    def item(self) -> Any:
        flags = self._int()
        return self._item(flags)

    def _item(self, flags: int) -> Any:  # noqa: C901 (one branch per SEXP type)
        kind = flags & 0xFF
        if kind == _NIL:
            return None
        if kind in _ENV_LIKE:
            return None
        if kind == _REF:
            index = flags >> 8 or self._int()
            return self.refs[index - 1]
        if kind == _SYM:
            name = RSymbol(self.item())
            self.refs.append(name)
            return name
        if kind == _CHAR:
            n = self._int()
            return None if n == -1 else self._read(n).decode("utf-8", "replace")
        if kind in _PAIRLIST_LIKE:
            return self._pairlist(flags)
        if kind == _ALTREP:
            return self._altrep()
        if kind == _RAW:
            value: Any = np.frombuffer(self._read(self._length()), dtype=np.uint8)
        elif kind in (_LGL, _INT):
            value = self._numeric(self._length(), ">i4", np.int32)
        elif kind == _REAL:
            value = self._numeric(self._length(), ">f8", np.float64)
        elif kind == _STR:
            value = [self.item() for _ in range(self._length())]
        elif kind in (_VEC, _EXPR):
            value = RList(self.item() for _ in range(self._length()))
        else:
            raise ValueError(f"Unsupported R object type {kind} in the stream")
        if flags & 0x200:  # attributes follow the data
            attrs = dict(self.item() or ())
            if isinstance(value, RList):
                value.attrs = attrs
            elif "levels" in attrs:
                raise ValueError("Factor columns are not supported")
        return value

    def _pairlist(self, flags: int) -> list[tuple[Any, Any]]:
        """A pairlist as ``[(tag, value), ...]`` (read iteratively along the cdr)."""
        items: list[tuple[Any, Any]] = []
        while True:
            if flags & 0x200:
                self.item()  # cell attributes are not needed
            tag = self.item() if flags & 0x400 else None
            items.append((tag, self.item()))
            flags = self._int()
            kind = flags & 0xFF
            if kind == _NIL:
                return items
            if kind not in _PAIRLIST_LIKE:
                raise ValueError(f"Unexpected pairlist tail of type {kind}")

    def _altrep(self) -> Any:
        info = self.item()  # pairlist (class symbol, package symbol, base type)
        state = self.item()
        self.item()  # attributes
        cls = str(info[0][1])
        if cls == "compact_intseq":
            n, start, step = (int(v) for v in state)
            return (start + step * np.arange(n)).astype(np.int32)
        if cls == "compact_realseq":
            count, first, increment = (float(v) for v in state)
            return first + increment * np.arange(int(count))
        raise ValueError(f"Unsupported ALTREP class {cls!r}")


def read_rdata(
    path: Union[str, Path], float_dtype: Optional[Any] = None
) -> dict[str, Any]:
    """The named objects of an ``.RData`` file.

    Args:
        path: A ``save()`` file (gzip, bzip2, xz or uncompressed XDR)
        float_dtype: Narrow every double vector to this dtype while reading (``None``
            keeps float64)

    Returns:
        Object name -> value: ``RList`` for lists and data frames (``.names`` gives the
        column names), numpy arrays for atomic vectors

    Raises:
        ValueError: If the file uses a feature this reader does not support
    """
    with _open(Path(path)) as stream:
        magic = stream.read(5)
        if magic not in (b"RDX2\n", b"RDX3\n"):
            raise ValueError(f"{path} is not an XDR RData file (magic {magic!r})")
        reader = _Reader(stream, float_dtype)
        reader.header()
        top = reader.item()
    if not isinstance(top, list):
        raise ValueError("Expected a pairlist of named objects at the top level")
    return {str(tag): value for tag, value in top}


def data_frame_columns(frame: Any) -> dict[str, np.ndarray]:
    """Column name -> array of a data frame returned by ``read_rdata``.

    Raises:
        ValueError: If ``frame`` is not a data frame of equally long columns
    """
    if not isinstance(frame, RList) or "data.frame" not in (
        frame.attrs.get("class") or ()
    ):
        raise ValueError("Not an R data frame")
    names = frame.names
    if len(set(names)) != len(names):
        raise ValueError("Duplicate column names")
    columns = dict(zip(names, frame))
    lengths = {len(col) for col in columns.values()}
    if len(lengths) > 1:
        raise ValueError(f"Columns have different lengths: {sorted(lengths)}")
    return columns
