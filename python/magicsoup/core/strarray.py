from collections.abc import Iterable, Iterator, Sequence
from typing import overload

import numpy as np
import numpy.typing as npt

# a selection of several positions: integer indices or a boolean mask
IntIndex = npt.NDArray[np.integer] | npt.NDArray[np.bool_] | Sequence[int]


def _starts_from_lens(lens: np.ndarray) -> np.ndarray:
    """Start offsets for a fresh, contiguous layout of the given lengths"""
    starts = np.zeros(len(lens), dtype=np.int64)
    if len(lens) > 1:
        np.cumsum(lens[:-1], out=starts[1:])
    return starts


class StrArray:
    """
    Mutable 1-D array of variable-length strings.

    Arguments:
        strings: Initial strings. Defaults to an empty array.

    Elements are stored UTF-8 encoded in a single byte buffer; element `i` is
    `buffer[starts[i] : starts[i] + lens[i]]`. Use [empty()][.empty] to create
    an array of `n` empty strings and [concat()][.concat] to join arrays.

    Indexing semantics mirror a NumPy string array:

    - `arr[i]` with an integer returns a `str`.
    - `arr[idxs]` with an integer array, list or boolean mask returns a new
      `StrArray` that shares the same underlying buffer (no bytes copied).
    - `arr[idxs] = value` scatters `value` (a `str`, an iterable of `str` or
      another `StrArray`) into the given positions.

    The representation is designed for cheap bookkeeping and zero-copy exchange
    with Rust:

    - gather (`arr[idxs]`), filter (`arr[mask]`), kill (`arr[idxs] = ""`) and
    copy (`arr[children] = arr[parents]`) only touch the handle table and never
    copy string bytes.
    - only genuinely new strings are ever written, and they are *appended* to the
    arena. Overwritten bytes become dead until [compact()][.compact] reclaims them.
    """

    def __init__(self, buf: np.ndarray, starts: np.ndarray, lens: np.ndarray) -> None:
        self._buf = buf
        self._starts = starts
        self._lens = lens

    @classmethod
    def from_strings(cls, strings: Iterable[str] = ()) -> "StrArray":
        blobs = [s.encode() for s in strings]
        lens = np.fromiter((len(b) for b in blobs), dtype=np.int64, count=len(blobs))
        buf = np.frombuffer(b"".join(blobs), dtype=np.uint8).copy()
        starts = _starts_from_lens(lens)
        return cls(buf=buf, starts=starts, lens=lens)

    # construction

    @classmethod
    def empty(cls, n: int = 0) -> "StrArray":
        """Create an array of `n` empty strings"""
        zeros = np.zeros(n, dtype=np.int64)
        return cls(buf=np.empty(0, dtype=np.uint8), starts=zeros, lens=zeros.copy())

    @classmethod
    def concat(cls, arrays: Iterable["StrArray"]) -> "StrArray":
        """Concatenate multiple arrays into a new, compact array"""
        arrays = list(arrays)
        if not arrays:
            return cls.empty(0)

        blobs: list[np.ndarray] = []
        starts: list[np.ndarray] = []
        lens: list[np.ndarray] = []
        base = 0
        for arr in arrays:
            blob, arr_lens = arr._contiguous()
            starts.append(base + _starts_from_lens(arr_lens))
            lens.append(arr_lens)
            blobs.append(blob)
            base += len(blob)

        return cls(
            buf=np.concatenate(blobs),
            starts=np.concatenate(starts),
            lens=np.concatenate(lens),
        )

    # sizing

    def __len__(self) -> int:
        return len(self._lens)

    @property
    def shape(self) -> tuple[int]:
        """Shape `(n,)` of the array (mirrors `numpy.ndarray.shape`)"""
        return (len(self),)

    @property
    def utilization(self) -> float:
        """Number of live bytes divided by the total buffer size (see [compact()][.compact])."""
        return float(self._lens.sum() / self._buf.size)

    # reading

    # TODO: add slice indexing?
    @overload
    def __getitem__(self, idx: int | np.integer) -> str: ...
    @overload
    def __getitem__(self, idx: IntIndex) -> "StrArray": ...
    def __getitem__(self, idx: "int | np.integer | IntIndex") -> "str | StrArray":
        if isinstance(idx, (int, np.integer)):
            start = int(self._starts[idx])
            length = int(self._lens[idx])
            return self._buf[start : start + length].tobytes().decode()

        ii = self._as_int_index(idx)
        return StrArray(buf=self._buf, starts=self._starts[ii], lens=self._lens[ii])

    def __iter__(self) -> Iterator[str]:
        buf = self._buf
        for start, length in zip(self._starts.tolist(), self._lens.tolist()):
            yield buf[start : start + length].tobytes().decode()

    def tolist(self) -> list[str]:
        """Return all strings as a Python list"""
        return list(self)

    def copy(self) -> "StrArray":
        """Return an independent, compact copy (does not share the buffer)"""
        blob, lens = self._contiguous()
        return StrArray(buf=blob, starts=_starts_from_lens(lens), lens=lens)

    # writing

    def __setitem__(
        self,
        idx: "int | np.integer | IntIndex",
        value: "str | Iterable[str] | StrArray",
    ) -> None:
        if isinstance(idx, (int, np.integer)):
            ii = np.array([idx], dtype=np.int64)
            if isinstance(value, str):
                value = [value]
        else:
            ii = self._as_int_index(idx)

        if isinstance(value, str):
            self._scatter_scalar(ii, value)
        elif isinstance(value, StrArray):
            self._scatter_strarray(ii, value)
        else:
            self._scatter_strings(ii, list(value))

    def _scatter_scalar(self, ii: np.ndarray, value: str) -> None:
        if value == "":
            self._starts[ii] = 0
            self._lens[ii] = 0
            return

        blob = value.encode()
        start = self._buf.size
        self._buf = np.concatenate([self._buf, np.frombuffer(blob, dtype=np.uint8)])
        self._starts[ii] = start  # all selected rows share the one appended region
        self._lens[ii] = len(blob)

    def _scatter_strings(self, ii: np.ndarray, strings: list[str]) -> None:
        if len(ii) != len(strings):
            raise ValueError(
                f"Cannot assign {len(strings)} strings to {len(ii)} positions"
            )

        if not strings:
            return

        blobs = [s.encode() for s in strings]
        lens = np.fromiter((len(b) for b in blobs), dtype=np.int64, count=len(blobs))
        base = self._buf.size
        self._buf = np.concatenate(
            [self._buf, np.frombuffer(b"".join(blobs), dtype=np.uint8)]
        )
        self._starts[ii] = base + _starts_from_lens(lens)
        self._lens[ii] = lens

    def _scatter_strarray(self, ii: np.ndarray, value: "StrArray") -> None:
        if len(ii) != len(value):
            raise ValueError(
                f"Cannot assign {len(value)} strings to {len(ii)} positions"
            )

        if value._buf is self._buf:
            self._starts[ii] = value._starts
            self._lens[ii] = value._lens
            return

        blob, lens = value._contiguous()
        base = self._buf.size
        self._buf = np.concatenate([self._buf, blob])
        self._starts[ii] = base + _starts_from_lens(lens)
        self._lens[ii] = lens

    # maintenance

    def compact(self) -> None:
        """
        Drop dead bytes, repacking the buffer to hold only live strings.

        Scatter and kill operations leave overwritten bytes behind. Call this
        (e.g. periodically, or after large deletions) to reclaim that memory.
        """
        blob, lens = self._contiguous()
        self._buf = blob
        self._starts = _starts_from_lens(lens)
        self._lens = lens

    # TODO: required?
    def as_buffers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Expose the backing arrays `(buffer, starts, lens)` for zero-copy exchange.

        Element `i` is `buffer[starts[i] : starts[i] + lens[i]]`. The buffer may
        contain dead bytes; always honour `starts`/`lens` rather than reading it
        contiguously. [compact()][.compact] first if a dense buffer is required.
        """
        return self._buf, self._starts, self._lens

    # internals

    def _as_int_index(self, idx: IntIndex) -> np.ndarray:
        """Normalize an index (array, list, mask, tensor) to int64 positions"""
        arr = np.asarray(idx)
        if arr.dtype == bool:
            arr = np.nonzero(arr)[0]

        return arr.astype(np.int64, copy=False)

    def _contiguous(self) -> tuple[np.ndarray, np.ndarray]:
        """Gather live bytes into a dense buffer; returns `(buffer, lens)`"""
        lens = self._lens
        total = int(lens.sum())
        if total == 0:
            return np.empty(0, dtype=np.uint8), lens.copy()

        starts = _starts_from_lens(lens)
        src = np.repeat(self._starts - starts, lens) + np.arange(total, dtype=np.int64)
        return self._buf[src], lens.copy()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n={len(self)}, utilization={self.utilization:.2f})"
        )
