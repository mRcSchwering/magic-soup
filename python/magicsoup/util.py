import math
import random
import string
from collections.abc import Iterable, Sequence
from itertools import product
from typing import Any

import torch

from magicsoup import _lib  # type: ignore
from magicsoup.constants import ALL_NTS, CODON_SIZE


def round_down(d: float, to: int = 3) -> int:
    """Round down to declared integer"""
    return math.floor(d / to) * to


def closest_value(values: Iterable[float], key: float) -> float:
    """Get closest value to key in values"""
    return min(values, key=lambda d: abs(d - key))


def randstr(n: int = 12) -> str:
    """
    Generate random string of length `n`.

    With `n=12` and the string consisting of 62 different characters,
    there's a 50% chance of encountering one collision after 5e10 draws.
    (birthday paradox)
    """
    return "".join(
        random.choices(
            string.ascii_uppercase + string.ascii_lowercase + string.digits, k=n
        )
    )


def random_genome(s: int = 500, excl: list[str] | None = None) -> str:
    """
    Generate a random nucleotide sequence string

    Parameters:
        s: Length of genome in nucleotides or base pairs
        excl: Exclude certain sequences from the genome

    Returns:
        Generated genome as string

    If `excl` is given, all sequences in `excl` will be removed.
    However, these sequences might still appear in the reverse-complement of
    the resulting genome.
    If you also want to get rid of those, you have to also provide their
    reverse-complement in `excl`.
    """
    n = s
    out = "".join(random.choices(ALL_NTS, k=s))

    if excl is not None:
        for seq in excl:
            out = "".join(out.split(seq))
        while len(out) != s:
            n = s - len(out)
            out += random_genome(s=n)
            for seq in excl:
                out = "".join(out.split(seq))

    return out


def variants(seq: str) -> list[str]:
    """
    Generate all possible nucleotide sequences from a template string.

    Apart from nucleotides, the template string can include special characters:
    - `N` refers to any nucleotide
    - `R` refers to purines (A or G)
    - `Y` refers to pyrimidines (C or T)
    """

    def apply(s: str, char: str, nts: tuple[str, ...]):
        n = s.count(char)
        for i in range(n):
            idx = s.find(char)
            s = s[:idx] + "{" + str(i) + "}" + s[idx + 1 :]
        ns = [nts] * n
        return [s.format(*d) for d in product(*ns)]

    seqs1 = apply(seq, "N", ("T", "C", "G", "A"))
    seqs2 = [ss for s in seqs1 for ss in apply(s, "R", ("A", "G"))]
    seqs3 = [ss for s in seqs2 for ss in apply(s, "Y", ("C", "T"))]
    return seqs3


def codons(n: int, excl_codons: list[str] | None = None) -> list[str]:
    """
    Return all possible nucleotide sequences of `n` codons,
    optionally excluding codons in `excl_codons`.
    """
    all_seqs = variants("N" * n * CODON_SIZE)
    if excl_codons is None:
        return all_seqs
    seqs = []
    for seq in all_seqs:
        has_stop = False
        for i in range(n):
            a = i * CODON_SIZE
            b = (i + 1) * CODON_SIZE
            if seq[a:b] in excl_codons:
                has_stop = True
        if not has_stop:
            seqs.append(seq)
    return seqs


def dist_1d(a: int, b: int, m: int) -> int:
    """Distance between `a` and `b` on circular 1D line of size `m`"""
    return _lib.dist_1d(a, b, m)


class TensorClass:

    def __init__(
        self,
        device: torch.device | str = "cpu",
        itype: torch.dtype = torch.int32,
        ftype: torch.dtype = torch.float32,
    ) -> None:
        self.device = device
        self.itype = itype
        self.ftype = ftype

    def idxtensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=torch.int32)

    def itensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=self.itype)

    def ftensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=self.ftype)

    def idxzeros(self, *args) -> torch.Tensor:
        return torch.zeros(*args, device=self.device, dtype=torch.int32)

    def izeros(self, *args) -> torch.Tensor:
        return torch.zeros(*args, device=self.device, dtype=self.itype)

    def fzeros(self, *args) -> torch.Tensor:
        return torch.zeros(*args, device=self.device, dtype=self.ftype)

    def __repr__(self) -> str:
        kwargs = {
            "device": self.device,
            "itype": self.itype,
            "ftype": self.ftype,
        }
        args = [f"{k}:{d!r}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"


IndexLike = slice | list[int] | tuple[int] | torch.Tensor


class Array[T]:
    """Convenience class for indexing a list like a tensor"""

    def __init__(self, items: list[T]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: IndexLike) -> list[T]:
        items = self.items

        if isinstance(idx, torch.Tensor):

            # 1) Boolean tensor mask
            if idx.dtype == torch.bool:
                if idx.ndim != 1:
                    raise ValueError("Boolean mask must be 1D")
                if idx.numel() != len(items):
                    raise ValueError("Boolean mask length must match items length")
                idxs = idx.nonzero(as_tuple=True)[0].tolist()
                return [items[i] for i in idxs]

            # 2) Integer tensor indices
            if idx.dtype in (torch.int32, torch.int64):
                if idx.ndim != 1:
                    raise ValueError("Integer index tensor must be 1D")
                idxs = idx.tolist()
                return [items[i] for i in idxs]

            raise TypeError(f"Unsupported tensor dtype for indexing: {idx.dtype}")

        # 3) Python list / tuple of ints
        if isinstance(idx, (list, tuple)):
            return [items[i] for i in idx]

        # 4) Plain slice: l[idx] (standard behavior)
        if isinstance(idx, slice):
            return items[idx]

        raise ValueError(f"Unsupported index type: {type(idx)}")

    def __setitem__(self, idx: IndexLike, value: Sequence[T]) -> None:
        items = self.items

        if isinstance(idx, torch.Tensor):

            # 1) Boolean tensor mask
            if idx.dtype == torch.bool:
                if idx.ndim != 1 or idx.numel() != len(items):
                    raise ValueError("Boolean mask must be 1D and match value length")
                idxs = idx.nonzero(as_tuple=True)[0].tolist()
                if len(idxs) != len(value):
                    raise ValueError("Number of True elements must match value length")
                for i, val in zip(idxs, value):
                    items[i] = val
                return

            # 2) Integer tensor indices
            if idx.dtype in (torch.int32, torch.int64):
                if idx.ndim != 1:
                    raise ValueError("Integer index tensor must be 1D")
                idxs = idx.tolist()
                if len(idxs) != len(value):
                    raise ValueError("Number of indices must match value length")
                for i, val in zip(idxs, value):
                    items[i] = val
                return

            raise ValueError(f"Unsupported index type: {type(idx)}")

        # 3) Python list / tuple of ints
        if isinstance(idx, (list, tuple)):
            if len(idx) != len(value):
                raise ValueError("Number of indices must match value length")
            for i, val in zip(idx, value):
                items[i] = val
            return

        # 4) Plain slice: l[idx] (standard behavior)
        if isinstance(idx, (int, slice)):
            items[idx] = value
            return

        raise ValueError(f"Unsupported index type: {type(idx)}")
