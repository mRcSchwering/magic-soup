from typing import TypeVar, Sequence
from itertools import product
import string
import random
import torch


def randstr(n: int = 12) -> str:
    """
    Generate random string of length `n`
    
    With `n=12` and the string consisting of 62 different characters,
    there's a 50% chance of encountering one collision after 5e10 draws.
    (birthday paradox)
    """
    return "".join(
        random.choices(
            string.ascii_uppercase + string.ascii_lowercase + string.digits, k=n
        )
    )


def trunc(tens: torch.Tensor, n_decs: int) -> torch.Tensor:
    """Round values of a tensor to `n_decs` decimals"""
    return torch.round(tens * 10 ** n_decs) / (10 ** n_decs)


def cpad1d(t: torch.Tensor, n=1) -> torch.Tensor:
    """Circular `n` padding of 3d tensor in 3rd dimention"""
    return torch.nn.functional.pad(t, (n, n), mode="circular")


def cpad2d(t: torch.Tensor, n=1) -> torch.Tensor:
    """Circular `n` padding of 2d tensor in 1st and 2nd dimension"""
    return (
        cpad1d(cpad1d(t.unsqueeze(0), n=n).permute(0, 2, 1), n=n)
        .permute(0, 2, 1)
        .squeeze(0)
    )


def pad_map(m: torch.Tensor) -> torch.Tensor:
    """Do a circular padding with size 1 on a 2d bool tensor"""
    return (
        cpad1d(cpad1d(m.to(torch.float).unsqueeze(0), n=1).permute(0, 2, 1), n=1)
        .permute(0, 2, 1)
        .squeeze(0)
        .to(torch.bool)
    )


def unpad_map(m: torch.Tensor) -> torch.Tensor:
    """Remove a padding with size 1 from a 2d tensor"""
    return m[1:-1, 1:-1]


def _padded_index(i: int, s: int) -> list[int]:
    if i == 0:
        return [i, s - 2]
    if i == 1:
        return [i, s - 1]
    if i == s - 1:
        return [i, 1]
    if i == s - 2:
        return [i, 0]
    return [i]


def padded_indices(x: int, y: int, s: int) -> tuple[list[int], list[int]]:
    """
    For a (x, y) position in a 2d tensor with circular padding of size 1
    get all other (x, y) positions that represent the same position.
    Returned as 2 lists of the same size with all x and all y values.
    """
    xs_ = _padded_index(i=x, s=s)
    ys_ = _padded_index(i=y, s=s)
    xs = []
    ys = []
    for x, y in product(xs_, ys_):
        xs.append(x)
        ys.append(y)
    return xs, ys


def pad_2_true_idx(idx: int, size: int, pad=1) -> int:
    """
    Convert index of a value in a circularly padded
    array to the index of the value in the corresponding
    non-padded array."""
    if idx == 0:
        return size - pad
    if idx == size + pad:
        return 0
    return idx - pad


def weight_map_fact(seqs: list[str], min_w: float, max_w: float) -> dict[str, float]:
    """
    Generate codon-to-weight mapping with uniformly distributed weights
    
    - `seqs` nucleotides sequences that are used as keys for mapping
    - `min_w` minimum weight
    - `max_w` maximum weight
    """
    return {d: random.uniform(min_w, max_w) for d in seqs}


def bool_map_fact(seqs: list[str], p: float = 0.5) -> dict[str, bool]:
    """
    Generate weighted codon-to-bool mapping 
    
    - `seqs` nucleotides sequences that are used as keys for mapping
    - `p` chance of `True` between 0.0 and 1.0
    """
    bls = random.choices((True, False), weights=(p, 1 - p), k=len(seqs))
    return {d: b for d, b in zip(seqs, bls)}


_Tv = TypeVar("_Tv")


def generic_map_fact(seqs: list[str], choices: Sequence[_Tv]) -> dict[str, _Tv]:
    """
    Generate mapping from nucleotide sequence to objects
    
    - `seqs` nucleotides sequences that are used as keys for mapping
    - `choices` objects to be mapped to
    """
    n = len(choices)
    if n < 1:
        return {}
    return {d: choices[i % n] for i, d in enumerate(seqs)}
