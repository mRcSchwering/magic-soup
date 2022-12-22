from typing import TypeVar, Sequence
from itertools import product
import string
import random
from magicsoup.constants import ALL_NTS


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


def reverse_complement(seq: str) -> str:
    """Reverse-complement of a nucleotide sequence"""
    rvsd = seq[::-1]
    return (
        rvsd.replace("A", "-")
        .replace("T", "A")
        .replace("-", "T")
        .replace("G", "-")
        .replace("C", "G")
        .replace("-", "C")
    )


def variants(seq: str) -> list[str]:
    """
    Generate all variants of sequence where
    'N' can be any nucleotide.
    """
    s = seq
    n = s.count("N")
    for i in range(n):
        idx = s.find("N")
        s = s[:idx] + "{" + str(i) + "}" + s[idx + 1 :]
    nts = [ALL_NTS] * n
    return [s.format(*d) for d in product(*nts)]


def rad1_nghbrhd(x: int, size: int) -> list[int]:
    """Radial 1D neigborhood of `x` with radius 1 in a circular map of size `size`"""
    if x == 0:
        return [size - 1, 1]
    if x == size - 1:
        return [x - 1, 0]
    return [x - 1, x + 1]


def moore_nghbrhd(x: int, y: int, size: int) -> list[tuple[int, int]]:
    """Moore's neighborhood of `x, y` with radius 1 in a circular map of size `size`"""
    xs = rad1_nghbrhd(x=x, size=size)
    ys = rad1_nghbrhd(x=y, size=size)
    res = set((x, y) for x, y in product(xs + [x], ys + [y]))
    return list(res - {(x, y)})


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
