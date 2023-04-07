from typing import TypeVar, Sequence
from itertools import product
import string
import random
import math
from magicsoup.constants import ALL_NTS


def round_down(d: float, to: int = 3) -> int:
    """Round down to declared integer"""
    return math.floor(d / to) * to


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


def random_genome(s=500, excl: list[str] | None = None) -> str:
    """
    Generate a random nucleotide sequence

    Parameters:
        s: Length of genome in nucleotides or base pairs
        excl: Exclude certain sequences from the genome

    Returns:
        Generated genome as string

    The resulting genome is a string of all possible nucleotide letters.
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


def nt_seqs(n: int) -> list[str]:
    """Return all possible nucleotide sequences of length `n`"""
    return variants("N" * n)


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
    Generate random mapping where each single string in `seqs` maps to a uniformly distributed
    float between `min_w` and `max_w`.

    - `seqs` strings that are used as keys
    - `min_w` minimum weight
    - `max_w` maximum weight
    """
    return {d: random.uniform(min_w, max_w) for d in seqs}


def log_weight_map_fact(
    seqs: list[str], min_w: float, max_w: float
) -> dict[str, float]:
    """
    Generate random mapping where each single string in `seqs` maps to a log uniformly
    distributed float between`min_w` and `max_w`.

    - `seqs` strings that are used as keys
    - `min_w` minimum weight (must be > 0.0)
    - `max_w` maximum weight (must be > 0.0)
    """
    l_min_w = math.log(min_w)
    l_max_w = math.log(max_w)
    return {d: math.exp(random.uniform(l_min_w, l_max_w)) for d in seqs}


def bool_map_fact(seqs: list[str], p: float = 0.5) -> dict[str, bool]:
    """
    Generate weighted random mapping where each single string of `seqs`
    maps to either `True` or `False`.

    - `seqs` strings that are used as keys
    - `p` chance of `True`, must be between 0.0 and 1.0
    """
    bls = random.choices((True, False), weights=(p, 1 - p), k=len(seqs))
    return {d: b for d, b in zip(seqs, bls)}


_Tv = TypeVar("_Tv")


def generic_map_fact(seqs: list[str], choices: Sequence[_Tv]) -> dict[str, _Tv]:
    """
    Generate a random mapping where each single string of `seqs` maps to one
    item in `choices`. Items in `choices` can appear multiple times.

    - `seqs` strings that are used as keys
    - `choices` objects that are used as values
    """
    n = len(choices)
    if n < 1:
        return {}
    return {d: choices[i % n] for i, d in enumerate(seqs)}
