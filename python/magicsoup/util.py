from typing import Iterable
from itertools import product
import string
import random
import math
from magicsoup.constants import ALL_NTS, CODON_SIZE
from magicsoup import _lib  # type:ignore


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


def free_moores_nghbhd(
    x: int, y: int, positions: list[tuple[int, int]], map_size: int
) -> list[tuple[int, int]]:
    """
    For position `x, y` get positions in Moore's neighborhood
    on circular 2D map of size `map_size`
    which are not already occupied as indicated by `positions`
    """
    return _lib.free_moores_nghbhd(x, y, positions, map_size)
