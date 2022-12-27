from typing import TypeVar, Sequence
from itertools import product
import string
import random
import torch
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


def random_genome(s=100) -> str:
    """Generate a random nucleotide sequence of length `s`"""
    return "".join(random.choices(ALL_NTS, k=s))


def substitution(seq: str, idx: int) -> str:
    """Create a 1 nucleotide substitution at index"""
    nt = random.choice(ALL_NTS)
    return seq[:idx] + nt + seq[idx + 1 :]


def indel(seq: str, idx: int) -> str:
    """Create a 1 nucleotide insertion or deletion at index"""
    if random.choice([True, False]):
        return seq[:idx] + seq[idx + 1 :]
    nt = random.choice(ALL_NTS)
    return seq[:idx] + nt + seq[idx:]


def point_mutations(
    seqs: list[str], p=1e-3, p_indel=0.1
) -> tuple[list[str], list[int]]:
    """
    Return new sequences mutated with point mutations.

    - `seqs` nucleotide sequences
    - `p` probability of a mutation per nucleotide
    - `p_indel` probability of any point mutation being a deletion or insertion
      (inverse probability of it being a substitution)
    
    Returns all sequences (muated or not).
    """
    n = len(seqs)
    lens = [len(d) for d in seqs]
    s_max = max(lens)

    mask = torch.zeros(n, s_max)
    for i, s in enumerate(lens):
        mask[i, :s] = True

    probs = torch.full((n, s_max), p)
    muts = torch.bernoulli(probs)
    mut_idxs = torch.argwhere(muts * mask).tolist()

    probs = torch.full((len(mut_idxs),), p_indel)
    indels = torch.bernoulli(probs).to(torch.bool).tolist()

    tmps = [d for d in seqs]
    for (seq_i, pos_i), is_indel in zip(mut_idxs, indels):
        if is_indel:
            tmps[seq_i] = indel(seq=tmps[seq_i], idx=pos_i)
        else:
            tmps[seq_i] = substitution(seq=tmps[seq_i], idx=pos_i)

    idxs = list(set(d[0] for d in mut_idxs))
    return [tmps[i] for i in idxs], idxs


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
    Generate random mapping where each single string in `seqs` maps to a weight,
    a float between `min_w` and `max_w`.
    
    - `seqs` strings that are used as keys
    - `min_w` minimum weight
    - `max_w` maximum weight
    """
    return {d: random.uniform(min_w, max_w) for d in seqs}


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
