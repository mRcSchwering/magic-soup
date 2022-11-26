from typing import TypeVar, Optional
from itertools import product
import random
import re
import numpy as np


CODON_SIZE = 3
ALL_NTS = tuple("TCGA")
ALL_CODONS = tuple(set("".join(d) for d in product(ALL_NTS, ALL_NTS, ALL_NTS)))


Tv = TypeVar("Tv")


def indices(lst: list[Tv], element: Tv) -> list[Tv]:
    """Get all indices of element in list"""
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset + 1)
        except ValueError:
            return result
        result.append(offset)


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


def rand_genome(len_range=(100, 500)) -> str:
    """
    Generate a random nucleotide sequence with length
    sampled within `len_range`.
    """
    k = random.randint(*len_range)
    return "".join(random.choices(ALL_NTS, k=k))


def weight_map_fact(
    n_nts: int, mu: float, sd: float, is_positive=False
) -> dict[str, float]:
    """
    Generate codon-to-weight mapping from Gauss samples
    
    - `n_nts` number of nucleotides which will encode weight
    - `mu` mu of the Gauss distribution sampled
    - `sd` sigma of the Gauss distribution sampled
    - `is_positive` whether to only return positive weights
    """
    codons = variants("N" * n_nts)
    c2w = {d: random.gauss(mu=mu, sigma=sd) for d in codons}
    if is_positive:
        c2w = {k: abs(v) for k, v in c2w.items()}
    return c2w


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


def get_coding_regions(seq: str) -> list[str]:
    """
    Get all possible coding regions in nucleotide sequence

    Assuming coding region can start at any start codon and
    is stopped with the first stop codon encountered in same
    frame.

    Ribosomes will stall without stop codon. So, a coding region
    without a stop codon is not considerd.
    (https://pubmed.ncbi.nlm.nih.gov/27934701/)
    
    22.11.22 takes 2.7s on 1000 coding regions from genomes
    of size (1000, 5000).
    """
    start_codons = ("TTG", "GTG", "ATG")
    stop_codons = ("TGA", "TAG", "TAA")

    pat = ".{1," + str(CODON_SIZE) + "}"
    res = []
    for frame_i in range(CODON_SIZE):
        codons = re.findall(pat, seq[frame_i:])

        starts = []
        for codon in start_codons:
            starts.extend(indices(codons, codon))

        stops = []
        for codon in stop_codons:
            stops.extend(indices(codons, codon))

        for start in sorted(starts):
            opts = [d for d in stops if d > start]
            if len(opts) > 0:
                res.append("".join(codons[start : min(opts) + 1]))

    return res


def _subst(seq: str, idx: int) -> str:
    nt = random.choice(ALL_NTS)
    return seq[:idx] + nt + seq[idx + 1 :]


def _indel(seq: str, idx: int) -> str:
    if random.choice([True, False]):
        return seq[:idx] + seq[idx + 1 :]
    nt = random.choice(ALL_NTS)
    return seq[:idx] + nt + seq[idx:]


def simulate_point_mutation(seq: str, p=1e-3, sub2indel=0.5) -> Optional[str]:
    """Mutate sequence. Returns None if no mutation happened"""
    n = len(seq)
    k = np.random.binomial(n=n, p=p)
    if k < 1:
        return None
    idxs = [random.randint(0, n) for _ in range(k)]
    tmp = seq
    for idx in idxs:
        if random.random() >= sub2indel:
            tmp = _subst(seq=tmp, idx=idx)
        else:
            tmp = _indel(seq=tmp, idx=idx)
    return tmp

