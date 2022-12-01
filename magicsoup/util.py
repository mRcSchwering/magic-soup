from typing import Optional
from itertools import product
import string
import random
import torch
import numpy as np


CODON_SIZE = 3
ALL_NTS = tuple("TCGA")
ALL_CODONS = tuple(set("".join(d) for d in product(ALL_NTS, ALL_NTS, ALL_NTS)))


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


def weight_map_fact(n_nts: int, min_w: float, max_w: float) -> dict[str, float]:
    """
    Generate codon-to-weight mapping with uniformly distributed weights
    
    - `n_nts` number of nucleotides which will encode weight
    - `min_w` minimum weight
    - `max_w` maximum weight
    """
    codons = variants("N" * n_nts)
    return {d: random.uniform(min_w, max_w) for d in codons}


def bool_map_fact(n_nts: int, p: float = 0.5) -> dict[str, bool]:
    """
    Generate weighted codon-to-bool mapping 
    
    - `n_nts` number of nucleotides which will encode weight
    - `p` chance of `True`
    """
    codons = variants("N" * n_nts)
    bls = random.choices((True, False), weights=(p, 1 - p), k=len(codons))
    return {d: b for d, b in zip(codons, bls)}


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


def trunc(tens: torch.Tensor, n_decs: int) -> torch.Tensor:
    """Round values of a tensor to `n_decs` decimals"""
    return torch.round(tens * 10 ** n_decs) / (10 ** n_decs)
