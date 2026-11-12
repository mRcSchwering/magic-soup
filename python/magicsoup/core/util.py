import random
import string
from collections.abc import Iterable
from itertools import product
from pathlib import Path

from magicsoup.core import rs
from magicsoup.core.config import CODON_SIZE, DEFAULT_RNG

_ALL_NTS = tuple("TCGA")


def reverse_complement(seq: str) -> str:
    """Get reverse complement of nucleotide sequence"""
    return rs.reverse_complement(seq=seq)


def randstr(n: int = 12, rng: random.Random = DEFAULT_RNG) -> str:
    """
    Generate random string of length `n`.

    With `n=12` and the string consisting of 62 different characters,
    there's a 50% chance of encountering one collision after 5e10 draws.
    (birthday paradox)
    """
    return "".join(
        rng.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=n)
    )


def random_genome(
    s: int = 500, excl: Iterable[str] | None = None, rng: random.Random = DEFAULT_RNG
) -> str:
    """
    Generate a random nucleotide sequence string

    Parameters:
        s: Length of genome in nucleotides or base pairs
        excl: Exclude certain sequences from the genome

    Returns:
        Generated genome as string

    If `excl` is given, all sequences in `excl` will be removed.
    However, these sequences might still appear in the reverse-complement of
    the resulting genome or across the seam where the end meets the start.
    """
    n = s
    out = "".join(rng.choices(_ALL_NTS, k=s))

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


def codons(n: int, excl_codons: Iterable[str] | None = None) -> list[str]:
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


def write_fasta(name_2_seq: dict[str, str], file: Path) -> None:
    """Write fasta file from name to sequences mapping"""
    lines = [f">{n}\n{s}" for n, s in name_2_seq.items()]
    file.write_text("\n".join(lines))


def read_fasta(file: Path) -> dict[str, str]:
    """Read name to sequences mapping from fasta file"""
    text = file.read_text()
    entries = [d for d in text.split(">") if len(d) > 0]

    name_2_seq: dict[str, str] = {}
    for entry in entries:
        parts = entry.split("\n")
        if len(parts) == 0:
            continue
        name = parts[0]
        name_2_seq[name] = "" if len(parts) < 2 else parts[1]

    return name_2_seq
