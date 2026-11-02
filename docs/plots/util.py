import math


def _parse_cigar(cigar: str) -> list[tuple[str, int]]:
    ops = []
    n = 0
    for ch in cigar:
        if ch.isdigit():
            n += int(ch)
        else:
            ops.append((ch, int(n)))
            n = 0
    return ops


def ani_from_cigar(cigar: str):
    """
    Get Average Nucleotide Identity (ANI) from a
    Concise Idiosyncratic Gapped Alignment Report (CIGAR)
    """
    m = 0
    mm = 0
    for op, n in _parse_cigar(cigar):
        if op == "=":
            m += n
        elif op == "X":
            mm += n
    aln = m + mm
    return m / aln


def _kmers(seq: str, k: int) -> set[str]:
    seq = seq.upper()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def jaccard(seqa: str, seqb: str, kmer: int = 21) -> float:
    """Get Jaccard similarity"""
    a = _kmers(seqa, k=kmer)
    b = _kmers(seqb, k=kmer)
    return len(a & b) / len(a | b)


def ani_from_jaccard(jacc: float, kmer: int = 21) -> float:
    """
    Get Average Nucleotide Identity (ANI)
    from Jaccard similarity using Mash equation.
    """
    if jacc <= 0:
        return 0.0  # no shared k-mers
    d = -math.log(2 * jacc / (1 + jacc)) / kmer  # Mash distance
    return 1 - d
