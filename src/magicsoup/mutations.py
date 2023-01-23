import random
import torch
from magicsoup.constants import ALL_NTS


def substitution(seq: str, idx: int) -> str:
    """Create a 1 nucleotide substitution at index"""
    nt = random.choice(ALL_NTS)
    return seq[:idx] + nt + seq[idx + 1 :]


def indel(seq: str, idx: int) -> str:
    """Create a 1 nucleotide insertion or deletion (1:1 chances) at index"""
    if random.choice([True, False]):
        return seq[:idx] + seq[idx + 1 :]
    nt = random.choice(ALL_NTS)
    return seq[:idx] + nt + seq[idx:]


def point_mutations(seqs: list[str], p=1e-3, p_indel=0.1) -> list[tuple[str, int]]:
    """
    Add point mutations to a list of nucleotide sequences.

    - `seqs` nucleotide sequences
    - `p` probability of a mutation per nucleotide
    - `p_indel` probability of any point mutation being a deletion or insertion
      (inverse probability of it being a substitution)
    
    Returns list of mutated sequences and their indices; which sequences of `seqs` got muated.

    If an indel occurs, there is a 1:1 chance of it being a deletion or insertion.
    """
    n = len(seqs)
    if n == 0:
        return []

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
    return [(tmps[i], i) for i in idxs]


def recombinations(
    seq_pairs: list[tuple[str, str]], p=1e-3
) -> list[tuple[str, str, int]]:
    """
    Add random recombinations to pairs of nucleotide sequences

    - `seq_pairs` nucleotide sequence pairs
    - `p` probability of a strand break per nucleotide
    
    Returns list of mutated sequence pairs and their indices.
    The index describes the index of `seq_pairs`.
    E.g. if it's 5, it means `seq_pairs[5]` was mutated and the resulting sequences are in this 3-tuple.

    The recombination happens by creating random strand breaks in both sequences (with likelihood `p`),
    then randomly shuffling the strands and randomly re-joining them.
    """
    n = len(seq_pairs)
    if n == 0:
        return []

    combined_seqs = [(a + b, len(a)) for a, b in seq_pairs]
    lens = [len(d) for d in combined_seqs]
    s_max = max(lens)

    mask = torch.zeros(n, s_max)
    for i, s in enumerate(lens):
        mask[i, :s] = True

    probs = torch.full((n, s_max), p)
    muts = torch.bernoulli(probs)
    mut_idxs = torch.argwhere(muts * mask)
    mut_rows = set(mut_idxs[:, 0].tolist())

    tmps: list[tuple[str, str, int]] = []
    for row_i in mut_rows:
        seq, cut = combined_seqs[row_i]

        cuts = mut_idxs[mut_idxs[:, 0] == row_i, 1].tolist()
        l = [0] + sorted(cuts + [cut]) + [len(seq)]
        parts = [seq[a:b] for a, b in zip(l, l[1:])]

        random.shuffle(parts)
        split = random.randint(0, len(parts))
        lft_new = "".join(parts[:split])
        rght_new = "".join(parts[split:])
        tmps.append((lft_new, rght_new, row_i))

    return tmps

