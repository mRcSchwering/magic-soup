from magicsoup import _lib  # type: ignore


def point_mutations(
    seqs: list[str], p: float = 1e-6, p_indel: float = 0.4, p_del: float = 0.66
) -> list[tuple[str, int]]:
    """
    Add point mutations to a list of nucleotide sequences.
    Mutations are substitutions and indels.

    Arguments:
        seqs: nucleotide sequences
        p: probability of a mutation per nucleotide
        p_indel: probability of any point mutation being an indel
                 (inverse probability of it being a substitution)
        p_del: probability of any indel being a deletion
               (inverse probability of it being an insertion)

    Returns:
        List of mutated sequences and their indices from `seqs`.

    The returned list only contains sequences which experienced at least one mutation.
    The new sequences are returned together with their `seqs` index.
    E.g. `("...", 5)` means `seqs[5]` was mutated and the resulting sequence is
    in the tuple.
    """
    return _lib.point_mutations(seqs, p, p_indel, p_del)


def recombinations(
    seq_pairs: list[tuple[str, str]], p: float = 1e-8
) -> list[tuple[str, str, int]]:
    """
    Add random recombinations to pairs of nucleotide sequences.
    The recombination happens by creating random strand breaks in the input sequence pairs
    and randomly re-joining them.

    Arguments:
        seq_pairs: nucleotide sequence pairs
        p: probability of a strand break per nucleotide

    Returns:
        List of mutated sequence pairs and their indices.

    The returned list only contains sequence pairs that experienced a recombination.
    The new sequence paris are returned and in addition to that their `seq_pairs` index.
    E.g. if it's 5, it means `seq_pairs[5]` was mutated and the resulting sequences are in this 3-tuple.
    """
    if len(seq_pairs) == 0:
        return []
    return _lib.recombinations(seq_pairs, p)
