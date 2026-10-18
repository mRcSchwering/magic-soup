from magicsoup import rs


def point_mutations(
    seqs: list[str], p: float = 1e-6, p_indel: float = 0.4, p_del: float = 0.66
) -> list[tuple[str, int]]:
    """
    Add point mutations to a list of base pair sequences.
    Mutations are substitutions and indels.

    Arguments:
        seqs: base pair sequences
        p: probability of a mutation per base pair
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
    # TODO: reducing to only the sequences that experience mutations
    #       would reduce serialization from Pythont o Rust
    return rs.point_mutations(seqs=seqs, p=p, p_indel=p_indel, p_del=p_del)


# TODO: test likelihoods again
def slippage(
    seqs: list[str], p: float = 1e-7, p_length: float = 10.0
) -> list[tuple[str, int]]:
    # TODO: implement, important for evolution
    return []


# TODO: check likelihoods again
def nahr(
    seqs: list[str], k: int = 5, min_homology: int = 10, max_occurences: int = 20
) -> list[tuple[str, int]]:
    # 5-mers occur reasonably often in random sequences of 100-1000 bp
    # min_homology would only allow a recombination if 2 sequences share a 10 bp stretch
    # and are not overlapping and don't occure more than 20 times (do avoid very repetitive sequences)
    # TODO: implement, important for evolution
    return []


def recombinations(
    seq_pairs: list[tuple[str, str]], p: float = 1e-7
) -> list[tuple[str, str, int]]:
    """
    Add random recombinations to pairs of base pair sequences.
    The recombination happens by creating random strand breaks in the input sequence pairs
    and randomly re-joining them.

    Arguments:
        seq_pairs: base pair sequence pairs
        p: probability of a strand break per base pair

    Returns:
        List of mutated sequence pairs and their indices.

    The returned list only contains sequence pairs that experienced a recombination.
    The new sequence paris are returned and in addition to that their `seq_pairs` index.
    E.g. if it's 5, it means `seq_pairs[5]` was mutated and the resulting sequences are in this 3-tuple.
    """
    # TODO: here I could also add a homology check to better imitate homologous recombination
    #       currently it rather resembles random strand breaks and re-joining
    # TODO: CGPT recommended p=1e-5 for 1k bp genomes
    if len(seq_pairs) == 0:
        return []
    return rs.recombinations(seq_pairs=seq_pairs, p=p)


class Mutations:

    def __init__(
        self, p_point: float = 1e-6, p_indel: float = 0.4, p_del: float = 0.66
    ) -> None:
        self.p_point = p_point
        self.p_indel = p_indel
        self.p_del = p_del
