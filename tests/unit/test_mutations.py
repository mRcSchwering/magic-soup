import magicsoup.mutations as muts


def test_point_mutations():
    # note: a sequence can be mutated but still unchance
    # e.g. if A is substituted with A or if A if deleted, then inserted
    seqs = ["AAAAAAAAAA"] * 100
    mutated = muts.point_mutations(seqs=seqs, p=0.1, p_indel=0.0)
    assert len(mutated) > 50  # chance >= 99.7%
    assert all(len(d) == 10 for d, _ in mutated)

    # 1 expected, sometimes up to 4
    n_ok = sum(0 < sum(dd != "A" for dd in d) < 5 for d, _ in mutated)
    assert n_ok > len(mutated) * 0.7

    mutated = muts.point_mutations(seqs=seqs, p=0.1, p_indel=1.0, p_del=1.0)
    assert len(mutated) > 50  # chance >= 99.7%
    assert all(len(d) < 10 for d, _ in mutated)

    mutated = muts.point_mutations(seqs=seqs, p=0.1, p_indel=1.0, p_del=0.0)
    assert len(mutated) > 50  # chance >= 99.7%
    assert all(len(d) > 10 for d, _ in mutated)

    seqs = ["TACCCAAGGA"] * 60 + ["AGTAACCA"] * 20 + ["AGACCGAATTAG"] * 20
    mutated = muts.point_mutations(seqs=seqs, p=0.1)
    assert len(mutated) > 50  # chance >= 99.7%

    n_ok = sum(seqs[i] != d for d, i in mutated)
    assert n_ok > len(mutated) * 0.7


def test_recombinations():
    seq_pair = ("AAAAAAAAAA", "CCCCCCCCCC")

    mutated = muts.recombinations(seq_pairs=[seq_pair] * 100, p=0.1)
    assert len(mutated) > 50  # chance >= 99.7%
    assert all(len(a) + len(b) == 20 for a, b, _ in mutated)

    n_ok = sum("A" in b or "C" in a for a, b, _ in mutated)
    assert n_ok > len(mutated) * 0.7

    mutated = muts.recombinations(
        seq_pairs=[("AAAA", "CCCC"), ("AAAACCCC", ""), ("", "AAAACCCC")], p=1.0
    )
    assert len(mutated) == 3  # chance >= 99.7%
    assert all(len(a) + len(b) == 8 for a, b, _ in mutated)
