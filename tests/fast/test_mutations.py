import magicsoup.mutations as muts


def test_point_mutations():
    # note: a sequence can be mutated but still unchance
    # e.g. if A is substituted with A or if A if deleted, then inserted
    seqs = ["AAAAAAAAAA"] * 100
    mutated = muts.point_mutations(seqs=seqs, p=0.1, p_indel=0.0)
    assert len(mutated) > 50  # chance >= 99.7%

    are_ok = 0
    for new, _ in mutated:
        assert len(new) == 10
        d = sum(d != "A" for d in new)
        if 0 < d < 5:  # 1 expected, sometimes up to 4
            are_ok += 1
    assert are_ok > len(mutated) * 0.7

    mutated = muts.point_mutations(seqs=seqs, p=0.1, p_indel=1.0, p_del=1.0)
    assert len(mutated) > 50  # chance >= 99.7%
    assert all(len(d) < 10 for d, _ in mutated)

    mutated = muts.point_mutations(seqs=seqs, p=0.1, p_indel=1.0, p_del=0.0)
    assert len(mutated) > 50  # chance >= 99.7%
    assert all(len(d) > 10 for d, _ in mutated)

    seqs = ["TACCCAAGGA"] * 60 + ["AGTAACCA"] * 20 + ["AGACCGAATTAG"] * 20
    mutated = muts.point_mutations(seqs=seqs, p=0.1)
    assert len(mutated) > 50  # chance >= 99.7%

    are_ok = 0
    for new, idx in mutated:
        if seqs[idx] != new:
            are_ok += 1
    assert are_ok > len(mutated) * 0.7


def test_recombinations():
    seq_pair = ("AAAAAAAAAA", "CCCCCCCCCC")

    mutated = muts.recombinations(seq_pairs=[seq_pair] * 50, p=1 / 20)
    assert len(mutated) > 20

    is_ok = 0
    for lft, rgt, _ in mutated:
        if set(lft) != {"A"} or set(rgt) != {"C"}:
            is_ok += 1
    assert is_ok > len(mutated) * 0.7
