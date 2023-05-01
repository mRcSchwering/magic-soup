import magicsoup.mutations as muts


def test_point_mutations():
    seqs = ["AAAAACCCCCTTTTTGGGGG"] * 50
    mutated = muts.point_mutations(seqs=seqs, p=1 / 20, p_indel=0.0)
    assert len(mutated) > 25

    are_ok = 0
    for new, idx in mutated:
        old = seqs[idx]
        assert len(new) == len(old)
        d = sum(a != b for a, b in zip(old, new))
        if 0 < d < 5:
            are_ok += 1
    assert are_ok > len(mutated) * 0.7

    mutated = muts.point_mutations(seqs=seqs, p=1 / 20, p_indel=1.0)
    assert len(mutated) > 25

    are_ok = 0
    for new, idx in mutated:
        old = seqs[idx]
        if len(new) != len(old):
            are_ok += 1
    assert are_ok > len(mutated) * 0.7


def test_recombinations():
    seq_pair = ("AAAAAAAAAA", "CCCCCCCCCC")

    mutated = muts.recombinations(seq_pairs=[seq_pair] * 50, p=1 / 20)
    assert len(mutated) > 25

    is_ok = 0
    for lft, rgt, _ in mutated:
        if set(lft) != {"A"} or set(rgt) != {"C"}:
            is_ok += 1
    assert is_ok > len(mutated) * 0.7
