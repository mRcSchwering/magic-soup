from ..conftest import gen_genomes
import magicsoup.mutations as muts


def test_point_mutations():
    n = 10_000
    s = 10_000
    r = 3
    for _ in range(r):
        genomes = gen_genomes(n=n, s=s)
        res = muts.point_mutations(seqs=genomes, p=1e-4)
        assert len(res) <= len(genomes)
        n_ok = sum(genomes[i] != d for d, i in res)
        assert n_ok / len(res) > 0.9  # highly likely


def test_recombinations():
    n = 10_000
    s = 10_000
    r = 3

    def totlen(iter):
        return len(iter[0]) + len(iter[1])

    for _ in range(r):
        genomes = gen_genomes(n=n, s=s)
        pairs = list(zip(genomes, reversed(genomes)))
        res = muts.recombinations(seq_pairs=pairs)
        assert len(res) <= len(genomes)
        assert all(totlen([a, b]) == totlen(pairs[i]) for a, b, i in res)
