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
        n_ok = sum(genomes[i] != d for i, d in enumerate(res))
        assert n_ok / len(res) > 0.9  # highly likely
