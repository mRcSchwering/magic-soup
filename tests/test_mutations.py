from magicsoup.mutations import Mutator
from magicsoup.util import random_genome


def test_mutator() -> None:
    mutator = Mutator(sub_rate=1e-6)

    genomes = [random_genome() for _ in range(10_000)]
    res = mutator.mutate(genomes=genomes)

    for idx, seqa in res:
        print(f"{idx}:")
        seqb = genomes[idx]
        for i in range(0, len(seqa), 100):
            a = seqa[i : i + 100]
            b = seqb[i : i + 100]
            print(" ", a)
            print(" ", "".join(["|" if u == v else " " for u, v in zip(a, b)]))
            print(" ", b)
            print()

    assert False
