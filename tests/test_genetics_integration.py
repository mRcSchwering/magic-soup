import magicsoup as ms


def test_genomes_are_always_translated_reproduceably():
    genetics = ms.Genetics()
    for i in range(100):
        g = ms.random_genome(s=500)
        original_proteome, *_ = genetics.translate_genomes(genomes=[g])

        proteomes = genetics.translate_genomes(genomes=[g] * 100)
        for proteome in proteomes:
            assert proteome == original_proteome, i
