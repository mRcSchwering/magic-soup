from argparse import ArgumentParser
from contextlib import contextmanager
import time
import torch
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS
from magicsoup.kinetics import Kinetics


@contextmanager
def timeit(msg: str):
    t0 = time.time()
    yield
    td = time.time() - t0
    print(f"{msg}: {td:.2f}s")


def main(loglevel: str, n_cells: int, n_steps: int, init_genome_size: int):

    genomes = [ms.random_genome(s=100) for _ in range(n_cells)]

    with timeit(f"check for {len(genomes):,} genomes"):
        # len(list(filter(lambda seq: sum(d == "A" for d in seq) / len(seq) > 0.25, genomes)))
        len([seq for seq in genomes if sum(d == "A" for d in seq) / len(seq) > 0.25])

    # with timeit(f"check for {len(genomes):,} genomes"):
    #     [seq for seq in genomes if sum(d == "A" for d in seq) > 0.3]

    return

    # fmt: off
    domains = {
        ms.CatalyticFact(): ms.variants("ACNTGN") + ms.variants("AGNTGN") + ms.variants("CCNTTN"),
        ms.TransporterFact(): ms.variants("ACNAGN") + ms.variants("ACNTAN") + ms.variants("AANTCN"),
        ms.AllostericFact(is_transmembrane=False, is_inhibiting=False): ms.variants("GGNANN"),
        ms.AllostericFact(is_transmembrane=False, is_inhibiting=True): ms.variants("GGNTNN"),
        ms.AllostericFact(is_transmembrane=True, is_inhibiting=False): ms.variants("GGNCNN"),
        ms.AllostericFact(is_transmembrane=True, is_inhibiting=True): ms.variants("GGNGNN"),
    }
    # fmt: on

    genetics = ms.Genetics(
        domain_facts=domains, molecules=MOLECULES, reactions=REACTIONS,
    )

    genomes = genetics.random_genomes(n=n_cells, s=init_genome_size)

    with timeit(f"get_coding_regions for {len(genomes):,} genomes"):
        _ = [genetics.get_coding_regions(d) for d in genomes]

    return

    with timeit(f"reverse_complement for {len(genomes):,} genomes"):
        bkwds = [ms.reverse_complement(d) for d in genomes]

    with timeit(f"get_coding_regions for {len(genomes):,} genomes with complement"):
        cdss = []
        for seq, bwk in zip(genomes, bkwds):
            cds = list(
                set(genetics.get_coding_regions(seq) + genetics.get_coding_regions(bwk))
            )
            cds = [d for d in cds if len(d) > genetics.min_n_seq_nts]
            cdss.append(cds)

    with timeit(f"translate_seq for {sum(len(d) for d in cdss):,} coding regions"):
        proteomes = []
        for cds in cdss:
            proteins = [genetics.translate_seq(d) for d in cds]
            proteins = [d for d in proteins if len(d) > 0]
            proteins = [d for d in proteins if not all(dd.is_allosteric for dd in d)]
            proteomes.append(proteins)

    with timeit(f"Protein for {len(proteomes):,} proteomes"):
        prots = [[ms.Protein(domains=d) for d in p] for p in proteomes]

    exit()

    with timeit(f"mutateGenomes for {len(genomes)} genomes of size {init_genome_size}"):
        new_gs, chgd_idxs = ms.point_mutatations(seqs=genomes)

    with timeit(f"getMutatedProteomes for {len(new_gs)} muated genomes"):
        new_ps = genetics.get_proteomes(sequences=new_gs)

    exit()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--log", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING",
    )
    parser.add_argument("--n_cells", default=1000, type=int)
    parser.add_argument("--n_steps", default=100, type=int)
    parser.add_argument("--init_genome_size", default=500, type=int)
    args = parser.parse_args()

    main(
        loglevel=args.log,
        n_cells=args.n_cells,
        n_steps=args.n_steps,
        init_genome_size=args.init_genome_size,
    )

