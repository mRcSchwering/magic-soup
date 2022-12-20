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
    genetics.summary()

    genomes = genetics.random_genomes(n=n_cells, s=init_genome_size)
    proteomes = genetics.get_proteomes(sequences=genomes)

    # test
    kinetics = Kinetics(n_signals=len(MOLECULES) * 2)
    kinetics.increase_max_cells(by_n=len(proteomes))
    kinetics.increase_max_proteins(max_n=max(len(d) for d in proteomes))

    with timeit(
        f"set cell params for {len(proteomes):,} proteomes from genomes of size {init_genome_size}"
    ):
        cell_prots = []
        for cell_i, cell in enumerate(proteomes):
            for prot_i, prot in enumerate(cell):
                cell_prots.append((cell_i, prot_i, prot))
        kinetics.set_cell_params(cell_prots=cell_prots)

    exit()

    n_threads = torch.get_num_threads()
    print(f"torch n threads {n_threads}")

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
    genetics.summary()

    world = ms.World(molecules=MOLECULES)


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

