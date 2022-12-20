from argparse import ArgumentParser
from contextlib import contextmanager
import logging
from multiprocessing import Pool
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from magicsoup.constants import EPS
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS, ATP

_log = logging.getLogger(__name__)


@contextmanager
def timeit(msg: str):
    t0 = time.time()
    yield
    td = time.time() - t0
    print(f"{msg}: {td:.2f}s")


def main(loglevel: str, n_cells: int, n_steps: int, init_genome_size: int):
    n_cells = 1000
    n_prots = 100
    n_mols = 20
    n_steps = 10

    # concentrations (c, s)
    X = torch.randn(n_cells, n_mols).abs() + EPS

    # reactions (c, p, s)
    N = torch.randint(low=-3, high=4, size=(n_cells, n_prots, n_mols))

    # affinities (c, p, s)
    Km = torch.randn(n_cells, n_prots, n_mols).abs() + EPS

    # max velocities (c, p)
    Vmax = torch.randn(n_cells, n_prots).abs() * 10

    # allosterics (c, p, s)
    A = torch.randint(low=-2, high=3, size=(n_cells, n_prots, n_mols))

    # equilibrium constants (c, p)
    Ke = torch.randn(n_cells, n_prots).abs() * 2

    # test
    t0 = time.time()
    for _ in range(n_steps):
        Xd = ms.integrate_signals(X=X, Km=Km, Vmax=Vmax, Ke=Ke, N=N, A=A)
        X = X + Xd
    print(time.time() - t0)

    exit()

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

    with timeit(
        f"point mutations for {len(proteomes):,} proteomes from genomes of size {init_genome_size}"
    ):
        new_gs, idxs = ms.point_mutatations(seqs=genomes)
        new_ps = genetics.get_proteomes(sequences=new_gs)

    with timeit(
        f"comparing {len(proteomes):,} proteomes from genomes of size {init_genome_size}"
    ):
        for idx, p2 in zip(idxs, new_ps):
            _ = proteomes[idx] == p2
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

