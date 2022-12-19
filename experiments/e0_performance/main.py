from argparse import ArgumentParser
from contextlib import contextmanager
import logging
import time
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS, ATP

_log = logging.getLogger(__name__)
_this_dir = Path(__file__).parent


@contextmanager
def timeit(label: str, step: int, writer: SummaryWriter):
    t0 = time.time()
    yield
    writer.add_scalar(f"Perf[s]/{label}", time.time() - t0, step)


def main(loglevel: str, n_cells: int, n_steps: int, rand_genome_size: int):
    logging.basicConfig(
        level=getattr(logging, loglevel.upper()),
        format="%(levelname)s::%(asctime)s::%(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    writer = SummaryWriter(log_dir=_this_dir / "runs")
    n_threads = torch.get_num_threads()
    _log.info("torch n threads %i", n_threads)

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
    world.summary()

    idx_ATP = world.get_intracellular_molecule_idxs(molecules=[ATP])[0]

    for step_i in range(n_steps):

        with timeit("addCells", step_i, writer):
            genomes = ms.random_genomes(n=n_cells, s=rand_genome_size)
            proteomes = genetics.get_proteomes(sequences=genomes)
            cells = [ms.Cell(genome=g, proteome=p) for g, p in zip(genomes, proteomes)]
            world.add_random_cells(cells=cells)

        with timeit("activity", step_i, writer):
            world.enzymatic_activity()

        with timeit("kill", step_i, writer):
            kill_idxs = (
                torch.argwhere(world.cell_molecules[:, idx_ATP] < 1.0)
                .flatten()
                .tolist()
            )
            world.kill_cells(cell_idxs=kill_idxs)

        with timeit("replicate", step_i, writer):
            rep_idxs = (
                torch.argwhere(world.cell_molecules[:, idx_ATP] > 5.0)
                .flatten()
                .tolist()
            )
            cells = world.get_cells(by_idxs=rep_idxs)
            world.cell_molecules[rep_idxs, idx_ATP] -= 5.0
            world.replicate_cells(cells=[d.copy() for d in cells])

        with timeit("mutateGenomes", step_i, writer):
            new_gs = ms.point_mutatations(seqs=[d.genome for d in world.cells])

        # TODO: Prio 2: takes > 0.6s
        with timeit("getMutatedProteomes", step_i, writer):
            mut_cells = []
            for new_g, cell in zip(new_gs, world.cells):
                if new_g != cell.genome:
                    new_p = genetics.get_proteome(seq=new_g)
                    new_cell = cell.copy(genome=new_g, proteome=new_p)
                    mut_cells.append(new_cell)

        # TODO: Prio 1, takes > 2s
        with timeit("updateMutatedCells", step_i, writer):
            world.update_cells(mut_cells)

        with timeit("wrapUp", step_i, writer):
            world.degrade_molecules()
            world.diffuse_molecules()
            world.increment_cell_survival()

        writer.add_scalar("Cells/total", len(world.cells), step_i)
        writer.add_scalar("Cells/MeanSurv", world.cell_survival.mean().item(), step_i)
        writer.add_scalar("Cells/MaxSurv", world.cell_survival.max().item(), step_i)
        writer.add_scalar("Other/MaxProteins", world.affinities.shape[1], step_i)
        writer.add_scalar(
            "Other/AvgATPint", world.cell_molecules[:, idx_ATP].mean(), step_i
        )
        writer.add_scalar("Other/AvgATPext", world.molecule_map[idx_ATP].mean(), step_i)

        if step_i % 5 == 0:
            writer.add_image("Maps/Cells", world.cell_map, step_i, dataformats="HW")
            writer.add_image(
                "Maps/ATP", world.molecule_map[idx_ATP], step_i, dataformats="HW"
            )

    writer.close()

    world.summary()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--log", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="WARNING",
    )
    parser.add_argument("--n_cells", default=100, type=int)
    parser.add_argument("--n_steps", default=100, type=int)
    parser.add_argument("--init_genome_size", default=500, type=int)
    args = parser.parse_args()

    main(
        loglevel=args.log,
        n_cells=args.n_cells,
        n_steps=args.n_steps,
        rand_genome_size=args.init_genome_size,
    )

