from argparse import ArgumentParser
from contextlib import contextmanager
import logging
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS, ATP

_log = logging.getLogger(__name__)


@contextmanager
def timeit(msg: str):
    t0 = time.time()
    yield
    td = time.time() - t0
    print(f"{msg}: {td:.2f}s")


def generate_genomes(n_cells: int, size: int):
    return ms.random_genomes(n=n_cells, s=size)


def derive_proteomes(genomes: list[str], genetics: ms.Genetics):
    return genetics.get_proteomes(sequences=genomes)


def add_new_cells(
    genomes: list[str], proteomes: list[list[ms.Protein]], world: ms.World
):
    cells = [ms.Cell(genome=g, proteome=p) for g, p in zip(genomes, proteomes)]
    world.add_random_cells(cells=cells)


def enzymatic_activity(world: ms.World):
    world.enzymatic_activity()


def wrap_up_step(world: ms.World):
    world.degrade_molecules()
    world.diffuse_molecules()
    world.increment_cell_survival()


def kill_cells(world: ms.World, idx: int):
    kill_idxs = torch.argwhere(world.cell_molecules[:, idx] < 0.1).flatten().tolist()
    world.kill_cells(cell_idxs=kill_idxs)


def replicate_cells(world: ms.World, idx: int):
    rep_idxs = torch.argwhere(world.cell_molecules[:, idx] > 2.5).flatten().tolist()
    cells = world.get_cells(by_idxs=rep_idxs)
    world.replicate_cells(cells=[d.copy() for d in cells])


def mutate_cells(world: ms.World, genetics: ms.Genetics):
    new_gs = ms.point_mutatations(seqs=[d.genome for d in world.cells])

    mut_cells = []
    for new_g, cell in zip(new_gs, world.cells):
        if new_g != cell.genome:
            new_p = genetics.get_proteome(seq=new_g)
            new_cell = cell.copy(genome=new_g, proteome=new_p)
            mut_cells.append(new_cell)

    world.update_cells(mut_cells)


def one_time_step(world: ms.World, genetics: ms.Genetics, idx: int):
    kill_cells(world=world, idx=idx)
    replicate_cells(world=world, idx=idx)
    mutate_cells(world=world, genetics=genetics)
    enzymatic_activity(world=world)
    wrap_up_step(world=world)


def main(loglevel: str, n_cells: int, n_steps: int, init_genome_size: int):
    logging.basicConfig(
        level=getattr(logging, loglevel.upper()),
        format="%(levelname)s::%(asctime)s::%(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    n_threads = torch.get_num_threads()
    _log.info("torch n threads %i", n_threads)

    # fmt: off
    domains = {
        ms.CatalyticFact(): ms.variants("ACNTGN") + ms.variants("AGNTGN") + ms.variants("CCNTTN"),
        ms.TransporterFact(): ms.variants("ACNAGN") + ms.variants("ACNTAN") + ms.variants("AANTCN"),
        ms.AllostericFact(is_transmembrane=False, is_inhibiting=False): ms.variants("GCNTGN"),
        ms.AllostericFact(is_transmembrane=False, is_inhibiting=True): ms.variants("GCNTAN"),
        ms.AllostericFact(is_transmembrane=True, is_inhibiting=False): ms.variants("AGNTCN"),
        ms.AllostericFact(is_transmembrane=True, is_inhibiting=True): ms.variants("CCNTGN"),
    }
    # fmt: on

    genetics = ms.Genetics(
        domain_facts=domains, molecules=MOLECULES, reactions=REACTIONS,
    )

    world = ms.World(molecules=MOLECULES)

    with timeit(f"Generating {n_cells} genomes"):
        genomes = generate_genomes(n_cells=n_cells, size=init_genome_size)

    with timeit(f"Getting {n_cells} proteomes"):
        proteomes = derive_proteomes(genomes=genomes, genetics=genetics)

    with timeit(f"Adding {n_cells} cells"):
        add_new_cells(genomes=genomes, proteomes=proteomes, world=world)

    with timeit("Integrating signals"):
        enzymatic_activity(world=world)

    with timeit("Degrade, diffuse, increment"):
        wrap_up_step(world=world)

    idx_ATP = world.get_intracellular_molecule_idxs(molecules=[ATP])[0]

    with timeit("Kill cells"):
        kill_cells(world=world, idx=idx_ATP)

    with timeit("Copy and replicate cells"):
        replicate_cells(world=world, idx=idx_ATP)

    with timeit("Mutate cells"):
        mutate_cells(world=world, genetics=genetics)

    with timeit("1 time step"):
        one_time_step(world=world, genetics=genetics, idx=idx_ATP)

    writer = SummaryWriter()

    with timeit(f"{n_steps} time steps"):
        for step_i in range(n_steps):
            t0 = time.time()
            one_time_step(world=world, genetics=genetics, idx=idx_ATP)
            t1 = time.time()

            writer.add_scalar("Cells/total", len(world.cells), step_i)
            writer.add_scalar(
                "Cells/MeanSurv", world.cell_survival.mean().item(), step_i
            )
            writer.add_scalar("Cells/MaxSurv", world.cell_survival.max().item(), step_i)
            writer.add_scalar("Other/MaxProteins", world.affinities.shape[1], step_i)
            writer.add_scalar("Other/SpStep", t1 - t0, step_i)

            if step_i % 2 == 0:
                writer.add_image("Maps/Cells", world.cell_map, step_i, dataformats="HW")
                writer.add_image(
                    "Maps/ATP", world.molecule_map[idx_ATP], step_i, dataformats="HW"
                )

    writer.close()


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

