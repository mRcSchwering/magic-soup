from argparse import ArgumentParser
from contextlib import contextmanager
import logging
import time
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from magicsoup.util import variants, bool_map_fact, weight_map_fact, generic_map_fact
from magicsoup.constants import NOW, CODON_SIZE
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

    writer = SummaryWriter(log_dir=_this_dir / "runs" / NOW)
    n_threads = torch.get_num_threads()
    _log.info("torch n threads %i", n_threads)

    n_region_codons = 2
    max_km = 10.0
    vmax_range = (1, 10)
    codons = variants("N" * n_region_codons * CODON_SIZE)
    reaction_map = generic_map_fact(codons, REACTIONS)
    molecule_map = generic_map_fact(codons, MOLECULES)
    affinity_map = weight_map_fact(codons, 1 / max_km, max_km)
    velocity_map = weight_map_fact(codons, *vmax_range)
    bool_map = bool_map_fact(codons)

    # fmt: off
    domains = {
        ms.CatalyticFact(reaction_map=reaction_map, affinity_map=affinity_map, velocity_map=velocity_map, orientation_map=bool_map):
            ms.variants("ACNTGN") + ms.variants("AGNTGN") + ms.variants("CCNTTN"),
        ms.TransporterFact(molecule_map=molecule_map, affinity_map=affinity_map, velocity_map=velocity_map, orientation_map=bool_map):
            ms.variants("ACNAGN") + ms.variants("ACNTAN") + ms.variants("AANTCN"),
        ms.AllostericFact(molecule_map=molecule_map, affinity_map=affinity_map, is_transmembrane=False, is_inhibiting=False):
            ms.variants("GGNANN"),
        ms.AllostericFact(molecule_map=molecule_map, affinity_map=affinity_map, is_transmembrane=False, is_inhibiting=True):
            ms.variants("GGNTNN"),
        ms.AllostericFact(molecule_map=molecule_map, affinity_map=affinity_map, is_transmembrane=True, is_inhibiting=False):
            ms.variants("GGNCNN"),
        ms.AllostericFact(molecule_map=molecule_map, affinity_map=affinity_map, is_transmembrane=True, is_inhibiting=True):
            ms.variants("GGNGNN"),
    }
    # fmt: on

    world = ms.World(domain_facts=domains, molecules=MOLECULES)
    world.summary()

    idx_ATP = ATP.int_idx

    for step_i in range(n_steps):

        with timeit("addCells", step_i, writer):
            genomes = [ms.random_genome(rand_genome_size) for _ in range(n_cells)]
            world.add_random_cells(genomes=genomes)

        # TODO: takes > 0.4s
        with timeit("activity", step_i, writer):
            world.enzymatic_activity()

        # TODO: takes > 0.15s
        with timeit("kill", step_i, writer):
            kill_idxs = (
                torch.argwhere(world.cell_molecules[:, idx_ATP] < 1.0)
                .flatten()
                .tolist()
            )
            world.kill_cells(cell_idxs=kill_idxs)

        # TODO: takes > 3s
        with timeit("replicate", step_i, writer):
            rep_idxs = (
                torch.argwhere(world.cell_molecules[:, idx_ATP] > 5.0)
                .flatten()
                .tolist()
            )
            succ_parents, children = world.replicate_cells(parent_idxs=rep_idxs)
            world.cell_molecules[succ_parents + children, idx_ATP] -= 4.0

        # TODO: takes > 0.3s
        with timeit("mutateGenomes", step_i, writer):
            new_gs, chgd_idxs = ms.point_mutations(seqs=[d.genome for d in world.cells])

        # TODO: takes > 2s
        with timeit("getMutatedProteomes", step_i, writer):
            world.update_cells(genomes=new_gs, idxs=chgd_idxs)

        with timeit("wrapUp", step_i, writer):
            world.degrade_molecules()
            world.diffuse_molecules()
            world.increment_cell_survival()

        writer.add_scalar("Cells/total", len(world.cells), step_i)
        writer.add_scalar("Cells/MeanSurv", world.cell_survival.mean().item(), step_i)
        writer.add_scalar("Cells/MaxSurv", world.cell_survival.max().item(), step_i)
        writer.add_scalar("Other/MaxProteins", world.kinetics.Km.shape[1], step_i)
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

