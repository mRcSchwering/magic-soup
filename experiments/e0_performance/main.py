from argparse import ArgumentParser
from contextlib import contextmanager
import time
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from magicsoup.constants import NOW
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS, ATP

_this_dir = Path(__file__).parent


@contextmanager
def timeit(label: str, step: int, writer: SummaryWriter):
    t0 = time.time()
    yield
    writer.add_scalar(f"Time[s]/{label}", time.time() - t0, step)


def main(n_cells: int, n_steps: int, rand_genome_size: int):
    writer = SummaryWriter(log_dir=_this_dir / "runs" / NOW)

    chemistry = ms.Chemistry(reactions=REACTIONS, molecules=MOLECULES)
    world = ms.World(chemistry=chemistry)
    world.save(outdir=_this_dir / "runs" / NOW)

    for step_i in range(n_steps):

        if step_i % 100 == 0:
            world.save_state(statedir=_this_dir / "runs" / NOW / f"step={step_i}")

        with timeit("perStep", step_i, writer):
            with timeit("addCells", step_i, writer):
                genomes = [ms.random_genome(rand_genome_size) for _ in range(n_cells)]
                world.add_random_cells(genomes=genomes)

            with timeit("activity", step_i, writer):
                world.enzymatic_activity()

            with timeit("kill", step_i, writer):
                kill_idxs = (
                    torch.argwhere(world.cell_molecules[:, ATP.idx] < 1.0)
                    .flatten()
                    .tolist()
                )
                world.kill_cells(cell_idxs=kill_idxs)

            with timeit("replicate", step_i, writer):
                rep_idxs = (
                    torch.argwhere(world.cell_molecules[:, ATP.idx] > 5.0)
                    .flatten()
                    .tolist()
                )
                world.cell_molecules[rep_idxs, ATP.idx] -= 4.0
                world.replicate_cells(parent_idxs=rep_idxs)

            with timeit("mutateGenomes", step_i, writer):
                chgd_genomes = ms.point_mutations(seqs=[d.genome for d in world.cells])
                new_gs, chgd_idxs = list(map(list, zip(*chgd_genomes)))

            with timeit("getMutatedProteomes", step_i, writer):
                world.update_cells(genomes=new_gs, idxs=chgd_idxs)  # type: ignore

            with timeit("wrapUp", step_i, writer):
                world.degrade_molecules()
                world.diffuse_molecules()
                world.increment_cell_survival()

            writer.add_scalar("Cells/total", len(world.cells), step_i)
            writer.add_scalar(
                "Cells/MeanSurv", world.cell_survival.mean().item(), step_i
            )
            writer.add_scalar("Cells/MaxSurv", world.cell_survival.max().item(), step_i)
            writer.add_scalar("Other/MaxProteins", world.kinetics.Km.shape[1], step_i)
            writer.add_scalar(
                "Other/AvgATPint", world.cell_molecules[:, ATP.idx].mean(), step_i
            )
            writer.add_scalar(
                "Other/AvgATPext", world.molecule_map[ATP.idx].mean(), step_i
            )

            if step_i % 5 == 0:
                writer.add_image("Maps/Cells", world.cell_map, step_i, dataformats="HW")
                writer.add_image(
                    "Maps/ATP", world.molecule_map[ATP.idx], step_i, dataformats="HW"
                )

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_cells", default=100, type=int)
    parser.add_argument("--n_steps", default=100, type=int)
    parser.add_argument("--init_genome_size", default=500, type=int)
    args = parser.parse_args()

    main(
        n_cells=args.n_cells,
        n_steps=args.n_steps,
        rand_genome_size=args.init_genome_size,
    )

