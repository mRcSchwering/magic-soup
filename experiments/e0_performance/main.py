from argparse import ArgumentParser
from contextlib import contextmanager
import time
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from magicsoup.constants import NOW
from magicsoup.examples.wood_ljungdahl import CHEMISTRY, ATP

_this_dir = Path(__file__).parent


@contextmanager
def timeit(label: str, step: int, writer: SummaryWriter):
    t0 = time.time()
    yield
    writer.add_scalar(f"Time[s]/{label}", time.time() - t0, step)


def main(n_cells: int, n_steps: int, rand_genome_size: int):
    writer = SummaryWriter(log_dir=_this_dir / "runs" / NOW)

    world = ms.World(chemistry=CHEMISTRY)
    world.save(outdir=_this_dir / "runs" / NOW)

    for step_i in range(n_steps):
        if step_i % 100 == 0:
            world.save_state(statedir=_this_dir / "runs" / NOW / f"step={step_i}")

        with timeit("perStep", step_i, writer):
            if len(world.cells) < 1000:
                with timeit("addCells", step_i, writer):
                    seqs = [ms.random_genome(rand_genome_size) for _ in range(n_cells)]
                    world.add_random_cells(genomes=seqs)

            with timeit("activity", step_i, writer):
                world.enzymatic_activity()

            with timeit("kill", step_i, writer):
                kill = (
                    torch.argwhere((world.cell_molecules[:, ATP.idx] < 1.0))
                    .flatten()
                    .tolist()
                )
                world.kill_cells(cell_idxs=kill)

            with timeit("replicate", step_i, writer):
                repl = (
                    torch.argwhere(world.cell_molecules[:, ATP.idx] > 5.0)
                    .flatten()
                    .tolist()
                )
                world.cell_molecules[repl, ATP.idx] -= 4.0
                parent_child_idxs = world.replicate_cells(parent_idxs=repl)

                seq_pairs = [
                    (world.cells[p].genome, world.cells[c].genome)
                    for p, c in parent_child_idxs
                ]
                chgnd = ms.recombinations(seq_pairs=seq_pairs)

                genome_idx_pairs = []
                for parent, child, idx in chgnd:
                    parent_i, child_i = parent_child_idxs[idx]
                    genome_idx_pairs.append((parent, parent_i))
                    genome_idx_pairs.append((child, child_i))
                world.update_cells(genome_idx_pairs=genome_idx_pairs)

            with timeit("mutateGenomes", step_i, writer):
                chgd_genomes = ms.point_mutations(seqs=[d.genome for d in world.cells])

            with timeit("getMutatedProteomes", step_i, writer):
                world.update_cells(genome_idx_pairs=chgd_genomes)

            with timeit("wrapUp", step_i, writer):
                world.degrade_molecules()
                world.diffuse_molecules()
                world.increment_cell_survival()

        writer.add_scalar("Cells/total", len(world.cells), step_i)
        writer.add_scalar("Cells/MeanSurv", world.cell_survival.mean().item(), step_i)
        writer.add_scalar("Cells/MaxSurv", world.cell_survival.max().item(), step_i)

        writer.add_scalar(
            "Molecules/AvgATP", world.cell_molecules[:, ATP.idx].mean(), step_i
        )

        if step_i % 10 == 0:
            writer.add_image("Maps/Cells", world.cell_map, step_i, dataformats="HW")

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

