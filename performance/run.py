"""
Dummy run to test simulation performance in realistic environment

    PYTHONPATH=./src python performance/run.py --n_steps=100

"""
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
import time
from pathlib import Path
import datetime as dt
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import CHEMISTRY

_this_dir = Path(__file__).parent
_now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")


@contextmanager
def timeit(label: str, step: int, writer: SummaryWriter):
    t0 = time.time()
    yield
    writer.add_scalar(f"Time[s]/{label}", time.time() - t0, step)


def main(args: Namespace):
    writer = SummaryWriter(log_dir=_this_dir / "runs" / _now)

    world = ms.World(chemistry=CHEMISTRY, device=args.device)
    world.save(rundir=_this_dir / "runs" / _now)

    mol_2_idx = {d.name: i for i, d in enumerate(CHEMISTRY.molecules)}
    ATP_IDX = mol_2_idx["ATP"]

    for step_i in range(args.n_steps):
        if step_i % 100 == 0:
            world.save_state(statedir=_this_dir / "runs" / _now / f"step={step_i}")

        with timeit("perStep", step_i, writer):
            n_cells = len(world.cells)
            if n_cells < 1000:
                with timeit("addCells", step_i, writer):
                    genomes = [
                        ms.random_genome(args.init_genome_size)
                        for _ in range(1000 - n_cells)
                    ]
                    world.add_random_cells(genomes=genomes)

            with timeit("activity", step_i, writer):
                world.enzymatic_activity()

            with timeit("kill", step_i, writer):
                kill = (
                    torch.argwhere((world.cell_molecules[:, ATP_IDX] < 1.0))
                    .flatten()
                    .tolist()
                )
                world.kill_cells(cell_idxs=kill)

            with timeit("replicate", step_i, writer):
                repl = (
                    torch.argwhere(world.cell_molecules[:, ATP_IDX] > 5.0)
                    .flatten()
                    .tolist()
                )
                world.cell_molecules[repl, ATP_IDX] -= 4.0
                replicated = world.replicate_cells(parent_idxs=repl)

                genomes = [
                    (world.cells[p].genome, world.cells[c].genome)
                    for p, c in replicated
                ]
                mutated = ms.recombinations(seq_pairs=genomes)

                genome_idx_pairs = []
                for parent, child, idx in mutated:
                    parent_i, child_i = replicated[idx]
                    genome_idx_pairs.append((parent, parent_i))
                    genome_idx_pairs.append((child, child_i))
                world.update_cells(genome_idx_pairs=genome_idx_pairs)

            with timeit("mutateGenomes", step_i, writer):
                mutated = ms.point_mutations(seqs=[d.genome for d in world.cells])

            with timeit("getMutatedProteomes", step_i, writer):
                world.update_cells(genome_idx_pairs=mutated)

            with timeit("wrapUp", step_i, writer):
                world.degrade_molecules()
                world.diffuse_molecules()
                world.increment_cell_survival()

        writer.add_scalar("Cells/total", len(world.cells), step_i)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_steps", default=100, type=int)
    parser.add_argument("--init_genome_size", default=500, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parsed_args = parser.parse_args()

    main(parsed_args)