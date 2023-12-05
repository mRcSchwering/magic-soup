"""
Dummy run to test simulation performance in realistic environment

    PYTHONPATH=./python python performance/run.py --help
    ...
    tensorboard --host 0.0.0.0 --logdir=./performance/runs

Last runs:

- 2023-06-08 EC2 GPU: 0.7s / 1k cells, 4.5s / 40k cells (0.9s activity, 1.2 mutate genomes, 2 replicate)
- 2023-11-21 EC2 GPU: 0.05s / 1k cells, 1.2s / 40k cells (0.3s activity, 0.7 mutate genomes, 0.03 replicate)
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

    world = ms.World(
        chemistry=CHEMISTRY,
        map_size=args.map_size,
        mol_map_init=args.init_molmap,
        device=args.device,
    )
    world.save(rundir=_this_dir / "runs" / _now)

    mol_2_idx = {d.name: i for i, d in enumerate(CHEMISTRY.molecules)}
    ATP_IDX = mol_2_idx["ATP"]

    for step_i in range(args.n_steps):
        if step_i % 100 == 0:
            world.save_state(statedir=_this_dir / "runs" / _now / f"step={step_i}")

        with timeit("perStep", step_i, writer):
            n_cells = world.n_cells
            if n_cells < 1000:
                with timeit("addCells", step_i, writer):
                    genomes = [
                        ms.random_genome(args.init_genome_size)
                        for _ in range(1000 - n_cells)
                    ]
                    world.spawn_cells(genomes=genomes)

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
                replicated = world.divide_cells(cell_idxs=repl)

                genome_pairs = [
                    (world.cell_genomes[p], world.cell_genomes[c])
                    for p, c in replicated
                ]
                recombs = ms.recombinations(seq_pairs=genome_pairs)

                genome_idx_pairs = []
                for parent, child, idx in recombs:
                    parent_i, child_i = replicated[idx]
                    genome_idx_pairs.append((parent, parent_i))
                    genome_idx_pairs.append((child, child_i))
                world.update_cells(genome_idx_pairs=genome_idx_pairs)

            with timeit("mutateGenomes", step_i, writer):
                snps = ms.point_mutations(seqs=world.cell_genomes)

            with timeit("getMutatedProteomes", step_i, writer):
                world.update_cells(genome_idx_pairs=snps)

            with timeit("wrapUp", step_i, writer):
                world.degrade_molecules()
                world.diffuse_molecules()
                world.increment_cell_lifetimes()

        writer.add_scalar("Cells/total", world.n_cells, step_i)

        molmap = world.molecule_map
        cellmols = world.cell_molecules
        for mol in CHEMISTRY.molecules:
            mol_i = mol_2_idx[mol.name]
            d = molmap[mol_i].sum().item()
            n = world.map_size**2
            if world.n_cells > 0:
                d += cellmols[:, mol_i].sum().item()
                n += world.n_cells
            writer.add_scalar(f"Molecules/{mol.name}", d / n, step_i)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--map-size", default=256, type=int)
    parser.add_argument("--n-steps", default=1000, type=int)
    parser.add_argument("--init-genome-size", default=500, type=int)
    parser.add_argument("--init-molmap", default="randn", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parsed_args = parser.parse_args()

    main(parsed_args)
