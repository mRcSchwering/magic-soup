"""
Dummy run to test simulation performance in realistic environment.
For local python package:

    PYTHONPATH=./python python performance/run.py --help

For installed python package:

    python run_simulation.py --help

Check running simulation:

    nvidia-smi -l 1
    tensorboard --host 0.0.0.0 --logdir=./performance/runs

Last runs:

- 2023-06-08 EC2 GPU: 0.7s / 1k cells, 4.5s / 40k cells (0.9s activity, 1.2 mutate genomes, 2 replicate)
- 2023-11-21 EC2 GPU: 0.05s / 1k cells, 0.35s / 40k cells (0.30s activity, 0.03 mutate genomes, 0.08 replicate peak)
- 2023-12-19 EC2 GPU: 0.03s / 1k cells, 0.30s / 40k cells (0.25s activity, 0.03 mutate genomes, 0.05 replicate peak)
"""

import datetime as dt
import time
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from pathlib import Path

import magicsoup as ms
import torch
from magicsoup.examples.wood_ljungdahl import CHEMISTRY
from torch.utils.tensorboard import SummaryWriter

_this_dir = Path(__file__).parent
_now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d_%H-%M")


@contextmanager
def timeit(label: str, step: int, writer: SummaryWriter):
    t0 = time.time()
    yield
    writer.add_scalar(f"Time[s]/{label}", time.time() - t0, step)


@torch.inference_mode()
def main(args: Namespace):
    writer = SummaryWriter(log_dir=_this_dir / "runs" / _now)

    world = ms.World(
        chemistry=CHEMISTRY,
        culture_kwargs={"map_size": args.map_size},
        tensor_kwargs={"device": args.device},
    )
    world.culture.molecule_map[:] = 10
    world.save(rundir=_this_dir / "runs" / _now)

    mol_2_idx = {d.name: i for i, d in enumerate(CHEMISTRY.molecules)}
    ATP_IDX = mol_2_idx["ATP"]

    for step_i in range(args.n_steps):
        if step_i % 100 == 0:
            world.save_state(statedir=_this_dir / "runs" / _now / f"step={step_i}")

        with timeit("perStep", step_i, writer):
            n_cells = world.cells.get_n_cells()
            if n_cells < 1000:
                with timeit("addCells", step_i, writer):
                    genomes = [
                        ms.random_genome(args.init_genome_size)
                        for _ in range(1000 - n_cells)
                    ]
                    world.spawn_cells(genomes=genomes)

            with timeit("advanceTime", step_i, writer):
                world.advance_time()

            with timeit("kill", step_i, writer):
                kill = torch.argwhere(world.cells.molecules[:, ATP_IDX] < 1.0)
                world.kill_cells(cell_idxs=kill)

            with timeit("replicate", step_i, writer):
                repl = torch.argwhere(world.cells.molecules[:, ATP_IDX] > 5.0)
                world.cells.molecules[repl, ATP_IDX] -= 4.0
                world.divide_cells(cell_idxs=repl)

            # with timeit("mutateGenomes", step_i, writer):
            #     world.mutate_cells()

        writer.add_scalar("Cells/total", n_cells, step_i)

        molmap = world.culture.molecule_map
        cellmols = world.cells.molecules
        for mol in CHEMISTRY.molecules:
            mol_i = mol_2_idx[mol.name]
            d = molmap[mol_i].sum().item()
            n = world.culture.map_size**2
            if n_cells > 0:
                d += cellmols[:, mol_i].sum().item() / 1000
            writer.add_scalar(f"Molecules/{mol.name}", d / n, step_i)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--map-size", default=256, type=int)
    parser.add_argument("--n-steps", default=200, type=int)
    parser.add_argument("--init-genome-size", default=500, type=int)
    parser.add_argument("--init-molmap", default="randn", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parsed_args = parser.parse_args()

    main(parsed_args)
