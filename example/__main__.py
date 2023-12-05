"""
Dummy run to test simulation performance in realistic environment

    PYTHONPATH=./python python -m example --help
    ...
    tensorboard --host 0.0.0.0 --logdir=./example/runs
"""
from argparse import ArgumentParser, Namespace
from pathlib import Path
import datetime as dt
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms  # pylint: disable=E0401
from magicsoup.examples.wood_ljungdahl import CHEMISTRY  # pylint: disable=E0401,E0611

_this_dir = Path(__file__).parent
_now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M")


def _log_scalars(
    step_i: int, writer: SummaryWriter, world: ms.World, mol_2_idx: dict[str, int]
):
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


def _log_imgs(step_i: int, writer: SummaryWriter, world: ms.World):
    writer.add_image("Maps/Cells", world.cell_map, step_i, dataformats="WH")


def _sample(p: torch.Tensor) -> list[int]:
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def _kill_cells(world: ms.World, atp: int):
    x0 = world.cell_molecules[:, atp]
    idxs = _sample(1.0**7 / (1.0**7 + x0**7))
    world.kill_cells(cell_idxs=idxs)


def _replicate_cells(world: ms.World, aca: int, hca: int):
    x = world.cell_molecules[:, aca]
    chosen = _sample(x**5 / (x**5 + 15.0**5))

    allowed = torch.argwhere(world.cell_molecules[:, aca] > 2.0).flatten().tolist()
    idxs = list(set(chosen) & set(allowed))

    replicated = world.divide_cells(cell_idxs=idxs)
    if len(replicated) > 0:
        parents, children = list(map(list, zip(*replicated)))
        world.cell_molecules[parents + children, aca] -= 1.0
        world.cell_molecules[parents + children, hca] += 1.0


def _mutate_cells(world: ms.World):
    mutated = ms.point_mutations(seqs=world.cell_genomes)
    world.update_cells(genome_idx_pairs=mutated)


def _activity(world: ms.World, atp: int, adp: int):
    world.cell_molecules[:, atp] -= 0.01
    world.cell_molecules[:, adp] += 0.01
    world.cell_molecules[world.cell_molecules[:, atp] < 0.0, atp] = 0.0
    world.enzymatic_activity()
    world.degrade_molecules()
    world.diffuse_molecules()
    world.increment_cell_lifetimes()


def main(args: Namespace):
    min_cells = int(args.init_n_cells * 0.01)
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
    ADP_IDX = mol_2_idx["ADP"]
    ACA_IDX = mol_2_idx["acetyl-CoA"]
    HCA_IDX = mol_2_idx["HS-CoA"]

    genomes = [
        ms.random_genome(args.init_genome_size) for _ in range(args.init_n_cells)
    ]
    world.spawn_cells(genomes=genomes)

    for step_i in range(args.n_steps):
        _activity(world=world, atp=ATP_IDX, adp=ADP_IDX)
        _kill_cells(world=world, atp=ATP_IDX)
        _replicate_cells(world=world, aca=ACA_IDX, hca=HCA_IDX)
        _mutate_cells(world=world)

        if step_i % 25 == 0:
            world.save_state(statedir=_this_dir / "runs" / _now / f"step={step_i}")
            _log_imgs(writer=writer, world=world, step_i=step_i)
            _log_scalars(writer=writer, world=world, step_i=step_i, mol_2_idx=mol_2_idx)

        if world.n_cells < min_cells:
            print(f"Only {world.n_cells:,} cells left, stopping")
            break

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--map-size", default=128, type=int)
    parser.add_argument("--n-steps", default=10_000, type=int)
    parser.add_argument("--init-n-cells", default=1000, type=int)
    parser.add_argument("--init-genome-size", default=500, type=int)
    parser.add_argument("--init-molmap", default="randn", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parsed_args = parser.parse_args()

    main(parsed_args)
