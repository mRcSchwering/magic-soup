"""
Run simple simulation to create data for visualizations

    PYTHONPATH=./python python docs/run.py --help

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


def _kill_cells(world: ms.World, i_atp: int, k: float):
    x = world.cell_molecules[:, i_atp]
    idxs = _sample(k**7 / (k**7 + x**7))
    world.kill_cells(cell_idxs=idxs)


def _replicate_cells(world: ms.World, i_aca: int, i_hca: int, k: float, cost=2.0):
    x = world.cell_molecules[:, i_aca]
    sampled_idxs = _sample(x**5 / (x**5 + k**5))

    can_replicate = world.cell_molecules[:, i_aca] > cost
    allowed_idxs = torch.argwhere(can_replicate).flatten().tolist()

    idxs = list(set(sampled_idxs) & set(allowed_idxs))
    replicated = world.divide_cells(cell_idxs=idxs)

    if len(replicated) > 0:
        descendants = [dd for d in replicated for dd in d]
        world.cell_molecules[descendants, i_aca] -= cost / 2
        world.cell_molecules[descendants, i_hca] += cost / 2


def _mutate_cells(world: ms.World, old=10):
    world.mutate_cells()
    is_old = torch.argwhere(world.cell_lifetimes > old)
    world.recombinate_cells(cell_idxs=is_old.flatten().tolist())


def _activity(world: ms.World, i_atp: int, i_adp: int):
    world.cell_molecules[:, i_atp] -= 0.01
    world.cell_molecules[:, i_adp] += 0.01
    world.cell_molecules[world.cell_molecules[:, i_atp] < 0.0, i_atp] = 0.0
    world.enzymatic_activity()
    world.degrade_molecules()
    world.diffuse_molecules()
    world.increment_cell_lifetimes()


def main(args: Namespace):
    writer = SummaryWriter(log_dir=_this_dir / "runs" / _now)

    world = ms.World(
        chemistry=CHEMISTRY,
        map_size=args.map_size,
        mol_map_init=args.init_molmap,
        device=args.device,
    )
    world.save(rundir=_this_dir / "runs" / _now)

    ATP_IDX = CHEMISTRY.molname_2_idx["ATP"]
    ADP_IDX = CHEMISTRY.molname_2_idx["ADP"]
    ACA_IDX = CHEMISTRY.molname_2_idx["acetyl-CoA"]
    HCA_IDX = CHEMISTRY.molname_2_idx["HS-CoA"]

    genomes = [
        ms.random_genome(args.init_genome_size) for _ in range(args.init_n_cells)
    ]
    world.spawn_cells(genomes=genomes)

    for step_i in range(args.n_steps):
        _activity(world=world, i_atp=ATP_IDX, i_adp=ADP_IDX)
        _kill_cells(world=world, i_atp=ATP_IDX, k=args.k_kill)
        _replicate_cells(world=world, i_aca=ACA_IDX, i_hca=HCA_IDX, k=args.k_replicate)
        _mutate_cells(world=world)

        if step_i % args.check_every_n == 0:
            if args.save_state:
                world.save_state(statedir=_this_dir / "runs" / _now / f"step={step_i}")
            _log_imgs(writer=writer, world=world, step_i=step_i)
            _log_scalars(
                writer=writer,
                world=world,
                step_i=step_i,
                mol_2_idx=CHEMISTRY.molname_2_idx,
            )

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--map-size", default=128, type=int)
    parser.add_argument("--n-steps", default=10_000, type=int)
    parser.add_argument("--init-n-cells", default=1000, type=int)
    parser.add_argument("--init-genome-size", default=500, type=int)
    parser.add_argument("--k-kill", default=1.0, type=float)
    parser.add_argument("--k-replicate", default=15.0, type=float)
    parser.add_argument("--check-every-n", default=25, type=int)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--init-molmap", default="randn", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parsed_args = parser.parse_args()

    main(parsed_args)
