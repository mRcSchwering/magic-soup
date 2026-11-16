"""
Run simple simulation to create data for visualizations

    uv run --no-sync python -m docs.gifs.run --help

"""

import datetime as dt
from argparse import ArgumentParser
from pathlib import Path

import magicsoup as ms  # pylint: disable=E0401
from magicsoup.examples.wood_ljungdahl import CHEMISTRY  # pylint: disable=E0401,E0611
from torch.utils.tensorboard import SummaryWriter

_this_dir = Path(__file__).parent
_now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d_%H-%M")


def _log_scalars(step_i: int, writer: SummaryWriter, world: ms.World) -> None:
    n_cells = world.cells.get_n_cells()
    writer.add_scalar("Cells/total", n_cells, step_i)

    m2i = world.chemistry.mol_2_idx
    molmap = world.culture.molecule_map
    cellmols = world.cells.molecules
    for mol in CHEMISTRY.molecules:
        mol_i = m2i[mol]
        d = molmap[mol_i].sum().item()
        n = world.culture.map_size**2
        if n_cells > 0:
            d += cellmols[:, mol_i].sum().item()
            n += n_cells
        writer.add_scalar(f"Molecules/{mol}", d / n, step_i)


def _log_imgs(step_i: int, writer: SummaryWriter, world: ms.World) -> None:
    writer.add_image("Maps/Cells", world.culture.cell_map >= 0, step_i, dataformats="WH")


def _kill_cells(world: ms.World, i_atp: int, thr: float) -> None:
    mask = world.cells.molecules[:, i_atp] < thr
    world.kill_cells(mask=mask)


def _divide_cells(world: ms.World, i_aca: int, i_hca: int, thr: float, cost=2.0) -> None:
    can_divide = world.cells.molecules[:, i_aca] > thr
    world.cells.molecules[can_divide, i_aca] -= cost
    world.cells.molecules[can_divide, i_hca] += cost
    is_desc = world.divide_cells(mask=can_divide)
    world.mutate_cells(mask=is_desc)


def main(kwargs: dict) -> None:
    writer = SummaryWriter(log_dir=_this_dir / "runs" / _now)

    world = ms.World(
        chemistry=CHEMISTRY,
        culture_kwargs={"map_size": kwargs["map_size"]},
        tensor_kwargs={"device": kwargs["device"]},
    )
    world.culture.molecule_map[:] = 10.0
    world.save(rundir=_this_dir / "runs" / _now)

    i_atp = world.chemistry.molname_2_idx["ATP"]
    i_aca = world.chemistry.molname_2_idx["acetyl-CoA"]
    i_hca = world.chemistry.molname_2_idx["HS-CoA"]

    genomes = [ms.random_genome(kwargs["init_genome_size"]) for _ in range(kwargs["init_n_cells"])]
    world.spawn_cells(genomes=genomes)

    for step_i in range(kwargs["n_steps"]):
        _kill_cells(world=world, i_atp=i_atp, thr=kwargs["thr_kill"])
        _divide_cells(world=world, i_aca=i_aca, i_hca=i_hca, thr=kwargs["thr_replicate"])

        if step_i % kwargs["check_every_n"] == 0:
            if kwargs["save_state"]:
                world.save_state(statedir=_this_dir / "runs" / _now / f"step={step_i}")
            _log_imgs(writer=writer, world=world, step_i=step_i)
            _log_scalars(writer=writer, world=world, step_i=step_i)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--map-size", default=128, type=int)
    parser.add_argument("--n-steps", default=10_000, type=int)
    parser.add_argument("--init-n-cells", default=1000, type=int)
    parser.add_argument("--init-genome-size", default=500, type=int)
    parser.add_argument("--thr-kill", default=1.0, type=float)
    parser.add_argument("--thr-replicate", default=15.0, type=float)
    parser.add_argument("--check-every-n", default=25, type=int)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--device", default="cpu", type=str)
    main(vars(parser.parse_args()))
