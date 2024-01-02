# type: ignore
from pathlib import Path
import pandas as pd
import torch
from plotnine import *
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import CHEMISTRY


def _plot_energy(nsteps=1000, map_size=8, confluency=0.5, gsize=1000):
    world = ms.World(map_size=map_size, chemistry=CHEMISTRY)
    molmap = torch.rand_like(world.molecule_map) * 100
    max_ = molmap.max().item()
    n_cells = int(map_size**2 * confluency)
    genomes = [ms.random_genome(s=gsize) for _ in range(n_cells)]

    def init_world() -> ms.World:
        world.kill_cells(cell_idxs=range(world.n_cells))
        world.molecule_map = molmap.clone()
        world.spawn_cells(genomes=genomes)
        return world

    def world_energy(world: ms.World) -> float:
        energies = torch.tensor([d.energy for d in world.chemistry.molecules])
        e_map = torch.einsum("mxy,m->xy", world.molecule_map, energies)
        e_cells = torch.einsum("cm,m->c", world.cell_molecules, energies)
        return (e_map.sum() + e_cells.sum()).item()

    def world_entropy(world: ms.World) -> float:
        t = (
            torch.cat([world.molecule_map.flatten(), world.cell_molecules.flatten()])
            / max_
        )
        return (-t * torch.log(t + 1e-10)).sum().item()

    def record_energy_entropy(record: dict, records: list, world: ms.World):
        records.append({**record, "value": world_energy(world), "variable": "U"})
        records.append({**record, "value": world_entropy(world), "variable": "S"})

    records = []

    world = init_world()
    for step_i in range(nsteps):
        world.diffuse_molecules()
        record_energy_entropy(
            record={"label": "diffusion", "step": step_i}, records=records, world=world
        )

    world = init_world()
    for step_i in range(nsteps):
        world.enzymatic_activity()
        record_energy_entropy(
            record={"label": "enzymatic-activity", "step": step_i},
            records=records,
            world=world,
        )

    world = init_world()
    for step_i in range(nsteps):
        world.enzymatic_activity()
        world.diffuse_molecules()
        record_energy_entropy(
            record={"label": "both", "step": step_i}, records=records, world=world
        )

    df = pd.DataFrame.from_records(records)
    df["label"] = pd.Categorical(
        df["label"], categories=["diffusion", "enzymatic-activity", "both"]
    )
    df["value"] = df["value"] / map_size**2
    df["variable"] = df["variable"] + "[pP]"

    return (
        ggplot(df)
        + geom_line(aes(x="step", y="value", color="label"))
        + scale_y_continuous(labels=lambda d: [f"{dd:.2E}" for dd in d])
        + facet_grid("variable ~ .", scales="free")
        + scale_color_brewer(type="Qualitative", palette="Set2")
        + theme(figure_size=(10, 4))
        + theme(
            legend_position="bottom",
            legend_title=element_blank(),
            axis_title=element_blank(),
            legend_margin=10,
        )
    ), "free_energy"


def create_plots(imgs_dir: Path):
    g, name = _plot_energy()
    g.save(imgs_dir / f"{name}.png", dpi=200)
