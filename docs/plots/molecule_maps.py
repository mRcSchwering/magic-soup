# type: ignore
from itertools import product
from pathlib import Path
import pandas as pd
from plotnine import *
import magicsoup as ms


def _plot_diff_degrad(chemistry: ms.Chemistry):
    world = ms.World(chemistry=chemistry, map_size=16, mol_map_init="zeros")
    world.molecule_map[:, 2:7, 2:7] = 10.0
    world.molecule_map[:, 3:6, 3:6] = 0.0
    world.molecule_map[:, 14:, 14:] = 10.0
    world.molecule_map[:, 2:7, 13] = 10.0
    world.molecule_map[:, 2, 9:13] = 10.0
    world.molecule_map[:, 7, 9] = 10.0
    world.molecule_map[:, 11, 5:11] = 10.0

    records = []
    for si in range(10):
        for mi, mol in enumerate(chemistry.molecules):
            record = {"step": si, "mol": mol.name}
            for x, y in product(range(world.map_size), range(world.map_size)):
                records.append(
                    {
                        **record,
                        "x": x,
                        "y": y,
                        "[x]": world.molecule_map[mi, x, y].item(),
                    }
                )
        world.diffuse_molecules()
        world.degrade_molecules()
    df = pd.DataFrame.from_records(records)
    df["mol"] = pd.Categorical(
        df["mol"], categories=[d.name for d in chemistry.molecules], ordered=True
    )

    return (
        ggplot(df)
        + geom_raster(aes(x="x", y="y", fill="[x]"))
        + facet_grid("mol ~ step")
        + coord_equal(expand=False)
        + theme(figure_size=(14, 6))
        + theme(axis_title=element_blank())
        + theme(
            legend_position=(0.2, -0.02),
            legend_direction="horizontal",
            legend_title=element_blank(),
        )
    ), "diffusion_degradation"


def _plot_molecule_gradients(chemistry: ms.Chemistry, mol_i=0):
    def gradient1d(world: ms.World):
        world.molecule_map[mol_i, [63, 64]] = 100.0
        world.molecule_map[mol_i, [0, -1]] = 1.0

    def gradient2d(world: ms.World):
        step = int(world.map_size / 4)
        half = int(step / 2)
        steps = [i * step for i in range(4)]
        for x in steps:
            for y in steps:
                world.molecule_map[mol_i, x + half, y + half] = 100.0
                world.molecule_map[mol_i, x, :] = 1.0
                world.molecule_map[mol_i, :, y] = 1.0

    gradient_map = {
        "1D": gradient1d,
        "2D": gradient2d,
    }

    records = []
    for label, fun in gradient_map.items():
        world = ms.World(chemistry=chemistry, mol_map_init="zeros")
        for si in range(601):
            fun(world=world)
            if si % 100 == 0:
                record = {"step": si, "grad": label}
                for x, y in product(range(128), range(128)):
                    records.append(
                        {
                            **record,
                            "x": x,
                            "y": y,
                            "[x]": world.molecule_map[mol_i, x, y].item(),
                        }
                    )
            world.diffuse_molecules()
            world.degrade_molecules()

    df = pd.DataFrame.from_records(records)

    return (
        ggplot(df)
        + geom_raster(aes(x="x", y="y", fill="[x]"))
        + facet_grid("grad ~ step", labeller="label_both")
        + coord_equal(expand=False)
        + theme(figure_size=(12, 6))
        + theme(axis_title=element_blank())
        + theme(legend_title=element_blank())
    ), "molecule_gradients"


def create_plots(imgs_dir: Path):
    ms.Molecule._instances = {}  # rm previous instances
    molecules = [
        ms.Molecule("fast-stable", 10, diffusivity=1.0, half_life=1_000),
        ms.Molecule("fast-unstable", 10, diffusivity=1.0, half_life=10),
        ms.Molecule("slow-stable", 10, diffusivity=0.01, half_life=1_000),
        ms.Molecule("slow-unstable", 10, diffusivity=0.01, half_life=10),
    ]

    chemistry = ms.Chemistry(molecules=molecules, reactions=[])
    g, name = _plot_diff_degrad(chemistry=chemistry)
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_molecule_gradients(chemistry=chemistry)
    g.save(imgs_dir / f"{name}.png", dpi=200)
