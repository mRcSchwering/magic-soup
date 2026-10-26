from itertools import product
from pathlib import Path

import magicsoup as ms
import pandas as pd
from plotnine import *  # pylint: disable=W0614,W0401


def _plot_diff(chemistry: ms.Chemistry):
    N = 32
    dt = 0.1
    world = ms.World(time_step=dt, chemistry=chemistry, culture_kwargs={"map_size": N})
    h = N // 2

    world.culture.molecule_map[:] = 1
    world.culture.molecule_map[:, h, h] = 100
    world.culture.molecule_map[:, h + 1, h] = 100
    world.culture.molecule_map[:, h, h + 1] = 100
    world.culture.molecule_map[:, h + 1, h + 1] = 100

    n_seconds = 10
    mols = ["CO2", "glucose", "NADH"]

    records = []
    for ti in range(n_seconds):
        for mol in mols:
            mi = chemistry.molname_2_idx[mol]
            record = {"second": round(ti * dt, 1), "molecule": mol}
            for x, y in product(range(N), range(N)):
                records.append(
                    {
                        **record,
                        "x": x * 10,
                        "y": y * 10,
                        "µM": world.culture.molecule_map[mi, x, y].item(),
                    }
                )

        world.advance_time()

    df = pd.DataFrame.from_records(records)
    df["molecule"] = pd.Categorical(
        df["molecule"], categories=[d.name for d in chemistry.molecules], ordered=True
    )

    return (
        ggplot(df)
        + geom_raster(aes(x="x", y="y", fill="µM"))
        + scale_fill_continuous(trans="log10")  # pylint: disable=all
        + facet_grid("molecule ~ second")
        + coord_equal(expand=False)
        + theme(figure_size=(14, 6))
        + theme(axis_title=element_blank())
        + theme(legend_position="bottom", legend_direction="horizontal")
    ), "diffusion"


def _plot_perm_degrad(chemistry: ms.Chemistry):
    # permeability
    n_seconds = 10
    dt = 0.01
    mols = ["CO2", "acetat", "glucose"]

    world = ms.World(time_step=dt, time_resolution=dt, chemistry=chemistry)
    n_cells = 1000
    idxs = world.spawn_cells(genomes=["" for _ in range(n_cells)])
    world.culture.molecule_map[:] = 100.0
    world.cells.molecules[:] = 1.0

    records = []
    for ti in range(n_seconds * int(1 / dt)):
        for mol in mols:
            mi = chemistry.molname_2_idx[mol]
            conc = float(world.cells.molecules[idxs, mi].sum()) / n_cells
            records.append({"second": ti * dt, "molecule": mol, "µM": conc})
        world.advance_time()

    dfp = pd.DataFrame.from_records(records)
    dfp["type"] = "permeability"

    # half-life
    dt = 1000
    n_seconds = 10_000_000
    mols = ["CO2", "NADH", "acetyl-phosphate"]

    world = ms.World(time_step=dt, time_resolution=dt, chemistry=chemistry)
    N = world.culture.map_size
    world.culture.molecule_map[:] = 100.0

    records = []
    for ti in range(n_seconds // dt):
        for mol in mols:
            mi = chemistry.molname_2_idx[mol]
            conc = float(world.culture.molecule_map[mi].sum()) / N**2
            records.append({"second": ti * dt, "molecule": mol, "µM": conc})
        world.advance_time()

    dfh = pd.DataFrame.from_records(records)
    dfh["type"] = "half-life"

    return (
        ggplot(pd.concat([dfp, dfh], ignore_index=True))
        + geom_line(aes(x="second", y="µM", color="molecule"))
        + scale_color_discrete()
        + scale_x_continuous(trans="log10")  # pylint: disable=all
        + facet_wrap("type", ncol=1, scales="free")
        + theme(figure_size=(12, 5))
        # + theme(axis_title=element_blank())
        # + theme(legend_position="bottom", legend_direction="horizontal")
    ), "permeation-degradation"


def create_plots(imgs_dir: Path):
    # half-lfie (days) diffusion coef (cm²/s), permeability coef (cm/s),
    molecules = [
        ms.Molecule("CO2", 1, diffusivity=1.9e-5, permeability=3.5e-1),
        ms.Molecule("glucose", 1, diffusivity=6.7e-6, permeability=1e-8),
        ms.Molecule("NADH", 1, half_life=100, diffusivity=2.5e-6, permeability=-1),
        ms.Molecule("acetat", 1, permeability=2.0e-4),
        ms.Molecule("acetyl-phosphate", 1, half_life=0.5),
    ]

    chemistry = ms.Chemistry(molecules=molecules, reactions=[])
    # g, name = _plot_diff(chemistry=chemistry)
    # g.save(imgs_dir / f"{name}.png", dpi=200, transparent=True)
    g, name = _plot_perm_degrad(chemistry=chemistry)
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=True)
    # g, name = _plot_molecule_gradients(chemistry=chemistry)
    # g.save(imgs_dir / f"{name}.png", dpi=200)
