from pathlib import Path

import magicsoup as ms
import pandas as pd
import torch
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS
from plotnine import *  # pylint: disable=W0614,W0401

# TODO: add physical units to plots in general


def _run_simulation(
    reactions: list[ms.Reaction], label: str, n_seconds=100
) -> pd.DataFrame:
    print("starting", label)
    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=reactions)
    E = torch.tensor([d.energy for d in chemistry.molecules], dtype=torch.double)
    world = ms.World(chemistry=chemistry, culture_kwargs={"map_size": 10})
    world.culture.molecule_map[:] = 10.0
    N = world.culture.map_size

    genomes = [ms.random_genome() for _ in range(N**2)]
    world.spawn_cells(genomes=genomes)

    def _get_energy_dens(world: ms.World) -> float:
        e_map = torch.einsum("mxy,m->xy", world.culture.molecule_map.to(E), E)
        e_cells = torch.einsum("cm,m->c", world.cells.molecules.to(E), E)
        return int(e_map.sum() + e_cells.sum() / 1000) / N**2

    records = []
    for t in range(n_seconds):
        print(f"  {t}s")
        records.append({"t": t, "e": _get_energy_dens(world)})
        world.advance_time()

    df = pd.DataFrame.from_records(records)
    df["e"] = df["e"] * 1e-3  # mmol/L * J/mol -> J/L
    df["de"] = df.loc[0, "e"] - df["e"]
    df["l"] = label
    return df


def _plot_energy():
    df0 = _run_simulation(reactions=[], label="without reactions")
    df1 = _run_simulation(reactions=REACTIONS, label="with reactions")
    df = pd.concat([df0, df1], ignore_index=True)
    return (
        ggplot(df, aes(x="t", y="e", color="l"))
        + geom_line()
        + xlab("time [s]")
        + ylab("energy density [J/L]")
        + theme(figure_size=(7, 3))
        + theme(legend_position="bottom", legend_title=element_blank())
    ), "energy-density"


def create_plots(imgs_dir: Path):
    g, name = _plot_energy()
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=True)
