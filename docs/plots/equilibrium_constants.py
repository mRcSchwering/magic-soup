# type: ignore
from itertools import combinations
from pathlib import Path
import pandas as pd
import torch
from plotnine import *
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import CHEMISTRY
from magicsoup.constants import GAS_CONSTANT


def _plot_specific_ke_distributions():
    world = ms.World(chemistry=CHEMISTRY)

    records = []
    for subs, prods in CHEMISTRY.reactions:
        name = (
            " + ".join(str(d) for d in subs)
            + " <-> "
            + " + ".join(str(d) for d in prods)
        )
        energy = sum(d.energy for d in prods) - sum(d.energy for d in subs)
        lke = -energy / GAS_CONSTANT / world.abs_temp / 2.303
        records.append({"name": name, "lKe": lke})
        records.append({"name": name, "lKe": -lke})

    genomes = [ms.random_genome(s=500) for _ in range(100)]
    world.spawn_cells(genomes=genomes)
    lKe = torch.log(world.kinetics.Kmb / world.kinetics.Kmf)
    for lke in lKe[lKe != 0.0].flatten().tolist():
        records.append({"name": "random proteins", "lKe": lke})

    df = pd.DataFrame.from_records(records)

    return (
        ggplot(df, aes(x="name", y="lKe"))
        + geom_hline(yintercept=0.0, linetype="dashed", alpha=0.5)
        + geom_hline(yintercept=[-4.6, 4.6], linetype="dashed", alpha=0.3)
        + geom_jitter(data=df, alpha=0.5, width=0.1, height=0.1)
        + coord_flip()
        + theme(figure_size=(7, 2))
    ), "specific_equilibrium_constant_distributions"


def _plot_random_ke_distributions(
    abs_tmps: list[float], energies: list[float], n_mols=10
):
    energies = [10, 100, 200]

    records = []
    for energy in energies:
        for abs_temp in abs_tmps:
            ms.Molecule._instances = {}  # rm previous instances
            molecules = [ms.Molecule(str(i), energy * 1e3) for i in range(n_mols)]
            reactions = [([a, b], [c]) for a, b, c in combinations(molecules, 3)]
            chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
            world = ms.World(chemistry=chemistry, abs_temp=abs_temp)
            genomes = [ms.random_genome(s=500) for _ in range(1000)]
            world.spawn_cells(genomes=genomes)
            lKe = torch.log(world.kinetics.Kmb / world.kinetics.Kmf).abs()
            label = f"{energy}kJ/mol\n{abs_temp}K"
            for lke in lKe[lKe != 0.0].flatten().tolist():
                records.append({"label": label, "|lKe|": lke, "T": abs_temp})

    df = pd.DataFrame.from_records(records)
    df["label"] = pd.Categorical(df["label"], categories=reversed(df["label"].unique()))

    return (
        ggplot(df)
        + geom_hline(yintercept=0, linetype="dashed", alpha=0.5)
        + geom_hline(yintercept=4.6, linetype="dashed", alpha=0.3)
        + geom_violin(aes(x="label", y="|lKe|", color="T", fill="T"), position="dodge")
        + coord_flip()
        + theme(figure_size=(8, 4))
    ), "random_equilibrium_constant_distributions"


def create_plots(imgs_dir: Path):
    g, name = _plot_specific_ke_distributions()
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_random_ke_distributions(
        abs_tmps=[250, 310, 370], energies=[10, 100, 200]
    )
    g.save(imgs_dir / f"{name}.png", dpi=200)
