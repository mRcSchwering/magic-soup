from pathlib import Path

import magicsoup as ms
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from magicsoup.examples.wood_ljungdahl import CHEMISTRY
from mizani.formatters import percent_format
from plotnine import *  # pylint: disable=W0614,W0401

PE_WHITE_OUTLINE = [
    path_effects.Stroke(linewidth=3, foreground="white"),
    path_effects.Normal(),
]


def _plot_distributions_continuous():
    world = ms.World(chemistry=CHEMISTRY)
    N = world.culture.map_size
    genomes = [ms.random_genome() for _ in range(N)]
    idxs = world.spawn_cells(genomes=genomes)
    is_prot = (world.cells.N_e[idxs] != 0).any(dim=2)
    k_e = world.cells.k_b[idxs] / world.cells.k_f[idxs]

    df = pd.DataFrame(
        {
            "Ke": k_e[is_prot].flatten(),
            "Vmax [mM/s]": world.cells.v_max[idxs][is_prot].flatten(),
            "Km [mM]": world.cells.k_f[idxs][is_prot].flatten(),
        }
    )
    stats = df.describe(percentiles=[0.1, 0.5, 0.9])
    med = stats.loc[["50%"]].melt()
    std = stats.loc[["10%", "90%"]].melt()

    return (
        ggplot(df.melt())
        + geom_histogram(aes(x="value", y=after_stat("width*density")), bins=10)
        + geom_vline(aes(xintercept="value"), med, linetype="dashed", alpha=0.5)
        + geom_vline(aes(xintercept="value"), std, linetype="dashed", alpha=0.2)
        + geom_text(
            aes(x=np.inf, y=np.inf, label="value"),
            ha="right",
            va="top",
            data=med,
            size=8,
            format_string="{:.2e}",
            path_effects=PE_WHITE_OUTLINE,
        )
        + facet_wrap("variable", ncol=1, scales="free")
        + scale_y_continuous(labels=percent_format())
        + scale_x_continuous(trans="log10")
        + theme(axis_title=element_blank(), legend_position="none")
        + theme(figure_size=(7, 6))
    ), "continuous-parameter-distributions"


def _plot_distributions_discrete():
    world = ms.World(chemistry=CHEMISTRY)
    N = world.culture.map_size
    genomes = [ms.random_genome() for _ in range(N**2)]
    idxs = world.spawn_cells(genomes=genomes)
    is_prot = (world.cells.N_e[idxs] != 0).any(dim=2)
    N_e = world.cells.N_b[idxs][is_prot] - world.cells.N_f[idxs][is_prot]
    N_e = N_e[N_e != 0].flatten()
    H = world.cells.N_h[idxs][is_prot]
    H = H[H != 0].flatten()

    df = pd.DataFrame(
        {
            "value": N_e.tolist() + H.tolist(),
            "variable": ["Stoichiometric number"] * N_e.numel()
            + ["Hill coefficient"] * H.numel(),
        }
    )
    df["variable"] = pd.Categorical(
        df["variable"], categories=df["variable"].unique(), ordered=True
    )

    return (
        ggplot(df)
        + geom_bar(aes(x="value", y=after_stat("count/sum(count)")))
        + facet_wrap("variable", ncol=1, scales="free")
        + scale_y_continuous(labels=percent_format())
        + theme(axis_title=element_blank(), legend_position="none")
        + theme(figure_size=(6, 3))
    ), "discrete-parameter-distributions"


def create_plots(imgs_dir: Path):
    g, name = _plot_distributions_continuous()
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=True)
    g, name = _plot_distributions_discrete()
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=True)
