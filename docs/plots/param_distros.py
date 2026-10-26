from pathlib import Path

import magicsoup as ms
import pandas as pd
from magicsoup.examples.wood_ljungdahl import CHEMISTRY
from plotnine import *  # pylint: disable=W0614,W0401


def _plot_distributions():
    world = ms.World(chemistry=CHEMISTRY)
    genomes = [ms.random_genome() for _ in range(1000)]
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
    stats = df.describe()
    med = stats.loc[["50%"]]
    std = stats.loc[["25%", "75%"]]

    return (
        ggplot(df.melt())
        + geom_hline(aes(yintercept="value"), med.melt(), linetype="dashed", alpha=0.5)
        + geom_hline(aes(yintercept="value"), std.melt(), linetype="dashed", alpha=0.2)
        + geom_violin(aes(y="value", color="variable", fill="variable"))
        + facet_wrap("variable", ncol=1, scales="free")
        + scale_y_continuous(trans="log10")
        + coord_flip()
        + theme(axis_title=element_blank(), legend_position="none")
        + theme(figure_size=(7, 4))
    ), "parameter-distributions"


def create_plots(imgs_dir: Path):
    # g, name = _plot_specific_ke_distributions()
    # g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_distributions()
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=True)


# TODO: in previous plots I used µM, actually it's mM
