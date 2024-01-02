# type: ignore
import math
from pathlib import Path
import pandas as pd
import torch
from plotnine import *
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import CHEMISTRY


def _plot_constant_distributions(gsize=1000, ncells=1000):
    world = ms.World(chemistry=CHEMISTRY)
    genomes = [ms.random_genome(s=gsize) for _ in range(ncells)]
    world.spawn_cells(genomes=genomes)
    vmaxs = world.kinetics.Vmax[world.kinetics.Vmax > 0.0].flatten().tolist()
    kmfs = world.kinetics.Kmf[world.kinetics.Vmax > 0.0].flatten().tolist()
    kmbs = world.kinetics.Kmb[world.kinetics.Vmax > 0.0].flatten().tolist()
    Ka = torch.pow(world.kinetics.Kmr, 1 / world.kinetics.A)  # they are exponentiated
    kmrs = Ka[world.kinetics.A != 0.0].flatten().tolist()

    records = []
    for d in vmaxs:
        records.append({"value": d, "variable": "Vmax", "scale": "linear"})
        records.append({"value": math.log(d, 10), "variable": "Vmax", "scale": "log10"})
    for a, b in zip(kmfs, kmbs):
        d = min(a, b)
        records.append({"value": d, "variable": "Km", "scale": "linear"})
        records.append({"value": math.log(d, 10), "variable": "Km", "scale": "log10"})
    for d in kmrs:
        records.append({"value": d, "variable": "Ka", "scale": "linear"})
        records.append({"value": math.log(d, 10), "variable": "Ka", "scale": "log10"})
    world.kill_cells(cell_idxs=list(range(world.n_cells)))
    df = pd.DataFrame.from_records(records)

    avgs = df.groupby(["variable", "scale"])["value"].median().reset_index(name="avg")
    avgs["l"] = [f"{d:.2e}" for d in avgs["avg"]]
    avgs["x"] = 30
    avgs["y"] = 4000
    avgs.loc[avgs["variable"] == "Ka", "y"] = 1000

    return (
        ggplot(df)
        + geom_histogram(aes(x="value"), bins=20)
        + geom_vline(aes(xintercept="avg"), linetype="dashed", alpha=0.5, data=avgs)
        + geom_text(
            aes(x="x", y="y", label="l"), data=avgs[avgs["scale"] == "linear"], size=10
        )
        + facet_wrap(("scale", "variable"), scales="free", ncol=3)
        + theme(figure_size=(12, 5))
        + theme(subplots_adjust={"wspace": 0.3, "hspace": 0.5})
    ), "constant_distributions"


def _plot_mm_kinetics():
    def mm(t: torch.Tensor, v: float, k: float, n: int) -> torch.Tensor:
        return t**n / (k + t**n) * v

    X = torch.arange(0, 10, 0.1)

    dfs = []
    for n in [1, 3, 5]:
        Y = mm(X, 1.0, 1.0, n)
        dfs.append(
            pd.DataFrame(
                {"x": X.tolist(), "y": Y.tolist(), "value": f"n={n}", "varying": "n"}
            )
        )
    for k in [1.0, 2.5, 5.0]:
        Y = mm(X, 1.0, k, 1)
        dfs.append(
            pd.DataFrame(
                {"x": X.tolist(), "y": Y.tolist(), "value": f"Km={k}", "varying": "Km"}
            )
        )
    for v in [0.8, 1.0, 1.2]:
        Y = mm(X, v, 1.0, 1)
        dfs.append(
            pd.DataFrame(
                {
                    "x": X.tolist(),
                    "y": Y.tolist(),
                    "value": f"Vmax={v}",
                    "varying": "Vmax",
                }
            )
        )
    df = pd.concat(dfs, ignore_index=True)

    return (
        ggplot(df, aes(x="x", y="y"))
        + geom_line(aes(color="value"), data=df[df["varying"] == "n"])
        + geom_line(aes(color="value"), data=df[df["varying"] == "Km"])
        + geom_line(aes(color="value"), data=df[df["varying"] == "Vmax"])
        + facet_grid(". ~ varying", labeller="label_both")
        + theme(
            figure_size=(10, 3),
            legend_position="bottom",
            legend_title=element_blank(),
            axis_title=element_blank(),
        )
    ), "mm_kinetics"


def _plot_allosteric_modulation():
    def allo(t: torch.Tensor, k: float, n: int) -> torch.Tensor:
        return t**n / (k**n + t**n)

    ns = [1, 3, 5]
    ks = [1.0, 2.0, 4.0]
    X = torch.arange(0, 10, 0.1)
    dfs = []
    for n in ns:
        for k in ks:
            for s in ("positive", "negative"):
                Y = allo(X, k, n if s == "positive" else -n)
                dfs.append(
                    pd.DataFrame(
                        {
                            "x": X.tolist(),
                            "y": Y.tolist(),
                            "n": n,
                            "Ka": k,
                            "cooperativity": s,
                        }
                    )
                )

    df = pd.concat(dfs, ignore_index=True)
    df.loc[df["y"].isna(), "y"] = 1.0
    df["n"] = pd.Categorical(df["n"], categories=ns)
    df["Ka"] = pd.Categorical(df["Ka"], categories=ks)

    return (
        ggplot(df, aes(x="x", y="y"))
        + geom_line(aes(color="n"))
        + facet_grid("cooperativity ~ Ka", labeller="label_both")
        + theme(
            figure_size=(10, 6),
            legend_position="bottom",
            legend_title=element_blank(),
            axis_title=element_blank(),
        )
    ), "allosteric_modulation"


def create_plots(imgs_dir: Path):
    g, name = _plot_mm_kinetics()
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_allosteric_modulation()
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_constant_distributions()
    g.save(imgs_dir / f"{name}.png", dpi=200)
