from pathlib import Path

import magicsoup as ms
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from magicsoup.genomics import Genomics, ProteinSpec
from mizani.formatters import percent_format
from plotnine import *  # pylint: disable=W0614,W0401

PE_WHITE_OUTLINE = [
    path_effects.Stroke(linewidth=3, foreground="white"),
    path_effects.Normal(),
]


def _record_proteome_stats(
    proteomes: list[list[ProteinSpec]],
    records: list[dict],
    size: int,
    record: dict,
    dom_size: int,
):
    for proteome in proteomes:
        n_prots = len(proteome)
        records.append({**record, "n": n_prots, "v": "proteins/genome"})
        if n_prots > 0:
            n_doms = sum(len(d[0]) for d in proteome)
            records.append({**record, "n": n_doms / n_prots, "v": "domains/protein"})
            records.append(
                {**record, "n": n_doms * dom_size / size, "v": "coding bps/bp"}
            )


def _plot(df: pd.DataFrame, by: str, facet: str) -> ggplot:
    df["v"] = pd.Categorical(
        df["v"],
        categories=["proteins/genome", "domains/protein", "coding bps/bp"],
        ordered=True,
    )
    avgs = df.groupby([by, "v"])["n"].median().reset_index()
    avgs["l"] = [f"{d:.1f}" for d in avgs["n"]]
    return (
        ggplot(df)
        + geom_vline(aes(xintercept="n"), data=avgs, linetype="dashed", alpha=0.5)
        + geom_histogram(aes(x="n", y=after_stat("width*density")), bins=10)
        + geom_text(
            aes(x=np.inf, y=np.inf, label="l"),
            ha="right",
            va="top",
            data=avgs,
            size=8,
            path_effects=PE_WHITE_OUTLINE,
        )
        + scale_y_continuous(labels=percent_format())
        + facet_grid(facet, scales="free")
        + theme(
            figure_size=(8, 6),
            axis_title_y=element_blank(),
            axis_title_x=element_blank(),
        )
    )


def _plot_genome_sizes(ncells: int):
    records: list[dict] = []
    for size in (500, 1000, 2000):
        genomics = Genomics()
        genomes = [ms.random_genome(s=size) for _ in range(ncells)]
        proteomes = genomics.translate_genomes(genomes=genomes)
        _record_proteome_stats(
            proteomes=proteomes,
            records=records,
            size=size,
            dom_size=genomics.dom_size_nt,
            record={"size": size},
        )

    df = pd.DataFrame.from_records(records)
    plot = _plot(df=df, by="size", facet="size ~ v")
    return plot, "different_genome_sizes"


def _plot_domain_probabilities(ncells: int, size: int):
    records: list[dict] = []
    for prob in (0.02, 0.05, 0.1):
        genetics = Genomics(p_reg_dom=prob, p_catal_dom=prob, p_transp_dom=prob)
        genomes = [ms.random_genome(s=size) for _ in range(ncells)]
        proteomes = genetics.translate_genomes(genomes=genomes)
        _record_proteome_stats(
            proteomes=proteomes,
            records=records,
            size=size,
            dom_size=genetics.dom_size_nt,
            record={"prob": prob},
        )

    df = pd.DataFrame.from_records(records)
    plot = _plot(df=df, by="prob", facet="prob ~ v")
    return plot, "different_domain_probabilities"


def _plot_start_stop_codons(ncells: int, size: int):
    kwargs_map = {
        "3-3": {},
        "2-3": {"start_codons": ("ATG", "GTG")},
        "1-3": {"start_codons": ("ATG",)},
        "3-2": {"stop_codons": ("TAG", "TAA")},
        "3-1": {"stop_codons": ("TAG",)},
    }

    records: list[dict] = []
    for label, kwargs in kwargs_map.items():
        genetics = Genomics(**kwargs)  # type: ignore
        genomes = [ms.random_genome(s=size) for _ in range(ncells)]
        proteomes = genetics.translate_genomes(genomes=genomes)
        _record_proteome_stats(
            proteomes=proteomes,
            records=records,
            size=size,
            dom_size=genetics.dom_size_nt,
            record={"start-stop": label},
        )

    df = pd.DataFrame.from_records(records)
    plot = _plot(df=df, by="start-stop", facet="start-stop ~ v")
    return plot, "different_start_stop_codons"


def create_plots(imgs_dir: Path):
    ncells = 1000
    size = 1000
    g, name = _plot_genome_sizes(ncells=ncells)
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_domain_probabilities(ncells=ncells, size=size)
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_start_stop_codons(ncells=ncells, size=size)
    g.save(imgs_dir / f"{name}.png", dpi=200)
