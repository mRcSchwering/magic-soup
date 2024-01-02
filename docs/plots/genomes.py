# type: ignore
from pathlib import Path
import pandas as pd
from plotnine import *
import magicsoup as ms


_IMGS_DIR = Path(__file__).parent.parent / "docs" / "img" / "supporting"
if not _IMGS_DIR.is_dir():
    raise ValueError(f"{_IMGS_DIR} is not a directory")


def _record_proteome_stats(
    proteomes: list[list[ms.ProteinSpecType]],
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


def _get_proteome_stat_avgs(
    df: pd.DataFrame, by: str, pad_coding=0.5, pad_doms=0.5, pad_prots=10.0
) -> pd.DataFrame:
    df["v"] = pd.Categorical(
        df["v"],
        categories=["proteins/genome", "domains/protein", "coding bps/bp"],
        ordered=True,
    )
    avgs = df.groupby([by, "v"])["n"].median().reset_index()
    avgs["l"] = [f"{d:.1f}" for d in avgs["n"]]
    avgs.loc[:, "x"] = avgs["n"]
    avgs.loc[avgs["v"] == "coding bps/bp", "x"] += pad_coding
    avgs.loc[avgs["v"] == "domains/protein", "x"] += pad_doms
    avgs.loc[avgs["v"] == "proteins/genome", "x"] += pad_prots
    return avgs


def _plot_genome_sizes(ncells: int):
    records = []
    for size in (200, 500, 1000, 2000):
        genetics = ms.Genetics()
        genomes = [ms.random_genome(s=size) for _ in range(ncells)]
        proteomes = genetics.translate_genomes(genomes=genomes)
        _record_proteome_stats(
            proteomes=proteomes,
            records=records,
            size=size,
            dom_size=genetics.dom_size,
            record={"size": size},
        )
    df = pd.DataFrame.from_records(records)

    avgs = _get_proteome_stat_avgs(df=df, by="size", pad_prots=15)
    avgs.loc[avgs["size"] == 200, "y"] = 400
    avgs.loc[avgs["size"] == 500, "y"] = 300
    avgs.loc[avgs["size"] == 1000, "y"] = 250
    avgs.loc[avgs["size"] == 2000, "y"] = 300

    return (
        ggplot(df)
        + geom_vline(aes(xintercept="n"), data=avgs, linetype="dashed", alpha=0.5)
        + geom_histogram(aes(x="n"), bins=15)
        + geom_text(aes(x="x", y="y", label="l"), data=avgs, size=10)
        + facet_grid("size ~ v", scales="free")
        + theme(figure_size=(8, 6))
    ), "different_genome_sizes"


def _plot_domain_probabilities(ncells: int, size: int):
    records = []
    for prob in (0.001, 0.01, 0.1):
        genetics = ms.Genetics(p_reg_dom=prob, p_catal_dom=prob, p_transp_dom=prob)
        genomes = [ms.random_genome(s=size) for _ in range(ncells)]
        proteomes = genetics.translate_genomes(genomes=genomes)
        _record_proteome_stats(
            proteomes=proteomes,
            records=records,
            size=size,
            dom_size=genetics.dom_size,
            record={"prob": prob},
        )
    df = pd.DataFrame.from_records(records)

    avgs = _get_proteome_stat_avgs(
        df=df, by="prob", pad_prots=15, pad_doms=0.7, pad_coding=1.0
    )
    avgs.loc[avgs["prob"] == 0.001, "y"] = 400
    avgs.loc[avgs["prob"] == 0.01, "y"] = 300
    avgs.loc[avgs["prob"] == 0.1, "y"] = 200

    return (
        ggplot(df)
        + geom_vline(aes(xintercept="n"), data=avgs, linetype="dashed", alpha=0.5)
        + geom_histogram(aes(x="n"), bins=15)
        + geom_text(aes(x="x", y="y", label="l"), data=avgs, size=10)
        + facet_grid("prob ~ v", scales="free")
        + theme(figure_size=(8, 4))
    ), "different_domain_probabilities"


def _plot_start_stop_codons(ncells: int, size: int):
    kwargs_map = {
        "3-3": {},
        "2-3": {"start_codons": ("ATG", "GTG")},
        "1-3": {"start_codons": ("ATG",)},
        "3-2": {"stop_codons": ("TAG", "TAA")},
        "3-1": {"stop_codons": ("TAG",)},
    }

    records = []
    for label, kwargs in kwargs_map.items():
        genetics = ms.Genetics(**kwargs)
        genomes = [ms.random_genome(s=size) for _ in range(ncells)]
        proteomes = genetics.translate_genomes(genomes=genomes)
        _record_proteome_stats(
            proteomes=proteomes,
            records=records,
            size=size,
            dom_size=genetics.dom_size,
            record={"start-stop": label},
        )
    df = pd.DataFrame.from_records(records)

    avgs = _get_proteome_stat_avgs(
        df=df, by="start-stop", pad_prots=14, pad_doms=0.7, pad_coding=0.8
    )
    avgs.loc[avgs["start-stop"] == "1-3", "y"] = 400
    avgs.loc[avgs["start-stop"] == "2-3", "y"] = 400
    avgs.loc[avgs["start-stop"] == "3-1", "y"] = 200
    avgs.loc[avgs["start-stop"] == "3-2", "y"] = 200
    avgs.loc[avgs["start-stop"] == "3-3", "y"] = 400

    return (
        ggplot(df)
        + geom_vline(aes(xintercept="n"), data=avgs, linetype="dashed", alpha=0.5)
        + geom_histogram(aes(x="n"), bins=15)
        + geom_text(aes(x="x", y="y", label="l"), data=avgs, size=10)
        + facet_grid("start-stop ~ v", scales="free")
        + theme(figure_size=(8, 6))
    ), "different_start_stop_codons"


def create_plots(imgs_dir: Path):
    ncells = 1000
    size = 1000
    g, name = _plot_genome_sizes(ncells=ncells)
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_domain_probabilities(ncells=ncells, size=size)
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_start_stop_codons(ncells=ncells, size=size)
    g.save(imgs_dir / f"{name}.png", dpi=200)
