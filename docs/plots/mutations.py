# type: ignore
from pathlib import Path
import pandas as pd
from Levenshtein import distance
from plotnine import *
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import CHEMISTRY


def _record_sequence_similarities(
    seqs0: list[str], seqs1: list[str], record: dict, records: list
):
    for i, (s0, s1) in enumerate((zip(seqs0, seqs1))):
        minlen, maxlen = sorted([len(s0), len(s1)])
        d = 1.0 - distance(s0, s1) / maxlen if minlen > 0 else 0.0
        records.append({**record, "value": d, "i": i})


def _plot_point_mutations(
    nsteps: int, probs: list[float], confluency: float, gsize=1000, map_size=32
):
    records = []
    for prob in probs:
        n_cells = int(map_size**2 * confluency)
        original_genomes = [ms.random_genome(s=gsize) for _ in range(n_cells)]
        genomes = [d for d in original_genomes]
        for step in range(nsteps):
            record = {"step": step, "p": prob}
            if step % 100 == 0:
                _record_sequence_similarities(
                    seqs0=genomes,
                    seqs1=original_genomes,
                    record=record,
                    records=records,
                )
            for seq, idx in ms.point_mutations(seqs=genomes, p=prob):
                genomes[idx] = seq
        _record_sequence_similarities(
            seqs0=genomes, seqs1=original_genomes, record=record, records=records
        )

    df = pd.DataFrame.from_records(records)
    df["p"] = pd.Categorical(df["p"], categories=probs)
    df["step"] = pd.Categorical(df["step"], categories=df["step"].unique())
    df["measure"] = "similarity"
    aggr = (
        df.groupby(["p", "step"])
        .apply(lambda d: (d["value"] < 1.0).sum() / len(d) * 100)
        .reset_index(name="value")
    )
    aggr["measure"] = "mutated[%]"
    df = pd.concat([df, aggr], ignore_index=True)

    return (
        ggplot(df)
        + geom_boxplot(
            aes(x="step", y="value", color="p"), data=df[df["measure"] == "similarity"]
        )
        + geom_col(
            aes(x="step", y="value", fill="p"),
            width=0.5,
            position="dodge",
            data=df[df["measure"] == "mutated[%]"],
        )
        + facet_grid("measure ~ .", scales="free")
        + scale_fill_brewer(type="Qualitative", palette="Set2")
        + scale_color_brewer(type="Qualitative", palette="Set2")
        + theme(figure_size=(6, 4))
    ), "point_mutations"


def _plot_genome_recombinations(
    confluency: float, probs: list[float], nsteps: int, gsize=1000, map_size=32
):
    records = []
    for prob in probs:
        n_cells = int(map_size**2 * confluency / 100)
        world = ms.World(map_size=map_size, chemistry=CHEMISTRY)
        world.spawn_cells(genomes=[ms.random_genome(s=gsize) for _ in range(n_cells)])
        nghbrs = world.get_neighbors(cell_idxs=list(range(world.n_cells)))
        original_genomes = [d for d in world.cell_genomes]
        genomes = [d for d in original_genomes]
        for step in range(nsteps):
            record = {"step": step, "p": prob}
            if step % 100 == 0:
                _record_sequence_similarities(
                    seqs0=genomes,
                    seqs1=original_genomes,
                    record=record,
                    records=records,
                )
            genome_pairs = [(genomes[a], genomes[b]) for a, b in nghbrs]
            for sa, sb, idx in ms.recombinations(seq_pairs=genome_pairs, p=prob):
                a, b = nghbrs[idx]
                genomes[a] = sa
                genomes[b] = sb
        _record_sequence_similarities(
            seqs0=genomes, seqs1=original_genomes, record=record, records=records
        )

    df = pd.DataFrame.from_records(records)
    df["p"] = pd.Categorical(df["p"], categories=probs)
    df["step"] = pd.Categorical(df["step"], categories=df["step"].unique())
    df["measure"] = "similarity"
    aggr = (
        df.groupby(["p", "step"])
        .apply(lambda d: (d["value"] < 1.0).sum() / len(d) * 100)
        .reset_index(name="value")
    )
    aggr["measure"] = "mutated[%]"
    df = pd.concat([df, aggr], ignore_index=True)

    return (
        ggplot(df)
        + geom_boxplot(
            aes(x="step", y="value", color="p"), data=df[df["measure"] == "similarity"]
        )
        + geom_col(
            aes(x="step", y="value", fill="p"),
            width=0.5,
            position="dodge",
            data=df[df["measure"] == "mutated[%]"],
        )
        + facet_grid("measure ~ .", scales="free")
        + scale_fill_brewer(type="Qualitative", palette="Set2")
        + scale_color_brewer(type="Qualitative", palette="Set2")
        + theme(figure_size=(6, 4))
    ), "genome_recombinations"


def create_plots(imgs_dir: Path):
    g, name = _plot_point_mutations(
        nsteps=1000, confluency=0.5, probs=[1e-6, 1e-5, 1e-4]
    )
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g.save(imgs_dir / f"{name}.png", dpi=200)
    for conf in [30, 50]:
        g, name = _plot_genome_recombinations(
            confluency=conf, nsteps=1000, probs=[1e-8, 1e-7, 1e-6]
        )
        g.save(imgs_dir / f"{name}{conf}.png", dpi=200)
