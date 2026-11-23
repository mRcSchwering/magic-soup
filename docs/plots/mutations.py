from pathlib import Path

import magicsoup as ms
import pandas as pd
from plotnine import *  # pylint: disable=W0614,W0401

from docs.plots.util import ani_from_jaccard, jaccard


def _get_stats(seqs0: list[str], seqs1: list[str]) -> tuple[int, float, float]:
    n_mutated = 0
    jacc_sum = 0.0
    ani_sum = 0.0
    for seq0, seq1 in zip(seqs0, seqs1):
        if seq0 != seq1:
            n_mutated += 1
            jacc = jaccard(seq0, seq1)
            ani = ani_from_jaccard(jacc)
            jacc_sum += jacc
            ani_sum += ani

    return n_mutated, jacc_sum / n_mutated, ani_sum / n_mutated


def _plot_default_mutations(n_gens: int = 1000) -> tuple[ggplot, str]:
    n_cells = 10_000
    nth = n_gens // 100
    mutations = [
        {"sub": 1, "indel": 1},
        {"sub": 10, "indel": 1},
        {"sub": 1, "indel": 10},
    ]

    records = []
    for mut in mutations:
        mutator = ms.Mutator()
        mutator = ms.Mutator(
            sub_rate=mutator.sub_rate * mut["sub"],
            indel_rate=mutator.indel_rate * mut["indel"],
        )
        orig_genomes = ms.StrArray.from_strings([ms.random_genome() for _ in range(n_cells)])
        genomes = ms.StrArray.from_strings([d for d in orig_genomes])
        label = ",".join(f"{v} x {k}" for k, v in mut.items() if v > 1)
        label = label or "default"

        for gen_i in range(n_gens + 1):
            idxs, mutated = mutator.mutate(genomes=genomes)
            genomes[idxs] = mutated

            if gen_i % nth == 0:
                n_mut, jacc, ani = _get_stats(genomes.tolist(), orig_genomes.tolist())
                records.append(
                    {
                        "generation": gen_i + 1,
                        "mutated": n_mut / n_cells,
                        "Jaccard": jacc,
                        "ANI": ani,
                        "mutations": label,
                    }
                )

    df = pd.DataFrame.from_records(records)
    df = df.melt(id_vars=["generation", "mutations"])
    df["mutations"] = pd.Categorical(
        df["mutations"], categories=df["mutations"].unique(), ordered=True
    )
    df["variable"] = pd.Categorical(
        df["variable"], categories=["mutated", "Jaccard", "ANI"], ordered=True
    )

    return (
        ggplot(df)
        + geom_line(aes(x="generation", y="value", color="mutations"))
        + scale_color_discrete()
        + facet_grid("variable ~ .", scales="free")
        + theme(figure_size=(10, 4), axis_title_y=element_blank())
    ), "mutations"


def create_plots(imgs_dir: Path, transparent: bool):
    g, name = _plot_default_mutations()
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=transparent)
