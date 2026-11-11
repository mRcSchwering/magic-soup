from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from plotnine import *  # pylint: disable=W0614,W0401


def _plot_random_splits(
    sample: Callable,
    kill: Callable,
    replicate: Callable,
    split_ratios: list[float],
    n_mols: list[int],
    ncells=1000,
    nsteps=1000,
    thresh=7000,
):
    def random_split(t: torch.Tensor, r: float) -> torch.Tensor:
        n = len(t)
        return t[torch.randint(n, size=(int(r * n),))]

    records = []
    for split_ratio in split_ratios:
        X = torch.cat([torch.ones(int(ncells / len(n_mols))) * d for d in n_mols])

        kwargs = {"step": 0, "split": split_ratio}
        for n_mol in n_mols:
            records.append({**kwargs, "[X]": n_mol, "cells": len(X[n_mol == X])})

        for step in range(nsteps):
            kwargs = {"step": step, "split": split_ratio}
            if len(X) > thresh:
                for n_mol in n_mols:
                    records.append({**kwargs, "[X]": n_mol, "cells": len(X[n_mol == X])})
                X = random_split(X, split_ratio)

            records.append({**kwargs, "[X]": -1, "cells": len(X)})

            kidxs = sample(kill(X))
            keep = torch.ones(X.size(), dtype=torch.bool)
            keep[kidxs] = False
            X = X[keep]

            ridxs = sample(replicate(X))
            X = torch.cat([X, X[ridxs].clone()])

    df = pd.DataFrame.from_records(records)

    return (
        ggplot(df, aes(x="step", y="cells"))
        + geom_area(data=df[df["[X]"] == -1], alpha=0.5)
        + geom_col(aes(fill="[X]", group="[X]"), data=df[df["[X]"] != -1], width=40)
        + facet_grid("split ~ .", labeller="label_both")
        + theme(figure_size=(8, 4))
    ), "random_splits"


def create_plots(imgs_dir: Path):
    def replicate(t: torch.Tensor, k=15, n=5) -> torch.Tensor:
        return t**n / (t**n + k**n)

    def kill(t: torch.Tensor, k=1, n=7) -> torch.Tensor:
        return k**n / (t**n + k**n)

    def sample(p: torch.Tensor) -> list[int]:
        idxs = torch.argwhere(torch.bernoulli(p))
        return idxs.flatten().tolist()

    g, name = _plot_random_splits(
        kill=kill,
        replicate=replicate,
        sample=sample,
        split_ratios=[0.1, 0.2, 0.3],
        n_mols=[3, 4, 5, 6],
    )
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=True)
