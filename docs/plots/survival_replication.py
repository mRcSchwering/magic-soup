# type: ignore
from typing import Callable
from pathlib import Path
import pandas as pd
import torch
from plotnine import *


def _plot_probability_functions():
    def increasing(t: torch.Tensor, k: float, n: int) -> torch.Tensor:
        return t**n / (t**n + k**n)

    def decreasing(t: torch.Tensor, k: float, n: int) -> torch.Tensor:
        return k**n / (t**n + k**n)

    X = torch.arange(0, 10, 0.01)

    dfs = []
    for k in (0.5, 1, 5, 10):
        for n in (1, 3, 5, 7):
            df = pd.DataFrame(
                {
                    "[X]": X.tolist() + X.tolist(),
                    "y": increasing(X, k, n).tolist() + decreasing(X, k, n).tolist(),
                    "d": ["increasing"] * len(X) + ["decreasing"] * len(X),
                    "k": 2 * [k] * len(X),
                    "n": 2 * [n] * len(X),
                }
            )
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return (
        ggplot(df, aes(y="y", x="[X]"))
        + geom_line(aes(color="n", group="n"))
        + facet_grid("d ~ k", scales="free", labeller="label_both")
        + theme(figure_size=(10, 4))
    ), "probability_functions"


def _plot_cell_dying_replicating(actions: dict, n_mols: list[float], nsteps=1000):
    records = []
    for n_mol in n_mols:
        for action, fun in actions.items():
            for step in range(nsteps):
                records.append(
                    {
                        "step": step,
                        "[X]": n_mol,
                        "p(action)": (1 - (1 - fun(n_mol)) ** step),
                        "action": action,
                    }
                )

    df = pd.DataFrame.from_records(records)

    return (
        ggplot(df, aes(y="p(action)", x="step"))
        + geom_line(aes(color="[X]", group="[X]"))
        + facet_grid(". ~ action", labeller="label_both")
        + theme(figure_size=(6, 2))
    ), "cell_dying_replicating"


def _plot_simulated_growth(
    sample: Callable,
    kill: Callable,
    replicate: Callable,
    n_mols: list[float],
    nsteps=100,
    ncells=1000,
):
    records = []
    for n_mol in n_mols:
        X = torch.ones(ncells) * n_mol
        for step in range(nsteps):
            kidxs = sample(kill(X))
            keep = torch.ones(X.size(), dtype=bool)
            keep[kidxs] = False
            X = X[keep]

            ridxs = sample(replicate(X))
            X = torch.cat([X, torch.ones(len(ridxs)) * n_mol])

            records.append(
                {
                    "step": step,
                    "[X]": n_mol,
                    "cells": len(X),
                }
            )

            if len(X) > 1e6:
                break

    df = pd.DataFrame.from_records(records)

    return (
        ggplot(df, aes(y="cells", x="step"))
        + geom_line(aes(color="[X]", group="[X]"))
        + theme(figure_size=(6, 2))
    ), "simulated_growth"


def _plot_multiple_die_divide_conditions(
    sample: Callable,
    kill: Callable,
    replicate: Callable,
    setups: list[tuple[int, int]],
    n_mol: float,
    nsteps=100,
    ncells=1000,
):
    records = []
    for n_kills, n_replications in setups:
        X = torch.ones(ncells) * n_mol
        label = f"{n_kills}k-{n_replications}r"
        for step in range(nsteps):
            for _ in range(n_kills):
                kidxs = sample(kill(X))
                keep = torch.ones(X.size(), dtype=bool)
                keep[kidxs] = False
                X = X[keep]

            for _ in range(n_replications):
                ridxs = sample(replicate(X))
                X = torch.cat([X, torch.ones(len(ridxs)) * n_mol])

            records.append(
                {
                    "step": step,
                    "cells": len(X),
                    "conditions": label,
                }
            )

            if len(X) > 1e6:
                break

    df = pd.DataFrame.from_records(records)

    return (
        ggplot(df, aes(y="cells", x="step"))
        + geom_line(aes(color="conditions"))
        + theme(figure_size=(6, 2))
    ), "multiple_die_divide_conditions"


def _plot_random_splits(
    sample: Callable,
    kill: Callable,
    replicate: Callable,
    split_ratios: list[float],
    n_mols: list[float],
    ncells=1000,
    nsteps=1000,
    thresh=7000,
):
    def random_split(t: torch.Tensor, r: float) -> list[int]:
        n = len(t)
        return t[torch.randint(n, size=(int(r * n),))]

    records = []
    for split_ratio in split_ratios:
        X = torch.cat([torch.ones(int(ncells / len(n_mols))) * d for d in n_mols])

        kwargs = {"step": 0, "split": split_ratio}
        for n_mol in n_mols:
            records.append({**kwargs, "[X]": n_mol, "cells": len(X[X == n_mol])})

        for step in range(nsteps):
            kwargs = {"step": step, "split": split_ratio}
            if len(X) > thresh:
                for n_mol in n_mols:
                    records.append(
                        {**kwargs, "[X]": n_mol, "cells": len(X[X == n_mol])}
                    )
                X = random_split(X, split_ratio)

            records.append({**kwargs, "[X]": -1, "cells": len(X)})

            kidxs = sample(kill(X))
            keep = torch.ones(X.size(), dtype=bool)
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


def _plot_biased_splits(
    sample: Callable,
    kill: Callable,
    replicate: Callable,
    biases: list[float],
    n_mols: list[float],
    split_ratio=0.2,
    ncells=1000,
    nsteps=1000,
    thresh=7000,
):
    def biased_split(t: torch.Tensor, b: float) -> list[int]:
        m = len(t) / len(t.unique())
        counts = {d.item(): len(t[t == d]) for d in t.unique()}
        samples = {k: (d * (1 - b) + m * b) * split_ratio for k, d in counts.items()}
        new_cells = [[k] * int(d) for k, d in samples.items()]
        return torch.tensor([dd for d in new_cells for dd in d])

    records = []
    for bias in biases:
        X = torch.cat([torch.ones(int(ncells / len(n_mols))) * d for d in n_mols])

        kwargs = {"step": 0, "bias": bias}
        for n_mol in n_mols:
            records.append({**kwargs, "[X]": n_mol, "cells": len(X[X == n_mol])})

        for step in range(nsteps):
            kwargs = {"step": step, "bias": bias}
            if len(X) > thresh:
                for n_mol in n_mols:
                    records.append(
                        {**kwargs, "[X]": n_mol, "cells": len(X[X == n_mol])}
                    )
                X = biased_split(X, bias)

            records.append({**kwargs, "[X]": -1, "cells": len(X)})

            kidxs = sample(kill(X))
            keep = torch.ones(X.size(), dtype=bool)
            keep[kidxs] = False
            X = X[keep]

            ridxs = sample(replicate(X))
            X = torch.cat([X, X[ridxs].clone()])

    df = pd.DataFrame.from_records(records)
    df["bias"] = pd.Categorical(df["bias"], categories=biases)

    return (
        ggplot(df, aes(x="step", y="cells"))
        + geom_area(data=df[df["[X]"] == -1], alpha=0.5)
        + geom_col(aes(fill="[X]", group="[X]"), data=df[df["[X]"] != -1], width=40)
        + facet_grid("bias ~ .", labeller="label_both")
        + theme(figure_size=(8, 4))
    ), "biased_splits"


def create_plots(imgs_dir: Path):
    def replicate(t: torch.Tensor, k=15, n=5) -> torch.Tensor:
        return t**n / (t**n + k**n)

    def kill(t: torch.Tensor, k=1, n=7) -> torch.Tensor:
        return k**n / (t**n + k**n)

    def sample(p: torch.Tensor) -> list[int]:
        idxs = torch.argwhere(torch.bernoulli(p))
        return idxs.flatten().tolist()

    g, name = _plot_probability_functions()
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_cell_dying_replicating(
        actions={"replicated": replicate, "killed": kill},
        n_mols=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_simulated_growth(
        kill=kill,
        replicate=replicate,
        sample=sample,
        n_mols=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_simulated_growth(
        kill=kill,
        replicate=replicate,
        sample=sample,
        n_mols=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_multiple_die_divide_conditions(
        kill=kill,
        replicate=replicate,
        sample=sample,
        setups=[(1, 1), (1, 2), (1, 4), (2, 1), (4, 1)],
        n_mol=3.07,  # to get 1k-1r in equilibrium
    )
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_random_splits(
        kill=kill,
        replicate=replicate,
        sample=sample,
        split_ratios=[0.1, 0.2, 0.3],
        n_mols=[3, 4, 5, 6],
    )
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_biased_splits(
        kill=kill,
        replicate=replicate,
        sample=sample,
        biases=[0.1, 0.5, 0.9],
        n_mols=[3, 4, 5, 6],
    )
    g.save(imgs_dir / f"{name}.png", dpi=200)
