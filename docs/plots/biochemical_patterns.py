from pathlib import Path

import magicsoup as ms
import pandas as pd
from plotnine import *  # pylint: disable=W0614,W0401

from docs.plots.util import set_catal, set_regul, set_trnsp


def _plot_relay_switch(
    chemistry: ms.Chemistry, n_seconds: int = 100, vmax: float = 1, km: float = 10, kr: float = 1
) -> tuple[ggplot, str]:
    m2i = chemistry.mol_2_idx
    A = chemistry.molname_2_mol["A"]
    B = chemistry.molname_2_mol["B"]
    C = chemistry.molname_2_mol["C"]
    E = chemistry.molname_2_mol["E"]

    switch_seconds = n_seconds // 4
    switches = [i for i in range(n_seconds) if i % switch_seconds == 0]

    records = []
    world = ms.World(chemistry=chemistry, culture_kwargs={"map_size": 1})
    c = int(world.spawn_cells(genomes=[""])[0])
    world.cells.update_p(p_req=3)

    set_catal(world=world, c=c, p=0, subs=[A, E], prods=[B], km=km, vmax=vmax)
    set_regul(world=world, c=c, p=0, kr=kr, mol=C, act=False)

    set_catal(world=world, c=c, p=1, subs=[B, E], prods=[A], km=km, vmax=vmax)
    set_regul(world=world, c=c, p=1, kr=kr, mol=C, act=True)

    set_trnsp(world=world, c=c, p=2, km=0.1, vmax=10, mol=E)

    world.culture.molecule_map[m2i[E]] = 10.0
    world.cells.molecules[:] = 0.0
    world.cells.molecules[c, m2i[E]] = 10.0
    world.cells.molecules[c, m2i[A]] = 2.0
    world.cells.molecules[c, m2i[B]] = 2.0

    has_c = False
    for t in range(n_seconds):
        if t in switches:
            has_c = not has_c

        world.cells.molecules[c, m2i[C]] = 10.0 if has_c else 0.0

        xs = {f"[{m}]": float(world.cells.molecules[c, m2i[m]]) for m in [A, B]}
        records.append({"t": t, **xs})
        world.advance_time()

    df = pd.DataFrame.from_records(records)
    df = df.melt(id_vars=["t"])

    return (
        ggplot(df)
        + geom_line(aes(x="t", y="value", color="variable"))
        + geom_vline(xintercept=switches, color="gray", linetype="dashed")
        + xlab("time[s]")
        + ylab("concentration[mM]")
        + theme(figure_size=(8, 2), legend_title=element_blank())
    ), "relay-switch"


def _plot_bistable_switch(
    chemistry: ms.Chemistry, n_seconds: int = 25, vmax: float = 1, km: float = 10, kr: float = 1
) -> tuple[ggplot, str]:
    m2i = chemistry.mol_2_idx
    A = chemistry.molname_2_mol["A"]
    B = chemistry.molname_2_mol["B"]
    E = chemistry.molname_2_mol["E"]
    favors = [A, B]

    records = []
    for favor in favors:
        world = ms.World(chemistry=chemistry, culture_kwargs={"map_size": 1})
        c = int(world.spawn_cells(genomes=[""])[0])
        world.cells.update_p(p_req=3)

        set_catal(world=world, c=c, p=0, subs=[A, E], prods=[B], km=km, vmax=vmax)
        set_regul(world=world, c=c, p=0, kr=kr, mol=A, act=False)

        set_catal(world=world, c=c, p=1, subs=[B, E], prods=[A], km=km, vmax=vmax)
        set_regul(world=world, c=c, p=1, kr=kr, mol=B, act=False)

        set_trnsp(world=world, c=c, p=2, km=0.1, vmax=10, mol=E)

        world.culture.molecule_map[m2i[E]] = 10.0
        world.cells.molecules[:] = 0.0
        world.cells.molecules[c, m2i[E]] = 10.0
        world.cells.molecules[c, m2i[A]] = 2.0
        world.cells.molecules[c, m2i[B]] = 2.0
        world.cells.molecules[c, m2i[favor]] += 0.01

        for t in range(n_seconds):
            xs = {f"[{m}]": float(world.cells.molecules[c, m2i[m]]) for m in [A, B]}
            records.append({"t": t, "favor": str(favor), **xs})
            world.advance_time()

    df = pd.DataFrame.from_records(records)
    df = df.melt(id_vars=["t", "favor"])

    return (
        ggplot(df)
        + geom_line(aes(x="t", y="value", color="variable"))
        + facet_grid(". ~ favor", labeller="label_both")
        + xlab("time[s]")
        + ylab("concentration[mM]")
        + theme(figure_size=(8, 2), legend_title=element_blank())
    ), "bistable-switch"


def _plot_bistable_switch_signal_propagation(
    chemistry: ms.Chemistry, n_seconds: int = 50, vmax: float = 10, km: float = 1, kr: float = 1
) -> tuple[ggplot, str]:
    m2i = chemistry.mol_2_idx
    A = chemistry.molname_2_mol["A"]
    B = chemistry.molname_2_mol["B"]
    E = chemistry.molname_2_mol["E"]

    records = []
    world = ms.World(chemistry=chemistry, culture_kwargs={"map_size": 2})
    world.spawn_cells(genomes=[""] * 4)[0]
    world.cells.update_p(p_req=5)

    for c in range(4):
        set_catal(world=world, c=c, p=0, subs=[A, E], prods=[B], km=km, vmax=vmax)
        set_regul(world=world, c=c, p=0, kr=kr, mol=A, act=False)

        set_catal(world=world, c=c, p=1, subs=[B, E], prods=[A], km=km, vmax=vmax)
        set_regul(world=world, c=c, p=1, kr=kr, mol=B, act=False)

        set_trnsp(world=world, c=c, p=2, km=1, vmax=10, mol=E)
        set_trnsp(world=world, c=c, p=3, km=1, vmax=1, mol=A)
        set_trnsp(world=world, c=c, p=4, km=1, vmax=1, mol=B)

    world.culture.molecule_map[m2i[A]] = 2.0
    world.culture.molecule_map[m2i[B]] = 2.0
    world.cells.molecules[:] = 0.0
    world.cells.molecules[:, m2i[A]] = 2.0
    world.cells.molecules[:, m2i[B]] = 2.0
    world.cells.molecules[0, m2i[B]] += 0.01

    for t in range(n_seconds):
        for c in range(4):
            xs = {f"[{m}]": float(world.cells.molecules[c, m2i[m]]) for m in [A, B]}
            records.append({"t": t, "cell": c, **xs})
        world.advance_time()
        world.culture.molecule_map[m2i[E]] = 10.0

    df = pd.DataFrame.from_records(records)
    df = df.melt(id_vars=["t", "cell"])

    return (
        ggplot(df)
        + geom_line(aes(x="t", y="value", color="variable"))
        + facet_grid("cell ~ .", labeller="label_both")
        + xlab("time[s]")
        + ylab("concentration[mM]")
        + theme(figure_size=(8, 5), legend_title=element_blank())
    ), "bistable-switch-signal-propagation"


def create_plots(imgs_dir: Path, transparent: bool):
    A = ms.Molecule("A", 1e3)
    B = ms.Molecule("B", 1e3)
    C = ms.Molecule("C", 1e3)
    E = ms.Molecule("E", 10 * 1e3)
    AB = ms.Reaction([A, E], [B])
    BA = ms.Reaction([B, E], [A])
    chemistry = ms.Chemistry(molecules=[A, B, C, E], reactions=[AB, BA])

    g, name = _plot_relay_switch(chemistry=chemistry)
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=transparent)
    g, name = _plot_bistable_switch(chemistry=chemistry)
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=transparent)
    g, name = _plot_bistable_switch_signal_propagation(chemistry=chemistry)
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=transparent)
