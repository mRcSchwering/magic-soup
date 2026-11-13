from itertools import product
from pathlib import Path

import magicsoup as ms
import pandas as pd
from plotnine import *  # pylint: disable=W0614,W0401

from docs.plots.util import set_catal, set_regul, set_trnsp


def _plot_simple_exporter_kinetics(
    chemistry: ms.Chemistry, n_seconds: int = 100
) -> tuple[ggplot, str]:
    m2i = chemistry.mol_2_idx
    A = chemistry.molname_2_mol["A"]
    vmaxs = [0.1, 1, 10]
    kms = [0.1, 1, 10]

    records = []
    for km, vmax in product(kms, vmaxs):
        world = ms.World(chemistry=chemistry, culture_kwargs={"map_size": 1})
        c = int(world.spawn_cells(genomes=[""])[0])
        world.cells.update_p(p_req=1)

        set_trnsp(world=world, c=c, p=0, mol=A, km=km, vmax=vmax)
        world.culture.molecule_map[m2i[A]] = 10.0
        world.cells.molecules[:] = 0.0

        for t in range(n_seconds):
            m_int = float(world.cells.molecules[c, m2i[A]])
            m_ext = float(world.culture.molecule_map[m2i[A], 0, 0])
            records.append({"t": t, "Km": km, "Vmax": vmax, "[int]": m_int, "[ext]": m_ext})
            world.advance_time()

    df = pd.DataFrame.from_records(records)
    df = df.melt(id_vars=["t", "Km", "Vmax"])

    return (
        ggplot(df)
        + geom_line(aes(x="t", y="value", color="variable"))
        + facet_grid("Km ~ Vmax", labeller="label_both")
        + xlab("time[s]")
        + ylab("concentration[mM]")
        + theme(figure_size=(8, 5), legend_title=element_blank())
    ), "simple-exporter-kinetics"


def _plot_simple_catalysis_kinetics(
    chemistry: ms.Chemistry,
    vmax: float = 0.3,
    n_seconds: int = 100,
) -> tuple[ggplot, str]:
    m2i = chemistry.mol_2_idx
    A = chemistry.molname_2_mol["A"]
    B = chemistry.molname_2_mol["B"]
    C = chemistry.molname_2_mol["C"]
    A_B = ms.Reaction([A], [B])
    A3_C = ms.Reaction([A] * 3, [C])
    AB_C = ms.Reaction([A, B], [C])
    kms = [0.1, 1, 10]

    records = []
    for km, react in product(kms, [A_B, A3_C, AB_C]):
        world = ms.World(chemistry=chemistry, culture_kwargs={"map_size": 1})
        c = int(world.spawn_cells(genomes=[""])[0])
        world.cells.update_p(p_req=1)
        set_catal(
            world=world, c=c, p=0, subs=react.substrates, prods=react.products, km=km, vmax=vmax
        )
        world.cells.molecules[c, m2i[A]] = 10.0
        world.cells.molecules[c, m2i[C]] = 10.0

        for t in range(n_seconds):
            xs = {f"[{m}]": float(world.cells.molecules[c, m2i[m]]) for m in chemistry.molecules}
            records.append({"t": t, "Km": f"Km: {km:.1f}", "reaction": str(react), **xs})
            world.advance_time()

    df = pd.DataFrame.from_records(records)
    df = df.melt(id_vars=["t", "Km", "reaction"])

    return (
        ggplot(df)
        + geom_line(aes(x="t", y="value", color="variable"))
        + facet_grid("Km ~ reaction")
        + xlab("time[s]")
        + ylab("concentration[mM]")
        + theme(figure_size=(8, 5), legend_title=element_blank())
    ), "simple-reaction-kinetics"


def _plot_coupled_kinetics(
    chemistry: ms.Chemistry,
    km: float = 1.0,
    vmax: float = 0.3,
    n_seconds: int = 100,
) -> tuple[ggplot, str]:
    m2i = chemistry.mol_2_idx
    A = chemistry.molname_2_mol["A"]
    B = chemistry.molname_2_mol["B"]
    C = chemistry.molname_2_mol["C"]
    conds = {
        "3 A <-> C | A + B <-> C": [(3 * [A], [C]), ([A, B], [C])],
        "4 A + B <-> 2 C": [(4 * [A] + [B], 2 * [C])],
        "3 A + C <-> A + B + C": [(3 * [A] + [C], [A, B, C])],
    }

    records = []
    for label, proteins in conds.items():
        world = ms.World(chemistry=chemistry, culture_kwargs={"map_size": 1})
        c = int(world.spawn_cells(genomes=[""])[0])
        world.cells.update_p(p_req=2)
        for p, (subs, prods) in enumerate(proteins):
            set_catal(world=world, c=c, p=p, subs=subs, prods=prods, km=km, vmax=vmax)

        world.cells.molecules[c, m2i[A]] = 10.0
        world.cells.molecules[c, m2i[B]] = 10.0
        world.cells.molecules[c, m2i[C]] = 10.0

        for t in range(n_seconds):
            xs = {f"[{m}]": float(world.cells.molecules[c, m2i[m]]) for m in chemistry.molecules}
            records.append({"t": t, "reaction": label, **xs})
            world.advance_time()

    df = pd.DataFrame.from_records(records)
    df = df.melt(id_vars=["t", "reaction"])
    df["reaction"] = pd.Categorical(df["reaction"], categories=list(conds), ordered=True)

    return (
        ggplot(df)
        + geom_line(aes(x="t", y="value", color="variable"))
        + facet_grid(". ~ reaction")
        + xlab("time[s]")
        + ylab("concentration[mM]")
        + theme(figure_size=(8, 2), legend_title=element_blank())
    ), "coupled-reaction-kinetics"


def _plot_simple_exporter_inhibition(
    chemistry: ms.Chemistry, n_seconds: int = 100, km: float = 1.0, vmax: float = 1.0
) -> tuple[ggplot, str]:
    m2i = chemistry.mol_2_idx
    A = chemistry.molname_2_mol["A"]
    B = chemistry.molname_2_mol["B"]
    krs = [0.1, 1, 10]
    xbs = [0.1, 1, 10]

    records = []
    for kr, xb in product(krs, xbs):
        world = ms.World(chemistry=chemistry, culture_kwargs={"map_size": 1})
        c = int(world.spawn_cells(genomes=[""])[0])
        world.cells.update_p(p_req=1)

        set_trnsp(world=world, c=c, p=0, mol=A, km=km, vmax=vmax)
        set_regul(world=world, c=c, p=0, kr=kr, mol=B, act=False)
        world.culture.molecule_map[m2i[A]] = 10.0
        world.cells.molecules[:] = 0.0
        world.cells.molecules[c, m2i[B]] = xb

        for t in range(n_seconds):
            m_int = float(world.cells.molecules[c, m2i[A]])
            m_ext = float(world.culture.molecule_map[m2i[A], 0, 0])
            records.append({"t": t, "Kr": kr, "[eff]": xb, "[int]": m_int, "[ext]": m_ext})
            world.advance_time()

    df = pd.DataFrame.from_records(records)
    df = df.melt(id_vars=["t", "Kr", "[eff]"])

    return (
        ggplot(df)
        + geom_line(aes(x="t", y="value", color="variable"))
        + facet_grid("Kr ~ [eff]", labeller="label_both")
        + xlab("time[s]")
        + ylab("concentration[mM]")
        + theme(figure_size=(8, 5), legend_title=element_blank())
    ), "simple-exporter-inhibition"


def _plot_negative_feedback(
    chemistry: ms.Chemistry, n_seconds: int = 100, vmax: float = 1, km: float = 1.0
) -> tuple[ggplot, str]:
    m2i = chemistry.mol_2_idx
    A = chemistry.molname_2_mol["A"]
    B = chemistry.molname_2_mol["B"]
    C = chemistry.molname_2_mol["C"]
    krs = [0.1, 1, 10]

    records = []
    for kr in krs:
        world = ms.World(chemistry=chemistry, culture_kwargs={"map_size": 1})
        c = int(world.spawn_cells(genomes=[""])[0])
        world.cells.update_p(p_req=2)

        set_catal(world=world, c=c, p=0, subs=[A, B], prods=[C], km=km, vmax=vmax)
        set_regul(world=world, c=c, p=0, kr=kr, mol=C, act=False)

        set_catal(world=world, c=c, p=1, subs=[A] * 3, prods=[C], km=km, vmax=vmax)
        set_regul(world=world, c=c, p=1, kr=kr, mol=B, act=False)

        world.cells.molecules[:] = 0.0
        world.cells.molecules[c, m2i[A]] = 10.0

        for t in range(n_seconds):
            xs = {f"[{m}]": float(world.cells.molecules[c, m2i[m]]) for m in chemistry.molecules}
            records.append({"t": t, "Kr": kr, **xs})
            world.advance_time()

    df = pd.DataFrame.from_records(records)
    df = df.melt(id_vars=["t", "Kr"])

    return (
        ggplot(df)
        + geom_line(aes(x="t", y="value", color="variable"))
        + facet_grid(". ~ Kr", labeller="label_both")
        + xlab("time[s]")
        + ylab("concentration[mM]")
        + theme(figure_size=(8, 2), legend_title=element_blank())
    ), "negative-feedback"


def create_plots(imgs_dir: Path, transparent: bool):
    _ma = ms.Molecule("A", 10 * 1e3)
    _mb = ms.Molecule("B", 5 * 1e3)
    _mc = ms.Molecule("C", 20 * 1e3)
    _r_a_b = ms.Reaction([_ma], [_mb])
    _r_3a_c = ms.Reaction([_ma] * 3, [_mc])
    _r_ab_c = ms.Reaction([_ma, _mb], [_mc])
    molecules = [_ma, _mb, _mc]
    reactions = [_r_a_b, _r_3a_c, _r_ab_c]
    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)

    g, name = _plot_simple_exporter_kinetics(chemistry=chemistry)
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=transparent)
    g, name = _plot_simple_catalysis_kinetics(chemistry=chemistry)
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=transparent)
    g, name = _plot_coupled_kinetics(chemistry=chemistry)
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=transparent)
    g, name = _plot_simple_exporter_inhibition(chemistry=chemistry)
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=transparent)
    g, name = _plot_negative_feedback(chemistry=chemistry)
    g.save(imgs_dir / f"{name}.png", dpi=200, transparent=transparent)
