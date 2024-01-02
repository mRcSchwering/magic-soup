# type: ignore
from pathlib import Path
import pandas as pd
from plotnine import *
import magicsoup as ms
from plots.util import generate_cell, record_concentrations


def _plot_switch_relay(nsteps=100, reg_km=1.0, low_high=(0.0, 4.0)):
    switches = [10, 30, 50, 70, 90]

    ms.Molecule._instances = {}  # rm previous instances
    _ma = ms.Molecule("A", 10 * 1e3)
    _mb = ms.Molecule("B", 10 * 1e3)
    _mc = ms.Molecule("C", 10 * 1e3)
    _me = ms.Molecule("E", 10 * 1e3)
    molecules = [_ma, _mb, _mc, _me]
    reactions = [([_ma, _me], [_mb]), ([_mb, _me], [_ma])]
    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry, map_size=2)

    p0 = [
        ms.CatalyticDomainFact(reaction=([_ma, _me], [_mb]), km=1.0, vmax=1.0),
        ms.RegulatoryDomainFact(
            effector=_mc, is_transmembrane=False, is_inhibiting=True, km=reg_km, hill=1
        ),
    ]
    p1 = [
        ms.CatalyticDomainFact(reaction=([_mb, _me], [_ma]), km=1.0, vmax=1.0),
        ms.RegulatoryDomainFact(
            effector=_mc, is_transmembrane=False, is_inhibiting=False, km=reg_km, hill=1
        ),
    ]

    ggen = ms.GenomeFact(world=world, proteome=[p0, p1])
    ci = generate_cell(ggen=ggen, world=world, max_tries=10)
    cell = world.get_cell(by_idx=ci)
    records = [{"cell": ci, "x": cell.position[0], "y": cell.position[1]}]
    cells_df = pd.DataFrame.from_records(records)

    world.cell_molecules[:] = 2.0
    is_high = False
    records = []
    for step in range(nsteps):
        if step in switches:
            is_high = not is_high
        world.cell_molecules[:, 2] = low_high[1 if is_high else 0]
        world.cell_molecules[:, 3] = 10.0
        record_concentrations(
            record={"step": step}, records=records, world=world, cells_df=cells_df
        )
        world.enzymatic_activity()

    sim_df = pd.DataFrame.from_records(records)
    df = pd.merge(cells_df, sim_df[sim_df["location"] == "int"], on="cell")

    return (
        ggplot(df[df["molecule"].isin(["A", "B"])])
        + geom_vline(
            aes(xintercept="x"),
            linetype="dashed",
            color="gray",
            data=pd.DataFrame({"x": switches}),
        )
        + geom_line(aes(x="step", y="[x]", color="molecule"))
        + scale_color_brewer(type="Qualitative", palette="Set2")
        + theme(figure_size=(8, 2))
    ), "switch_relay"


def _plot_bistable_switch(nsteps=25, reg_km=1.0):
    ms.Molecule._instances = {}  # rm previous instances
    _ma = ms.Molecule("A", 10 * 1e3)
    _mb = ms.Molecule("B", 10 * 1e3)
    _me = ms.Molecule("E", 100 * 1e3)
    molecules = [_ma, _mb, _me]
    reactions = [([_ma, _me], [_mb]), ([_mb, _me], [_ma])]
    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry, map_size=2)

    p0 = [
        ms.CatalyticDomainFact(reaction=([_ma, _me], [_mb]), km=1.0, vmax=1.0),
        ms.RegulatoryDomainFact(
            effector=_ma, is_transmembrane=False, is_inhibiting=True, km=reg_km, hill=3
        ),
    ]
    p1 = [
        ms.CatalyticDomainFact(reaction=([_mb, _me], [_ma]), km=1.0, vmax=1.0),
        ms.RegulatoryDomainFact(
            effector=_mb, is_transmembrane=False, is_inhibiting=True, km=reg_km, hill=3
        ),
    ]
    ggen = ms.GenomeFact(world=world, proteome=[p0, p1])

    records = []
    for mol in (_ma, _mb):
        ci = generate_cell(ggen=ggen, world=world, max_tries=10)
        cell = world.get_cell(by_idx=ci)
        records.append(
            {
                "cell": ci,
                "x": cell.position[0],
                "y": cell.position[1],
                "favor": mol.name,
            }
        )
        world.cell_molecules[ci, :] = 2.0
        world.cell_molecules[ci, world.chemistry.mol_2_idx[mol]] += 0.1
    cells_df = pd.DataFrame.from_records(records)

    records = []
    for step in range(nsteps):
        world.cell_molecules[:, 2] = 10.0
        record_concentrations(
            record={"step": step}, records=records, world=world, cells_df=cells_df
        )
        world.enzymatic_activity()

    sim_df = pd.DataFrame.from_records(records)
    df = pd.merge(cells_df, sim_df[sim_df["location"] == "int"], on="cell")

    return (
        ggplot(df[df["molecule"].isin(["A", "B"])])
        + geom_line(aes(x="step", y="[x]", color="molecule"))
        + facet_grid(". ~ favor", labeller="label_both")
        + scale_color_brewer(type="Qualitative", palette="Set2")
        + theme(figure_size=(8, 2))
    ), "bistable_switch"


def _plot_bistable_switch_cascade(nsteps=50, reg_km=1.0):
    ms.Molecule._instances = {}  # rm previous instances
    _ma = ms.Molecule("A", 10 * 1e3, permeability=0.1)
    _mb = ms.Molecule("B", 10 * 1e3, permeability=0.1)
    _me = ms.Molecule("E", 100 * 1e3)
    molecules = [_ma, _mb, _me]
    reactions = [([_ma, _me], [_mb]), ([_mb, _me], [_ma])]
    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry, map_size=2, mol_map_init="zeros")

    p0 = [
        ms.CatalyticDomainFact(reaction=([_ma, _me], [_mb]), km=1.0, vmax=1.0),
        ms.RegulatoryDomainFact(
            effector=_ma, is_transmembrane=False, is_inhibiting=True, km=reg_km, hill=3
        ),
    ]
    p1 = [
        ms.CatalyticDomainFact(reaction=([_mb, _me], [_ma]), km=1.0, vmax=1.0),
        ms.RegulatoryDomainFact(
            effector=_mb, is_transmembrane=False, is_inhibiting=True, km=reg_km, hill=3
        ),
    ]
    ggen = ms.GenomeFact(world=world, proteome=[p0, p1])

    records = []
    for _ in range(world.map_size**2):
        ci = generate_cell(ggen=ggen, world=world, max_tries=10)
        cell = world.get_cell(by_idx=ci)
        records.append({"cell": ci, "x": cell.position[0], "y": cell.position[1]})
    cells_df = pd.DataFrame.from_records(records)

    world.molecule_map[:] = 2.0
    world.cell_molecules[:] = 2.0
    records = []
    for step in range(nsteps):
        world.cell_molecules[:, 2] = 10.0
        record_concentrations(
            record={"step": step}, records=records, world=world, cells_df=cells_df
        )
        world.enzymatic_activity()
        world.diffuse_molecules()

    sim_df = pd.DataFrame.from_records(records)
    df = pd.merge(cells_df, sim_df[sim_df["location"] == "int"], on="cell")

    return (
        ggplot(df[df["molecule"].isin(["A", "B"])])
        + geom_line(aes(x="step", y="[x]", color="molecule"))
        + facet_grid("cell ~ .", labeller="label_both")
        + scale_color_brewer(type="Qualitative", palette="Set2")
        + theme(figure_size=(6, 4))
    ), "bistable_switch_cascade"


def _plot_cyclic_pathway(km=10.0, vmax=0.3, nsteps=50):
    ms.Molecule._instances = {}  # rm previous instances
    _ma = ms.Molecule("A", 10 * 1e3, permeability=0.1)
    _mb = ms.Molecule("B", 10 * 1e3, permeability=0.1)
    _mc = ms.Molecule("C", 10 * 1e3, permeability=0.1)
    _md = ms.Molecule("D", 10 * 1e3, permeability=0.1)
    _me = ms.Molecule("E", 100 * 1e3)
    molecules = [_ma, _mb, _mc, _md, _me]
    reactions = [
        ([_ma, _me], [_mb]),
        ([_mb, _me], [_mc]),
        ([_mc, _me], [_md]),
        ([_md, _me], [_ma]),
    ]
    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry, map_size=2, mol_map_init="zeros")

    p0 = [ms.CatalyticDomainFact(reaction=([_ma, _me], [_mb]), km=km, vmax=vmax)]
    p1 = [ms.CatalyticDomainFact(reaction=([_mb, _me], [_mc]), km=km, vmax=vmax)]
    p2 = [ms.CatalyticDomainFact(reaction=([_mc, _me], [_md]), km=km, vmax=vmax)]
    p3 = [ms.CatalyticDomainFact(reaction=([_md, _me], [_ma]), km=km, vmax=vmax)]
    ggen = ms.GenomeFact(world=world, proteome=[p0, p1, p2, p3])

    ci = generate_cell(ggen=ggen, world=world, max_tries=10)
    cell = world.get_cell(by_idx=ci)
    record = {"cell": ci, "x": cell.position[0], "y": cell.position[1]}
    cells_df = pd.DataFrame.from_records([record])

    world.cell_molecules[0, 0] = 5.0
    records = []
    for step in range(nsteps):
        world.cell_molecules[:, 4] = 10.0
        record_concentrations(
            record={"step": step}, records=records, world=world, cells_df=cells_df
        )
        world.enzymatic_activity()
        world.diffuse_molecules()

    sim_df = pd.DataFrame.from_records(records)
    df = pd.merge(cells_df, sim_df[sim_df["location"] == "int"], on="cell")

    return (
        ggplot(df[df["molecule"] != "E"])
        + geom_line(aes(x="step", y="[x]", color="molecule"))
        + scale_color_brewer(type="Qualitative", palette="Set2")
        + theme(figure_size=(8, 3))
    ), "cyclic_pathway"


def create_plots(imgs_dir: Path):
    g, name = _plot_switch_relay()
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_bistable_switch()
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_bistable_switch_cascade()
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_cyclic_pathway()
    g.save(imgs_dir / f"{name}.png", dpi=200)
