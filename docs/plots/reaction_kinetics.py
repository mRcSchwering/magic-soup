# type: ignore
from pathlib import Path
import pandas as pd
from plotnine import *
import magicsoup as ms
from plots.util import record_concentrations, generate_cell


def _run_enzymatic_activity(
    n_steps: int, world: ms.World, cells_df: pd.DataFrame
) -> pd.DataFrame:
    records = []
    for si in range(n_steps):
        record_concentrations(
            record={"step": si}, records=records, world=world, cells_df=cells_df
        )
        world.enzymatic_activity()
    return pd.DataFrame.from_records(records)


def _plot_simple_exporter_kinetics(
    world: ms.World, mol: ms.Molecule, kms: list[float], vmaxs: list[float], nsteps=100
):
    records = []
    for km in kms:
        for vmax in vmaxs:
            p0 = [
                ms.TransporterDomainFact(
                    molecule=mol, km=km, vmax=vmax, is_exporter=True
                )
            ]
            ggen = ms.GenomeFact(world=world, proteome=[p0])
            ci = generate_cell(ggen=ggen, world=world)
            cell = world.get_cell(by_idx=ci)
            records.append(
                {
                    "cell": ci,
                    "Vmax": vmax,
                    "Km": km,
                    "x": cell.position[0],
                    "y": cell.position[1],
                }
            )

    cells_df = pd.DataFrame.from_records(records)
    cells_df["Km"] = pd.Categorical(cells_df["Km"], categories=kms)
    cells_df["Vmax"] = pd.Categorical(cells_df["Vmax"], categories=vmaxs)

    world.molecule_map[:] = 0.0
    world.cell_molecules[:] = 2.0
    sim_df = _run_enzymatic_activity(world=world, n_steps=nsteps, cells_df=cells_df)
    df = pd.merge(cells_df, sim_df[sim_df["molecule"] == "A"], on="cell")
    world.kill_cells(cell_idxs=list(range(world.n_cells)))

    return (
        ggplot(df)
        + geom_line(aes(x="step", y="[x]", color="location"))
        + facet_grid("Km ~ Vmax", labeller="label_both", scales="fixed")
        + scale_color_brewer(type="Qualitative", palette="Set2")
        + theme(figure_size=(8, 4))
    ), "simple_exporter_kinetics"


def _plot_simple_catalysis_kinetics(
    world: ms.World,
    reaction,
    kms: list[float],
    vmaxs: list[float],
    nsteps=100,
):
    records = []
    for km in kms:
        for vmax in vmaxs:
            p0 = [ms.CatalyticDomainFact(reaction=reaction, km=km, vmax=vmax)]
            ggen = ms.GenomeFact(world=world, proteome=[p0])
            ci = generate_cell(ggen=ggen, world=world)
            cell = world.get_cell(by_idx=ci)
            records.append(
                {
                    "cell": ci,
                    "Vmax": vmax,
                    "Km": km,
                    "x": cell.position[0],
                    "y": cell.position[1],
                }
            )

    cells_df = pd.DataFrame.from_records(records)
    cells_df["Km"] = pd.Categorical(cells_df["Km"], categories=kms)
    cells_df["Vmax"] = pd.Categorical(cells_df["Vmax"], categories=vmaxs)

    world.cell_molecules[:] = 0.0
    world.cell_molecules[:, 0] = 2.0
    sim_df = _run_enzymatic_activity(world=world, n_steps=nsteps, cells_df=cells_df)
    df = pd.merge(cells_df, sim_df[sim_df["location"] == "int"], on="cell")
    world.kill_cells(cell_idxs=list(range(world.n_cells)))

    return (
        ggplot(df[df["molecule"] != "B"])
        + geom_line(aes(x="step", y="[x]", color="molecule"))
        + facet_grid("Km ~ Vmax", labeller="label_both", scales="fixed")
        + scale_color_brewer(type="Qualitative", palette="Set2")
        + theme(figure_size=(8, 4))
    ), "simple_catalysis_kinetics"


def _plot_simple_catalysis_kinetics2(
    world: ms.World,
    reaction,
    kms: list[float],
    vmaxs: list[float],
    nsteps=100,
):
    records = []
    for km in kms:
        for vmax in vmaxs:
            p0 = [ms.CatalyticDomainFact(reaction=reaction, km=km, vmax=vmax)]
            ggen = ms.GenomeFact(world=world, proteome=[p0])
            ci = generate_cell(ggen=ggen, world=world)
            cell = world.get_cell(by_idx=ci)
            records.append(
                {
                    "cell": ci,
                    "Vmax": vmax,
                    "Km": km,
                    "x": cell.position[0],
                    "y": cell.position[1],
                }
            )

    cells_df = pd.DataFrame.from_records(records)
    cells_df["Km"] = pd.Categorical(cells_df["Km"], categories=kms)
    cells_df["Vmax"] = pd.Categorical(cells_df["Vmax"], categories=vmaxs)

    world.cell_molecules[:, 0] = 1.0
    world.cell_molecules[:, 1] = 2.0
    world.cell_molecules[:, 2] = 3.0
    sim_df = _run_enzymatic_activity(world=world, n_steps=nsteps, cells_df=cells_df)
    df = pd.merge(cells_df, sim_df[sim_df["location"] == "int"], on="cell")
    world.kill_cells(cell_idxs=list(range(world.n_cells)))

    return (
        ggplot(df)
        + geom_line(aes(x="step", y="[x]", color="molecule"))
        + facet_grid("Km ~ Vmax", labeller="label_both", scales="fixed")
        + scale_color_brewer(type="Qualitative", palette="Set2")
        + theme(figure_size=(8, 4))
    ), "simple_catalysis_kinetics2"


def _plot_coupled_vs_seperate_kinetics(
    world: ms.World, react1, react2, km=1.0, vmax=0.1, nsteps=100
):
    records = []
    world.kill_cells(cell_idxs=list(range(world.n_cells)))
    p0 = [ms.CatalyticDomainFact(reaction=react1, km=km, vmax=vmax)]
    p1 = [ms.CatalyticDomainFact(reaction=react2, km=km, vmax=vmax)]
    ggen = ms.GenomeFact(world=world, proteome=[p0, p1])
    ci = generate_cell(ggen=ggen, world=world)
    cell = world.get_cell(by_idx=ci)
    records.append(
        {"cell": ci, "type": "2 proteins", "x": cell.position[0], "y": cell.position[1]}
    )

    p0 = [
        ms.CatalyticDomainFact(reaction=react1, km=km, vmax=vmax),
        ms.CatalyticDomainFact(reaction=react2, km=km, vmax=vmax),
    ]
    ggen = ms.GenomeFact(world=world, proteome=[p0])
    ci = generate_cell(ggen=ggen, world=world, max_tries=10)
    cell = world.get_cell(by_idx=ci)
    records.append(
        {"cell": ci, "type": "1 protein", "x": cell.position[0], "y": cell.position[1]}
    )

    cells_df = pd.DataFrame.from_records(records)

    world.cell_molecules[:, 0] = 3.0
    world.cell_molecules[:, 1] = 0.5
    world.cell_molecules[:, 2] = 1.0
    sim_df = _run_enzymatic_activity(world=world, n_steps=nsteps, cells_df=cells_df)
    df = pd.merge(cells_df, sim_df[sim_df["location"] == "int"], on="cell")
    world.kill_cells(cell_idxs=list(range(world.n_cells)))

    return (
        ggplot(df)
        + geom_line(aes(x="step", y="[x]", color="molecule"))
        + facet_grid(". ~ type")
        + scale_color_brewer(type="Qualitative", palette="Set2")
        + theme(figure_size=(8, 2))
    ), "coupled_vs_seperate_kinetics"


def _plot_inhibited_exporter_kinetics(
    world: ms.World, mol, eff, kms: list[float], concentrations: list[float], nsteps=100
):
    records = []
    world.kill_cells(cell_idxs=list(range(world.n_cells)))
    for km in kms:
        for x in concentrations:
            p0 = [
                ms.TransporterDomainFact(
                    molecule=mol, km=1.0, vmax=1.0, is_exporter=True
                ),
                ms.RegulatoryDomainFact(
                    effector=eff,
                    km=km,
                    is_inhibiting=True,
                    is_transmembrane=False,
                    hill=1,
                ),
            ]
            ggen = ms.GenomeFact(world=world, proteome=[p0])
            ci = generate_cell(ggen=ggen, world=world, max_tries=10)
            cell = world.get_cell(by_idx=ci)
            records.append(
                {
                    "cell": ci,
                    "Km(inh)": km,
                    "[x](inh)": x,
                    "x": cell.position[0],
                    "y": cell.position[1],
                }
            )

    cells_df = pd.DataFrame.from_records(records)
    cells_df["Km(inh)"] = pd.Categorical(cells_df["Km(inh)"], categories=kms)
    cells_df["[x](inh)"] = pd.Categorical(
        cells_df["[x](inh)"], categories=concentrations
    )

    world.molecule_map[:] = 0.0
    world.cell_molecules[:, 0] = 2.0
    for x in concentrations:
        is_x = cells_df.loc[cells_df["[x](inh)"] == x, "cell"].tolist()
        world.cell_molecules[is_x, 1] = x

    sim_df = _run_enzymatic_activity(world=world, n_steps=nsteps, cells_df=cells_df)
    df = pd.merge(cells_df, sim_df[sim_df["molecule"] == "A"], on="cell")
    world.kill_cells(cell_idxs=list(range(world.n_cells)))

    return (
        ggplot(df)
        + geom_line(aes(x="step", y="[x]", color="location"))
        + facet_grid("Km(inh) ~ [x](inh)", labeller="label_both", scales="fixed")
        + scale_color_brewer(type="Qualitative", palette="Set2")
        + theme(figure_size=(8, 4))
    ), "inhibited_exporter_kinetics"


def _plot_inhibited_catalysis_kinetics(
    world: ms.World, react1, eff1, react2, eff2, kms: list[float], nsteps=100
):
    records = []
    world.kill_cells(cell_idxs=list(range(world.n_cells)))
    for km in kms:
        p0 = [
            ms.CatalyticDomainFact(reaction=react1, km=1.0, vmax=0.3),
            ms.RegulatoryDomainFact(
                effector=eff1, km=km, is_inhibiting=True, is_transmembrane=False, hill=1
            ),
        ]
        p1 = [
            ms.CatalyticDomainFact(reaction=react2, km=1.0, vmax=0.3),
            ms.RegulatoryDomainFact(
                effector=eff2, km=km, is_inhibiting=True, is_transmembrane=False, hill=1
            ),
        ]
        ggen = ms.GenomeFact(world=world, proteome=[p0, p1])
        ci = generate_cell(ggen=ggen, world=world, max_tries=20, allow_more_prots=True)
        cell = world.get_cell(by_idx=ci)
        records.append(
            {"cell": ci, "Km(inh)": km, "x": cell.position[0], "y": cell.position[1]}
        )

    cells_df = pd.DataFrame.from_records(records)
    cells_df["Km(inh)"] = pd.Categorical(cells_df["Km(inh)"], categories=kms)

    world.cell_molecules[:] = 0.0
    world.cell_molecules[:, 0] = 3.0
    sim_df = _run_enzymatic_activity(world=world, n_steps=nsteps, cells_df=cells_df)
    df = pd.merge(cells_df, sim_df[sim_df["location"] == "int"], on="cell")

    return (
        ggplot(df)
        + geom_line(aes(x="step", y="[x]", color="molecule"))
        + facet_grid(". ~ Km(inh)", labeller="label_both", scales="fixed")
        + scale_color_brewer(type="Qualitative", palette="Set2")
        + theme(figure_size=(10, 2))
    ), "inhibited_catalysis_kinetics"


def create_plots(imgs_dir: Path):
    ms.Molecule._instances = {}  # rm previous instances
    _ma = ms.Molecule("A", 10 * 1e3)
    _mb = ms.Molecule("B", 5 * 1e3)
    _mc = ms.Molecule("C", 20 * 1e3)
    molecules = [_ma, _mb, _mc]
    reactions = [([_ma, _mb], [_mc]), ([_ma, _ma, _ma], [_mc])]
    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry, map_size=8)

    g, name = _plot_simple_exporter_kinetics(
        world=world, mol=_ma, kms=[0.1, 1.0, 10.0], vmaxs=[0.1, 1.0, 10.0]
    )
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_simple_catalysis_kinetics(
        world=world,
        reaction=([_ma, _ma, _ma], [_mc]),
        kms=[0.1, 1.0, 10.0],
        vmaxs=[0.1, 1.0, 10.0],
    )
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_simple_catalysis_kinetics2(
        world=world,
        reaction=([_ma, _mb], [_mc]),
        kms=[0.1, 1.0, 10.0],
        vmaxs=[0.1, 1.0, 10.0],
    )
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_coupled_vs_seperate_kinetics(
        world=world, react1=([_ma, _mb], [_mc]), react2=([_ma, _ma, _ma], [_mc])
    )
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_inhibited_exporter_kinetics(
        world=world,
        mol=_ma,
        eff=_mb,
        kms=[0.1, 1.0, 10.0],
        concentrations=[0.1, 1.0, 10.0],
    )
    g.save(imgs_dir / f"{name}.png", dpi=200)
    g, name = _plot_inhibited_catalysis_kinetics(
        world=world,
        react1=([_ma, _mb], [_mc]),
        eff1=_mc,
        react2=([_ma, _ma, _ma], [_mc]),
        eff2=_mb,
        kms=[0.1, 1.0, 10.0],
    )
    g.save(imgs_dir / f"{name}.png", dpi=200)
