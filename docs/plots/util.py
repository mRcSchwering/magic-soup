import pandas as pd
import magicsoup as ms


def generate_cell(
    ggen: ms.GenomeFact, world: ms.World, max_tries=3, allow_more_prots=False
) -> int:
    n_prots = len(ggen.proteome)
    prot_dom_counts = [
        {
            ms.CatalyticDomain: sum(isinstance(dd, ms.CatalyticDomainFact) for dd in d),
            ms.TransporterDomain: sum(
                isinstance(dd, ms.TransporterDomainFact) for dd in d
            ),
            ms.RegulatoryDomain: sum(
                isinstance(dd, ms.RegulatoryDomainFact) for dd in d
            ),
        }
        for d in ggen.proteome
    ]

    i = 0
    success = False
    idxs: list[int] = []
    while not success:
        world.kill_cells(cell_idxs=idxs)
        i += 1
        if i > max_tries:
            raise ValueError(f"Could not generate cell after {i} tries")
        g = ggen.generate()
        idxs = world.spawn_cells(genomes=[g])
        if len(idxs) != 1:
            continue
        cell = world.get_cell(by_idx=idxs[0])
        if not allow_more_prots and len(cell.proteome) != n_prots:
            continue
        found_prots = [False] * n_prots
        for i, dom_counts in enumerate(prot_dom_counts):
            for prot in cell.proteome:
                found_doms = {d: False for d in dom_counts}
                for domtype, target in dom_counts.items():
                    if sum(isinstance(d, domtype) for d in prot.domains) == target:
                        found_doms[domtype] = True
                if all(found_doms.values()):
                    found_prots[i] = True
        if all(found_prots):
            return idxs[0]
    raise ValueError("unreachable")


def record_concentrations(
    record: dict, records: list, world: ms.World, cells_df: pd.DataFrame
):
    for _, row in cells_df.iterrows():
        for mol, mi in world.chemistry.mol_2_idx.items():
            records.append(
                {
                    **record,
                    "cell": int(row["cell"]),
                    "location": "int",
                    "molecule": mol.name,
                    "[x]": world.cell_molecules[int(row["cell"]), mi].item(),
                }
            )
            records.append(
                {
                    **record,
                    "cell": int(row["cell"]),
                    "location": "ext",
                    "molecule": mol.name,
                    "[x]": world.molecule_map[mi, int(row["x"]), int(row["y"])].item(),
                }
            )
