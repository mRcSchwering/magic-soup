import pytest
import torch
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import MOLECULES


def test_diffuse():
    # fmt: off
    layer0 = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    layer1 = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    exp0 = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    exp1 = [
        [0.2, 0.1, 0.0, 0.0, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.2],
        [0.0, 0.0, 0.2, 0.3, 0.2],
        [0.0, 0.0, 0.2, 0.3, 0.2],
        [0.1, 0.1, 0.1, 0.1, 0.2]
    ]
    # fmt: on

    m0 = ms.Molecule("m0", 10, diffusivity=0.0)
    m1 = ms.Molecule("m1", 10, diffusivity=0.5)

    chemistry = ms.Chemistry(molecules=[m0, m1], reactions=[])
    world = ms.World(chemistry=chemistry, map_size=5)
    world.molecule_map = torch.tensor([layer0, layer1])

    world.diffuse_molecules()

    assert world.molecule_map.shape == (2, 5, 5)
    assert ((world.molecule_map[0] - torch.tensor(exp0)).abs() < 1e-7).all()
    assert ((world.molecule_map[1] - torch.tensor(exp1)).abs() < 1e-7).all()


def test_degrade():
    # fmt: off
    layer0 = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    # fmt: on

    chemistry = ms.Chemistry(molecules=MOLECULES[:2], reactions=[])
    world = ms.World(chemistry=chemistry, map_size=5)

    world._mol_degrads = [0.8, 0.5]
    world.molecule_map[0] = torch.tensor([layer0])
    world.molecule_map[1] = torch.tensor([layer0])

    world.degrade_molecules()
    assert world.molecule_map.shape == (2, 5, 5)

    layer0[2][2] = 0.8
    assert (world.molecule_map[0] == torch.tensor(layer0)).all()
    layer0[2][2] = 0.5
    assert (world.molecule_map[1] == torch.tensor(layer0)).all()


def test_add_cells():
    chemistry = ms.Chemistry(molecules=MOLECULES[:2], reactions=[])
    world = ms.World(chemistry=chemistry, map_size=5)
    old_molmap = world.molecule_map.clone()

    genomes = [ms.random_genome(s=500) for _ in range(3)]
    world.add_random_cells(genomes=genomes)

    xs = []
    ys = []
    for cell in world.cells:
        x, y = cell.position
        xs.append(x)
        ys.append(y)

    assert torch.all(old_molmap[:, xs, ys] / 2 == world.molecule_map[:, xs, ys])
    assert torch.all(old_molmap[:, xs, ys] / 2 == world.cell_molecules.T)


def test_replicate_cells():
    chemistry = ms.Chemistry(molecules=MOLECULES[:2], reactions=[])
    world = ms.World(chemistry=chemistry, map_size=5)

    genomes = [ms.random_genome(s=500) for _ in range(3)]
    cell_idxs = world.add_random_cells(genomes=genomes)
    parent_child_idxs = world.replicate_cells(parent_idxs=cell_idxs)

    parents, children = list(map(list, zip(*parent_child_idxs)))
    assert torch.all(world.cell_molecules[parents] == world.cell_molecules[children])


def test_molecule_amount_integrity_when_changing_cells():
    tolerance = 1e-1

    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=[])
    world = ms.World(chemistry=chemistry, map_size=128)
    exp = world.molecule_map.sum(dim=[1, 2]) + world.cell_molecules.sum(dim=0)

    genomes = [ms.random_genome(s=500) for _ in range(1000)]
    cell_idxs = world.add_random_cells(genomes=genomes)
    res0 = world.molecule_map.sum(dim=[1, 2]) + world.cell_molecules.sum(dim=0)
    assert torch.all(torch.abs(res0 - exp) < tolerance)

    replicated = world.replicate_cells(parent_idxs=cell_idxs)
    res1 = world.molecule_map.sum(dim=[1, 2]) + world.cell_molecules.sum(dim=0)
    assert torch.all(torch.abs(res1 - exp) < tolerance)

    world.kill_cells(cell_idxs=cell_idxs + [d[1] for d in replicated])
    res2 = world.molecule_map.sum(dim=[1, 2]) + world.cell_molecules.sum(dim=0)
    assert torch.all(torch.abs(res2 - exp) < tolerance)


def test_cell_index_integrity_when_changing_cells():
    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=[])
    world = ms.World(chemistry=chemistry, map_size=128)

    n = len(world.cells)
    assert n == 0
    assert world.cell_map.sum().item() == n

    # there can be unviable genomes
    # with with s=1000 almost all genomes should be viable
    genomes = [ms.random_genome(s=1000) for _ in range(1000)]
    cell_idxs = world.add_random_cells(genomes=genomes)
    n = len(world.cells)
    assert n > 900
    assert len(cell_idxs) == n
    assert set(cell_idxs) == set(d.idx for d in world.cells)
    assert set(d.idx for d in world.cells) == set(range(len(cell_idxs)))
    assert world.cell_map.sum().item() == n

    replicated = world.replicate_cells(parent_idxs=cell_idxs)
    n = len(world.cells)
    new_idxs = [d[1] for d in replicated]
    assert len(replicated) > 1
    assert n == len(replicated) + len(cell_idxs)
    assert set(d.idx for d in world.cells) == set(cell_idxs) | set(new_idxs)
    assert set(d.idx for d in world.cells) == set(range(n))
    assert world.cell_map.sum().item() == n

    world.kill_cells(cell_idxs=cell_idxs)
    n = len(world.cells)
    assert n == len(replicated)
    assert set(d.idx for d in world.cells) == set(range(n))
    assert world.cell_map.sum().item() == n

    world.kill_cells(cell_idxs=list(range(n)))
    n = len(world.cells)
    assert n == 0
    assert world.cell_map.sum().item() == n


def test_molecule_amount_integrity_during_diffusion():
    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=[])
    world = ms.World(chemistry=chemistry, map_size=128)

    exp = world.molecule_map.sum(dim=[1, 2])
    for step_i in range(1000):
        world.diffuse_molecules()
        res = world.molecule_map.sum(dim=[1, 2])
        assert (res.sum() - exp.sum()).abs() < 10.0, step_i
        assert torch.all(torch.abs(res - exp) < 1.0), step_i


def test_molecule_amount_integrity_during_reactions():
    # X and Y can react back and forth but X + Y <-> Z
    # so if Z is counted as 2, n should stay equal
    mi = ms.Molecule("i", 10 * 1e3)
    mj = ms.Molecule("j", 20 * 1e3)
    mk = ms.Molecule("k", 30 * 1e3)
    molecules = [mi, mj, mk]
    reactions = [([mi], [mj]), ([mi, mj], [mk])]

    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry, map_size=128)
    genomes = [ms.random_genome(s=500) for _ in range(1000)]
    world.add_random_cells(genomes=genomes)

    def count(world: ms.World) -> float:
        mij = world.molecule_map[[0, 1]].sum().item()
        mk = world.molecule_map[2].sum().item() * 2
        cij = world.cell_molecules[:, [0, 1]].sum().item()
        ck = world.cell_molecules[:, 2].sum().item() * 2
        return mij + mk + cij + ck

    n0 = count(world)
    for step_i in range(1000):
        world.enzymatic_activity()
        n = count(world)
        assert n == pytest.approx(n0, abs=1.0), step_i


def test_generate_genome():
    mi = ms.Molecule("i", 10 * 1e3)
    mj = ms.Molecule("j", 20 * 1e3)
    mk = ms.Molecule("k", 30 * 1e3)
    molecules = [mi, mj, mk]
    reactions = [([mi], [mj]), ([mi, mj], [mk])]

    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry, map_size=128)

    g = world.generate_genome(proteome=[], size=10)
    assert len(g) == 10
    g = world.generate_genome(proteome=[], size=100)
    assert len(g) == 100

    p0 = ms.ProteinFact(
        domain_facts=[
            ms.CatalyticDomainFact(reaction=([mi], [mj])),
            ms.CatalyticDomainFact(reaction=([mk], [mi, mj])),
        ]
    )
    with pytest.raises(ValueError):
        g = world.generate_genome(proteome=[p0], size=10)

    g = world.generate_genome(proteome=[p0], size=50)
    assert len(g) == 50

    g = world.generate_genome(proteome=[p0], size=100)
    assert len(g) == 100

    p0 = ms.ProteinFact(
        domain_facts=[
            ms.CatalyticDomainFact(reaction=([mi], [mj])),
            ms.CatalyticDomainFact(reaction=([mk], [mi, mj])),
        ]
    )
    p1 = ms.ProteinFact(
        domain_facts=[
            ms.TransporterDomainFact(molecule=mi),
            ms.RegulatoryDomainFact(effector=mk, is_transmembrane=True),
        ]
    )

    # codon-to-indices mappings (1-codon/2-codon maps) can also map to
    # stop codons, so there's a 4.6% chance per such codon that a domain
    # is pre-maturely terminated by a stop codon. With 5 such domains
    # in every domain, there's only a 78% chance that everything goes right
    # with 4 domains in the proteome, theres only a 38% chance of succeeding
    # (having 0 pre-mature terminations)
    # repeating the genome generation 7 times, reduces the chance of 7 pre-mature
    # terminations to below 5% (>95% chance that at least one proteome is correct)
    success = False
    max_i = 7
    i = 0
    p0_found = 0
    p1_found = 0
    while not success:
        g = world.generate_genome(proteome=[p0, p1], size=100)
        assert len(g) == 100

        world.add_random_cells(genomes=[g])
        cell = world.get_cell(by_idx=0)
        has_p0 = False
        has_p1 = False
        for prot in cell.proteome:
            print(prot)
            has_cij = False
            has_ckij = False
            has_ti = False
            has_rk = False

            for dom in prot.domains:
                if isinstance(dom, ms.CatalyticDomain):
                    subs = dom.substrates
                    prods = dom.products
                    if subs == [mk] and prods == [mi, mj]:
                        has_ckij = True
                    elif subs == [mi] and prods == [mj]:
                        has_cij = True
                if isinstance(dom, ms.TransporterDomain):
                    if dom.molecule == mi:
                        has_ti = True
                if isinstance(dom, ms.RegulatoryDomain):
                    if dom.effector == mk and dom.is_inhibiting:
                        has_rk = True

            if has_ckij and has_cij:
                has_p0 = True
                p0_found += 1
            if has_ti and has_rk:
                has_p1 = True
                p1_found += 1

        world.kill_cells(cell_idxs=[d.idx for d in world.cells])
        if has_p0 and has_p1:
            success = True
            break
        else:
            i += 1

        if i > max_i:
            raise AssertionError(
                f"Was not able to recreate proteome from generated genome after {max_i} tries."
                f" P0 was found {p0_found} times, P1 was found {p1_found} times."
            )
