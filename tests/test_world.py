import tempfile
from pathlib import Path
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


def test_spawn_cells():
    chemistry = ms.Chemistry(molecules=MOLECULES[:2], reactions=[])
    world = ms.World(chemistry=chemistry, map_size=5)
    old_molmap = world.molecule_map.clone()

    genomes = [ms.random_genome(s=500) for _ in range(3)]
    world.spawn_cells(genomes=genomes)

    assert world.n_cells == 3
    xs = world.cell_positions[:, 0]
    ys = world.cell_positions[:, 1]
    assert torch.all(old_molmap[:, xs, ys] / 2 == world.molecule_map[:, xs, ys])
    assert torch.all(old_molmap[:, xs, ys] / 2 == world.cell_molecules.T)

    genomes = ["" for _ in range(2)]
    world.spawn_cells(genomes=genomes)

    assert world.n_cells == 5
    xs = world.cell_positions[:, 0]
    ys = world.cell_positions[:, 1]
    assert torch.all(old_molmap[:, xs, ys] / 2 == world.molecule_map[:, xs, ys])
    assert torch.all(old_molmap[:, xs, ys] / 2 == world.cell_molecules.T)

    genomes = [ms.random_genome(100), "", ms.random_genome(100)]
    world.spawn_cells(genomes=genomes)

    assert world.n_cells == 8
    xs = world.cell_positions[:, 0]
    ys = world.cell_positions[:, 1]
    assert torch.all(old_molmap[:, xs, ys] / 2 == world.molecule_map[:, xs, ys])
    assert torch.all(old_molmap[:, xs, ys] / 2 == world.cell_molecules.T)


def test_add_cells():
    chemistry = ms.Chemistry(molecules=MOLECULES[:2], reactions=[])
    world = ms.World(chemistry=chemistry, map_size=5)
    old_molmap = world.molecule_map.clone()

    genomes = [ms.random_genome(s=d) for d in [500, 300, 100]]
    cells = [
        ms.Cell(
            label="C0",
            genome=genomes[0],
            int_molecules=torch.tensor([0.5, 0.1]),
            ext_molecules=torch.tensor([1.0, 1.0]),
        ),
        ms.Cell(
            label="C1",
            genome=genomes[1],
            int_molecules=torch.tensor([0.3, 0.4]),
            ext_molecules=torch.tensor([1.0, 1.0]),
        ),
        ms.Cell(
            label="C2",
            genome=genomes[2],
            int_molecules=torch.tensor([0.9, 0.7]),
            ext_molecules=torch.tensor([1.0, 1.0]),
            n_steps_alive=3,
            n_divisions=1,
        ),
    ]

    world.add_cells(cells=cells[:2])

    assert world.n_cells == 2
    xs = world.cell_positions[:, 0]
    ys = world.cell_positions[:, 1]
    assert torch.all(old_molmap[:, xs, ys] == world.molecule_map[:, xs, ys])
    assert (world.cell_molecules[0] == torch.tensor([0.5, 0.1])).all()
    assert (world.cell_molecules[1] == torch.tensor([0.3, 0.4])).all()
    assert (world.cell_divisions == torch.tensor([0, 0])).all()
    assert (world.cell_lifetimes == torch.tensor([0, 0])).all()
    assert world.cell_genomes == genomes[:2]
    assert world.cell_labels == ["C0", "C1"]

    world.add_cells(cells=cells[2:])

    assert world.n_cells == 3
    xs = world.cell_positions[:, 0]
    ys = world.cell_positions[:, 1]
    assert torch.all(old_molmap[:, xs, ys] == world.molecule_map[:, xs, ys])
    assert (world.cell_molecules[0] == torch.tensor([0.5, 0.1])).all()
    assert (world.cell_molecules[1] == torch.tensor([0.3, 0.4])).all()
    assert (world.cell_molecules[2] == torch.tensor([0.9, 0.7])).all()
    assert (world.cell_divisions == torch.tensor([0, 0, 1])).all()
    assert (world.cell_lifetimes == torch.tensor([0, 0, 3])).all()
    assert world.cell_genomes == genomes
    assert world.cell_labels == ["C0", "C1", "C2"]


def test_divide_cells():
    chemistry = ms.Chemistry(molecules=MOLECULES[:2], reactions=[])
    world = ms.World(chemistry=chemistry, map_size=5)

    genomes = [ms.random_genome(s=500) for _ in range(3)]
    cell_idxs = world.spawn_cells(genomes=genomes)
    parent_child_idxs = world.divide_cells(cell_idxs=cell_idxs)

    parents, children = list(map(list, zip(*parent_child_idxs)))
    assert torch.all(world.cell_molecules[parents] == world.cell_molecules[children])


def test_molecule_amount_integrity_when_changing_cells():
    tolerance = 1e-1

    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=[])
    world = ms.World(chemistry=chemistry, map_size=128)
    exp = world.molecule_map.sum(dim=[1, 2]) + world.cell_molecules.sum(dim=0)

    genomes = [ms.random_genome(s=500) for _ in range(1000)]
    cell_idxs = world.spawn_cells(genomes=genomes)
    res0 = world.molecule_map.sum(dim=[1, 2]) + world.cell_molecules.sum(dim=0)
    assert torch.all(torch.abs(res0 - exp) < tolerance)

    replicated = world.divide_cells(cell_idxs=cell_idxs)
    res1 = world.molecule_map.sum(dim=[1, 2]) + world.cell_molecules.sum(dim=0)
    assert torch.all(torch.abs(res1 - exp) < tolerance)

    world.kill_cells(cell_idxs=cell_idxs + [d[1] for d in replicated])
    res2 = world.molecule_map.sum(dim=[1, 2]) + world.cell_molecules.sum(dim=0)
    assert torch.all(torch.abs(res2 - exp) < tolerance)


def test_cell_index_integrity_when_changing_cells():
    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=[])
    world = ms.World(chemistry=chemistry, map_size=128)

    n = world.n_cells
    assert n == 0
    assert world.cell_map.sum().item() == n

    # there can be unviable genomes
    # with with s=1000 almost all genomes should be viable
    genomes = [ms.random_genome(s=1000) for _ in range(1000)]
    cell_idxs = world.spawn_cells(genomes=genomes)
    n0 = world.n_cells
    assert n0 > 900
    assert len(cell_idxs) == n0
    assert len(world.cell_genomes) == n0
    assert len(world.cell_labels) == n0
    assert len(set(world.cell_labels)) == n0
    assert world.cell_map.sum().item() == n0

    replicated = world.divide_cells(cell_idxs=cell_idxs)
    n1 = world.n_cells
    assert len(replicated) > 1
    assert n1 == len(replicated) + len(cell_idxs)
    assert len(world.cell_genomes) == n1
    assert len(world.cell_labels) == n1
    assert len(set(world.cell_labels)) == n0
    assert world.cell_map.sum().item() == n1

    parents = [d[0] for d in replicated]
    children = [d[1] for d in replicated]
    assert set(parents) <= set(cell_idxs)
    assert set(children) & set(cell_idxs) == set()

    world.kill_cells(cell_idxs=cell_idxs)
    n2 = world.n_cells
    assert n2 == len(replicated)
    assert len(world.cell_genomes) == n2
    assert len(world.cell_labels) == n2
    assert world.cell_map.sum().item() == n2

    world.kill_cells(cell_idxs=list(range(n2)))
    n3 = world.n_cells
    assert n3 == 0
    assert world.cell_map.sum().item() == n3


def test_cells_unable_to_divide():
    chemistry = ms.Chemistry(molecules=MOLECULES[:2], reactions=[])
    world = ms.World(chemistry=chemistry, map_size=3)

    genomes = [ms.random_genome(s=500) for _ in range(9)]
    world.spawn_cells(genomes=genomes)

    descendants = world.divide_cells(cell_idxs=list(range(9)))
    assert len(descendants) == 0


def test_get_cell_by_position():
    chemistry = ms.Chemistry(molecules=MOLECULES[:2], reactions=[])
    world = ms.World(chemistry=chemistry, map_size=5)

    genomes = [ms.random_genome(s=500) for _ in range(3)]
    world.spawn_cells(genomes=genomes)

    pos = tuple(world.cell_positions[1].tolist())
    cell = world.get_cell(by_position=pos)  # type: ignore

    assert cell.idx == 1
    assert cell.position == pos


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
            ms.RegulatoryDomainFact(effector=mk, is_transmembrane=True, hill=3),
        ]
    )

    # codon-to-indices mappings (1-codon/2-codon maps) can also map to
    # stop codons, so there's a 4.6% chance per such codon that a domain
    # is pre-maturely terminated by a stop codon. With 5 such domains
    # in every domain, there's only a 78% chance that everything goes right
    # with 4 domains in the proteome, theres only a 38% chance of succeeding
    # (having 0 pre-mature terminations)
    # repeating the genome generation 10 times, reduces the chance of 10 pre-mature
    # terminations to below 1% (>95% chance that at least one proteome is correct)
    success = False
    max_i = 10
    i = 0
    p0_found = 0
    p1_found = 0
    while not success:
        g = world.generate_genome(proteome=[p0, p1], size=100)
        assert len(g) == 100

        world.spawn_cells(genomes=[g])
        cell = world.get_cell(by_idx=0)
        has_p0 = False
        has_p1 = False
        proteome = cell.get_proteome(world=world)
        for prot in proteome:
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
                    if dom.effector == mk and dom.is_inhibiting and dom.hill == 3:
                        has_rk = True

            if has_ckij and has_cij:
                has_p0 = True
                p0_found += 1
            if has_ti and has_rk:
                has_p1 = True
                p1_found += 1

        world.kill_cells(cell_idxs=list(range(world.n_cells)))
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


def test_generate_genome_with_different_reaction_sorting():
    mi = ms.Molecule("i", 10 * 1e3)
    mj = ms.Molecule("j", 20 * 1e3)
    mk = ms.Molecule("k", 30 * 1e3)
    molecules = [mi, mj, mk]
    reactions = [([mj, mi], [mk])]

    # reaction gets sorted to i + j <-> k
    # when initializing chemistry object
    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry, map_size=128)

    # if reaction is not properly reordered
    # it will fail
    doms = [ms.CatalyticDomainFact(reaction=reactions[0])]
    g = world.generate_genome(proteome=[ms.ProteinFact(domain_facts=doms)], size=100)
    assert len(g) == 100

    # in contrast this should fail,
    # because the reaction wasnt defined
    doms = [ms.CatalyticDomainFact(reaction=([mj], [mk]))]
    with pytest.raises(ValueError):
        world.generate_genome(proteome=[ms.ProteinFact(domain_facts=doms)], size=100)


def test_saving_and_loading_world_obj():
    mi = ms.Molecule("i", 10 * 1e3)
    mj = ms.Molecule("j", 20 * 1e3)
    chemistry = ms.Chemistry(molecules=[mi, mj], reactions=[([mi], [mj])])
    world = ms.World(chemistry=chemistry, map_size=7)

    for i, mol in enumerate(chemistry.molecules):
        assert world.chemistry.mol_2_idx[mol] == i
        assert world.chemistry.molname_2_idx[mol.name] == i

    with tempfile.TemporaryDirectory() as tmpdir:
        rundir = Path(tmpdir)
        world.save(rundir=rundir)
        del world
        world = ms.World.from_file(rundir=rundir)

    assert world.abs_temp == 310.0
    assert world.map_size == 7
    assert world.workers == 2
    assert world.genetics.workers == 2
    assert world.device == "cpu"
    assert world.chemistry.molecules[0] is mi
    assert world.chemistry.molecules[1] is mj
    assert world.chemistry.reactions == [([mi], [mj])]

    for i, mol in enumerate(chemistry.molecules):
        assert world.chemistry.mol_2_idx[mol] == i
        assert world.chemistry.molname_2_idx[mol.name] == i

    del world

    world = ms.World(chemistry=chemistry, map_size=9, workers=2, abs_temp=300.0)
    with tempfile.TemporaryDirectory() as tmpdir:
        rundir = Path(tmpdir)
        world.save(rundir=rundir)
        del world
        world = ms.World.from_file(rundir=rundir, workers=4)

    assert world.abs_temp == 300.0
    assert world.map_size == 9
    assert world.workers == 4
    assert world.genetics.workers == 4
    assert world.device == "cpu"
    assert world.chemistry.molecules[0] is mi
    assert world.chemistry.molecules[1] is mj
    assert world.chemistry.reactions == [([mi], [mj])]

    for i, mol in enumerate(chemistry.molecules):
        assert world.chemistry.mol_2_idx[mol] == i
        assert world.chemistry.molname_2_idx[mol.name] == i

    del world


def test_saving_and_loading_state():
    mi = ms.Molecule("i", 10 * 1e3)
    mj = ms.Molecule("j", 20 * 1e3)
    chemistry = ms.Chemistry(molecules=[mi, mj], reactions=[([mi], [mj])])

    world = ms.World(chemistry=chemistry, map_size=7)
    world.spawn_cells(genomes=[ms.random_genome(s=500) for _ in range(3)])

    cell_map = world.cell_map.clone()
    molecule_map = world.molecule_map.clone()
    assert cell_map.sum() == 3.0
    assert molecule_map.mean() > 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        statedir = Path(tmpdir)
        world.save_state(statedir=statedir)

        del world
        world = ms.World(chemistry=chemistry, map_size=7)
        world.load_state(statedir=statedir)

    assert (cell_map == world.cell_map).all()
    assert (molecule_map == world.molecule_map).all()


def test_loading_multiple_states():
    mi = ms.Molecule("i", 10 * 1e3)
    mj = ms.Molecule("j", 20 * 1e3)
    chemistry = ms.Chemistry(molecules=[mi, mj], reactions=[([mi], [mj])])

    with tempfile.TemporaryDirectory() as tmpdir:
        world = ms.World(chemistry=chemistry, map_size=7)

        state0 = Path(tmpdir) / "state0"
        world.spawn_cells(genomes=[ms.random_genome(s=500) for _ in range(3)])
        world.save_state(statedir=state0)
        assert world.n_cells == 3
        assert len(world.cell_genomes) == 3
        assert world.kinetics.N.size(0) == 3

        state1 = Path(tmpdir) / "state1"
        world.spawn_cells(genomes=[ms.random_genome(s=500) for _ in range(3)])
        world.save_state(statedir=state1)
        assert world.n_cells == 6
        assert len(world.cell_genomes) == 6
        assert world.kinetics.N.size(0) == 6

        state2 = Path(tmpdir) / "state2"
        world.kill_cells(cell_idxs=list(range(4)))
        world.save_state(statedir=state2)
        assert world.n_cells == 2
        assert len(world.cell_genomes) == 2
        assert world.kinetics.N.size(0) == 2

        del world
        world = ms.World(chemistry=chemistry, map_size=7)

        world.load_state(statedir=state0)
        assert world.n_cells == 3
        assert len(world.cell_genomes) == 3
        assert world.kinetics.N.size(0) == 3

        world.load_state(statedir=state1)
        assert world.n_cells == 6
        assert len(world.cell_genomes) == 6
        assert world.kinetics.N.size(0) == 6

        world.load_state(statedir=state2)
        assert world.n_cells == 2
        assert len(world.cell_genomes) == 2
        assert world.kinetics.N.size(0) == 2


def test_divisions_and_survival_after_replication():
    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=[])

    world = ms.World(chemistry=chemistry)
    world.spawn_cells(genomes=[ms.random_genome(s=500) for _ in range(2)])
    assert world.n_cells == 2
    world.cell_divisions[0] = 1
    world.cell_divisions[1] = 11
    world.cell_lifetimes[0] = 5
    world.cell_lifetimes[1] = 15

    world.divide_cells(cell_idxs=[0])
    assert world.n_cells == 3
    assert world.cell_divisions[0] == 2
    assert world.cell_divisions[1] == 11
    assert world.cell_divisions[2] == 2
    assert world.cell_lifetimes[0] == 0
    assert world.cell_lifetimes[1] == 15
    assert world.cell_lifetimes[2] == 0

    world.increment_cell_lifetimes()
    assert world.n_cells == 3
    assert world.cell_divisions[0] == 2
    assert world.cell_divisions[1] == 11
    assert world.cell_divisions[2] == 2
    assert world.cell_lifetimes[0] == 1
    assert world.cell_lifetimes[1] == 16
    assert world.cell_lifetimes[2] == 1


def test_reference_to_tensors_not_lost():
    class A:
        def __init__(self, world: ms.World):
            self.world = world

        def f(self, d: float):
            self.world.molecule_map = torch.full_like(self.world.molecule_map, d)
            self.world.cell_molecules = torch.full_like(self.world.cell_molecules, d)

    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=[])
    world = ms.World(chemistry=chemistry, map_size=3, mol_map_init="zeros")
    world.spawn_cells(genomes=[ms.random_genome(s=500) for _ in range(2)])
    a = A(world=world)

    assert id(world.molecule_map) == id(a.world.molecule_map)
    assert id(world.cell_molecules) == id(a.world.cell_molecules)
    assert id(world.cell_map) == id(a.world.cell_map)
    assert world.molecule_map.mean() == 0.0

    world.diffuse_molecules()
    world.enzymatic_activity()
    world.degrade_molecules()

    assert id(world.molecule_map) == id(a.world.molecule_map)
    assert id(world.cell_molecules) == id(a.world.cell_molecules)
    assert id(world.cell_map) == id(a.world.cell_map)

    a.f(2.0)
    assert id(world.molecule_map) == id(a.world.molecule_map)
    assert id(world.cell_molecules) == id(a.world.cell_molecules)
    assert id(world.cell_map) == id(a.world.cell_map)
    assert world.molecule_map.mean() == 2.0

    world.diffuse_molecules()
    world.enzymatic_activity()
    world.degrade_molecules()

    assert id(world.molecule_map) == id(a.world.molecule_map)
    assert id(world.cell_molecules) == id(a.world.cell_molecules)
    assert id(world.cell_map) == id(a.world.cell_map)


def test_reposition_cells():
    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=[])
    world = ms.World(chemistry=chemistry)
    world.spawn_cells(genomes=[ms.random_genome(s=500) for _ in range(3)])

    c0_0 = world.get_cell(by_idx=0)
    c1_0 = world.get_cell(by_idx=1)
    c2_0 = world.get_cell(by_idx=2)

    world.reposition_cells(cell_idxs=[0, 2])
    c0_1 = world.get_cell(by_idx=0)
    c1_1 = world.get_cell(by_idx=1)
    c2_1 = world.get_cell(by_idx=2)

    assert c0_1.position != c0_0.position
    assert c1_1.position == c1_0.position
    assert c2_1.position != c2_0.position
    assert (c0_1.int_molecules == c0_0.int_molecules).all()
    assert (c1_1.int_molecules == c1_0.int_molecules).all()
    assert (c2_1.int_molecules == c2_0.int_molecules).all()
    assert c0_1.genome == c0_0.genome
    assert c1_1.genome == c1_0.genome
    assert c2_1.genome == c2_0.genome
    assert c0_1.label == c0_0.label
    assert c1_1.label == c1_0.label
    assert c2_1.label == c2_0.label


def test_change_genomes():
    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=[])
    world = ms.World(chemistry=chemistry)

    g0 = ms.random_genome(s=500)
    world.spawn_cells(genomes=[g0] * 2)
    assert world.cell_genomes == [g0] * 2
    assert (world.kinetics.N[0] == world.kinetics.N[1]).all()

    g1 = ms.random_genome(s=1000)
    world.spawn_cells(genomes=[g1])
    assert world.cell_genomes == [g0] * 2 + [g1]
    assert (world.kinetics.N[0] == world.kinetics.N[1]).all()

    g2 = ms.random_genome(s=600)
    world.update_cells(genome_idx_pairs=[(g2, 1)])
    assert world.cell_genomes == [g0, g2, g1]

    world.kill_cells(cell_idxs=[0])
    assert world.cell_genomes == [g2, g1]

    g3 = ms.random_genome(s=700)
    world.update_cells(genome_idx_pairs=[(g3, 1)])
    assert world.cell_genomes == [g2, g3]

    g4 = ms.random_genome(s=500)
    world.spawn_cells(genomes=[g4])
    assert world.cell_genomes == [g2, g3, g4]

    g5 = ""
    world.update_cells(genome_idx_pairs=[(g4, 0), (g5, 1), (g4, 2)])
    assert world.cell_genomes == [g4, g5, g4]
    assert (world.kinetics.N[0] == world.kinetics.N[2]).all()
    assert (world.kinetics.N[1] == 0.0).all()

    world.update_cells(genome_idx_pairs=[(g5, 0), (g5, 1), (g5, 2)])
    assert world.cell_genomes == [g5, g5, g5]
    assert (world.kinetics.N[0] == 0.0).all()
    assert (world.kinetics.N[1] == 0.0).all()
    assert (world.kinetics.N[2] == 0.0).all()


def test_get_neighbours():
    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=[])
    world = ms.World(chemistry=chemistry, map_size=9)

    # fmt: off
    world.cell_map = torch.tensor([
        [0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1]
    ]).bool()
    
    world.cell_positions = torch.tensor([
        [1, 1],  # 0
        [1, 2],  # 1
        [0, 1],  # 2
        [2, 1],  # 3
        [8, 8],  # 4
        [0, 8],  # 5
        [2, 0],  # 6
        [8, 6],  # 7
        [8, 7],  # 8
    ])

    # always sorted lower idx first
    nghbrs = [
        {(0, 2), (0, 1), (0, 6), (0, 3)},  # 0
        {(1, 2), (0, 1), (1, 3)},  # 1
        {(0, 2), (1, 2)},  # 2
        {(0, 3), (1, 3), (3, 6)},  # 3
        {(4, 8), (4, 5)},  # 4
        {(5, 8), (4, 5)},  # 5
        {(0, 6), (3, 6)},  # 6
        {(7, 8)},  # 7
        {(7, 8), (4, 8), (5, 8)},  # 8
    ]
    # fmt: on

    for idx, exp in enumerate(nghbrs):
        res = world.get_neighbors(cell_idxs=[idx])
        assert len(res) == len(exp), idx
        assert set(res) == exp, idx

        # should effectively be the same
        res = world.get_neighbors(cell_idxs=[idx], nghbr_idxs=list(range(len(nghbrs))))
        assert len(res) == len(exp), idx
        assert set(res) == exp, idx

    res = world.get_neighbors(cell_idxs=[0, 1, 2])
    exp = nghbrs[0] | nghbrs[1] | nghbrs[2]
    assert len(res) == len(exp)
    assert set(res) == exp

    res = world.get_neighbors(cell_idxs=[1, 4, 8])
    exp = nghbrs[1] | nghbrs[4] | nghbrs[8]
    assert len(res) == len(exp)
    assert set(res) == exp

    res = world.get_neighbors(cell_idxs=[1, 4, 8])
    exp = nghbrs[1] | nghbrs[4] | nghbrs[8]
    assert len(res) == len(exp)
    assert set(res) == exp

    nghbr_idxs = [2, 3]
    res = world.get_neighbors(cell_idxs=[0], nghbr_idxs=nghbr_idxs)
    exp = nghbrs[0]
    exp = set(d for d in exp if d[0] in nghbr_idxs or d[1] in nghbr_idxs)
    assert len(res) == len(exp)
    assert set(res) == exp

    nghbr_idxs = [6, 2]
    res = world.get_neighbors(cell_idxs=[0, 3], nghbr_idxs=nghbr_idxs)
    exp = nghbrs[0] | nghbrs[3]
    exp = set(d for d in exp if d[0] in nghbr_idxs or d[1] in nghbr_idxs)
    assert len(res) == len(exp)
    assert set(res) == exp


def test_empty_proteome_and_genome_cells():
    genomes = [ms.random_genome(10), "", ms.random_genome(10)]

    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=[])
    world = ms.World(chemistry=chemistry, map_size=9)

    world.spawn_cells(genomes=genomes)
    assert world.n_cells == 3

    # get cells with empty proteomes
    assert len(world.get_cell(by_idx=0).get_proteome(world=world)) == 0
    assert len(world.get_cell(by_idx=1).get_proteome(world=world)) == 0
    assert len(world.get_cell(by_idx=2).get_proteome(world=world)) == 0

    # load cells with empty fastAs
    with tempfile.TemporaryDirectory() as tmpdir:
        statedir = Path(tmpdir)
        world.save_state(statedir=statedir)

        del world
        world = ms.World(chemistry=chemistry, map_size=9)
        world.load_state(statedir=statedir)

    assert world.cell_genomes == genomes
