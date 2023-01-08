import torch
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import MOLECULES, co2, formiat, NADPH, NADP


TOLERANCE = 1e-4


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

    m0 = ms.Molecule("m0", 10, diff_coef=0.0)
    m1 = ms.Molecule("m1", 10, diff_coef=0.5 / 1e6)

    chemistry = ms.Chemistry(molecules=[m0, m1], reactions=[])
    world = ms.World(chemistry=chemistry, map_size=5)
    world.molecule_map = torch.tensor([layer0, layer1])

    world.diffuse_molecules()

    assert world.molecule_map.shape == (2, 5, 5)
    assert (world.molecule_map[0] == torch.tensor(exp0)).all()
    assert (world.molecule_map[1] == torch.tensor(exp1)).all()


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

    world.add_random_cells(genomes=["A" * 50] * 3)

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

    cell_idxs = world.add_random_cells(genomes=["A" * 50] * 3)
    parent_child_idxs = world.replicate_cells(parent_idxs=cell_idxs)

    parents, children = list(map(list, zip(*parent_child_idxs)))
    assert torch.all(world.cell_molecules[parents] == world.cell_molecules[children])


def test_molecule_amount_integrity_when_changing_cells():
    tolerance = 1e-2

    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=[])
    world = ms.World(chemistry=chemistry, map_size=128)
    exp = world.molecule_map.sum(dim=[1, 2]) + world.cell_molecules.sum(dim=0)

    cell_idxs = world.add_random_cells(genomes=["A" * 50] * 1000)
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

    cell_idxs = world.add_random_cells(genomes=["A" * 50] * 1000)
    n = len(world.cells)
    assert len(cell_idxs) == 1000
    assert len(world.cells) == 1000
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
    tolerance = 1e-2

    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=[])
    world = ms.World(chemistry=chemistry, map_size=128)
    exp = world.molecule_map.sum(dim=[1, 2])

    for _ in range(10):
        world.diffuse_molecules()
        res = world.molecule_map.sum(dim=[1, 2])
        assert torch.all(torch.abs(res - exp) < tolerance)


def test_molecule_amount_integrity_during_reactions():
    tolerance = 1e-2

    # 2 molecules react to 2 other molecules, total count should be constant
    reactions = [([co2, NADPH], [formiat, NADP])]
    molecules = [co2, NADPH, NADP, formiat]

    chemistry = ms.Chemistry(molecules=molecules, reactions=reactions)
    world = ms.World(chemistry=chemistry, map_size=128)
    genomes = [ms.random_genome(s=500) for _ in range(1)]
    world.add_random_cells(genomes=genomes)
    exp = world.molecule_map.sum() + world.cell_molecules.sum()

    for _ in range(10):
        world.enzymatic_activity()
        res = world.molecule_map.sum() + world.cell_molecules.sum()
        assert torch.all(torch.abs(res - exp) < tolerance)

