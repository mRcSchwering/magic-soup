import torch
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import MOLECULES


TOLERANCE = 1e-4

DOMAIN_FACT = {
    ms.CatalyticFact(reactions=[([MOLECULES[0]], [MOLECULES[1]])]): ["AAAAAA"]
}


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
        [0.0, 0.1, 0.1, 0.1, 0.0],
        [0.0, 0.1, 0.2, 0.1, 0.0],
        [0.0, 0.1, 0.1, 0.1, 0.0],
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

    world = ms.World(
        domain_facts=DOMAIN_FACT,
        molecules=MOLECULES[:2],
        map_size=5,
        mol_diff_coef=0.5 / 1e6,
    )
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

    world = ms.World(
        domain_facts=DOMAIN_FACT, molecules=MOLECULES[:2], map_size=5, mol_halflife=1.0
    )
    world.mol_degrad = 0.8
    world.molecule_map[0] = torch.tensor([layer0])

    world.degrade_molecules()

    layer0[2][2] = 0.8
    assert world.molecule_map.shape == (2, 5, 5)
    assert (world.molecule_map[0] == torch.tensor(layer0)).all()


def test_add_cells():
    world = ms.World(
        domain_facts=DOMAIN_FACT, molecules=MOLECULES[:2], map_size=5, mol_halflife=1.0
    )
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
    world = ms.World(
        domain_facts=DOMAIN_FACT, molecules=MOLECULES[:2], map_size=5, mol_halflife=1.0
    )

    cell_idxs = world.add_random_cells(genomes=["A" * 50] * 3)
    parent_idxs, child_idxs = world.replicate_cells(parent_idxs=cell_idxs)

    assert torch.all(
        world.cell_molecules[parent_idxs] == world.cell_molecules[child_idxs]
    )
