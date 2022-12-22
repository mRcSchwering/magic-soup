import torch
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import MOLECULES


TOLERANCE = 1e-4

DOMAIN_FACT = {
    ms.CatalyticFact(
        {"AAA": ([MOLECULES[0]], [MOLECULES[1]])},
        {"AAA": 1.0},
        {"AAA": 1.0},
        {"AAA": True},
    ): ["AAAAAA"]
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

    genetics = ms.Genetics(domain_facts=DOMAIN_FACT, molecules=MOLECULES[:2])
    world = ms.World(genetics=genetics, map_size=5, mol_diff_coef=0.5 / 1e6)
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

    genetics = ms.Genetics(domain_facts=DOMAIN_FACT, molecules=MOLECULES[:2])
    world = ms.World(genetics=genetics, map_size=5, mol_halflife=1.0)
    world.mol_degrad = 0.8
    world.molecule_map[0] = torch.tensor([layer0])

    world.degrade_molecules()

    layer0[2][2] = 0.8
    assert world.molecule_map.shape == (2, 5, 5)
    assert (world.molecule_map[0] == torch.tensor(layer0)).all()
