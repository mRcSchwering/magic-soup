import time
import torch
from world import World


TOLERANCE = 1e-4


def test_performance():
    world = World(size=128, n_molecules=4, mol_map_init="randn")

    t0 = time.time()
    for _ in range(100):
        world.diffuse_molecules()
    td = time.time() - t0

    assert td < 0.2, "Used to take 0.127"


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

    world = World(size=5, n_molecules=2, mol_diff_rate=0.5)
    world.molecule_map = torch.tensor([[layer0], [layer1]])

    world.diffuse_molecules()

    assert world.molecule_map.shape == (2, 1, 5, 5)
    assert (world.molecule_map[0, 0] == torch.tensor(exp0)).all()
    assert (world.molecule_map[1, 0] == torch.tensor(exp1)).all()


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

    world = World(size=5, n_molecules=2, mol_degrad=0.8)
    world.molecule_map[0, 0] = torch.tensor([layer0])

    world.degrade_molecules()

    layer0[2][2] = 0.8
    assert world.molecule_map.shape == (2, 1, 5, 5)
    assert (world.molecule_map[0, 0] == torch.tensor(layer0)).all()

