import time
import torch
from world import World


def test_performance():
    world = World(size=128, layers=4, map_init="randn")

    t0 = time.time()
    for _ in range(100):
        world.diffuse()
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
    kernel = [
        [0.1, 0.0, 0.1],
        [0.1, 0.4, 0.1],
        [0.1, 0.0, 0.1],
    ]
    exp0 = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.1, 0.0],
        [0.0, 0.1, 0.4, 0.1, 0.0],
        [0.0, 0.1, 0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    exp1 = [
        [0.4, 0.1, 0.0, 0.0, 0.1],
        [0.0, 0.1, 0.1, 0.0, 0.2],
        [0.0, 0.0, 0.2, 0.4, 0.2],
        [0.0, 0.0, 0.2, 0.4, 0.2],
        [0.0, 0.1, 0.1, 0.0, 0.2]
    ]
    # fmt: on

    world = World(size=5, layers=2, kernel=torch.tensor([[kernel]]))
    world.map = torch.tensor([[layer0], [layer1]])

    world.diffuse()

    assert world.map.shape == (2, 1, 5, 5)
    assert (world.map[0, 0] == torch.tensor(exp0)).all()
    assert (world.map[1, 0] == torch.tensor(exp1)).all()


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

    exp0 = layer0.copy()
    exp0[2][2] = 0.8

    world = World(size=5, layers=2, degrad=0.8)
    world.map = torch.tensor([[layer0]])

    world.degrade()

    assert world.map.shape == (1, 1, 5, 5)
    assert (world.map[0, 0] == torch.tensor(exp0)).all()

