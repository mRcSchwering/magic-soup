import time
import numpy as np
import torch
from util import rand_genome
from genetics import (
    assert_config,
    get_proteome,
    get_cell_params,
    simulate_protein_work,
    Signal,
)
from world import World


if __name__ == "__main__":
    assert_config()

    gs = [rand_genome((1000, 5000)) for _ in range(1000)]

    t0 = time.time()
    cells = [get_proteome(g=d, ignore_cds=True) for d in gs]
    print(f"get proteome 1000x: {time.time() - t0:.2f}s")

    t0 = time.time()
    A, B = get_cell_params(cells=cells)
    print(f"get cell params for 1000 cells: {time.time() - t0:.2f}s")

    C = np.random.random((len(cells), len(Signal)))

    t0 = time.time()
    res = simulate_protein_work(C=C, A=A, B=B)
    print(f"simulate_protein_work for 1000 cells: {time.time() - t0:.2f}s")

    world = World(size=128, layers=4)
    world.map = torch.randn(4, 1, 128, 128)
    t0 = time.time()
    for _ in range(1000):
        world.diffuse()
        world.degrade()
    print(f"128x128 world 1000 time steps: {time.time() - t0:.2f}s")

