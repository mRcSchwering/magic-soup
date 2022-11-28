import time
import torch
from util import rand_genome
from genetics import Genetics, CellSignal, WorldSignal
from world import World
from cells import Cells


def time_step(cells: Cells, world: World):
    # get C0 from cell and map molecule concentrations
    C0 = torch.zeros(len(cells.positions), len(CellSignal) + len(WorldSignal))
    for cell_i, pos in enumerate(cells.positions):
        for idx, wsig in enumerate(WorldSignal):
            C0[cell_i, wsig.value] = world.map[idx, 0, pos[0], pos[1]]
        for idx, csig in enumerate(CellSignal):
            C0[cell_i, csig.value] = cells.cell_signals[cell_i, idx]

    # integrate signals
    res = cells.simulate_protein_work(C=C0, A=cells.A, B=cells.B)

    # degrade cell and map
    world.diffuse()
    world.degrade()
    cells.degrade_signals()

    # add cell's produces molecules to map and cell
    for cell_i, pos in enumerate(cells.positions):
        for idx, wsig in enumerate(WorldSignal):
            world.map[idx, 0, pos[0], pos[1]] += res[0, wsig.value]
        for idx, csig in enumerate(CellSignal):
            cells.cell_signals[cell_i, idx] += res[cell_i, csig.value]


if __name__ == "__main__":
    genetics = Genetics()
    world = World(size=128, layers=4)
    cells = Cells(n_cell_signals=len(CellSignal), n_world_signals=len(WorldSignal))

    gs = [rand_genome((1000, 5000)) for _ in range(1000)]
    positions = world.position_cells(n_cells=len(gs))

    t0 = time.time()
    prtms = [genetics.get_proteome(g=d) for d in gs]
    print(f"get proteome 1000x: {time.time() - t0:.2f}s")

    cells.add_cells(proteomes=prtms, positions=positions)

    t0 = time.time()
    A, B = cells.get_cell_params(cells=prtms)
    print(f"get cell params for 1000 cells: {time.time() - t0:.2f}s")

    C = torch.randn(len(prtms), len(CellSignal) + len(WorldSignal))

    t0 = time.time()
    res = cells.simulate_protein_work(C=C, A=A, B=B)
    print(f"simulate_protein_work for 1000 cells: {time.time() - t0:.2f}s")

    world.map = torch.randn(4, 1, 128, 128)
    t0 = time.time()
    for _ in range(1000):
        world.diffuse()
        world.degrade()
    print(f"128x128 world 1000 time steps: {time.time() - t0:.2f}s")

    t0 = time.time()
    cdss = [genetics.get_coding_regions(seq=d) for d in gs]
    print(f"get_coding_regions 1000x: {time.time() - t0:.2f}s")

    t0 = time.time()
    a = [genetics.translate_seq(seq=dd) for d in cdss for dd in d]
    print(f"translate_seq 1000x: {time.time() - t0:.2f}s")

    #############

    genetics = Genetics()
    cells = Cells(
        n_cell_signals=len(CellSignal),
        n_world_signals=len(WorldSignal),
        max_proteins=10_000,
    )
    world = World(size=128, layers=len(WorldSignal), map_init="randn")

    # get initial cell params
    init_gs = [rand_genome() for _ in range(10)]
    init_pts = [genetics.get_proteome(g=d) for d in init_gs]
    init_positions = world.position_cells(n_cells=len(init_pts))
    cells.add_cells(proteomes=init_pts, positions=init_positions)

    t0 = time.time()
    for _ in range(1000):
        time_step(world=world, cells=cells)
    print(f"time_step 1000x: {time.time() - t0:.2f}s")

