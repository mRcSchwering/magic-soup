import time
import torch
from util import rand_genome
from genetics import Genetics, MOLECULES, ACTIONS, DOMAINS
from world import World
from cells import Cells


def time_step(cells: Cells, world: World):
    # get X0 from cell and map molecule concentrations
    X0 = cells.get_signals(world_mol_map=world.molecule_map)

    # integrate signals
    res = cells.simulate_protein_work(X=X0, A=cells.A, B=cells.B, Z=cells.Z)

    # degrade cell and map
    world.diffuse_molecules()
    world.degrade_molecules()
    cells.degrade_molecules()

    # add cell's produces molecules to map and cell
    cells.update_signal_maps(X=res, world_mol_map=world.molecule_map)


if __name__ == "__main__":
    genetics = Genetics(domain_map=DOMAINS)
    world = World(size=128, n_molecules=4)
    cells = Cells(molecules=MOLECULES, actions=ACTIONS)

    gs = [rand_genome((1000, 5000)) for _ in range(1000)]
    positions = world.add_cells(n_cells=len(gs))

    t0 = time.time()
    prtms = [genetics.get_proteome(seq=d) for d in gs]
    print(f"get proteome 1000x: {time.time() - t0:.2f}s")

    cells.add_cells(genomes=gs, proteomes=prtms, positions=positions)

    t0 = time.time()
    A, B, Z = cells.get_cell_params(proteomes=prtms)
    print(f"get cell params for 1000 cells: {time.time() - t0:.2f}s")

    X = torch.randn(len(prtms), cells.n_infos)

    t0 = time.time()
    res = cells.simulate_protein_work(X=X, A=A, B=B, Z=Z)
    print(f"simulate_protein_work for 1000 cells: {time.time() - t0:.2f}s")

    world.molecule_map = torch.randn(4, 1, 128, 128)
    t0 = time.time()
    for _ in range(1000):
        world.diffuse_molecules()
        world.degrade_molecules()
    print(f"128x128 world 1000 time steps: {time.time() - t0:.2f}s")

    t0 = time.time()
    cdss = [genetics.get_coding_regions(seq=d) for d in gs]
    print(f"get_coding_regions 1000x: {time.time() - t0:.2f}s")

    t0 = time.time()
    a = [genetics.translate_seq(seq=dd) for d in cdss for dd in d]
    print(f"translate_seq 1000x: {time.time() - t0:.2f}s")

    #############

    genetics = Genetics(domain_map=DOMAINS)
    cells = Cells(molecules=MOLECULES, actions=ACTIONS)
    world = World(size=128, n_molecules=len(MOLECULES), mol_map_init="randn")

    # get initial cell params
    init_gs = [rand_genome() for _ in range(10)]
    init_pts = [genetics.get_proteome(seq=d) for d in init_gs]
    init_positions = world.add_cells(n_cells=len(init_pts))
    cells.add_cells(genomes=init_gs, proteomes=init_pts, positions=init_positions)

    t0 = time.time()
    for i in range(1000):
        time_step(world=world, cells=cells)
    print(f"time_step 1000x: {time.time() - t0:.2f}s")

