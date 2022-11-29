import time
import torch
from util import rand_genome
from genetics import Genetics, MOLECULES, ACTIONS, DOMAINS
from world import World
from cells import Cells


def time_step(cells: Cells, world: World):
    # get C0 from cell and map molecule concentrations
    # TODO: probably wrong by now
    C0 = torch.zeros(len(cells.positions), cells.n_infos)
    for cell_i, pos in enumerate(cells.positions):
        for widx, mol in enumerate(MOLECULES):
            intern_idx = cells.info_2_cell_mol_idx[mol]
            extern_idx = cells.info_2_world_mol_idx[mol]
            C0[cell_i, intern_idx] = cells.molecule_map[cell_i, widx]
            C0[cell_i, extern_idx] = world.map[widx, 0, pos[0], pos[1]]
        for aidx, act in enumerate(ACTIONS):
            idx = cells.info_2_cell_act_idx[act]
            C0[cell_i, idx] = cells.action_map[cell_i, aidx]

    # integrate signals
    res = cells.simulate_protein_work(C=C0, A=cells.A, B=cells.B)

    # degrade cell and map
    world.diffuse()
    world.degrade()
    cells.degrade_molecules()

    # add cell's produces molecules to map and cell
    # TODO: probably wrong by now
    for cell_i, pos in enumerate(cells.positions):
        for widx, mol in enumerate(MOLECULES):
            intern_idx = cells.info_2_cell_mol_idx[mol]
            extern_idx = cells.info_2_world_mol_idx[mol]
            world.map[widx, 0, pos[0], pos[1]] += res[0, extern_idx]
            cells.molecule_map[cell_i, widx] += res[cell_i, intern_idx]
        for aidx, act in enumerate(ACTIONS):
            idx = cells.info_2_cell_act_idx[act]
            cells.action_map[cell_i, aidx] += res[cell_i, idx]


if __name__ == "__main__":
    genetics = Genetics(domain_map=DOMAINS)
    world = World(size=128, layers=4)
    cells = Cells(molecules=MOLECULES, actions=ACTIONS)

    gs = [rand_genome((1000, 5000)) for _ in range(1000)]
    positions = world.position_cells(n_cells=len(gs))

    t0 = time.time()
    prtms = [genetics.get_proteome(seq=d) for d in gs]
    print(f"get proteome 1000x: {time.time() - t0:.2f}s")

    cells.add_cells(proteomes=prtms, positions=positions)

    t0 = time.time()
    A, B = cells.get_cell_params(cells=prtms)
    print(f"get cell params for 1000 cells: {time.time() - t0:.2f}s")

    C = torch.randn(len(prtms), cells.n_infos)

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

    genetics = Genetics(domain_map=DOMAINS)
    cells = Cells(molecules=MOLECULES, actions=ACTIONS)
    world = World(size=128, layers=len(MOLECULES), map_init="randn")

    # get initial cell params
    init_gs = [rand_genome() for _ in range(10)]
    init_pts = [genetics.get_proteome(seq=d) for d in init_gs]
    init_positions = world.position_cells(n_cells=len(init_pts))
    cells.add_cells(proteomes=init_pts, positions=init_positions)

    t0 = time.time()
    for _ in range(1000):
        time_step(world=world, cells=cells)
    print(f"time_step 1000x: {time.time() - t0:.2f}s")

