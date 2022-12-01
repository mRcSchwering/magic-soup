import time
import torch
import magicsoup as ms
from magicsoup.util import rand_genome
from magicsoup.examples.default import MOLECULES, ACTIONS, DOMAINS


def time_step(world: ms.World, A: torch.Tensor, B: torch.Tensor, Z: torch.Tensor):
    # get X0 from cell and map molecule concentrations
    X = world.get_signals()

    # integrate signals
    Xd = world.integrate_signals(X=X, A=A, B=B, Z=Z)
    X = X + Xd

    # add cell's produces molecules to map and cell
    world.update_signals(X=X)

    # degrade cell and map
    world.diffuse_molecules()
    world.degrade_molecules()


if __name__ == "__main__":
    genetics = ms.Genetics(domain_map=DOMAINS)
    world = ms.World(molecules=MOLECULES, actions=ACTIONS, map_size=128)

    gs = [rand_genome((1000, 5000)) for _ in range(1000)]
    world.add_cells(genomes=gs)

    t0 = time.time()
    prtms = [genetics.get_proteome(seq=d) for d in gs]
    print(f"get proteome 1000x: {time.time() - t0:.2f}s")

    world.add_cells(genomes=gs)

    X = torch.randn(len(prtms), world.n_signals)

    t0 = time.time()
    A, B, Z = world.get_cell_params(proteomes=prtms)
    print(f"get cell params for 1000 cells: {time.time() - t0:.2f}s")

    t0 = time.time()
    res = world.integrate_signals(X=X, A=A, B=B, Z=Z)
    print(f"simulate_protein_work for 1000 cells: {time.time() - t0:.2f}s")

    world.molecule_map = torch.randn(len(MOLECULES), 128, 128)
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

    genetics = ms.Genetics(domain_map=DOMAINS)
    world = ms.World(
        molecules=MOLECULES, actions=ACTIONS, map_size=128, mol_map_init="randn"
    )

    # get initial cell params
    init_gs = [rand_genome() for _ in range(10)]
    init_pts = [genetics.get_proteome(seq=d) for d in init_gs]
    world.add_cells(genomes=init_gs)
    A, B, Z = world.get_cell_params(proteomes=init_pts)

    t0 = time.time()
    for i in range(1000):
        time_step(world=world, A=A, B=B, Z=Z)
    print(f"time_step 1000x: {time.time() - t0:.2f}s")

