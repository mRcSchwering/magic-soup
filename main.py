import time
import torch
import magicsoup as ms
from magicsoup.util import variants
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS


if __name__ == "__main__":
    n_threads = torch.get_num_threads()
    print(f"n threads {n_threads}")
    n_interop_threads = torch.get_num_interop_threads()
    print(f"n interop threads {n_interop_threads}")

    # fmt: off
    domains = {
        ms.CatalyticFact(): variants("ACNTGN") + variants("AGNTGN") + variants("CCNTTN"),
        ms.TransporterFact(): variants("ACNAGN") + variants("ACNTAN") + variants("AANTCN"),
        ms.AllostericFact(is_transmembrane=False, is_inhibitor=False): variants("GCNTGN"),
        ms.AllostericFact(is_transmembrane=False, is_inhibitor=True): variants("GCNTAN"),
        ms.AllostericFact(is_transmembrane=True, is_inhibitor=False): variants("AGNTCN"),
        ms.AllostericFact(is_transmembrane=True, is_inhibitor=True): variants("CCNTGN"),
    }
    # fmt: on

    genetics = ms.Genetics(
        domain_facts=domains, molecules=MOLECULES, reactions=REACTIONS,
    )

    world = ms.World(molecules=MOLECULES)

    n_cells = 1000
    n_steps = 100

    t0 = time.time()
    genomes = genetics.get_genomes(n=n_cells)
    td = time.time() - t0
    print(f"Generating {n_cells} genomes: {td:.2f}s")

    t0 = time.time()
    proteomes = genetics.get_proteomes(sequences=genomes)
    td = time.time() - t0
    print(f"Getting {n_cells} proteomes: {td:.2f}s")

    t0 = time.time()
    cells = [ms.Cell(genome=g, proteome=p) for g, p in zip(genomes, proteomes)]
    world.add_random_cells(cells=cells)
    td = time.time() - t0
    print(f"Adding {n_cells} cells: {td:.2f}s")
    print(f"{int(world.affinities.shape[1])} max proteins")

    t0 = time.time()
    world.integrate_signals()
    td = time.time() - t0
    print(f"Integrating signals: {td:.2f}s")

    t0 = time.time()
    world.degrade_molecules()
    world.diffuse_molecules()
    world.increment_cell_survival()
    td = time.time() - t0
    print(f"Degrade, diffuse, increment: {td:.2f}s")

    t0 = time.time()
    world.integrate_signals()
    world.degrade_molecules()
    world.diffuse_molecules()
    world.increment_cell_survival()
    td = time.time() - t0
    print(f"One step: {td:.2f}s")

    t0 = time.time()
    for _ in range(n_steps):
        world.integrate_signals()
        world.degrade_molecules()
        world.diffuse_molecules()
        world.increment_cell_survival()
        td = time.time() - t0
    print(f"Doing {n_steps} steps: {td:.2f}s")

