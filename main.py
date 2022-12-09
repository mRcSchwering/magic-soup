import time
import magicsoup as ms
from magicsoup.util import variants
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS


if __name__ == "__main__":
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
        domain_facts=domains, molecules=MOLECULES, reactions=REACTIONS
    )

    world = ms.World(molecules=MOLECULES, n_max_proteins=1000)

    n_cells = 1000
    n_steps = 10

    t0 = time.time()
    genomes = genetics.get_genomes(n=n_cells)
    td = time.time() - t0
    print(f"Generating {n_cells} genomes: {td:.2f}s")

    t0 = time.time()
    proteomes = genetics.get_proteomes(sequences=genomes)
    td = time.time() - t0
    print(f"Getting {n_cells} proteomes: {td:.2f}s")

    t0 = time.time()
    world.add_random_cells(genomes=genomes, proteomes=proteomes)
    td = time.time() - t0
    print(f"Adding {n_cells} cells: {td:.2f}s")

    t0 = time.time()
    world.integrate_signals()
    print(f"Integrating signals: {td:.2f}s")

    t0 = time.time()
    world.degrade_molecules()
    world.diffuse_molecules()
    world.increment_cell_survival()
    td = time.time() - t0
    print(f"Degrade, diffuse, increment: {td:.2f}s")

    t0 = time.time()
    world._send_molecules_from_world_to_x()
    world._send_molecules_from_x_to_world()
    td = time.time() - t0
    print(f"Send molecules to X and back: {td:.2f}s")

    t0 = time.time()
    world.integrate_signals()
    world.degrade_molecules()
    world.diffuse_molecules()
    world.increment_cell_survival()
    td = time.time() - t0
    print(f"One step: {td:.2f}s")

    exit()
    t0 = time.time()
    for _ in range(n_steps):
        world.integrate_signals()
        world.degrade_molecules()
        world.diffuse_molecules()
        world.increment_cell_survival()
        td = time.time() - t0
    print(f"Doing {n_steps} steps: {td:.2f}s")

