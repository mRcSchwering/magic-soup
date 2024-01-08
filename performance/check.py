"""
Little helper script for checking the performance of some functions.
For local python package:

    PYTHONPATH=./python python performance/check.py

For installed python package:

    python check.py

v0.14.1 GPU (g4dn.xlarge):
Running spawn_cells, update_cells, replicate_cells, enzymatic_activity, mutations
10,000 cells, 1,000 genome size, on cuda
(6.64+-1.08)s - spawn cells
(5.95+-0.22)s - update cells
(0.28+-0.00)s - replicate cells
(0.16+-0.00)s - enzymatic activity
(0.46+-0.00)s - mutations

v0.14.1 CPU (Intel Core i5-10210U):
10,000 cells, 1,000 genome size, on cpu
(7.66+-0.51)s - spawn cells
(7.17+-0.10)s - update cells
(0.37+-0.01)s - replicate cells
(4.51+-0.12)s - enzymatic activity
(0.40+-0.00)s - mutations
"""
import time
import random
from argparse import ArgumentParser
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import CHEMISTRY

R = 5


def _summary(tds: list[float]) -> str:
    mu = sum(tds) / R
    sd = sum((d - mu) ** 2 / R for d in tds) ** (1 / 2)
    return f"({mu:.2f}+-{sd:.2f})s"


def _gen_genomes(n: int, s: int, d=0.1) -> list[str]:
    pop = [-int(s * d), s, int(s * d)]
    return [ms.random_genome(s + random.choice(pop)) for _ in range(n)]


def spawn_cells(device: str, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, device=device)
        genomes = _gen_genomes(n=n, s=s)
        t0 = time.time()
        world.spawn_cells(genomes=genomes)
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def update_cells(device: str, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, device=device)
        genomes = _gen_genomes(n=n, s=s)
        world.spawn_cells(genomes=genomes)
        t0 = time.time()
        pairs = [(d, i) for i, d in enumerate(world.cell_genomes)]
        world.update_cells(genome_idx_pairs=pairs)
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def replicate_cells(device: str, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, device=device)
        genomes = _gen_genomes(n=n, s=s)
        idxs = world.spawn_cells(genomes=genomes)
        t0 = time.time()
        world.divide_cells(cell_idxs=idxs)
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def enzymatic_activity(device: str, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, device=device)
        genomes = _gen_genomes(n=n, s=s)
        world.spawn_cells(genomes=genomes)
        t0 = time.time()
        world.enzymatic_activity()
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def mutations(device: str, n: int, s: int):
    world = ms.World(chemistry=CHEMISTRY, device=device)
    genomes = _gen_genomes(n=n, s=s)
    world.spawn_cells(genomes=genomes)
    tds = []
    for _ in range(R):
        t0 = time.time()
        _ = ms.point_mutations(seqs=genomes)
        nghbrs = world.get_neighbors(cell_idxs=list(range(world.n_cells)))
        nghbr_genomes = [(genomes[a], genomes[b]) for a, b in nghbrs]
        _ = ms.recombinations(seq_pairs=nghbr_genomes)
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def get_test(device: str, n: int, s: int):
    proteome: list[list[ms.DomainFactType]] = []
    for react in CHEMISTRY.reactions:
        p: list[ms.DomainFactType] = [
            ms.CatalyticDomainFact(reaction=react),
            ms.RegulatoryDomainFact(effector=react[0][0], is_transmembrane=False),
        ]
        proteome.append(p)
    for mol in CHEMISTRY.molecules:
        p = [
            ms.TransporterDomainFact(molecule=mol),
            ms.RegulatoryDomainFact(effector=mol, is_transmembrane=True),
        ]
        proteome.append(p)

    world = ms.World(chemistry=CHEMISTRY, device=device)
    genome_fact = ms.GenomeFact(world=world, proteome=proteome, target_size=s)
    tds = []
    for _ in range(R):
        t0 = time.time()
        for _ in range(n):
            _ = genome_fact.generate()
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def main(parts: list, n: int, s: int, device: str):
    print(f"Running {', '.join(parts)}")
    print(f"{n:,} cells, {s:,} genome size, on {device}")

    if "spawn_cells" in parts:
        smry = spawn_cells(device=device, n=n, s=s)
        print(f"{smry} - spawn cells")

    if "update_cells" in parts:
        smry = update_cells(device=device, n=n, s=s)
        print(f"{smry} - update cells")

    if "replicate_cells" in parts:
        smry = replicate_cells(device=device, n=n, s=s)
        print(f"{smry} - replicate cells")

    if "enzymatic_activity" in parts:
        smry = enzymatic_activity(device=device, n=n, s=s)
        print(f"{smry} - enzymatic activity")

    if "mutations" in parts:
        smry = mutations(device=device, n=n, s=s)
        print(f"{smry} - mutations")

    if "test" in parts:
        smry = get_test(n=n, s=s, device=device)
        print(f"{smry} - test")


if __name__ == "__main__":
    default_parts: list[str] = [
        "spawn_cells",
        "update_cells",
        "replicate_cells",
        "enzymatic_activity",
        "mutations",
    ]

    parser = ArgumentParser()
    parser.add_argument("--parts", default=default_parts, nargs="*", action="store")
    parser.add_argument("--n-cells", default=10_000, type=int)
    parser.add_argument("--genome-size", default=1_000, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()
    main(parts=args.parts, n=args.n_cells, s=args.genome_size, device=args.device)
