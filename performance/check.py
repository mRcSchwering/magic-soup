"""
Little helper script for checking the performance of some functions.
For local python package:

    PYTHONPATH=./python python performance/check.py

For installed python package:

    python check.py

v0.12.1 CPU:
10,000 cells, 1,000 genome size, on cpu
(9.95+-0.33)s - spawn cells
(10.36+-0.39)s - update cells
(0.70+-0.01)s - replicate cells
(5.08+-0.36)s - enzymatic activity
(6.34+-0.25)s - mutations
(5.68+-0.11)s - mutations (smaller types)

v0.12.1 GPU (g4dn.xlarge):
10,000 cells, 1,000 genome size, 4 workers
(13.75+-0.72)s - add cells
(9.60+-0.54)s - update cells
(1.44+-0.03)s - replicate cells
(0.19+-0.01)s - enzymatic activity
(4.00+-0.01)s - get neighbors
(0.01+-0.00)s - point mutations
"""
import time
import random
from argparse import ArgumentParser
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import CHEMISTRY

R = 5

# TODO: torch: try f32 for speed


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


def get_test(n: int, s: int):
    genetics = ms.Genetics()
    tds = []
    for _ in range(R):
        genomes = _gen_genomes(n=n, s=s)
        t0 = time.time()
        _ = genetics.translate_genomes(genomes=genomes)
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
        smry = get_test(n=n, s=s)
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
