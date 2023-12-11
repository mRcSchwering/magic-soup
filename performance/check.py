"""
Little helper script for checking the performance of some functions

    PYTHONPATH=./python python performance/check.py

v0.9.0:
10,000 cells, 1,000 genome size, 4 workers
(13.21+-0.50)s - add cells
(11.85+-0.25)s - update cells
(0.81+-0.06)s - replicate cells
(5.82+-0.79)s - enzymatic activity
(6.21+-0.13)s - get neighbors
(0.25+-0.01)s - point mutations
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


def add_cells(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, workers=w)
        genomes = _gen_genomes(n=n, s=s)
        t0 = time.time()
        world.spawn_cells(genomes=genomes)
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def update_cells(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, workers=w)
        genomes = _gen_genomes(n=n, s=s)
        world.spawn_cells(genomes=genomes)
        t0 = time.time()
        pairs = [(d, i) for i, d in enumerate(world.cell_genomes)]
        world.update_cells(genome_idx_pairs=pairs)
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def replicate_cells(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, workers=w)
        genomes = _gen_genomes(n=n, s=s)
        idxs = world.spawn_cells(genomes=genomes)
        t0 = time.time()
        world.divide_cells(cell_idxs=idxs)
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def enzymatic_activity(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, workers=w)
        genomes = _gen_genomes(n=n, s=s)
        world.spawn_cells(genomes=genomes)
        t0 = time.time()
        world.enzymatic_activity()
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def get_neighbors(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, workers=w)
        genomes = _gen_genomes(n=n, s=s)
        world.spawn_cells(genomes=genomes)
        t0 = time.time()
        _ = world.get_neighbors(cell_idxs=list(range(world.n_cells)))
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def get_point_mutations(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        genomes = _gen_genomes(n=n, s=s)
        t0 = time.time()
        _ = ms.point_mutations(seqs=genomes)
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def get_test(w: int, n: int, s: int):
    genetics = ms.Genetics()
    tds = []
    for _ in range(R):
        genomes = _gen_genomes(n=n, s=s)
        t0 = time.time()
        genetics.translate_genomes(genomes=genomes)
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def main(parts: list, n: int, s: int, w: int):
    print(f"Running {', '.join(parts)}")
    print(f"{n:,} cells, {s:,} genome size, {w} workers")

    if "add_cells" in parts:
        smry = add_cells(w=w, n=n, s=s)
        print(f"{smry} - add cells")

    if "update_cells" in parts:
        smry = update_cells(w=w, n=n, s=s)
        print(f"{smry} - update cells")

    if "replicate_cells" in parts:
        smry = replicate_cells(w=w, n=n, s=s)
        print(f"{smry} - replicate cells")

    if "enzymatic_activity" in parts:
        smry = enzymatic_activity(w=w, n=n, s=s)
        print(f"{smry} - enzymatic activity")

    if "get_neighbors" in parts:
        smry = get_neighbors(w=w, n=n, s=s)
        print(f"{smry} - get neighbors")

    if "point_mutations" in parts:
        smry = get_point_mutations(w=w, n=n, s=s)
        print(f"{smry} - point mutations")

    if "test" in parts:
        smry = get_test(w=w, n=n, s=s)
        print(f"{smry} - test")


if __name__ == "__main__":
    default_parts: list[str] = [
        "add_cells",
        "update_cells",
        "replicate_cells",
        "enzymatic_activity",
        "get_neighbors",
        "point_mutations",
    ]

    parser = ArgumentParser()
    parser.add_argument("--parts", default=default_parts, nargs="*", action="store")
    parser.add_argument("--n-cells", default=10_000, type=int)
    parser.add_argument("--genome-size", default=1_000, type=int)
    parser.add_argument("--n-workers", default=4, type=int)
    args = parser.parse_args()
    main(parts=args.parts, n=args.n_cells, s=args.genome_size, w=args.n_workers)
