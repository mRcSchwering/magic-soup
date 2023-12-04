"""
Little helper script for checking the performance of some functions

    PYTHONPATH=./src python performance/check.py --n=1000 --s=1000

v0.7.0:

1,000 genomes, 500 size, 4 workers
(0.42+-0.05)s - add cells
(0.43+-0.02)s - update cells
(0.09+-0.01)s - replicate cells
(0.22+-0.06)s - enzymatic activity
(0.06+-0.01)s - get neighbors

Running update_cells
10,000 cells, 1,000 genome size, 4 workers
(8.23+-0.41)s - update cells
"""
import random
import time
from argparse import ArgumentParser
import magicsoup as ms
from magicsoup import functions as fn
from magicsoup.examples.wood_ljungdahl import CHEMISTRY

R = 5


def _gen_genomes(n: int, s: int, d=0.1) -> list[str]:
    pop = [-int(n * d), 0, int(n * d)]
    return [ms.random_genome(s + random.choice(pop)) for _ in range(n)]


def add_cells(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, workers=w)
        genomes = _gen_genomes(n, s)

        t0 = time.time()
        world.spawn_cells(genomes=genomes)
        tds.append(time.time() - t0)

    m = sum(tds) / R
    s = sum((d - m) ** 2 / R for d in tds) ** (1 / 2)
    return m, s


def update_cells(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, workers=w)
        genomes = _gen_genomes(n, s)
        world.spawn_cells(genomes=genomes)

        t0 = time.time()
        pairs = [(d, i) for i, d in enumerate(world.cell_genomes)]
        world.update_cells(genome_idx_pairs=pairs)
        tds.append(time.time() - t0)

    m = sum(tds) / R
    s = sum((d - m) ** 2 / R for d in tds) ** (1 / 2)
    return m, s


def replicate_cells(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, workers=w)
        genomes = _gen_genomes(n, s)
        idxs = world.spawn_cells(genomes=genomes)

        t0 = time.time()
        world.divide_cells(cell_idxs=idxs)
        tds.append(time.time() - t0)

    m = sum(tds) / R
    s = sum((d - m) ** 2 / R for d in tds) ** (1 / 2)
    return m, s


def enzymatic_activity(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, workers=w)
        genomes = _gen_genomes(n, s)
        world.spawn_cells(genomes=genomes)

        t0 = time.time()
        world.enzymatic_activity()
        tds.append(time.time() - t0)

    m = sum(tds) / R
    s = sum((d - m) ** 2 / R for d in tds) ** (1 / 2)
    return m, s


def get_neighbors(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, workers=w)
        genomes = _gen_genomes(n, s)
        world.spawn_cells(genomes=genomes)

        t0 = time.time()
        _ = world.get_neighbors(cell_idxs=list(range(world.n_cells)))
        tds.append(time.time() - t0)

    m = sum(tds) / R
    s = sum((d - m) ** 2 / R for d in tds) ** (1 / 2)
    return m, s


def get_point_mutations(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        gs = _gen_genomes(n, s)
        t0 = time.time()
        pairs = ms.point_mutations(seqs=gs)
        tds.append(time.time() - t0)
        assert len(pairs) > 0
    m = sum(tds) / R
    s = sum((d - m) ** 2 / R for d in tds) ** (1 / 2)
    return m, s


def get_point_mutations_np(w: int, n: int, s: int):
    tds = []
    gs = _gen_genomes(n, s)
    pairs = fn.point_mutations(seqs=gs)
    for _ in range(R):
        gs = _gen_genomes(n, s)
        t0 = time.time()
        pairs = fn.point_mutations(seqs=gs)
        tds.append(time.time() - t0)
        assert len(pairs) > 0
    m = sum(tds) / R
    s = sum((d - m) ** 2 / R for d in tds) ** (1 / 2)
    return m, s


def get_point_mutations_np2(w: int, n: int, s: int):
    tds = []
    gs = _gen_genomes(n, s)
    pairs = fn.point_mutations2(seqs=gs)
    for _ in range(R):
        gs = _gen_genomes(n, s)
        t0 = time.time()
        pairs = fn.point_mutations2(seqs=gs)
        tds.append(time.time() - t0)
        assert len(pairs) > 0
    m = sum(tds) / R
    s = sum((d - m) ** 2 / R for d in tds) ** (1 / 2)
    return m, s


def main(parts: list, n: int, s: int, w: int):
    print(f"Running {', '.join(parts)}")
    print(f"{n:,} cells, {s:,} genome size, {w} workers")

    if "add_cells" in parts:
        mu, sd = add_cells(w=w, n=n, s=s)
        print(f"({mu:.2f}+-{sd:.2f})s - add cells")

    if "update_cells" in parts:
        mu, sd = update_cells(w=w, n=n, s=s)
        print(f"({mu:.2f}+-{sd:.2f})s - update cells")

    if "replicate_cells" in parts:
        mu, sd = replicate_cells(w=w, n=n, s=s)
        print(f"({mu:.2f}+-{sd:.2f})s - replicate cells")

    if "enzymatic_activity" in parts:
        mu, sd = enzymatic_activity(w=w, n=n, s=s)
        print(f"({mu:.2f}+-{sd:.2f})s - enzymatic activity")

    if "get_neighbors" in parts:
        mu, sd = get_neighbors(w=w, n=n, s=s)
        print(f"({mu:.2f}+-{sd:.2f})s - get neighbors")

    if "point_mutations" in parts:
        mu, sd = get_point_mutations(w=w, n=n, s=s)
        print(f"({mu:.2f}+-{sd:.2f})s - point_mutations")
        mu, sd = get_point_mutations_np(w=w, n=n, s=s)
        print(f"({mu:.2f}+-{sd:.2f})s - point_mutations_np")
        mu, sd = get_point_mutations_np2(w=w, n=n, s=s)
        print(f"({mu:.2f}+-{sd:.2f})s - point_mutations_np2")


if __name__ == "__main__":
    default_parts: list[str] = [
        "add_cells",
        "update_cells",
        "replicate_cells",
        "enzymatic_activity",
        "get_neighbors",
    ]

    parser = ArgumentParser()
    parser.add_argument("--parts", default=default_parts, nargs="*", action="store")
    parser.add_argument("--n-cells", default=10_000, type=int)
    parser.add_argument("--genome-size", default=1_000, type=int)
    parser.add_argument("--n-workers", default=4, type=int)
    args = parser.parse_args()
    main(parts=args.parts, n=args.n_cells, s=args.genome_size, w=args.n_workers)
