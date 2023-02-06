import time
from argparse import ArgumentParser
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import CHEMISTRY

CODON_SIZE = 3
MIN_CDS_SIZE = 12
START_CODONS = ("TTG", "GTG", "ATG")
STOP_CODONS = ("TGA", "TAG", "TAA")
NON_POLAR_AAS = ["F", "L", "I", "M", "V", "P", "A", "W", "G"]
POLAR_AAS = ["S", "T", "Y", "Q", "N", "C"]
BASIC_AAS = ["H", "K", "R"]
ACIDIC_AAS = ["D", "E"]


def timeit(callback, *args, r=3) -> tuple[float, float]:
    tds = []
    for _ in range(r):
        t0 = time.time()
        callback(*args)
        tds.append(time.time() - t0)
    m = sum(tds) / r
    s = sum((d - m) ** 2 / r for d in tds) ** (1 / 2)
    return m, s


def classic_add_random_cells(world: ms.World, genomes: list[str]):
    world.add_random_cells(genomes=genomes)


def main(n=1000, s=500, w=4):
    print(f"{n:,} genomes, {s:,} size, {w} workers")
    genomes = [ms.random_genome(s) for _ in range(n)]

    world = ms.World(chemistry=CHEMISTRY, workers=w)
    mu, sd = timeit(classic_add_random_cells, world, genomes)
    print(f"({mu:.2f}+-{sd:.2f})s - classic")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n", default=1_000, type=int)
    parser.add_argument("--s", default=500, type=int)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()
    main(n=args.n, s=args.s, w=args.workers)
