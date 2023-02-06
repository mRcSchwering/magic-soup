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

# fmt: off
CODON_TABLE = {
    "TTT": 1,   "TCT": 2,   "TAT": 3,   "TGT": 4,
    "TTC": 5,   "TCC": 6,   "TAC": 7,   "TGC": 8,
    "TTA": 9,   "TCA": 10,  "TAA": 11,  "TGA": 12,
    "TTG": 13,  "TCG": 14,  "TAG": 15,  "TGG": 16,
    "CTT": 17,  "CCT": 18,  "CAT": 19,  "CGT": 20,
    "CTC": 21,  "CCC": 22,  "CAC": 23,  "CGC": 24,
    "CTA": 25,  "CCA": 26,  "CAA": 27,  "CGA": 28,
    "CTG": 29,  "CCG": 30,  "CAG": 31,  "CGG": 32,
    "ATT": 33,  "ACT": 34,  "AAT": 35,  "AGT": 36,
    "ATC": 37,  "ACC": 38,  "AAC": 39,  "AGC": 40,
    "ATA": 41,  "ACA": 42,  "AAA": 43,  "AGA": 44,
    "ATG": 45,  "ACG": 46,  "AAG": 47,  "AGG": 48,
    "GTT": 49,  "GCT": 50,  "GAT": 51,  "GGT": 52,
    "GTC": 53,  "GCC": 54,  "GAC": 55,  "GGC": 56,
    "GTA": 57,  "GCA": 58,  "GAA": 59,  "GGA": 60,
    "GTG": 61,  "GCG": 62,  "GAG": 63,  "GGG": 64
}
AA_MAP = {
    "Stop": 0,  # not defined
    "A": 1, "F": 2, "G": 3, "I": 4, "L": 5, "M": 6, "P": 7, "V": 8, "W": 9,  # non-polar 1-9
    "C": 10, "N": 11, "Q": 12, "S": 13, "T": 14, "Y": 15,  # polar 10-15
    "H": 16, "K": 17, "R": 18,  # basic 16-18
    "D": 19, "E": 20  # acidic 19-20
}
# fmt: on


def timeit(callback, *args, r=3) -> tuple[float, float]:
    tds = []
    for _ in range(r):
        t0 = time.time()
        callback(*args)
        tds.append(time.time() - t0)
    m = sum(tds) / r
    s = sum((d - m) ** 2 / r for d in tds) ** (1 / 2)
    return m, s


def add_cells(world: ms.World, genomes: list[str]):
    world.add_random_cells(genomes=genomes)


def update_cells(world: ms.World, genomes: list[str]):
    pairs = [(d, i) for d, i in zip(genomes, range(len(world.cells)))]
    world.update_cells(genome_idx_pairs=pairs)


def main(n=1000, s=500, w=4):
    print(f"{n:,} genomes, {s:,} size, {w} workers")
    genomes = [ms.random_genome(s) for _ in range(n)]

    world = ms.World(chemistry=CHEMISTRY, workers=w)
    mu, sd = timeit(add_cells, world, genomes)
    print(f"({mu:.2f}+-{sd:.2f})s - add cells")

    world = ms.World(chemistry=CHEMISTRY, workers=w)
    genomes = [ms.random_genome(s) for _ in range(n)]
    world.add_random_cells(genomes=genomes)
    mu, sd = timeit(update_cells, world, genomes)
    print(f"({mu:.2f}+-{sd:.2f})s - update")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n", default=1_000, type=int)
    parser.add_argument("--s", default=500, type=int)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()
    main(n=args.n, s=args.s, w=args.workers)
