"""
Little helper script for checking the performance of some functions

    PYTHONPATH=./src python performance/check.py --n=1000 --s=1000

It seems with v0.3.0 everything got a bit slower :/
- v0.2.2 (python3.9,torch1.13) add/update cells is around 0.3s
- v0.3.0 (python10.10,torch2.0) add/update cell is around 0.4s

"""
import time
from argparse import ArgumentParser
import magicsoup as ms
from magicsoup.kinetics2 import Kinetics as Kinetics2
from magicsoup.examples.wood_ljungdahl import CHEMISTRY

R = 5
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


def add_cells(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, workers=w)
        genomes = [ms.random_genome(s) for _ in range(n)]

        t0 = time.time()
        world.add_cells(genomes=genomes)
        tds.append(time.time() - t0)

    m = sum(tds) / R
    s = sum((d - m) ** 2 / R for d in tds) ** (1 / 2)
    return m, s


def update_cells(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, workers=w)
        genomes = [ms.random_genome(s) for _ in range(n)]
        world.add_cells(genomes=genomes)

        t0 = time.time()
        pairs = [(d, i) for i, d in enumerate(world.genomes)]
        world.update_cells(genome_idx_pairs=pairs)
        tds.append(time.time() - t0)

    m = sum(tds) / R
    s = sum((d - m) ** 2 / R for d in tds) ** (1 / 2)
    return m, s


def replicate_cells(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, workers=w)
        genomes = [ms.random_genome(s) for _ in range(n)]
        idxs = world.add_cells(genomes=genomes)

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
        genomes = [ms.random_genome(s) for _ in range(n)]
        world.add_cells(genomes=genomes)

        t0 = time.time()
        world.enzymatic_activity()
        tds.append(time.time() - t0)

    m = sum(tds) / R
    s = sum((d - m) ** 2 / R for d in tds) ** (1 / 2)
    return m, s


def enzymatic_activity2(w: int, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, workers=w)
        world.kinetics = Kinetics2(
            molecules=CHEMISTRY.molecules,
            reactions=CHEMISTRY.reactions,
            km_range=(0.01, 100),
            vmax_range=(0.01, 100),
            n_computations=9,
        )

        genomes = [ms.random_genome(s) for _ in range(n)]
        world.add_cells(genomes=genomes)

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
        genomes = [ms.random_genome(s) for _ in range(n)]
        world.add_cells(genomes=genomes)

        t0 = time.time()
        _ = world.get_neighbors(cell_idxs=list(range(world.n_cells)))
        tds.append(time.time() - t0)

    m = sum(tds) / R
    s = sum((d - m) ** 2 / R for d in tds) ** (1 / 2)
    return m, s


def main(n: int, s: int, w: int):
    print(f"{n:,} genomes, {s:,} size, {w} workers")

    # mu, sd = add_cells(w=w, n=n, s=s)
    # print(f"({mu:.2f}+-{sd:.2f})s - add cells")

    # mu, sd = update_cells(w=w, n=n, s=s)
    # print(f"({mu:.2f}+-{sd:.2f})s - update cells")

    # mu, sd = replicate_cells(w=w, n=n, s=s)
    # print(f"({mu:.2f}+-{sd:.2f})s - replicate cells")

    mu, sd = enzymatic_activity(w=w, n=n, s=s)
    print(f"({mu:.2f}+-{sd:.2f})s - enzymatic activity")

    mu, sd = enzymatic_activity2(w=w, n=n, s=s)
    print(f"({mu:.2f}+-{sd:.2f})s - enzymatic activity2")

    # mu, sd = get_neighbors(w=w, n=n, s=s)
    # print(f"({mu:.2f}+-{sd:.2f})s - get neighbors")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n", default=1_000, type=int)
    parser.add_argument("--s", default=500, type=int)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()
    main(n=args.n, s=args.s, w=args.workers)
