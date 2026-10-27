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

0ba7586781640e8ae85f6ce089798f8b20dc5104 (Intel Core i5-10210U):
10,000 cells, 1,000 genome size, on cpu
(16.51+-0.76)s - spawn cells
(15.38+-0.42)s - update cells
(0.35+-0.01)s - replicate cells
forever - advance time
"""

import random
import time
from argparse import ArgumentParser

import magicsoup as ms
import torch
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
        world = ms.World(chemistry=CHEMISTRY, tensor_kwargs={"device": device})
        genomes = _gen_genomes(n=n, s=s)
        t0 = time.time()
        world.spawn_cells(genomes=genomes)
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def update_cells(device: str, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, tensor_kwargs={"device": device})
        genomes = _gen_genomes(n=n, s=s)
        idxs = world.spawn_cells(genomes=genomes)
        t0 = time.time()
        world.update_cells(genomes=genomes, cell_idxs=idxs)
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def replicate_cells(device: str, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, tensor_kwargs={"device": device})
        genomes = _gen_genomes(n=n, s=s)
        idxs = world.spawn_cells(genomes=genomes)
        t0 = time.time()
        world.divide_cells(cell_idxs=idxs)
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def advance_time(device: str, n: int, s: int):
    tds = []
    for _ in range(R):
        world = ms.World(chemistry=CHEMISTRY, tensor_kwargs={"device": device})
        genomes = _gen_genomes(n=n, s=s)
        world.spawn_cells(genomes=genomes)
        t0 = time.time()
        world.advance_time()
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def mutations(device: str, n: int, s: int):
    world = ms.World(chemistry=CHEMISTRY, tensor_kwargs={"device": device})
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


@torch.inference_mode()
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

    if "advance_time" in parts:
        smry = advance_time(device=device, n=n, s=s)
        print(f"{smry} - advance_time")

    if "mutations" in parts:
        smry = mutations(device=device, n=n, s=s)
        print(f"{smry} - mutations")


if __name__ == "__main__":
    default_parts: list[str] = [
        "spawn_cells",
        "update_cells",
        "replicate_cells",
        "advance_time",
        # "mutations",
    ]

    parser = ArgumentParser()
    parser.add_argument("--parts", default=default_parts, nargs="*", action="store")
    parser.add_argument("--n-cells", default=10_000, type=int)
    parser.add_argument("--genome-size", default=1_000, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    args = parser.parse_args()
    main(parts=args.parts, n=args.n_cells, s=args.genome_size, device=args.device)
