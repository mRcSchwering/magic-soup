"""
Little helper script for checking the performance of some functions.
"""

import random
import statistics
import time

import magicsoup as ms
import torch
from magicsoup.examples.wood_ljungdahl import chemistry


def _summary(tds: list[float]) -> str:
    mu = statistics.mean(tds)
    sd = statistics.stdev(tds)
    return f"({mu:.2f}+-{sd:.2f})s"


def _gen_genomes(n: int, s: int, d=0.1) -> list[str]:
    pop = [-int(s * d), s, int(s * d)]
    return [ms.random_genome(s + random.choice(pop)) for _ in range(n)]


def spawn_cells(device: str, n: int, s: int, n_rep: int):
    tds = []
    for _ in range(n_rep):
        world = ms.World(chemistry=chemistry, device=device)
        genomes = _gen_genomes(n=n, s=s)
        t0 = time.time()
        world.spawn_cells(genomes=genomes)
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def replicate_cells(device: str, n: int, s: int, n_rep: int):
    tds = []
    for _ in range(n_rep):
        world = ms.World(chemistry=chemistry, device=device)
        genomes = _gen_genomes(n=n, s=s)
        world.spawn_cells(genomes=genomes)
        t0 = time.time()
        world.divide_cells()
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def advance_time(device: str, n: int, s: int, n_rep: int):
    tds = []
    for _ in range(n_rep):
        world = ms.World(chemistry=chemistry, device=device)
        genomes = _gen_genomes(n=n, s=s)
        world.spawn_cells(genomes=genomes)
        t0 = time.time()
        world.advance_time()
        tds.append(time.time() - t0)
    return _summary(tds=tds)


def mutations(device: str, n: int, s: int, n_rep: int):
    world = ms.World(chemistry=chemistry, device=device)
    genomes = _gen_genomes(n=n, s=s)
    world.spawn_cells(genomes=genomes)
    tds = []
    for _ in range(n_rep):
        world = ms.World(chemistry=chemistry, device=device)
        genomes = _gen_genomes(n=n, s=s)
        world.spawn_cells(genomes=genomes)
        t0 = time.time()
        world.mutate_cells()
        tds.append(time.time() - t0)
    return _summary(tds=tds)


default_parts = {
    "spawn_cells": spawn_cells,
    "replicate_cells": replicate_cells,
    "advance_time": advance_time,
    "mutations": mutations,
}


@torch.inference_mode()
def main(parts: list, n_cells: int, genome_size: int, device: str, n_rep: int = 5, **_):
    print(f"Running {', '.join(parts)}")
    print(f"{n_cells:,} cells, {genome_size:,} genome size, on {device}")

    for part in parts:
        smry = default_parts[part](device=device, n=n_cells, s=genome_size, n_rep=n_rep)
        print(f"{smry} - {part}")
