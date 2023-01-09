from argparse import ArgumentParser
import time
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from magicsoup.constants import NOW
from magicsoup.examples.wood_ljungdahl import (
    CHEMISTRY as wl_chem,
    ATP,
    ADP,
    NADPH,
    NADP,
    HSCoA,
    acetylCoA,
    co2,
)
from magicsoup.examples.reverse_krebs import CHEMISTRY as rk_chem

_this_dir = Path(__file__).parent

import json
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import NADPH

chem = rk_chem & wl_chem
with open("asd.json", "w") as fh:
    json.dump(chem.to_dict(), fh)


with open("asd.json") as fh:
    chem_dict = json.load(fh)

chem = ms.Chemistry.from_dict(chem_dict)
id(chem.molecules[0])
id(NADPH)
NADPH.idx
chem.molecules[0].idx
# 139995581757136
# 139995581757136
# 139995581757136


def sign_sample(t: torch.Tensor, k=1.0, rev=False) -> list[int]:
    # sample based on molecules
    p = t ** 3 / (t ** 3 + k ** 3)
    if rev:
        p = 1.0 - p
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def add_co2(world: ms.World):
    # add CO2 in some places
    xs = [32, 32, 32, 64, 64, 64, 96, 96, 96]
    ys = [32, 64, 96, 32, 64, 96, 32, 64, 96]
    if world.molecule_map[co2.idx, xs, ys].mean() < 10.0:
        world.molecule_map[co2.idx, xs, ys] += 1.0


def add_energy(world: ms.World):
    # restore energy carriers in some places
    xs = [48, 48, 80, 80]
    ys = [48, 80, 48, 80]
    world.molecule_map[ATP.idx, xs, ys] += world.molecule_map[ADP.idx, xs, ys] * 0.2
    world.molecule_map[ADP.idx, xs, ys] *= 0.8
    world.molecule_map[NADPH.idx, xs, ys] += world.molecule_map[NADP.idx, xs, ys] * 0.2
    world.molecule_map[NADP.idx, xs, ys] *= 0.8


def add_random_cells(world: ms.World, s: int, n: int):
    # have at least n cells living
    d = n - len(world.cells)
    if d > 0:
        seqs = [ms.random_genome(s) for _ in range(d)]
        world.add_random_cells(genomes=seqs)


def kill_cells(world: ms.World):
    # kill cells with low Acetyl-CoA, and cells that never replicate
    acoa_idx = acetylCoA.idx
    acoa_idxs = sign_sample(t=world.cell_molecules[:, acoa_idx], k=2.0, rev=True)
    surv_idxs = sign_sample(t=world.cell_survival - world.cell_divisions, k=250.0)
    unq_idxs = set(acoa_idxs + surv_idxs)
    world.kill_cells(cell_idxs=list(unq_idxs))


def replicate_cells(world: ms.World):
    # cells must have enough Acetyl-CoA to replicate
    # more AcetylCoA will increase their chances of replicating
    acoa_idx = acetylCoA.idx
    coa_idx = HSCoA.idx
    idxs1 = sign_sample(t=world.cell_molecules[:, acoa_idx], k=10.0)
    idxs2 = torch.argwhere(world.cell_molecules[:, acoa_idx] > 5.0).flatten().tolist()
    idxs = list(set(idxs1) & set(idxs2))
    world.cell_molecules[idxs, acoa_idx] -= 5.0
    world.cell_molecules[idxs, coa_idx] += 5.0
    replicated = world.replicate_cells(parent_idxs=idxs)

    genomes = [(world.cells[p].genome, world.cells[c].genome) for p, c in replicated]
    mutated = ms.recombinations(seq_pairs=genomes)

    genome_idx_pairs = []
    for parent, child, idx in mutated:
        parent_i, child_i = replicated[idx]
        genome_idx_pairs.append((parent, parent_i))
        genome_idx_pairs.append((child, child_i))
    world.update_cells(genome_idx_pairs=genome_idx_pairs)


def random_mutations(world: ms.World):
    mutated = ms.point_mutations(seqs=[d.genome for d in world.cells])
    world.update_cells(genome_idx_pairs=mutated)


def write_scalars(world: ms.World, writer: SummaryWriter, step: int, td: float):
    writer.add_scalar("Other/TimePerStep[s]", td, step)

    writer.add_scalar("Cells/total", len(world.cells), step)
    writer.add_scalar("Cells/SurvAvg", world.cell_survival.mean().item(), step)
    writer.add_scalar("Cells/SurvMax", world.cell_survival.max().item(), step)
    writer.add_scalar("Cells/DivisAvg", world.cell_divisions.mean().item(), step)
    writer.add_scalar("Cells/DivisMax", world.cell_divisions.max().item(), step)


def write_images(world: ms.World, writer: SummaryWriter, step: int):
    if step % 10 == 0:
        mm = world.molecule_map
        writer.add_image(
            "Maps/CO2", mm[co2.idx] / 10.0, step, dataformats="HW",
        )
        writer.add_image(
            "Maps/ATP",
            mm[ATP.idx] / (mm[ADP.idx] + mm[ATP.idx]),
            step,
            dataformats="HW",
        )
        writer.add_image(
            "Maps/NADPH",
            mm[NADPH.idx] / (mm[NADP.idx] + mm[NADPH.idx]),
            step,
            dataformats="HW",
        )
        writer.add_image("Maps/Cells", world.cell_map, step, dataformats="HW")


def main(n_cells: int, n_steps: int, genome_size: int):
    writer = SummaryWriter(log_dir=_this_dir / "runs" / NOW)

    world = ms.World(chemistry=rk_chem & wl_chem)
    world.molecule_map += 2.0
    world.molecule_map *= 2.0
    world.save(outdir=_this_dir / "runs" / NOW)

    for step_i in range(n_steps):
        t0 = time.time()

        add_co2(world=world)
        add_energy(world=world)

        add_random_cells(world=world, s=genome_size, n=n_cells)
        world.enzymatic_activity()

        kill_cells(world=world)
        replicate_cells(world=world)
        random_mutations(world=world)

        world.degrade_molecules()
        world.diffuse_molecules()
        world.increment_cell_survival()

        write_scalars(world=world, writer=writer, step=step_i, td=time.time() - t0)
        write_images(world=world, writer=writer, step=step_i)

        if step_i % 100 == 0:
            print(f"Finished step {step_i:,}")
            world.save_state(statedir=_this_dir / "runs" / NOW / f"step={step_i}")

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_cells", default=1000, type=int)
    parser.add_argument("--n_steps", default=10_000, type=int)
    parser.add_argument("--genome_size", default=300, type=int)
    args = parser.parse_args()

    main(
        n_cells=args.n_cells, n_steps=args.n_steps, genome_size=args.genome_size,
    )

