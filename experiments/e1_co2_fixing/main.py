from argparse import ArgumentParser
import time
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from magicsoup.constants import NOW
from magicsoup.examples.wood_ljungdahl import (
    CHEMISTRY,
    ATP,
    ADP,
    NADPH,
    NADP,
    HSCoA,
    acetylCoA,
    co2,
    FH4,
    NiACS,
)

_this_dir = Path(__file__).parent

# TODO: mol concentrations drop to 2.5?!
# TODO: does mol diffusion change molecule content?
# TODO: does adding/killing replicating cells change molecule content?
# TODO: do reactions change molecule content in a way that molecules are lost overall?


def sign_sample(t: torch.Tensor, k=1.0, rev=False) -> list[int]:
    p = t ** 3 / (t ** 3 + k ** 3)
    if rev:
        p = 1.0 - p
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def add_co2(world: ms.World):
    # keep CO2 on a constant high level
    if world.molecule_map[co2.idx].mean() < 10.0:
        world.molecule_map[co2.idx] += 1.0


def add_energy(world: ms.World):
    # keep energy carriers in their high energy state
    high_idxs = [ATP.idx, NADPH.idx]
    low_idxs = [ADP.idx, NADP.idx]
    world.molecule_map[high_idxs] += world.molecule_map[low_idxs] * 0.5
    world.molecule_map[low_idxs] *= 0.5


def add_random_cells(world: ms.World, s: int, n: int):
    # keep at least up to 1k cells alive
    d = n - len(world.cells)
    if d > 0:
        seqs = [ms.random_genome(s) for _ in range(d)]
        world.add_random_cells(genomes=seqs)


def kill_cells(world: ms.World):
    # kill cells with low Acetyl-CoA, and cells that never replicate
    acoa_idx = acetylCoA.idx
    acoa_idxs = sign_sample(t=world.cell_molecules[:, acoa_idx], k=1.0, rev=True)
    surv_idxs = sign_sample(t=world.cell_survival - world.cell_divisions, k=250.0)
    unq_idxs = set(acoa_idxs + surv_idxs)
    world.kill_cells(cell_idxs=list(unq_idxs))


def replicate_cells(world: ms.World):
    # cells must have enough Acetyl-CoA to replicate
    # more AcetylCoA will increase their chances of replicating
    acoa_idx = acetylCoA.idx
    coa_idx = HSCoA.idx
    idxs1 = sign_sample(t=world.cell_molecules[:, acoa_idx], k=2.5)
    idxs2 = torch.argwhere(world.cell_molecules[:, acoa_idx] > 2.0).flatten().tolist()
    idxs = list(set(idxs1) & set(idxs2))
    world.cell_molecules[idxs, acoa_idx] -= 2.0
    world.cell_molecules[idxs, coa_idx] += 2.0
    parent_child_idxs = world.replicate_cells(parent_idxs=idxs)

    genome_pairs = [
        (world.cells[p].genome, world.cells[c].genome) for p, c in parent_child_idxs
    ]
    mutated = ms.recombinations(seq_pairs=genome_pairs, p=1e-4)

    genome_idx_pairs = []
    for parent, child, idx in mutated:
        parent_i, child_i = parent_child_idxs[idx]
        genome_idx_pairs.append((parent, parent_i))
        genome_idx_pairs.append((child, child_i))
    world.update_cells(genome_idx_pairs=genome_idx_pairs)


def random_mutations(world: ms.World):
    mutated = ms.point_mutations(seqs=[d.genome for d in world.cells])
    world.update_cells(genome_idx_pairs=mutated)


def write_scalars(world: ms.World, writer: SummaryWriter, step: int, td: float):
    # fmt: off
    writer.add_scalar("Other/TimePerStep[s]", td, step)

    writer.add_scalar("Cells/total", len(world.cells), step)
    writer.add_scalar("Cells/SurvAvg", world.cell_survival.mean().item(), step)
    writer.add_scalar("Cells/SurvMax", world.cell_survival.max().item(), step)
    writer.add_scalar("Cells/DivisAvg", world.cell_divisions.mean().item(), step)
    writer.add_scalar("Cells/DivisMax", world.cell_divisions.max().item(), step)
    writer.add_scalar("Cells/ReplSurvAvg", world.cell_survival[world.cell_divisions > 0].mean().item(), step)
    writer.add_scalar("Cells/ReplSurvMax", world.cell_survival[world.cell_divisions > 0].max().item(), step)

    writer.add_scalar("MoleculeMap/ATP", world.cell_molecules[:, ATP.idx].mean(), step)
    writer.add_scalar("MoleculeMap/NADPH", world.cell_molecules[:, NADPH.idx].mean(), step)
    writer.add_scalar("MoleculeMap/CO2", world.cell_molecules[:, co2.idx].mean(), step)
    writer.add_scalar("MoleculeMap/HSCoA", world.cell_molecules[:, HSCoA.idx].mean(), step)
    writer.add_scalar("MoleculeMap/FH4", world.cell_molecules[:, FH4.idx].mean(), step)
    writer.add_scalar("MoleculeMap/NiACS", world.cell_molecules[:, NiACS.idx].mean(), step)

    writer.add_scalar("CellMolecules/CO2", world.cell_molecules[:, co2.idx].mean(), step)
    writer.add_scalar("CellMolecules/AcetylCoA", world.cell_molecules[:, acetylCoA.idx].mean(), step)
    # fmt: on


def write_images(world: ms.World, writer: SummaryWriter, step: int):
    if step % 10 == 0:
        writer.add_image("Other/Cells", world.cell_map, step, dataformats="HW")


def main(n_cells: int, n_steps: int, genome_size: int):
    writer = SummaryWriter(log_dir=_this_dir / "runs" / NOW)

    world = ms.World(chemistry=CHEMISTRY)
    world.molecule_map += 10.0
    world.molecule_map *= 10.0
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

        # world.degrade_molecules()
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
    parser.add_argument("--n_steps", default=100_000, type=int)
    parser.add_argument("--genome_size", default=500, type=int)
    args = parser.parse_args()

    main(
        n_cells=args.n_cells, n_steps=args.n_steps, genome_size=args.genome_size,
    )

