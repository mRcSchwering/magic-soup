from argparse import ArgumentParser, Namespace
import time
import json
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from magicsoup.constants import NOW
from magicsoup.examples.wood_ljungdahl import CHEMISTRY as wl_chem, HSCoA, acetylCoA

THIS_DIR = Path(__file__).parent

X = ms.Molecule("X", 50.0 * 1e3)
CHEMISTRY = ms.Chemistry(
    molecules=wl_chem.molecules + [X],
    reactions=wl_chem.reactions + [([acetylCoA], [HSCoA, X])],
)


def replicate_sample(t: torch.Tensor, k: float) -> list[int]:
    p = t ** 3 / (t ** 3 + k ** 3)
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def kill_sample(t: torch.Tensor, k: float) -> list[int]:
    p = k / (t + k)
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def init_map(world: ms.World, x_idx: int, co2_idx: int):
    init_molecules = torch.randn(world.molecule_map.size()) + 10.0
    world.molecule_map = init_molecules.clamp(min=1.0)
    world.molecule_map[x_idx] = 1.0

    n = int(world.map_size / 2)
    map_ones = torch.ones((world.map_size, world.map_size))
    gradient = torch.cat([torch.linspace(1.0, 100.0, n), torch.linspace(100.0, 1.0, n)])
    world.molecule_map[co2_idx] = torch.einsum("xy,x->xy", map_ones, gradient)


def add_co2(world: ms.World, co2_idx: int):
    n = int(world.map_size / 2)
    world.molecule_map[co2_idx, [n - 1, n]] = 100.0
    world.molecule_map[co2_idx, [0, -1]] = 1.0


def add_energy(
    world: ms.World, atp_idx: int, adp_idx: int, nadph_idx: int, nadp_idx: int
):
    for high, low in [(atp_idx, adp_idx), (nadph_idx, nadp_idx)]:
        high_avg = world.molecule_map[high].mean().item()
        low_avg = world.molecule_map[low].mean().item()
        if high_avg / low_avg < 0.6:
            world.molecule_map[high] += world.molecule_map[low] * 0.2
            world.molecule_map[low] *= 0.8


def add_random_cells(world: ms.World, s: int, n: int):
    d = n - len(world.cells)
    if d > 0:
        seqs = [ms.random_genome(s) for _ in range(d)]
        world.add_random_cells(genomes=seqs)


def kill_cells(world: ms.World, aca_idx: int):
    idxs = kill_sample(t=world.cell_molecules[:, aca_idx], k=0.001)
    world.kill_cells(cell_idxs=idxs)


def replicate_cells(world: ms.World, x_idx: int):
    idxs1 = replicate_sample(t=world.cell_molecules[:, x_idx], k=15.0)

    # successful cells will share their n(X), which will then be reduced by 1.0
    # so a cell must have n(X)>=2.0 to replicate
    idxs2 = torch.argwhere(world.cell_molecules[:, x_idx] > 2.5).flatten().tolist()
    idxs = list(set(idxs1) & set(idxs2))

    replicated = world.replicate_cells(parent_idxs=idxs)
    if len(replicated) == 0:
        return

    # these cells have successfully divided and shared n(X)
    parents, children = list(map(list, zip(*replicated)))
    world.cell_molecules[parents + children, x_idx] -= 1.0

    # add random recombination
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


def write_scalars(
    world: ms.World,
    writer: SummaryWriter,
    step: int,
    dtime: float,
    x_idx: int,
    aca_idx: int,
    co2_idx: int,
):
    n_cells = len(world.cells)
    if n_cells > 0:
        writer.add_scalar("Cells/total[n]", n_cells, step)
        writer.add_scalar("Cells/Survival[avg]", world.cell_survival.mean(), step)
        writer.add_scalar("Cells/Divisions[avg]", world.cell_divisions.mean(), step)

    writer.add_scalar("Map/X[avg]", world.molecule_map[x_idx].mean(), step)
    writer.add_scalar("Map/Acetyl-CoA[avg]", world.molecule_map[aca_idx].mean(), step)
    writer.add_scalar("Map/CO2[avg]", world.molecule_map[co2_idx].mean(), step)

    writer.add_scalar("Other/TimePerStep[s]", dtime, step)


def write_images(world: ms.World, writer: SummaryWriter, step: int):
    if step % 10 == 0:
        writer.add_image("Maps/Cells", world.cell_map, step, dataformats="HW")


def save_state(world: ms.World, step: int, n: int):
    if len(world.cells) >= n and step % 100 == 0:
        print(f"Finished step {step:,}")
        world.save_state(statedir=THIS_DIR / "runs" / NOW / f"step={step}")
        return
    if step % 1000 == 0:
        world.save_state(statedir=THIS_DIR / "runs" / NOW / f"step={step}")


def main(args: Namespace):
    mol_2_idx = {d.name: i for i, d in enumerate(CHEMISTRY.molecules)}
    CO2_IDX = mol_2_idx["CO2"]
    ATP_IDX = mol_2_idx["ATP"]
    ADP_IDX = mol_2_idx["ADP"]
    NADPH_IDX = mol_2_idx["NADPH"]
    NADP_IDX = mol_2_idx["NADP"]
    ACA_IDX = mol_2_idx["acetyl-CoA"]
    X_IDX = mol_2_idx["X"]

    writer = SummaryWriter(log_dir=THIS_DIR / "runs" / NOW)
    with open(THIS_DIR / "runs" / NOW / "hparams.json", "w") as fh:
        json.dump(vars(args), fh)

    world = ms.World(chemistry=CHEMISTRY, map_size=args.map_size)
    world.save(rundir=THIS_DIR / "runs" / NOW)
    init_map(world=world, co2_idx=CO2_IDX, x_idx=X_IDX)

    for step_i in range(args.n_steps):
        t0 = time.time()

        add_co2(world=world, co2_idx=CO2_IDX)
        add_energy(
            world=world,
            atp_idx=ATP_IDX,
            adp_idx=ADP_IDX,
            nadph_idx=NADPH_IDX,
            nadp_idx=NADP_IDX,
        )

        add_random_cells(world=world, s=args.genome_size, n=args.n_cells)
        world.enzymatic_activity()

        kill_cells(world=world, aca_idx=ACA_IDX)
        replicate_cells(world=world, x_idx=X_IDX)
        random_mutations(world=world)

        world.degrade_molecules()
        world.diffuse_molecules()
        world.increment_cell_survival()
        save_state(world=world, step=step_i, n=2 * args.n_cells)

        write_scalars(
            world=world,
            writer=writer,
            step=step_i,
            dtime=time.time() - t0,
            x_idx=X_IDX,
            co2_idx=CO2_IDX,
            aca_idx=ACA_IDX,
        )

        write_images(world=world, writer=writer, step=step_i)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_cells", default=1_000, type=int)
    parser.add_argument("--n_steps", default=100_000, type=int)
    parser.add_argument("--genome_size", default=300, type=int)
    parser.add_argument("--map_size", default=128, type=int)
    args = parser.parse_args()
    main(args)

