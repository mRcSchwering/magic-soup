from argparse import ArgumentParser, Namespace
import time
import json
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import magicsoup as ms
from magicsoup.constants import NOW
from magicsoup.examples.wood_ljungdahl import CHEMISTRY

THIS_DIR = Path(__file__).parent


def replicate_sample(t: torch.Tensor, k=15.0) -> list[int]:
    p = t ** 5 / (t ** 5 + k ** 5)
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def kill_sample(t: torch.Tensor, k=0.5) -> list[int]:
    p = k ** 4 / (t ** 4 + k ** 4)
    idxs = torch.argwhere(torch.bernoulli(p))
    return idxs.flatten().tolist()


def init_map(world: ms.World, co2_idx: int, add_mol_idxs: list[int]):
    world.molecule_map += 1.5
    world.molecule_map = world.molecule_map.clamp(min=0.5)
    world.molecule_map[add_mol_idxs] += 10.0

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
        if high_avg / (low_avg + 1e-4) < 5.0:
            world.molecule_map[high] += world.molecule_map[low] * 0.99
            world.molecule_map[low] *= 0.01


def add_random_cells(world: ms.World, s: int, n: int):
    d = n - len(world.cells)
    if d > 0:
        seqs = [ms.random_genome(s) for _ in range(d)]
        world.add_random_cells(genomes=seqs)


def kill_cells(world: ms.World, nadph_idx: int, atp_idx: int):
    low_atp = kill_sample(t=world.cell_molecules[:, atp_idx])
    low_nadph = kill_sample(t=world.cell_molecules[:, nadph_idx])
    idxs = list(set(low_atp + low_nadph))
    world.kill_cells(cell_idxs=idxs)


def replicate_cells(world: ms.World, aca_idx: int):
    idxs1 = replicate_sample(t=world.cell_molecules[:, aca_idx])

    # successful cells will share their n(X), which will then be reduced by 1.0
    # so a cell must have n(X)>=2.0 to replicate
    idxs2 = torch.argwhere(world.cell_molecules[:, aca_idx] > 2.2).flatten().tolist()
    idxs = list(set(idxs1) & set(idxs2))

    replicated = world.replicate_cells(parent_idxs=idxs)
    if len(replicated) == 0:
        return

    # these cells have successfully divided and shared n(X)
    parents, children = list(map(list, zip(*replicated)))
    world.cell_molecules[parents + children, aca_idx] -= 1.0

    # add random recombinations
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
    mol_name_idx_list: list[tuple[str, int]],
):
    if step % 10 != 0:
        return

    n_cells = len(world.cells)
    molecules = {f"Molecules/{s}": i for s, i in mol_name_idx_list}

    if n_cells == 0:
        for scalar, idx in molecules.items():
            writer.add_scalar(scalar, world.molecule_map[idx].mean().item(), step)
    else:
        writer.add_scalar("Cells/total[n]", n_cells, step)
        cell_surv = world.cell_survival.float()
        writer.add_scalar("Cells/Survival[avg]", cell_surv.mean(), step)
        cell_divis = world.cell_divisions.float()
        writer.add_scalar("Cells/Divisions[avg]", cell_divis.mean(), step)

        n = world.map_size ** 2 + n_cells
        for scalar, idx in molecules.items():
            mm = world.molecule_map[idx].sum().item()
            cm = world.cell_molecules[:, idx].sum().item()
            writer.add_scalar(scalar, (mm + cm) / n, step)

    writer.add_scalar("Other/TimePerStep[s]", dtime, step)


def write_images(world: ms.World, writer: SummaryWriter, step: int):
    if step % 100 == 0:
        writer.add_image("Maps/Cells", world.cell_map, step, dataformats="WH")


def save_state(world: ms.World, step: int, n: int):
    if len(world.cells) >= n and step % 100 == 0:
        print(f"Finished step {step:,}")
        world.save_state(statedir=THIS_DIR / "runs" / NOW / f"step={step}")
        return
    if step % 1000 == 0:
        print(f"Finished step {step:,}")
        world.save_state(statedir=THIS_DIR / "runs" / NOW / f"step={step}")


def main(args: Namespace):
    mol_2_idx = {d.name: i for i, d in enumerate(CHEMISTRY.molecules)}
    CO2_IDX = mol_2_idx["CO2"]
    ATP_IDX = mol_2_idx["ATP"]
    ADP_IDX = mol_2_idx["ADP"]
    NADPH_IDX = mol_2_idx["NADPH"]
    NADP_IDX = mol_2_idx["NADP"]
    ACA_IDX = mol_2_idx["acetyl-CoA"]
    HCA_IDX = mol_2_idx["HS-CoA"]
    ACS_IDX = mol_2_idx["Ni-ACS"]
    FH4_IDX = mol_2_idx["FH4"]

    observe_molecules = ["CO2", "ATP", "NADPH", "FH4", "HS-CoA", "Ni-ACS"]
    mol_name_idx_list = [(d, mol_2_idx[d]) for d in observe_molecules]

    writer = SummaryWriter(log_dir=THIS_DIR / "runs" / NOW)
    with open(THIS_DIR / "runs" / NOW / "hparams.json", "w") as fh:
        json.dump(vars(args), fh)

    world = ms.World(chemistry=CHEMISTRY, map_size=args.map_size, mol_map_init="zeros")
    world.save(rundir=THIS_DIR / "runs" / NOW)
    init_map(
        world=world,
        co2_idx=CO2_IDX,
        add_mol_idxs=[ATP_IDX, NADPH_IDX, FH4_IDX, HCA_IDX, ACS_IDX],
    )

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

        kill_cells(world=world, nadph_idx=NADPH_IDX, atp_idx=ATP_IDX)
        replicate_cells(world=world, aca_idx=ACA_IDX)
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
            mol_name_idx_list=mol_name_idx_list,
        )

        write_images(world=world, writer=writer, step=step_i)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_cells", default=1_000, type=int)
    parser.add_argument("--n_steps", default=100_000, type=int)
    parser.add_argument("--genome_size", default=500, type=int)
    parser.add_argument("--map_size", default=128, type=int)
    args = parser.parse_args()
    main(args)

