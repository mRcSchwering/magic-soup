"""
Dummy run to test simulation performance in realistic environment.
"""

import statistics
import time

import magicsoup as ms
import torch
from magicsoup.examples.wood_ljungdahl import CHEMISTRY


@torch.inference_mode()
def main(map_size: int, n_steps: int, init_genome_size: int, device: str, **_):
    world = ms.World(
        chemistry=CHEMISTRY,
        time_step=1,  # 1s
        culture_kwargs={"map_size": map_size},
        tensor_kwargs={"device": device},
    )
    m2i = world.chemistry.molname_2_idx
    i_atp = m2i["ATP"]
    i_acoa = m2i["acetyl-CoA"]
    i_mfh4 = m2i["methylen-FH4"]

    # init conditions
    world.culture.molecule_map[:] = 10
    world.culture.molecule_map[i_atp] = 100
    genomes = [ms.random_genome(init_genome_size) for _ in range(1000)]
    world.spawn_cells(genomes=genomes)

    print("Starting")
    t0 = time.time()
    cells = []
    for step_i in range(n_steps):
        if step_i % 10 == 0:
            n_cells = world.cells.get_n_cells()
            cells.append(n_cells)
            print(f"step: {step_i}, {n_cells:,} cells")

            if n_cells > 12_000:
                print("Passage cells")
                world.kill_cells(mask=torch.rand(world.cells.c) > 0.3)
                world.resuspend_cells()

            world.compact(util_lt=0.5)

        # always fresh medium
        world.culture.molecule_map[:] = 10
        world.culture.molecule_map[i_atp] = 100

        # advance time
        world.advance_time()
        cell_molecules = world.cells.molecules

        # kill if ATP low
        world.kill_cells(mask=cell_molecules[:, i_atp] < 1)

        # divide if C-loaded molecules high
        acoa = cell_molecules[:, i_acoa]
        mfh4 = cell_molecules[:, i_mfh4]
        divide = (mfh4 + 2 * acoa) > 10
        rm_acoa = torch.minimum(acoa, torch.full_like(acoa, 5.0))
        rm_mfh4 = 10 - 2 * rm_acoa
        cell_molecules[:, i_acoa] -= rm_acoa
        cell_molecules[:, i_mfh4] -= rm_mfh4
        cell_molecules.clamp_(0)
        divided_idxs = world.divide_cells(mask=divide)

        # mutate on division
        world.mutate_cells(mask=divided_idxs)

    td = (time.time() - t0) / n_steps
    mu = int(statistics.mean(cells))
    std = int(statistics.stdev(cells))
    print(f"Finished: {td:.2e}s per simulated second ({mu:,} +/- {std:,} cells on average)")
