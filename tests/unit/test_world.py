import logging

import pytest
import torch
from magicsoup.cells import Cells
from magicsoup.culture import Culture
from magicsoup.examples.wood_ljungdahl import CHEMISTRY
from magicsoup.util import random_genome
from magicsoup.world import World

_ATOL = 1e-4


@pytest.fixture
def world() -> World:
    culture = Culture(chemistry=CHEMISTRY, map_size=10)
    culture.molecule_map[:] = 10.0
    return World(chemistry=CHEMISTRY, culture=culture)


def _assert_cells(
    cells: Cells,
    exp_n: int,
    exp_x: float,
    exp_gen: int,
    cell_idxs: torch.Tensor | None = None,
) -> None:
    idxs = cells.get_cell_idxs() if cell_idxs is None else cell_idxs
    n = idxs.numel()

    # assert number of cells
    assert n == exp_n
    assert cells.alive.sum() >= n
    assert cells.alive[idxs].all()

    # assert intracellular concentrations
    cellmols = cells.molecules
    assert ((cellmols[idxs] - exp_x).abs() < _ATOL).all()
    assert (cellmols[~cells.alive] == 0.0).all()

    # check generations
    assert (cells.generations[idxs] == exp_gen).all()

    if n < 1:
        return

    # check at least some non-placeholder values
    denstensors = [cells.v_max, cells.k_b, cells.k_f, cells.N_f, cells.N_b]
    for tens in denstensors:
        per_cell = (tens[idxs] != 0).sum(tuple(range(1, tens.ndim)))
        assert (per_cell > 0).sum() / n > 0.8  # a few empty proteomes possible


def _assert_culture(culture: Culture, exp_n: int, exp_x: float) -> None:
    molmap = culture.molecule_map
    cellmap = culture.cell_map
    exp_n_hat = culture.map_size**2 - exp_n
    exp_idxs = [-1] * exp_n_hat + list(range(exp_n))

    # map is occupied by cells and they picked up molecules
    assert sorted(culture.cell_map_flat.tolist()) == exp_idxs
    assert ((molmap[:, cellmap >= 0] - exp_x).abs() < _ATOL).all()
    assert (molmap[:, cellmap < 0] == 10.0).all()


def test_spawn_cells(caplog, world: World) -> None:
    # spawn 90 cells
    with caplog.at_level(logging.WARNING):
        world.spawn_cells(genomes=[random_genome() for _ in range(90)])

    # all 90 cells had space
    assert len(caplog.records) == 0

    _assert_culture(world.culture, exp_n=90, exp_x=9.99)
    _assert_cells(world.cells, exp_n=90, exp_x=9.99, exp_gen=0)

    with caplog.at_level(logging.WARNING):
        world.spawn_cells(genomes=[random_genome() for _ in range(20)])

    # 10 cells did not have space
    assert len(caplog.records) > 0

    _assert_culture(world.culture, exp_n=100, exp_x=9.99)
    _assert_cells(world.cells, exp_n=100, exp_x=9.99, exp_gen=0)


# TODO: test add cells


def test_resuspend_cells(world: World) -> None:
    # spawn 50 cells
    world.spawn_cells(genomes=[random_genome() for _ in range(50)])
    world.culture.molecule_map[:] = 10.0
    _assert_culture(world.culture, exp_n=50, exp_x=10.0)
    _assert_cells(world.cells, exp_n=50, exp_x=9.99, exp_gen=0)

    idxs0 = world.cells.get_cell_idxs().clone()
    pos0 = world.culture.get_cell_positions().clone()

    # resuspend cells
    world.resuspend_cells()

    # parameters didnt change
    _assert_culture(world.culture, exp_n=50, exp_x=10.0)
    _assert_cells(world.cells, exp_n=50, exp_x=9.99, exp_gen=0)

    idxs1 = world.cells.get_cell_idxs().clone()
    pos1 = world.culture.get_cell_positions().clone()

    # indexes are still the same but positions changed
    assert torch.equal(idxs0, idxs1)
    assert not torch.equal(pos0, pos1)


def test_migrate_cells(world: World) -> None:
    # spawn 50 cells
    world.spawn_cells(genomes=[random_genome() for _ in range(50)])
    world.culture.molecule_map[:] = 10.0
    _assert_culture(world.culture, exp_n=50, exp_x=10.0)
    _assert_cells(world.cells, exp_n=50, exp_x=9.99, exp_gen=0)

    idxs0 = world.cells.get_cell_idxs().clone()
    pos0 = world.culture.get_cell_positions().clone()

    # migrate cells
    world.migrate_cells()

    # parameters didnt change
    _assert_culture(world.culture, exp_n=50, exp_x=10.0)
    _assert_cells(world.cells, exp_n=50, exp_x=9.99, exp_gen=0)

    idxs1 = world.cells.get_cell_idxs().clone()
    pos1 = world.culture.get_cell_positions().clone()

    # indexes are still the same but positions changed
    assert torch.equal(idxs0, idxs1)
    assert not torch.equal(pos0, pos1)


@pytest.mark.parametrize(
    "n_spawn, n_divide, n_succ",
    [
        (10, 10, 10),  # all divide successfully
        (10, 5, 5),  # half divide successfully)
        (90, 90, 10),  # some cant divide
    ],
)
def test_divide_cells(world: World, n_spawn: int, n_divide: int, n_succ: int) -> None:
    n_total = n_spawn + n_succ

    # spawn 10 cells
    world.spawn_cells(genomes=[random_genome() for _ in range(n_spawn)])
    world.culture.molecule_map[:] = 10.0
    _assert_culture(world.culture, exp_n=n_spawn, exp_x=10.0)
    _assert_cells(world.cells, exp_n=n_spawn, exp_x=9.99, exp_gen=0)

    # divide cells
    all_idxs0 = world.cells.get_cell_idxs()
    divide_idxs = all_idxs0[:n_divide]
    non_divide_idxs = all_idxs0[n_divide:]
    parent_idxs, child_idxs = world.divide_cells(cell_idxs=divide_idxs)

    # some cells divided
    assert parent_idxs.shape == (n_succ,)
    assert parent_idxs.shape == child_idxs.shape
    unsucc_parent_idxs = divide_idxs[~torch.isin(divide_idxs, parent_idxs)]

    # new cells exist
    _assert_culture(world.culture, exp_n=n_total, exp_x=10.0)

    # non divided cells unchanged
    unchanged_idxs = torch.cat([non_divide_idxs, unsucc_parent_idxs])
    _assert_cells(
        world.cells,
        exp_n=unchanged_idxs.numel(),
        exp_x=9.99,
        exp_gen=0,
        cell_idxs=unchanged_idxs,
    )

    # successful descendant concentrations halfed, generation incremented
    desc_idxs = torch.cat([parent_idxs, child_idxs])
    _assert_cells(
        world.cells,
        exp_n=desc_idxs.numel(),
        exp_x=9.99 / 2,
        exp_gen=1,
        cell_idxs=desc_idxs,
    )

    # descendant cell parameters should be equal
    assert world.cells.genomes[parent_idxs] == world.cells.genomes[child_idxs]
    assert world.cells.labels[parent_idxs] == world.cells.labels[child_idxs]
    assert torch.equal(world.cells.ages[parent_idxs], world.cells.ages[child_idxs])
    assert torch.equal(world.cells.v_max[parent_idxs], world.cells.v_max[child_idxs])
    assert torch.equal(world.cells.k_f[parent_idxs], world.cells.k_f[child_idxs])
    assert torch.equal(world.cells.k_b[parent_idxs], world.cells.k_b[child_idxs])
    assert torch.equal(world.cells.K_r[parent_idxs], world.cells.K_r[child_idxs])
    assert torch.equal(world.cells.N_f[parent_idxs], world.cells.N_f[child_idxs])
    assert torch.equal(world.cells.N_b[parent_idxs], world.cells.N_b[child_idxs])
    assert torch.equal(world.cells.N_h[parent_idxs], world.cells.N_h[child_idxs])
    assert torch.equal(world.cells.N_e[parent_idxs], world.cells.N_e[child_idxs])
