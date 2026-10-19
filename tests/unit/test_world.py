import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
from magicsoup.cells import Cells
from magicsoup.culture import Culture
from magicsoup.examples.wood_ljungdahl import CHEMISTRY
from magicsoup.util import Array, random_genome
from magicsoup.world import World

_ATOL = 1e-4
_RTOL = 1e-4


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
        assert (per_cell > 0).sum() / n > 0.7  # a few empty proteomes possible


def _assert_culture(culture: Culture, exp_n: int, exp_x: float | None = None) -> None:
    molmap = culture.molecule_map
    cellmap = culture.cell_map
    exp_n_hat = culture.map_size**2 - exp_n
    exp_idxs = [-1] * exp_n_hat + list(range(exp_n))

    # map is occupied by cells and they picked up molecules
    assert sorted(culture.cell_map_flat.tolist()) == exp_idxs
    if exp_x is not None:
        assert ((molmap[:, cellmap >= 0] - exp_x).abs() < _ATOL).all()
        assert (molmap[:, cellmap < 0] == 10.0).all()


def _get_mol_mass(world: World) -> torch.Tensor:
    return (
        world.culture.molecule_map.sum() * world.v_voxel
        + world.cells.molecules.sum() * world.v_cell
    )


@contextmanager
def _assert_mol_mass(world: World) -> Iterator:
    before = _get_mol_mass(world)
    yield
    after = _get_mol_mass(world)
    torch.testing.assert_close(before, after, atol=_ATOL, rtol=_RTOL)


@contextmanager
def _assert_genomes_proteomes_unchanged(
    world: World, cell_idxs: torch.Tensor | None = None
) -> Iterator:
    cell_idxs = world.cells.get_cell_idxs() if cell_idxs is None else cell_idxs
    genomes0 = world.cells.genomes[cell_idxs]
    v_max0 = world.cells.v_max[cell_idxs].clone()
    k_f0 = world.cells.k_f[cell_idxs].clone()
    k_b0 = world.cells.k_b[cell_idxs].clone()
    K_r0 = world.cells.K_r[cell_idxs].clone()
    N_f0 = world.cells.N_f[cell_idxs].clone()
    N_b0 = world.cells.N_b[cell_idxs].clone()
    N_h0 = world.cells.N_h[cell_idxs].clone()
    N_e0 = world.cells.N_e[cell_idxs].clone()

    yield

    genomes1 = world.cells.genomes[cell_idxs]
    v_max1 = world.cells.v_max[cell_idxs].clone()
    k_f1 = world.cells.k_f[cell_idxs].clone()
    k_b1 = world.cells.k_b[cell_idxs].clone()
    K_r1 = world.cells.K_r[cell_idxs].clone()
    N_f1 = world.cells.N_f[cell_idxs].clone()
    N_b1 = world.cells.N_b[cell_idxs].clone()
    N_h1 = world.cells.N_h[cell_idxs].clone()
    N_e1 = world.cells.N_e[cell_idxs].clone()

    assert genomes0 == genomes1
    assert torch.equal(v_max0, v_max1)
    assert torch.equal(k_f0, k_f1)
    assert torch.equal(k_b0, k_b1)
    assert torch.equal(K_r0, K_r1)
    assert torch.equal(N_f0, N_f1)
    assert torch.equal(N_b0, N_b1)
    assert torch.equal(N_h0, N_h1)
    assert torch.equal(N_e0, N_e1)


def test_spawn_cells(caplog, world: World) -> None:
    # spawn 90 cells
    with caplog.at_level(logging.WARNING), _assert_mol_mass(world):
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


@pytest.mark.parametrize(
    "n_spawn, n_resp",
    [
        (10, 10),  # resuspend all
        (100, 100),  # resuspend all
        (50, 25),  # resuspend half
        (50, 0),  # resuspend none
    ],
)
def test_resuspend_cells(world: World, n_spawn: int, n_resp: int) -> None:
    # spawn cells
    world.spawn_cells(genomes=[random_genome() for _ in range(n_spawn)])
    world.culture.molecule_map[:] = 10.0
    _assert_culture(world.culture, exp_n=n_spawn, exp_x=10.0)
    _assert_cells(world.cells, exp_n=n_spawn, exp_x=9.99, exp_gen=0)

    idxs0 = world.cells.get_cell_idxs().clone()
    pos0 = world.culture.get_cell_positions().clone()

    # resuspend cells
    susp_idxs = idxs0[:n_resp]
    non_susp_idxs = idxs0[n_resp:]
    with _assert_mol_mass(world):
        world.resuspend_cells(cell_idxs=susp_idxs)

    # parameters didnt change
    _assert_culture(world.culture, exp_n=n_spawn, exp_x=10.0)
    _assert_cells(world.cells, exp_n=n_spawn, exp_x=9.99, exp_gen=0)

    idxs1 = world.cells.get_cell_idxs().clone()
    pos1 = world.culture.get_cell_positions().clone()

    # indexes are still the same but positions changed (by chance a few could be the same)
    assert torch.equal(idxs0, idxs1)
    assert torch.equal(pos0[non_susp_idxs], pos1[non_susp_idxs])
    if n_resp > 0:
        assert ((pos0[susp_idxs] != pos1[susp_idxs]).sum(1) > 0).sum() > 0.9 * n_resp


@pytest.mark.parametrize(
    "n_spawn, n_migrate, n_succ",
    [
        (10, 10, 10),  # all migrate successfully
        (10, 5, 5),  # half migrate successfully
        (90, 90, 30),  # only a few can migrate (10 for each round)
        (50, 0, 0),  # none want to migrate
    ],
)
def test_migrate_cells(world: World, n_spawn: int, n_migrate: int, n_succ: int) -> None:
    # spawn cells
    world.spawn_cells(genomes=[random_genome() for _ in range(n_spawn)])
    world.culture.molecule_map[:] = 10.0
    _assert_culture(world.culture, exp_n=n_spawn, exp_x=10.0)
    _assert_cells(world.cells, exp_n=n_spawn, exp_x=9.99, exp_gen=0)

    idxs0 = world.cells.get_cell_idxs().clone()
    pos0 = world.culture.get_cell_positions().clone()

    # migrate cells
    migrate_idxs = idxs0[:n_migrate]
    non_migrate_idxs = idxs0[n_migrate:]
    with _assert_mol_mass(world):
        scc_idx = world.migrate_cells(cell_idxs=migrate_idxs)

    assert scc_idx.numel() == n_succ

    # parameters didnt change
    _assert_culture(world.culture, exp_n=n_spawn, exp_x=10.0)
    _assert_cells(world.cells, exp_n=n_spawn, exp_x=9.99, exp_gen=0)

    idxs1 = world.cells.get_cell_idxs().clone()
    pos1 = world.culture.get_cell_positions().clone()

    # indexes are still the same but positions changed
    assert torch.equal(idxs0, idxs1)
    assert torch.equal(pos0[non_migrate_idxs], pos1[non_migrate_idxs])
    assert ((pos0[scc_idx] != pos1[scc_idx]).sum(1) > 0).all()


@pytest.mark.parametrize(
    "n_spawn, n_divide, n_succ",
    [
        (10, 10, 10),  # all divide successfully
        (10, 5, 5),  # half divide successfully)
        (90, 90, 10),  # some cant divide
        (50, 0, 0),  # nobody wants to divide
    ],
)
def test_divide_cells(world: World, n_spawn: int, n_divide: int, n_succ: int) -> None:
    n_total = n_spawn + n_succ

    # spawn cells
    world.spawn_cells(genomes=[random_genome() for _ in range(n_spawn)])
    world.culture.molecule_map[:] = 10.0
    _assert_culture(world.culture, exp_n=n_spawn, exp_x=10.0)
    _assert_cells(world.cells, exp_n=n_spawn, exp_x=9.99, exp_gen=0)

    # divide cells
    all_idxs0 = world.cells.get_cell_idxs()
    divide_idxs = all_idxs0[:n_divide]
    non_divide_idxs = all_idxs0[n_divide:]
    with _assert_mol_mass(world):
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


@pytest.mark.parametrize(
    "n_spawn, n_update",
    [
        (50, 50),  # update all
        (50, 25),  # update some
        (50, 0),  # update none
    ],
)
def test_update_cells(world: World, n_spawn: int, n_update: int) -> None:
    # spawn cells
    world.spawn_cells(genomes=[random_genome() for _ in range(n_spawn)])
    world.culture.molecule_map[:] = 10.0
    _assert_culture(world.culture, exp_n=n_spawn, exp_x=10.0)
    _assert_cells(world.cells, exp_n=n_spawn, exp_x=9.99, exp_gen=0)

    # remember before
    idxs0 = world.cells.get_cell_idxs().clone()
    pos0 = world.culture.get_cell_positions().clone()
    genomes0 = Array([d for d in world.cells.genomes.items])
    labels0 = Array([d for d in world.cells.labels.items])
    ages0 = world.cells.ages.clone()
    v_max0 = world.cells.v_max.clone()
    k_f0 = world.cells.k_f.clone()
    k_b0 = world.cells.k_b.clone()
    K_r0 = world.cells.K_r.clone()
    N_f0 = world.cells.N_f.clone()
    N_b0 = world.cells.N_b.clone()
    N_h0 = world.cells.N_h.clone()
    N_e0 = world.cells.N_e.clone()
    p0 = world.cells.p

    # update cells
    all_idxs0 = world.cells.get_cell_idxs()
    update_idxs = all_idxs0[:n_update]
    no_update_idxs = all_idxs0[n_update:]
    new_genomes = [random_genome() for _ in range(n_update)]
    with _assert_mol_mass(world):
        world.update_cells(genomes=new_genomes, cell_idxs=update_idxs)

    # remember after
    # remember before
    idxs1 = world.cells.get_cell_idxs().clone()
    pos1 = world.culture.get_cell_positions().clone()
    genomes1 = Array([d for d in world.cells.genomes.items])
    labels1 = Array([d for d in world.cells.labels.items])
    ages1 = world.cells.ages.clone()
    v_max1 = world.cells.v_max.clone()
    k_f1 = world.cells.k_f.clone()
    k_b1 = world.cells.k_b.clone()
    K_r1 = world.cells.K_r.clone()
    N_f1 = world.cells.N_f.clone()
    N_b1 = world.cells.N_b.clone()
    N_h1 = world.cells.N_h.clone()
    N_e1 = world.cells.N_e.clone()
    p1 = world.cells.p

    # cells and cell culture didn't change
    _assert_culture(world.culture, exp_n=n_spawn, exp_x=10.0)
    _assert_cells(world.cells, exp_n=n_spawn, exp_x=9.99, exp_gen=0)
    assert torch.equal(idxs0, idxs1)
    assert torch.equal(pos0, pos1)
    assert labels0.items == labels1.items
    assert torch.equal(ages0, ages1)

    # p could have changed
    p = min(p0, p1)

    # genomes and proteomes of unchanged cells didnt change
    assert genomes0[no_update_idxs] == genomes1[no_update_idxs]
    assert torch.equal(v_max0[no_update_idxs, :p], v_max1[no_update_idxs, :p])
    assert torch.equal(k_f0[no_update_idxs, :p], k_f1[no_update_idxs, :p])
    assert torch.equal(k_b0[no_update_idxs, :p], k_b1[no_update_idxs, :p])
    assert torch.equal(K_r0[no_update_idxs, :p], K_r1[no_update_idxs, :p])
    assert torch.equal(N_f0[no_update_idxs, :p], N_f1[no_update_idxs, :p])
    assert torch.equal(N_b0[no_update_idxs, :p], N_b1[no_update_idxs, :p])
    assert torch.equal(N_h0[no_update_idxs, :p], N_h1[no_update_idxs, :p])
    assert torch.equal(N_e0[no_update_idxs, :p], N_e1[no_update_idxs, :p])

    # genomes and proteomes of updated cells changed
    if n_update > 0:
        assert genomes0[update_idxs] != genomes1[update_idxs]
        assert not torch.equal(v_max0[update_idxs, :p], v_max1[update_idxs, :p])
        assert not torch.equal(k_f0[update_idxs, :p], k_f1[update_idxs, :p])
        assert not torch.equal(k_b0[update_idxs, :p], k_b1[update_idxs, :p])
        assert not torch.equal(K_r0[update_idxs, :p], K_r1[update_idxs, :p])
        assert not torch.equal(N_f0[update_idxs, :p], N_f1[update_idxs, :p])
        assert not torch.equal(N_b0[update_idxs, :p], N_b1[update_idxs, :p])
        assert not torch.equal(N_h0[update_idxs, :p], N_h1[update_idxs, :p])
        assert not torch.equal(N_e0[update_idxs, :p], N_e1[update_idxs, :p])


@pytest.mark.parametrize(
    "n_spawn, n_kill",
    [
        (50, 50),  # kill all
        (50, 25),  # kill some
        (50, 0),  # kill none
    ],
)
def test_kill_cells(world: World, n_spawn: int, n_kill: int) -> None:
    n_remain = n_spawn - n_kill

    # spawn cells
    world.spawn_cells(genomes=[random_genome() for _ in range(n_spawn)])
    world.culture.molecule_map[:] = 10.0
    world.cells.ages[:] = 1.0
    world.cells.generations[:] = 1
    _assert_culture(world.culture, exp_n=n_spawn, exp_x=10.0)
    _assert_cells(world.cells, exp_n=n_spawn, exp_x=9.99, exp_gen=1)

    # remember before
    idxs0 = world.cells.get_cell_idxs().clone()
    pos0 = world.culture.get_cell_positions().clone()
    genomes0 = Array([d for d in world.cells.genomes.items])
    labels0 = Array([d for d in world.cells.labels.items])
    ages0 = world.cells.ages.clone()
    v_max0 = world.cells.v_max.clone()
    k_f0 = world.cells.k_f.clone()
    k_b0 = world.cells.k_b.clone()
    K_r0 = world.cells.K_r.clone()
    N_f0 = world.cells.N_f.clone()
    N_b0 = world.cells.N_b.clone()
    N_h0 = world.cells.N_h.clone()
    N_e0 = world.cells.N_e.clone()

    # kill cells
    kill_idxs = idxs0[n_remain : n_remain + n_kill]
    no_kill_idxs = idxs0[:n_remain]
    with _assert_mol_mass(world):
        world.kill_cells(cell_idxs=kill_idxs)

    # remember after
    idxs1 = world.cells.get_cell_idxs().clone()
    pos1 = world.culture.get_cell_positions().clone()
    genomes1 = Array([d for d in world.cells.genomes.items])
    labels1 = Array([d for d in world.cells.labels.items])
    ages1 = world.cells.ages.clone()
    gens1 = world.cells.generations.clone()
    v_max1 = world.cells.v_max.clone()
    k_f1 = world.cells.k_f.clone()
    k_b1 = world.cells.k_b.clone()
    K_r1 = world.cells.K_r.clone()
    N_f1 = world.cells.N_f.clone()
    N_b1 = world.cells.N_b.clone()
    N_h1 = world.cells.N_h.clone()
    N_e1 = world.cells.N_e.clone()

    # some cells should have died
    assert world.cells.get_n_cells() == n_remain
    assert set(idxs0.tolist()) == set(idxs1.tolist() + kill_idxs.tolist())

    # surviving cells are the same
    _assert_culture(world.culture, exp_n=n_remain, exp_x=None)
    _assert_cells(world.cells, exp_n=n_remain, exp_x=9.99, exp_gen=1)

    # unchanged cells didn't change
    assert torch.equal(pos0[no_kill_idxs], pos1[no_kill_idxs])
    assert torch.equal(pos0[no_kill_idxs], pos1[no_kill_idxs])

    # genomes and proteomes of unchanged cells didnt change
    assert genomes0[no_kill_idxs] == genomes1[no_kill_idxs]
    assert labels0[no_kill_idxs] == labels1[no_kill_idxs]
    assert torch.equal(ages0[no_kill_idxs], ages1[no_kill_idxs])
    assert torch.equal(v_max0[no_kill_idxs], v_max1[no_kill_idxs])
    assert torch.equal(k_f0[no_kill_idxs], k_f1[no_kill_idxs])
    assert torch.equal(k_b0[no_kill_idxs], k_b1[no_kill_idxs])
    assert torch.equal(K_r0[no_kill_idxs], K_r1[no_kill_idxs])
    assert torch.equal(N_f0[no_kill_idxs], N_f1[no_kill_idxs])
    assert torch.equal(N_b0[no_kill_idxs], N_b1[no_kill_idxs])
    assert torch.equal(N_h0[no_kill_idxs], N_h1[no_kill_idxs])
    assert torch.equal(N_e0[no_kill_idxs], N_e1[no_kill_idxs])

    # killed cells parameters back to defaults
    if n_kill > 0:
        assert genomes1[kill_idxs] == [""] * n_kill
        assert labels1[kill_idxs] == [""] * n_kill
        assert (gens1[kill_idxs] == 0).all()
        assert (ages1[kill_idxs] == 0.0).all()
        assert (v_max1[kill_idxs] == 0.0).all()
        assert (k_f1[kill_idxs] == 0.0).all()
        assert (k_b1[kill_idxs] == 0.0).all()
        assert (K_r1[kill_idxs] == 0.0).all()
        assert (N_f1[kill_idxs] == 0).all()
        assert (N_b1[kill_idxs] == 0).all()
        assert (N_h1[kill_idxs] == 0).all()
        assert (N_e1[kill_idxs] == 0.0).all()


@pytest.mark.parametrize(
    "n_spawn",
    [
        100,  # full
        50,  # 50% confl
        0,  # empty
    ],
)
def test_advance_time(world: World, n_spawn: int) -> None:
    # spawn cells
    world.spawn_cells(genomes=[random_genome() for _ in range(n_spawn)])
    world.culture.molecule_map[:] = 10.0
    world.cells.ages[:] = 1.0
    world.cells.generations[:] = 1

    _assert_culture(world.culture, exp_n=n_spawn, exp_x=10.0)
    _assert_cells(world.cells, exp_n=n_spawn, exp_x=9.99, exp_gen=1)

    # advance time
    with _assert_genomes_proteomes_unchanged(world):
        world.advance_time()
