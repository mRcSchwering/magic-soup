import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from magicsoup.cells import Cells
from magicsoup.culture import Culture
from magicsoup.examples.wood_ljungdahl import CHEMISTRY
from magicsoup.util import random_genome
from magicsoup.world import World

_ATOL = 1e-4
_RTOL = 1e-4


@pytest.fixture
def world() -> World:
    culture = Culture(chemistry=CHEMISTRY, map_size=10)
    culture.molecule_map[:] = 10.0
    return World(chemistry=CHEMISTRY, culture=culture)


def _spawn_cells(world: World, n: int, x: float = 10.0) -> World:
    world.spawn_cells(genomes=[random_genome() for _ in range(n)])
    world.culture.molecule_map[:] = x
    world.cells.molecules[world.cells.alive] = x
    world.cells.ages[:] = 1.0
    world.cells.generations[:] = 1
    return world


def _get_mol_mass(world: World) -> torch.Tensor:
    return (
        world.culture.molecule_map.sum() * world.v_voxel
        + world.cells.molecules.sum() * world.v_cell
    )


def _get_cell_arrays(cells: Cells) -> dict[str, np.ndarray]:
    return {
        "genomes": cells.genomes.copy(),
        "lineages": cells.lineages.copy(),
    }


def _get_cell_tensors(cells: Cells) -> dict[str, torch.Tensor]:
    return {
        "alive": cells.alive.clone(),
        "ages": cells.ages.clone(),
        "generations": cells.generations.clone(),
        "molecules": cells.molecules.clone(),
    }


def _get_proteome_tensors(cells: Cells) -> dict[str, torch.Tensor]:
    return {
        "v_max": cells.v_max.clone(),
        "k_f": cells.k_f.clone(),
        "k_b": cells.k_b.clone(),
        "K_r": cells.K_r.clone(),
        "N_f": cells.N_f.clone(),
        "N_b": cells.N_b.clone(),
        "N_h": cells.N_h.clone(),
        "N_e": cells.N_e.clone(),
    }


def _get_culture_tensors(culture: Culture) -> dict[str, torch.Tensor]:
    return {
        "cell_map": culture.cell_map.clone(),
        "molecule_map": culture.molecule_map.clone(),
    }


@dataclass
class _State:
    p: int
    idx: torch.Tensor
    pos: torch.Tensor
    mass: torch.Tensor
    cell_tensors: dict[str, torch.Tensor]
    cell_arrays: dict[str, np.ndarray]
    proteome_tensors: dict[str, torch.Tensor]
    culture_tensors: dict[str, torch.Tensor]

    @classmethod
    def from_world(cls, world: World) -> "_State":
        return cls(
            p=world.cells.p,
            idx=world.cells.get_cell_idxs().clone(),
            pos=world.culture.get_cell_positions().clone(),
            mass=_get_mol_mass(world),
            cell_tensors=_get_cell_tensors(world.cells),
            cell_arrays=_get_cell_arrays(world.cells),
            proteome_tensors=_get_proteome_tensors(world.cells),
            culture_tensors=_get_culture_tensors(world.culture),
        )


@contextmanager
def _capture_state_change(world: World) -> Iterator[dict[str, _State]]:
    out = {}
    out["t0"] = _State.from_world(world)
    yield out
    out["t1"] = _State.from_world(world)


@pytest.mark.parametrize(
    "n_spawn, exp_n_spawn",
    [
        (0, 0),  # spawn none
        (50, 50),  # to 100% confl
        (60, 50),  # 10 won't have space
    ],
)
def test_spawn_cells(caplog, world: World, n_spawn: int, exp_n_spawn: int) -> None:
    _spawn_cells(world, n=50)

    # spawn cells
    with caplog.at_level(logging.WARNING), _capture_state_change(world) as state:
        world.spawn_cells(genomes=[random_genome() for _ in range(n_spawn)])

    t0 = state["t0"]
    t1 = state["t1"]

    # check warning
    if exp_n_spawn != n_spawn:
        assert len(caplog.records) > 0
    else:
        assert len(caplog.records) == 0

    # new cells were created
    assert t0.idx.numel() + exp_n_spawn == t1.idx.numel()
    new_idx = t1.idx[~torch.isin(t1.idx, t0.idx)]

    # mass is conserved
    torch.testing.assert_close(t0.mass, t1.mass, atol=_ATOL, rtol=_RTOL)

    # old cells didn't change
    assert torch.equal(t0.pos[t0.idx], t1.pos[t0.idx])
    p = min(t0.p, t1.p) - 1

    for key in t0.cell_tensors:
        ten0 = t0.cell_tensors[key][t0.idx]
        ten1 = t1.cell_tensors[key][t0.idx]
        assert torch.equal(ten0, ten1), key

    for key in t0.proteome_tensors:
        ten0 = t0.proteome_tensors[key][t0.idx, p]
        ten1 = t1.proteome_tensors[key][t0.idx, p]
        assert torch.equal(ten0, ten1), key

    for key in t0.cell_arrays:
        arr0 = t0.cell_arrays[key][t0.idx]
        arr1 = t1.cell_arrays[key][t0.idx]
        assert np.all(arr0 == arr1), key

    if n_spawn == 0:
        return

    # new cells are sound
    assert t1.cell_tensors["alive"][new_idx].all()
    assert (t1.cell_tensors["ages"][new_idx] == 0.0).all()
    assert (t1.cell_tensors["generations"][new_idx] == 0).all()
    assert (t1.proteome_tensors["v_max"][new_idx] != 0).any()
    assert all(len(d) > 0 for d in t1.cell_arrays["genomes"][new_idx])
    assert all(len(d) > 0 for d in t1.cell_arrays["lineages"][new_idx])

    # molecules were picked up
    molmap = world.culture.molecule_map
    cellmap = world.culture.cell_map
    cellmols = world.cells.molecules
    xs = t1.pos[new_idx, 0]
    ys = t1.pos[new_idx, 1]
    assert ((molmap[:, xs, ys] - 9.99).abs()).max() < _ATOL
    assert (cellmols[new_idx] - 9.99).abs().max() < _ATOL
    assert set(cellmap[xs, ys].tolist()) == set(new_idx.tolist())


def test_get_cell(world: World) -> None:
    _spawn_cells(world, n=50)

    pos = world.culture.get_cell_positions()

    c0 = world.get_cell(by_idx=25)
    assert c0.lineage == world.cells.lineages[25]
    assert c0.genome == world.cells.genomes[25]
    assert torch.equal(c0.molecules, world.cells.molecules[25])
    assert len(c0.proteome) > 0

    c1 = world.get_cell(by_position=(int(pos[20, 0]), int(pos[20, 1])))
    assert c1.lineage == world.cells.lineages[20]
    assert c1.genome == world.cells.genomes[20]
    assert torch.equal(c1.molecules, world.cells.molecules[20])
    assert len(c1.proteome) > 0


@pytest.mark.parametrize(
    "n_add, exp_n_add",
    [
        (0, 0),  # add none
        (50, 50),  # to 100% confl
        (60, 50),  # 10 won't have space
    ],
)
def test_add_cells(caplog, world: World, n_add: int, exp_n_add: int) -> None:
    _spawn_cells(world, n=50)

    # get clones of the first cell
    cell = world.get_cell(by_idx=0)

    # spawn cells
    with caplog.at_level(logging.WARNING), _capture_state_change(world) as state:
        world.add_cells(cells=[cell] * n_add)

    t0 = state["t0"]
    t1 = state["t1"]

    # check warning and mass
    if exp_n_add != n_add:
        assert len(caplog.records) > 0
    else:
        assert len(caplog.records) == 0

    # new cells were created
    assert t0.idx.numel() + exp_n_add == t1.idx.numel()
    new_idx = t1.idx[~torch.isin(t1.idx, t0.idx)]

    # mass was added if cells were added
    if n_add == 0:
        torch.testing.assert_close(t0.mass, t1.mass, atol=_ATOL, rtol=_RTOL)
    else:
        assert t0.mass < t1.mass

    # old cells didn't change
    assert torch.equal(t0.pos[t0.idx], t1.pos[t0.idx])
    p = min(t0.p, t1.p) - 1

    for key in t0.cell_tensors:
        ten0 = t0.cell_tensors[key][t0.idx]
        ten1 = t1.cell_tensors[key][t0.idx]
        assert torch.equal(ten0, ten1), key

    for key in t0.proteome_tensors:
        ten0 = t0.proteome_tensors[key][t0.idx, p]
        ten1 = t1.proteome_tensors[key][t0.idx, p]
        assert torch.equal(ten0, ten1), key

    for key in t0.cell_arrays:
        arr0 = t0.cell_arrays[key][t0.idx]
        arr1 = t1.cell_arrays[key][t0.idx]
        assert np.all(arr0 == arr1), key

    if n_add == 0:
        return

    # new cells are sound
    assert t1.cell_tensors["alive"][new_idx].all()
    assert (t1.cell_tensors["ages"][new_idx] == 1.0).all()
    assert (t1.cell_tensors["generations"][new_idx] == 1).all()
    assert (t1.proteome_tensors["v_max"][new_idx] != 0).any()
    assert all(len(d) > 0 for d in t1.cell_arrays["genomes"][new_idx])
    assert all(len(d) > 0 for d in t1.cell_arrays["lineages"][new_idx])

    # molecule map is as before
    assert torch.equal(
        t0.culture_tensors["molecule_map"], t1.culture_tensors["molecule_map"]
    )

    # molecules were carried over
    assert (world.cells.molecules[new_idx] == cell.molecules).all()


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
    _spawn_cells(world, n=n_spawn)

    # resuspend cells
    with _capture_state_change(world) as state:
        t0 = state["t0"]
        susp_idxs = t0.idx[:n_resp]
        non_susp_idxs = t0.idx[n_resp:]
        world.resuspend_cells(cell_idxs=susp_idxs)

    t1 = state["t1"]

    # indexes are the same
    assert torch.equal(t0.idx, t1.idx)

    # mass is conserved
    torch.testing.assert_close(t0.mass, t1.mass, atol=_ATOL, rtol=_RTOL)

    # cells didn't change
    for key in t0.cell_tensors:
        ten0 = t0.cell_tensors[key][t0.idx]
        ten1 = t1.cell_tensors[key][t0.idx]
        assert torch.equal(ten0, ten1), key

    for key in t0.proteome_tensors:
        ten0 = t0.proteome_tensors[key][t0.idx]
        ten1 = t1.proteome_tensors[key][t0.idx]
        assert torch.equal(ten0, ten1), key

    for key in t0.cell_arrays:
        arr0 = t0.cell_arrays[key][t0.idx]
        arr1 = t1.cell_arrays[key][t0.idx]
        assert np.all(arr0 == arr1), key

    # non-resuspended cells didn't change position
    assert torch.equal(t0.pos[non_susp_idxs], t1.pos[non_susp_idxs])

    if n_resp == 0:
        return

    # only positions of resuspended cells changed (by chance a few could be the same)
    assert ((t0.pos[susp_idxs] != t1.pos[susp_idxs]).sum(1) > 0).sum() > 0.8 * n_resp


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
    _spawn_cells(world, n=n_spawn)

    # migrate cells
    with _capture_state_change(world) as state:
        t0 = state["t0"]
        migrate_idxs = t0.idx[:n_migrate]
        non_migrate_idxs = t0.idx[n_migrate:]
        scc_idx = world.migrate_cells(cell_idxs=migrate_idxs)

    t1 = state["t1"]

    # cells migrated (in full map case 1 may fail to migrate by chance)
    assert abs(scc_idx.numel() - n_succ) < 2

    # mass is conserved
    torch.testing.assert_close(t0.mass, t1.mass, atol=_ATOL, rtol=_RTOL)

    # cells didn't change
    assert torch.equal(t0.idx, t1.idx)

    for key in t0.cell_tensors:
        ten0 = t0.cell_tensors[key]
        ten1 = t1.cell_tensors[key]
        assert torch.equal(ten0, ten1), key

    for key in t0.proteome_tensors:
        ten0 = t0.proteome_tensors[key]
        ten1 = t1.proteome_tensors[key]
        assert torch.equal(ten0, ten1), key

    for key in t0.cell_arrays:
        arr0 = t0.cell_arrays[key][:]
        arr1 = t1.cell_arrays[key][:]
        assert np.all(arr0 == arr1), key

    # non-migrated and non-successful cells didn't change position
    non_succ_idxs = migrate_idxs[~torch.isin(migrate_idxs, scc_idx)]
    not_moved_idxs = torch.cat([non_migrate_idxs, non_succ_idxs])
    assert torch.equal(t0.pos[not_moved_idxs], t1.pos[not_moved_idxs])

    if n_succ == 0:
        return

    # only positions of successful cells changed
    assert ((t0.pos[scc_idx] != t1.pos[scc_idx]).sum(1) > 0).all()


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
    _spawn_cells(world, n=n_spawn)

    # divide cells
    with _capture_state_change(world) as state:
        t0 = state["t0"]
        divide_idxs = t0.idx[:n_divide]
        non_divide_idxs = t0.idx[n_divide:]
        parent_idxs, child_idxs = world.divide_cells(cell_idxs=divide_idxs)

    t1 = state["t1"]

    # some cells divided
    assert parent_idxs.numel() == n_succ
    assert parent_idxs.numel() == child_idxs.numel()
    unsucc_parent_idxs = divide_idxs[~torch.isin(divide_idxs, parent_idxs)]

    # mass is conserved
    torch.testing.assert_close(t0.mass, t1.mass, atol=_ATOL, rtol=_RTOL)

    # original cells didn't change except for generation, age, molecules
    p = min(t0.p, t1.p) - 1

    for key in t0.proteome_tensors:
        ten0 = t0.proteome_tensors[key][t0.idx, :p]
        ten1 = t1.proteome_tensors[key][t0.idx, :p]
        assert torch.equal(ten0, ten1), key

    for key in t0.cell_arrays:
        arr0 = t0.cell_arrays[key][t0.idx]
        arr1 = t1.cell_arrays[key][t0.idx]
        assert np.all(arr0 == arr1), key

    # original cells didn't change position
    torch.equal(t0.pos[t0.idx], t1.pos[t0.idx])

    # non-divided cells have same molecules, ages, generations
    non_divide_idxs = torch.cat([non_divide_idxs, unsucc_parent_idxs])
    for key in t0.cell_tensors:
        ten0 = t0.cell_tensors[key][non_divide_idxs]
        ten1 = t1.cell_tensors[key][non_divide_idxs]
        assert torch.equal(ten0, ten1), key

    if n_succ == 0:
        return

    # successful parents and children share molecules
    desc_idxs = torch.cat([parent_idxs, child_idxs])
    assert (t1.cell_tensors["molecules"][desc_idxs] == 5.0).all()

    # they increment generation and start at age 0
    assert (t1.cell_tensors["generations"][parent_idxs] == 2).all()
    assert (t1.cell_tensors["generations"][child_idxs] == 2).all()
    assert (t1.cell_tensors["ages"][parent_idxs] == 0.0).all()
    assert (t1.cell_tensors["ages"][child_idxs] == 0.0).all()

    # their genomes, lineages, proteomes should be equal
    for key in t1.cell_arrays:
        arr0 = t1.cell_arrays[key][parent_idxs]
        arr1 = t1.cell_arrays[key][child_idxs]
        assert np.all(arr0 == arr1), key

    for key in t1.proteome_tensors:
        ten0 = t1.proteome_tensors[key][parent_idxs, :p]
        ten1 = t1.proteome_tensors[key][child_idxs, :p]
        assert torch.equal(ten0, ten1), key


@pytest.mark.parametrize(
    "n_spawn, n_update",
    [
        (50, 50),  # update all
        (50, 25),  # update some
        (50, 0),  # update none
    ],
)
def test_update_cells(world: World, n_spawn: int, n_update: int) -> None:
    _spawn_cells(world, n=n_spawn)

    # update cells
    with _capture_state_change(world) as state:
        t0 = state["t0"]
        update_idxs = t0.idx[:n_update]
        no_update_idxs = t0.idx[n_update:]
        new_genomes = [random_genome() for _ in range(n_update)]
        world.update_cells(genomes=new_genomes, cell_idxs=update_idxs)

    t1 = state["t1"]

    # mass is conserved
    torch.testing.assert_close(t0.mass, t1.mass, atol=_ATOL, rtol=_RTOL)

    # cell didn't change except for genomes and proteomes
    assert torch.equal(t0.idx, t1.idx)
    assert torch.equal(t0.pos, t1.pos)
    p = min(t0.p, t1.p) - 1

    arr0 = t0.cell_arrays["lineages"][:]
    arr1 = t1.cell_arrays["lineages"][:]
    assert np.all(arr0 == arr1)

    for key in t0.cell_tensors:
        ten0 = t0.cell_tensors[key]
        ten1 = t1.cell_tensors[key]
        assert torch.equal(ten0, ten1), key

    for key in t0.culture_tensors:
        ten0 = t0.culture_tensors[key]
        ten1 = t1.culture_tensors[key]
        assert torch.equal(ten0, ten1), key

    # non-updated cells have same genomes and proteome
    arr0 = t0.cell_arrays["genomes"][no_update_idxs]
    arr1 = t1.cell_arrays["genomes"][no_update_idxs]
    assert np.all(arr0 == arr1)

    for key in t0.proteome_tensors:
        ten0 = t0.proteome_tensors[key][no_update_idxs, :p]
        ten1 = t1.proteome_tensors[key][no_update_idxs, :p]
        assert torch.equal(ten0, ten1), key

    if n_update == 0:
        return

    # updated cells have different genomes and proteomes
    arr0 = t0.cell_arrays["genomes"][update_idxs]
    arr1 = t1.cell_arrays["genomes"][update_idxs]
    assert not np.all(arr0 == arr1)

    for key in t0.proteome_tensors:
        ten0 = t0.proteome_tensors[key][update_idxs, :p]
        ten1 = t1.proteome_tensors[key][update_idxs, :p]
        assert not torch.equal(ten0, ten1), key


@pytest.mark.parametrize(
    "n_spawn, n_kill",
    [
        (50, 50),  # kill all
        (50, 25),  # kill some
        (50, 0),  # kill none
    ],
)
def test_kill_cells(world: World, n_spawn: int, n_kill: int) -> None:
    world = _spawn_cells(world, n_spawn)
    n_remain = n_spawn - n_kill

    # kill cells
    with _capture_state_change(world) as state:
        t0 = state["t0"]
        kill_idxs = t0.idx[n_remain : n_remain + n_kill]
        no_kill_idxs = t0.idx[:n_remain]
        world.kill_cells(cell_idxs=kill_idxs)

    t1 = state["t1"]

    # cells died
    assert world.cells.get_n_cells() == n_remain
    assert set(t0.idx.tolist()) == set(t1.idx.tolist() + kill_idxs.tolist())

    # mass is conserved
    torch.testing.assert_close(t0.mass, t1.mass, atol=_ATOL, rtol=_RTOL)

    # survivors didn't change
    assert torch.equal(t0.pos[no_kill_idxs], t1.pos[no_kill_idxs])

    for key in t0.cell_tensors:
        ten0 = t0.cell_tensors[key][no_kill_idxs]
        ten1 = t1.cell_tensors[key][no_kill_idxs]
        assert torch.equal(ten0, ten1), key

    for key in t0.proteome_tensors:
        ten0 = t0.proteome_tensors[key][no_kill_idxs]
        ten1 = t1.proteome_tensors[key][no_kill_idxs]
        assert torch.equal(ten0, ten1), key

    for key in t0.cell_arrays:
        arr0 = t0.cell_arrays[key][no_kill_idxs]
        arr1 = t1.cell_arrays[key][no_kill_idxs]
        assert np.all(arr0 == arr1), key

    if n_kill == 0:
        return

    # killed cells are gone
    pos = t0.pos[kill_idxs]
    assert (world.culture.cell_map[pos[:, 0], pos[:, 1]] == -1).all()

    for key, tens in t1.cell_tensors.items():
        assert (tens[kill_idxs] == 0).all(), key

    for key, tens in t1.proteome_tensors.items():
        assert (tens[kill_idxs] == 0).all(), key

    for key, arr in t1.cell_arrays.items():
        assert np.all(arr[kill_idxs] == ""), key

    for key in t0.culture_tensors:
        ten0 = t0.culture_tensors[key]
        ten1 = t1.culture_tensors[key]
        assert not torch.equal(ten0, ten1), key


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_spawn",
    [
        100,  # full
        50,  # 50% confl
        0,  # empty
    ],
)
def test_advance_time(world: World, n_spawn: int) -> None:
    world = _spawn_cells(world, n_spawn)

    # advance time
    with _capture_state_change(world) as state:
        world.advance_time()

    t0 = state["t0"]
    t1 = state["t1"]

    # except for molecule distribution and ages everything is the same
    assert torch.equal(t0.idx, t1.idx)
    assert torch.equal(t0.pos, t1.pos)

    for key in [d for d in t0.cell_tensors if d not in ["ages", "molecules"]]:
        ten0 = t0.cell_tensors[key]
        ten1 = t1.cell_tensors[key]
        assert torch.equal(ten0, ten1), key

    for key in t0.proteome_tensors:
        ten0 = t0.proteome_tensors[key]
        ten1 = t1.proteome_tensors[key]
        assert torch.equal(ten0, ten1), key

    for key in t0.cell_arrays:
        arr0 = t0.cell_arrays[key][:]
        arr1 = t1.cell_arrays[key][:]
        assert np.all(arr0 == arr1), key

    if n_spawn == 0:
        return

    # mass is likely not the same
    assert not torch.equal(t0.mass, t1.mass)

    # ages incremented, molecules changed
    assert (t1.cell_tensors["ages"] == 2.0).all()
    assert not torch.equal(t0.cell_tensors["molecules"], t1.cell_tensors["molecules"])
    assert not torch.equal(
        t0.culture_tensors["molecule_map"], t1.culture_tensors["molecule_map"]
    )


def test_saving_loading_state(world: World, tmp_path: Path) -> None:
    statedir = tmp_path / "world_state"
    world = _spawn_cells(world, 50)

    with _capture_state_change(world) as state:
        # save world with 50 cells
        world.save_state(statedir=statedir)

        # clear world
        world.kill_cells()
        world.cells.update_p(p_req=0)
        world.cells.update_c(c_req=0)

        # it was really cleared
        assert world.cells.get_n_cells() == 0
        assert world.cells.get_cell_idxs().numel() == 0
        assert (world.culture.get_cell_positions() == -1).all()

        # load state
        world.load_state(statedir=statedir)

    t0 = state["t0"]
    t1 = state["t1"]

    # everything should be the same
    assert torch.equal(t0.idx, t1.idx)
    assert torch.equal(t0.pos, t1.pos)
    assert torch.equal(t0.mass, t1.mass)
    assert t0.p == t1.p

    for key in t0.culture_tensors:
        ten0 = t0.culture_tensors[key]
        ten1 = t1.culture_tensors[key]
        assert torch.equal(ten0, ten1), key

    for key in t0.cell_tensors:
        ten0 = t0.cell_tensors[key]
        ten1 = t1.cell_tensors[key]
        assert torch.equal(ten0, ten1), key

    for key in t0.proteome_tensors:
        ten0 = t0.proteome_tensors[key]
        ten1 = t1.proteome_tensors[key]
        assert torch.equal(ten0, ten1), key

    for key in t0.cell_arrays:
        arr0 = t0.cell_arrays[key][:]
        arr1 = t1.cell_arrays[key][:]
        assert np.all(arr0 == arr1), key


def test_saving_loading_obj(tmp_path: Path) -> None:
    world = World(
        chemistry=CHEMISTRY,
        kinetics_kwargs={"eps": 1e-3},
        diffusion_kwargs={"dx": 0.5},
        permeation_kwargs={"a_cell": 9},
        culture_kwargs={"map_size": 12},
        cells_kwargs={"dim_scaling": 0.3},
        genomics_kwargs={"stop_codons": ("TGA", "TAG")},
        proteomics_kwargs={"abs_temp": 100},
    )

    world = _spawn_cells(world, 50)

    # save world with 50 cells
    world.save(rundir=tmp_path)
    t0 = _State.from_world(world)

    # delete and load
    del world
    world = World.from_file(tmp_path)
    t1 = _State.from_world(world)

    # everything should be the same
    assert torch.equal(t0.idx, t1.idx)
    assert torch.equal(t0.pos, t1.pos)
    assert torch.equal(t0.mass, t1.mass)
    assert t0.p == t1.p

    for key in t0.culture_tensors:
        ten0 = t0.culture_tensors[key]
        ten1 = t1.culture_tensors[key]
        assert torch.equal(ten0, ten1), key

    for key in t0.cell_tensors:
        ten0 = t0.cell_tensors[key]
        ten1 = t1.cell_tensors[key]
        assert torch.equal(ten0, ten1), key

    for key in t0.proteome_tensors:
        ten0 = t0.proteome_tensors[key]
        ten1 = t1.proteome_tensors[key]
        assert torch.equal(ten0, ten1), key

    for key in t0.cell_arrays:
        arr0 = t0.cell_arrays[key][:]
        arr1 = t1.cell_arrays[key][:]
        assert np.all(arr0 == arr1), key

    # check class attrs
    assert world.kinetics.eps == 1e-3
    assert world.diffusion.N == 12  # map size
    assert world.culture.map_size == 12
    assert world.cells.dim_scaling == 0.3
    assert world.cells.genomics.stop_codons == ["TGA", "TAG"]
    assert world.cells.proteomics.abs_temp == 100


# TODO: test that I can add 0 cells and run simulation

# TODO: test that I can add a cell with 0 proteins (p=0) and run simulation

# TODO: K_e values created by random proteins should be on average 1
#       and most values should be between 1e-4 and 1e4
#       most people model it as log-normal distribution with center 1
#       and +/- 100 STDEV, values <1e-10 or >1e10 should be virtually impossible

# TODO: test that checks that Vmax, Km values are in proper range (see proteomics)
