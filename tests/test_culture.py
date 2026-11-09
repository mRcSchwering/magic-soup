import random
from itertools import product
from pathlib import Path

import pytest
import torch
from magicsoup.culture import Culture
from magicsoup.examples.wood_ljungdahl import CHEMISTRY

from tests.util import (
    fzeros,
    idxarange,
    idxfull,
    idxrandperm,
    idxtensor,
    idxzeros,
    is_moore_nbd,
)


def _gen_rand_cell_map(N: int, confl: float) -> torch.Tensor:
    n = round(N * N * confl)
    flat = idxfull(N * N, value=-1)
    perm = idxrandperm(N * N)[:n]
    flat[perm] = idxarange(n)
    return flat.reshape(N, N)


@pytest.mark.parametrize(
    "occ, c, req",
    [
        ([0, 1], 9, [0, 1]),  # 2 requested 7 free
        ([], 9, [2, 3]),  # 2 requested 9 free
        ([2, 3], 9, []),  # 0 requested 7 free
        ([4, 5], 2, [0, 1]),  # 2 requested 7 free (c=2)
        (list(range(8)), 9, [0, 1]),  # 2 requested 1 free
        ([0, 1], None, []),  # all requested 7 free
    ],
)
def test_get_random_free_voxels(occ: list[int], c: int | None, req: list[int]) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=3)

    # occupy cell map
    flat = culture.cell_map.view(-1)
    flat[idxtensor(occ)] = 0

    # setup mask
    if c is None:
        mask = None
    else:
        mask = idxfull(c, value=0).bool()
        mask[idxtensor(req)] = True

    voxels = culture.get_random_free_voxels(mask=mask)

    # voxels were only chosen for mask
    if mask is None:
        (voxels >= 0).all()
    else:
        assert (voxels[~mask] == -1).all()

    # result voxels are unique
    res_voxels = voxels[voxels >= 0].tolist()
    assert len(set(res_voxels)) == len(res_voxels)

    # result voxels were not already occupied
    occ_voxels = (flat >= 0).argwhere().flatten().tolist()
    assert len(set(occ_voxels) & set(res_voxels)) == 0


@pytest.mark.parametrize(
    "voxels",
    [
        ([-1, -1, -1, 1, 2, 3]),  # 3 new cells on free voxels
        ([-1, -1]),  # move none
        ([]),  # empty
        ([-1, 7, 6, 4, -1]),  # 2 existing, 1 new cell
    ],
)
def test_place_cells(voxels: list[int]) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=3)

    # pre-occupy cell map
    i2v = {0: 0, 1: 4, 2: 7}
    flat = culture.cell_map.view(-1)
    flat[idxtensor(list(i2v.values()))] = idxtensor(list(i2v))

    culture.place_cells(voxels=idxtensor(voxels))
    print(flat)

    for idx, vox in enumerate(voxels):
        if vox >= 0:
            i2v[idx] = vox

    # voxels map to indices as expected
    for vox, idx in enumerate(flat.tolist()):
        if idx in i2v:
            assert vox == i2v[idx]
        else:
            assert idx == -1


@pytest.mark.parametrize(
    "bfr, o2n, exp",
    [
        ([[0, -1], [-1, 1]], [2, 3], [[2, -1], [-1, 3]]),
        ([[2, -1], [-1, 3]], [-1, -1, 0, 1], [[0, -1], [-1, 1]]),
    ],
)
def test_update_cell_map(bfr: list, o2n: list, exp: list) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=2)
    culture.cell_map[:] = idxtensor(bfr)
    culture.update_cell_map(idxtensor(o2n))
    assert (culture.cell_map == idxtensor(exp)).all()


def test_maps() -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=3)

    # should be mutable on both 2D and 1D
    culture.cell_map[0, 0] = 0
    culture.cell_map[0, 2] = 1
    culture.cell_map[1, 1] = 2
    culture.cell_map.view(-1)[-1] = 9

    # expected maps
    exp_coords = idxtensor([tuple(d) for d in product(range(3), range(3))])
    exp_cell_map = idxtensor(
        [
            [0, -1, 1],
            [-1, 2, -1],
            [-1, -1, 9],
        ]
    )
    exp_cell_map_flat = idxtensor([0, -1, 1, -1, 2, -1, -1, -1, 9])
    exp_mol_map = fzeros(len(CHEMISTRY.molecules), 3, 3)

    # check expectations
    torch.testing.assert_close(culture.all_positions, exp_coords)
    torch.testing.assert_close(culture.cell_map, exp_cell_map)
    torch.testing.assert_close(culture.cell_map.view(-1), exp_cell_map_flat)
    torch.testing.assert_close(culture.molecule_map, exp_mol_map)


@pytest.mark.parametrize(
    "cell_map, exp_pos",
    [
        (
            # empty map
            [
                [-1, -1],
                [-1, -1],
            ],
            [(-1, -1), (-1, -1), (-1, -1), (-1, -1)],
        ),
        (
            # full map
            [
                [0, 1],
                [2, 3],
            ],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        ),
        (
            # interspersed cells
            [
                [0, -1],
                [2, -1],
            ],
            [(0, 0), (-1, -1), (1, 0), (-1, -1)],
        ),
    ],
)
def test_get_cell_coords(cell_map: list, exp_pos: list) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=len(cell_map))
    culture.cell_map[:] = idxtensor(cell_map)
    coords = culture.get_cell_coords()
    torch.testing.assert_close(coords, idxtensor(exp_pos))


@pytest.mark.parametrize(
    "cell_map, exp_pairs",
    [
        # one possible pair
        (
            [
                [-1, 0, 1],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
            [(0, 1)],
        ),
        # no cells
        (
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
            [],
        ),
        # multiple cell pairs
        (
            [
                [-1, 0, -1],
                [2, -1, -1],
                [-1, -1, 5],
            ],
            [(0, 2), (2, 5), (5, 0)],
        ),
        # no pair possible
        (
            [
                [-1, 1, -1, -1],
                [-1, -1, -1, -1],
                [-1, 1, -1, -1],
                [-1, -1, -1, -1],
            ],
            [],
        ),
    ],
)
def test_get_cell_neighbor_pairs(cell_map: list, exp_pairs: list) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=len(cell_map))
    culture.cell_map[:] = idxtensor(cell_map)
    pairs = culture.get_cell_neighbor_pairs()
    exp = idxtensor(exp_pairs) if exp_pairs else idxzeros(0, 2)
    torch.testing.assert_close(pairs, exp)


@pytest.mark.parametrize(
    "cell_map, parents, n_exp",
    [
        (
            # empty map
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
            [],
            0,
        ),
        (
            # no cell divides
            [
                [0, -1, 1],
                [2, 3, 4],
                [5, 6, 7],
            ],
            [],
            0,
        ),
        (
            # 1 cell wants to divide (only 1 choice)
            [
                [3, -1, 1],
                [7, 0, 4],
                [5, 6, 2],
            ],
            [0],
            1,
        ),
        (
            # 1 cell wants to devide but cant (no free neighborhood)
            [
                [0, -1, 1, 2],
                [3, 4, 5, 6],
                [7, 8, 9, 10],
                [11, 12, 13, 14],
            ],
            [10],
            0,
        ),
        (
            # many cells can divide and there's competition
            [
                [0, -1, 1, -1],
                [-1, 2, -1, -1],
                [3, -1, 4, -1],
                [-1, -1, -1, -1],
            ],
            [0, 1, 2, 3],
            4,
        ),
        (
            # many cells want to divide, only 1 free voxel
            [
                [0, 7, 1],
                [2, 3, 4],
                [5, 6, -1],
            ],
            [3, 4, 6],
            1,
        ),
        (
            # 2 cells want to divide, plenty of room
            [
                [0, -1, 1],
                [2, 3, 4],
                [-1, 6, -1],
            ],
            [0, 6],
            2,
        ),
        (
            # many cells divide, plenty of free voxels
            [
                [-1, -1, 1],
                [2, -1, -1],
                [-1, 6, -1],
            ],
            [1, 2, 6],
            3,
        ),
    ],
)
def test_divide_cells(cell_map: list, parents: list[int], n_exp: int) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=len(cell_map))
    culture.cell_map[:] = idxtensor(cell_map)

    # which cell slots want to divide (mask over all possible slots)
    mask = idxzeros(culture.N2).bool()
    mask[idxtensor(parents)] = True

    # remember positions before
    coords_0 = culture.get_cell_coords()
    cell_map_before = culture.cell_map.clone()

    # resolve division contention (reserves won voxels on the map)
    success, won_voxel = culture.divide_cells(mask=mask)
    scc_parents = success.nonzero(as_tuple=True)[0].to(torch.int32)

    # expected number of successful divisions (= won free voxels)
    assert scc_parents.numel() == n_exp

    # successful parents are a subset of requested parents
    assert set(scc_parents.tolist()) <= set(parents)

    # originally occupied voxels are unchanged (reservations only touch free voxels)
    occ = cell_map_before >= 0
    assert torch.equal(culture.cell_map[occ], cell_map_before[occ])

    # won voxels are distinct
    scc_voxels = won_voxel[success]
    assert scc_voxels.unique().numel() == scc_voxels.numel()

    # each won voxel was free before, is reserved (with parent id) after, and
    # lies in the parent's Moore neighbourhood
    for p in scc_parents.tolist():
        v = int(won_voxel[p])
        assert v >= 0
        x, y = v // culture.map_size, v % culture.map_size
        assert cell_map_before[x, y] < 0
        assert int(culture.cell_map[x, y]) == p
        assert is_moore_nbd(tuple(coords_0[p].tolist()), (x, y), n=culture.map_size)


@pytest.mark.slow
def test_divide_cells_randomly() -> None:
    N = 100
    for _ in range(100):
        culture = Culture(chemistry=CHEMISTRY, map_size=N)

        # genrate cell map
        culture.cell_map[:] = _gen_rand_cell_map(N=N, confl=0.7)
        n = int(culture.cell_map.max()) + 1

        # remember positions before
        coords_0 = culture.get_cell_coords()
        cell_map_before = culture.cell_map.clone()

        # pick random parents that want to divide
        parents_0 = idxrandperm(n)[: random.randint(5, n)]
        mask = idxzeros(culture.N2).bool()
        mask[parents_0] = True

        # divide cells (resolves voxel contention, reserves won voxels)
        success, won_voxel = culture.divide_cells(mask=mask)
        scc_parents = success.nonzero(as_tuple=True)[0].to(torch.int32)

        # some should have divided
        assert scc_parents.numel() > 0

        # successful parents are a subset of requested parents
        assert set(scc_parents.tolist()) <= set(parents_0.tolist())

        # originally occupied voxels are unchanged (reservations touch free voxels)
        occ = cell_map_before >= 0
        assert torch.equal(culture.cell_map[occ], cell_map_before[occ])

        # won voxels are distinct, were free before, and are reserved after
        scc_voxels = won_voxel[success]
        assert scc_voxels.unique().numel() == scc_voxels.numel()
        xs = scc_voxels // culture.map_size
        ys = scc_voxels % culture.map_size
        assert (cell_map_before[xs, ys] < 0).all()
        assert (culture.cell_map[xs, ys] == scc_parents).all()

        # each won voxel is in the parent's Moore neighborhood
        par_pos = coords_0[scc_parents].tolist()
        vox_pos = torch.stack([xs, ys], dim=1).tolist()
        for p0, p1 in zip(par_pos, vox_pos):
            assert is_moore_nbd(p0, p1, n=culture.map_size)


@pytest.mark.parametrize(
    "cell_map, idxs, exp_n",
    [
        (
            # empty map
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
            [],
            0,
        ),
        (
            # no cell wants to migrate
            [
                [0, -1, 1],
                [2, 3, 4],
                [5, 6, 7],
            ],
            [],
            0,
        ),
        (
            # 1 cell can migrate (only 1 choice)
            [
                [0, -1, 1],
                [2, 3, 4],
                [5, 6, 7],
            ],
            [0],
            1,
        ),
        (
            # 1 cell cant migrate (no free neighborhood)
            [
                [0, -1, 1, 2],
                [3, 4, 5, 6],
                [7, 8, 9, 10],
                [11, 12, 13, 14],
            ],
            [10],
            0,
        ),
        (
            # 1 cell can migrate
            [
                [0, -1, 5],
                [2, 1, 7],
                [3, 6, 8],
            ],
            [5],
            1,
        ),
        (
            # many cells can migrate and there's competition
            [
                [0, -1, 1, -1],
                [-1, 2, -1, -1],
                [3, -1, 4, -1],
                [-1, -1, -1, -1],
            ],
            [0, 1, 2, 3],
            4,
        ),
    ],
)
def test_migrate_cells_randomly(cell_map: list, idxs: list[int], exp_n: int) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=len(cell_map))
    culture.cell_map[:] = idxtensor(cell_map)

    # which cell slots want to migrate (mask over all possible slots)
    req_idxs = idxtensor(idxs)
    mask = idxzeros(culture.N2).bool()
    mask[req_idxs] = True

    # remember idxs and positions before
    cell_idxs_0 = culture.cell_map[culture.cell_map >= 0].flatten().sort()[0]
    cell_pos_0 = culture.get_cell_coords()

    # migrate
    migrated = culture.migrate_cells_randomly(mask=mask)
    succ_idxs = migrated.nonzero(as_tuple=True)[0].to(torch.int32)

    # remember idxs and positions after
    cell_idxs_1 = culture.cell_map[culture.cell_map >= 0].flatten().sort()[0]
    cell_pos_1 = culture.get_cell_coords()
    unsucc_idxs = req_idxs[~torch.isin(req_idxs, succ_idxs)]

    # check expected number of successful migrations
    assert succ_idxs.numel() == exp_n

    # successful idxs are subset of requested idxs
    assert set(idxs) >= set(succ_idxs.tolist())

    # cell idxs are unchanged
    assert torch.equal(cell_idxs_0, cell_idxs_1)

    # unsuccessful cells are still in same positions
    assert torch.equal(cell_pos_0[unsucc_idxs], cell_pos_1[unsucc_idxs])

    # successful cells all have new positions
    assert ((cell_pos_0[succ_idxs] != cell_pos_1[succ_idxs]).sum(dim=1) > 0).all()

    # they are all in Moore neighborhood of each other
    for p0, p1 in zip(cell_pos_0[succ_idxs].tolist(), cell_pos_1[succ_idxs].tolist()):
        assert is_moore_nbd(p0, p1, n=culture.map_size)


@pytest.mark.slow
def test_randomly_migrate_cells_randomly() -> None:
    N = 100
    for _ in range(100):
        culture = Culture(chemistry=CHEMISTRY, map_size=N)

        # genrate cell map
        culture.cell_map[:] = _gen_rand_cell_map(N=N, confl=0.7)
        n = int(culture.cell_map.max())

        # remember idxs and positions before
        cell_idxs_0 = culture.cell_map[culture.cell_map >= 0].flatten().sort()[0]
        coords_0 = culture.get_cell_coords()

        # migrate
        k = random.randint(10, n)
        req_idxs = idxrandperm(n)[:k]
        mask = idxzeros(culture.N2).bool()
        mask[req_idxs] = True
        migrated = culture.migrate_cells_randomly(mask=mask)
        succ_idxs = migrated.nonzero(as_tuple=True)[0].to(torch.int32)

        # remember idxs and positions after
        cell_idxs_1 = culture.cell_map[culture.cell_map >= 0].flatten().sort()[0]
        coords_1 = culture.get_cell_coords()
        unsucc_idxs = req_idxs[~torch.isin(req_idxs, succ_idxs)]

        # at least some should have migrated
        assert succ_idxs.numel() > 0

        # successful idxs are subset of requested idxs
        assert set(req_idxs.tolist()) >= set(succ_idxs.tolist())

        # cell idxs are unchanged
        assert torch.equal(cell_idxs_0, cell_idxs_1)

        # unsuccessful cells are still in same positions
        assert torch.equal(coords_0[unsucc_idxs], coords_1[unsucc_idxs])

        # successful cells all have new positions
        assert ((coords_0[succ_idxs] != coords_1[succ_idxs]).sum(dim=1) > 0).all()

        # they are all in Moore neighborhood of each other
        for p0, p1 in zip(
            coords_0[succ_idxs].tolist(),
            coords_1[succ_idxs].tolist(),
        ):
            assert is_moore_nbd(p0, p1, n=culture.map_size)


def test_saving_loading(tmp_path: Path) -> None:
    statedir = tmp_path / "culture_state"

    # create culture with 70% confluency
    culture = Culture(chemistry=CHEMISTRY)
    culture.molecule_map[:] = torch.rand_like(culture.molecule_map)
    culture.cell_map[:] = _gen_rand_cell_map(N=culture.map_size, confl=0.7)

    # keep original state for comparison
    molecule_map = culture.molecule_map.clone()
    cell_map = culture.cell_map.clone()

    # save state
    culture.save_state(statedir=statedir)

    # change stuff
    culture.molecule_map = torch.rand_like(culture.molecule_map)
    culture.cell_map[:] = _gen_rand_cell_map(N=culture.map_size, confl=0.5)

    # load state
    culture.load_state(statedir=statedir)

    # check that the loaded state matches the original state
    assert (molecule_map == culture.molecule_map).all()
    assert (cell_map == culture.cell_map).all()
