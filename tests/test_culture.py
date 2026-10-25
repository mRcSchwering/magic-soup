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


def test_maps() -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=3)

    # should be mutable on both 2D and 1D
    culture.cell_map[0, 0] = 0
    culture.cell_map[0, 2] = 1
    culture.cell_map[1, 1] = 2
    culture.cell_map_flat[-1] = 9

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
    torch.testing.assert_close(culture.cell_map_flat, exp_cell_map_flat)
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
def test_get_cell_positions(cell_map: list, exp_pos: list) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=len(cell_map))
    culture.cell_map[:] = idxtensor(cell_map)
    pos = culture.get_cell_positions()
    torch.testing.assert_close(pos, idxtensor(exp_pos))


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
    "cell_map, n, exp_n",
    [
        (
            # empty map: ask 4, get 4
            [
                [-1, -1],
                [-1, -1],
            ],
            4,
            4,
        ),
        (
            # empty map: ask 2, get 2
            [
                [-1, -1],
                [-1, -1],
            ],
            2,
            2,
        ),
        (
            # full map, ask 4, get 0
            [
                [0, 1],
                [2, 3],
            ],
            4,
            0,
        ),
        (
            # interspersed cells: ask 2 get 2
            [
                [0, -1],
                [2, -1],
            ],
            2,
            2,
        ),
        (
            # interspersed cells: ask 4 get 2
            [
                [0, -1],
                [2, -1],
            ],
            4,
            2,
        ),
    ],
)
def test_get_free_positions(cell_map: list, n: int, exp_n: int) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=len(cell_map))
    culture.cell_map[:] = idxtensor(cell_map)
    pos = culture.get_free_positions(n)

    # check correct number of positions
    assert pos.shape == (exp_n, 2)

    # make sure positions are free
    xs = pos[:, 0]
    ys = pos[:, 1]
    assert (culture.cell_map[xs, ys] < 0).all()


@pytest.mark.parametrize(
    "cell_map, parents, children, n_exp",
    [
        (
            # empty map
            [
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
            [],
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
            [8],
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
            [15],
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
            [5, 6, 7, 8],
            4,
        ),
        (
            # many cells want to divide, only 1 available
            [
                [0, 7, 1],
                [2, 3, 4],
                [5, 6, -1],
            ],
            [3, 4, 6],
            [8],
            1,
        ),
        (
            # many children available only 2 want to divide
            [
                [0, -1, 1],
                [2, 3, 4],
                [-1, 6, -1],
            ],
            [0, 6],
            [5, 7, 8],
            2,
        ),
        (
            # many can divide but only 1 child available
            [
                [-1, -1, 1],
                [2, -1, -1],
                [-1, 6, -1],
            ],
            [1, 2, 6],
            [5],
            1,
        ),
    ],
)
def test_cell_division(
    cell_map: list, parents: list[int], children: list[int], n_exp: int
) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=len(cell_map))
    culture.cell_map[:] = idxtensor(cell_map)

    parents_0 = idxtensor(parents)
    children_0 = idxtensor(children)

    # remember idxs and positions before
    cell_idxs_0 = culture.cell_map[culture.cell_map >= 0].flatten().sort()[0]
    cell_pos_0 = culture.get_cell_positions()

    # divide cells
    scc_parents, scc_children = culture.cell_division(
        parent_idxs=parents_0, child_idxs=children_0
    )

    # remember idxs and positions after
    cell_idxs_1 = culture.cell_map[culture.cell_map >= 0].flatten().sort()[0]
    cell_pos_1 = culture.get_cell_positions()
    fld_parents = parents_0[~torch.isin(parents_0, scc_parents)]

    # check expected number of successful divisions
    assert scc_parents.numel() == n_exp
    assert scc_children.numel() == n_exp

    # basic consistency checks
    assert set(fld_parents.tolist()) & set(scc_parents.tolist()) == set()
    assert set(scc_parents.tolist()) & set(scc_children.tolist()) == set()
    assert set(fld_parents.tolist() + scc_parents.tolist()) == set(parents_0.tolist())
    assert fld_parents.numel() + scc_parents.numel() == parents_0.numel()

    # test only provided parents and children are in the results
    assert set(children_0.tolist()) >= set(scc_children.tolist())
    assert set(parents_0.tolist()) >= set(scc_parents.tolist())

    # test map now contains new idxs
    assert set(cell_idxs_0.tolist()) | set(scc_children.tolist()) == set(
        cell_idxs_1.tolist()
    )

    # test all original cells are unchanged
    assert torch.equal(cell_pos_0[cell_idxs_0], cell_pos_1[cell_idxs_0])

    # test all children are in Moore neighborhood of their parents
    for p0, p1 in zip(
        cell_pos_0[scc_parents].tolist(),
        cell_pos_1[scc_children].tolist(),
    ):
        assert is_moore_nbd(p0, p1, n=culture.map_size)


@pytest.mark.slow
def test_cell_division_randomly() -> None:
    N = 100
    for _ in range(100):
        culture = Culture(chemistry=CHEMISTRY, map_size=N)

        # genrate cell map
        culture.cell_map[:] = _gen_rand_cell_map(N=N, confl=0.7)
        n = int(culture.cell_map.max()) + 1

        # remember idxs and positions before
        cell_idxs_0 = culture.cell_map[culture.cell_map >= 0].flatten().sort()[0]
        cell_pos_0 = culture.get_cell_positions()

        # migrate
        parents_0 = idxrandperm(n)[: random.randint(5, n)]
        free = [i for i in range(n, N * N) if i not in parents_0.tolist()]
        children_0 = idxtensor(free)[: random.randint(0, len(free))]

        # divide cells
        scc_parents, scc_children = culture.cell_division(
            parent_idxs=parents_0, child_idxs=children_0
        )

        # remember idxs and positions after
        cell_idxs_1 = culture.cell_map[culture.cell_map >= 0].flatten().sort()[0]
        cell_pos_1 = culture.get_cell_positions()
        fld_parents = parents_0[~torch.isin(parents_0, scc_parents)]

        # basic consistency checks
        assert set(fld_parents.tolist()) & set(scc_parents.tolist()) == set()
        assert set(scc_parents.tolist()) & set(scc_children.tolist()) == set()
        assert set(fld_parents.tolist() + scc_parents.tolist()) == set(
            parents_0.tolist()
        )
        assert fld_parents.numel() + scc_parents.numel() == parents_0.numel()

        # test only provided parents and children are in the results
        assert set(children_0.tolist()) >= set(scc_children.tolist())
        assert set(parents_0.tolist()) >= set(scc_parents.tolist())

        # test map now contains new idxs
        assert set(cell_idxs_0.tolist()) | set(scc_children.tolist()) == set(
            cell_idxs_1.tolist()
        )

        # test all original cells are unchanged
        assert torch.equal(cell_pos_0[cell_idxs_0], cell_pos_1[cell_idxs_0])

        # test all children are in Moore neighborhood of their parents
        for p0, p1 in zip(
            cell_pos_0[scc_parents].tolist(),
            cell_pos_1[scc_children].tolist(),
        ):
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
                [9, 6, 8],
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
def test_undirected_cell_migration(cell_map: list, idxs: list[int], exp_n: int) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=len(cell_map))
    culture.cell_map[:] = idxtensor(cell_map)
    cell_idxs = idxtensor(idxs)

    # remember idxs and positions before
    cell_idxs_0 = culture.cell_map[culture.cell_map >= 0].flatten().sort()[0]
    cell_pos_0 = culture.get_cell_positions()

    # migrate
    succ_idxs = culture.undirected_cell_migration(cell_idxs=cell_idxs)

    # remember idxs and positions after
    cell_idxs_1 = culture.cell_map[culture.cell_map >= 0].flatten().sort()[0]
    cell_pos_1 = culture.get_cell_positions()
    unsucc_idxs = cell_idxs[~torch.isin(cell_idxs, succ_idxs)]

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
def test_undirected_cell_migration_randomly() -> None:
    N = 100
    for _ in range(100):
        culture = Culture(chemistry=CHEMISTRY, map_size=N)

        # genrate cell map
        culture.cell_map[:] = _gen_rand_cell_map(N=N, confl=0.7)
        n = int(culture.cell_map.max())

        # remember idxs and positions before
        cell_idxs_0 = culture.cell_map[culture.cell_map >= 0].flatten().sort()[0]
        cell_pos_0 = culture.get_cell_positions()

        # migrate
        k = random.randint(10, n)
        cell_idxs = idxrandperm(n)[:k]
        succ_idxs = culture.undirected_cell_migration(cell_idxs=cell_idxs)

        # remember idxs and positions after
        cell_idxs_1 = culture.cell_map[culture.cell_map >= 0].flatten().sort()[0]
        cell_pos_1 = culture.get_cell_positions()
        unsucc_idxs = cell_idxs[~torch.isin(cell_idxs, succ_idxs)]

        # at least some should have migrated
        assert succ_idxs.numel() > 0

        # successful idxs are subset of requested idxs
        assert set(cell_idxs.tolist()) >= set(succ_idxs.tolist())

        # cell idxs are unchanged
        assert torch.equal(cell_idxs_0, cell_idxs_1)

        # unsuccessful cells are still in same positions
        assert torch.equal(cell_pos_0[unsucc_idxs], cell_pos_1[unsucc_idxs])

        # successful cells all have new positions
        assert ((cell_pos_0[succ_idxs] != cell_pos_1[succ_idxs]).sum(dim=1) > 0).all()

        # they are all in Moore neighborhood of each other
        for p0, p1 in zip(
            cell_pos_0[succ_idxs].tolist(),
            cell_pos_1[succ_idxs].tolist(),
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
