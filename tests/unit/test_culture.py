import random
from itertools import product
from pathlib import Path

import pytest
import torch
from magicsoup.chemistry import Chemistry, Molecule
from magicsoup.culture import Culture
from magicsoup.examples.wood_ljungdahl import CHEMISTRY

from tests.config import DEVICE
from tests.util import fzeros, idxarange, idxfull, idxrandperm, idxtensor, seed

_ATOL = 1e-4
_RTOL = 1e-4


def _gen_rand_cell_map(L: int, confl: float) -> torch.Tensor:
    n = round(L * L * confl)
    flat = idxfull(L * L, value=-1)
    perm = idxrandperm(L * L)[:n]
    flat[perm] = idxarange(n)
    return flat.reshape(L, L)


def test_maps() -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=3, device=DEVICE)

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
    torch.testing.assert_close(culture.coord_map, exp_coords)
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
    culture = Culture(chemistry=CHEMISTRY, map_size=len(cell_map), device=DEVICE)
    culture.cell_map[:] = idxtensor(cell_map)
    pos = culture.get_cell_positions()
    torch.testing.assert_close(pos, idxtensor(exp_pos))


@pytest.mark.parametrize(
    "cell_map, exp_pairs",
    [
        (
            [
                [-1, 0, 1],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
            [(0, 1)],
        ),
        (
            [
                [-1, 0, -1],
                [2, -1, -1],
                [-1, -1, 5],
            ],
            [(0, 2), (2, 5), (5, 0)],
        ),
        (
            [
                [0, 1, -1],
                [2, 3, -1],
                [-1, -1, -1],
            ],
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        ),
    ],
)
def test_cell_neighbor_pairs(cell_map: list, exp_pairs: list) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=len(cell_map), device=DEVICE)
    culture.cell_map[:] = idxtensor(cell_map)
    pairs = culture.cell_neighbor_pairs()
    torch.testing.assert_close(pairs, idxtensor(exp_pairs))


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
def test_random_free_positions(cell_map: list, n: int, exp_n: int) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=len(cell_map), device=DEVICE)
    culture.cell_map[:] = idxtensor(cell_map)
    pos = culture.random_free_positions(n)

    # check correct number of positions
    assert pos.shape == (exp_n, 2)

    # make sure positions are free
    xs = pos[:, 0]
    ys = pos[:, 1]
    assert (culture.cell_map[xs, ys] < 0).all()


@pytest.mark.parametrize(
    "cell_map, parents, children, exp_cell_map, exp_succ_parents, exp_succ_children",
    [
        (
            # no cell divides
            [
                [0, -1, 1],
                [2, 3, 4],
                [5, 6, 7],
            ],
            [],
            [],
            [
                [0, -1, 1],
                [2, 3, 4],
                [5, 6, 7],
            ],
            [],
            [],
        ),
        (
            # 1 cell can divide (only 1 choice)
            [
                [0, -1, 1],
                [2, 3, 4],
                [5, 6, 7],
            ],
            [0],
            [8],
            [
                [0, 8, 1],
                [2, 3, 4],
                [5, 6, 7],
            ],
            [0],
            [8],
        ),
        (
            # 1 cell cant divide (no free neighborhood)
            [
                [0, -1, 1, 2],
                [3, 4, 5, 6],
                [7, 8, 9, 10],
                [11, 12, 13, 14],
            ],
            [10],
            [15],
            [
                [0, -1, 1, 2],
                [3, 4, 5, 6],
                [7, 8, 9, 10],
                [11, 12, 13, 14],
            ],
            [],
            [],
        ),
        (
            # 1 cell can divide (interspersed)
            [
                [0, -1, 5],
                [2, 1, 7],
                [9, 6, 8],
            ],
            [5],
            [3],
            [
                [0, 3, 5],
                [2, 1, 7],
                [9, 6, 8],
            ],
            [5],
            [3],
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
            [
                [0, 5, 1, 6],
                [-1, 2, 7, -1],
                [3, -1, 4, -1],
                [-1, -1, -1, 8],
            ],
            [0, 1, 2, 3],
            [5, 6, 7, 8],
        ),
    ],
)
def test_divide_cells(
    cell_map: list,
    parents: list[int],
    children: list[int],
    exp_cell_map: list,
    exp_succ_parents: list[int],
    exp_succ_children: list[int],
) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=len(cell_map), device=DEVICE)
    culture.cell_map[:] = idxtensor(cell_map)

    seed()  # seed for RNG
    succ_parents, succ_children = culture.divide_cells(
        parent_idxs=idxtensor(parents), child_idxs=idxtensor(children)
    )

    # check results
    torch.testing.assert_close(culture.cell_map, idxtensor(exp_cell_map))
    torch.testing.assert_close(succ_parents, idxtensor(exp_succ_parents))
    torch.testing.assert_close(succ_children, idxtensor(exp_succ_children))


@pytest.mark.parametrize(
    "cell_map, idxs, exp_cell_map, exp_succ_idxs",
    [
        (
            # no cell migrates
            [
                [0, -1, 1],
                [2, 3, 4],
                [5, 6, 7],
            ],
            [],
            [
                [0, -1, 1],
                [2, 3, 4],
                [5, 6, 7],
            ],
            [],
        ),
        (
            # 1 cell can migrate (only 1 choice)
            [
                [0, -1, 1],
                [2, 3, 4],
                [5, 6, 7],
            ],
            [0],
            [
                [-1, 0, 1],
                [2, 3, 4],
                [5, 6, 7],
            ],
            [0],
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
            [
                [0, -1, 1, 2],
                [3, 4, 5, 6],
                [7, 8, 9, 10],
                [11, 12, 13, 14],
            ],
            [],
        ),
        (
            # 1 cell can migrate (interspersed)
            [
                [0, -1, 5],
                [2, 1, 7],
                [9, 6, 8],
            ],
            [5],
            [
                [0, 5, -1],
                [2, 1, 7],
                [9, 6, 8],
            ],
            [5],
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
            [
                [-1, 0, -1, -1],
                [-1, 1, 2, -1],
                [-1, -1, 4, -1],
                [-1, -1, -1, 3],
            ],
            [0, 1, 2, 3],
        ),
    ],
)
def test_migrate_cells(
    cell_map: list,
    idxs: list[int],
    exp_cell_map: list,
    exp_succ_idxs: list[int],
) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=len(cell_map), device=DEVICE)
    culture.cell_map[:] = idxtensor(cell_map)

    seed()  # seed for RNG
    succ_idxs = culture.migrate_cells(cell_idxs=idxtensor(idxs))

    # check results
    torch.testing.assert_close(culture.cell_map, idxtensor(exp_cell_map))
    torch.testing.assert_close(succ_idxs, idxtensor(exp_succ_idxs))


@pytest.mark.slow
def test_migrate_cells_randomly() -> None:
    L = 100
    for _ in range(100):
        culture = Culture(chemistry=CHEMISTRY, map_size=L, device=DEVICE)

        # genrate cell map
        culture.cell_map[:] = _gen_rand_cell_map(L=L, confl=0.7)
        n = int(culture.cell_map.max())

        # remember cell map
        cell_map_0 = culture.cell_map.clone()
        n_cells_0 = (cell_map_0 >= 0).sum()

        k = random.randint(10, n)
        idxs = idxrandperm(n)[:k]
        succ_idxs = culture.migrate_cells(cell_idxs=idxs)

        k_succ = succ_idxs.shape[0]
        cell_map_1 = culture.cell_map
        n_cells_1 = (cell_map_1 >= 0).sum()

        assert k_succ <= k
        assert n_cells_0 == n_cells_1

        # at least some cells should have been updated
        assert (cell_map_0 != cell_map_1).any()


def test_diffuse():
    molecule_map = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    exp = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.2, 0.1, 0.0, 0.0, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.2],
                [0.0, 0.0, 0.2, 0.3, 0.2],
                [0.0, 0.0, 0.2, 0.3, 0.2],
                [0.1, 0.1, 0.1, 0.1, 0.2],
            ],
        ]
    )

    m0 = Molecule("m0", 10, diffusivity=0.0)
    m1 = Molecule("m1", 10, diffusivity=0.5)

    chemistry = Chemistry(molecules=[m0, m1], reactions=[])
    culture = Culture(chemistry=chemistry, map_size=5)
    culture.molecule_map = molecule_map

    culture.diffuse_molecules()

    torch.testing.assert_close(culture.molecule_map, exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
def test_diffuse_randomly() -> None:
    L = 100
    culture = Culture(chemistry=CHEMISTRY, map_size=L, device=DEVICE)
    culture.molecule_map[:] = torch.rand_like(culture.molecule_map) * 10
    totals_0 = culture.molecule_map.sum(dim=(1, 2))

    for _ in range(1000):
        culture.diffuse_molecules()

    totals_1 = culture.molecule_map.sum(dim=(1, 2))
    torch.testing.assert_close(totals_0, totals_1, atol=_ATOL, rtol=_RTOL)


def test_saving_loading(tmp_path: Path) -> None:
    statedir = tmp_path / "culture_state"

    # create culture with 70% confluency
    culture = Culture(chemistry=CHEMISTRY, device=DEVICE)
    culture.molecule_map[:] = torch.rand_like(culture.molecule_map)
    culture.cell_map[:] = _gen_rand_cell_map(L=culture.map_size, confl=0.7)

    # keep original state for comparison
    molecule_map = culture.molecule_map.clone()
    cell_map = culture.cell_map.clone()

    # save state
    culture.save_state(statedir=statedir)

    # change stuff
    culture.molecule_map = torch.rand_like(culture.molecule_map)
    culture.cell_map[:] = _gen_rand_cell_map(L=culture.map_size, confl=0.5)

    # load state
    culture.load_state(statedir=statedir)

    # check that the loaded state matches the original state
    assert (molecule_map == culture.molecule_map).all()
    assert (cell_map == culture.cell_map).all()
