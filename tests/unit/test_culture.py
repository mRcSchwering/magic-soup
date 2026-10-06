import pytest
import torch
from magicsoup.chemistry import Chemistry, Molecule
from magicsoup.culture import Culture
from magicsoup.examples.wood_ljungdahl import CHEMISTRY, MOLECULES

_ATOL = 1e-4
_RTOL = 1e-4


def test_diffuse():
    molecules = torch.tensor(
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
    culture = Culture(chemistry=chemistry, size=5)
    culture.molecules = molecules

    culture.diffuse_molecules()

    torch.testing.assert_close(culture.molecules, exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize(
    "x, y, exp",
    [
        (2, 2, [(1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3), (2, 1), (2, 3)]),
        (0, 0, [(4, 4), (4, 0), (4, 1), (1, 4), (1, 0), (1, 1), (0, 4), (0, 1)]),
        (4, 4, [(3, 3), (3, 4), (3, 0), (0, 3), (0, 4), (0, 0), (4, 3), (4, 0)]),
        (0, 4, [(4, 3), (4, 4), (4, 0), (1, 3), (1, 4), (1, 0), (0, 3), (0, 0)]),
        (4, 0, [(3, 4), (4, 4), (0, 4), (3, 1), (4, 1), (0, 1), (3, 0), (0, 0)]),
    ],
)
def test_free_moores_nghbhd(x: int, y: int, exp: list[tuple[int, int]]):
    culture = Culture(chemistry=CHEMISTRY, size=5)

    res = culture.free_moores_nghbhd(x=x, y=y, positions=[])
    assert set(res) == set(exp)

    occ = res[0]
    res1 = culture.free_moores_nghbhd(x=x, y=y, positions=[occ])
    assert set(res1) == set(exp) - {occ}

    res2 = culture.free_moores_nghbhd(x=x, y=y, positions=res)
    assert len(res2) == 0


# TODO: move_cells
# TODO: get_neighbors
# TODO: find_free_random_positions


@pytest.mark.parametrize(
    "cells, exp_parent_idxs, exp_child_idxs, exp_new_positions",
    [
        # first cell can divice (only 1 choice)
        (
            [
                [1, 0, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [0],
            [8],
            [(0, 1)],
        ),
        (
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [],
            [],
            [],
        ),
    ],
)
def test_divide_cells_if_possible(
    cells: list[list[int]],
    exp_parent_idxs: list[int],
    exp_child_idxs: list[int],
    exp_new_positions: list[tuple[int, int]],
) -> None:
    chemistry = Chemistry(molecules=MOLECULES[:2], reactions=[])
    culture = Culture(chemistry=chemistry, size=3)
    culture.cells[:] = torch.tensor(cells, dtype=torch.bool)

    positions = [tuple(d) for d in culture.cells.nonzero().tolist()]
    n = len(positions)
    idxs = list(range(n))
    parent_idxs, child_idxs, new_positions = culture.divide_cells_if_possible(
        cell_idxs=idxs, positions=positions, n_cells=n
    )
    assert parent_idxs == exp_parent_idxs
    assert child_idxs == exp_child_idxs
    assert new_positions == exp_new_positions


# TODO: save/load state
