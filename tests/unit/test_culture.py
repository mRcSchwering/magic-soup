import random
from pathlib import Path

import pytest
import torch
from magicsoup.chemistry import Chemistry, Molecule
from magicsoup.culture import Culture
from magicsoup.examples.wood_ljungdahl import CHEMISTRY

from tests.config import DEVICE

_ATOL = 1e-4
_RTOL = 1e-4


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


# TODO: move_cells
# TODO: get_neighbors
# TODO: find_free_random_positions


@pytest.mark.parametrize(
    "cells, idxs, migrate, exp_cells, exp_new_positions",
    [
        # no cell divides
        (
            [
                [1, 0, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [],
            False,
            [
                [1, 0, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [],
        ),
        # 1 cell can divide (only 1 choice)
        (
            [
                [1, 0, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [0],
            False,
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [(0, 1)],
        ),
        # all but one cell can divide
        (
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1],
            ],
            [0, 1, 2, 3, 4],
            False,
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [(0, 1), (1, 0), (1, 2), (2, 1)],
        ),
        # 1 cell cant divide (no free neighborhood)
        (
            [
                [1, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            [10],
            False,
            [
                [1, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            [],
        ),
        # 1 cell can migrate (only 1 choice)
        (
            [
                [1, 0, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [0],
            True,
            [
                [0, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [(0, 1)],
        ),
    ],
)
def test_resolve_placements(
    cells: list[list[int]],
    idxs: list[int],
    migrate: bool,
    exp_cells: list[list[int]],
    exp_new_positions: list[tuple[int, int]],
) -> None:
    culture = Culture(chemistry=CHEMISTRY, map_size=len(cells), device=DEVICE)
    culture.cell_map[:] = torch.tensor(cells, dtype=torch.bool, device=DEVICE)

    idx = torch.tensor(idxs, dtype=torch.int32, device=DEVICE)
    pos = culture.cell_map.nonzero().to(dtype=torch.int32)

    # pos will be edited in place for migrating cells
    successful_idxs, new_positions = culture._resolve_placements(
        cell_idxs=idx, cell_positions=pos, is_migration=migrate
    )

    # cell maps match
    cell_map = torch.tensor(exp_cells, dtype=torch.bool, device=DEVICE)
    torch.testing.assert_close(culture.cell_map, cell_map)

    # new positions match
    assert len(successful_idxs) == len(exp_new_positions)
    assert {tuple(d) for d in new_positions.tolist()} == set(exp_new_positions)

    # pos was correctly edited if migrating
    if migrate:
        cell_pos = culture.cell_map.nonzero().tolist()
        assert {tuple(d) for d in cell_pos} == {tuple(d) for d in pos.tolist()}


@pytest.mark.slow
@pytest.mark.parametrize("migrate", [True, False])
def test_resolve_placements_randomly(migrate: bool) -> None:
    n = 100
    for _ in range(100):
        culture = Culture(chemistry=CHEMISTRY, map_size=n, device=DEVICE)
        culture.cell_map[:] = torch.rand((n, n), device=DEVICE) < 0.7  # 70% confluency
        cells_before = culture.cell_map.clone()

        n_cells = int(culture.cell_map.sum().item())
        k = random.randint(10, n_cells)

        idx = torch.randperm(n_cells, dtype=torch.int32, device=DEVICE)[:k]
        pos = culture.cell_map.nonzero().to(dtype=torch.int32)

        # pos will be edited in place for migrating cells
        successful_idxs, new_positions = culture._resolve_placements(
            cell_idxs=idx, cell_positions=pos, is_migration=migrate, n_max_rounds=5
        )

        k_succ = new_positions.shape[0]
        cells_after = culture.cell_map

        # check dimensions and numbers make sense
        assert new_positions.shape[1] == 2
        assert successful_idxs.numel() == k_succ
        assert k_succ <= k

        if migrate:
            assert n_cells == cells_after.sum()
        else:
            assert k_succ + n_cells == cells_after.sum()

        # at least some cells should have been updated
        assert (cells_before != cells_after).any()

        # positions should have been updated for migrating cells
        if migrate:
            cell_pos = culture.cell_map.nonzero().tolist()
            assert {tuple(d) for d in cell_pos} == {tuple(d) for d in pos.tolist()}


def test_saving_loading(tmp_path: Path) -> None:
    statedir = tmp_path / "culture_state"

    # create culture with 70% confluency
    culture = Culture(chemistry=CHEMISTRY, device=DEVICE)
    culture.molecule_map[:] = torch.rand_like(culture.molecule_map)
    culture.cell_map[:] = torch.rand(culture.cell_map.shape, device=DEVICE) < 0.7

    # keep original state for comparison
    molecule_map = culture.molecule_map.clone()
    cell_map = culture.cell_map.clone()

    # save state
    culture.save_state(statedir=statedir)

    # change stuff
    culture.molecule_map = torch.rand_like(culture.molecule_map)
    culture.cell_map[:] = torch.rand(culture.cell_map.shape, device=DEVICE) < 0.5

    # load state
    culture.load_state(statedir=statedir)

    # check that the loaded state matches the original state
    assert (molecule_map == culture.molecule_map).all()
    assert (cell_map == culture.cell_map).all()
