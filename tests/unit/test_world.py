import logging

import pytest
from magicsoup.culture import Culture
from magicsoup.examples.wood_ljungdahl import CHEMISTRY
from magicsoup.util import random_genome
from magicsoup.world import World


@pytest.fixture
def world() -> World:
    culture = Culture(chemistry=CHEMISTRY, size=10)
    culture.molecules[:] = 10.0
    return World(chemistry=CHEMISTRY, culture=culture)


def test_spawn_cells(caplog, world: World) -> None:
    # spawn 90 cells
    with caplog.at_level(logging.WARNING):
        world.spawn_cells(genomes=[random_genome() for _ in range(90)])

    # all 90 cells had space
    assert len(caplog.records) == 0

    # map is occupied by 90 cells and they picked up half of the molecules
    assert world.culture.cells.sum() == 90
    assert (world.culture.molecules[:, world.culture.cells] == 5.0).all()
    assert (world.culture.molecules[:, ~world.culture.cells] == 10.0).all()

    # 90 cells are alive and picked up half the molecules
    assert world.cells.c >= 90
    assert world.cells.p > 0
    assert world.cells.alive.sum() == 90
    assert (world.cells.molecules[world.cells.alive] == 5.0).all()
    assert (world.cells.molecules[~world.cells.alive] == 0.0).all()

    with caplog.at_level(logging.WARNING):
        world.spawn_cells(genomes=[random_genome() for _ in range(20)])

    # 10 cells did not have space
    assert len(caplog.records) > 0

    # map is at full confluency
    assert world.culture.cells.sum() == 100
    assert (world.culture.molecules == 5.0).all()

    # all cells are alive and picked up half the molecules
    assert world.cells.c == 100
    assert world.cells.p > 0
    assert world.cells.alive.sum() == 100
    assert (world.cells.molecules == 5.0).all()


# TODO: test add cells


def test_enzymatic_activity(world: World) -> None:
    world.spawn_cells(genomes=[random_genome() for _ in range(100)])

    # initially cells pickup molecules
    assert (world.cells.molecules == 5.0).all()

    world.enzymatic_activity()

    # at least some cells should have converted molecules
    assert (world.cells.molecules != 5.0).any()


def test_diffuse_molecules(world: World) -> None:
    world.culture.molecules[0, 5, 5] = 0.0

    # molecules should diffuse into empty spot
    world.diffuse_molecules()
    assert world.culture.molecules[0, 5, 5] > 0.0
