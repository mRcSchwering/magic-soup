import random
from pathlib import Path

import numpy as np
import pytest
import torch
from magicsoup.cells import Cells
from magicsoup.examples.wood_ljungdahl import CHEMISTRY
from magicsoup.genomics import Genomics
from magicsoup.proteomics import Proteomics
from magicsoup.util import random_genome

from tests.util import FTYPE, ITYPE, idxtensor


@pytest.fixture
def cells() -> Cells:
    genomics = Genomics()
    proteomics = Proteomics(chemistry=CHEMISTRY)
    return Cells(chemistry=CHEMISTRY, genomics=genomics, proteomics=proteomics)


def _assert_empty_values(cells: Cells) -> None:
    # test empty values of cell parameters
    assert np.all(cells.genomes == "")
    assert np.all(cells.lineages == "")
    assert (cells.alive == False).all()
    assert (cells.ages == 0.0).all()
    assert (cells.generations == 0).all()
    assert (cells.molecules == 0.0).all()

    # test empty value of kinetics parameters
    assert (cells.v_max == 0.0).all()
    assert (cells.k_f == 0.0).all()
    assert (cells.k_b == 0.0).all()
    assert (cells.K_r == 0.0).all()
    assert (cells.N_f == 0).all()
    assert (cells.N_b == 0).all()
    assert (cells.N_h == 0).all()
    assert (cells.N_e == 0).all()


@pytest.mark.parametrize(
    "c_req, c_exp",
    [
        (100, 100),  # stays at 100
        (101, 122),  # updates including margin (20% of 101)
        (800, 960),  # updates including margin (20% of 800)
        (900, 1000),  # update is limited to 1000 (max)
        (1000, 1000),  # stays at 1000 (max)
        (83, 100),  # reduction too small (20% of 83)
        (82, 99),  # reduced and margin added (20% of 82)
        (40, 48),  # reduced and margin added (20% of 40)
        (1, 2),  # reduced and margin added (20% of 1)
        (0, 0),  # reduced to 0
    ],
)
def test_update_c(cells: Cells, c_req: int, c_exp: int) -> None:
    cells.dim_scaling = 0.2
    cells.c_max = 1000
    cells.update_c(c_req=83)  # leaves exactly c=100
    cells.update_c(c_req=c_req)

    # check c
    assert cells.c == c_exp

    # check exact shape of cell parameters
    assert cells.genomes.shape == (cells.c,)
    assert cells.lineages.shape == (cells.c,)
    assert cells.alive.shape == (cells.c,)
    assert cells.ages.shape == (cells.c,)
    assert cells.generations.shape == (cells.c,)
    assert cells.molecules.shape == (cells.c, len(cells.chemistry.molecules))

    # check exact shape of kinetics parameters
    assert cells.v_max.shape == (cells.c, cells.p)
    assert cells.k_f.shape == (cells.c, cells.p)
    assert cells.k_b.shape == (cells.c, cells.p)
    assert cells.K_r.shape == (cells.c, cells.p, cells.m)
    assert cells.N_f.shape == (cells.c, cells.p, cells.m)
    assert cells.N_b.shape == (cells.c, cells.p, cells.m)
    assert cells.N_h.shape == (cells.c, cells.p, cells.m)
    assert cells.N_e.shape == (cells.c, cells.p, cells.m)

    _assert_empty_values(cells=cells)


def test_update_c_keeps_living_cells(cells: Cells) -> None:
    cells.dim_scaling = 0.2
    cells.c_max = 1000
    cells.update_c(c_req=83)  # leaves exactly c=100

    # simulate 50 interspersed living cells
    idxs = list(range(0, 100, 2))
    cells.alive[idxs] = True
    assert cells.alive.sum() == 50

    # reduce to 60 (50 + 20%) cells
    cells.update_c(c_req=50)
    assert cells.c == 60

    # living cells should be preserved
    assert cells.alive.sum() == 50

    # reducing below 50 doesn't work with 50 living cells
    with pytest.raises(ValueError):
        cells.update_c(c_req=40)


@pytest.mark.parametrize(
    "p_req, p_exp",
    [
        (100, 100),  # stays at 100
        (101, 122),  # updates including margin (20% of 101)
        (800, 960),  # updates including margin (20% of 800)
        (900, 1000),  # update is limited to 1000 (max)
        (1000, 1000),  # stays at 1000 (max)
        (83, 100),  # reduction too small (20% of 83)
        (82, 99),  # reduced and margin added (20% of 82)
        (40, 48),  # reduced and margin added (20% of 40)
        (1, 2),  # reduced and margin added (20% of 1)
        (0, 0),  # reduced to 0
    ],
)
def test_update_p(cells: Cells, p_req: int, p_exp: int) -> None:
    cells.dim_scaling = 0.2
    cells.p_max = 1000
    cells.update_p(p_req=83)  # leaves exactly c=100
    cells.update_p(p_req=p_req)

    # check p
    assert cells.p == p_exp

    # check exact shape of cell parameters
    assert cells.genomes.shape == (cells.c,)
    assert cells.lineages.shape == (cells.c,)
    assert cells.alive.shape == (cells.c,)
    assert cells.ages.shape == (cells.c,)
    assert cells.generations.shape == (cells.c,)
    assert cells.molecules.shape == (cells.c, len(cells.chemistry.molecules))

    # check exact shape of kinetics parameters
    assert cells.v_max.shape == (cells.c, cells.p)
    assert cells.k_f.shape == (cells.c, cells.p)
    assert cells.k_b.shape == (cells.c, cells.p)
    assert cells.K_r.shape == (cells.c, cells.p, cells.m)
    assert cells.N_f.shape == (cells.c, cells.p, cells.m)
    assert cells.N_b.shape == (cells.c, cells.p, cells.m)
    assert cells.N_h.shape == (cells.c, cells.p, cells.m)
    assert cells.N_e.shape == (cells.c, cells.p, cells.m)

    _assert_empty_values(cells=cells)


@pytest.mark.slow
def test_update_genomes_randomly(cells: Cells) -> None:
    c = 100
    cells.update_c(c_req=c)
    p_init = cells.p

    for _ in range(100):
        n = random.randint(10, c)
        idxs = idxtensor(random.sample(range(c), n))
        genomes = [random_genome(s=500) for _ in range(n)]
        cells.update_genomes(genomes=genomes, idxs=idxs)

        # check genomes set
        assert np.all(cells.genomes[idxs] == genomes)

        # check kinetics parameters types
        ftensors = [cells.v_max, cells.k_b, cells.k_f, cells.K_r, cells.N_e]
        itensors = [cells.N_b, cells.N_f, cells.N_h]
        for tens in ftensors:
            assert tens.dtype == FTYPE
        for tens in itensors:
            assert tens.dtype == ITYPE

        # check at least some non-placeholder values
        denstensors = [cells.v_max, cells.k_b, cells.k_f, cells.N_f, cells.N_b]
        for tens in denstensors:
            per_cell = (tens[idxs] != 0).sum(tuple(range(1, tens.ndim)))
            assert (per_cell > 0).sum() / n > 0.8  # a few empty proteomes possible

    assert cells.p > p_init


@pytest.mark.slow
def test_copy_genomes_randomly(cells: Cells) -> None:
    c = 100
    cells.update_c(c_req=c)
    p_init = cells.p

    for _ in range(100):
        n = random.randint(1, c // 2)
        idxs0 = idxtensor(random.sample(range(c), n))
        genomes0 = [random_genome(s=500) for _ in range(n)]
        cells.update_genomes(genomes=genomes0, idxs=idxs0)

        idxs_left = list(set(range(c)) - set(idxs0.tolist()))
        idxs1 = idxtensor(random.sample(idxs_left, n))
        cells.copy_genomes(from_idxs=idxs0, to_idxs=idxs1)

        # check genomes set
        assert np.all(cells.genomes[idxs0] == cells.genomes[idxs1])

        # check kinetics parameters types
        ftensors = [cells.v_max, cells.k_b, cells.k_f, cells.K_r, cells.N_e]
        itensors = [cells.N_b, cells.N_f, cells.N_h]
        for tens in ftensors:
            assert tens.dtype == FTYPE
        for tens in itensors:
            assert tens.dtype == ITYPE

        # check at least some non-placeholder values
        for tens in ftensors + itensors:
            torch.testing.assert_close(tens[idxs0], tens[idxs1])

    assert cells.p > p_init


def test_saving_loading(cells: Cells, tmp_path: Path) -> None:
    statedir = tmp_path / "cells_state"

    # create interspersed cells
    c = 60
    cells.update_c(c_req=c)
    idxs = idxtensor(list(range(0, c, 2)))
    genomes_ = [random_genome() for _ in range(idxs.numel())]
    cells.update_genomes(genomes=genomes_, idxs=idxs)
    cells.alive[idxs] = True

    # keep original state for comparison
    genomes = cells.genomes.copy()
    lineages = cells.lineages.copy()
    alive = cells.alive.clone()
    ages = cells.ages.clone()
    generations = cells.generations.clone()
    molecules = cells.molecules.clone()

    v_max = cells.v_max.clone()
    k_f = cells.k_f.clone()
    k_b = cells.k_b.clone()
    K_r = cells.K_r.clone()
    N_f = cells.N_f.clone()
    N_b = cells.N_b.clone()
    N_h = cells.N_h.clone()
    N_e = cells.N_e.clone()

    # save state
    cells.save_state(statedir=statedir)

    # clear cells
    cells.alive[:] = False
    cells.update_p(p_req=0)
    cells.update_c(c_req=0)

    # load state
    cells.load_state(statedir=statedir)

    # check that the loaded state matches the original state
    assert np.all(cells.genomes == genomes)
    assert np.all(cells.lineages == lineages)
    assert (cells.alive == alive).all()
    assert (cells.ages == ages).all()
    assert (cells.generations == generations).all()
    assert (cells.molecules == molecules).all()

    # check kinetics parameters
    assert (cells.v_max == v_max).all()
    assert (cells.k_f == k_f).all()
    assert (cells.k_b == k_b).all()
    assert (cells.K_r == K_r).all()
    assert (cells.N_f == N_f).all()
    assert (cells.N_b == N_b).all()
    assert (cells.N_h == N_h).all()
    assert (cells.N_e == N_e).all()
