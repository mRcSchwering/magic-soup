import random

import pytest
from magicsoup.cells import Cells
from magicsoup.examples.wood_ljungdahl import CHEMISTRY
from magicsoup.genomics import Genomics
from magicsoup.proteomics import Proteomics
from magicsoup.util import random_genome


@pytest.fixture
def cells() -> Cells:
    genomics = Genomics()
    proteomics = Proteomics(chemistry=CHEMISTRY)
    return Cells(chemistry=CHEMISTRY, genomics=genomics, proteomics=proteomics)


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
    assert len(cells.genomes) == cells.c
    assert len(cells.labels) == cells.c
    assert cells.alive.shape == (cells.c,)
    assert cells.ages.shape == (cells.c,)
    assert cells.positions.shape == (cells.c, 2)
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
    assert cells.N.shape == (cells.c, cells.p, cells.m)


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
    assert len(cells.genomes) == cells.c
    assert len(cells.labels) == cells.c
    assert cells.alive.shape == (cells.c,)
    assert cells.ages.shape == (cells.c,)
    assert cells.positions.shape == (cells.c, 2)
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
    assert cells.N.shape == (cells.c, cells.p, cells.m)


@pytest.mark.slow
def test_update_genomes_randomly(cells: Cells) -> None:
    c = 100
    cells.update_c(c_req=c)
    p_init = cells.p

    for _ in range(100):
        n = random.randint(1, c)
        idxs = random.sample(range(c), n)
        genomes = [random_genome(s=500) for _ in range(n)]
        cells.update_genomes(genomes=genomes, idxs=idxs)

    assert cells.p > p_init
