import random
from pathlib import Path

import pytest
import torch
from magicsoup.cells import Cells
from magicsoup.core.strarray import StrArray
from magicsoup.core.util import random_genome
from magicsoup.examples.wood_ljungdahl import CHEMISTRY
from magicsoup.genomics import Genomics
from magicsoup.proteomics import Proteomics

from tests.config import FTYPE, IDXTYPE, ITYPE
from tests.util import idxfull, idxtensor


@pytest.fixture
def cells() -> Cells:
    genomics = Genomics()
    proteomics = Proteomics(chemistry=CHEMISTRY)
    return Cells(chemistry=CHEMISTRY, genomics=genomics, proteomics=proteomics)


def _assert_empty_values(cells: Cells) -> None:
    # test empty values of cell parameters
    assert all(d == "" for d in cells.genomes)
    assert all(d == "" for d in cells.lineages)
    assert (cells.alive == False).all()
    assert (cells.ages == 0.0).all()
    assert (cells.generations == 0).all()
    assert (cells.proteins == 0).all()
    assert (cells.molecules == 0.0).all()

    # test empty value of kinetics parameters
    assert (cells.v_max == 0.0).all()
    assert (cells.k_f == 0.0).all()
    assert (cells.k_b == 0.0).all()
    assert (cells.k_r == 0.0).all()
    assert (cells.N_f == 0).all()
    assert (cells.N_b == 0).all()
    assert (cells.N_h == 0).all()
    assert (cells.N_e == 0).all()


def _assert_shapes(cells: Cells) -> None:
    # check exact shape of cell parameters
    assert cells.genomes.shape == (cells.c,)
    assert cells.lineages.shape == (cells.c,)
    assert cells.alive.shape == (cells.c,)
    assert cells.ages.shape == (cells.c,)
    assert cells.generations.shape == (cells.c,)
    assert cells.proteins.shape == (cells.c,)
    assert cells.molecules.shape == (cells.c, len(cells.chemistry.molecules))

    # check exact shape of kinetics parameters
    assert cells.v_max.shape == (cells.c, cells.p)
    assert cells.k_f.shape == (cells.c, cells.p)
    assert cells.k_b.shape == (cells.c, cells.p)
    assert cells.k_r.shape == (cells.c, cells.p)
    assert cells.N_f.shape == (cells.c, cells.p, cells.m)
    assert cells.N_b.shape == (cells.c, cells.p, cells.m)
    assert cells.N_h.shape == (cells.c, cells.p, cells.m)
    assert cells.N_e.shape == (cells.c, cells.p, cells.m)


def _asser_dtypes(cells: Cells) -> None:
    # check exact shape of cell parameters
    assert isinstance(cells.genomes, StrArray)
    assert isinstance(cells.lineages, StrArray)
    assert cells.alive.dtype == torch.bool
    assert cells.ages.dtype == FTYPE
    assert cells.generations.dtype == IDXTYPE
    assert cells.proteins.dtype == IDXTYPE
    assert cells.molecules.dtype == FTYPE

    # check exact shape of kinetics parameters
    assert cells.v_max.dtype == FTYPE
    assert cells.k_f.dtype == FTYPE
    assert cells.k_b.dtype == FTYPE
    assert cells.k_r.dtype == FTYPE
    assert cells.N_f.dtype == ITYPE
    assert cells.N_b.dtype == ITYPE
    assert cells.N_h.dtype == ITYPE
    assert cells.N_e.dtype == FTYPE


@pytest.mark.parametrize(
    "c_req, c_exp",
    [
        (100, 100),  # stays at 100
        (101, 122),  # grows including margin (20% of 101)
        (800, 960),  # grows including margin (20% of 800)
        (900, 1000),  # grow limited to 1000 (max)
        (1000, 1000),  # stays at 1000 (max)
        (83, 100),  # no shrink: stays at 100
        (82, 100),  # no shrink: stays at 100
        (40, 100),  # no shrink: stays at 100
        (1, 100),  # no shrink: stays at 100
        (0, 100),  # no shrink: stays at 100
    ],
)
def test_update_c(cells: Cells, c_req: int, c_exp: int) -> None:
    cells.dim_scaling = 0.2
    cells.c_max = 1000
    cells.update_c(c_req=83)  # leaves exactly c=100
    c = cells.update_c(c_req=c_req)

    # check c (update_c only grows)
    assert cells.c == c_exp
    assert c == cells.c
    _assert_shapes(cells)
    _assert_empty_values(cells=cells)


@pytest.mark.parametrize(
    "p_req, p_exp",
    [
        (100, 100),  # stays at 100
        (101, 122),  # grows including margin (20% of 101)
        (800, 960),  # grows including margin (20% of 800)
        (900, 1000),  # grow limited to 1000 (max)
        (1000, 1000),  # stays at 1000 (max)
        (83, 100),  # no shrink: stays at 100
        (82, 100),  # no shrink: stays at 100
        (40, 100),  # no shrink: stays at 100
        (1, 100),  # no shrink: stays at 100
        (0, 100),  # no shrink: stays at 100
    ],
)
def test_update_p(cells: Cells, p_req: int, p_exp: int) -> None:
    cells.dim_scaling = 0.2
    cells.p_max = 1000
    cells.update_p(p_req=83)  # leaves exactly p=100
    p = cells.update_p(p_req=p_req)

    # check p (update_p only grows)
    assert cells.p == p_exp
    assert cells.p == p
    _assert_shapes(cells)
    _assert_empty_values(cells=cells)


def test_compact_keeps_living_cells(cells: Cells) -> None:
    cells.dim_scaling = 0.2
    cells.c_max = 1000
    cells.update_c(c_req=83)  # leaves exactly c=100

    # simulate 50 interspersed living cells
    idxs = list(range(0, 100, 2))
    cells.alive[idxs] = True
    assert cells.alive.sum() == 50
    alive_old = cells.alive.clone()

    new_idx = cells.compact()

    # shrinks to 50 living + 20% hysteresis, keeping all living cells
    assert cells.c == 60
    assert cells.alive.sum() == 50

    # old -> new map: 100 old slots, 60 kept, 40 dropped
    assert new_idx.shape == (100,)
    assert (new_idx == -1).sum() == 40

    # living cells get distinct, valid new indices that point to living slots
    living_new = new_idx[alive_old]
    assert (living_new >= 0).all()
    assert living_new.unique().numel() == 50
    assert cells.alive[living_new].all()

    _assert_shapes(cells)


def test_get_utilization(cells: Cells) -> None:
    cells.dim_scaling = 0.2
    cells.c_max = 1000
    cells.p_max = 1000

    # nothing allocated -> fully utilized
    assert cells.get_utilization() == 1.0

    # fill 100 living cells with real proteomes
    cells.update_c(c_req=100)
    idxs = idxtensor(list(range(100)))
    genomes = StrArray.from_strings([random_genome(s=500) for _ in range(100)])
    cells.update_genomes(genomes=genomes, idxs=idxs)
    cells.alive[idxs] = True
    u_full = cells.get_utilization()
    assert 0.0 < u_full <= 1.0

    # "kill" 90 cells (fewer alive, params zeroed) lowers utilization
    dead = idxtensor(list(range(90)))
    cells.alive[dead] = False
    cells.v_max[dead] = 0.0
    cells.k_r[dead] = 0.0
    u_sparse = cells.get_utilization()
    assert u_sparse < u_full

    # compacting reclaims the slack and restores utilization
    cells.compact()
    assert cells.get_utilization() > u_sparse


def test_update_empty_genome(cells: Cells) -> None:
    idxs = idxtensor([0])
    genomes = StrArray.from_strings([random_genome(s=0) for _ in range(1)])
    cells.update_genomes(genomes=genomes, idxs=idxs)
    _asser_dtypes(cells)


# TODO: test proteome param distros by random genomes make sense
#       K_e: med 1, 70% (1e-4, 1e4), 99% (1e-8, 1e8)
#       N_h: med 1.5, 95% (1, 3)
#       K_m: med 0.1, 90% (0.003, 3)
#       Vmax: med 1e-2, 90% (1e-4, 1)


@pytest.mark.slow
def test_update_genomes_randomly(cells: Cells) -> None:
    c = 100
    cells.update_c(c_req=c)
    p_init = cells.p

    for _ in range(100):
        n = random.randint(10, c)
        idxs = idxtensor(random.sample(range(c), n))
        genomes = StrArray.from_strings([random_genome(s=500) for _ in range(n)])
        cells.update_genomes(genomes=genomes, idxs=idxs)

        _asser_dtypes(cells)

        # check genomes set
        assert cells.genomes[idxs.cpu().numpy()].tolist() == genomes.tolist()

        # check at least some non-placeholder values
        denstensors = [cells.v_max, cells.k_b, cells.k_f, cells.N_f, cells.N_b]
        for tens in denstensors:
            per_cell = (tens[idxs] != 0).sum(tuple(range(1, tens.ndim)))
            assert (per_cell > 0).sum() / n > 0.8  # a few empty proteomes possible

        # check protein counts
        prot_exp = ((cells.v_max != 0.0) | (cells.k_r != 0)).sum(dim=1).to(IDXTYPE)
        torch.testing.assert_close(cells.proteins, prot_exp)

    assert cells.p > p_init


@pytest.mark.slow
def atest_copy_proteomes_randomly(cells: Cells) -> None:
    # TODO: test for update_descendants instead
    # TODO: also test that proteins was copied

    c = 100
    cells.update_c(c_req=c)
    p_init = cells.p

    for _ in range(100):
        n = random.randint(1, c // 2)
        from_idxs = idxtensor(random.sample(range(c), n))
        genomes0 = StrArray.from_strings([random_genome(s=500) for _ in range(n)])
        cells.update_genomes(genomes=genomes0, idxs=from_idxs)

        idxs_left = list(set(range(c)) - set(from_idxs.tolist()))
        to_idxs = idxtensor(random.sample(idxs_left, n))
        parent_of = idxfull(cells.c, value=-1)
        parent_of[from_idxs] = to_idxs
        cells.copy_proteomes(parent_of=parent_of)

        _asser_dtypes(cells)

        # values were copied
        ftensors = [cells.v_max, cells.k_b, cells.k_f, cells.k_r, cells.N_e]
        itensors = [cells.N_b, cells.N_f, cells.N_h]
        for tens in ftensors + itensors:
            torch.testing.assert_close(tens[from_idxs], tens[to_idxs])

    assert cells.p > p_init


def test_saving_loading(cells: Cells, tmp_path: Path) -> None:
    statedir = tmp_path / "cells_state"

    # create interspersed cells
    c = 60
    cells.update_c(c_req=c)
    idxs = idxtensor(list(range(0, c, 2)))
    genomes_ = StrArray.from_strings([random_genome() for _ in range(idxs.numel())])
    cells.update_genomes(genomes=genomes_, idxs=idxs)
    cells.alive[idxs] = True

    # keep original state for comparison
    genomes = cells.genomes.copy()
    lineages = cells.lineages.copy()
    alive = cells.alive.clone()
    ages = cells.ages.clone()
    generations = cells.generations.clone()
    proteins = cells.proteins.clone()
    molecules = cells.molecules.clone()

    v_max = cells.v_max.clone()
    k_f = cells.k_f.clone()
    k_b = cells.k_b.clone()
    k_r = cells.k_r.clone()
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
    assert cells.genomes.tolist() == genomes.tolist()
    assert cells.lineages.tolist() == lineages.tolist()
    assert (cells.alive == alive).all()
    assert (cells.ages == ages).all()
    assert (cells.generations == generations).all()
    assert (cells.proteins == proteins).all()
    assert (cells.molecules == molecules).all()

    # check kinetics parameters
    assert (cells.v_max == v_max).all()
    assert (cells.k_f == k_f).all()
    assert (cells.k_b == k_b).all()
    assert (cells.k_r == k_r).all()
    assert (cells.N_f == N_f).all()
    assert (cells.N_b == N_b).all()
    assert (cells.N_h == N_h).all()
    assert (cells.N_e == N_e).all()
