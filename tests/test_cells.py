import time
import pytest
import torch
from util import rand_genome
from genetics import (
    Genetics,
    ReceptorDomain,
    SynthesisDomain,
    MA,
    MB,
    MC,
    MD,
    ME,
    MOLECULES,
    ACTIONS,
)
from cells import Cells

TOLERANCE = 1e-3


def test_cell_time_step():
    # fmt: off
    cell = [
        {
            ReceptorDomain(MA): (0.1, None),
            SynthesisDomain(MB): (None, 0.8),
            SynthesisDomain(MC): (None, 0.5),
            SynthesisDomain(ME): (None, 0.3),
        },
        {
            ReceptorDomain(MD): (0.4, None),
            SynthesisDomain(MC): (None, 0.9),
            SynthesisDomain(MD): (None, 0.4),
        },
        {
            ReceptorDomain(MC): (0.2, None),
            SynthesisDomain(MD): (None, 0.3)
        },
        {
            ReceptorDomain(ME): (0.6, None),
            SynthesisDomain(MB): (None, 0.7),
            SynthesisDomain(ME): (None, 0.8),
        },
    ]
    # fmt: on

    dim1 = len(MOLECULES) * 2

    # initial concentrations
    C0 = torch.zeros(1, dim1)
    C0[0, 0] = 1.5  # c0_a
    C0[0, 1] = 1.1  # c0_b
    C0[0, 2] = 1.2  # c0_c
    C0[0, 3] = 1.3  # c0_d
    C0[0, 4] = 1.4  # c0_e

    # by hand with:
    # def f(x: float) -> float:
    #    return (1 - math.exp(-(x ** 3)))
    c1_a = 0.0  # no edge points to a
    c1_b = 0.3157  # f(c0_a * 0.1) * 0.8 + f(c0_e * 0.6) * 0.7
    c1_c = 0.1197  # f(c0_a * 0.1) * 0.5 + f(c0_d * 0.4) * 0.9
    c1_d = 0.0565  # f(c0_d * 0.4) * 0.4 + f(c0_c * 0.2) * 0.3
    c1_e = 0.3587  # f(c0_a * 0.1) * 0.3 + f(c0_e * 0.6) * 0.8

    cells = Cells(molecules=MOLECULES, actions=[], max_proteins=4)

    A, B = cells.get_cell_params(cells=[cell])

    assert A.shape == (1, dim1, 4)
    assert B.shape == (1, dim1, 4)

    C1 = cells.simulate_protein_work(C=C0, A=A, B=B)

    assert C1.shape == (1, dim1)
    assert C1[0, 0] == pytest.approx(c1_a, abs=TOLERANCE)
    assert C1[0, 1] == pytest.approx(c1_b, abs=TOLERANCE)
    assert C1[0, 2] == pytest.approx(c1_c, abs=TOLERANCE)
    assert C1[0, 3] == pytest.approx(c1_d, abs=TOLERANCE)
    assert C1[0, 4] == pytest.approx(c1_e, abs=TOLERANCE)
    for i in range(5, dim1):
        assert C1[0, i] == 0.0


def test_performance():
    genetics = Genetics()
    cells = Cells(molecules=MOLECULES, actions=ACTIONS)
    gs = [rand_genome((1000, 5000)) for _ in range(100)]
    prtms = [genetics.get_proteome(g=d) for d in gs]

    # TODO: improve get cell params performance
    t0 = time.time()
    A, B = cells.get_cell_params(cells=prtms)
    td = time.time() - t0
    assert td < 0.5, "get_cell_params performance degraded a lot"

    C = torch.randn(len(prtms), len(ACTIONS) + 2 * len(MOLECULES))

    t0 = time.time()
    _ = cells.simulate_protein_work(C=C, A=A, B=B)
    td = time.time() - t0
    assert td < 0.01, "get_cell_params performance degraded a lot"

