import time
import pytest
import torch
from util import rand_genome
from genetics import ReceptorDomain, SynthesisDomain, Messenger, Cytokine, Genetics
from cells import Cells

TOLERANCE = 1e-3


def test_cell_time_step():
    # fmt: off
    cell = [
        {
            ReceptorDomain(Cytokine.F): 0.1,
            SynthesisDomain(Messenger.MA): 0.8,
            SynthesisDomain(Messenger.MB): 0.5,
            SynthesisDomain(Messenger.MD): 0.3,
        },
        {
            ReceptorDomain(Messenger.MC): 0.4,
            SynthesisDomain(Messenger.MB): 0.9,
            SynthesisDomain(Messenger.MC): 0.4,
        },
        {
            ReceptorDomain(Messenger.MB): 0.2,
            SynthesisDomain(Messenger.MC): 0.3
        },
        {
            ReceptorDomain(Messenger.MD): 0.6,
            SynthesisDomain(Messenger.MA): 0.7,
            SynthesisDomain(Messenger.MD): 0.8,
        },
    ]
    # fmt: on

    # initial concentrations
    c0_a = 1.1
    c0_b = 1.2
    c0_c = 1.3
    c0_d = 1.4
    c0_f = 1.5
    C0 = torch.tensor([[0.0, c0_a, c0_b, c0_c, c0_d, c0_f, 0.0, 0.0]])

    # by hand with:
    # def f(x: float) -> float:
    #    return (1 - math.exp(-(x ** 3)))
    c1_a = 0.3157  # f(c0_f * 0.1) * 0.8 + f(c0_d * 0.6) * 0.7
    c1_b = 0.1197  # f(c0_f * 0.1) * 0.5 + f(c0_c * 0.4) * 0.9
    c1_c = 0.0565  # f(c0_c * 0.4) * 0.4 + f(c0_b * 0.2) * 0.3
    c1_d = 0.3587  # f(c0_f * 0.1) * 0.3 + f(c0_d * 0.6) * 0.8
    c1_f = 0.0  # no edge points to f

    cells = Cells(n_world_signals=3, n_cell_signals=5, max_proteins=4)

    A, B = cells.get_cell_params(cells=[cell])
    assert A.shape == (1, 8, 4)
    assert B.shape == (1, 8, 4)

    C1 = cells.simulate_protein_work(C=C0, A=A, B=B)
    assert C1.shape == (1, 8)
    assert C1[0, 0] == 0.0
    assert C1[0, 1] == pytest.approx(c1_a, abs=TOLERANCE)
    assert C1[0, 2] == pytest.approx(c1_b, abs=TOLERANCE)
    assert C1[0, 3] == pytest.approx(c1_c, abs=TOLERANCE)
    assert C1[0, 4] == pytest.approx(c1_d, abs=TOLERANCE)
    assert C1[0, 5] == pytest.approx(c1_f, abs=TOLERANCE)
    assert C1[0, 6] == 0.0
    assert C1[0, 7] == 0.0


def test_performance():
    genetics = Genetics()
    cells = Cells(n_world_signals=3, n_cell_signals=5)
    gs = [rand_genome((1000, 5000)) for _ in range(100)]
    prtms = [genetics.get_proteome(g=d) for d in gs]

    # TODO: improve get cell params performance
    t0 = time.time()
    A, B = cells.get_cell_params(cells=prtms)
    td = time.time() - t0
    assert td < 0.5, "get_cell_params performance degraded a lot"

    C = torch.randn(len(prtms), 8)

    t0 = time.time()
    _ = cells.simulate_protein_work(C=C, A=A, B=B)
    td = time.time() - t0
    assert td < 0.01, "get_cell_params performance degraded a lot"

