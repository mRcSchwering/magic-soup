import time
import pytest
import numpy as np
from util import rand_genome
from genetics import (
    Signal,
    get_cell_params,
    simulate_protein_work,
    assert_config,
    get_proteome,
)

TOLERANCE = 1e-3


def test_cell_time_step():
    # fmt: off
    cell = [
        {
            ("RcF", Signal.F, True): 0.1,
            ("OutA", Signal.MA, False): 0.8,
            ("OutB", Signal.MB, False): 0.5,
            ("OutD", Signal.MD, False): 0.3,
        },
        {
            ("InC", Signal.MC, True): 0.4,
            ("OutB", Signal.MB, False): 0.9,
            ("OutC", Signal.MC, False): 0.4,
        },
        {
            ("InB", Signal.MB, True): 0.2,
            ("OutC", Signal.MC, False): 0.3
        },
        {
            ("InD", Signal.MD, True): 0.6,
            ("OutA", Signal.MA, False): 0.7,
            ("OutD", Signal.MD, False): 0.8,
        },
    ]
    # fmt: on

    # initial concentrations
    c0_a = 1.1
    c0_b = 1.2
    c0_c = 1.3
    c0_d = 1.4
    c0_f = 1.5
    C0 = np.array([[c0_f, 0.0, c0_a, c0_b, c0_c, c0_d, 0.0, 0.0]])

    # by hand with:
    # def f(x: float) -> float:
    #    return (1 - math.exp(-(x ** 3)))
    c1_f = 0.0  # no edge points to f
    c1_a = 0.3157  # f(c0_f * 0.1) * 0.8 + f(c0_d * 0.6) * 0.7
    c1_b = 0.1197  # f(c0_f * 0.1) * 0.5 + f(c0_c * 0.4) * 0.9
    c1_c = 0.0565  # f(c0_c * 0.4) * 0.4 + f(c0_b * 0.2) * 0.3
    c1_d = 0.3587  # f(c0_f * 0.1) * 0.3 + f(c0_d * 0.6) * 0.8

    A, B = get_cell_params(cells=[cell])
    assert A.shape == (1, 8, 4)
    assert B.shape == (1, 8, 4)

    C1 = simulate_protein_work(C=C0, A=A, B=B)
    assert C1.shape == (1, 8)
    assert C1[0, 0] == pytest.approx(c1_f, abs=TOLERANCE)
    assert C1[0, 1] == 0.0
    assert C1[0, 2] == pytest.approx(c1_a, abs=TOLERANCE)
    assert C1[0, 3] == pytest.approx(c1_b, abs=TOLERANCE)
    assert C1[0, 4] == pytest.approx(c1_c, abs=TOLERANCE)
    assert C1[0, 5] == pytest.approx(c1_d, abs=TOLERANCE)
    assert C1[0, 6] == 0.0
    assert C1[0, 7] == 0.0


def test_performance():
    gs = [rand_genome((1000, 5000)) for _ in range(100)]

    # TODO: improve CDS translation performance
    t0 = time.time()
    cells = [get_proteome(g=d, ignore_cds=False) for d in gs]
    td = time.time() - t0
    assert td < 0.4, "get_proteome performance degraded a lot"

    t0 = time.time()
    cells = [get_proteome(g=d, ignore_cds=True) for d in gs]
    td = time.time() - t0
    assert td - t0 < 0.05, "get_proteome performance degraded a lot"

    t0 = time.time()
    A, B = get_cell_params(cells=cells)
    td = time.time() - t0
    assert td < 0.01, "get_cell_params performance degraded a lot"

    C = np.random.random((len(cells), len(Signal)))

    t0 = time.time()
    _ = simulate_protein_work(C=C, A=A, B=B)
    td = time.time() - t0
    assert td < 0.01, "get_cell_params performance degraded a lot"
