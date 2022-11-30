import time
import pytest
import torch
from util import rand_genome
from genetics import (
    Genetics,
    ReceptorDomainFact,
    SynthesisDomainFact,
    Protein,
    MA,
    MB,
    MC,
    MD,
    ME,
    MOLECULES,
    ACTIONS,
    DOMAINS,
)
from cells import Cells

TOLERANCE = 1e-4


def test_cell_signal_integration():
    # fmt: off
    domains = [
        [
            ReceptorDomainFact(MA)(0.1),
            SynthesisDomainFact(MB)(0.8),
            SynthesisDomainFact(MC)(0.5),
            SynthesisDomainFact(ME)(0.3),
        ],
        [
            ReceptorDomainFact(MD)(0.4),
            SynthesisDomainFact(MC)(0.9),
            SynthesisDomainFact(MD)(0.4),
        ],
        [
            ReceptorDomainFact(MC)(0.2),
            SynthesisDomainFact(MD)(0.3)
        ],
        [
            ReceptorDomainFact(ME)(0.6),
            SynthesisDomainFact(MB)(0.7),
            SynthesisDomainFact(ME)(0.8),
        ],
    ]
    # fmt: on

    prtm = [Protein(domains=d, energy=0) for d in domains]
    dim1 = len(MOLECULES) * 2

    # initial concentrations
    X0 = torch.zeros(1, dim1)
    X0[0, 0] = 1.5  # x0_a
    X0[0, 1] = 1.1  # x0_b
    X0[0, 2] = 1.2  # x0_c
    X0[0, 3] = 1.3  # x0_d
    X0[0, 4] = 1.4  # x0_e

    # by hand with:
    # def f(x: float) -> float:
    #    return (1 - math.exp(-(x ** 3)))
    x1_a = 0.0  # no edge points to a
    x1_b = 0.3157  # f(x0_a * 0.1) * 0.8 + f(x0_e * 0.6) * 0.7
    x1_c = 0.1197  # f(x0_a * 0.1) * 0.5 + f(x0_d * 0.4) * 0.9
    x1_d = 0.0565  # f(x0_d * 0.4) * 0.4 + f(x0_c * 0.2) * 0.3
    x1_e = 0.3587  # f(x0_a * 0.1) * 0.3 + f(x0_e * 0.6) * 0.8

    cells = Cells(molecules=MOLECULES, actions=[], n_max_proteins=4, trunc_n_decs=5)

    A, B, Z = cells.get_cell_params(proteomes=[prtm])

    assert A.shape == (1, dim1, 4)
    assert B.shape == (1, dim1, 4)
    assert Z.shape == (1, 4)

    X1 = cells.simulate_protein_work(X=X0, A=A, B=B, Z=Z)

    assert X1.shape == (1, dim1)
    assert X1[0, 0] == pytest.approx(x1_a, abs=TOLERANCE)
    assert X1[0, 1] == pytest.approx(x1_b, abs=TOLERANCE)
    assert X1[0, 2] == pytest.approx(x1_c, abs=TOLERANCE)
    assert X1[0, 3] == pytest.approx(x1_d, abs=TOLERANCE)
    assert X1[0, 4] == pytest.approx(x1_e, abs=TOLERANCE)
    for i in range(5, dim1):
        assert X1[0, i] == 0.0


def test_switching_off_proteins_by_energy():
    # fmt: off
    domains = [
        [
            ReceptorDomainFact(MA)(0.1),
            SynthesisDomainFact(MB)(0.8),
            SynthesisDomainFact(MC)(0.5),
            SynthesisDomainFact(ME)(0.3),
        ],
        [
            ReceptorDomainFact(MD)(0.4),
            SynthesisDomainFact(MC)(0.9),
            SynthesisDomainFact(MD)(0.4),
        ],
        [
            ReceptorDomainFact(MC)(0.2),
            SynthesisDomainFact(MD)(0.3)
        ],
        [
            ReceptorDomainFact(ME)(0.6),
            SynthesisDomainFact(MB)(0.7),
            SynthesisDomainFact(ME)(0.8),
        ],
    ]
    # fmt: on

    prtm = [Protein(domains=d, energy=0) for d in domains]
    dim1 = len(MOLECULES) * 2

    # switch protein 0, 1 off
    prtm[0].energy = 1  # energetically not preferred
    prtm[1].energy = 1  # energetically not preferred

    # initial concentrations
    X0 = torch.zeros(1, dim1)
    X0[0, 0] = 1.5  # x0_a
    X0[0, 1] = 1.1  # x0_b
    X0[0, 2] = 1.2  # x0_c
    X0[0, 3] = 1.3  # x0_d
    X0[0, 4] = 1.4  # x0_e

    # by hand with:
    # def f(x: float) -> float:
    #    return (1 - math.exp(-(x ** 3)))
    x1_a = 0.0  # no edge points to a
    x1_b = 0.3130  # f(x0_e * 0.6) * 0.7
    x1_c = 0.0  # no edges to c left
    x1_d = 0.0041  # f(x0_c * 0.2) * 0.3
    x1_e = 0.3577  # f(x0_e * 0.6) * 0.8

    cells = Cells(molecules=MOLECULES, actions=[], n_max_proteins=4, trunc_n_decs=5)

    A, B, Z = cells.get_cell_params(proteomes=[prtm])

    assert A.shape == (1, dim1, 4)
    assert B.shape == (1, dim1, 4)
    assert Z.shape == (1, 4)

    X1 = cells.simulate_protein_work(X=X0, A=A, B=B, Z=Z)

    assert X1.shape == (1, dim1)
    assert X1[0, 0] == pytest.approx(x1_a, abs=TOLERANCE)
    assert X1[0, 1] == pytest.approx(x1_b, abs=TOLERANCE)
    assert X1[0, 2] == pytest.approx(x1_c, abs=TOLERANCE)
    assert X1[0, 3] == pytest.approx(x1_d, abs=TOLERANCE)
    assert X1[0, 4] == pytest.approx(x1_e, abs=TOLERANCE)
    for i in range(5, dim1):
        assert X1[0, i] == 0.0


def test_performance():
    genetics = Genetics(domain_map=DOMAINS)
    cells = Cells(molecules=MOLECULES, actions=ACTIONS)
    gs = [rand_genome((1000, 5000)) for _ in range(100)]
    prtms = [genetics.get_proteome(seq=d) for d in gs]

    t0 = time.time()
    A, B, Z = cells.get_cell_params(proteomes=prtms)
    td = time.time() - t0
    assert td < 0.1, "Used to take 0.043"

    C = torch.randn(len(prtms), len(ACTIONS) + 2 * len(MOLECULES))

    t0 = time.time()
    _ = cells.simulate_protein_work(X=C, A=A, B=B, Z=Z)
    td = time.time() - t0
    assert td < 0.01, "Used to take 0.003"

