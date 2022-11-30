import time
import pytest
import torch
from util import rand_genome
from genetics import (
    Genetics,
    Molecule,
    ReceptorDomainFact,
    SynthesisDomainFact,
    Protein,
    MOLECULES,
    ACTIONS,
    DOMAINS,
)
from world import World
from cells import Cells

TOLERANCE = 1e-4


def test_cell_signal_integration():
    # molecules have all 0 energy, so all proteins can work
    ma = Molecule("MA", 0)
    mb = Molecule("MB", 0)
    mc = Molecule("MC", 0)
    md = Molecule("MD", 0)
    me = Molecule("ME", 0)
    molecules = [ma, mb, mc, md, me]

    # fmt: off
    proteins = [
        [
            ReceptorDomainFact(ma)(0.1),
            SynthesisDomainFact(mb)(0.8),
            SynthesisDomainFact(mc)(0.5),
            SynthesisDomainFact(me)(0.3),
        ],
        [
            ReceptorDomainFact(md)(0.4),
            SynthesisDomainFact(mc)(0.9),
            SynthesisDomainFact(md)(0.4),
        ],
        [
            ReceptorDomainFact(mc)(0.2),
            SynthesisDomainFact(md)(0.3)
        ],
        [
            ReceptorDomainFact(me)(0.6),
            SynthesisDomainFact(mb)(0.7),
            SynthesisDomainFact(me)(0.8),
        ],
    ]
    # fmt: on

    prtm = [Protein(domains=d) for d in proteins]
    dim1 = len(molecules) * 2

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

    cells = Cells(molecules=molecules, actions=[], n_max_proteins=4, trunc_n_decs=5)

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
    # molecule E needs energy, so proteins 0 and 3 dont work
    ma = Molecule("MA", 0)
    mb = Molecule("MB", 0)
    mc = Molecule("MC", 0)
    md = Molecule("MD", 0)
    me = Molecule("ME", 1)
    molecules = [ma, mb, mc, md, me]

    # fmt: off
    proteins = [
        [
            ReceptorDomainFact(ma)(0.1),
            SynthesisDomainFact(mb)(0.8),
            SynthesisDomainFact(mc)(0.5),
            SynthesisDomainFact(me)(0.3),
        ],
        [
            ReceptorDomainFact(md)(0.4),
            SynthesisDomainFact(mc)(0.9),
            SynthesisDomainFact(md)(0.4),
        ],
        [
            ReceptorDomainFact(mc)(0.2),
            SynthesisDomainFact(md)(0.3)
        ],
        [
            ReceptorDomainFact(me)(0.6),
            SynthesisDomainFact(mb)(0.7),
            SynthesisDomainFact(me)(0.8),
        ],
    ]
    # fmt: on

    prtm = [Protein(domains=d) for d in proteins]
    dim1 = len(molecules) * 2

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
    x1_b = 0.0  # no edges to b left
    x1_c = 0.1180  # f(x0_d * 0.4) * 0.9
    x1_d = 0.0565  # f(x0_d * 0.4) * 0.4 + f(x0_c * 0.2) * 0.3
    x1_e = 0.0  # no edges to e left

    cells = Cells(molecules=molecules, actions=[], n_max_proteins=4, trunc_n_decs=5)

    A, B, Z = cells.get_cell_params(proteomes=[prtm])
    X1 = cells.simulate_protein_work(X=X0, A=A, B=B, Z=Z)

    assert X1.shape == (1, dim1)
    assert X1[0, 0] == pytest.approx(x1_a, abs=TOLERANCE)
    assert X1[0, 1] == pytest.approx(x1_b, abs=TOLERANCE)
    assert X1[0, 2] == pytest.approx(x1_c, abs=TOLERANCE)
    assert X1[0, 3] == pytest.approx(x1_d, abs=TOLERANCE)
    assert X1[0, 4] == pytest.approx(x1_e, abs=TOLERANCE)
    for i in range(5, dim1):
        assert X1[0, i] == 0.0


def test_molecule_deconstruction_with_abundant_concentration():
    # molecule B gets deconstructed while C is being synthesized
    ma = Molecule("MA", 0)
    mb = Molecule("MB", 1)
    mc = Molecule("MC", 1)
    molecules = [ma, mb, mc]

    # fmt: off
    proteins = [
        [
            ReceptorDomainFact(ma)(0.9),
            SynthesisDomainFact(mb)(-0.5),
            SynthesisDomainFact(mc)(0.5),
        ]
    ]
    # fmt: on
    prtm = [Protein(domains=d) for d in proteins]
    dim1 = len(molecules) * 2

    # initial concentrations
    X0 = torch.zeros(1, dim1)
    X0[0, 0] = 2.5  # x0_a
    X0[0, 1] = 2.1  # x0_b
    X0[0, 2] = 1.2  # x0_c

    # by hand with:
    # def f(x: float) -> float:
    #    return (1 - math.exp(-(x ** 3)))
    x1_a = 0.0  # no edge points to a
    x1_b = -0.4999  # f(x0_a * 0.9) * -0.5
    x1_c = 0.4999  # f(x0_a * 0.9) * 0.5

    cells = Cells(molecules=molecules, actions=[], n_max_proteins=4, trunc_n_decs=5)

    A, B, Z = cells.get_cell_params(proteomes=[prtm])
    X1 = cells.simulate_protein_work(X=X0, A=A, B=B, Z=Z)

    assert X1.shape == (1, dim1)
    assert X1[0, 0] == pytest.approx(x1_a, abs=TOLERANCE)
    assert X1[0, 1] == pytest.approx(x1_b, abs=TOLERANCE)
    assert X1[0, 2] == pytest.approx(x1_c, abs=TOLERANCE)
    for i in range(3, dim1):
        assert X1[0, i] == 0.0


def test_molecule_deconstruction_with_small_concentration():
    # molecule B gets deconstructed while C is being synthesized
    ma = Molecule("MA", 0)
    mb = Molecule("MB", 1)
    mc = Molecule("MC", 1)
    molecules = [ma, mb, mc]

    # fmt: off
    proteins = [
        [
            ReceptorDomainFact(ma)(0.9),
            SynthesisDomainFact(mb)(-0.5),
            SynthesisDomainFact(mc)(0.5),
        ]
    ]
    # fmt: on
    prtm = [Protein(domains=d) for d in proteins]
    dim1 = len(molecules) * 2

    # initial concentrations
    X0 = torch.zeros(1, dim1)
    X0[0, 0] = 2.5  # x0_a
    X0[0, 1] = 0.2  # x0_b (only little left)
    X0[0, 2] = 1.2  # x0_c

    # by hand with:
    # def f(x: float) -> float:
    #    return (1 - math.exp(-(x ** 3)))
    # protein activity would be f(x0_a * 0.9) * 0.5 = 0.4999
    x1_a = 0.0  # no edge points to a
    x1_b = -0.2000  # but it can only get 0.2 X0_b
    x1_c = 0.2000  # and thus only creates 0.2 X0_b

    cells = Cells(molecules=molecules, actions=[], n_max_proteins=4, trunc_n_decs=5)

    A, B, Z = cells.get_cell_params(proteomes=[prtm])
    X1 = cells.simulate_protein_work(X=X0, A=A, B=B, Z=Z)

    assert X1.shape == (1, dim1)
    assert X1[0, 0] == pytest.approx(x1_a, abs=TOLERANCE)
    assert X1[0, 1] == pytest.approx(x1_b, abs=TOLERANCE)
    assert X1[0, 2] == pytest.approx(x1_c, abs=TOLERANCE)
    for i in range(3, dim1):
        assert X1[0, i] == 0.0


def test_proteins_cannot_produce_negative_concentrations():
    genetics = Genetics(domain_map=DOMAINS)
    cells = Cells(molecules=MOLECULES, actions=ACTIONS)
    gs = [rand_genome((1000, 5000)) for _ in range(100)]
    prtms = [genetics.get_proteome(seq=d) for d in gs]
    world = World(n_molecules=len(MOLECULES))
    pos = world.add_cells(n_cells=len(prtms))

    cells.add_cells(genomes=gs, proteomes=prtms, positions=pos)
    A, B, Z = cells.get_cell_params(proteomes=prtms)
    X = torch.randn(len(prtms), len(ACTIONS) + 2 * len(MOLECULES))

    for _ in range(10):
        Xd = cells.simulate_protein_work(X=X, A=A, B=B, Z=Z)
        X = X + Xd
        assert not torch.any(X < 0)


def test_performance():
    genetics = Genetics(domain_map=DOMAINS)
    cells = Cells(molecules=MOLECULES, actions=ACTIONS)
    gs = [rand_genome((1000, 5000)) for _ in range(100)]
    prtms = [genetics.get_proteome(seq=d) for d in gs]
    world = World(n_molecules=len(MOLECULES))
    pos = world.add_cells(n_cells=len(prtms))

    cells.add_cells(genomes=gs, proteomes=prtms, positions=pos)

    t0 = time.time()
    A, B, Z = cells.get_cell_params(proteomes=prtms)
    td = time.time() - t0
    assert td < 0.1, "Used to take 0.043"

    C = torch.randn(len(prtms), len(ACTIONS) + 2 * len(MOLECULES))

    t0 = time.time()
    _ = cells.simulate_protein_work(X=C, A=A, B=B, Z=Z)
    td = time.time() - t0
    assert td < 0.01, "Used to take 0.003"

