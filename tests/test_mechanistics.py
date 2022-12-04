import time
import pytest
import torch
from magicsoup.util import rand_genome
from magicsoup.genetics import (
    Molecule,
    ReceptorDomainFact,
    SynthesisDomainFact,
    Protein,
    Genetics,
)
from magicsoup.examples.default import DOMAINS, MOLECULES, ACTIONS
from magicsoup.world import World

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

    world = World(molecules=molecules, actions=[], n_max_proteins=4, trunc_n_decs=5)

    A, B, Z = world.get_cell_params(proteomes=[prtm])

    assert A.shape == (1, dim1, 4)
    assert B.shape == (1, dim1, 4)
    assert Z.shape == (1, 4)

    X1 = world.integrate_signals(X=X0, A=A, B=B, Z=Z)

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

    world = World(molecules=molecules, actions=[], n_max_proteins=4, trunc_n_decs=5)

    A, B, Z = world.get_cell_params(proteomes=[prtm])
    X1 = world.integrate_signals(X=X0, A=A, B=B, Z=Z)

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

    world = World(molecules=molecules, actions=[], n_max_proteins=4, trunc_n_decs=5)

    A, B, Z = world.get_cell_params(proteomes=[prtm])
    X1 = world.integrate_signals(X=X0, A=A, B=B, Z=Z)

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

    world = World(molecules=molecules, actions=[], n_max_proteins=4, trunc_n_decs=5)

    A, B, Z = world.get_cell_params(proteomes=[prtm])
    X1 = world.integrate_signals(X=X0, A=A, B=B, Z=Z)

    assert X1.shape == (1, dim1)
    assert X1[0, 0] == pytest.approx(x1_a, abs=TOLERANCE)
    assert X1[0, 1] == pytest.approx(x1_b, abs=TOLERANCE)
    assert X1[0, 2] == pytest.approx(x1_c, abs=TOLERANCE)
    for i in range(3, dim1):
        assert X1[0, i] == 0.0


def test_molecule_deconstruction_with_small_concentration_multiple_proteins_and_cells():
    # molecule B gets deconstructed while C is being synthesized
    ma = Molecule("MA", 0)
    mb = Molecule("MB", 1)
    mc = Molecule("MC", 1)
    md = Molecule("MD", 0)
    molecules = [ma, mb, mc, md]

    # fmt: off
    proteins_c1 = [
        [
            ReceptorDomainFact(ma)(0.9),
            SynthesisDomainFact(mb)(-0.5),
            SynthesisDomainFact(mc)(0.5),
        ],
        [
            ReceptorDomainFact(ma)(0.2),
            SynthesisDomainFact(mb)(-0.9),
            SynthesisDomainFact(mc)(0.3),
        ],
        [
            ReceptorDomainFact(ma)(0.5),
            SynthesisDomainFact(md)(0.2)
        ],
    ]
    proteins_c2 = [
        [
            ReceptorDomainFact(ma)(0.9),
            SynthesisDomainFact(mb)(0.5),
            SynthesisDomainFact(mc)(-0.5),
        ],
        [
            ReceptorDomainFact(ma)(0.2),
            SynthesisDomainFact(mb)(0.3),
            SynthesisDomainFact(mc)(-0.9),
        ],
        [
            ReceptorDomainFact(ma)(0.6),
            SynthesisDomainFact(md)(-0.3)
        ],
    ]
    # fmt: on
    prtm_c1 = [Protein(domains=d) for d in proteins_c1]
    prtm_c2 = [Protein(domains=d) for d in proteins_c2]

    dim1 = len(molecules) * 2
    X0 = torch.zeros(2, dim1)
    X0[0, 0] = 2.5  # x0_0_a (plenty activation)
    X0[0, 1] = 0.2  # x0_0_b (only little left)
    X0[0, 2] = 0.1  # x0_0_c
    X0[0, 3] = 0.1  # x0_0_d
    X0[1, 0] = 2.0  # x0_1_a (plenty activation)
    X0[1, 1] = 0.2  # x0_1_b
    X0[1, 2] = 0.1  # x0_1_c (only little left)
    X0[1, 3] = 2.0  # x0_1_d

    # by hand with:
    #
    # def f(x: float) -> float:
    #    return (1 - math.exp(-(x ** 3)))
    #
    # x1_0_b would be -0.6057 = f(x0_0_a * 0.9) * -0.5 + f(x0_0_a * 0.2) * -0.9
    # x1_0_c would be 0.5352 = f(x0_0_a * 0.9) * 0.5 + f(x0_0_a * 0.2) * 0.3
    # but only 0.2 mb left in cell 0: 0.3301 = -0.2 / -0.6057 to correct
    #
    # x1_1_b would be 0.5171 = f(x0_1_a * 0.9) * 0.5 + f(x0_1_a * 0.2) * 0.3
    # x1_1_c would be -0.5543 = f(x0_1_a * 0.9) * -0.5 + f(x0_1_a * 0.2) * -0.9
    # but only 0.1 mc left in cell 0: 0.1804 = -0.1 / -0.5543 to correct
    #
    # md production by the last protein is untouched from all that
    x1_0_a = 0.0  # no edge points to a
    x1_0_b = -0.2000  # -0.6057 * 0.3301
    x1_0_c = 0.1767  # 0.5352 * 0.3301
    x1_0_d = 0.1716  # f(x0_0_d * 0.5) * 0.2
    x1_1_a = 0.0  # no edge points to a
    x1_1_b = 0.0932  # 0.5171 * 0.1804
    x1_1_c = -0.1000  # -0.5543 * 0.1804
    x1_1_d = -0.2467  # f(x0_1_d * 0.6) * -0.3

    world = World(molecules=molecules, actions=[], n_max_proteins=3, trunc_n_decs=5)

    A, B, Z = world.get_cell_params(proteomes=[prtm_c1, prtm_c2])
    X1 = world.integrate_signals(X=X0, A=A, B=B, Z=Z)

    assert X1.shape == (2, dim1)
    assert X1[0, 0] == pytest.approx(x1_0_a, abs=TOLERANCE)
    assert X1[0, 1] == pytest.approx(x1_0_b, abs=TOLERANCE)
    assert X1[0, 2] == pytest.approx(x1_0_c, abs=TOLERANCE)
    assert X1[0, 3] == pytest.approx(x1_0_d, abs=TOLERANCE)
    assert X1[1, 0] == pytest.approx(x1_1_a, abs=TOLERANCE)
    assert X1[1, 1] == pytest.approx(x1_1_b, abs=TOLERANCE)
    assert X1[1, 2] == pytest.approx(x1_1_c, abs=TOLERANCE)
    assert X1[1, 3] == pytest.approx(x1_1_d, abs=TOLERANCE)
    for i in range(4, dim1):
        assert X1[0, i] == 0.0
        assert X1[1, i] == 0.0


def test_proteins_cannot_produce_negative_concentrations():
    genetics = Genetics(domain_map=DOMAINS)
    world = World(molecules=MOLECULES, actions=ACTIONS)
    gs = [rand_genome((1000, 5000)) for _ in range(100)]
    prtms = [genetics.get_proteome(seq=d) for d in gs]

    A, B, Z = world.get_cell_params(proteomes=prtms)
    X = torch.randn(len(prtms), len(ACTIONS) + 2 * len(MOLECULES)).abs()

    for _ in range(10):
        Xd = world.integrate_signals(X=X, A=A, B=B, Z=Z)
        X = X + Xd
        assert not torch.any(X < 0)


def test_performance():
    genetics = Genetics(domain_map=DOMAINS)
    world = World(molecules=MOLECULES, actions=ACTIONS)
    gs = [rand_genome((1000, 5000)) for _ in range(100)]
    prtms = [genetics.get_proteome(seq=d) for d in gs]

    t0 = time.time()
    A, B, Z = world.get_cell_params(proteomes=prtms)
    td = time.time() - t0
    assert td < 0.1, "Used to take 0.043"

    C = torch.randn(len(prtms), len(ACTIONS) + 2 * len(MOLECULES))

    t0 = time.time()
    _ = world.integrate_signals(X=C, A=A, B=B, Z=Z)
    td = time.time() - t0
    assert td < 0.01, "Used to take 0.003"
