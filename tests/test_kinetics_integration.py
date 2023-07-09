import pytest
import torch
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS

# mark all tests in this module as slow
pytestmark = pytest.mark.slow


def _get_kinetics(n_computations=1) -> ms.Kinetics:
    chemistry = ms.Chemistry(molecules=MOLECULES, reactions=REACTIONS)
    kinetics = ms.Kinetics(
        molecules=chemistry.molecules,
        reactions=chemistry.reactions,
        n_computations=n_computations,
        abs_temp=310,
        scalar_enc_size=61,
        vector_enc_size=3904,
    )
    return kinetics


def test_equilibrium_is_reached():
    # molecules should reach equilibrium after a few steps
    # with Vmax > 1.0 higher order reactions cant reach equilibrium
    # because they overshoot

    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b (Ke=1.0), P1: c -> d (Ke=20.0)
    # cell 1: P0: a,2b -> c (Ke=10.0)

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [2.0, 20.0, 5.0, 5.0],
        [2.0, 3.1, 1.3, 2.9],
    ])

    # reactions (c, p, s)
    N = torch.tensor([
        [   [-1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
        [   [-1.0, -2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])
    Nf = torch.where(N < 0.0, -N, 0.0)
    Nb = torch.where(N > 0.0, N, 0.0)

    # affinities (c, p)
    Kmf = torch.tensor([
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    Kmb = torch.tensor([
        [1.0, 20.0, 0.0],
        [10.0, 0.0, 0.0],
    ])
    Kmr = torch.zeros(2, 3)

    # max velocities (c, p)
    Vmax = torch.tensor([
        [100.0, 100.0, 0.0],
        [100.0, 0.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4)

    # test
    kinetics = _get_kinetics(n_computations=11)
    kinetics.N = N
    kinetics.Nf = Nf
    kinetics.Nb = Nb
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A

    for _ in range(10):
        X0 = kinetics.integrate_signals(X=X0)

    q_c0_0 = X0[0, 1] / X0[0, 0]
    q_c0_1 = X0[0, 3] / X0[0, 2]
    q_c1_0 = X0[1, 2] / (X0[1, 0] * X0[1, 1] * X0[1, 1])

    assert (q_c0_0 - 1.0).abs() < 0.1
    assert (q_c0_1 - 20.0).abs() < 2.0
    assert (q_c1_0 - 10.0).abs() < 1.0


def test_random_kinetics_stay_zero():
    # Typical reasons for cells creating signals from 0:
    # 1. when using exp(ln(x)) 1s can accidentally be created
    # 2. when doing a^b 1s can be created (0^1=0 but 0^0=1)
    n_cells = 100
    n_prots = 100
    n_steps = 1000

    kinetics = _get_kinetics()
    n_mols = len(MOLECULES) * 2

    # concentrations (c, s)
    X = torch.zeros(n_cells, n_mols).abs()

    # reactions (c, p, s)
    kinetics.N = torch.randint(low=-8, high=9, size=(n_cells, n_prots, n_mols)).float()
    kinetics.Nf = torch.where(kinetics.N < 0.0, -kinetics.N, 0.0)
    kinetics.Nb = torch.where(kinetics.N > 0.0, kinetics.N, 0.0)

    # max velocities (c, p)
    kinetics.Vmax = torch.randn(n_cells, n_prots).abs() * 100

    # allosterics (c, p, s)
    kinetics.A = torch.randint(low=-2, high=3, size=(n_cells, n_prots, n_mols))

    # affinities (c, p)
    Ke = torch.randn(n_cells, n_prots)
    kinetics.Kmf = torch.randn(n_cells, n_prots).abs()
    kinetics.Kmb = kinetics.Kmf * Ke
    kinetics.Kmr = torch.randn(n_cells, n_prots).abs()

    # test
    for _ in range(n_steps):
        X = kinetics.integrate_signals(X=X)
        assert X.min() == 0.0
        assert X.max() == 0.0


def test_random_kinetics_dont_explode():
    # interaction terms can overflow float32 when they become too big
    # i.g. exponents too high and many substrates
    # this will create infinites which will eventually become NaN
    n_cells = 100
    n_prots = 100
    n_steps = 1000

    kinetics = _get_kinetics()
    n_mols = len(MOLECULES) * 2

    # concentrations (c, s)
    X = torch.randn(n_cells, n_mols).abs().clamp(max=1.0) * 100

    # reactions (c, p, s)
    kinetics.N = torch.randint(low=-8, high=9, size=(n_cells, n_prots, n_mols)).float()
    kinetics.Nf = torch.where(kinetics.N < 0.0, -kinetics.N, 0.0)
    kinetics.Nb = torch.where(kinetics.N > 0.0, kinetics.N, 0.0)

    # max velocities (c, p)
    kinetics.Vmax = torch.randn(n_cells, n_prots).abs().clamp(max=1.0) * 100

    # allosterics (c, p, s)
    kinetics.A = torch.randint(low=-2, high=3, size=(n_cells, n_prots, n_mols))

    # affinities (c, p)
    Ke = torch.randn(n_cells, n_prots)
    kinetics.Kmf = torch.randn(n_cells, n_prots).abs().clamp(0.001)
    kinetics.Kmb = (kinetics.Kmf * Ke).clamp(0.001)
    kinetics.Kmr = torch.randn(n_cells, n_prots).abs().clamp(0.001)

    # test
    for _ in range(n_steps):
        X = kinetics.integrate_signals(X=X)
        assert not torch.any(X < 0.0), X[X < 0.0].min()
        assert not torch.any(X.isnan()), X.isnan().sum()
        assert torch.all(X.isfinite()), ~X.isfinite().sum()
