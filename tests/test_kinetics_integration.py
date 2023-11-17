import torch
import random
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS


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


def test_cell_params_are_always_set_reproduceably():
    n_cells = 100
    n_tries = 10

    for i in range(n_tries):
        proteomes = []
        for _ in range(n_cells):
            proteins = []
            for _ in range(random.choice(list(range(20)))):
                domains = []
                for _ in range(random.choice([1, 1, 2])):
                    domtype = random.choice([1, 2, 3])
                    i0 = random.choice(list(range(61)))
                    i1 = random.choice(list(range(61)))
                    i2 = random.choice(list(range(61)))
                    i3 = random.choice(list(range(3904)))
                    domains.append(((domtype, i0, i1, i2, i3), 1, 2))
                proteins.append(domains)
            proteomes.append(proteins)

        n_max_prots = max(len(d) for d in proteomes)
        idxs = list(range(n_cells))

        kinetics = _get_kinetics()
        kinetics.increase_max_cells(by_n=n_cells)
        kinetics.increase_max_proteins(max_n=n_max_prots)
        kinetics.set_cell_params(cell_idxs=idxs, proteomes=proteomes)

        N_orig = kinetics.N.clone()
        Nb_orig = kinetics.Nb.clone()
        Nf_orig = kinetics.Nf.clone()
        A_orig = kinetics.A.clone()
        Kmf_orig = kinetics.Kmf.clone()
        Kmb_orig = kinetics.Kmb.clone()
        Kmr_orig = kinetics.Kmr.clone()
        Vmax_orig = kinetics.Vmax.clone()

        kinetics.remove_cell_params(keep=torch.full((n_cells,), False))
        kinetics.increase_max_cells(by_n=n_cells)
        kinetics.increase_max_proteins(max_n=n_max_prots)
        kinetics.set_cell_params(cell_idxs=idxs, proteomes=proteomes)

        assert torch.equal(kinetics.N, N_orig), i
        assert torch.equal(kinetics.Nb, Nb_orig), i
        assert torch.equal(kinetics.Nf, Nf_orig), i
        assert torch.equal(kinetics.A, A_orig), i
        assert torch.equal(kinetics.Kmf, Kmf_orig), i
        assert torch.equal(kinetics.Kmb, Kmb_orig), i
        assert torch.equal(kinetics.Kmr, Kmr_orig), i
        assert torch.equal(kinetics.Vmax, Vmax_orig), i


def test_equilibrium_is_reached_with_zeros():
    # P0: A + B <-> C | +5 kJ
    # P1: 3A <-> C | -10 kJ
    # t0: A = 3.0, others 0.0
    # expected: A drops, B rises, C first rises then drops
    # equilibrium has a lot of B, very little A and C, A > C
    # has to create stuff from zeros (except for A)
    # there used to be a bug that didnt allow that

    # fmt: off

    # concentrations (c, s)
    X = torch.tensor([
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    # reactions (c, p, s)
    N = torch.tensor([[
        [-1., -1.,  1.,  0.,  0.,  0.],
        [-3.,  0.,  1.,  0.,  0.,  0.]]])
    
    Nf = torch.where(N < 0.0, -N, 0.0)
    Nb = torch.where(N > 0.0, N, 0.0)

    # affinities (c, p)
    Kmf = torch.tensor([[7.3328, 1.0539]])
    Kmb = torch.tensor([[1.0539, 5.1021]])
    Kmr = torch.zeros(1, 2)

    # max velocities (c, p)
    Vmax = torch.tensor([[0.3, 0.3]])

    # allosterics (c, p, s)
    A = torch.zeros(1, 2, 6)

    # fmt: on

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

    for _ in range(100):
        X = kinetics.integrate_signals(X=X)

    assert 0.5 > X[0][0] > 0.0
    assert 3.0 > X[0][1] > 1.0
    assert 0.5 > X[0][2] > 0.0
    assert X[0][0] > X[0][2]


def test_equilibrium_is_reached_with_high_vmax():
    # higher order reactions have trouble reaching equilibrium
    # because they tend to overshoot
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

    # fmt: on

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
        assert torch.all(X < 10_000), X[X >= 10_000]
