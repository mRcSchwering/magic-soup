import torch
from magicsoup.containers import Molecule
from magicsoup.kinetics import Kinetics

TOLERANCE = 1e-4
EPS = 1e-8


ma = Molecule("a", energy=15 * 1e3)
mb = Molecule("b", energy=10 * 1e3)
mc = Molecule("c", energy=10 * 1e3)
md = Molecule("d", energy=5 * 1e3)
MOLECULES = [ma, mb, mc, md]

r_a_b = ([ma], [mb])
r_b_c = ([mb], [mc])
r_bc_d = ([mb, mc], [md])
r_d_bb = ([md], [mb, mb])
REACTIONS = [r_a_b, r_b_c, r_bc_d, r_d_bb]

# fmt: off
KM_WEIGHTS = torch.tensor([
    torch.nan, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  # idxs 0-9
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,  # idxs 10-19
    2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9  # idxs 20-29
])

VMAX_WEIGHTS = torch.tensor([
    torch.nan, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,  # idxs 0-9
    2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,  # idxs 10-19
])

SIGNS = torch.tensor([0.0, 1.0, -1.0])  # idxs 0-2

TRANSPORT_M = torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # idx 0: none
    [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], # idx 1: a in->out
    [0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # idx 2: b in->out
    [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0], # idx 3: c in->out
    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0], # idx 4: d in->out
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

EFFECTOR_M = torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # idx 0: none
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # idx 1: a in
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # idx 2: b in
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], # idx 3: c in
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], # idx 4: d in
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], # idx 5: a out
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], # idx 6: b out
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], # idx 7: c out
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # idx 8: d out
])

REACTION_M = torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # idx 0: none
    [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # idx 1: a -> b
    [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # idx 2: b -> c
    [0.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0], # idx 3: b,c -> d
    [0.0, 2.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],  # idx 4: d -> 2b
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

# fmt: on

# TODO: refactor
#       I think _get_kinetics should include random kinetics


def _get_kinetics(n_computations=1) -> Kinetics:
    kinetics = Kinetics(
        molecules=MOLECULES,
        reactions=REACTIONS,
        n_computations=n_computations,
        abs_temp=310,
    )
    # kinetics.km_map.weights = KM_WEIGHTS.clone()
    # kinetics.vmax_map.weights = VMAX_WEIGHTS.clone()
    # kinetics.sign_map.signs = SIGNS.clone()
    # kinetics.transport_map.M = TRANSPORT_M.clone()
    # kinetics.effector_map.M = EFFECTOR_M.clone()
    # kinetics.reaction_map.M = REACTION_M.clone()
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
    kinetics.N = torch.randint(low=-5, high=6, size=(n_cells, n_prots, n_mols)).float()
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
    kinetics.N = torch.randint(low=-5, high=6, size=(n_cells, n_prots, n_mols)).float()
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
