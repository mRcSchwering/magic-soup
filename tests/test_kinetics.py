import pytest
import torch
from magicsoup.containers import Protein, Domain, Molecule
from magicsoup.constants import EPS
from magicsoup.kinetics import integrate_signals, calc_cell_params

TOLERANCE = 1e-4

# fmt: off

ma = Molecule("a", energy=15)
mb = Molecule("b", energy=10)
mc = Molecule("c", energy=10)
md = Molecule("d", energy=5)

mol_2_idx = {
    (ma, False): 0, (mb, False): 1, (mc, False): 2, (md, False): 3,
    (ma, True): 4, (mb, True): 5, (mc, True): 6, (md, True): 7,
}

r_a_b = [[ma], [mb]]
r_b_c = [[mb], [mc]]
r_bc_d = [[mb, mc], [md]]

# fmt: on


def avg(*x):
    return sum(x) / len(x)


def test_simple_cell_params():
    # fmt: off
    p0 = Protein(domains=[
        Domain(*r_a_b, affinity=0.5, velocity=1.0, orientation=True, is_catalytic=True),
        Domain(*r_bc_d, affinity=1.5, velocity=1.2, orientation=False, is_catalytic=True),
    ])
    p1 = Protein(domains=[
        Domain(*r_b_c, affinity=0.9, velocity=2.0, orientation=True, is_catalytic=True),
        Domain(*r_bc_d, affinity=1.2, velocity=1.3, orientation=False, is_catalytic=True),
    ])
    c0 = [p0, p1]

    p0 = Protein(domains=[
        Domain(*r_a_b, affinity=0.3, velocity=1.1, orientation=False, is_catalytic=True),
        Domain(*r_bc_d, affinity=1.4, velocity=2.1, orientation=False, is_catalytic=True),
    ])
    p1 = Protein(domains=[
        Domain(*r_b_c, affinity=0.3, velocity=1.9, orientation=True, is_catalytic=True),
        Domain(*r_bc_d, affinity=1.7, velocity=2.3, orientation=True, is_catalytic=True),
    ])
    c1 = [p0, p1]

    # fmt: on

    Km = torch.zeros(2, 3, 8)
    Vmax = torch.zeros(2, 3)
    E = torch.zeros(2, 3)
    N = torch.zeros(2, 3, 8)
    A = torch.zeros(2, 3, 8)

    calc_cell_params(
        proteomes=[c0, c1],
        n_signals=4,
        mol_2_idx=mol_2_idx,
        cell_idxs=[0, 1],
        Km=Km,
        Vmax=Vmax,
        E=E,
        N=N,
        A=A,
    )

    assert Km[0, 0, 0] == pytest.approx(0.5, abs=TOLERANCE)
    assert Km[0, 0, 1] == pytest.approx(avg(1 / 0.5, 1 / 1.5), abs=TOLERANCE)
    assert Km[0, 0, 2] == pytest.approx(1 / 1.5, abs=TOLERANCE)
    assert Km[0, 0, 3] == pytest.approx(1.5, abs=TOLERANCE)
    assert Km[0, 0, 4] == 0.0
    assert Km[0, 0, 5] == 0.0
    assert Km[0, 0, 6] == 0.0
    assert Km[0, 0, 7] == 0.0
    assert Km[0, 1, 0] == 0.0
    assert Km[0, 1, 1] == pytest.approx(avg(0.9, 1 / 1.2), abs=TOLERANCE)
    assert Km[0, 1, 2] == pytest.approx(avg(1 / 0.9, 1 / 1.2), abs=TOLERANCE)
    assert Km[0, 1, 3] == pytest.approx(1.2, abs=TOLERANCE)
    assert Km[0, 1, 4] == 0.0
    assert Km[0, 1, 5] == 0.0
    assert Km[0, 1, 6] == 0.0
    assert Km[0, 1, 7] == 0.0

    assert Km[1, 0, 0] == pytest.approx(1 / 0.3, abs=TOLERANCE)
    assert Km[1, 0, 1] == pytest.approx(avg(0.3, 1 / 1.4), abs=TOLERANCE)
    assert Km[1, 0, 2] == pytest.approx(1 / 1.4, abs=TOLERANCE)
    assert Km[1, 0, 3] == pytest.approx(1.4, abs=TOLERANCE)
    assert Km[1, 0, 4] == 0.0
    assert Km[1, 0, 5] == 0.0
    assert Km[1, 0, 6] == 0.0
    assert Km[1, 0, 7] == 0.0
    assert Km[1, 1, 0] == 0.0
    assert Km[1, 1, 1] == pytest.approx(avg(0.3, 1.7), abs=TOLERANCE)
    assert Km[1, 1, 2] == pytest.approx(avg(1 / 0.3, 1.7), abs=TOLERANCE)
    assert Km[1, 1, 3] == pytest.approx(1 / 1.7, abs=TOLERANCE)
    assert Km[1, 1, 4] == 0.0
    assert Km[1, 1, 5] == 0.0
    assert Km[1, 1, 6] == 0.0
    assert Km[1, 1, 7] == 0.0

    assert Vmax[0, 0] == pytest.approx(avg(1.0, 1.2), abs=TOLERANCE)
    assert Vmax[0, 1] == pytest.approx(avg(2.0, 1.3), abs=TOLERANCE)
    assert Vmax[0, 2] == 0.0

    assert Vmax[1, 0] == pytest.approx(avg(1.1, 2.1), abs=TOLERANCE)
    assert Vmax[1, 1] == pytest.approx(avg(1.9, 2.3), abs=TOLERANCE)
    assert Vmax[1, 2] == 0.0

    assert E[0, 0] == 10 - 15 + 10 + 10 - 5
    assert E[0, 1] == pytest.approx(avg(2.0, 1.3), abs=TOLERANCE)
    assert E[0, 2] == 0.0

    assert E[1, 0] == pytest.approx(avg(1.1, 2.1), abs=TOLERANCE)
    assert E[1, 1] == pytest.approx(avg(1.9, 2.3), abs=TOLERANCE)
    assert E[1, 2] == 0.0


def test_simple_mm_kinetic():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b, P1: b -> d
    # cell 1: P0: c -> d, P1: a -> d

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [1.1, 0.1, 2.9, 0.8],
        [2.9, 3.1, 2.1, 1.0],
    ])

    # reactions (c, p, s)
    N = torch.tensor([
        [   [-1.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
        [   [0.0, 0.0, -1.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])

    # affinities (c, p, s)
    Km = torch.tensor([
        [   [1.2, 1.5, 0.0, 0.0],
            [0.0, 0.5, 0.0, 3.5],
            [0.0, 0.0, 0.0, 0.0] ],
        [   [0.0, 0.0, 0.5, 1.5],
            [1.2, 0.0, 0.0, 1.9],
            [0.0, 0.0, 0.0, 0.0] ],
    ])

    # max velocities (c, p)
    Vmax = torch.tensor([
        [2.1, 1.0, 0.0],
        [1.1, 2.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4)

    # equilibrium constants (c, p)
    Ke = torch.full((2, 3), 999.9)

    # fmt: on

    def mm(x, k, v):
        return v * x / (k + x)

    # expected outcome
    dx_c0_b = mm(x=X0[0, 0], v=Vmax[0, 0], k=Km[0, 0, 0])
    dx_c0_a = -dx_c0_b
    dx_c0_d = mm(x=X0[0, 1], v=Vmax[0, 1], k=Km[0, 1, 1])
    dx_c0_b = dx_c0_b - dx_c0_d

    dx_c1_d_1 = mm(x=X0[1, 2], v=Vmax[1, 0], k=Km[1, 0, 2])
    dx_c1_c = -dx_c1_d_1
    dx_c1_d_2 = mm(x=X0[1, 0], v=Vmax[1, 1], k=Km[1, 1, 0])
    dx_c1_a = -dx_c1_d_2
    dx_c1_d = dx_c1_d_1 + dx_c1_d_2

    # test
    Xd = integrate_signals(X=X0, Km=Km, Vmax=Vmax, Ke=Ke, N=N, A=A)

    assert Xd[0, 0] == pytest.approx(dx_c0_a, abs=TOLERANCE)
    assert Xd[0, 1] == pytest.approx(dx_c0_b, abs=TOLERANCE)
    assert Xd[0, 2] == 0.0
    assert Xd[0, 3] == pytest.approx(dx_c0_d, abs=TOLERANCE)

    assert Xd[1, 0] == pytest.approx(dx_c1_a, abs=TOLERANCE)
    assert Xd[1, 1] == 0.0
    assert Xd[1, 2] == pytest.approx(dx_c1_c, abs=TOLERANCE)
    assert Xd[1, 3] == pytest.approx(dx_c1_d, abs=TOLERANCE)


def test_mm_kinetic_with_proportions():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> 2b, P1: 2c -> d
    # cell 1: P0: 3b -> 2c

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [1.1, 0.1, 2.9, 0.8],
        [1.2, 3.1, 1.5, 1.4],
    ])

    # reactions (c, p, s)
    N = torch.tensor([
        [   [-1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, -2.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
        [   [0.0, -3.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])

    # affinities (c, p, s)
    Km = torch.tensor([
        [   [1.2, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.9, 1.2],
            [0.0, 0.0, 0.0, 0.0] ],
        [   [0.0, 1.5, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0] ],
    ])
    
    # max velocities (c, p)
    Vmax = torch.tensor([
        [2.1, 1.1, 0.0],
        [1.9, 0.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4)

    # equilibrium constants (c, p)
    Ke = torch.full((2, 3), 999.9)

    # fmt: on

    def mm(x, k, v, n):
        return v * x ** n / (k + x) ** n

    # expected outcome
    dx_c0_b = 2 * mm(x=X0[0, 0], v=Vmax[0, 0], k=Km[0, 0, 0], n=1)
    dx_c0_a = -dx_c0_b / 2
    dx_c0_d = mm(x=X0[0, 2], v=Vmax[0, 1], k=Km[0, 1, 2], n=2)
    dx_c0_c = -2 * dx_c0_d

    v0_c1 = mm(x=X0[1, 1], v=Vmax[1, 0], k=Km[1, 0, 1], n=3)
    dx_c1_a = 0.0
    dx_c1_b = -3 * v0_c1
    dx_c1_c = 2 * v0_c1
    dx_c1_d = 0.0

    # test
    Xd = integrate_signals(X=X0, Km=Km, Vmax=Vmax, Ke=Ke, N=N, A=A)

    assert Xd[0, 0] == pytest.approx(dx_c0_a, abs=TOLERANCE)
    assert Xd[0, 1] == pytest.approx(dx_c0_b, abs=TOLERANCE)
    assert Xd[0, 2] == pytest.approx(dx_c0_c, abs=TOLERANCE)
    assert Xd[0, 3] == pytest.approx(dx_c0_d, abs=TOLERANCE)

    assert Xd[1, 0] == pytest.approx(dx_c1_a, abs=TOLERANCE)
    assert Xd[1, 1] == pytest.approx(dx_c1_b, abs=TOLERANCE)
    assert Xd[1, 2] == pytest.approx(dx_c1_c, abs=TOLERANCE)
    assert Xd[1, 3] == pytest.approx(dx_c1_d, abs=TOLERANCE)


def test_mm_kinetic_with_multiple_substrates():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a,b -> c, P1: b,d -> a2,c
    # cell 1: P0: a,d -> b

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [1.1, 2.1, 2.9, 0.8],
        [2.3, 0.4, 1.1, 3.2]
    ])

    # reactions (c, p, s)
    N = torch.tensor([
        [   [-1.0, -1.0, 1.0, 0.0],
            [2.0, -1.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
        [   [-1.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])

    # affinities (c, p, s)
    Km = torch.tensor([
        [   [2.2, 1.2, 0.2, 0.0],
            [0.8, 1.9, 0.4, 1.2],
            [0.0, 0.0, 0.0, 0.0] ],
        [   [2.3, 0.2, 0.0, 1.2],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0] ],
    ])

    # max velocities (c, p)
    Vmax = torch.tensor([
        [2.1, 1.1, 0.0],
        [1.2, 0.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4)

    # equilibrium constants (c, p)
    Ke = torch.full((2, 3), 999.9)

    # fmt: on

    def mm(x1, x2, k1, k2, v):
        return v * x1 * x2 / ((k1 + x1) * (k2 + x2))

    # expected outcome
    c0_v0 = mm(x1=X0[0, 0], k1=Km[0, 0, 0], x2=X0[0, 1], k2=Km[0, 0, 1], v=Vmax[0, 0])
    c0_v1 = mm(x1=X0[0, 1], k1=Km[0, 1, 1], x2=X0[0, 3], k2=Km[0, 1, 3], v=Vmax[0, 1])
    dx_c0_a = 2 * c0_v1 - c0_v0
    dx_c0_b = -c0_v0 - c0_v1
    dx_c0_c = c0_v0 + c0_v1
    dx_c0_d = -c0_v1
    c1_v0 = mm(x1=X0[1, 0], k1=Km[1, 0, 0], x2=X0[1, 3], k2=Km[1, 0, 3], v=Vmax[1, 0])
    dx_c1_a = -c1_v0
    dx_c1_b = c1_v0
    dx_c1_c = 0.0
    dx_c1_d = -c1_v0

    # test
    Xd = integrate_signals(X=X0, Km=Km, Vmax=Vmax, Ke=Ke, N=N, A=A)

    assert Xd[0, 0] == pytest.approx(dx_c0_a, abs=TOLERANCE)
    assert Xd[0, 1] == pytest.approx(dx_c0_b, abs=TOLERANCE)
    assert Xd[0, 2] == pytest.approx(dx_c0_c, abs=TOLERANCE)
    assert Xd[0, 3] == pytest.approx(dx_c0_d, abs=TOLERANCE)

    assert Xd[1, 0] == pytest.approx(dx_c1_a, abs=TOLERANCE)
    assert Xd[1, 1] == pytest.approx(dx_c1_b, abs=TOLERANCE)
    assert Xd[1, 2] == pytest.approx(dx_c1_c, abs=TOLERANCE)
    assert Xd[1, 3] == pytest.approx(dx_c1_d, abs=TOLERANCE)


def test_mm_kinetic_with_allosteric_action():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b, inhibitor=c, P1: c -> d, activator=a
    # cell 1: P0: a -> b, inhibitor=c,d, P1: c -> d, activator=a,b

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [2.1, 3.5, 1.9, 2.0],
        [3.2, 1.6, 4.0, 1.9],
    ])

    # reactions (c, p, s)
    N = torch.tensor([
        [   [-1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]   ],
        [   [-1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]   ],
    ])

    # affinities (c, p, s)
    Km = torch.tensor([
        [   [1.2, 3.1, 0.7, 0.0],
            [1.0, 0.0, 0.9, 1.2],
            [0.0, 0.0, 0.0, 0.0] ],
        [   [2.1, 1.3, 1.0, 0.6],
            [1.3, 0.8, 0.3, 1.1],
            [0.0, 0.0, 0.0, 0.0] ],
    ])

    # max velocities (c, p)
    Vmax = torch.tensor([
        [2.1, 2.0, 0.0],
        [3.2, 2.5, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.tensor([
        [   [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]   ],
        [   [0.0, 0.0, -1.0, -1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]   ],
    ])

    # equilibrium constants (c, p)
    Ke = torch.full((2, 3), 999.9)

    # fmt: on

    def mm(x, kx, v):
        return v * x / (kx + x)

    def fi(i, ki):
        return 1 - i / (ki + i)

    def fi2(i1, ki1, i2, ki2):
        return max(0, 1 - i1 * i2 / ((ki1 + i1) * (ki2 + i2)))

    def fa(a, ka):
        return a / (ka + a)

    def fa2(a1, ka1, a2, ka2):
        return min(1, a1 * a2 / ((ka1 + a1) * (ka2 + a2)))

    # expected outcome
    v0_c0 = mm(x=X0[0, 0], v=Vmax[0, 0], kx=Km[0, 0, 0]) * fi(
        i=X0[0, 2], ki=Km[0, 0, 2]
    )
    v1_c0 = mm(x=X0[0, 2], v=Vmax[0, 1], kx=Km[0, 1, 2]) * fa(
        a=X0[0, 0], ka=Km[0, 1, 0]
    )
    dx_c0_b = v0_c0
    dx_c0_a = -v0_c0
    dx_c0_c = -v1_c0
    dx_c0_d = v1_c0

    v0_c1 = mm(x=X0[1, 0], v=Vmax[1, 0], kx=Km[1, 0, 0]) * fi2(
        i1=X0[1, 2], ki1=Km[1, 0, 2], i2=X0[1, 3], ki2=Km[1, 0, 3]
    )
    v1_c1 = mm(x=X0[1, 2], v=Vmax[1, 1], kx=Km[1, 1, 2]) * fa2(
        a1=X0[1, 0], ka1=Km[1, 1, 0], a2=X0[1, 1], ka2=Km[1, 1, 1],
    )
    dx_c1_a = -v0_c1
    dx_c1_b = v0_c1
    dx_c1_c = -v1_c1
    dx_c1_d = v1_c1

    # test
    Xd = integrate_signals(X=X0, Km=Km, Vmax=Vmax, Ke=Ke, N=N, A=A)

    assert Xd[0, 0] == pytest.approx(dx_c0_a, abs=TOLERANCE)
    assert Xd[0, 1] == pytest.approx(dx_c0_b, abs=TOLERANCE)
    assert Xd[0, 2] == pytest.approx(dx_c0_c, abs=TOLERANCE)
    assert Xd[0, 3] == pytest.approx(dx_c0_d, abs=TOLERANCE)

    assert Xd[1, 0] == pytest.approx(dx_c1_a, abs=TOLERANCE)
    assert Xd[1, 1] == pytest.approx(dx_c1_b, abs=TOLERANCE)
    assert Xd[1, 2] == pytest.approx(dx_c1_c, abs=TOLERANCE)
    assert Xd[1, 3] == pytest.approx(dx_c1_d, abs=TOLERANCE)


def test_reduce_velocity_to_avoid_zero_concentrations():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b, P1: b -> d
    # cell 1: P0: 2c -> d

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [0.1, 0.1, 2.9, 0.8],
        [2.9, 3.1, 0.1, 1.0],
    ])

    # reactions (c, p, s)
    N = torch.tensor([
        [   [-1.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
        [   [0.0, 0.0, -2.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])

    # affinities (c, p, s)
    Km = torch.tensor([
        [   [1.2, 1.5, 0.0, 0.0],
            [0.0, 0.5, 0.0, 3.5],
            [0.0, 0.0, 0.0, 0.0] ],
        [   [0.0, 0.0, 0.5, 1.5],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0] ],
    ])

    # max velocities (c, p)
    Vmax = torch.tensor([
        [2.1, 1.0, 0.0],
        [3.1, 0.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4)

    # equilibrium constants (c, p)
    Ke = torch.full((2, 3), 999.9)

    # fmt: on

    def mm(x, k, v, n=1):
        return v * x ** n / (k + x) ** n

    # expected outcome
    v0_c0 = mm(x=X0[0, 0], v=Vmax[0, 0], k=Km[0, 0, 0])
    # but this would lead to Xd + X0 = -0.0615 (for a)
    assert X0[0, 0] - v0_c0 < EPS
    # so velocity should be reduced to a (it will be eps afterwards)
    v0_c0 = X0[0, 0] - EPS
    # the other protein was unaffected
    v1_c1 = mm(x=X0[0, 1], v=Vmax[0, 1], k=Km[0, 1, 1])
    dx_c0_a = -v0_c0
    dx_c0_b = v0_c0 - v1_c1
    dx_c0_c = 0.0
    dx_c0_d = v1_c1

    v0_c1 = mm(x=X0[1, 2], v=Vmax[1, 0], k=Km[1, 0, 2], n=2)
    # but this would lead to Xd + X0 = -0.0722 (for c)
    assert X0[1, 2] - 2 * v0_c1 < EPS
    # so velocity should be reduced to c (it will be 0 afterwards)
    v0_c1 = X0[1, 2] / 2 - EPS
    dx_c1_a = 0.0
    dx_c1_b = 0.0
    dx_c1_c = -2 * v0_c1
    dx_c1_d = v0_c1

    # test
    Xd = integrate_signals(X=X0, Km=Km, Vmax=Vmax, Ke=Ke, N=N, A=A)

    assert Xd[0, 0] == pytest.approx(dx_c0_a, abs=TOLERANCE)
    assert Xd[0, 1] == pytest.approx(dx_c0_b, abs=TOLERANCE)
    assert Xd[0, 2] == pytest.approx(dx_c0_c, abs=TOLERANCE)
    assert Xd[0, 3] == pytest.approx(dx_c0_d, abs=TOLERANCE)

    assert Xd[1, 0] == pytest.approx(dx_c1_a, abs=TOLERANCE)
    assert Xd[1, 1] == pytest.approx(dx_c1_b, abs=TOLERANCE)
    assert Xd[1, 2] == pytest.approx(dx_c1_c, abs=TOLERANCE)
    assert Xd[1, 3] == pytest.approx(dx_c1_d, abs=TOLERANCE)

    X1 = X0 + Xd
    assert not torch.any(X1 < EPS)


def test_reactions_are_turned_around():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b (but b/a > Ke), P1: b -> d
    # cell 1: P0: c -> d, P1: a -> d (but d/a > Ke)

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [0.9, 2.1, 2.9, 0.8],
        [1.0, 3.1, 2.1, 2.9],
    ])

    # reactions (c, p, s)
    N = torch.tensor([
        [   [-1.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
        [   [0.0, 0.0, -1.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])

    # affinities (c, p, s)
    Km = torch.tensor([
        [   [1.2, 1.5, 0.0, 0.0],
            [0.0, 0.5, 0.0, 3.5],
            [0.0, 0.0, 0.0, 0.0] ],
        [   [0.0, 0.0, 0.5, 1.5],
            [1.2, 0.0, 0.0, 1.9],
            [0.0, 0.0, 0.0, 0.0] ],
    ])

    # max velocities (c, p)
    Vmax = torch.tensor([
        [2.1, 1.0, 0.0],
        [1.1, 2.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4)

    # equilibrium constants (c, p)
    Ke = torch.full((2, 3), 999.9)
    Ke[0, 0] = 2.2 # b/a = 2.3, so b -> a
    Ke[1, 1] = 2.8 # d/a = 2.9, so d -> a

    # fmt: on

    def mm(x, k, v):
        return v * x / (k + x)

    # expected outcome
    v0_c0 = mm(x=X0[0, 1], v=Vmax[0, 0], k=Km[0, 0, 1])
    v1_c0 = mm(x=X0[0, 1], v=Vmax[0, 1], k=Km[0, 1, 1])
    dx_c0_a = v0_c0
    dx_c0_b = -v0_c0 - v1_c0
    dx_c0_c = 0.0
    dx_c0_d = v1_c0

    v0_c1 = mm(x=X0[1, 2], v=Vmax[1, 0], k=Km[1, 0, 2])
    v1_c1 = mm(x=X0[1, 3], v=Vmax[1, 1], k=Km[1, 1, 3])
    dx_c1_a = v1_c1
    dx_c1_b = 0.0
    dx_c1_c = -v0_c1
    dx_c1_d = v0_c1 - v1_c1

    # test
    Xd = integrate_signals(X=X0, Km=Km, Vmax=Vmax, Ke=Ke, N=N, A=A)

    assert Xd[0, 0] == pytest.approx(dx_c0_a, abs=TOLERANCE)
    assert Xd[0, 1] == pytest.approx(dx_c0_b, abs=TOLERANCE)
    assert Xd[0, 2] == pytest.approx(dx_c0_c, abs=TOLERANCE)
    assert Xd[0, 3] == pytest.approx(dx_c0_d, abs=TOLERANCE)

    assert Xd[1, 0] == pytest.approx(dx_c1_a, abs=TOLERANCE)
    assert Xd[1, 1] == pytest.approx(dx_c1_b, abs=TOLERANCE)
    assert Xd[1, 2] == pytest.approx(dx_c1_c, abs=TOLERANCE)
    assert Xd[1, 3] == pytest.approx(dx_c1_d, abs=TOLERANCE)


def test_substrate_concentrations_never_get_too_low():
    n_cells = 1000
    n_prots = 100
    n_mols = 20

    # concentrations (c, s)
    X0 = torch.randn(n_cells, n_mols).abs()

    # reactions (c, p, s)
    N = torch.randint(low=-3, high=4, size=(n_cells, n_prots, n_mols))

    # affinities (c, p, s)
    Km = torch.randn(n_cells, n_prots, n_mols).abs()

    # max velocities (c, p)
    Vmax = torch.randn(n_cells, n_prots).abs()

    # allosterics (c, p, s)
    A = torch.randint(low=-1, high=2, size=(n_cells, n_prots, n_mols))

    # equilibrium constants (c, p)
    Ke = torch.full((n_cells, n_prots), 999.9)

    # test
    Xd = integrate_signals(X=X0, Km=Km, Vmax=Vmax, Ke=Ke, N=N, A=A)

    X1 = X0 + Xd
    assert not torch.any(X1 + TOLERANCE < EPS)
