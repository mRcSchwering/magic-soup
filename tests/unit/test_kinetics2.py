import random

import pytest
import torch
from magicsoup.kinetics2 import Kinetics

_INF = float("inf")
_ATOL = 1e-4
_RTOL = 1e-4
_KINETICS = Kinetics(device="cpu")


def test_get_bounds():
    # 2 cells, 3 proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: 2a -> 5b, P1: 10b -> d
    # cell 1: P0: 3c -> 7d, P1: a -> d

    # stoichiometry
    N = torch.tensor(
        [
            [
                [-2, 5, 0, 0],
                [0, -10, 0, 1],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, -3, 7],
                [-1, 0, 0, 1],
                [0, 0, 0, 0],
            ],
        ],
        dtype=torch.float32,
    )

    # background
    B = torch.tensor(
        [
            [
                [10, 10, 10, 10],
                [0, 1, 0, 1],
                [10, 10, 10, 10],
            ],
            [
                [3, 7, 5, 1],
                [0, 0, 0, 1],
                [10, 10, 10, 10],
            ],
        ],
        dtype=torch.float32,
    )

    # expected boundaries
    lo_exp = torch.tensor(
        [
            [-2, -1, -_INF],
            [-1 / 7, -1, -_INF],
        ],
        dtype=torch.float32,
    )
    hi_exp = torch.tensor(
        [
            [5, 1 / 10, _INF],
            [5 / 3, 0, _INF],
        ],
        dtype=torch.float32,
    )

    lo, hi = _KINETICS._get_bounds(B=B, N=N)

    torch.testing.assert_close(lo, lo_exp, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(hi, hi_exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
@pytest.mark.parametrize("_", range(100))
def test_get_bounds_randomly(_: int):
    c = random.randint(2, 100)
    p = random.randint(1, 100)
    m = random.randint(2, 100)
    N = torch.randint(-10, 10, (c, p, m), dtype=torch.float32)
    N[0, 0, :] = 0.0
    B = torch.rand(c, p, m, dtype=torch.float32)
    lo, hi = _KINETICS._get_bounds(B=B, N=N)
    assert lo.shape == (c, p)
    assert hi.shape == (c, p)
    assert lo[0, 0] == -float("inf")
    assert hi[0, 0] == float("inf")
    assert lo.isnan().sum() == 0
    assert hi.isnan().sum() == 0
    assert (lo <= hi).all()


def test_get_alpha_cat_simple_mm_kinetic():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b, P1: b -> d
    # cell 1: P0: c -> d, P1: a -> d

    # concentrations
    x0 = torch.tensor(
        [
            [2.1, 1.9, 0.0, 0.8],
            [2.9, 3.1, 2.1, 1.0],
        ],
        dtype=torch.float32,
    )
    x0_ = torch.broadcast_to(x0.unsqueeze(1), (2, 3, 4))

    # stoichiometry
    N = torch.tensor(
        [
            [
                [-1, 1, 0, 0],
                [0, -1, 0, 1],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, -1, 1],
                [-1, 0, 0, 1],
                [0, 0, 0, 0],
            ],
        ],
        dtype=torch.float32,
    )
    N_f = torch.where(N < 0, -N, 0)
    N_b = torch.where(N > 0, N, 0)

    # affinities
    k_f = torch.tensor(
        [
            [1.3, 2.1, 0.0],
            [1.0, 1.7, 0.0],
        ],
        dtype=torch.float32,
    )
    k_b = torch.tensor(
        [
            [0.3, 1.1, 0.0],
            [1.5, 0.7, 0.0],
        ],
        dtype=torch.float32,
    )

    def f(s, p, kf, kb):
        vf = (s / kf - p / kb) / (1 + s / kf + p / kb)
        vb = (p / kb - s / kf) / (1 + s / kf + p / kb)
        return (vf - vb) / 2

    # expected outcome
    a_c0_0 = f(x0[0, 0], x0[0, 1], k_f[0, 0], k_b[0, 0])
    a_c0_1 = f(x0[0, 1], x0[0, 3], k_f[0, 1], k_b[0, 1])

    a_c1_0 = f(x0[1, 2], x0[1, 3], k_f[1, 0], k_b[1, 0])
    a_c1_1 = f(x0[1, 0], x0[1, 3], k_f[1, 1], k_b[1, 1])

    a_exp = torch.tensor(
        [
            [a_c0_0, a_c0_1, 0.0],
            [a_c1_0, a_c1_1, 0.0],
        ],
        dtype=torch.float32,
    )

    # test
    a = _KINETICS._get_alpha_cat(x=x0_, N_f=N_f, N_b=N_b, k_f=k_f, k_b=k_b)
    torch.testing.assert_close(a, a_exp, atol=_ATOL, rtol=_RTOL)


def test_get_alpha_cat_mm_kinetic_with_proportions():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> 2b, P1: 2c -> d
    # cell 1: P0: 3b -> 2c

    # concentrations
    x0 = torch.tensor(
        [
            [1.1, 0.1, 2.9, 0.8],
            [1.2, 4.9, 5.1, 1.4],
        ],
        dtype=torch.float32,
    )
    x0_ = torch.broadcast_to(x0.unsqueeze(1), (2, 3, 4))

    # reactions
    N = torch.tensor(
        [
            [
                [-1, 2, 0, 0],
                [0, 0, -2, 1],
                [0, 0, 0, 0],
            ],
            [
                [0, -3, 2, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
        dtype=torch.float32,
    )
    N_f = torch.where(N < 0, -N, 0)
    N_b = torch.where(N > 0, N, 0)

    # affinities
    k_f = torch.tensor(
        [
            [1.3, 2.1, 0.0],
            [1.4, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    k_b = torch.tensor(
        [
            [0.3, 1.1, 0.0],
            [1.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    def f12(s, p, kf, kb):
        af = (s / kf - p**2 / kb) / (1 + s / kf + p**2 / kb)
        ab = (p**2 / kb - s / kf) / (1 + s / kf + p**2 / kb)
        return (af - ab) / 2

    def f21(s, p, kf, kb):
        af = (s**2 / kf - p / kb) / (1 + s**2 / kf + p / kb)
        ab = (p / kb - s**2 / kf) / (1 + s**2 / kf + p / kb)
        return (af - ab) / 2

    def f32(s, p, kf, kb):
        af = (s**3 / kf - p**2 / kb) / (1 + s**3 / kf + p**2 / kb)
        ab = (p**2 / kb - s**3 / kf) / (1 + s**3 / kf + p**2 / kb)
        return (af - ab) / 2

    # expected outcome
    a_c0_0 = f12(x0[0, 0], x0[0, 1], k_f[0, 0], k_b[0, 0])
    a_c0_1 = f21(x0[0, 2], x0[0, 3], k_f[0, 1], k_b[0, 1])

    a_c1_0 = f32(x0[1, 1], x0[1, 2], k_f[1, 0], k_b[1, 0])

    a_exp = torch.tensor(
        [
            [a_c0_0, a_c0_1, 0.0],
            [a_c1_0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    # test
    a = _KINETICS._get_alpha_cat(x=x0_, N_f=N_f, N_b=N_b, k_f=k_f, k_b=k_b)
    torch.testing.assert_close(a, a_exp, atol=_ATOL, rtol=_RTOL)


def test_get_alpha_cat_mm_kinetic_with_multiple_substrates():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a,b -> c, P1: b,d -> 2a,c
    # cell 1: P0: a,d -> b

    # concentrations
    x0 = torch.tensor(
        [
            [1.1, 2.1, 2.9, 0.8],
            [2.3, 0.4, 0.0, 3.2],
        ],
        dtype=torch.float32,
    )
    x0_ = torch.broadcast_to(x0.unsqueeze(1), (2, 3, 4))

    # reactions
    N = torch.tensor(
        [
            [
                [-1, -1, 1, 0],
                [2, -1, 1, -1],
                [0, 0, 0, 0],
            ],
            [
                [-1, 1, 0, -1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
        dtype=torch.float32,
    )
    N_f = torch.where(N < 0, -N, 0)
    N_b = torch.where(N > 0, N, 0)

    # affinities
    k_f = torch.tensor(
        [
            [1.3, 2.1, 0.0],
            [1.4, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    k_b = torch.tensor(
        [
            [0.3, 1.1, 0.0],
            [1.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    def f111(s1, s2, p, kf, kb):
        af = (s1 * s2 / kf - p / kb) / (1 + s1 * s2 / kf + p / kb)
        ab = (p / kb - s1 * s2 / kf) / (1 + s1 * s2 / kf + p / kb)
        return (af - ab) / 2

    def f1121(s1, s2, p1, p2, kf, kb):
        base = 1 + s1 * s2 / kf + p1**2 * p2 / kb
        af = (s1 * s2 / kf - p1**2 * p2 / kb) / base
        ab = (p1**2 * p2 / kb - s1 * s2 / kf) / base
        return (af - ab) / 2

    # expected outcome
    a_c0_0 = f111(x0[0, 0], x0[0, 1], x0[0, 2], k_f[0, 0], k_b[0, 0])
    a_c0_1 = f1121(x0[0, 1], x0[0, 3], x0[0, 0], x0[0, 2], k_f[0, 1], k_b[0, 1])

    a_c1_0 = f111(x0[1, 0], x0[1, 3], x0[1, 1], k_f[1, 0], k_b[1, 0])

    a_exp = torch.tensor(
        [
            [a_c0_0, a_c0_1, 0.0],
            [a_c1_0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    # test
    a = _KINETICS._get_alpha_cat(x=x0_, N_f=N_f, N_b=N_b, k_f=k_f, k_b=k_b)
    torch.testing.assert_close(a, a_exp, atol=_ATOL, rtol=_RTOL)


def test_get_alpha_cat_mm_kinetic_with_cofactors():
    # 2 cell, 3 proteins, 4 molecules (a, b, c, d)
    # N for a molecule might be 0 but it's still required
    # cell 0: P0: a + b -> b + c
    # cell 1: P0: a + c -> b + c

    # concentrations
    x0 = torch.tensor(
        [
            [10.0, 0.1, 3.0, 0.8],
            [10.0, 3.0, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )
    x0_ = torch.broadcast_to(x0.unsqueeze(1), (2, 3, 4))

    # reactions
    N_f = torch.tensor(
        [
            [
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
        dtype=torch.float32,
    )
    N_b = torch.tensor(
        [
            [
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
        dtype=torch.float32,
    )

    # affinities
    k_f = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    k_b = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    def f(s1, s2, p1, p2, kf, kb):
        af = (s1 * s2 / kf - p1 * p2 / kb) / (1 + s1 * s2 / kf + p1 * p2 / kb)
        ab = (p1 * p2 / kb - s1 * s2 / kf) / (1 + s1 * s2 / kf + p1 * p2 / kb)
        return (af - ab) / 2

    # expected outcome
    a_c0_0 = f(x0[0, 0], x0[0, 1], x0[0, 1], x0[0, 2], k_f[0, 0], k_b[0, 0])

    a_c1_0 = f(x0[1, 0], x0[1, 2], x0[1, 1], x0[1, 2], k_f[1, 0], k_b[1, 0])

    a_exp = torch.tensor(
        [
            [a_c0_0, 0.0, 0.0],
            [a_c1_0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    # test
    a = _KINETICS._get_alpha_cat(x=x0_, N_f=N_f, N_b=N_b, k_f=k_f, k_b=k_b)
    torch.testing.assert_close(a, a_exp, atol=_ATOL, rtol=_RTOL)


def test_mm_kinetic_with_allosteric_action():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b, inhibitor=c, P1: c -> d, activator=a, P2: a -> b, inh=c, act=d
    # cell 1: P0: a -> b, inhibitor=c,d, P1: c -> d, activator=a,b

    # fmt: off

    # concentrations (c, s)
    x0 = torch.tensor([
        [2.1, 3.5, 1.9, 2.0],
        [3.2, 1.6, 4.0, 1.9],
    ])

    # reactions (c, p, s)
    N = torch.tensor([
        [   [-1, 1, 0, 0],
            [0, 0, -1, 1],
            [-1, 1, 0, 0]   ],
        [   [-1, 1, 0, 0],
            [0, 0, -1, 1],
            [0, 0, 0, 0]   ],
    ], dtype=torch.int32)
    Nf = torch.where(N < 0, -N, 0)
    Nb = torch.where(N > 0, N, 0)

    # affinities (c, p, s)
    Kmf = torch.tensor([
        [1.3, 2.1, 0.9],
        [1.4, 2.2, 0.0],
    ])
    Kmb = torch.tensor([
        [1.1, 1.1, 1.0],
        [1.5, 1.9, 0.0],
    ])
    Kmr = torch.tensor([
        [   [0.0, 0.0, 1.3, 0.0],
            [2.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.9]   ],
        [   [0.0, 0.0, 1.4, 1.4],
            [2.2, 2.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]   ],
    ])

    # max velocities (c, p)
    Vmax = torch.tensor([
        [2.1, 2.0, 1.0],
        [3.2, 2.5, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.tensor([
        [   [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, -1, 1]   ],
        [   [0, 0, -1, -1],
            [1, 1, 0, 0],
            [0, 0, 0, 0]   ],
    ], dtype=torch.int32)

    def mm(s: float, p: float, kf: float, kb: float, v: float) -> float:
        vf = v * (s / kf - p / kb) / (1 + s / kf + p / kb)
        vb = v * (p / kb - s / kf) / (1 + s / kf + p / kb)
        return (vf - vb) / 2

    def al(x, k, n):
        return x**n / (k**n + x**n)

    # expected outcome
    v_c0_0 = (
        mm(x0[0, 0], x0[0, 1], Kmf[0, 0], Kmb[0, 0], Vmax[0, 0])
        * al(x0[0, 2], Kmr[0, 0, 2], A[0, 0, 2])
    )
    v_c0_1 = (
        mm(x0[0, 2], x0[0, 3], Kmf[0, 1], Kmb[0, 1], Vmax[0, 1])
        * al(x0[0, 0], Kmr[0, 1, 0], A[0, 1, 0])
    )
    v_c0_2 = (
        mm(x0[0, 0], x0[0, 1], Kmf[0, 2], Kmb[0, 2], Vmax[0, 2])
        * al(x0[0, 2], Kmr[0, 2, 2], A[0, 2, 2])
        * al(x0[0, 3], Kmr[0, 2, 3], A[0, 2, 3])
    )
    dx_c0_a = -v_c0_0 - v_c0_2
    dx_c0_b = v_c0_0 + v_c0_2
    dx_c0_c = -v_c0_1
    dx_c0_d = v_c0_1

    v_c1_0 = (
        mm(x0[1, 0], x0[1, 1], Kmf[1, 0], Kmb[1, 0], Vmax[1, 0])
        * al(x0[1, 2], Kmr[1, 0, 2], A[1, 0, 2])
        * al(x0[1, 3], Kmr[1, 0, 3], A[1, 0, 3])
    )
    v_c1_1 = (
        mm(x0[1, 2], x0[1, 3], Kmf[1, 1], Kmb[1, 1], Vmax[1, 1])
        * al(x0[1, 0], Kmr[1, 1, 0], A[1, 1, 0])
        * al(x0[1, 1], Kmr[1, 1, 1], A[1, 1, 1])
    )
    dx_c1_a = -v_c1_0
    dx_c1_b = v_c1_0
    dx_c1_c = -v_c1_1
    dx_c1_d = v_c1_1
    # fmt: on

    # test
    kinetics = _get_kinetics()
    kinetics.N = N
    kinetics.Nf = Nf
    kinetics.Nb = Nb
    kinetics.Ke = Kmb / Kmf
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = torch.pow(Kmr, A)
    kinetics.Vmax = Vmax
    kinetics.A = A
    dx = kinetics.integrate_signals(X=x0) - x0

    assert (dx[0, 0] - dx_c0_a).abs() < _TOLERANCE
    assert (dx[0, 1] - dx_c0_b).abs() < _TOLERANCE
    assert (dx[0, 2] - dx_c0_c).abs() < _TOLERANCE
    assert (dx[0, 3] - dx_c0_d).abs() < _TOLERANCE

    assert (dx[1, 0] - dx_c1_a).abs() < _TOLERANCE
    assert (dx[1, 1] - dx_c1_b).abs() < _TOLERANCE
    assert (dx[1, 2] - dx_c1_c).abs() < _TOLERANCE
    assert (dx[1, 3] - dx_c1_d).abs() < _TOLERANCE


def test_reduce_velocity_to_avoid_negative_concentrations():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b, P1: b -> d
    # cell 1: P0: 2c -> d

    # fmt: off

    # concentrations (c, s)
    x0 = torch.tensor([
        [0.1, 1.0, 2.9, 0.8],
        [2.9, 3.1, 0.1, 0.3],
    ])

    # reactions (c, p, s)
    N = torch.tensor([
        [   [-1, 1, 0, 0],
            [0, -1, 0, 1],
            [0, 0, 0, 0]    ],
        [   [0, 0, -2, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]    ],
    ], dtype=torch.int32)
    Nf = torch.where(N < 0, -N, 0)
    Nb = torch.where(N > 0, N, 0)

    # affinities (c, p, s)
    Kmf = torch.tensor([
        [0.1, 2.1, 0.0],
        [0.1, 0.0, 0.0],
    ])
    Kmb = torch.tensor([
        [10.3, 1.1, 0.0],
        [10.5, 0.0, 0.0],
    ])
    Kmr = torch.zeros(2, 3, 4)

    # max velocities (c, p)
    Vmax = torch.tensor([
        [2.1, 1.0, 0.0],
        [3.1, 0.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4, dtype=torch.int32)

    # fmt: on

    def mm(s: float, p: float, kf: float, kb: float, v: float) -> float:
        vf = v * (s / kf - p / kb) / (1 + s / kf + p / kb)
        vb = v * (p / kb - s / kf) / (1 + s / kf + p / kb)
        return (vf - vb) / 2

    def mm21(s: float, p: float, kf: float, kb: float, v: float) -> float:
        vf = v * (s**2 / kf - p / kb) / (1 + s**2 / kf + p / kb)
        vb = v * (p / kb - s**2 / kf) / (1 + s**2 / kf + p / kb)
        return (vf - vb) / 2

    # expected outcome
    v_c0_0 = mm(x0[0, 0], x0[0, 1], Kmf[0, 0], Kmb[0, 0], Vmax[0, 0])
    v_c0_1 = mm(x0[0, 1], x0[0, 3], Kmf[0, 1], Kmb[0, 1], Vmax[0, 1])
    # but this would lead to dx + x0 = -0.804 (for a)
    assert x0[0, 0] - v_c0_0 < 0.0
    # so velocity should be reduced by a factor depending on current a
    # only P0 is reduced by this factor because its the only one reducing a
    f = (v_c0_0 - (v_c0_0 - x0[0, 0].item())) / v_c0_0
    v_c0_0 = f * v_c0_0
    dx_c0_a = -v_c0_0
    dx_c0_b = v_c0_0 - v_c0_1
    dx_c0_c = 0.0
    dx_c0_d = v_c0_1

    v_c1_0 = mm21(x0[1, 2], x0[1, 3], Kmf[1, 0], Kmb[1, 0], Vmax[1, 0])
    # but this would lead to dx + x0 = -0.0722 (for c)
    assert x0[1, 2] - 2 * v_c1_0 < 0.0
    # as above, velocities are reduced
    f = (v_c1_0 - (v_c1_0 - x0[1, 2].item())) / v_c1_0 / 2
    v_c1_0 = f * v_c1_0
    dx_c1_a = 0.0
    dx_c1_b = 0.0
    dx_c1_c = -2 * v_c1_0
    dx_c1_d = v_c1_0

    # test
    kinetics = _get_kinetics()
    kinetics.N = N
    kinetics.Nf = Nf
    kinetics.Nb = Nb
    kinetics.Ke = Kmb / Kmf
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A

    dx = kinetics.integrate_signals(X=x0) - x0

    assert (dx[0, 0] - dx_c0_a).abs() < _TOLERANCE
    assert (dx[0, 1] - dx_c0_b).abs() < _TOLERANCE
    assert (dx[0, 2] - dx_c0_c).abs() < _TOLERANCE
    assert (dx[0, 3] - dx_c0_d).abs() < _TOLERANCE

    assert (dx[1, 0] - dx_c1_a).abs() < _TOLERANCE
    assert (dx[1, 1] - dx_c1_b).abs() < _TOLERANCE
    assert (dx[1, 2] - dx_c1_c).abs() < _TOLERANCE
    assert (dx[1, 3] - dx_c1_d).abs() < _TOLERANCE

    X1 = x0 + dx
    assert not torch.any(X1 < 0.0)


def test_reduce_velocity_in_multiple_proteins():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b, P1: 2a -> d
    # cell 1: P0: a -> b

    # fmt: off

    # concentrations (c, s)
    x0 = torch.tensor([
        [2.0, 1.2, 2.9, 1.5],
        [2.9, 3.1, 0.1, 1.0],
    ])

    # reactions (c, p, s)
    N = torch.tensor([
        [   [-1, 1, 0, 0],
            [-2, 0, 0, 1],
            [0, 0, 0, 0]    ],
        [   [-1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]    ],
    ], dtype=torch.int32)
    Nf = torch.where(N < 0, -N, 0)
    Nb = torch.where(N > 0, N, 0)

    # affinities (c, p, s)
    Kmf = torch.tensor([
        [0.1, 2.1, 0.0],
        [0.1, 0.0, 0.0],
    ])
    Kmb = torch.tensor([
        [10.3, 1.1, 0.0],
        [1.5, 0.0, 0.0],
    ])
    Kmr = torch.zeros(2, 3, 4)

    # max velocities (c, p)
    Vmax = torch.tensor([
        [3.1, 2.0, 0.0],
        [3.1, 0.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4, dtype=torch.int32)

    # fmt: on

    def mm(s: float, p: float, kf: float, kb: float, v: float) -> float:
        vf = v * (s / kf - p / kb) / (1 + s / kf + p / kb)
        vb = v * (p / kb - s / kf) / (1 + s / kf + p / kb)
        return (vf - vb) / 2

    def mm21(s: float, p: float, kf: float, kb: float, v: float) -> float:
        vf = v * (s**2 / kf - p / kb) / (1 + s**2 / kf + p / kb)
        vb = v * (p / kb - s**2 / kf) / (1 + s**2 / kf + p / kb)
        return (vf - vb) / 2

    # expected outcome
    v_c0_0 = mm(x0[0, 0], x0[0, 1], Kmf[0, 0], Kmb[0, 0], Vmax[0, 0])
    v_c0_1 = mm21(x0[0, 0], x0[0, 3], Kmf[0, 1], Kmb[0, 1], Vmax[0, 1])
    # but this would lead to a < 0.0
    naive_dx_c0_a = -v_c0_0 - 2 * v_c0_1
    assert x0[0, 0] + naive_dx_c0_a < 0.0
    # so velocity should be reduced to by a factor to not deconstruct too much a
    # all other proteins have to be reduced by the same factor to not cause downstream problems
    f = x0[0, 0] / -naive_dx_c0_a
    v_c0_0 = v_c0_0 * f
    v_c0_1 = v_c0_1 * f
    dx_c0_a = -v_c0_0 - v_c0_1 * 2
    dx_c0_b = v_c0_0
    dx_c0_c = 0.0
    dx_c0_d = v_c0_1

    # cell1 is business as usual
    v_c1_0 = mm(x0[1, 0], x0[1, 1], Kmf[1, 0], Kmb[1, 0], Vmax[1, 0])
    dx_c1_a = -v_c1_0
    dx_c1_b = v_c1_0
    dx_c1_c = 0.0
    dx_c1_d = 0.0

    # test
    kinetics = _get_kinetics()
    kinetics.N = N
    kinetics.Nf = Nf
    kinetics.Nb = Nb
    kinetics.Ke = Kmb / Kmf
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A

    dx = kinetics.integrate_signals(X=x0) - x0

    assert (dx[0, 0] - dx_c0_a).abs() < _TOLERANCE
    assert (dx[0, 1] - dx_c0_b).abs() < _TOLERANCE
    assert (dx[0, 2] - dx_c0_c).abs() < _TOLERANCE
    assert (dx[0, 3] - dx_c0_d).abs() < _TOLERANCE

    assert (dx[1, 0] - dx_c1_a).abs() < _TOLERANCE
    assert (dx[1, 1] - dx_c1_b).abs() < _TOLERANCE
    assert (dx[1, 2] - dx_c1_c).abs() < _TOLERANCE
    assert (dx[1, 3] - dx_c1_d).abs() < _TOLERANCE

    X1 = x0 + dx
    assert not torch.any(X1 < 0.0)


def test_multiply_signals():
    kinetics = _get_kinetics()
    # 4 signals: s0, s1, s2, s3
    # 2 proteins: p0, p1, p2
    # fmt: off

    # signals (c, s)
    X = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],  # low concentrations
        [100.0, 200.0, 300.0, 400.0],  # high concentrations
        [0.0, 0.0, 3.0, 4.0],  # some zeros
        [0.0, 0.0, 0.0, 0.0],  # all zero
    ])

    # one side of stoichiometry (c, p, s)
    N = torch.tensor([
        [
            [0, 1, 2, 0],  # B + 2C
            [3, 0, 0, 0],  # 3A
            [0, 0, 0, 0]
        ],
        [
            [10, 10, 5, 0],  # 10A + 10B + 5C
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ],
        [
            [2, 1, 2, 0],  # 2A + B + 2C
            [0, 0, 1, 2],  # 2D + C
            [0, 0, 0, 0]
        ],
        [
            [1, 1, 1, 1],  # A + B + C + D
            [1, 2, 0, 0],  # A + 2B
            [0, 0, 0, 0]
        ],
    ], dtype=torch.int32)

    # fmt: on

    # note: prots is a mask that identifies proteins
    #       which are involved
    #       xx of non-involved prots is useless
    xx, prots = kinetics._multiply_signals(X=X, N=N)
    assert xx.size() == (4, 3)
    assert prots.size() == (4, 3)

    # cell 0:
    p = prots[0]
    x = xx[0]
    assert p[0].item() is True
    assert p[1].item() is True
    assert p[2].item() is False
    assert x[0] == X[0, 1] * X[0, 2] ** 2  # B + 2C
    assert x[1] == X[0, 0] ** 3  # 3A

    # cell 1:
    p = prots[1]
    x = xx[1]
    assert p[0].item() is True
    assert p[1].item() is False
    assert p[2].item() is False
    assert x[0] == _MAX  # 10A + 10B + 5C

    # cell 2:
    p = prots[2]
    x = xx[2]
    assert p[0].item() is True
    assert p[1].item() is True
    assert p[2].item() is False
    assert x[0] == 0.0  # 2A + B + 2C, where A,B=0
    assert x[1] == X[2, 3] ** 2 * X[2, 2]  # 2D + C

    # cell 3:
    p = prots[3]
    x = xx[3]
    assert p[0].item() is True
    assert p[1].item() is True
    assert p[2].item() is False
    assert x[0] == 0.0  # A + B + C + D, where all 0
    assert x[1] == 0.0  # A + B + C + D, where all 0
