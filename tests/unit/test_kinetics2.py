import logging
import random

import pytest
import torch
from magicsoup.kinetics2 import Kinetics

from tests.config import DEVICE

_ATOL = 1e-4
_RTOL = 1e-4
_INT = torch.int8
_FLOAT = torch.float32

_kinetics = Kinetics(device=DEVICE)


def test_get_bounds():
    # 2 cells, 3 proteins, 4 molecules (a, b, c, d)
    c = 2
    p = 3
    m = 4
    h = 0.5
    v_max = torch.zeros((c, p), dtype=_FLOAT)
    N = torch.zeros((c, p, m), dtype=_FLOAT)
    B = torch.zeros((c, p, m), dtype=_FLOAT)

    # cell 0: P0: 2a -> 5b, P1: 10b -> d
    v_max[0, :] = torch.tensor([10, 10, 10], dtype=_FLOAT)
    N[0, 0, :] = torch.tensor([-2, 5, 0, 0], dtype=_FLOAT)
    N[0, 1, :] = torch.tensor([0, -10, 0, 1], dtype=_FLOAT)
    B[0, 0, :] = torch.tensor([10, 10, 10, 10], dtype=_FLOAT)
    B[0, 1, :] = torch.tensor([0, 1, 0, 1], dtype=_FLOAT)
    B[0, 2, :] = torch.tensor([10, 10, 10, 10], dtype=_FLOAT)

    # cell 1: P0: 3c -> 7d, P1: a -> d
    v_max[1, :] = torch.tensor([5, 5, 5], dtype=_FLOAT)
    N[1, 0, :] = torch.tensor([0, 0, -3, 7], dtype=_FLOAT)
    N[1, 1, :] = torch.tensor([-1, 0, 0, 1], dtype=_FLOAT)
    B[1, 0, :] = torch.tensor([3, 7, 5, 1], dtype=_FLOAT)
    B[1, 1, :] = torch.tensor([0, 0, 0, 1], dtype=_FLOAT)
    B[1, 2, :] = torch.tensor([10, 10, 10, 10], dtype=_FLOAT)

    # expected boundaries
    lo_exp = torch.tensor(
        [
            [-2, -1, -v_max[0, 2] * h],
            [-1 / 7, -1, -v_max[1, 2] * h],
        ],
        dtype=_FLOAT,
    )
    hi_exp = torch.tensor(
        [
            [5, 1 / 10, v_max[0, 2] * h],
            [5 / 3, 0, v_max[1, 2] * h],
        ],
        dtype=_FLOAT,
    )

    lo, hi = _kinetics._get_bounds(B=B, N=N, v_max=v_max, h=h)

    torch.testing.assert_close(lo, lo_exp, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(hi, hi_exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
def test_get_bounds_randomly():
    for _ in range(100):
        c = random.randint(2, 1000)
        p = random.randint(1, 100)
        m = random.randint(2, 100)
        h = random.uniform(0, 10)
        v_max = torch.rand(c, p, dtype=_FLOAT) * 100
        B = torch.rand(c, p, m, dtype=_FLOAT) * 1000
        N = torch.randint(-100, 100, (c, p, m), dtype=_FLOAT)
        N[0, 0, :] = 0.0
        lo, hi = _kinetics._get_bounds(B=B, N=N, v_max=v_max, h=h)
        assert lo.shape == (c, p)
        assert hi.shape == (c, p)
        assert lo[0, 0] == -v_max[0, 0] * h
        assert hi[0, 0] == v_max[0, 0] * h
        assert lo.isnan().sum() == 0
        assert hi.isnan().sum() == 0
        assert (lo <= hi).all()


# TODO: refactor
def test_get_alpha_cat():
    # n cells, 3 max proteins, 4 molecules (a, b, c, d)
    # simple MM kinetics
    # cell 0: P0: a -> b, P1: b -> d
    # cell 1: P0: c -> d, P1: a -> d
    # multiple substrates and products
    # cell 2: P0: a -> 2b, P1: 2c -> d
    # cell 3: P0: 3b -> 2c
    # cell 4: P0: a,b -> c, P1: b,d -> 2a,c
    # cell 5: P0: a,d -> b
    # co-factors involved (required but n_f+n_b=0)
    # cell 6: P0: a + b -> b + c
    # cell 7: P0: a + c -> b + c

    x0 = torch.tensor(
        [
            [2.1, 1.9, 0.0, 0.8],
            [2.9, 3.1, 2.1, 1.0],
            [1.1, 0.1, 2.9, 0.8],
            [1.2, 4.9, 5.1, 1.4],
            [1.1, 2.1, 2.9, 0.8],
            [2.3, 0.4, 0.0, 3.2],
            [10.0, 0.1, 3.0, 0.8],
            [10.0, 3.0, 0.1, 0.0],
        ],
        dtype=_FLOAT,
    )
    x0_ = torch.broadcast_to(x0.unsqueeze(1), (len(x0), 3, 4))
    N_f = torch.tensor(
        [
            [
                [1, 0, 0, 0],  # P0: a -> b
                [0, 1, 0, 0],  # P1: b -> d
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 0],  # P0: c -> d
                [1, 0, 0, 0],  # P1: a -> d
                [0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 0],  # P0: a -> 2b
                [0, 0, 2, 0],  # P1: 2c -> d
                [0, 0, 0, 0],
            ],
            [
                [0, 3, 0, 0],  # P0: 3b -> 2c
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 1, 0, 0],  # P0: a,b -> c
                [0, 1, 0, 1],  # P1: b,d -> 2a,c
                [0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 1],  # P0: a,d -> b
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 1, 0, 0],  # P0: a + b -> b + c
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 0, 1, 0],  # P0: a + c -> b + c
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
        dtype=_INT,
    )
    N_b = torch.tensor(
        [
            [
                [0, 1, 0, 0],  # P0: a -> b
                [0, 0, 0, 1],  # P1: b -> d
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 1],  # P0: c -> d
                [0, 0, 0, 1],  # P1: a -> d
                [0, 0, 0, 0],
            ],
            [
                [0, 2, 0, 0],  # P0: a -> 2b
                [0, 0, 0, 1],  # P1: 2c -> d
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 2, 0],  # P0: 3b -> 2c
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 0],  # P0: a,b -> c
                [2, 0, 1, 0],  # P1: b,d -> 2a,c
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 0, 0],  # P0: a,d -> b
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 1, 0],  # P0: a + b -> b + c
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 1, 0],  # P0: a + c -> b + c
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
        dtype=_INT,
    )
    k_f = torch.tensor(
        [
            [1.3, 2.1, 0.0],  # P0: a -> b, P1: b -> d
            [1.0, 1.7, 0.0],  # P0: c -> d, P1: a -> d
            [1.3, 2.1, 0.0],  # P0: a -> 2b, P1: 2c -> d
            [1.4, 0.0, 0.0],  # P0: 3b -> 2c
            [1.3, 2.1, 0.0],  # P0: a,b -> c, P1: b,d -> 2a,c
            [1.4, 0.0, 0.0],  # P0: a,d -> b
            [2.0, 0.0, 0.0],  # P0: a + b -> b + c
            [2.0, 0.0, 0.0],  # P0: a + c -> b + c
        ],
        dtype=_FLOAT,
    )
    k_b = torch.tensor(
        [
            [0.3, 1.1, 0.0],  # P0: a -> b, P1: b -> d
            [1.5, 0.7, 0.0],  # P0: c -> d, P1: a -> d
            [0.3, 1.1, 0.0],  # P0: a -> 2b, P1: 2c -> d
            [1.5, 0.0, 0.0],  # P0: 3b -> 2c
            [0.3, 1.1, 0.0],  # P0: a,b -> c, P1: b,d -> 2a,c
            [1.5, 0.0, 0.0],  # P0: a,d -> b
            [1.0, 0.0, 0.0],  # P0: a + b -> b + c
            [1.0, 0.0, 0.0],  # P0: a + c -> b + c
        ],
        dtype=_FLOAT,
    )

    def f(s, p, kf, kb):
        vf = (s / kf - p / kb) / (1 + s / kf + p / kb)
        vb = (p / kb - s / kf) / (1 + s / kf + p / kb)
        return (vf - vb) / 2

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

    def f111(s1, s2, p, kf, kb):
        af = (s1 * s2 / kf - p / kb) / (1 + s1 * s2 / kf + p / kb)
        ab = (p / kb - s1 * s2 / kf) / (1 + s1 * s2 / kf + p / kb)
        return (af - ab) / 2

    def f1121(s1, s2, p1, p2, kf, kb):
        base = 1 + s1 * s2 / kf + p1**2 * p2 / kb
        af = (s1 * s2 / kf - p1**2 * p2 / kb) / base
        ab = (p1**2 * p2 / kb - s1 * s2 / kf) / base
        return (af - ab) / 2

    def f1111(s1, s2, p1, p2, kf, kb):
        af = (s1 * s2 / kf - p1 * p2 / kb) / (1 + s1 * s2 / kf + p1 * p2 / kb)
        ab = (p1 * p2 / kb - s1 * s2 / kf) / (1 + s1 * s2 / kf + p1 * p2 / kb)
        return (af - ab) / 2

    # cell 0: P0: a -> b, P1: b -> d
    a_c0_0 = f(x0[0, 0], x0[0, 1], k_f[0, 0], k_b[0, 0])
    a_c0_1 = f(x0[0, 1], x0[0, 3], k_f[0, 1], k_b[0, 1])

    # cell 1: P0: c -> d, P1: a -> d
    a_c1_0 = f(x0[1, 2], x0[1, 3], k_f[1, 0], k_b[1, 0])
    a_c1_1 = f(x0[1, 0], x0[1, 3], k_f[1, 1], k_b[1, 1])

    # cell 2: P0: a -> 2b, P1: 2c -> d
    a_c2_0 = f12(x0[2, 0], x0[2, 1], k_f[2, 0], k_b[2, 0])
    a_c2_1 = f21(x0[2, 2], x0[2, 3], k_f[2, 1], k_b[2, 1])

    # cell 3: P0: 3b -> 2c
    a_c3_0 = f32(x0[3, 1], x0[3, 2], k_f[3, 0], k_b[3, 0])

    # cell 4: P0: a,b -> c, P1: b,d -> 2a,c
    a_c4_0 = f111(x0[4, 0], x0[4, 1], x0[4, 2], k_f[4, 0], k_b[4, 0])
    a_c4_1 = f1121(x0[4, 1], x0[4, 3], x0[4, 0], x0[4, 2], k_f[4, 1], k_b[4, 1])

    # cell 5: P0: a,d -> b
    a_c5_0 = f111(x0[5, 0], x0[5, 3], x0[5, 1], k_f[5, 0], k_b[5, 0])

    # cell 6: P0: a + b -> b + c
    a_c6_0 = f1111(x0[6, 0], x0[6, 1], x0[6, 1], x0[6, 2], k_f[6, 0], k_b[6, 0])

    # cell 7: P0: a + c -> b + c
    a_c7_0 = f1111(x0[7, 0], x0[7, 2], x0[7, 1], x0[7, 2], k_f[7, 0], k_b[7, 0])

    # expected outcome
    a_exp = torch.tensor(
        [
            [a_c0_0, a_c0_1, 0.0],
            [a_c1_0, a_c1_1, 0.0],
            [a_c2_0, a_c2_1, 0.0],
            [a_c3_0, 0.0, 0.0],
            [a_c4_0, a_c4_1, 0.0],
            [a_c5_0, 0.0, 0.0],
            [a_c6_0, 0.0, 0.0],
            [a_c7_0, 0.0, 0.0],
        ],
        dtype=_FLOAT,
    )

    # test
    a = _kinetics._get_alpha_cat(x=x0_, N_f=N_f, N_b=N_b, k_f=k_f, k_b=k_b)
    torch.testing.assert_close(a, a_exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
def test_get_alpha_cat_randomly():
    for _ in range(100):
        c = random.randint(2, 1000)
        p = random.randint(1, 100)
        m = random.randint(2, 100)
        x = torch.rand(c, p, m, dtype=_FLOAT) * 1000
        N_f = torch.randint(-100, 100, (c, p, m), dtype=_INT)
        N_b = torch.randint(-100, 100, (c, p, m), dtype=_INT)
        k_f = torch.rand(c, p, dtype=_FLOAT) * 10
        k_b = torch.rand(c, p, dtype=_FLOAT) * 10
        a = _kinetics._get_alpha_cat(x=x, N_f=N_f, N_b=N_b, k_f=k_f, k_b=k_b)
        assert a.shape == (c, p)
        assert a.isnan().sum() == 0
        assert a.isfinite().all()


def test_get_alpha_reg():
    # n cells, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: inh=c, P1: act=a, P2: inh=c, act=d
    # cell 1: P0: inh=c,d, P1: act=a,b
    # cell 2: P0: act=10a, inh=10b, P1: act=c (but no c), P2: inh=d (but no d)

    # concentrations
    x0 = torch.tensor(
        [
            [2.1, 0.0, 1.9, 2.0],
            [3.2, 1.6, 4.0, 1.9],
            [2.6, 1.2, 0.0, 0.0],
        ],
        dtype=_FLOAT,
    )
    x0_ = torch.broadcast_to(x0.unsqueeze(1), (len(x0), 3, 4))
    N_h = torch.tensor(
        [
            [
                [0, 0, -1, 0],  # P0: inh=c
                [1, 0, 0, 0],  # P1: act=a
                [0, 0, -1, 1],  # P2: inh=c, act=d
            ],
            [
                [0, 0, -1, -1],  # P0: inh=c,d
                [1, 1, 0, 0],  # P1: act=a,b
                [0, 0, 0, 0],
            ],
            [
                [10, -10, 0, 0],  # P0: act=10a, inh=10b
                [0, 0, 1, 0],  # P1: act=c (but no c)
                [0, 0, 0, -1],  # P2: inh=d (but no d)
            ],
        ],
        dtype=_INT,
    )
    K_r = torch.tensor(
        [
            [
                [0.0, 0.0, 1.3, 0.0],  # P0: inh=c
                [2.1, 0.0, 0.0, 0.0],  # P1: act=a
                [0.0, 0.0, 0.9, 0.9],  # P2: inh=c, act=d
            ],
            [
                [0.0, 0.0, 1.4, 1.4],  # P0: inh=c,d
                [2.2, 2.2, 0.0, 0.0],  # P1: act=a,b
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 2.0, 0.0, 0.0],  # P0: act=10a, inh=10b
                [0.0, 0.0, 3.0, 0.0],  # P1: act=c (but no c)
                [0.0, 0.0, 0.0, 4.0],  # P2: inh=d (but no d)
            ],
        ],
        dtype=_FLOAT,
    )

    def f(x, k, n):
        return x**n / (k**n + x**n)

    # cell 0: P0: inh=c, P1: act=a, P2: inh=c, act=d
    a_c0_0 = f(x0[0, 2], K_r[0, 0, 2], N_h[0, 0, 2])
    a_c0_1 = f(x0[0, 0], K_r[0, 1, 0], N_h[0, 1, 0])
    a_c0_2 = f(x0[0, 2], K_r[0, 2, 2], N_h[0, 2, 2]) * f(
        x0[0, 3], K_r[0, 2, 3], N_h[0, 2, 3]
    )

    # cell 1: P0: inh=c,d, P1: act=a,b
    a_c1_0 = f(x0[1, 2], K_r[1, 0, 2], N_h[1, 0, 2]) * f(
        x0[1, 3], K_r[1, 0, 3], N_h[1, 0, 3]
    )
    a_c1_1 = f(x0[1, 0], K_r[1, 1, 0], N_h[1, 1, 0]) * f(
        x0[1, 1], K_r[1, 1, 1], N_h[1, 1, 1]
    )

    # cell 2: P0: act=10a, inh=10b, P1: act=c (but no c), P2: inh=d (but no d)
    a_c2_0 = f(x0[2, 0], K_r[2, 0, 0], N_h[2, 0, 0]) * f(
        x0[2, 1], K_r[2, 0, 1], N_h[2, 0, 1]
    )

    # expected outcome
    a_exp = torch.tensor(
        [
            [a_c0_0, a_c0_1, a_c0_2],
            [a_c1_0, a_c1_1, 1.0],
            [a_c2_0, 0.0, 1.0],
        ],
        dtype=_FLOAT,
    )

    # test
    a = _kinetics._get_alpha_reg(x=x0_, N_h=N_h, K_r=K_r)
    torch.testing.assert_close(a, a_exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
def test_get_alpha_reg_randomly():
    for _ in range(100):
        c = random.randint(2, 1000)
        p = random.randint(1, 100)
        m = random.randint(2, 100)
        x = torch.rand(c, p, m, dtype=_FLOAT) * 1000
        N_h = torch.randint(-100, 100, (c, p, m), dtype=_INT)
        K_r = torch.rand(c, p, m, dtype=_FLOAT) * 10
        a = _kinetics._get_alpha_reg(x=x, N_h=N_h, K_r=K_r)
        assert a.shape == (c, p)
        assert a.isnan().sum() == 0
        assert a.isfinite().all()
        assert (a >= 0).all()


def test_step_protein_activity():
    # c cells, p max proteins, m molecules (a, b, c, d)
    c = 4
    p = 3
    m = 4

    # molecular masses (a, b, c, d)
    c_mass = torch.tensor([9.0, 6.0, 3.0, 3.0])

    x0 = torch.full((c, m), 10.0, dtype=_FLOAT)
    v_max = torch.full((c, p), 10.0, dtype=_FLOAT)
    k_f = torch.zeros((c, p), dtype=_FLOAT)
    k_b = torch.zeros((c, p), dtype=_FLOAT)
    K_r = torch.zeros(c, p, m, dtype=_FLOAT)
    N_f = torch.zeros(c, p, m, dtype=_INT)
    N_b = torch.zeros(c, p, m, dtype=_INT)
    N_h = torch.zeros(c, p, m, dtype=_INT)

    # cell 0: P0: a -> 3c, P1: b -> 2d (uncoupled)
    N_f[0, 0, 0] = 1
    N_b[0, 0, 2] = 3
    k_f[0, 0] = 5.0
    k_b[0, 0] = 5.0

    N_f[0, 1, 1] = 1
    N_b[0, 1, 3] = 2
    k_f[0, 1] = 5.0
    k_b[0, 1] = 5.0

    # cell 1: P0: a -> b + c, P1: b -> c + d, P2: c -> d (coupled)
    N_f[1, 0, 0] = 1
    N_b[1, 0, 1] = 1
    N_b[1, 0, 2] = 1
    k_f[1, 0] = 5.0
    k_b[1, 0] = 5.0

    N_f[1, 1, 1] = 1
    N_b[1, 1, 2] = 1
    N_b[1, 1, 3] = 1
    k_f[1, 1] = 5.0
    k_b[1, 1] = 5.0

    N_f[1, 2, 2] = 1
    N_b[1, 2, 3] = 1
    k_f[1, 2] = 1.0
    k_b[1, 2] = 9.0

    # cell 3: P0: c -> d (at equilibrium)
    N_f[2, 0, 2] = 1
    N_b[2, 0, 3] = 1
    k_f[2, 0] = 5.0
    k_b[2, 0] = 5.0

    # cell 4: P0: a -> 3c | b-inh, P1: b -> 2d | a-act (like cell 0 but regulated)
    N_f[3, 0, 0] = 1
    N_b[3, 0, 2] = 3
    N_h[3, 0, 1] = -1
    k_f[3, 0] = 5.0
    k_b[3, 0] = 5.0
    K_r[3, 0, 1] = 5.0

    N_f[3, 1, 1] = 1
    N_b[3, 1, 3] = 2
    N_h[3, 1, 0] = 1
    k_f[3, 1] = 5.0
    k_b[3, 1] = 5.0
    K_r[3, 1, 0] = 5.0

    # test
    x1 = _kinetics.step_protein_activity(
        h=1.0,
        x0=x0,
        N_f=N_f,
        N_b=N_b,
        N_h=N_h,
        k_f=k_f,
        k_b=k_b,
        K_r=K_r,
        v_max=v_max,
    )

    # useful results
    assert x1.shape == x0.shape
    assert x1.isnan().sum() == 0
    assert x1.isfinite().all()
    assert (x1 >= 0).all()

    # conservation of mass
    xd = x1 - x0
    assert (xd @ c_mass).abs().max() < 1e-4

    # cell 0: moves towards K_e
    assert xd[0, 0] > 0
    assert xd[0, 1] > 0
    assert xd[0, 2] < 0
    assert xd[0, 3] < 0

    # cell 1: moves towards K_e
    assert xd[1, 0] > 0
    assert xd[1, 1] > 0
    assert xd[1, 2] < 0
    assert xd[1, 3] < 0

    # cell 3: already at K_e
    assert xd[2, 0] == 0
    assert xd[2, 1] == 0
    assert xd[2, 2].abs() < 0.01
    assert xd[2, 3].abs() < 0.01

    # cell 4: like cell 0 but regulated (so should be slower)
    assert xd[3, 0] > 0
    assert xd[3, 1] > 0
    assert xd[3, 2] < 0
    assert xd[3, 3] < 0
    assert xd[3, 0].abs() < xd[0, 0].abs()
    assert xd[3, 1].abs() < xd[0, 1].abs()
    assert xd[3, 2].abs() < xd[0, 2].abs()
    assert xd[3, 3].abs() < xd[0, 3].abs()


def test_step_protein_activity_convergence():
    c = 3
    p = 1
    m = 4

    # molecular masses (a, b, c, d)
    c_mass = torch.tensor([9.0, 6.0, 3.0, 3.0])

    x0 = torch.full((c, m), 10.0, dtype=_FLOAT)
    v_max = torch.full((c, p), 10.0, dtype=_FLOAT)
    k_f = torch.zeros((c, p), dtype=_FLOAT)
    k_b = torch.zeros((c, p), dtype=_FLOAT)
    K_r = torch.zeros(c, p, m, dtype=_FLOAT)
    N_f = torch.zeros(c, p, m, dtype=_INT)
    N_b = torch.zeros(c, p, m, dtype=_INT)
    N_h = torch.zeros(c, p, m, dtype=_INT)

    # cell 0: P0: c -> d
    N_f[0, 0, 2] = 1
    N_b[0, 0, 3] = 1
    k_f[0, 0] = 2.0
    k_b[0, 0] = 9.0
    v_max[0, 0] = 10.0

    # cell 1: P0: a -> 3c
    N_f[1, 0, 0] = 1
    N_b[1, 0, 2] = 3
    k_f[1, 0] = 1.0
    k_b[1, 0] = 50.0
    v_max[1, 0] = 10.0

    # cell 2: P0: P0: b + c -> a
    N_f[2, 0, 1] = 1
    N_f[2, 0, 2] = 1
    N_b[2, 0, 0] = 1
    k_f[2, 0] = 1.0
    k_b[2, 0] = 100.0
    v_max[2, 0] = 100.0

    # should converge within 5s
    for _ in range(6):
        x1 = _kinetics.step_protein_activity(
            h=1.0,
            x0=x0,
            N_f=N_f,
            N_b=N_b,
            N_h=N_h,
            k_f=k_f,
            k_b=k_b,
            K_r=K_r,
            v_max=v_max,
        )

        # useful results
        assert x1.shape == x0.shape
        assert x1.isnan().sum() == 0
        assert x1.isfinite().all()
        assert (x1 >= 0).all()

        # conservation of mass
        xd = x1 - x0
        assert (xd @ c_mass).abs().max() < 1e-4

        x0 = x1

    # should have converged by now
    N = (N_b - N_f).float()
    q = torch.prod(x0.unsqueeze(1) ** N, dim=2)
    torch.testing.assert_close(k_b / k_f, q, atol=0.1, rtol=0.1)


@pytest.mark.slow
def test_step_protein_activity_randomly(caplog):
    for _ in range(10):
        c = random.randint(2, 1000)
        p = random.randint(1, 100)
        m = random.randint(2, 100)
        h = random.uniform(0, 10)
        x = torch.rand(c, m, dtype=_FLOAT) * 1000
        N_f = torch.randint(-100, 100, (c, p, m), dtype=_INT)
        N_b = torch.randint(-100, 100, (c, p, m), dtype=_INT)
        N_h = torch.randint(-100, 100, (c, p, m), dtype=_INT)
        v_max = torch.rand(c, p, dtype=_FLOAT) * 100
        k_f = torch.rand(c, p, dtype=_FLOAT) * 10
        k_b = torch.rand(c, p, dtype=_FLOAT) * 10
        K_r = torch.rand(c, p, m, dtype=_FLOAT) * 10

        for _ in range(10):
            with caplog.at_level(logging.WARNING):
                x = _kinetics.step_protein_activity(
                    h=h,
                    x0=x,
                    N_f=N_f,
                    N_b=N_b,
                    N_h=N_h,
                    k_f=k_f,
                    k_b=k_b,
                    K_r=K_r,
                    v_max=v_max,
                )
            assert x.shape == (c, m)
            assert x.isnan().sum() == 0
            assert x.isfinite().all()
            assert (x >= 0).all()
            assert len(caplog.records) == 0
