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
def test_get_bounds_randomly():
    for _ in range(100):
        c = random.randint(2, 1000)
        p = random.randint(1, 100)
        m = random.randint(2, 100)
        N = torch.randint(-100, 100, (c, p, m), dtype=torch.float32)
        N[0, 0, :] = 0.0
        B = torch.rand(c, p, m, dtype=torch.float32) * 1000
        lo, hi = _KINETICS._get_bounds(B=B, N=N)
        assert lo.shape == (c, p)
        assert hi.shape == (c, p)
        assert lo[0, 0] == -float("inf")
        assert hi[0, 0] == float("inf")
        assert lo.isnan().sum() == 0
        assert hi.isnan().sum() == 0
        assert (lo <= hi).all()


def test_get_alpha_cat():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
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

    # concentrations
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
        dtype=torch.float32,
    )
    x0_ = torch.broadcast_to(x0.unsqueeze(1), (len(x0), 3, 4))

    # stoichiometry
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
        dtype=torch.float32,
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
        dtype=torch.float32,
    )

    # affinities
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
        dtype=torch.float32,
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
        dtype=torch.float32,
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
        dtype=torch.float32,
    )

    # test
    a = _KINETICS._get_alpha_cat(x=x0_, N_f=N_f, N_b=N_b, k_f=k_f, k_b=k_b)
    torch.testing.assert_close(a, a_exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
def test_get_alpha_cat_randomly():
    for _ in range(100):
        c = random.randint(2, 1000)
        p = random.randint(1, 100)
        m = random.randint(2, 100)
        x = torch.rand(c, p, m, dtype=torch.float32) * 1000
        N_f = torch.randint(-100, 100, (c, p, m), dtype=torch.float32)
        N_b = torch.randint(-100, 100, (c, p, m), dtype=torch.float32)
        k_f = torch.rand(c, p, dtype=torch.float32) * 10  # TODO: better distro
        k_b = torch.rand(c, p, dtype=torch.float32) * 10  # TODO: better distro
        a = _KINETICS._get_alpha_cat(x=x, N_f=N_f, N_b=N_b, k_f=k_f, k_b=k_b)
        assert a.shape == (c, p)
        assert a.isnan().sum() == 0
        assert a.isfinite().all()


def test_get_alpha_reg():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
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
        dtype=torch.float32,
    )
    x0_ = torch.broadcast_to(x0.unsqueeze(1), (len(x0), 3, 4))

    # hill coefficients
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
        dtype=torch.int32,
    )

    # affinities
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
        dtype=torch.float32,
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
        dtype=torch.float32,
    )

    # test
    a = _KINETICS._get_alpha_reg(x=x0_, N_h=N_h, K_r=K_r)
    torch.testing.assert_close(a, a_exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
def test_get_alpha_reg_randomly():
    for _ in range(100):
        c = random.randint(2, 1000)
        p = random.randint(1, 100)
        m = random.randint(2, 100)
        x = torch.rand(c, p, m, dtype=torch.float32) * 1000
        N_h = torch.randint(-100, 100, (c, p, m), dtype=torch.float32)
        K_r = torch.rand(c, p, m, dtype=torch.float32) * 10  # TODO: better distro
        a = _KINETICS._get_alpha_reg(x=x, N_h=N_h, K_r=K_r)
        assert a.shape == (c, p)
        assert a.isnan().sum() == 0
        assert a.isfinite().all()
        assert (a >= 0).all()


def test_step_protein_activity():
    # c cells, p max proteins, m molecules (a, b, c, d)
    # coupled and uncoupled reactions
    # cell 0: P0: a -> b, P1: c -> d
    # cell 1: P0: a -> 2b, P1: b -> c, P2: 2a -> d
    # overshooting K_e and negative concentrations
    c = 2
    p = 3
    m = 4

    # concentrations
    x0 = torch.full((c, m), 10.0, dtype=torch.float32)
    v_max = torch.full((c, p), 10.0, dtype=torch.float32)

    N_f = torch.tensor(
        [
            [
                [1, 0, 0, 0],  # P0: a -> b
                [0, 0, 1, 0],  # P1: 2 -> d
                [0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 0],  # P0: a -> 2b
                [0, 2, 0, 0],  # P1: b -> c
                [2, 0, 0, 0],  # P2: 2a -> d
            ],
        ],
        dtype=torch.int32,
    )
    N_b = torch.tensor(
        [
            [
                [0, 1, 0, 0],  # P0: a -> b
                [0, 0, 0, 1],  # P1: 2 -> d
                [0, 0, 0, 0],
            ],
            [
                [0, 2, 0, 0],  # P0: a -> 2b
                [0, 0, 1, 0],  # P1: b -> c
                [0, 0, 0, 1],  # P2: 2a -> d
            ],
        ],
        dtype=torch.int32,
    )
    k_f = torch.tensor(
        [
            [5.0, 5.0, 0.0],  # P0: a -> b, P1: c -> d
            [5.0, 5.0, 5.0],  # P0: a -> 2b, P1: b -> c, P2: 2a -> d
        ],
        dtype=torch.float32,
    )
    k_b = torch.tensor(
        [
            [9.0, 9.0, 0.0],  # P0: a -> b, P1: c -> d
            [9.0, 9.0, 9.0],  # P0: a -> 2b, P1: b -> c, P2: 2a -> d
        ],
        dtype=torch.float32,
    )

    # regulation
    K_r = torch.zeros(2, 3, 4, dtype=torch.float32)
    N_h = torch.zeros(2, 3, 4, dtype=torch.int32)

    # expected outcome

    # test
    x1 = _KINETICS.step_protein_activity(
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

    assert x1.shape == x0.shape
    assert x1.isnan().sum() == 0
    assert x1.isfinite().all()
    assert (x1 >= 0).all()

    # TODO: check balance of decoupled reactions
    # TODO: check direction of all reactions

    # TODO: use c_mass (m,) to check conservation of mass
    #       it should be c_mass * dx = 0 for each cell


@pytest.mark.slow
def test_step_protein_activity_randomly():
    for _ in range(100):
        c = random.randint(2, 1000)
        p = random.randint(1, 100)
        m = random.randint(2, 100)
        h = random.uniform(0, 10)
        x = torch.rand(c, m, dtype=torch.float32) * 1000
        N_f = torch.randint(-100, 100, (c, p, m), dtype=torch.int32)
        N_b = torch.randint(-100, 100, (c, p, m), dtype=torch.int32)
        N_h = torch.randint(-100, 100, (c, p, m), dtype=torch.int32)
        v_max = torch.rand(c, p, dtype=torch.float32) * 100
        k_f = torch.rand(c, p, dtype=torch.float32) * 10  # TODO: better distro
        k_b = torch.rand(c, p, dtype=torch.float32) * 10  # TODO: better distro
        K_r = torch.rand(c, p, m, dtype=torch.float32) * 10  # TODO: better distro
        for _ in range(10):
            x = _KINETICS.step_protein_activity(
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
