import logging
import random

import pytest
import torch
from magicsoup.kinetics import Kinetics

from tests.config import DEVICE
from tests.util import FTYPE, ffull, frand, ftensor, fzeros, irand, itensor, izeros

_ATOL = 1e-4
_RTOL = 1e-4

_kinetics = Kinetics(dt=1.0, device=DEVICE)


def test_get_bounds():
    # 2 cells, 3 proteins, 4 molecules (a, b, c, d)
    c = 2
    p = 3
    m = 4
    dt = 0.5
    v_max = fzeros(c, p)
    N_e = fzeros(c, p, m)
    B = fzeros(c, p, m)

    # cell 0: P0: 2a -> 5b, P1: 10b -> d
    v_max[0, :] = ftensor([10, 10, 10])
    N_e[0, 0, :] = ftensor([-2, 5, 0, 0])
    N_e[0, 1, :] = ftensor([0, -10, 0, 1])
    B[0, 0, :] = ftensor([10, 10, 10, 10])
    B[0, 1, :] = ftensor([0, 1, 0, 1])
    B[0, 2, :] = ftensor([10, 10, 10, 10])

    # cell 1: P0: 3c -> 7d, P1: a -> d
    v_max[1, :] = ftensor([5, 5, 5])
    N_e[1, 0, :] = ftensor([0, 0, -3, 7])
    N_e[1, 1, :] = ftensor([-1, 0, 0, 1])
    B[1, 0, :] = ftensor([3, 7, 5, 1])
    B[1, 1, :] = ftensor([0, 0, 0, 1])
    B[1, 2, :] = ftensor([10, 10, 10, 10])

    # expected boundaries
    lo_exp = ftensor(
        [
            [-2, -1, -v_max[0, 2] * dt],
            [-1 / 7, -1, -v_max[1, 2] * dt],
        ]
    )
    hi_exp = ftensor(
        [
            [5, 1 / 10, v_max[0, 2] * dt],
            [5 / 3, 0, v_max[1, 2] * dt],
        ]
    )

    lo, hi = _kinetics._get_bounds(B=B, N_e=N_e, v_max=v_max, dt=dt)

    torch.testing.assert_close(lo, lo_exp, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(hi, hi_exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
def test_get_bounds_randomly():
    for _ in range(100):
        c = random.randint(2, 1000)
        p = random.randint(1, 100)
        m = random.randint(2, 100)
        dt = random.uniform(0, 10)
        v_max = frand(c, p) * 1e4
        B = frand(c, p, m) * 1000
        N_e = irand(-100, 100, c, p, m)
        N_e[0, 0, :] = 0.0
        lo, hi = _kinetics._get_bounds(B=B, N_e=N_e, v_max=v_max, dt=dt)
        assert lo.shape == (c, p)
        assert hi.shape == (c, p)
        assert lo[0, 0] == -v_max[0, 0] * dt
        assert hi[0, 0] == v_max[0, 0] * dt
        assert lo.isnan().sum() == 0
        assert hi.isnan().sum() == 0
        assert (lo <= hi).all()


def test_get_alpha_cat():
    # 10 cells, 3 max proteins, 4 molecules (a, b, c, d)
    c = 10
    p = 3
    m = 4
    x0 = fzeros(c, m)
    N_f = izeros(c, p, m)
    N_b = izeros(c, p, m)
    k_f = fzeros(c, p)
    k_b = fzeros(c, p)

    # simple MM kinetics

    # cell 0: P0: a -> b, P1: b -> d
    x0[0, :] = ftensor([2.1, 1.9, 0.0, 0.8])

    N_f[0, 0, :] = itensor([1, 0, 0, 0])
    N_b[0, 0, :] = itensor([0, 1, 0, 0])
    k_f[0, 0] = 1.3
    k_b[0, 0] = 0.3

    N_f[0, 1, :] = itensor([0, 1, 0, 0])
    N_b[0, 1, :] = itensor([0, 0, 0, 1])
    k_f[0, 1] = 2.1
    k_b[0, 1] = 1.1

    # cell 1: P0: c -> d, P1: a -> d
    x0[1, :] = ftensor([2.9, 3.1, 2.1, 1.0])

    N_f[1, 0, :] = itensor([0, 0, 1, 0])
    N_b[1, 0, :] = itensor([0, 0, 0, 1])
    k_f[1, 0] = 1.0
    k_b[1, 0] = 1.5

    N_f[1, 1, :] = itensor([1, 0, 0, 0])
    N_b[1, 1, :] = itensor([0, 0, 0, 1])
    k_f[1, 1] = 1.7
    k_b[1, 1] = 0.7

    # multiple substrates and products

    # cell 2: P0: a -> 2b, P1: 2c -> d
    x0[2, :] = ftensor([1.1, 0.1, 2.9, 0.8])

    N_f[2, 0, :] = itensor([1, 0, 0, 0])
    N_b[2, 0, :] = itensor([0, 2, 0, 0])
    k_f[2, 0] = 1.3
    k_b[2, 0] = 0.3

    N_f[2, 1, :] = itensor([0, 0, 2, 0])
    N_b[2, 1, :] = itensor([0, 0, 0, 1])
    k_f[2, 1] = 2.1
    k_b[2, 1] = 1.1

    # cell 3: P0: 3b -> 2c
    x0[3, :] = ftensor([1.2, 4.9, 5.1, 1.4])

    N_f[3, 0, :] = itensor([0, 3, 0, 0])
    N_b[3, 0, :] = itensor([0, 0, 2, 0])
    k_f[3, 0] = 1.4
    k_b[3, 0] = 1.5

    # cell 4: P0: a,b -> c, P1: b,d -> 2a,c
    x0[4, :] = ftensor([1.1, 2.1, 2.9, 0.8])

    N_f[4, 0, :] = itensor([1, 1, 0, 0])
    N_b[4, 0, :] = itensor([0, 0, 1, 0])
    k_f[4, 0] = 1.3
    k_b[4, 0] = 0.3

    N_f[4, 1, :] = itensor([0, 1, 0, 1])
    N_b[4, 1, :] = itensor([2, 0, 1, 0])
    k_f[4, 1] = 2.1
    k_b[4, 1] = 1.1

    # cell 5: P0: a,d -> b
    x0[5, :] = ftensor([2.3, 0.4, 0.0, 3.2])

    N_f[5, 0, :] = itensor([1, 0, 0, 1])
    N_b[5, 0, :] = itensor([0, 1, 0, 0])
    k_f[5, 0] = 1.4
    k_b[5, 0] = 1.5

    # co-factors involved (required but n_f+n_b=0)

    # cell 6: P0: a + b -> b + c
    x0[6, :] = ftensor([10.0, 0.1, 3.0, 0.8])

    N_f[6, 0, :] = itensor([1, 1, 0, 0])
    N_b[6, 0, :] = itensor([0, 1, 1, 0])
    k_f[6, 0] = 2.0
    k_b[6, 0] = 1.0

    # cell 7: P0: a + c -> b + c
    x0[7, :] = ftensor([10.0, 3.0, 0.1, 0.0])

    N_f[7, 0, :] = itensor([1, 0, 1, 0])
    N_b[7, 0, :] = itensor([0, 1, 1, 0])
    k_f[7, 0] = 2.0
    k_b[7, 0] = 1.0

    # expected outcome
    a_f = 1 / k_f * (x0.unsqueeze(1) ** N_f).prod(dim=-1)
    a_b = 1 / k_b * (x0.unsqueeze(1) ** N_b).prod(dim=-1)
    a_exp = ((a_f - a_b) / (1 + a_f + a_b)).nan_to_num(0.0)

    # test
    x0_ = torch.broadcast_to(x0.unsqueeze(1), (c, p, m))
    a = _kinetics._get_alpha_cat(x=x0_, N_f=N_f, N_b=N_b, k_f=k_f, k_b=k_b)
    torch.testing.assert_close(a, a_exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
def test_get_alpha_cat_randomly():
    for _ in range(100):
        c = random.randint(2, 1000)
        p = random.randint(1, 100)
        m = random.randint(2, 100)
        x = frand(c, p, m) * 1000
        N_f = irand(-100, 100, c, p, m)
        N_b = irand(-100, 100, c, p, m)
        k_f = frand(c, p) * 10
        k_b = frand(c, p) * 10
        a = _kinetics._get_alpha_cat(x=x, N_f=N_f, N_b=N_b, k_f=k_f, k_b=k_b)
        assert a.shape == (c, p)
        assert a.isnan().sum() == 0
        assert a.isfinite().all()


def test_get_alpha_reg():
    # 6 cells, 3 max proteins, 4 molecules (a, b, c, d)
    c = 6
    p = 3
    m = 4

    x0 = fzeros(c, m)
    K_r = fzeros(c, p, m)
    N_h = izeros(c, p, m)

    # cell 0: P0: inh=c, P1: act=a, P2: inh=c, act=d
    x0[0, :] = ftensor([2.1, 0.0, 1.9, 2.0])

    N_h[0, 0, :] = itensor([0, 0, -1, 0])
    K_r[0, 0, :] = ftensor([0.0, 0.0, 1.3, 0.0])

    N_h[0, 1, :] = itensor([1, 0, 0, 0])
    K_r[0, 1, :] = ftensor([2.1, 0.0, 0.0, 0.0])

    N_h[0, 2, :] = itensor([0, 0, -1, 1])
    K_r[0, 2, :] = ftensor([0.0, 0.0, 0.9, 0.9])

    # cell 1: P0: inh=c,d, P1: act=a,b
    x0[1, :] = ftensor([3.2, 1.6, 4.0, 1.9])

    N_h[1, 0, :] = itensor([0, 0, -1, -1])
    K_r[1, 0, :] = ftensor([0.0, 0.0, 1.4, 1.4])

    N_h[1, 1, :] = itensor([1, 1, 0, 0])
    K_r[1, 1, :] = ftensor([2.2, 2.2, 0.0, 0.0])

    # cell 2: P0: act=10a, inh=10b, P1: act=c (but no c), P2: inh=d (but no d)
    x0[2, :] = ftensor([2.6, 1.2, 0.0, 0.0])

    N_h[2, 0, :] = itensor([10, -10, 0, 0])
    K_r[2, 0, :] = ftensor([1.0, 2.0, 0.0, 0.0])

    N_h[2, 1, :] = itensor([0, 0, 1, 0])
    K_r[2, 1, :] = ftensor([0.0, 0.0, 3.0, 0.0])

    N_h[2, 2, :] = itensor([0, 0, 0, -1])
    K_r[2, 2, :] = ftensor([0.0, 0.0, 0.0, 4.0])

    # expected
    a_r = x0.unsqueeze(1) ** N_h
    k = K_r**N_h
    a_exp = torch.where(N_h != 0, (a_r / (k + a_r)), 1.0).prod(dim=-1).nan_to_num(1.0)

    # test
    x0_ = torch.broadcast_to(x0.unsqueeze(1), (c, p, m))
    a = _kinetics._get_alpha_reg(x=x0_, N_h=N_h, K_r=K_r)
    torch.testing.assert_close(a, a_exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
def test_get_alpha_reg_randomly():
    for _ in range(100):
        c = random.randint(2, 1000)
        p = random.randint(1, 100)
        m = random.randint(2, 100)
        x = frand(c, p, m) * 1000
        N_h = irand(-100, 100, c, p, m)
        K_r = frand(c, p, m) * 10
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
    c_mass = ftensor([9.0, 6.0, 3.0, 3.0])

    x0 = ffull(c, m, value=10.0)
    v_max = ffull(c, p, value=10.0)
    k_f = fzeros(c, p)
    k_b = fzeros(c, p)
    K_r = fzeros(c, p, m)
    N_f = izeros(c, p, m)
    N_b = izeros(c, p, m)
    N_h = izeros(c, p, m)

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
    x1 = _kinetics.get_x1(
        x0=x0,
        N_e=(N_b - N_f).to(FTYPE),
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
    dx = x1 - x0
    assert (dx @ c_mass).abs().max() < 1e-4

    # cell 0: moves towards K_e
    assert dx[0, 0] > 0
    assert dx[0, 1] > 0
    assert dx[0, 2] < 0
    assert dx[0, 3] < 0

    # cell 1: moves towards K_e
    assert dx[1, 0] > 0
    assert dx[1, 1] > 0
    assert dx[1, 2] < 0
    assert dx[1, 3] < 0

    # cell 3: already at K_e
    assert dx[2, 0] == 0
    assert dx[2, 1] == 0
    assert dx[2, 2].abs() < 0.01
    assert dx[2, 3].abs() < 0.01

    # cell 4: like cell 0 but regulated (so should be slower)
    assert dx[3, 0] > 0
    assert dx[3, 1] > 0
    assert dx[3, 2] < 0
    assert dx[3, 3] < 0
    assert dx[3, 0].abs() < dx[0, 0].abs()
    assert dx[3, 1].abs() < dx[0, 1].abs()
    assert dx[3, 2].abs() < dx[0, 2].abs()
    assert dx[3, 3].abs() < dx[0, 3].abs()


def test_step_protein_activity_different_volumes():
    # c cells, p max proteins, m molecules (a, b, c, d)
    c = 1
    p = 3
    m = 4

    # molecular masses (a_int, b_int, a_ext, b_ext)
    c_mass = ftensor([3.0, 6.0, 3.0, 6.0])

    # volumes and concentration ratios
    v_cell = 1
    v_med = 10
    v_mols = ftensor([v_cell, v_cell, v_med, v_med])

    x0 = ffull(c, m, value=10.0)
    v_max = ffull(c, p, value=10.0)
    k_f = fzeros(c, p)
    k_b = fzeros(c, p)
    K_r = fzeros(c, p, m)
    N_f = izeros(c, p, m)
    N_b = izeros(c, p, m)
    N_h = izeros(c, p, m)

    # cell 0: P0: a_int -> a_ext, P1: b_ext -> b_int
    N_f[0, 0, 0] = 1
    N_b[0, 0, 2] = 1
    k_f[0, 0] = 1.0
    k_b[0, 0] = 10.0

    N_f[0, 1, 1] = 1
    N_b[0, 1, 3] = 1
    k_f[0, 1] = 10.0
    k_b[0, 1] = 1.0

    # adj N_e for effective concentration change
    N_eff = (N_b - N_f) / v_mols

    # test
    x1 = _kinetics.get_x1(
        x0=x0,
        N_e=N_eff,
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
    dx = x1 - x0
    assert ((dx * v_mols) @ c_mass).abs().max() < 1e-4

    # cell 0: moves towards K_e
    assert dx[0, 0] < 0
    assert dx[0, 1] > 0
    assert dx[0, 2] > 0
    assert dx[0, 3] < 0


def test_step_protein_activity_convergence():
    c = 3
    p = 1
    m = 4

    # molecular masses (a, b, c, d)
    c_mass = ftensor([9.0, 6.0, 3.0, 3.0])

    x0 = ffull(c, m, value=10.0)
    v_max = ffull(c, p, value=1e4)
    k_f = fzeros(c, p)
    k_b = fzeros(c, p)
    K_r = fzeros(c, p, m)
    N_f = izeros(c, p, m)
    N_b = izeros(c, p, m)
    N_h = izeros(c, p, m)

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
        x1 = _kinetics.get_x1(
            x0=x0,
            N_e=N_b - N_f,
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
        dx = x1 - x0
        assert (dx @ c_mass).abs().max() < 1e-4

        x0 = x1

    # should have converged by now
    N_e = (N_b - N_f).float()
    q = torch.prod(x0.unsqueeze(1) ** N_e, dim=2)
    torch.testing.assert_close(k_b / k_f, q, atol=0.1, rtol=0.1)


@pytest.mark.slow
def test_step_protein_activity_randomly(caplog):
    for i in range(10):
        c = random.randint(2, 1000)
        p = random.randint(1, 100)
        m = random.randint(2, 100)
        dt = random.uniform(0, 10)
        x = frand(c, m) * 1000
        N_f = irand(-100, 100, c, p, m)
        N_b = irand(-100, 100, c, p, m)
        N_h = irand(-100, 100, c, p, m)
        v_max = frand(c, p) * 1e3
        k_f = frand(c, p) * 10
        k_b = frand(c, p) * 10
        K_r = frand(c, p, m) * 10

        kinetics = Kinetics(dt=dt, device=DEVICE)

        for j in range(10):
            with caplog.at_level(logging.WARNING):
                x = kinetics.get_x1(
                    x0=x,
                    N_e=(N_b - N_f).to(FTYPE),
                    N_f=N_f,
                    N_b=N_b,
                    N_h=N_h,
                    k_f=k_f,
                    k_b=k_b,
                    K_r=K_r,
                    v_max=v_max,
                )
            assert x.shape == (c, m), f"repetition {i}, step {j}"
            assert x.isnan().sum() == 0, f"repetition {i}, step {j}"
            assert x.isfinite().all(), f"repetition {i}, step {j}"
            assert (x >= 0).all(), f"repetition {i}, step {j}"
            assert len(caplog.records) == 0, f"repetition {i}, step {j}"
