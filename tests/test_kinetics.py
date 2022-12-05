import pytest
import torch

TOLERANCE = 1e-4


def f(
    X: torch.Tensor, K: torch.Tensor, V: torch.Tensor, Z: torch.Tensor
) -> torch.Tensor:
    # TODO: masked tensors for Zs
    #       https://pytorch.org/docs/stable/masked.html
    #       also padding

    # input signals only
    Zi = torch.where(Z < 0.0, 1.0, 0.0)

    # input proportions
    Ni = torch.where(Z < 0.0, -Z, 0.0)

    # protein substrates
    Ps = torch.einsum("cps,cs->cps", Zi, X)

    # protein affinities (Km)
    Pkm = torch.einsum("cps,cp->cps", Zi, K)

    # protein maximum velocities
    Pvm = Zi.max(dim=2).values * V

    # protein velocities (c, p)
    Nom = torch.pow(Ps, Ni).prod(2)
    Den = torch.pow(Pkm + Ps, Ni).prod(2)
    Pv = Pvm * Nom / Den

    # concentration deltas
    Xd = torch.einsum("cps,cp->cs", Z, Pv)

    return Xd


def test_simple_mm_kinetic():
    # 2 cells, 3 max proteins
    # cell 0: P0: a -> b, P1: b -> d
    # cell 1: P0: c -> d, P1: a -> d

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [1.1, 0.1, 2.9, 0.8],
        [2.9, 3.1, 2.1, 1.0],
    ])

    # reactions (c, p, s)
    Z = torch.tensor([
        [   [-1.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
        [   [0.0, 0.0, -1.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])

    # proteins (c, p)
    K = torch.tensor([
        [1.2, 0.5, 1.9],
        [0.5, 1.2, 2.1],
    ])
    V = torch.tensor([
        [2.1, 1.0, 3.0],
        [1.1, 2.0, 2.2],
    ])
    # fmt: on

    def mm(x, k, v):
        return v * x / (k + x)

    # expected outcome
    dx_c0_b = mm(x=X0[0, 0], v=V[0, 0], k=K[0, 0])
    dx_c0_a = -dx_c0_b
    dx_c0_d = mm(x=X0[0, 1], v=V[0, 1], k=K[0, 1])
    dx_c0_b = dx_c0_b - dx_c0_d

    dx_c1_d_1 = mm(x=X0[1, 2], v=V[1, 0], k=K[1, 0])
    dx_c1_c = -dx_c1_d_1
    dx_c1_d_2 = mm(x=X0[1, 0], v=V[1, 1], k=K[1, 1])
    dx_c1_a = -dx_c1_d_2
    dx_c1_d = dx_c1_d_1 + dx_c1_d_2

    # test
    Xd = f(X=X0, K=K, V=V, Z=Z)

    assert Xd[0, 0] == pytest.approx(dx_c0_a, abs=TOLERANCE)
    assert Xd[0, 1] == pytest.approx(dx_c0_b, abs=TOLERANCE)
    assert Xd[0, 2] == 0.0
    assert Xd[0, 3] == pytest.approx(dx_c0_d, abs=TOLERANCE)

    assert Xd[1, 0] == pytest.approx(dx_c1_a, abs=TOLERANCE)
    assert Xd[1, 1] == 0.0
    assert Xd[1, 2] == pytest.approx(dx_c1_c, abs=TOLERANCE)
    assert Xd[1, 3] == pytest.approx(dx_c1_d, abs=TOLERANCE)


def test_mm_kinetic_with_proportions():
    # 1 cell, 3 max proteins
    # cell 0: P0: a -> 2b, P1: 2c -> d

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [1.1, 0.1, 2.9, 0.8],
    ])

    # reactions (c, p, s)
    Z = torch.tensor([
        [   [-1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, -2.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])

    # proteins (c, p)
    K = torch.tensor([
        [1.2, 0.9, 0.5],
    ])
    V = torch.tensor([
        [2.1, 1.1, 0.8],
    ])
    # fmt: on

    def mm(x, k, v, n):
        return v * x ** n / (k + x) ** n

    # expected outcome
    dx_c0_b = 2 * mm(x=X0[0, 0], v=V[0, 0], k=K[0, 0], n=1)
    dx_c0_a = -dx_c0_b / 2
    dx_c0_d = mm(x=X0[0, 2], v=V[0, 1], k=K[0, 1], n=2)
    dx_c0_c = -2 * dx_c0_d

    # test
    Xd = f(X=X0, K=K, V=V, Z=Z)

    assert Xd[0, 0] == pytest.approx(dx_c0_a, abs=TOLERANCE)
    assert Xd[0, 1] == pytest.approx(dx_c0_b, abs=TOLERANCE)
    assert Xd[0, 2] == pytest.approx(dx_c0_c, abs=TOLERANCE)
    assert Xd[0, 3] == pytest.approx(dx_c0_d, abs=TOLERANCE)


def test_mm_kinetic_with_multiple_substrates():
    # 1 cell, 3 max proteins
    # cell 0: P0: a,b -> c, P1: b,d -> a2,c

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [1.1, 2.1, 2.9, 0.8],
    ])

    # reactions (c, p, s)
    Z = torch.tensor([
        [   [-1.0, -1.0, 1.0, 0.0],
            [2.0, -1.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])

    # proteins (c, p)
    K = torch.tensor([
        [1.2, 0.9, 0.5],
    ])
    V = torch.tensor([
        [2.1, 1.1, 0.8],
    ])
    # fmt: on

    def mm(x1, x2, k, v):
        return v * x1 * x2 / ((k + x1) * (k + x2))

    # expected outcome
    v0 = mm(x1=X0[0, 0], x2=X0[0, 1], v=V[0, 0], k=K[0, 0])
    v1 = mm(x1=X0[0, 1], x2=X0[0, 3], v=V[0, 1], k=K[0, 1])
    print(f"activation v0={v0:.2f}")
    print(f"activation v1={v1:.2f}")
    dx_c0_a = 2 * v1 - v0
    dx_c0_b = -v0 - v1
    dx_c0_c = v0 + v1
    dx_c0_d = -v1

    # test
    Xd = f(X=X0, K=K, V=V, Z=Z)

    assert Xd[0, 0] == pytest.approx(dx_c0_a, abs=TOLERANCE)
    assert Xd[0, 1] == pytest.approx(dx_c0_b, abs=TOLERANCE)
    assert Xd[0, 2] == pytest.approx(dx_c0_c, abs=TOLERANCE)
    assert Xd[0, 3] == pytest.approx(dx_c0_d, abs=TOLERANCE)
