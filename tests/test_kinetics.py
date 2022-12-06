import pytest
import torch

TOLERANCE = 1e-4

# TODO: masked tensor (pytorch.org/docs/stable/masked.html)


# TODO: rethink K (c, p, s) vs (c, p)
#       what if 2 domains both a -> b?
#       same for inhibitors?
#       what if 2 i inhibitor domains?


def f(
    X: torch.Tensor, K: torch.Tensor, V: torch.Tensor, Z: torch.Tensor, I: torch.Tensor
) -> torch.Tensor:
    """
    - `X` signal concentrations (c, s)
    - `K` affinities (c, p, s)
    - `V` velocities (c, p)
    - `Z` reactions (c, p, s)
    - `I` inhibitors (c, p, s)
    """
    # inhibitors
    Mi = torch.where(I < 0.0, 1.0, 0.0)  # mask (c, p, s)
    Ki = torch.where(I < 0.0, -I, 0.0)  # affinities (c, p, s)
    Xi = torch.einsum("cps,cs->cps", Mi, X)  # concentrations (c, p, s)
    Vi = torch.nan_to_num(Xi / (Ki + Xi), 0.0).sum(dim=2).clamp(0, 1)  # activity (c, p)

    # substrates
    Ms = torch.where(Z < 0.0, 1.0, 0.0)  # mask (c, p s)
    Ns = torch.where(Z < 0.0, -Z, 0.0)  # proportions (c, p, s)
    Xs = torch.einsum("cps,cs->cps", Ms, X)  # concentrations (c, p, s)
    # Ks = torch.einsum("cps,cp->cps", Ms, K)  # affinities (c, p, s)

    # velocities
    Vmax = Ms.max(dim=2).values * V  # max velocities (c, p)
    XXs = torch.pow(Xs, Ns).prod(2)  # concentration interactions
    KKs = torch.pow(K + Xs, Ns).prod(2)  # affinity interactions
    Pv = Vmax * XXs / KKs * (1 - Vi)  # velocities (c, p)

    # concentration deltas (c, s)
    Xd = torch.einsum("cps,cp->cs", Z, Pv)

    return Xd


def test_mm_kinetic_with_inhibitor():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b, inhibitor=c
    # cell 1: P0: a -> b, inhibitor=c,d

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [1.1, 2.5, 0.9, 0.0],
        [2.3, 1.6, 3.0, 0.9],
    ])

    # reactions (c, p, s)
    Z = torch.tensor([
        [   [-1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]   ],
        [   [-1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]   ],
    ])

    # affinities (c, p, s)
    K = torch.tensor([
        [   [1.2, 3.1, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0] ],
        [   [2.1, 1.3, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0] ],
    ])

    # max velocities (c, p)
    V = torch.tensor([
        [2.1, 0.0, 0.0],
        [3.2, 0.0, 0.0],
    ])

    # inhibitors (c, p, s)
    I = torch.tensor([
        [   [0.0, 0.0, -0.7, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]   ],
        [   [0.0, 0.0, -1.0, -0.6],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]   ],
    ])
    # fmt: on

    def mmi(x, kx, v, i, ki):
        return v * x / (kx + x) * (1 - i / (ki + i))

    def mm2i(x, kx, v, i1, ki1, i2, ki2):
        return v * x / (kx + x) * max(0, 1 - i1 / (ki1 + i1) - i2 / (ki2 + i2))

    # expected outcome
    dx_c0_b = mmi(x=X0[0, 0], v=V[0, 0], kx=K[0, 0, 0], i=X0[0, 2], ki=-I[0, 0, 2])
    dx_c0_a = -dx_c0_b
    dx_c0_c = 0.0
    dx_c0_d = 0.0

    dx_c1_b = mm2i(
        x=X0[1, 0],
        v=V[1, 0],
        kx=K[1, 0, 0],
        i1=X0[1, 2],
        ki1=-I[1, 0, 2],
        i2=X0[1, 3],
        ki2=-I[1, 0, 3],
    )
    dx_c1_a = -dx_c1_b
    dx_c1_c = 0.0
    dx_c1_d = 0.0

    # test
    Xd = f(X=X0, K=K, V=V, Z=Z, I=I)

    assert Xd[0, 0] == pytest.approx(dx_c0_a, abs=TOLERANCE)
    assert Xd[0, 1] == pytest.approx(dx_c0_b, abs=TOLERANCE)
    assert Xd[0, 2] == pytest.approx(dx_c0_c, abs=TOLERANCE)
    assert Xd[0, 3] == pytest.approx(dx_c0_d, abs=TOLERANCE)

    assert Xd[1, 0] == pytest.approx(dx_c1_a, abs=TOLERANCE)
    assert Xd[1, 1] == pytest.approx(dx_c1_b, abs=TOLERANCE)
    assert Xd[1, 2] == pytest.approx(dx_c1_c, abs=TOLERANCE)
    assert Xd[1, 3] == pytest.approx(dx_c1_d, abs=TOLERANCE)


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
    Z = torch.tensor([
        [   [-1.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
        [   [0.0, 0.0, -1.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])

    # affinities (c, p, s)
    K = torch.tensor([
        [   [1.2, 1.5, 0.0, 0.0],
            [0.0, 0.5, 0.0, 3.5],
            [0.0, 0.0, 0.0, 0.0] ],
        [   [0.0, 0.0, 0.5, 1.5],
            [1.2, 0.0, 0.0, 1.9],
            [0.0, 0.0, 0.0, 0.0] ],
    ])

    # max velocities (c, p)
    V = torch.tensor([
        [2.1, 1.0, 0.0],
        [1.1, 2.0, 0.0],
    ])

    # inhibitors (c, p, s)
    I = torch.zeros(2, 3, 4)
    # fmt: on

    def mm(x, k, v):
        return v * x / (k + x)

    # expected outcome
    dx_c0_b = mm(x=X0[0, 0], v=V[0, 0], k=K[0, 0, 0])
    dx_c0_a = -dx_c0_b
    dx_c0_d = mm(x=X0[0, 1], v=V[0, 1], k=K[0, 1, 1])
    dx_c0_b = dx_c0_b - dx_c0_d

    dx_c1_d_1 = mm(x=X0[1, 2], v=V[1, 0], k=K[1, 0, 2])
    dx_c1_c = -dx_c1_d_1
    dx_c1_d_2 = mm(x=X0[1, 0], v=V[1, 1], k=K[1, 1, 0])
    dx_c1_a = -dx_c1_d_2
    dx_c1_d = dx_c1_d_1 + dx_c1_d_2

    # test
    Xd = f(X=X0, K=K, V=V, Z=Z, I=I)

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
    Z = torch.tensor([
        [   [-1.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, -2.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
        [   [0.0, -3.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])

    # affinities (c, p, s)
    K = torch.tensor([
        [   [1.2, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.9, 1.2],
            [0.0, 0.0, 0.0, 0.0] ],
        [   [0.0, 1.5, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0] ],
    ])
    
    # max velocities (c, p)
    V = torch.tensor([
        [2.1, 1.1, 0.0],
        [1.9, 0.0, 0.0],
    ])

    # inhibitors (c, p, s)
    I = torch.zeros(2, 3, 4)
    # fmt: on

    def mm(x, k, v, n):
        return v * x ** n / (k + x) ** n

    # expected outcome
    dx_c0_b = 2 * mm(x=X0[0, 0], v=V[0, 0], k=K[0, 0, 0], n=1)
    dx_c0_a = -dx_c0_b / 2
    dx_c0_d = mm(x=X0[0, 2], v=V[0, 1], k=K[0, 1, 2], n=2)
    dx_c0_c = -2 * dx_c0_d

    v0_c1 = mm(x=X0[1, 1], v=V[1, 0], k=K[1, 0, 1], n=3)
    dx_c1_a = 0.0
    dx_c1_b = -3 * v0_c1
    dx_c1_c = 2 * v0_c1
    dx_c1_d = 0.0

    # test
    Xd = f(X=X0, K=K, V=V, Z=Z, I=I)

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
    Z = torch.tensor([
        [   [-1.0, -1.0, 1.0, 0.0],
            [2.0, -1.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
        [   [-1.0, 1.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])

    # affinities (c, p, s)
    K = torch.tensor([
        [   [2.2, 1.2, 0.2, 0.0],
            [0.8, 1.9, 0.4, 1.2],
            [0.0, 0.0, 0.0, 0.0] ],
        [   [2.3, 0.2, 0.0, 1.2],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0] ],
    ])

    # max velocities (c, p)
    V = torch.tensor([
        [2.1, 1.1, 0.0],
        [1.2, 0.0, 0.0],
    ])

    # inhibitors (c, p, s)
    I = torch.zeros(2, 3, 4)
    # fmt: on

    def mm(x1, x2, k1, k2, v):
        return v * x1 * x2 / ((k1 + x1) * (k2 + x2))

    # expected outcome
    c0_v0 = mm(x1=X0[0, 0], k1=K[0, 0, 0], x2=X0[0, 1], k2=K[0, 0, 1], v=V[0, 0])
    c0_v1 = mm(x1=X0[0, 1], k1=K[0, 1, 1], x2=X0[0, 3], k2=K[0, 1, 3], v=V[0, 1])
    dx_c0_a = 2 * c0_v1 - c0_v0
    dx_c0_b = -c0_v0 - c0_v1
    dx_c0_c = c0_v0 + c0_v1
    dx_c0_d = -c0_v1
    c1_v0 = mm(x1=X0[1, 0], k1=K[1, 0, 0], x2=X0[1, 3], k2=K[1, 0, 3], v=V[1, 0])
    dx_c1_a = -c1_v0
    dx_c1_b = c1_v0
    dx_c1_c = 0.0
    dx_c1_d = -c1_v0

    # test
    Xd = f(X=X0, K=K, V=V, Z=Z, I=I)

    assert Xd[0, 0] == pytest.approx(dx_c0_a, abs=TOLERANCE)
    assert Xd[0, 1] == pytest.approx(dx_c0_b, abs=TOLERANCE)
    assert Xd[0, 2] == pytest.approx(dx_c0_c, abs=TOLERANCE)
    assert Xd[0, 3] == pytest.approx(dx_c0_d, abs=TOLERANCE)

    assert Xd[1, 0] == pytest.approx(dx_c1_a, abs=TOLERANCE)
    assert Xd[1, 1] == pytest.approx(dx_c1_b, abs=TOLERANCE)
    assert Xd[1, 2] == pytest.approx(dx_c1_c, abs=TOLERANCE)
    assert Xd[1, 3] == pytest.approx(dx_c1_d, abs=TOLERANCE)

