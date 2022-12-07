import pytest
import torch

TOLERANCE = 1e-4


def f(
    X: torch.Tensor,
    Km: torch.Tensor,
    Vmax: torch.Tensor,
    Ke: torch.Tensor,
    N: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate new signals (molecules/actions) created by all proteins of all cells
    after 1 timestep. Returns the change in signals in shape `(c, s)`.
    There are `c` cells, `p` proteins, `s` signals/molecules.

    - `X` Signal/molecule concentrations (c, s). Must all be >= 0.0.
    - `Km` Domain affinities of all proteins for each signal (c, p, s). Must all be >= 0.0.
    - `Vmax` Maximum velocities of all proteins (c, p). Must all be >= 0.0.
    - `Ke` Equilibrium constants of all proteins (c, p). If the current reaction quotient of a protein
      exceeds its equilibrium constant, it will work in the other direction. Must all be >= 0.0.
    - `N` Reaction stoichiometry of all proteins for each signal (c, p, s). Numbers < 0.0 indicate this
      amount of this molecule is being used up by the protein. Numbers > 0.0 indicate this amount of this
      molecule is being created by the protein. 0.0 means this molecule is not part of the reaction of
      this protein.
    - `A` allosteric control of all proteins for each signal (c, p, s). Must all be -1.0, 0.0, or 1.0.
      1.0 means this molecule acts as an activating effector on this protein. -1.0 means this molecule
      acts as an inhibiting effector on this protein. 0.0 means this molecule does not allosterically
      effect the protein.
    
    Everything is based on Michaelis Menten kinetics where protein velocity
    depends on substrate concentration:

    ```
        v = Vmax * x / (Km + x)
    ```

    `Vmax` is the maximum velocity of the protein and `Km` the substrate affinity. Multiple
    substrates create interaction terms such as:

    ```
        v = Vmax * x1 * / (Km1 + x1) * x2 / (Km2 + x2)
    ```

    Allosteric effectors work non-competitively such that they reduce or raise `Vmax` but
    leave any `Km` unaffected. Effectors themself also use Michaelis Menten kinteics.
    Here is substrate `x` with inhibitor `i`:
    
    ```
        v = Vmax * x / (Kmx + x) * (1 - Vi)
        Vi = i / (Kmi + i)
    ```

    Activating effectors effectively make the protein dependent on that activator:

    ```
        v = Vmax * x / (Kmx + x) * Va
        Va = a / (Kma + a)
    ```

    Multiple effectors are summed up in `Vi` and `Va` and each is clamped to `[0;1]`
    before being multiplied with `Vmax`.

    ```
        v = Vmax * x / (Km + x) * Va * (1 - Vi)
        Va = Va1 + Va2 + ...
        Vi = Vi1 + Vi2 + ...
    ```

    Equilibrium constants `Ke` define whether the reaction of a protein (as defined in `N`)
    can take place. If the reaction quotient (the combined concentrations of all products
    devided by all substrates) is smaller than its `Ke` the reaction proceeds. If it is
    greater than `Ke` the reverse reaction will take place (`N * -1.0` for this protein).
    The idea is to link signal integration to energy conversion.

    ```
        -dG0 = R * T * ln(Ke)
        Ke = exp(-dG0 / R / T)
    ```

    where `dG0` is the standard Gibb's free energy of this reaction, `R` is gas constant,
    `T` is absolute temperature.

    There's currently a chance that proteins in a cell can deconstruct more
    of a molecule than available. This would lead to a negative concentration
    in the resulting signals `X`. To avoid this, there is a correction heuristic
    which will downregulate these proteins by so much, that the molecule will
    only be reduced to 0.

    Limitations:
    - all based on Michaelis-Menten kinetics, no cooperativity
    - all allosteric control is non-competitive (activating or inhibiting)
    - there are substrate-substrate interactions but no interactions among effectors
    - 1 protein can have multiple substrates and products but there is only 1 Km for each type of molecule
    - there can only be 1 effector per molecule (e.g. not 2 different allosteric controls for the same type of molecule)
    """
    # TODO: refactor:
    #       - it seems like some of the masks are unnecessary (e.g. if I know that in
    #         N there is 0.0 in certain places, do I really need a mask like sub_M?)
    #       - can I first do the nom / denom, then pow and prod afterwards?
    #       - are there some things which I basically calculate 2 times? Can I avoid it?
    #         or are there some intermediates which could be reused if I would calculate
    #         them slightly different/in different order?
    #       - does it help to use masked tensors? (pytorch.org/docs/stable/masked.html)
    #       - better names, split into a few functions?

    # substrates
    sub_M = torch.where(N < 0.0, 1.0, 0.0)  # (c, p s)
    sub_X = torch.einsum("cps,cs->cps", sub_M, X)  # (c, p, s)
    sub_N = torch.where(N < 0.0, -N, 0.0)  # (c, p, s)

    # products
    pro_M = torch.where(N > 0.0, 1.0, 0.0)  # (c, p s)
    pro_X = torch.einsum("cps,cs->cps", pro_M, X)  # (c, p, s)
    pro_N = torch.where(N > 0.0, N, 0.0)  # (c, p, s)

    # quotients
    nom = torch.pow(pro_X, pro_N).prod(2)  # (c, p)
    denom = torch.pow(sub_X, sub_N).prod(2)  # (c, p)
    Q = nom / denom  # (c, p)

    # adjust direction
    adj_N = torch.where(Q - Ke > 0.0, -1.0, 1.0)  # (c, p)
    N_adj = torch.einsum("cps,cp->cps", N, adj_N)

    # inhibitors
    inh_M = torch.where(A < 0.0, 1.0, 0.0)  # (c, p, s)
    inh_X = torch.einsum("cps,cs->cps", inh_M, X)  # (c, p, s)
    inh_V = torch.nansum(inh_X / (Km + inh_X), dim=2).clamp(0, 1)  # (c, p)

    # activators
    act_M = torch.where(A > 0.0, 1.0, 0.0)  # (c, p, s)
    act_X = torch.einsum("cps,cs->cps", act_M, X)  # (c, p, s)
    act_V = torch.nansum(act_X / (Km + act_X), dim=2).clamp(0, 1)  # (c, p)
    act_V_adj = torch.where(act_M.sum(dim=2) > 0, act_V, 1.0)  # (c, p)

    # substrates
    sub_M = torch.where(N_adj < 0.0, 1.0, 0.0)  # (c, p s)
    sub_X = torch.einsum("cps,cs->cps", sub_M, X)  # (c, p, s)
    sub_N = torch.where(N_adj < 0.0, -N_adj, 0.0)  # (c, p, s)

    # proteins
    prot_Vmax = sub_M.max(dim=2).values * Vmax  # (c, p)
    nom = torch.pow(sub_X, sub_N).prod(2)  # (c, p)
    denom = torch.pow(Km + sub_X, sub_N).prod(2)  # (c, p)
    prot_V = prot_Vmax * nom / denom * (1 - inh_V) * act_V_adj  # (c, p)

    # concentration deltas (c, s)
    Xd = torch.einsum("cps,cp->cs", N_adj, prot_V)

    X1 = X + Xd
    if torch.any(X1 < 0):
        neg_conc = torch.where(X1 < 0, 1.0, 0.0)  # (c, s)
        candidates = torch.where(N_adj < 0.0, 1.0, 0.0)  # (c, p, s)

        # which proteins need to be down-regulated (c, p)
        BX_mask = torch.einsum("cps,cs->cps", candidates, neg_conc)
        prot_M = BX_mask.max(dim=2).values

        # what are the correction factors (c,)
        correct = torch.where(neg_conc > 0.0, -X / Xd, 1.0).min(dim=1).values

        # correction for protein velocities (c, p)
        prot_V_adj = torch.einsum("cp,c->cp", prot_M, correct)
        prot_V_adj = torch.where(prot_V_adj == 0.0, 1.0, prot_V_adj)

        # new concentration deltas (c, s)
        Xd = torch.einsum("cps,cp->cs", N_adj, prot_V * prot_V_adj)

        # TODO: can I already predict this before calculated Xd the first time?
        #       maybe at the time when I know the intended protein velocities?

    # TODO: correct for X1 Q -Ke changes
    #       if the new concentration (x1) would have changed the direction
    #       of the reaction (a -> b) -> (b -> a), I would have a constant
    #       back and forth between the 2 reactions always overshooting the
    #       actual equilibrium
    #       It would be better in this case to arrive at Q = Ke, so that
    #       the reaction stops with x1
    #       could be a correction term (like X1 < 0) or better solution
    #       that can be calculated ahead of time

    return Xd


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
    Xd = f(X=X0, Km=Km, Vmax=Vmax, Ke=Ke, N=N, A=A)

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
    Xd = f(X=X0, Km=Km, Vmax=Vmax, Ke=Ke, N=N, A=A)

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
    Xd = f(X=X0, Km=Km, Vmax=Vmax, Ke=Ke, N=N, A=A)

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
        [1.1, 2.5, 0.9, 1.0],
        [2.3, 1.6, 3.0, 0.9],
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
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]   ],
    ])

    # equilibrium constants (c, p)
    Ke = torch.full((2, 3), 999.9)

    # fmt: on

    def mmi(x, kx, v, i, ki):
        return v * x / (kx + x) * (1 - i / (ki + i))

    def mm2i(x, kx, v, i1, ki1, i2, ki2):
        return v * x / (kx + x) * max(0, 1 - i1 / (ki1 + i1) - i2 / (ki2 + i2))

    def mma(x, kx, v, a, ka):
        return v * x / (kx + x) * a / (ka + a)

    def mm2a(x, kx, v, a1, ka1, a2, ka2):
        return v * x / (kx + x) * min(1, a1 / (ka1 + a1) + a2 / (ka2 + a2))

    # expected outcome
    v0_c0 = mmi(x=X0[0, 0], v=Vmax[0, 0], kx=Km[0, 0, 0], i=X0[0, 2], ki=Km[0, 0, 2])
    v1_c0 = mma(x=X0[0, 2], v=Vmax[0, 1], kx=Km[0, 1, 2], a=X0[0, 0], ka=Km[0, 1, 0])
    dx_c0_b = v0_c0
    dx_c0_a = -v0_c0
    dx_c0_c = -v1_c0
    dx_c0_d = v1_c0

    v0_c1 = mm2i(
        x=X0[1, 0],
        v=Vmax[1, 0],
        kx=Km[1, 0, 0],
        i1=X0[1, 2],
        ki1=Km[1, 0, 2],
        i2=X0[1, 3],
        ki2=Km[1, 0, 3],
    )
    v1_c1 = mm2a(
        x=X0[1, 2],
        v=Vmax[1, 1],
        kx=Km[1, 1, 2],
        a1=X0[1, 0],
        ka1=Km[1, 1, 0],
        a2=X0[1, 1],
        ka2=Km[1, 1, 1],
    )
    dx_c1_a = -v0_c1
    dx_c1_b = v0_c1
    dx_c1_c = -v1_c1
    dx_c1_d = v1_c1

    # test
    Xd = f(X=X0, Km=Km, Vmax=Vmax, Ke=Ke, N=N, A=A)

    assert Xd[0, 0] == pytest.approx(dx_c0_a, abs=TOLERANCE)
    assert Xd[0, 1] == pytest.approx(dx_c0_b, abs=TOLERANCE)
    assert Xd[0, 2] == pytest.approx(dx_c0_c, abs=TOLERANCE)
    assert Xd[0, 3] == pytest.approx(dx_c0_d, abs=TOLERANCE)

    assert Xd[1, 0] == pytest.approx(dx_c1_a, abs=TOLERANCE)
    assert Xd[1, 1] == pytest.approx(dx_c1_b, abs=TOLERANCE)
    assert Xd[1, 2] == pytest.approx(dx_c1_c, abs=TOLERANCE)
    assert Xd[1, 3] == pytest.approx(dx_c1_d, abs=TOLERANCE)


def test_reduce_velocity_to_avoid_negative_concentrations():
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
    assert X0[0, 0] - v0_c0 < 0.0
    # so velocity should be reduced to a (it will be 0 afterwards)
    v0_c0 = X0[0, 0]
    # the other protein was unaffected
    v1_c1 = mm(x=X0[0, 1], v=Vmax[0, 1], k=Km[0, 1, 1])
    dx_c0_a = -v0_c0
    dx_c0_b = v0_c0 - v1_c1
    dx_c0_c = 0.0
    dx_c0_d = v1_c1

    v0_c1 = mm(x=X0[1, 2], v=Vmax[1, 0], k=Km[1, 0, 2], n=2)
    # but this would lead to Xd + X0 = -0.0722 (for c)
    assert X0[1, 2] - 2 * v0_c1 < 0.0
    # so velocity should be reduced to c (it will be 0 afterwards)
    v0_c1 = X0[1, 2] / 2
    dx_c1_a = 0.0
    dx_c1_b = 0.0
    dx_c1_c = -2 * v0_c1
    dx_c1_d = v0_c1

    # test
    Xd = f(X=X0, Km=Km, Vmax=Vmax, Ke=Ke, N=N, A=A)

    assert Xd[0, 0] == pytest.approx(dx_c0_a, abs=TOLERANCE)
    assert Xd[0, 1] == pytest.approx(dx_c0_b, abs=TOLERANCE)
    assert Xd[0, 2] == pytest.approx(dx_c0_c, abs=TOLERANCE)
    assert Xd[0, 3] == pytest.approx(dx_c0_d, abs=TOLERANCE)

    assert Xd[1, 0] == pytest.approx(dx_c1_a, abs=TOLERANCE)
    assert Xd[1, 1] == pytest.approx(dx_c1_b, abs=TOLERANCE)
    assert Xd[1, 2] == pytest.approx(dx_c1_c, abs=TOLERANCE)
    assert Xd[1, 3] == pytest.approx(dx_c1_d, abs=TOLERANCE)


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
    Xd = f(X=X0, Km=Km, Vmax=Vmax, Ke=Ke, N=N, A=A)

    assert Xd[0, 0] == pytest.approx(dx_c0_a, abs=TOLERANCE)
    assert Xd[0, 1] == pytest.approx(dx_c0_b, abs=TOLERANCE)
    assert Xd[0, 2] == pytest.approx(dx_c0_c, abs=TOLERANCE)
    assert Xd[0, 3] == pytest.approx(dx_c0_d, abs=TOLERANCE)

    assert Xd[1, 0] == pytest.approx(dx_c1_a, abs=TOLERANCE)
    assert Xd[1, 1] == pytest.approx(dx_c1_b, abs=TOLERANCE)
    assert Xd[1, 2] == pytest.approx(dx_c1_c, abs=TOLERANCE)
    assert Xd[1, 3] == pytest.approx(dx_c1_d, abs=TOLERANCE)
