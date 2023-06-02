import pytest
import torch
from magicsoup.containers import (
    Molecule,
    CatalyticDomain,
    RegulatoryDomain,
    TransporterDomain,
)
from magicsoup.kinetics2 import Kinetics

TOLERANCE = 1e-4
EPS = 1e-8


ma = Molecule("a", energy=15)
mb = Molecule("b", energy=10)
mc = Molecule("c", energy=10)
md = Molecule("d", energy=5)
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


def get_kinetics() -> Kinetics:
    kinetics = Kinetics(molecules=MOLECULES, reactions=REACTIONS, n_computations=1)
    kinetics.km_map.weights = KM_WEIGHTS.clone()
    kinetics.vmax_map.weights = VMAX_WEIGHTS.clone()
    kinetics.sign_map.signs = SIGNS.clone()
    kinetics.transport_map.M = TRANSPORT_M.clone()
    kinetics.effector_map.M = EFFECTOR_M.clone()
    kinetics.reaction_map.M = REACTION_M.clone()
    return kinetics


def avg(*x):
    return sum(x) / len(x)


def atest_cell_params_with_catalytic_domains_and_co_factors():
    # Dealing with stoichiometric numbers that cancel each other out
    # in general, intermediate molecules should be 0 in N
    # but the reaction must depend on abundance of the starting molecules
    # the first domain defines these starting molecules
    # if these are 0 in N, they must appear in A

    # C0, P0:
    # bc->d then d->2b so bc->2b, with N b=1 c=-1 d=0
    # b needs to be added as activating effector A b=1
    # C0, P1:
    # d->2b then bc->d, with N b=1 c=-1 d=0
    # d needs to be added as activating effector A d=1
    #
    # the stoichiometry in both proteins in C0 is the same, but
    # the order in which domains were defined is different
    # That's why they differ in A

    # Domain spec indexes: (dom_types, reacts_trnspts_effctrs, Vmaxs, Kms, signs)
    # fmt: off
    c0 = [
        [
            (1, 2, 15, 1, 3), # catal, Vmax 1.2, Km 1.5, fwd, bc->d
            (1, 1, 5, 1, 4), # catal, Vmax 1.1, Km 0.5, fwd, d->2b
        ],
        [
            (1, 1, 5, 1, 4), # catal, Vmax 1.1, Km 0.5, fwd, d->2b
            (1, 2, 15, 1, 3), # catal, Vmax 1.2, Km 1.5, fwd, bc->d
        ]
    ]

    # fmt: on

    Km = torch.zeros(1, 3, 8)
    Vmax = torch.zeros(1, 3)
    E = torch.zeros(1, 3)
    N = torch.zeros(1, 3, 8)
    A = torch.zeros(1, 3, 8)

    # test
    kinetics = get_kinetics()
    kinetics.Km = Km
    kinetics.Vmax = Vmax
    kinetics.E = E
    kinetics.N = N
    kinetics.A = A
    kinetics.set_cell_params(cell_idxs=[0], proteomes=[c0])

    # proteins in C0 only differ in A
    for p in [0, 1]:
        assert Km[0, p, 1] == pytest.approx(avg(1.5, 1 / 0.5), abs=TOLERANCE)
        assert Km[0, p, 2] == pytest.approx(1.5, abs=TOLERANCE)
        assert Km[0, p, 3] == pytest.approx(avg(0.5, 1 / 1.5), abs=TOLERANCE)
        for i in [0, 4, 5, 6, 7]:
            assert Km[0, p, i] == 0.0

        assert Vmax[0, p] == pytest.approx(avg(1.1, 1.2), abs=TOLERANCE)

        assert E[0, p] == 10 - 10

        assert N[0, p, 1] == 1
        assert N[0, p, 2] == -1
        assert N[0, p, 3] == 0
        for i in [0, 4, 5, 6, 7]:
            assert N[0, p, i] == 0

    assert A[0, 0, 1] == 1
    assert A[0, 1, 3] == 1
    assert A.sum() == 2

    # test proteome representation

    proteins = kinetics.get_proteome(proteome=c0)

    p0 = proteins[0]
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].substrates == [mb, mc]
    assert p0.domains[0].products == [md]
    assert p0.domains[0].vmax == pytest.approx(1.2, abs=TOLERANCE)
    assert p0.domains[0].km == pytest.approx(1.5, abs=TOLERANCE)
    assert isinstance(p0.domains[1], CatalyticDomain)
    assert p0.domains[1].substrates == [md]
    assert p0.domains[1].products == [mb, mb]
    assert p0.domains[1].vmax == pytest.approx(1.1, abs=TOLERANCE)
    assert p0.domains[1].km == pytest.approx(0.5, abs=TOLERANCE)

    p1 = proteins[1]
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].substrates == [md]
    assert p1.domains[0].products == [mb, mb]
    assert p1.domains[0].vmax == pytest.approx(1.1, abs=TOLERANCE)
    assert p1.domains[0].km == pytest.approx(0.5, abs=TOLERANCE)
    assert isinstance(p1.domains[1], CatalyticDomain)
    assert p1.domains[1].substrates == [mb, mc]
    assert p1.domains[1].products == [md]
    assert p1.domains[1].vmax == pytest.approx(1.2, abs=TOLERANCE)
    assert p1.domains[1].km == pytest.approx(1.5, abs=TOLERANCE)


def atest_cell_params_with_transporter_domains():
    # Domain spec indexes: (dom_types, reacts_trnspts_effctrs, Vmaxs, Kms, signs)
    # fmt: off
    c0 = [
        [
            (2, 5, 5, 1, 1)  # transporter, Vmax 1.5, Km 0.5, fwd, mol a
        ],
        [
            (2, 5, 5, 1, 1), # transporter, Vmax 1.5, Km 0.5, fwd, mol a
            (2, 1, 2, 2, 1)  # transporter, Vmax 1.1, Km 0.2, bwd, mol a
        ],
    ]
    c1 = [
        [
            (2, 5, 4, 1, 1), # transporter, Vmax 1.5, Km 0.4, fwd, mol a
            (2, 4, 5, 1, 1), # transporter, Vmax 1.4, Km 0.5, fwd, mol a
            (2, 3, 6, 1, 2), # transporter, Vmax 1.3, Km 0.6, fwd, mol b
            (2, 2, 7, 1, 3)  # transporter, Vmax 1.2, Km 0.7, fwd, mol c
        ],
        [
            (1, 10, 5, 1, 1), # catal, Vmax 2.0, Km 0.5, fwd, a->b
            (2, 5, 5, 1, 1)   # transporter, Vmax 1.5, Km 0.5, fwd, mol a
        ],
    ]
    # fmt: on

    Km = torch.zeros(2, 3, 8)
    Vmax = torch.zeros(2, 3)
    E = torch.zeros(2, 3)
    N = torch.zeros(2, 3, 8)
    A = torch.zeros(2, 3, 8)

    # test
    kinetics = get_kinetics()
    kinetics.Km = Km
    kinetics.Vmax = Vmax
    kinetics.E = E
    kinetics.N = N
    kinetics.A = A
    kinetics.set_cell_params(cell_idxs=[0, 1], proteomes=[c0, c1])

    assert Km[0, 0, 0] == pytest.approx(0.5, abs=TOLERANCE)
    assert Km[0, 0, 4] == pytest.approx(1 / 0.5, abs=TOLERANCE)
    for i in [1, 2, 3, 5, 6, 7]:
        assert Km[0, 0, i] == 0.0
    assert Km[0, 1, 0] == pytest.approx(avg(0.5, 1 / 0.2), abs=TOLERANCE)
    assert Km[0, 1, 4] == pytest.approx(avg(1 / 0.5, 0.2), abs=TOLERANCE)
    for i in [1, 2, 3, 5, 6, 7]:
        assert Km[0, 1, i] == 0.0

    assert Km[1, 0, 0] == pytest.approx(avg(0.4, 0.5), abs=TOLERANCE)
    assert Km[1, 0, 1] == pytest.approx(0.6, abs=TOLERANCE)
    assert Km[1, 0, 2] == pytest.approx(0.7, abs=TOLERANCE)
    assert Km[1, 0, 4] == pytest.approx(avg(1 / 0.4, 1 / 0.5), abs=TOLERANCE)
    assert Km[1, 0, 5] == pytest.approx(1 / 0.6, abs=TOLERANCE)
    assert Km[1, 0, 6] == pytest.approx(1 / 0.7, abs=TOLERANCE)
    for i in [3, 7]:
        assert Km[1, 0, i] == 0.0
    assert Km[1, 1, 0] == pytest.approx(avg(0.5, 0.5), abs=TOLERANCE)
    assert Km[1, 1, 1] == pytest.approx(1 / 0.5, abs=TOLERANCE)
    assert Km[1, 1, 4] == pytest.approx(1 / 0.5, abs=TOLERANCE)
    for i in [2, 3, 5, 6, 7]:
        assert Km[1, 1, i] == 0.0

    assert Vmax[0, 0] == pytest.approx(1.5, abs=TOLERANCE)
    assert Vmax[0, 1] == pytest.approx(avg(1.5, 1.1), abs=TOLERANCE)
    assert Vmax[0, 2] == 0.0

    assert Vmax[1, 0] == pytest.approx(avg(1.5, 1.4, 1.3, 1.2), abs=TOLERANCE)
    assert Vmax[1, 1] == pytest.approx(avg(2.0, 1.5), abs=TOLERANCE)
    assert Vmax[1, 2] == 0.0

    assert E[0, 0] == 0
    assert E[0, 1] == 0
    assert E[0, 2] == 0

    assert E[1, 0] == 0
    assert E[1, 1] == 10 - 15
    assert E[1, 2] == 0

    assert N[0, 0, 0] == -1
    assert N[0, 0, 4] == 1
    for i in [1, 2, 3, 5, 6, 7]:
        assert N[0, 0, i] == 0
    assert N[0, 1, 0] == 0
    assert N[0, 1, 4] == 0
    for i in [1, 2, 3, 5, 6, 7]:
        assert N[0, 1, i] == 0

    assert N[1, 0, 0] == -2
    assert N[1, 0, 1] == -1
    assert N[1, 0, 2] == -1
    assert N[1, 0, 4] == 2
    assert N[1, 0, 5] == 1
    assert N[1, 0, 6] == 1
    for i in [3, 7]:
        assert N[1, 0, i] == 0
    assert N[1, 1, 0] == -2
    assert N[1, 1, 1] == 1
    assert N[1, 1, 4] == 1
    for i in [2, 3, 5, 6, 7]:
        assert N[1, 1, i] == 0

    # a was imported and exported, so its N=0
    # but it must be added as effector
    assert A[0, 1, 0] == 1
    assert A.sum() == 1

    # test proteome representation

    proteins = kinetics.get_proteome(proteome=c0)

    p0 = proteins[0]
    assert isinstance(p0.domains[0], TransporterDomain)
    assert p0.domains[0].molecule is ma
    assert p0.domains[0].vmax == pytest.approx(1.5, abs=TOLERANCE)
    assert p0.domains[0].km == pytest.approx(0.5, abs=TOLERANCE)

    p1 = proteins[1]
    assert isinstance(p1.domains[0], TransporterDomain)
    assert p1.domains[0].molecule is ma
    assert p1.domains[0].vmax == pytest.approx(1.5, abs=TOLERANCE)
    assert p1.domains[0].km == pytest.approx(0.5, abs=TOLERANCE)
    assert isinstance(p1.domains[1], TransporterDomain)
    assert p1.domains[1].molecule is ma
    assert p1.domains[1].vmax == pytest.approx(1.1, abs=TOLERANCE)
    assert p1.domains[1].km == pytest.approx(1 / 0.2, abs=TOLERANCE)

    proteins = kinetics.get_proteome(proteome=c1)

    p0 = proteins[0]
    assert isinstance(p0.domains[0], TransporterDomain)
    assert p0.domains[0].molecule is ma
    assert p0.domains[0].vmax == pytest.approx(1.5, abs=TOLERANCE)
    assert p0.domains[0].km == pytest.approx(0.4, abs=TOLERANCE)
    assert isinstance(p0.domains[1], TransporterDomain)
    assert p0.domains[1].molecule is ma
    assert p0.domains[1].vmax == pytest.approx(1.4, abs=TOLERANCE)
    assert p0.domains[1].km == pytest.approx(0.5, abs=TOLERANCE)
    assert isinstance(p0.domains[2], TransporterDomain)
    assert p0.domains[2].molecule is mb
    assert p0.domains[2].vmax == pytest.approx(1.3, abs=TOLERANCE)
    assert p0.domains[2].km == pytest.approx(0.6, abs=TOLERANCE)
    assert isinstance(p0.domains[3], TransporterDomain)
    assert p0.domains[3].molecule is mc
    assert p0.domains[3].vmax == pytest.approx(1.2, abs=TOLERANCE)
    assert p0.domains[3].km == pytest.approx(0.7, abs=TOLERANCE)

    p1 = proteins[1]
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].substrates == [ma]
    assert p1.domains[0].products == [mb]
    assert p1.domains[0].vmax == pytest.approx(2.0, abs=TOLERANCE)
    assert p1.domains[0].km == pytest.approx(0.5, abs=TOLERANCE)
    assert isinstance(p1.domains[1], TransporterDomain)
    assert p1.domains[1].molecule is ma
    assert p1.domains[1].vmax == pytest.approx(1.5, abs=TOLERANCE)
    assert p1.domains[1].km == pytest.approx(0.5, abs=TOLERANCE)


def atest_cell_params_with_regulatory_domains():
    # Domain spec indexes: (dom_types, reacts_trnspts_effctrs, Vmaxs, Kms, signs)
    # fmt: off
    c0 = [
        [
            (1, 10, 5, 1, 1), # catal, Vmax 2.0, Km 0.5, fwd, a->b
            (3, 0, 10, 1, 3), # reg, Km 1.0, cyto, act, c
            (3, 0, 20, 2, 4), # reg, Km 2.0, cyto, inh, d
        ],
        [
            (1, 10, 5, 1, 1), # catal, Vmax 2.0, Km 0.5, fwd, a->b
            (3, 0, 10, 1, 1), # reg, Km 1.0, cyto, act, a
            (3, 0, 15, 1, 5), # reg, Km 1.5, transm, act, a
        ]
    ]

    c1 = [
        [
            (1, 10, 5, 1, 1), # catal, Vmax 2.0, Km 0.5, fwd, a->b
            (3, 0, 10, 2, 2), # reg, Km 1.0, cyto, inh, b
            (3, 0, 15, 2, 6), # reg, Km 1.5, transm, inh, b
        ],
        [
            (1, 10, 5, 1, 1), # catal, Vmax 2.0, Km 0.5, fwd, a->b
            (3, 0, 10, 1, 4), # reg, Km 1.0, cyto, act, d
            (3, 0, 15, 1, 4), # reg, Km 1.5, cyto, act, d
        ]
    ]
    # fmt: on

    Km = torch.zeros(2, 3, 8)
    Vmax = torch.zeros(2, 3)
    E = torch.zeros(2, 3)
    N = torch.zeros(2, 3, 8)
    A = torch.zeros(2, 3, 8)

    # test
    kinetics = get_kinetics()
    kinetics.Km = Km
    kinetics.Vmax = Vmax
    kinetics.E = E
    kinetics.N = N
    kinetics.A = A
    kinetics.set_cell_params(cell_idxs=[0, 1], proteomes=[c0, c1])

    assert Km[0, 0, 0] == pytest.approx(0.5, abs=TOLERANCE)
    assert Km[0, 0, 1] == pytest.approx(1 / 0.5, abs=TOLERANCE)
    assert Km[0, 0, 2] == pytest.approx(1.0, abs=TOLERANCE)
    assert Km[0, 0, 3] == pytest.approx(2.0, abs=TOLERANCE)
    for i in [4, 5, 6, 7]:
        assert Km[0, 0, i] == 0.0
    assert Km[0, 1, 0] == pytest.approx(avg(0.5, 1.0), abs=TOLERANCE)
    assert Km[0, 1, 1] == pytest.approx(1 / 0.5, abs=TOLERANCE)
    assert Km[0, 1, 2] == pytest.approx(0.0, abs=TOLERANCE)
    assert Km[0, 1, 3] == pytest.approx(0.0, abs=TOLERANCE)
    assert Km[0, 1, 4] == pytest.approx(1.5, abs=TOLERANCE)
    for i in [5, 6, 7]:
        assert Km[0, 1, i] == 0.0
    for i in range(8):
        assert Km[0, 2, i] == 0.0

    assert Km[1, 0, 0] == pytest.approx(0.5, abs=TOLERANCE)
    assert Km[1, 0, 1] == pytest.approx(avg(1 / 0.5, 1.0), abs=TOLERANCE)
    assert Km[1, 0, 2] == pytest.approx(0.0, abs=TOLERANCE)
    assert Km[1, 0, 3] == pytest.approx(0.0, abs=TOLERANCE)
    assert Km[1, 0, 5] == pytest.approx(1.5, abs=TOLERANCE)
    for i in [4, 6, 7]:
        assert Km[1, 0, i] == 0.0
    assert Km[1, 1, 0] == pytest.approx(0.5, abs=TOLERANCE)
    assert Km[1, 1, 1] == pytest.approx(1 / 0.5, abs=TOLERANCE)
    assert Km[1, 1, 2] == pytest.approx(0.0, abs=TOLERANCE)
    assert Km[1, 1, 3] == pytest.approx(avg(1.0, 1.5), abs=TOLERANCE)
    for i in [4, 5, 6, 7]:
        assert Km[1, 1, i] == 0.0
    for i in range(8):
        assert Km[1, 2, i] == 0.0

    assert Vmax[0, 0] == pytest.approx(2.0, abs=TOLERANCE)
    assert Vmax[0, 1] == pytest.approx(2.0, abs=TOLERANCE)
    assert Vmax[0, 2] == 0.0

    assert Vmax[1, 0] == pytest.approx(2.0, abs=TOLERANCE)
    assert Vmax[1, 1] == pytest.approx(2.0, abs=TOLERANCE)
    assert Vmax[1, 2] == 0.0

    assert E[0, 0] == 10 - 15
    assert E[0, 1] == 10 - 15
    assert E[0, 2] == 0

    assert E[1, 0] == 10 - 15
    assert E[1, 1] == 10 - 15
    assert E[1, 2] == 0

    assert N[0, 0, 0] == -1
    assert N[0, 0, 1] == 1
    assert N[0, 0, 2] == 0
    assert N[0, 0, 3] == 0
    for i in [4, 5, 6, 7]:
        assert N[0, 0, i] == 0
    assert N[0, 1, 0] == -1
    assert N[0, 1, 1] == 1
    assert N[0, 1, 2] == 0
    assert N[0, 1, 3] == 0
    for i in [4, 5, 6, 7]:
        assert N[0, 1, i] == 0
    for i in range(8):
        assert N[0, 2, i] == 0

    assert N[1, 0, 0] == -1
    assert N[1, 0, 1] == 1
    assert N[1, 0, 2] == 0
    assert N[1, 0, 3] == 0
    for i in [4, 5, 6, 7]:
        assert N[1, 0, i] == 0
    assert N[1, 1, 0] == -1
    assert N[1, 1, 1] == 1
    assert N[1, 1, 2] == 0
    assert N[1, 1, 3] == 0
    for i in [4, 5, 6, 7]:
        assert N[1, 1, i] == 0
    for i in range(8):
        assert N[1, 2, i] == 0

    assert A[0, 0, 0] == 0
    assert A[0, 0, 1] == 0
    assert A[0, 0, 2] == 1
    assert A[0, 0, 3] == -1
    for i in [4, 5, 6, 7]:
        assert A[0, 0, i] == 0
    assert A[0, 1, 0] == 1
    assert A[0, 1, 1] == 0
    assert A[0, 1, 2] == 0
    assert A[0, 1, 3] == 0
    assert A[0, 1, 4] == 1
    for i in [5, 6, 7]:
        assert A[0, 1, i] == 0
    for i in range(8):
        assert A[0, 2, i] == 0

    assert A[1, 0, 0] == 0
    assert A[1, 0, 1] == -1
    assert A[1, 0, 2] == 0
    assert A[1, 0, 3] == 0
    assert A[1, 0, 5] == -1
    for i in [4, 6, 7]:
        assert A[0, 0, i] == 0
    assert A[1, 1, 0] == 0
    assert A[1, 1, 1] == 0
    assert A[1, 1, 2] == 0
    assert A[1, 1, 3] == 2
    for i in [4, 5, 6, 7]:
        assert A[1, 1, i] == 0
    for i in range(8):
        assert A[1, 2, i] == 0

    # test protein representation

    proteins = kinetics.get_proteome(proteome=c0)

    p0 = proteins[0]
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].substrates == [ma]
    assert p0.domains[0].products == [mb]
    assert p0.domains[0].vmax == pytest.approx(2.0, abs=TOLERANCE)
    assert p0.domains[0].km == pytest.approx(0.5, abs=TOLERANCE)
    assert isinstance(p0.domains[1], RegulatoryDomain)
    assert p0.domains[1].effector is mc
    assert not p0.domains[1].is_inhibiting
    assert not p0.domains[1].is_transmembrane
    assert p0.domains[1].km == pytest.approx(1.0, abs=TOLERANCE)
    assert isinstance(p0.domains[2], RegulatoryDomain)
    assert p0.domains[2].effector is md
    assert p0.domains[2].is_inhibiting
    assert not p0.domains[2].is_transmembrane
    assert p0.domains[2].km == pytest.approx(2.0, abs=TOLERANCE)

    p1 = proteins[1]
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].substrates == [ma]
    assert p1.domains[0].products == [mb]
    assert p1.domains[0].vmax == pytest.approx(2.0, abs=TOLERANCE)
    assert p1.domains[0].km == pytest.approx(0.5, abs=TOLERANCE)
    assert isinstance(p1.domains[1], RegulatoryDomain)
    assert p1.domains[1].effector is ma
    assert not p1.domains[1].is_inhibiting
    assert not p1.domains[1].is_transmembrane
    assert p1.domains[1].km == pytest.approx(1.0, abs=TOLERANCE)
    assert isinstance(p1.domains[2], RegulatoryDomain)
    assert p1.domains[2].effector is ma
    assert not p1.domains[2].is_inhibiting
    assert p1.domains[2].is_transmembrane
    assert p1.domains[2].km == pytest.approx(1.5, abs=TOLERANCE)

    proteins = kinetics.get_proteome(proteome=c1)

    p0 = proteins[0]
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].substrates == [ma]
    assert p0.domains[0].products == [mb]
    assert p0.domains[0].vmax == pytest.approx(2.0, abs=TOLERANCE)
    assert p0.domains[0].km == pytest.approx(0.5, abs=TOLERANCE)
    assert isinstance(p0.domains[1], RegulatoryDomain)
    assert p0.domains[1].effector is mb
    assert p0.domains[1].is_inhibiting
    assert not p0.domains[1].is_transmembrane
    assert p0.domains[1].km == pytest.approx(1.0, abs=TOLERANCE)
    assert isinstance(p0.domains[2], RegulatoryDomain)
    assert p0.domains[2].effector is mb
    assert p0.domains[2].is_inhibiting
    assert p0.domains[2].is_transmembrane
    assert p0.domains[2].km == pytest.approx(1.5, abs=TOLERANCE)

    p1 = proteins[1]
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].substrates == [ma]
    assert p1.domains[0].products == [mb]
    assert p1.domains[0].vmax == pytest.approx(2.0, abs=TOLERANCE)
    assert p1.domains[0].km == pytest.approx(0.5, abs=TOLERANCE)
    assert isinstance(p1.domains[1], RegulatoryDomain)
    assert p1.domains[1].effector is md
    assert not p1.domains[1].is_inhibiting
    assert not p1.domains[1].is_transmembrane
    assert p1.domains[1].km == pytest.approx(1.0, abs=TOLERANCE)
    assert isinstance(p1.domains[2], RegulatoryDomain)
    assert p1.domains[2].effector is md
    assert not p1.domains[2].is_inhibiting
    assert not p1.domains[2].is_transmembrane
    assert p1.domains[2].km == pytest.approx(1.5, abs=TOLERANCE)


def atest_cell_params_with_catalytic_domains():
    # Domain spec indexes: (dom_types, reacts_trnspts_effctrs, Vmaxs, Kms, signs)
    # fmt: off
    c0 = [
        [
            (1, 1, 5, 1, 1), # catal, Vmax 1.1, Km 0.5, fwd, a->b
            (1, 2, 15, 2, 3), # catal, Vmax 1.2, Km 1.5, bwd, bc->d
        ],
        [
            (1, 10, 9, 1, 2), # catal, Vmax 2.0, Km 0.9, fwd, b->c
            (1, 3, 12, 2, 3), # catal, Vmax 1.3, Km 1.2, bwd, bc->d
        ],
        [
            (1, 19, 29, 1, 4), # catal, Vmax 2.9, Km 2.9, fwd, d->bb
        ]
    ]
    c1 = [
        [
            (1, 1, 3, 2, 1), # catal, Vmax 1.1, Km 0.3, bwd, a->b
            (1, 11, 14, 2, 3), # catal, Vmax 2.1, Km 1.4, bwd, bc->d
        ],
        [
            (1, 9, 3, 1, 2), # catal, Vmax 1.9, Km 0.3, fwd, b->c
            (1, 13, 17, 1, 3), # catal, Vmax 2.3, Km 1.7, fwd, bc->d
        ]
    ]

    # fmt: on

    Km = torch.zeros(2, 3, 8)
    Vmax = torch.zeros(2, 3)
    E = torch.zeros(2, 3)
    N = torch.zeros(2, 3, 8)
    A = torch.zeros(2, 3, 8)

    # test
    kinetics = get_kinetics()
    kinetics.Km = Km
    kinetics.Vmax = Vmax
    kinetics.E = E
    kinetics.N = N
    kinetics.A = A
    kinetics.set_cell_params(cell_idxs=[0, 1], proteomes=[c0, c1])

    assert Km[0, 0, 0] == pytest.approx(0.5, abs=TOLERANCE)
    assert Km[0, 0, 1] == pytest.approx(avg(1 / 0.5, 1 / 1.5), abs=TOLERANCE)
    assert Km[0, 0, 2] == pytest.approx(1 / 1.5, abs=TOLERANCE)
    assert Km[0, 0, 3] == pytest.approx(1.5, abs=TOLERANCE)
    for i in [4, 5, 6, 7]:
        assert Km[0, 0, i] == 0.0
    assert Km[0, 1, 0] == 0.0
    assert Km[0, 1, 1] == pytest.approx(avg(0.9, 1 / 1.2), abs=TOLERANCE)
    assert Km[0, 1, 2] == pytest.approx(avg(1 / 0.9, 1 / 1.2), abs=TOLERANCE)
    assert Km[0, 1, 3] == pytest.approx(1.2, abs=TOLERANCE)
    for i in [4, 5, 6, 7]:
        assert Km[0, 1, i] == 0.0
    assert Km[0, 2, 0] == 0.0
    assert Km[0, 2, 1] == pytest.approx(1 / 2.9, abs=TOLERANCE)
    assert Km[0, 2, 2] == 0.0
    assert Km[0, 2, 3] == pytest.approx(2.9, abs=TOLERANCE)
    for i in [4, 5, 6, 7]:
        assert Km[0, 2, i] == 0.0

    assert Km[1, 0, 0] == pytest.approx(1 / 0.3, abs=TOLERANCE)
    assert Km[1, 0, 1] == pytest.approx(avg(0.3, 1 / 1.4), abs=TOLERANCE)
    assert Km[1, 0, 2] == pytest.approx(1 / 1.4, abs=TOLERANCE)
    assert Km[1, 0, 3] == pytest.approx(1.4, abs=TOLERANCE)
    for i in [4, 5, 6, 7]:
        assert Km[1, 0, i] == 0.0
    assert Km[1, 1, 0] == 0.0
    assert Km[1, 1, 1] == pytest.approx(avg(0.3, 1.7), abs=TOLERANCE)
    assert Km[1, 1, 2] == pytest.approx(avg(1 / 0.3, 1.7), abs=TOLERANCE)
    assert Km[1, 1, 3] == pytest.approx(1 / 1.7, abs=TOLERANCE)
    for i in [4, 5, 6, 7]:
        assert Km[1, 1, i] == 0.0
    for i in range(8):
        assert Km[1, 2, i] == 0.0

    assert Vmax[0, 0] == pytest.approx(avg(1.1, 1.2), abs=TOLERANCE)
    assert Vmax[0, 1] == pytest.approx(avg(2.0, 1.3), abs=TOLERANCE)
    assert Vmax[0, 2] == pytest.approx(2.9, abs=TOLERANCE)

    assert Vmax[1, 0] == pytest.approx(avg(1.1, 2.1), abs=TOLERANCE)
    assert Vmax[1, 1] == pytest.approx(avg(1.9, 2.3), abs=TOLERANCE)
    assert Vmax[1, 2] == 0.0

    assert E[0, 0] == 10 - 15 + 10 + 10 - 5
    assert E[0, 1] == 10 - 10 - 5 + 10 + 10
    assert E[0, 2] == 10 + 10 - 5

    assert E[1, 0] == 15 - 10 - 5 + 10 + 10
    assert E[1, 1] == 10 - 10 + 5 - 10 - 10
    assert E[1, 2] == 0

    assert N[0, 0, 0] == -1
    assert N[0, 0, 1] == 2
    assert N[0, 0, 2] == 1
    assert N[0, 0, 3] == -1
    for i in [4, 5, 6, 7]:
        assert N[0, 0, i] == 0
    assert N[0, 1, 0] == 0
    assert N[0, 1, 1] == 0  # b is added and removed
    assert N[0, 1, 2] == 2
    assert N[0, 1, 3] == -1
    for i in [4, 5, 6, 7]:
        assert N[0, 1, i] == 0
    assert N[0, 2, 0] == 0
    assert N[0, 2, 1] == 2
    assert N[0, 2, 2] == 0
    assert N[0, 2, 3] == -1
    for i in [4, 5, 6, 7]:
        assert N[0, 2, i] == 0

    assert N[1, 0, 0] == 1
    assert N[1, 0, 1] == 0  # b is added and removed
    assert N[1, 0, 2] == 1
    assert N[1, 0, 3] == -1
    for i in [4, 5, 6, 7]:
        assert N[1, 0, i] == 0
    assert N[1, 1, 0] == 0
    assert N[1, 1, 1] == -2
    assert N[1, 1, 2] == 0  # c is added and removed
    assert N[1, 1, 3] == 1
    for i in [4, 5, 6, 7]:
        assert N[1, 1, i] == 0
    for i in range(8):
        assert N[1, 2, i] == 0

    # b was a reactant in the first domain,
    # but was then created again yielding N b=0
    # it thus has to be added as activating effector
    assert A[0, 1, 1] == 1
    assert A[1, 0, 1] == 1
    assert A.sum() == 2

    # test protein representation

    proteins = kinetics.get_proteome(proteome=c0)

    p0 = proteins[0]
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].substrates == [ma]
    assert p0.domains[0].products == [mb]
    assert p0.domains[0].vmax == pytest.approx(1.1, abs=TOLERANCE)
    assert p0.domains[0].km == pytest.approx(0.5, abs=TOLERANCE)
    assert isinstance(p0.domains[1], CatalyticDomain)
    assert p0.domains[1].substrates == [md]
    assert p0.domains[1].products == [mb, mc]
    assert p0.domains[1].vmax == pytest.approx(1.2, abs=TOLERANCE)
    assert p0.domains[1].km == pytest.approx(1.5, abs=TOLERANCE)

    p1 = proteins[1]
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].substrates == [mb]
    assert p1.domains[0].products == [mc]
    assert p1.domains[0].vmax == pytest.approx(2.0, abs=TOLERANCE)
    assert p1.domains[0].km == pytest.approx(0.9, abs=TOLERANCE)
    assert isinstance(p1.domains[1], CatalyticDomain)
    assert p1.domains[1].substrates == [md]
    assert p1.domains[1].products == [mb, mc]
    assert p1.domains[1].vmax == pytest.approx(1.3, abs=TOLERANCE)
    assert p1.domains[1].km == pytest.approx(1.2, abs=TOLERANCE)

    p2 = proteins[2]
    assert isinstance(p2.domains[0], CatalyticDomain)
    assert p2.domains[0].substrates == [md]
    assert p2.domains[0].products == [mb, mb]
    assert p2.domains[0].vmax == pytest.approx(2.9, abs=TOLERANCE)
    assert p2.domains[0].km == pytest.approx(2.9, abs=TOLERANCE)

    proteins = kinetics.get_proteome(proteome=c1)

    p0 = proteins[0]
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].substrates == [mb]
    assert p0.domains[0].products == [ma]
    assert p0.domains[0].vmax == pytest.approx(1.1, abs=TOLERANCE)
    assert p0.domains[0].km == pytest.approx(0.3, abs=TOLERANCE)
    assert isinstance(p0.domains[1], CatalyticDomain)
    assert p0.domains[1].substrates == [md]
    assert p0.domains[1].products == [mb, mc]
    assert p0.domains[1].vmax == pytest.approx(2.1, abs=TOLERANCE)
    assert p0.domains[1].km == pytest.approx(1.4, abs=TOLERANCE)

    p1 = proteins[1]
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].substrates == [mb]
    assert p1.domains[0].products == [mc]
    assert p1.domains[0].vmax == pytest.approx(1.9, abs=TOLERANCE)
    assert p1.domains[0].km == pytest.approx(0.3, abs=TOLERANCE)
    assert isinstance(p1.domains[1], CatalyticDomain)
    assert p1.domains[1].substrates == [mb, mc]
    assert p1.domains[1].products == [md]
    assert p1.domains[1].vmax == pytest.approx(2.3, abs=TOLERANCE)
    assert p1.domains[1].km == pytest.approx(1.7, abs=TOLERANCE)


def test_simple_mm_kinetic():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b, P1: b -> d
    # cell 1: P0: c -> d, P1: a -> d

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [1.1, 0.9, 2.9, 0.8],
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

    # affinities (c, p)
    Kmf = torch.tensor([
        [1.3, 2.1, 0.0],
        [1.0, 1.7, 0.0],
    ])
    Kmb = torch.tensor([
        [0.3, 1.1, 0.0],
        [1.5, 0.7, 0.0],
    ])
    Kmr = torch.zeros(2, 3)

    # max velocities (c, p)
    Vmax = torch.tensor([
        [2.1, 1.0, 0.0],
        [1.1, 2.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4)

    # fmt: on

    def mm(s, p, kf, kb, v):
        vf = v * (s / kf - p / kb) / (1 + s / kf + p / kb)
        vb = v * (p / kb - s / kf) / (1 + s / kf + p / kb)
        return (vf - vb) / 2

    # expected outcome
    v_c0_0 = mm(X0[0, 0], X0[0, 1], Kmf[0, 0], Kmb[0, 0], Vmax[0, 0])
    v_c0_1 = mm(X0[0, 1], X0[0, 3], Kmf[0, 1], Kmb[0, 1], Vmax[0, 1])
    dx_c0_a = -v_c0_0
    dx_c0_b = v_c0_0 - v_c0_1
    dx_c0_c = 0.0
    dx_c0_d = v_c0_1

    v_c1_0 = mm(X0[1, 2], X0[1, 3], Kmf[1, 0], Kmb[1, 0], Vmax[1, 0])
    v_c1_1 = mm(X0[1, 0], X0[1, 3], Kmf[1, 1], Kmb[1, 1], Vmax[1, 1])
    dx_c1_a = -v_c1_1
    dx_c1_b = 0.0
    dx_c1_c = -v_c1_0
    dx_c1_d = v_c1_0 + v_c1_1

    # test
    kinetics = get_kinetics()
    kinetics.N = N
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    assert (Xd[0, 0] - dx_c0_a).abs() < TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < TOLERANCE
    assert (Xd[0, 2] - dx_c0_c).abs() < TOLERANCE
    assert (Xd[0, 3] - dx_c0_d).abs() < TOLERANCE

    assert (Xd[1, 0] - dx_c1_a).abs() < TOLERANCE
    assert (Xd[1, 1] - dx_c1_b).abs() < TOLERANCE
    assert (Xd[1, 2] - dx_c1_c).abs() < TOLERANCE
    assert (Xd[1, 3] - dx_c1_d).abs() < TOLERANCE


def test_mm_kinetic_with_proportions():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> 2b, P1: 2c -> d
    # cell 1: P0: 3b -> 2c

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [1.1, 0.1, 2.9, 0.8],
        [1.2, 4.9, 5.1, 1.4],
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
    Kmf = torch.tensor([
        [1.3, 2.1, 0.0],
        [1.4, 0.0, 0.0],
    ])
    Kmb = torch.tensor([
        [0.3, 1.1, 0.0],
        [1.5, 0.0, 0.0],
    ])
    Kmr = torch.zeros(2, 3)
    
    # max velocities (c, p)
    Vmax = torch.tensor([
        [2.1, 1.1, 0.0],
        [1.9, 0.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4)

    # fmt: on

    def mm12(s, p, kf, kb, v):
        vf = v * (s / kf - p**2 / kb) / (1 + s / kf + p**2 / kb)
        vb = v * (p**2 / kb - s / kf) / (1 + s / kf + p**2 / kb)
        return (vf - vb) / 2

    def mm21(s, p, kf, kb, v):
        vf = v * (s**2 / kf - p / kb) / (1 + s**2 / kf + p / kb)
        vb = v * (p / kb - s**2 / kf) / (1 + s**2 / kf + p / kb)
        return (vf - vb) / 2

    def mm32(s, p, kf, kb, v):
        vf = v * (s**3 / kf - p**2 / kb) / (1 + s**3 / kf + p**2 / kb)
        vb = v * (p**2 / kb - s**3 / kf) / (1 + s**3 / kf + p**2 / kb)
        return (vf - vb) / 2

    # expected outcome
    v_c0_0 = mm12(X0[0, 0], X0[0, 1], Kmf[0, 0], Kmb[0, 0], Vmax[0, 0])
    v_c0_1 = mm21(X0[0, 2], X0[0, 3], Kmf[0, 1], Kmb[0, 1], Vmax[0, 1])
    dx_c0_a = -v_c0_0
    dx_c0_b = 2 * v_c0_0
    dx_c0_c = -2 * v_c0_1
    dx_c0_d = v_c0_1

    v_c1_0 = mm32(X0[1, 1], X0[1, 2], Kmf[1, 0], Kmb[1, 0], Vmax[1, 0])
    dx_c1_a = 0.0
    dx_c1_b = -3 * v_c1_0
    dx_c1_c = 2 * v_c1_0
    dx_c1_d = 0.0

    # test
    kinetics = get_kinetics()
    kinetics.N = N
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    assert (Xd[0, 0] - dx_c0_a).abs() < TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < TOLERANCE
    assert (Xd[0, 2] - dx_c0_c).abs() < TOLERANCE
    assert (Xd[0, 3] - dx_c0_d).abs() < TOLERANCE

    assert (Xd[1, 0] - dx_c1_a).abs() < TOLERANCE
    assert (Xd[1, 1] - dx_c1_b).abs() < TOLERANCE
    assert (Xd[1, 2] - dx_c1_c).abs() < TOLERANCE
    assert (Xd[1, 3] - dx_c1_d).abs() < TOLERANCE


def test_mm_kinetic_with_multiple_substrates():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a,b -> c, P1: b,d -> 2a,c
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
    Kmf = torch.tensor([
        [1.3, 2.1, 0.0],
        [1.4, 0.0, 0.0],
    ])
    Kmb = torch.tensor([
        [0.3, 1.1, 0.0],
        [1.5, 0.0, 0.0],
    ])
    Kmr = torch.zeros(2, 3)

    # max velocities (c, p)
    Vmax = torch.tensor([
        [2.1, 1.1, 0.0],
        [1.2, 0.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4)

    # fmt: on

    def mm111(s1, s2, p, kf, kb, v):
        vf = v * (s1 * s2 / kf - p / kb) / (1 + s1 * s2 / kf + p / kb)
        vb = v * (p / kb - s1 * s2 / kf) / (1 + s1 * s2 / kf + p / kb)
        return (vf - vb) / 2

    def mm1121(s1, s2, p1, p2, kf, kb, v):
        base = 1 + s1 * s2 / kf + p1**2 * p2 / kb
        vf = v * (s1 * s2 / kf - p1**2 * p2 / kb) / base
        vb = v * (p1**2 * p2 / kb - s1 * s2 / kf) / base
        return (vf - vb) / 2

    # expected outcome
    v_c0_0 = mm111(X0[0, 0], X0[0, 1], X0[0, 2], Kmf[0, 0], Kmb[0, 0], Vmax[0, 0])
    v_c0_1 = mm1121(
        X0[0, 1], X0[0, 3], X0[0, 0], X0[0, 2], Kmf[0, 1], Kmb[0, 1], Vmax[0, 1]
    )
    dx_c0_a = -v_c0_0 + 2 * v_c0_1
    dx_c0_b = -v_c0_0 + -v_c0_1
    dx_c0_c = v_c0_0 + v_c0_1
    dx_c0_d = -v_c0_1

    v_c1_0 = mm111(X0[1, 0], X0[1, 3], X0[1, 1], Kmf[1, 0], Kmb[1, 0], Vmax[1, 0])
    dx_c1_a = -v_c1_0
    dx_c1_b = v_c1_0
    dx_c1_c = 0.0
    dx_c1_d = -v_c1_0

    # test
    kinetics = get_kinetics()
    kinetics.N = N
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    assert (Xd[0, 0] - dx_c0_a).abs() < TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < TOLERANCE
    assert (Xd[0, 2] - dx_c0_c).abs() < TOLERANCE
    assert (Xd[0, 3] - dx_c0_d).abs() < TOLERANCE

    assert (Xd[1, 0] - dx_c1_a).abs() < TOLERANCE
    assert (Xd[1, 1] - dx_c1_b).abs() < TOLERANCE
    assert (Xd[1, 2] - dx_c1_c).abs() < TOLERANCE
    assert (Xd[1, 3] - dx_c1_d).abs() < TOLERANCE


def test_mm_kinetic_with_allosteric_action():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b, inhibitor=c, P1: c -> d, activator=a, P2: a -> b, inh=c, act=d
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
            [-1.0, 1.0, 0.0, 0.0]   ],
        [   [-1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]   ],
    ])

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
        [1.3, 2.1, 0.9],
        [1.4, 2.2, 0.0],
    ])

    # max velocities (c, p)
    Vmax = torch.tensor([
        [2.1, 2.0, 1.0],
        [3.2, 2.5, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.tensor([
        [   [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0]   ],
        [   [0.0, 0.0, -1.0, -1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]   ],
    ])

    # fmt: on

    def mm(s, p, kf, kb, v):
        vf = v * (s / kf - p / kb) / (1 + s / kf + p / kb)
        vb = v * (p / kb - s / kf) / (1 + s / kf + p / kb)
        return (vf - vb) / 2

    def fi(i, ki):
        return 1 - i / (ki + i)

    def fi2(i1, i2, ki):
        return max(0, 1 - i1 * i2 / (ki + i1 * i2))

    def fa(a, ka):
        return a / (ka + a)

    def fa2(a1, a2, ka):
        return min(1, a1 * a2 / (ka + a1 * a2))

    # expected outcome
    v_c0_0 = mm(X0[0, 0], X0[0, 1], Kmf[0, 0], Kmb[0, 0], Vmax[0, 0]) * fi(
        X0[0, 2], Kmr[0, 0]
    )
    v_c0_1 = mm(X0[0, 2], X0[0, 3], Kmf[0, 1], Kmb[0, 1], Vmax[0, 1]) * fa(
        X0[0, 0], Kmr[0, 1]
    )
    v_c0_2 = (
        mm(X0[0, 0], X0[0, 1], Kmf[0, 2], Kmb[0, 2], Vmax[0, 2])
        * fi(X0[0, 2], Kmr[0, 2])
        * fa(X0[0, 3], Kmr[0, 2])
    )
    dx_c0_a = -v_c0_0 - v_c0_2
    dx_c0_b = v_c0_0 + v_c0_2
    dx_c0_c = -v_c0_1
    dx_c0_d = v_c0_1

    i_c1_0 = fi2(X0[1, 2], X0[1, 3], Kmr[1, 0])
    v_c1_0 = mm(X0[1, 0], X0[1, 1], Kmf[1, 0], Kmb[1, 0], Vmax[1, 0]) * i_c1_0
    v_c1_1 = mm(X0[1, 2], X0[1, 3], Kmf[1, 1], Kmb[1, 1], Vmax[1, 1]) * fa2(
        X0[1, 0], X0[1, 1], Kmr[1, 1]
    )
    dx_c1_a = -v_c1_0
    dx_c1_b = v_c1_0
    dx_c1_c = -v_c1_1
    dx_c1_d = v_c1_1

    # test
    kinetics = get_kinetics()
    kinetics.N = N
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    assert (Xd[0, 0] - dx_c0_a).abs() < TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < TOLERANCE
    assert (Xd[0, 2] - dx_c0_c).abs() < TOLERANCE
    assert (Xd[0, 3] - dx_c0_d).abs() < TOLERANCE

    assert (Xd[1, 0] - dx_c1_a).abs() < TOLERANCE
    assert (Xd[1, 1] - dx_c1_b).abs() < TOLERANCE
    assert (Xd[1, 2] - dx_c1_c).abs() < TOLERANCE
    assert (Xd[1, 3] - dx_c1_d).abs() < TOLERANCE


def test_reduce_velocity_to_avoid_negative_concentrations():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b, P1: b -> d
    # cell 1: P0: 2c -> d

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [0.1, 1.0, 2.9, 0.8],
        [2.9, 3.1, 0.1, 0.3],
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
    Kmf = torch.tensor([
        [0.1, 2.1, 0.0],
        [0.1, 0.0, 0.0],
    ])
    Kmb = torch.tensor([
        [10.3, 1.1, 0.0],
        [10.5, 0.0, 0.0],
    ])
    Kmr = torch.zeros(2, 3)

    # max velocities (c, p)
    Vmax = torch.tensor([
        [2.1, 1.0, 0.0],
        [3.1, 0.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4)

    # fmt: on

    def mm(s, p, kf, kb, v):
        vf = v * (s / kf - p / kb) / (1 + s / kf + p / kb)
        vb = v * (p / kb - s / kf) / (1 + s / kf + p / kb)
        return (vf - vb) / 2

    def mm21(s, p, kf, kb, v):
        vf = v * (s**2 / kf - p / kb) / (1 + s**2 / kf + p / kb)
        vb = v * (p / kb - s**2 / kf) / (1 + s**2 / kf + p / kb)
        return (vf - vb) / 2

    # expected outcome
    v_c0_0 = mm(X0[0, 0], X0[0, 1], Kmf[0, 0], Kmb[0, 0], Vmax[0, 0])
    v_c0_1 = mm(X0[0, 1], X0[0, 3], Kmf[0, 1], Kmb[0, 1], Vmax[0, 1])
    # but this would lead to Xd + X0 = -0.0615 (for a)
    assert X0[0, 0] - v_c0_0 < 0.0
    # so velocity should be reduced by a factor depending on current a
    # all other proteins in this cell are reduce by the same factor
    # to avoid follow up problems with other molecules being destroyed by other proteins
    f = (X0[0, 0] - EPS) / v_c0_0
    v_c0_0 = f * v_c0_0
    v_c0_1 = f * v_c0_1
    dx_c0_a = -v_c0_0
    dx_c0_b = v_c0_0 - v_c0_1
    dx_c0_c = 0.0
    dx_c0_d = v_c0_1

    v_c1_0 = mm21(X0[1, 2], X0[1, 3], Kmf[1, 0], Kmb[1, 0], Vmax[1, 0])
    # but this would lead to Xd + X0 = -0.0722 (for c)
    assert X0[1, 2] - 2 * v_c1_0 < 0.0
    # as above, velocities are reduced. here its only this protein
    v_c1_0 = (X0[1, 2] - EPS) / 2
    dx_c1_a = 0.0
    dx_c1_b = 0.0
    dx_c1_c = -2 * v_c1_0
    dx_c1_d = v_c1_0

    # test
    kinetics = get_kinetics()
    kinetics.N = N
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    assert (Xd[0, 0] - dx_c0_a).abs() < TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < TOLERANCE
    assert (Xd[0, 2] - dx_c0_c).abs() < TOLERANCE
    assert (Xd[0, 3] - dx_c0_d).abs() < TOLERANCE

    assert (Xd[1, 0] - dx_c1_a).abs() < TOLERANCE
    assert (Xd[1, 1] - dx_c1_b).abs() < TOLERANCE
    assert (Xd[1, 2] - dx_c1_c).abs() < TOLERANCE
    assert (Xd[1, 3] - dx_c1_d).abs() < TOLERANCE

    X1 = X0 + Xd
    assert not torch.any(X1 < 0.0)


def test_reduce_velocity_in_multiple_proteins():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b, P1: 2a -> d
    # cell 1: P0: a -> b

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [2.0, 1.2, 2.9, 1.5],
        [2.9, 3.1, 0.1, 1.0],
    ])

    # reactions (c, p, s)
    N = torch.tensor([
        [   [-1.0, 1.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0]    ],
        [   [-1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])

    # affinities (c, p, s)
    Kmf = torch.tensor([
        [0.1, 2.1, 0.0],
        [0.1, 0.0, 0.0],
    ])
    Kmb = torch.tensor([
        [10.3, 1.1, 0.0],
        [1.5, 0.0, 0.0],
    ])
    Kmr = torch.zeros(2, 3)

    # max velocities (c, p)
    Vmax = torch.tensor([
        [3.1, 2.0, 0.0],
        [3.1, 0.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4)

    # reaction energies (c, p)
    torch.full((2, 3), -99999.9)

    # fmt: on

    def mm(s, p, kf, kb, v):
        vf = v * (s / kf - p / kb) / (1 + s / kf + p / kb)
        vb = v * (p / kb - s / kf) / (1 + s / kf + p / kb)
        return (vf - vb) / 2

    def mm21(s, p, kf, kb, v):
        vf = v * (s**2 / kf - p / kb) / (1 + s**2 / kf + p / kb)
        vb = v * (p / kb - s**2 / kf) / (1 + s**2 / kf + p / kb)
        return (vf - vb) / 2

    # expected outcome
    v_c0_0 = mm(X0[0, 0], X0[0, 1], Kmf[0, 0], Kmb[0, 0], Vmax[0, 0])
    v_c0_1 = mm21(X0[0, 0], X0[0, 3], Kmf[0, 1], Kmb[0, 1], Vmax[0, 1])
    # but this would lead to a < 0.0
    naive_dx_c0_a = -v_c0_0 - 2 * v_c0_1
    assert X0[0, 0] + naive_dx_c0_a < 0.0
    # so velocity should be reduced to by a factor to not deconstruct too much a
    # all other proteins have to be reduced by the same factor to not cause downstream problems
    f = (X0[0, 0] - EPS) / -naive_dx_c0_a
    v_c0_0 = v_c0_0 * f
    v_c0_1 = v_c0_1 * f
    dx_c0_a = -v_c0_0 - v_c0_1 * 2
    dx_c0_b = v_c0_0
    dx_c0_c = 0.0
    dx_c0_d = v_c0_1

    # cell1 is business as usual
    v_c1_0 = mm(X0[1, 0], X0[1, 1], Kmf[1, 0], Kmb[1, 0], Vmax[1, 0])
    dx_c1_a = -v_c1_0
    dx_c1_b = v_c1_0
    dx_c1_c = 0.0
    dx_c1_d = 0.0

    # test
    kinetics = get_kinetics()
    kinetics.N = N
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    assert (Xd[0, 0] - dx_c0_a).abs() < TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < TOLERANCE
    assert (Xd[0, 2] - dx_c0_c).abs() < TOLERANCE
    assert (Xd[0, 3] - dx_c0_d).abs() < TOLERANCE

    assert (Xd[1, 0] - dx_c1_a).abs() < TOLERANCE
    assert (Xd[1, 1] - dx_c1_b).abs() < TOLERANCE
    assert (Xd[1, 2] - dx_c1_c).abs() < TOLERANCE
    assert (Xd[1, 3] - dx_c1_d).abs() < TOLERANCE

    X1 = X0 + Xd
    assert not torch.any(X1 < 0.0)


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

    # affinities (c, p, s)
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
    kinetics = get_kinetics()
    kinetics.n_computations = 9
    kinetics.N = N
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A

    for _ in range(10):
        X0 = kinetics.integrate_signals(X=X0)
        print(X0[1, 2] / (X0[1, 0] * X0[1, 1] * X0[1, 1]))

    q_c0_0 = X0[0, 1] / X0[0, 0]
    q_c0_1 = X0[0, 3] / X0[0, 2]
    q_c1_0 = X0[1, 2] / (X0[1, 0] * X0[1, 1] * X0[1, 1])

    assert (q_c0_0 - 1.0).abs() < 0.1
    assert (q_c0_1 - 20.0).abs() < 0.1
    assert (q_c1_0 - 10.0).abs() < 0.1


def test_zero_substrates_stay_zero():
    # Typical reasons for cells creating signals from 0:
    # 1. when using exp(ln(x)) 1s can accidentally be created
    # 2. when doing a^b 1s can be created (0^1=0 but 0^0=1)

    n_cells = 100
    n_prots = 100
    n_steps = 100

    kinetics = get_kinetics()
    n_mols = len(MOLECULES) * 2

    # concentrations (c, s)
    X = torch.zeros(n_cells, n_mols).abs()

    # reactions (c, p, s)
    kinetics.N = torch.randint(low=-3, high=4, size=(n_cells, n_prots, n_mols)).float()

    # max velocities (c, p)
    kinetics.Vmax = torch.randn(n_cells, n_prots).abs() * 100

    # allosterics (c, p, s)
    kinetics.A = torch.randint(low=-2, high=3, size=(n_cells, n_prots, n_mols))

    # reaction energies (c, p)
    kinetics.Ke = torch.randn(n_cells, n_prots)

    # affinities (c, p)
    kinetics.Kmf = torch.randn(n_cells, n_prots).abs()
    kinetics.Kmb = kinetics.Kmf * kinetics.Ke
    kinetics.Kmr = kinetics.Kmf

    # test
    for _ in range(n_steps):
        X = kinetics.integrate_signals(X=X)
        assert X.min() == 0.0
        assert X.max() == 0.0


def test_substrate_concentrations_are_always_finite_and_positive():
    # interaction terms can overflow float32 when they become too big
    # i.g. exponents too high and many substrates
    # this will create infinites which will eventually become NaN
    n_cells = 100
    n_prots = 100
    n_steps = 100

    kinetics = get_kinetics()
    n_mols = len(MOLECULES) * 2

    # concentrations (c, s)
    X = torch.randn(n_cells, n_mols).abs()

    # reactions (c, p, s)
    kinetics.N = torch.randint(low=-3, high=4, size=(n_cells, n_prots, n_mols)).float()

    # max velocities (c, p)
    kinetics.Vmax = torch.randn(n_cells, n_prots).abs() * 100

    # allosterics (c, p, s)
    kinetics.A = torch.randint(low=-2, high=3, size=(n_cells, n_prots, n_mols))

    # reaction energies (c, p)
    kinetics.Ke = torch.randn(n_cells, n_prots)

    # affinities (c, p)
    kinetics.Kmf = torch.randn(n_cells, n_prots).abs().clamp(0.001)
    kinetics.Kmb = (kinetics.Kmf * kinetics.Ke).clamp(0.001)
    kinetics.Kmr = kinetics.Kmf

    # test
    for _ in range(n_steps):
        X = kinetics.integrate_signals(X=X)
        assert not torch.any(X < 0.0), X[X < 0.0].min()
        assert not torch.any(X.isnan()), X.isnan().sum()
        assert torch.all(X.isfinite()), ~X.isfinite().sum()
