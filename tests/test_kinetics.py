import pytest
import torch
from magicsoup.containers import (
    Molecule,
    CatalyticDomain,
    RegulatoryDomain,
    TransporterDomain,
)
from magicsoup.kinetics import Kinetics

TOLERANCE = 1e-4


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
    kinetics = Kinetics(molecules=MOLECULES, reactions=REACTIONS)
    kinetics.km_map.weights = KM_WEIGHTS.clone()
    kinetics.vmax_map.weights = VMAX_WEIGHTS.clone()
    kinetics.sign_map.signs = SIGNS.clone()
    kinetics.transport_map.M = TRANSPORT_M.clone()
    kinetics.effector_map.M = EFFECTOR_M.clone()
    kinetics.reaction_map.M = REACTION_M.clone()
    return kinetics


def avg(*x):
    return sum(x) / len(x)


def test_cell_params_with_catalytic_domains_and_co_factors():
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


def test_cell_params_with_transporter_domains():
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


def test_cell_params_with_regulatory_domains():
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


def test_cell_params_with_catalytic_domains():
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

    # reaction energies (c, p)
    E = torch.full((2, 3), -99999.9)

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
    kinetics = get_kinetics()
    kinetics.N = N
    kinetics.Km = Km
    kinetics.Vmax = Vmax
    kinetics.E = E
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    assert (Xd[0, 0] - dx_c0_a).abs() < TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < TOLERANCE
    assert Xd[0, 2] == 0.0
    assert (Xd[0, 3] - dx_c0_d).abs() < TOLERANCE

    assert (Xd[1, 0] - dx_c1_a).abs() < TOLERANCE
    assert Xd[1, 1] == 0.0
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

    # reaction energies (c, p)
    E = torch.full((2, 3), -99999.9)

    # fmt: on

    def mm(x, k, v, n):
        return v * x**n / (k + x) ** n

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
    kinetics = get_kinetics()
    kinetics.N = N
    kinetics.Km = Km
    kinetics.Vmax = Vmax
    kinetics.E = E
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

    # reaction energies (c, p)
    E = torch.full((2, 3), -99999.9)

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
    kinetics = get_kinetics()
    kinetics.N = N
    kinetics.Km = Km
    kinetics.Vmax = Vmax
    kinetics.E = E
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
    # cell 0: P0: a -> b, inhibitor=c, P1: c -> d, activator=a, P3: a -> b, inh=c, act=d
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
    Km = torch.tensor([
        [   [1.2, 3.1, 0.7, 0.0],
            [1.0, 0.0, 0.9, 1.2],
            [1.0, 0.1, 1.0, 1.0] ],
        [   [2.1, 1.3, 1.0, 0.6],
            [1.3, 0.8, 0.3, 1.1],
            [0.0, 0.0, 0.0, 0.0] ],
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

    # reaction energies (c, p)
    E = torch.full((2, 3), -99999.9)

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
    v2_c0 = (
        mm(x=X0[0, 0], v=Vmax[0, 2], kx=Km[0, 2, 0])
        * fi(i=X0[0, 2], ki=Km[0, 2, 2])
        * fa(a=X0[0, 3], ka=Km[0, 2, 3])
    )
    dx_c0_b = v0_c0 + v2_c0
    dx_c0_a = -v0_c0 - v2_c0
    dx_c0_c = -v1_c0
    dx_c0_d = v1_c0

    v0_c1 = mm(x=X0[1, 0], v=Vmax[1, 0], kx=Km[1, 0, 0]) * fi2(
        i1=X0[1, 2], ki1=Km[1, 0, 2], i2=X0[1, 3], ki2=Km[1, 0, 3]
    )
    v1_c1 = mm(x=X0[1, 2], v=Vmax[1, 1], kx=Km[1, 1, 2]) * fa2(
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
    kinetics = get_kinetics()
    kinetics.N = N
    kinetics.Km = Km
    kinetics.Vmax = Vmax
    kinetics.E = E
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

    # reaction energies (c, p)
    E = torch.full((2, 3), -99999.9)

    # fmt: on

    def mm(x, k, v, n=1):
        return v * x**n / (k + x) ** n

    # expected outcome
    v0_c0 = mm(x=X0[0, 0], v=Vmax[0, 0], k=Km[0, 0, 0])
    v1_c0 = mm(x=X0[0, 1], v=Vmax[0, 1], k=Km[0, 1, 1])
    # but this would lead to Xd + X0 = -0.0615 (for a)
    assert X0[0, 0] - v0_c0 < 0.0
    # so velocity should be reduced by a factor depending on current a
    # all other proteins in this cell are reduce by the same factor
    # to avoid follow up problems with other molecules being destroyed by other proteins
    f = X0[0, 0] * 0.99 / v0_c0
    v0_c0 = f * v0_c0
    v1_c1 = f * v1_c0
    dx_c0_a = -v0_c0
    dx_c0_b = v0_c0 - v1_c1
    dx_c0_c = 0.0
    dx_c0_d = v1_c1

    v0_c1 = mm(x=X0[1, 2], v=Vmax[1, 0], k=Km[1, 0, 2], n=2)
    # but this would lead to Xd + X0 = -0.0722 (for c)
    assert X0[1, 2] - 2 * v0_c1 < 0.0
    # as above, velocities are reduced. here its only this protein
    v0_c1 = X0[1, 2] * 0.99 / 2
    dx_c1_a = 0.0
    dx_c1_b = 0.0
    dx_c1_c = -2 * v0_c1
    dx_c1_d = v0_c1

    # test
    kinetics = get_kinetics()
    kinetics.N = N
    kinetics.Km = Km
    kinetics.Vmax = Vmax
    kinetics.E = E
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
    Km = torch.tensor([
        [   [0.5, 1.5, 0.0, 0.0],
            [0.5, 0.0, 0.0, 3.5],
            [0.0, 0.0, 0.0, 0.0] ],
        [   [0.5, 1.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0] ],
    ])

    # max velocities (c, p)
    Vmax = torch.tensor([
        [3.1, 2.0, 0.0],
        [3.1, 0.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4)

    # reaction energies (c, p)
    E = torch.full((2, 3), -99999.9)

    # fmt: on

    def mm(x, k, v, n=1):
        return v * x**n / (k + x) ** n

    # expected outcome
    v0_c0 = mm(x=X0[0, 0], v=Vmax[0, 0], k=Km[0, 0, 0])
    v1_c0 = mm(x=X0[0, 0], v=Vmax[0, 1], k=Km[0, 1, 0], n=2)
    # but this would lead to a < 0.0
    naive_dx_c0_a = -v0_c0 - 2 * v1_c0
    assert X0[0, 0] + naive_dx_c0_a < 0.0
    # so velocity should be reduced to by a factor to not deconstruct too much a
    # all other proteins have to be reduced by the same factor to not cause downstream problems
    f = X0[0, 0] * 0.99 / -naive_dx_c0_a
    v0_c0 = v0_c0 * f
    v1_c0 = v1_c0 * f
    dx_c0_a = -v0_c0 - v1_c0 * 2
    dx_c0_b = v0_c0
    dx_c0_c = 0.0
    dx_c0_d = v1_c0

    # cell1 is business as usual
    v0_c0 = mm(x=X0[1, 0], v=Vmax[1, 0], k=Km[1, 0, 0])
    dx_c1_a = -v0_c0
    dx_c1_b = v0_c0
    dx_c1_c = 0.0
    dx_c1_d = 0.0

    # test
    kinetics = get_kinetics()
    kinetics.N = N
    kinetics.Km = Km
    kinetics.Vmax = Vmax
    kinetics.E = E
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


def test_equilibrium_constants():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # cell 0: P0: a -> b (but b/a > Ke), P1: b -> d (but Q almost Ke)
    # cell 1: P0: c -> d (but Q ~= Ke), P1: a -> d (but d/a > Ke, and Q almost Ke)

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [2.0, 20.0, 2.9, 20.0],
        [1.0, 3.1, 1.3, 2.9],
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
        [2.1, 4.0, 0.0],
        [1.1, 3.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4)

    # reaction energies (c, p)
    E = torch.full((2, 3), -99999.9)
    # E = -2000 ^= # ln(Ke) = 0.77
    E[0, 0] = -2000 # ln(Q) - ln(Ke) = 1.53 (switch N)
    E[0, 1] = -2000 # ln(Q) - ln(Ke) = -0.76 (reduce V)
    E[1, 0] = -2000 # ln(Q) - ln(Ke) = 0.02 (switch off)
    E[1, 1] = -2000 # ln(Q) - ln(Ke) = 0.28 (switch N, reduce V)

    # fmt: on

    def mm(x, k, v):
        return v * x / (k + x)

    # expected outcome
    # v0 is turned around, v1 is reduced by a factor of around 0.37
    v0_c0 = mm(x=X0[0, 1], v=Vmax[0, 0], k=Km[0, 0, 1]) * -1.0
    v1_c0 = mm(x=X0[0, 1], v=Vmax[0, 1], k=Km[0, 1, 1]) * 0.76
    dx_c0_a = -v0_c0
    dx_c0_b = v0_c0 - v1_c0
    dx_c0_c = 0.0
    dx_c0_d = v1_c0

    # v0 is switched off, v1 is reduced by 0.28 and turned around
    v0_c1 = 0.0
    v1_c1 = mm(x=X0[1, 3], v=Vmax[1, 1], k=Km[1, 1, 3]) * -0.28
    dx_c1_a = -v1_c1
    dx_c1_b = 0.0
    dx_c1_c = -v0_c1
    dx_c1_d = v0_c1 + v1_c1

    # test
    kinetics = get_kinetics()
    kinetics.N = N
    kinetics.Km = Km
    kinetics.Vmax = Vmax
    kinetics.E = E
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    tolerance = 1e-1  # floating point problems
    assert (Xd[0, 0] - dx_c0_a).abs() < TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < tolerance
    assert (Xd[0, 2] - dx_c0_c).abs() < TOLERANCE
    assert (Xd[0, 3] - dx_c0_d).abs() < tolerance

    assert (Xd[1, 0] - dx_c1_a).abs() < tolerance
    assert (Xd[1, 1] - dx_c1_b).abs() < TOLERANCE
    assert (Xd[1, 2] - dx_c1_c).abs() < TOLERANCE
    assert (Xd[1, 3] - dx_c1_d).abs() < tolerance


@pytest.mark.parametrize("gen", [torch.zeros, torch.randn])
def test_substrate_concentrations_are_always_finite_and_positive(gen):
    n_cells = 100
    n_prots = 100
    n_steps = 100

    kinetics = get_kinetics()
    n_mols = len(MOLECULES) * 2

    # concentrations (c, s)
    X = gen(n_cells, n_mols).abs()

    # reactions (c, p, s)
    kinetics.N = torch.randint(low=-3, high=4, size=(n_cells, n_prots, n_mols))

    # affinities (c, p, s)
    kinetics.Km = torch.randn(n_cells, n_prots, n_mols).abs()

    # max velocities (c, p)
    kinetics.Vmax = torch.randn(n_cells, n_prots).abs() * 10

    # allosterics (c, p, s)
    kinetics.A = torch.randint(low=-2, high=3, size=(n_cells, n_prots, n_mols))

    # reaction energies (c, p)
    kinetics.E = torch.randn(n_cells, n_prots)

    # test
    for _ in range(n_steps):
        Xd = kinetics.integrate_signals(X=X)
        X = X + Xd
        assert not torch.any(X < 0.0), X[X < 0.0].min()
        assert not torch.any(X.isnan()), X.isnan().sum()
        assert torch.all(X.isfinite()), ~X.isfinite().sum()
