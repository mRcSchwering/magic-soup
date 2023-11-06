import pytest
import math
import torch
from magicsoup.containers import (
    Molecule,
    CatalyticDomain,
    RegulatoryDomain,
    TransporterDomain,
)
from magicsoup.constants import GAS_CONSTANT
from magicsoup.kinetics import Kinetics

_TOLERANCE = 1e-4
_EPS = 1e-8


_ma = Molecule("a", energy=15 * 1e3)
_mb = Molecule("b", energy=10 * 1e3)
_mc = Molecule("c", energy=10 * 1e3)
_md = Molecule("d", energy=5 * 1e3)
_MOLECULES = [_ma, _mb, _mc, _md]

_r_a_b = ([_ma], [_mb])
_r_b_c = ([_mb], [_mc])
_r_bc_d = ([_mb, _mc], [_md])
_r_d_bb = ([_md], [_mb, _mb])
_REACTIONS = [_r_a_b, _r_b_c, _r_bc_d, _r_d_bb]

# fmt: off
_KM_WEIGHTS = torch.tensor([
    torch.nan, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  # idxs 0-9
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,  # idxs 10-19
    2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9  # idxs 20-29
])

_VMAX_WEIGHTS = torch.tensor([
    torch.nan, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,  # idxs 0-9
    2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,  # idxs 10-19
])

_SIGNS = torch.tensor([0.0, 1.0, -1.0])  # idxs 0-2

_TRANSPORT_M = torch.tensor([
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

_EFFECTOR_M = torch.tensor([
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

_REACTION_M = torch.tensor([
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


def _get_kinetics(n_computations=1) -> Kinetics:
    kinetics = Kinetics(
        molecules=_MOLECULES,
        reactions=_REACTIONS,
        n_computations=n_computations,
        abs_temp=310,
    )
    kinetics.km_map.weights = _KM_WEIGHTS.clone()
    kinetics.vmax_map.weights = _VMAX_WEIGHTS.clone()
    kinetics.sign_map.signs = _SIGNS.clone()
    kinetics.transport_map.M = _TRANSPORT_M.clone()
    kinetics.effector_map.M = _EFFECTOR_M.clone()
    kinetics.reaction_map.M = _REACTION_M.clone()
    return kinetics


def _ke(subs: list[Molecule], prods: list[Molecule]):
    e = sum(d.energy for d in prods) - sum(d.energy for d in subs)
    return math.exp(-e / 310 / GAS_CONSTANT)


def _avg(*x):
    return sum(x) / len(x)


def test_cell_params_with_transporter_domains():
    # Protein: (domains, cds_start, cds_end, is_fwd)
    # Domain: (domain_spec, dom_start, dom_end)
    # Domain spec indexes: (dom_types, reacts_trnspts_effctrs, Vmaxs, Kms, signs)
    # fmt: off
    c0 = [
        (
            [
                (
                    (2, 5, 5, 1, 1),  # transporter, Vmax 1.5, Km 0.5, fwd, mol a
                    6, 27
                )
            ],
            13, 27, True
        )
        ,
        (
            [
                (
                    (2, 5, 5, 1, 1), # transporter, Vmax 1.5, Km 0.5, fwd, mol a
                    5, 13
                ),
                (
                    (2, 1, 2, 2, 1),  # transporter, Vmax 1.1, Km 0.2, bwd, mol a
                    7, 12
                )
            ],
            36, 74, False
        ),
    ]
    c1 = [
        (
            [
                (
                    (2, 5, 4, 1, 1), # transporter, Vmax 1.5, Km 0.4, fwd, mol a
                    1, 10
                ),
                (
                    (2, 4, 5, 1, 1), # transporter, Vmax 1.4, Km 0.5, fwd, mol a
                    2, 20
                ),
                (
                    (2, 3, 6, 1, 2), # transporter, Vmax 1.3, Km 0.6, fwd, mol b
                    3, 30
                ),
                (
                    (2, 2, 7, 1, 3),  # transporter, Vmax 1.2, Km 0.7, fwd, mol c
                    4, 40
                )
            ],
            91, 112, False
        ),
        (
            [
                (
                    (1, 10, 5, 1, 1), # catal, Vmax 2.0, Km 0.5, fwd, a->b
                    5, 50
                ),
                (
                    (2, 5, 5, 1, 1),   # transporter, Vmax 1.5, Km 0.5, fwd, mol a
                    6, 60
                )
            ],
            1, 10, False
        ),
    ]
    # fmt: on

    Kmf = torch.zeros(2, 3)
    Kmb = torch.zeros(2, 3)
    Kmr = torch.zeros(2, 3)
    Vmax = torch.zeros(2, 3)
    N = torch.zeros(2, 3, 8)
    Nf = torch.zeros(2, 3, 8)
    Nb = torch.zeros(2, 3, 8)
    A = torch.zeros(2, 3, 8)

    # test
    kinetics = _get_kinetics()
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.N = N
    kinetics.Nf = Nf
    kinetics.Nb = Nb
    kinetics.A = A
    proteomes = [[d[0] for d in c0], [d[0] for d in c1]]
    kinetics.set_cell_params(cell_idxs=[0, 1], proteomes=proteomes)

    assert Kmf[0, 0] == pytest.approx(0.5, abs=_TOLERANCE)
    assert Kmb[0, 0] == pytest.approx(0.5, abs=_TOLERANCE)
    assert Kmf[0, 1] == pytest.approx(_avg(0.5, 0.2), abs=_TOLERANCE)
    assert Kmb[0, 1] == pytest.approx(_avg(0.5, 0.2), abs=_TOLERANCE)
    assert Kmf[0, 2] == pytest.approx(0.0, _TOLERANCE)
    assert Kmb[0, 2] == pytest.approx(0.0, _TOLERANCE)

    ke_c1_1 = _ke([_ma], [_mb])
    assert Kmf[1, 0] == pytest.approx(_avg(0.4, 0.5, 0.6, 0.7), abs=_TOLERANCE)
    assert Kmb[1, 0] == pytest.approx(_avg(0.4, 0.5, 0.6, 0.7), abs=_TOLERANCE)
    assert Kmf[1, 1] == pytest.approx(_avg(0.5, 0.5), abs=_TOLERANCE)
    assert Kmb[1, 1] == pytest.approx(_avg(0.5, 0.5) * ke_c1_1, abs=_TOLERANCE)
    assert Kmf[1, 2] == pytest.approx(0.0, _TOLERANCE)
    assert Kmb[1, 2] == pytest.approx(0.0, _TOLERANCE)

    assert (Kmr < _TOLERANCE).all()

    assert Vmax[0, 0] == pytest.approx(1.5, abs=_TOLERANCE)
    assert Vmax[0, 1] == pytest.approx(_avg(1.5, 1.1), abs=_TOLERANCE)
    assert Vmax[0, 2] == 0.0

    assert Vmax[1, 0] == pytest.approx(_avg(1.5, 1.4, 1.3, 1.2), abs=_TOLERANCE)
    assert Vmax[1, 1] == pytest.approx(_avg(2.0, 1.5), abs=_TOLERANCE)
    assert Vmax[1, 2] == 0.0

    assert N[0, 0, 0] == -1
    assert N[0, 0, 4] == 1
    assert (N[0, 0, [1, 2, 3, 5, 6, 7]] == 0).all()
    assert Nf[0, 0, 0] == 1
    assert (Nf[0, 0, [1, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert Nb[0, 1, 4] == 1
    assert (Nb[0, 0, [0, 1, 2, 3, 5, 6, 7]] == 0).all()

    assert N[1, 0, 0] == -2
    assert N[1, 0, 1] == -1
    assert N[1, 0, 2] == -1
    assert N[1, 0, 4] == 2
    assert N[1, 0, 5] == 1
    assert N[1, 0, 6] == 1
    assert (N[1, 0, [3, 7]] == 0).all()
    assert Nf[1, 0, 0] == 2
    assert Nf[1, 0, 1] == 1
    assert Nf[1, 0, 2] == 1
    assert (Nf[1, 0, [4, 5, 6, 3, 7]] == 0).all()
    assert Nb[1, 0, 4] == 2
    assert Nb[1, 0, 5] == 1
    assert Nb[1, 0, 6] == 1
    assert (Nb[1, 0, [0, 1, 2, 3, 7]] == 0).all()
    assert N[1, 1, 0] == -2
    assert N[1, 1, 1] == 1
    assert N[1, 1, 4] == 1
    assert (N[1, 1, [2, 3, 5, 6, 7]] == 0).all()
    assert Nf[1, 1, 0] == 2
    assert (Nf[1, 1, [1, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert Nb[1, 1, 1] == 1
    assert Nb[1, 1, 4] == 1
    assert (Nb[1, 1, [0, 2, 3, 5, 6, 7]] == 0).all()

    assert (A == 0).all()

    # test proteome representation

    proteins = kinetics.get_proteome(proteome=c0)

    p0 = proteins[0]
    assert p0.cds_start == 13
    assert p0.cds_end == 27
    assert p0.is_fwd is True
    assert isinstance(p0.domains[0], TransporterDomain)
    assert p0.domains[0].molecule is _ma
    assert p0.domains[0].vmax == pytest.approx(1.5, abs=_TOLERANCE)
    assert p0.domains[0].km == pytest.approx(0.5, abs=_TOLERANCE)
    assert p0.domains[0].start == 6
    assert p0.domains[0].end == 27

    p1 = proteins[1]
    assert p1.cds_start == 36
    assert p1.cds_end == 74
    assert p1.is_fwd is False
    assert isinstance(p1.domains[0], TransporterDomain)
    assert p1.domains[0].molecule is _ma
    assert p1.domains[0].vmax == pytest.approx(1.5, abs=_TOLERANCE)
    assert p1.domains[0].km == pytest.approx(0.5, abs=_TOLERANCE)
    assert p1.domains[0].start == 5
    assert p1.domains[0].end == 13
    assert isinstance(p1.domains[1], TransporterDomain)
    assert p1.domains[1].molecule is _ma
    assert p1.domains[1].vmax == pytest.approx(1.1, abs=_TOLERANCE)
    assert p1.domains[1].km == pytest.approx(0.2, abs=_TOLERANCE)
    assert p1.domains[1].start == 7
    assert p1.domains[1].end == 12

    proteins = kinetics.get_proteome(proteome=c1)

    p0 = proteins[0]
    assert p0.cds_start == 91
    assert p0.cds_end == 112
    assert p0.is_fwd is False
    assert isinstance(p0.domains[0], TransporterDomain)
    assert p0.domains[0].molecule is _ma
    assert p0.domains[0].vmax == pytest.approx(1.5, abs=_TOLERANCE)
    assert p0.domains[0].km == pytest.approx(0.4, abs=_TOLERANCE)
    assert p0.domains[0].start == 1
    assert p0.domains[0].end == 10
    assert isinstance(p0.domains[1], TransporterDomain)
    assert p0.domains[1].molecule is _ma
    assert p0.domains[1].vmax == pytest.approx(1.4, abs=_TOLERANCE)
    assert p0.domains[1].km == pytest.approx(0.5, abs=_TOLERANCE)
    assert p0.domains[1].start == 2
    assert p0.domains[1].end == 20
    assert isinstance(p0.domains[2], TransporterDomain)
    assert p0.domains[2].molecule is _mb
    assert p0.domains[2].vmax == pytest.approx(1.3, abs=_TOLERANCE)
    assert p0.domains[2].km == pytest.approx(0.6, abs=_TOLERANCE)
    assert p0.domains[2].start == 3
    assert p0.domains[2].end == 30
    assert isinstance(p0.domains[3], TransporterDomain)
    assert p0.domains[3].molecule is _mc
    assert p0.domains[3].vmax == pytest.approx(1.2, abs=_TOLERANCE)
    assert p0.domains[3].km == pytest.approx(0.7, abs=_TOLERANCE)
    assert p0.domains[3].start == 4
    assert p0.domains[3].end == 40

    p1 = proteins[1]
    assert p1.cds_start == 1
    assert p1.cds_end == 10
    assert p1.is_fwd is False
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].substrates == [_ma]
    assert p1.domains[0].products == [_mb]
    assert p1.domains[0].vmax == pytest.approx(2.0, abs=_TOLERANCE)
    assert p1.domains[0].km == pytest.approx(0.5, abs=_TOLERANCE)
    assert p1.domains[0].start == 5
    assert p1.domains[0].end == 50
    assert isinstance(p1.domains[1], TransporterDomain)
    assert p1.domains[1].molecule is _ma
    assert p1.domains[1].vmax == pytest.approx(1.5, abs=_TOLERANCE)
    assert p1.domains[1].km == pytest.approx(0.5, abs=_TOLERANCE)
    assert p1.domains[1].start == 6
    assert p1.domains[1].end == 60


def test_cell_params_with_regulatory_domains():
    # Protein: (domains, cds_start, cds_end, is_fwd)
    # Domain: (domain_spec, dom_start, dom_end)
    # Domain spec indexes: (dom_types, reacts_trnspts_effctrs, Vmaxs, Kms, signs)
    # fmt: off
    c0 = [
        (
            [
                (
                    (1, 10, 5, 1, 1), # catal, Vmax 2.0, Km 0.5, fwd, a->b
                    1, 10
                ),
                (
                    (3, 0, 10, 1, 3), # reg, Km 1.0, cyto, act, c
                    2, 20
                ),
                (
                    (3, 0, 20, 2, 4), # reg, Km 2.0, cyto, inh, d
                    3, 30
                )
            ],
            1, 100, False
        ),
        (
            [
                (
                    (1, 10, 5, 1, 1), # catal, Vmax 2.0, Km 0.5, fwd, a->b
                    4, 40
                ),
                (
                    (3, 0, 10, 1, 1), # reg, Km 1.0, cyto, act, a
                    5, 50
                ),
                (
                    (3, 0, 15, 1, 5), # reg, Km 1.5, transm, act, a
                    6, 60
                )
            ],
            2, 200, True
        )
    ]

    c1 = [
        (
            [
                (
                    (1, 10, 5, 1, 1), # catal, Vmax 2.0, Km 0.5, fwd, a->b
                    7, 70
                ),
                (
                    (3, 0, 10, 2, 2), # reg, Km 1.0, cyto, inh, b
                    8, 80
                ),
                (
                    (3, 0, 15, 2, 6), # reg, Km 1.5, transm, inh, b
                    9, 90
                )
            ],
            3, 300, False
        ),
        (
            [
                (
                    (1, 10, 5, 1, 1), # catal, Vmax 2.0, Km 0.5, fwd, a->b
                    10, 100
                ),
                (
                    (3, 0, 10, 1, 4), # reg, Km 1.0, cyto, act, d
                    11, 110
                ),
                (
                    (3, 0, 15, 1, 4), # reg, Km 1.5, cyto, act, d
                    12, 120
                )
            ],
            4, 400, True
        )
    ]
    # fmt: on

    Kmf = torch.zeros(2, 3)
    Kmb = torch.zeros(2, 3)
    Kmr = torch.zeros(2, 3)
    Vmax = torch.zeros(2, 3)
    N = torch.zeros(2, 3, 8)
    Nf = torch.zeros(2, 3, 8)
    Nb = torch.zeros(2, 3, 8)
    A = torch.zeros(2, 3, 8)

    # test
    kinetics = _get_kinetics()
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.N = N
    kinetics.Nf = Nf
    kinetics.Nb = Nb
    kinetics.A = A
    proteomes = [[d[0] for d in c0], [d[0] for d in c1]]
    kinetics.set_cell_params(cell_idxs=[0, 1], proteomes=proteomes)

    ke_c0_0 = _ke([_ma], [_mb])
    ke_c0_1 = _ke([_ma], [_mb])
    assert Kmf[0, 0] == pytest.approx(0.5, abs=_TOLERANCE)
    assert Kmb[0, 0] == pytest.approx(0.5 * ke_c0_0, abs=_TOLERANCE)
    assert Kmr[0, 0] == pytest.approx(_avg(1.0, 2.0), abs=_TOLERANCE)
    assert Kmf[0, 1] == pytest.approx(0.5, abs=_TOLERANCE)
    assert Kmb[0, 1] == pytest.approx(0.5 * ke_c0_1, abs=_TOLERANCE)
    assert Kmr[0, 1] == pytest.approx(_avg(1.0, 1.5), abs=_TOLERANCE)
    assert Kmf[0, 2] == pytest.approx(0.0, _TOLERANCE)
    assert Kmb[0, 2] == pytest.approx(0.0, _TOLERANCE)
    assert Kmr[0, 2] == pytest.approx(0.0, _TOLERANCE)

    assert Kmf[1, 0] == pytest.approx(0.5, abs=_TOLERANCE)
    assert Kmb[1, 0] == pytest.approx(0.5 * ke_c0_0, abs=_TOLERANCE)
    assert Kmf[1, 1] == pytest.approx(0.5, abs=_TOLERANCE)
    assert Kmb[1, 1] == pytest.approx(0.5 * ke_c0_1, abs=_TOLERANCE)
    assert Kmf[1, 2] == pytest.approx(0.0, _TOLERANCE)
    assert Kmb[1, 2] == pytest.approx(0.0, _TOLERANCE)

    assert Vmax[0, 0] == pytest.approx(2.0, abs=_TOLERANCE)
    assert Vmax[0, 1] == pytest.approx(2.0, abs=_TOLERANCE)
    assert Vmax[0, 2] == 0.0

    assert Vmax[1, 0] == pytest.approx(2.0, abs=_TOLERANCE)
    assert Vmax[1, 1] == pytest.approx(2.0, abs=_TOLERANCE)
    assert Vmax[1, 2] == 0.0

    assert N[0, 0, 0] == -1
    assert N[0, 0, 1] == 1
    assert (N[0, 0, [2, 3, 4, 5, 6]] == 0).all()
    assert Nf[0, 0, 0] == 1
    assert (Nf[0, 0, [1, 2, 3, 4, 5, 6]] == 0).all()
    assert Nb[0, 0, 1] == 1
    assert (Nb[0, 0, [0, 2, 3, 4, 5, 6]] == 0).all()
    assert N[0, 1, 0] == -1
    assert N[0, 1, 1] == 1
    assert (N[0, 1, [2, 3, 4, 5, 6, 7]] == 0).all()
    assert Nf[0, 1, 0] == 1
    assert (Nf[0, 1, [1, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert Nb[0, 1, 1] == 1
    assert (Nb[0, 1, [0, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert (N[0, 2] == 0).all()
    assert (Nf[0, 2] == 0).all()
    assert (Nb[0, 2] == 0).all()

    assert N[1, 0, 0] == -1
    assert N[1, 0, 1] == 1
    assert (N[1, 0, [2, 3, 4, 5, 6, 7]] == 0).all()
    assert Nf[1, 0, 0] == 1
    assert (Nf[1, 0, [1, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert Nb[1, 0, 1] == 1
    assert (Nb[1, 0, [0, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert N[1, 1, 0] == -1
    assert N[1, 1, 1] == 1
    assert (N[1, 1, [2, 3, 4, 5, 6, 7]] == 0).all()
    assert Nf[1, 1, 0] == 1
    assert (Nf[1, 1, [1, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert Nb[1, 1, 1] == 1
    assert (Nb[1, 1, [0, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert (N[1, 2] == 0).all()
    assert (Nf[1, 2] == 0).all()
    assert (Nb[1, 2] == 0).all()

    assert A[0, 0, 2] == 1
    assert A[0, 0, 3] == -1
    assert (A[0, 0, [0, 1, 4, 5, 6, 7]] == 0).all()
    assert A[0, 1, 0] == 1
    assert A[0, 1, 4] == 1
    assert (A[0, 1, [1, 2, 3, 5, 6, 7]] == 0).all()
    assert (A[0, 2] == 0).all()

    assert A[1, 0, 1] == -1
    assert A[1, 0, 5] == -1
    assert (A[1, 0, [0, 2, 3, 4, 6, 7]] == 0).all()
    assert A[1, 1, 0] == 0
    assert A[1, 1, 3] == 2
    assert (A[1, 1, [1, 2, 4, 5, 6, 7]] == 0).all()
    assert (A[1, 2] == 0).all()

    # test protein representation

    proteins = kinetics.get_proteome(proteome=c0)

    p0 = proteins[0]
    assert p0.cds_start == 1
    assert p0.cds_end == 100
    assert p0.is_fwd is False
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].substrates == [_ma]
    assert p0.domains[0].products == [_mb]
    assert p0.domains[0].vmax == pytest.approx(2.0, abs=_TOLERANCE)
    assert p0.domains[0].km == pytest.approx(0.5, abs=_TOLERANCE)
    assert p0.domains[0].start == 1
    assert p0.domains[0].end == 10
    assert isinstance(p0.domains[1], RegulatoryDomain)
    assert p0.domains[1].effector is _mc
    assert not p0.domains[1].is_inhibiting
    assert not p0.domains[1].is_transmembrane
    assert p0.domains[1].km == pytest.approx(1.0, abs=_TOLERANCE)
    assert p0.domains[1].start == 2
    assert p0.domains[1].end == 20
    assert isinstance(p0.domains[2], RegulatoryDomain)
    assert p0.domains[2].effector is _md
    assert p0.domains[2].is_inhibiting
    assert not p0.domains[2].is_transmembrane
    assert p0.domains[2].km == pytest.approx(2.0, abs=_TOLERANCE)
    assert p0.domains[2].start == 3
    assert p0.domains[2].end == 30

    p1 = proteins[1]
    assert p1.cds_start == 2
    assert p1.cds_end == 200
    assert p1.is_fwd is True
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].substrates == [_ma]
    assert p1.domains[0].products == [_mb]
    assert p1.domains[0].vmax == pytest.approx(2.0, abs=_TOLERANCE)
    assert p1.domains[0].km == pytest.approx(0.5, abs=_TOLERANCE)
    assert p1.domains[0].start == 4
    assert p1.domains[0].end == 40
    assert isinstance(p1.domains[1], RegulatoryDomain)
    assert p1.domains[1].effector is _ma
    assert not p1.domains[1].is_inhibiting
    assert not p1.domains[1].is_transmembrane
    assert p1.domains[1].km == pytest.approx(1.0, abs=_TOLERANCE)
    assert p1.domains[1].start == 5
    assert p1.domains[1].end == 50
    assert isinstance(p1.domains[2], RegulatoryDomain)
    assert p1.domains[2].effector is _ma
    assert not p1.domains[2].is_inhibiting
    assert p1.domains[2].is_transmembrane
    assert p1.domains[2].km == pytest.approx(1.5, abs=_TOLERANCE)
    assert p1.domains[2].start == 6
    assert p1.domains[2].end == 60

    proteins = kinetics.get_proteome(proteome=c1)

    p0 = proteins[0]
    assert p0.cds_start == 3
    assert p0.cds_end == 300
    assert p0.is_fwd is False
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].substrates == [_ma]
    assert p0.domains[0].products == [_mb]
    assert p0.domains[0].vmax == pytest.approx(2.0, abs=_TOLERANCE)
    assert p0.domains[0].km == pytest.approx(0.5, abs=_TOLERANCE)
    assert p0.domains[0].start == 7
    assert p0.domains[0].end == 70
    assert isinstance(p0.domains[1], RegulatoryDomain)
    assert p0.domains[1].effector is _mb
    assert p0.domains[1].is_inhibiting
    assert not p0.domains[1].is_transmembrane
    assert p0.domains[1].km == pytest.approx(1.0, abs=_TOLERANCE)
    assert p0.domains[1].start == 8
    assert p0.domains[1].end == 80
    assert isinstance(p0.domains[2], RegulatoryDomain)
    assert p0.domains[2].effector is _mb
    assert p0.domains[2].is_inhibiting
    assert p0.domains[2].is_transmembrane
    assert p0.domains[2].km == pytest.approx(1.5, abs=_TOLERANCE)
    assert p0.domains[2].start == 9
    assert p0.domains[2].end == 90

    p1 = proteins[1]
    assert p1.cds_start == 4
    assert p1.cds_end == 400
    assert p1.is_fwd is True
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].substrates == [_ma]
    assert p1.domains[0].products == [_mb]
    assert p1.domains[0].vmax == pytest.approx(2.0, abs=_TOLERANCE)
    assert p1.domains[0].km == pytest.approx(0.5, abs=_TOLERANCE)
    assert p1.domains[0].start == 10
    assert p1.domains[0].end == 100
    assert isinstance(p1.domains[1], RegulatoryDomain)
    assert p1.domains[1].effector is _md
    assert not p1.domains[1].is_inhibiting
    assert not p1.domains[1].is_transmembrane
    assert p1.domains[1].km == pytest.approx(1.0, abs=_TOLERANCE)
    assert p1.domains[1].start == 11
    assert p1.domains[1].end == 110
    assert isinstance(p1.domains[2], RegulatoryDomain)
    assert p1.domains[2].effector is _md
    assert not p1.domains[2].is_inhibiting
    assert not p1.domains[2].is_transmembrane
    assert p1.domains[2].km == pytest.approx(1.5, abs=_TOLERANCE)
    assert p1.domains[2].start == 12
    assert p1.domains[2].end == 120


def test_cell_params_with_catalytic_domains():
    # Protein: (domains, cds_start, cds_end, is_fwd)
    # Domain: (domain_spec, dom_start, dom_end)
    # Domain spec indexes: (dom_types, reacts_trnspts_effctrs, Vmaxs, Kms, signs)
    # fmt: off
    c0 = [
        (
            [
                (
                    (1, 1, 5, 1, 1), # catal, Vmax 1.1, Km 0.5, fwd, a->b
                    1, 10
                ),
                (
                    (1, 2, 15, 2, 3), # catal, Vmax 1.2, Km 1.5, bwd, bc->d
                    2, 20
                )
            ],
            1, 100, False
        ),
        (
            [
                (
                    (1, 10, 9, 1, 2), # catal, Vmax 2.0, Km 0.9, fwd, b->c
                    3, 30
                ),
                (
                    (1, 3, 12, 2, 3), # catal, Vmax 1.3, Km 1.2, bwd, bc->d
                    4, 40
                )
            ],
            2, 200, True
        ),
        (
            [
                (
                    (1, 19, 29, 1, 4), # catal, Vmax 2.9, Km 2.9, fwd, d->bb
                    5, 50
                )
            ],
            3, 300, False
        )
    ]
    c1 = [
        (
            [
                (
                    (1, 1, 3, 2, 1), # catal, Vmax 1.1, Km 0.3, bwd, a->b
                    6, 60
                ),
                (
                    (1, 11, 14, 2, 3), # catal, Vmax 2.1, Km 1.4, bwd, bc->d
                    7, 70
                )
            ],
            4, 400, True
        ),
        (
            [
                (
                    (1, 9, 3, 1, 2), # catal, Vmax 1.9, Km 0.3, fwd, b->c
                    8, 80
                ),
                (
                    (1, 13, 17, 1, 3), # catal, Vmax 2.3, Km 1.7, fwd, bc->d
                    9, 90
                )
            ],
            5, 500, False
        )
    ]
    # fmt: on

    Kmf = torch.zeros(2, 3)
    Kmb = torch.zeros(2, 3)
    Kmr = torch.zeros(2, 3)
    Vmax = torch.zeros(2, 3)
    N = torch.zeros(2, 3, 8)
    Nf = torch.zeros(2, 3, 8)
    Nb = torch.zeros(2, 3, 8)
    A = torch.zeros(2, 3, 8)

    # test
    kinetics = _get_kinetics()
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.N = N
    kinetics.Nf = Nf
    kinetics.Nb = Nb
    kinetics.A = A
    proteomes = [[d[0] for d in c0], [d[0] for d in c1]]
    kinetics.set_cell_params(cell_idxs=[0, 1], proteomes=proteomes)

    ke_c0_0 = _ke([_ma, _md], [_mb, _mb, _mc])
    ke_c0_1 = _ke([_mb, _md], [_mc, _mb, _mc])
    ke_c0_2 = _ke([_md], [_mb, _mb])
    assert Kmf[0, 0] == pytest.approx(_avg(0.5, 1.5) / ke_c0_0, abs=_TOLERANCE)
    assert Kmb[0, 0] == pytest.approx(_avg(0.5, 1.5), abs=_TOLERANCE)
    assert Kmf[0, 1] == pytest.approx(_avg(0.9, 1.2) / ke_c0_1, abs=_TOLERANCE)
    assert Kmb[0, 1] == pytest.approx(_avg(0.9, 1.2), abs=_TOLERANCE)
    assert Kmf[0, 2] == pytest.approx(2.9 / ke_c0_2, abs=_TOLERANCE)
    assert Kmb[0, 2] == pytest.approx(2.9, abs=_TOLERANCE)

    ke_c1_0 = _ke([_mb, _md], [_ma, _mb, _mc])
    ke_c1_1 = _ke([_mb, _mb, _mc], [_mc, _md])
    assert Kmf[1, 0] == pytest.approx(_avg(0.3, 1.4) / ke_c1_0, _TOLERANCE)
    assert Kmb[1, 0] == pytest.approx(_avg(0.3, 1.4), _TOLERANCE)
    assert Kmf[1, 1] == pytest.approx(_avg(0.3, 1.7), _TOLERANCE)
    assert Kmb[1, 1] == pytest.approx(_avg(0.3, 1.7) * ke_c1_1, _TOLERANCE)
    assert Kmf[1, 2] == pytest.approx(0.0, _TOLERANCE)
    assert Kmb[1, 2] == pytest.approx(0.0, _TOLERANCE)

    assert (Kmr < _TOLERANCE).all()

    assert Vmax[0, 0] == pytest.approx(_avg(1.1, 1.2), abs=_TOLERANCE)
    assert Vmax[0, 1] == pytest.approx(_avg(2.0, 1.3), abs=_TOLERANCE)
    assert Vmax[0, 2] == pytest.approx(2.9, abs=_TOLERANCE)

    assert Vmax[1, 0] == pytest.approx(_avg(1.1, 2.1), abs=_TOLERANCE)
    assert Vmax[1, 1] == pytest.approx(_avg(1.9, 2.3), abs=_TOLERANCE)
    assert Vmax[1, 2] == 0.0

    assert N[0, 0, 0] == -1
    assert N[0, 0, 1] == 2
    assert N[0, 0, 2] == 1
    assert N[0, 0, 3] == -1
    assert (N[0, 0, [4, 5, 6, 7]] == 0).all()
    assert Nf[0, 0, 0] == 1
    assert Nf[0, 0, 3] == 1
    assert (Nf[0, 0, [1, 2, 4, 5, 6, 7]] == 0).all()
    assert Nb[0, 0, 1] == 2
    assert Nb[0, 0, 2] == 1
    assert (Nb[0, 0, [0, 3, 4, 5, 6, 7]] == 0).all()
    assert N[0, 1, 2] == 2
    assert N[0, 1, 3] == -1
    assert (N[0, 1, [0, 1, 4, 5, 6, 7]] == 0).all()
    assert Nf[0, 1, 1] == 1
    assert Nf[0, 1, 3] == 1
    assert (Nf[0, 1, [0, 2, 4, 5, 6, 7]] == 0).all()
    assert Nb[0, 1, 1] == 1
    assert Nb[0, 1, 2] == 2
    assert (Nb[0, 1, [0, 3, 4, 5, 6, 7]] == 0).all()
    assert N[0, 2, 0] == 0
    assert N[0, 2, 1] == 2
    assert N[0, 2, 2] == 0
    assert N[0, 2, 3] == -1
    assert (N[0, 2, [4, 5, 6, 7]] == 0).all()
    assert Nf[0, 2, 3] == 1
    assert (Nf[0, 2, [0, 1, 2, 4, 5, 6, 7]] == 0).all()
    assert Nb[0, 2, 1] == 2
    assert (Nb[0, 2, [0, 2, 3, 4, 5, 6, 7]] == 0).all()

    assert N[1, 0, 0] == 1
    assert N[1, 0, 1] == 0  # b is added and removed
    assert N[1, 0, 2] == 1
    assert N[1, 0, 3] == -1
    assert (N[1, 0, [4, 5, 6, 7]] == 0).all()
    assert Nf[1, 0, 1] == 1
    assert Nf[1, 0, 3] == 1
    assert (Nf[1, 0, [0, 2, 4, 5, 6, 7]] == 0).all()
    assert Nb[1, 0, 0] == 1
    assert Nb[1, 0, 1] == 1
    assert Nb[1, 0, 2] == 1
    assert (Nb[1, 0, [3, 4, 5, 6, 7]] == 0).all()
    assert N[1, 1, 0] == 0
    assert N[1, 1, 1] == -2
    assert N[1, 1, 2] == 0  # c is added and removed
    assert N[1, 1, 3] == 1
    assert (N[1, 1, [4, 5, 6, 7]] == 0).all()
    assert Nf[1, 1, 1] == 2
    assert Nf[1, 1, 2] == 1
    assert (Nf[1, 1, [0, 3, 4, 5, 6, 7]] == 0).all()
    assert Nb[1, 1, 2] == 1
    assert Nb[1, 1, 3] == 1
    assert (Nb[1, 1, [0, 1, 4, 5, 6, 7]] == 0).all()
    assert (N[1, 2] == 0).all()
    assert (Nf[1, 2] == 0).all()
    assert (Nb[1, 2] == 0).all()

    assert (A == 0).all()

    # test protein representation

    proteins = kinetics.get_proteome(proteome=c0)

    p0 = proteins[0]
    assert p0.cds_start == 1
    assert p0.cds_end == 100
    assert p0.is_fwd is False
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].substrates == [_ma]
    assert p0.domains[0].products == [_mb]
    assert p0.domains[0].vmax == pytest.approx(1.1, abs=_TOLERANCE)
    assert p0.domains[0].km == pytest.approx(0.5, abs=_TOLERANCE)
    assert p0.domains[0].start == 1
    assert p0.domains[0].end == 10
    assert isinstance(p0.domains[1], CatalyticDomain)
    assert p0.domains[1].substrates == [_md]
    assert p0.domains[1].products == [_mb, _mc]
    assert p0.domains[1].vmax == pytest.approx(1.2, abs=_TOLERANCE)
    assert p0.domains[1].km == pytest.approx(1.5, abs=_TOLERANCE)
    assert p0.domains[1].start == 2
    assert p0.domains[1].end == 20

    p1 = proteins[1]
    assert p1.cds_start == 2
    assert p1.cds_end == 200
    assert p1.is_fwd is True
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].substrates == [_mb]
    assert p1.domains[0].products == [_mc]
    assert p1.domains[0].vmax == pytest.approx(2.0, abs=_TOLERANCE)
    assert p1.domains[0].km == pytest.approx(0.9, abs=_TOLERANCE)
    assert p1.domains[0].start == 3
    assert p1.domains[0].end == 30
    assert isinstance(p1.domains[1], CatalyticDomain)
    assert p1.domains[1].substrates == [_md]
    assert p1.domains[1].products == [_mb, _mc]
    assert p1.domains[1].vmax == pytest.approx(1.3, abs=_TOLERANCE)
    assert p1.domains[1].km == pytest.approx(1.2, abs=_TOLERANCE)
    assert p1.domains[1].start == 4
    assert p1.domains[1].end == 40

    p2 = proteins[2]
    assert p2.cds_start == 3
    assert p2.cds_end == 300
    assert p2.is_fwd is False
    assert isinstance(p2.domains[0], CatalyticDomain)
    assert p2.domains[0].substrates == [_md]
    assert p2.domains[0].products == [_mb, _mb]
    assert p2.domains[0].vmax == pytest.approx(2.9, abs=_TOLERANCE)
    assert p2.domains[0].km == pytest.approx(2.9, abs=_TOLERANCE)
    assert p2.domains[0].start == 5
    assert p2.domains[0].end == 50

    proteins = kinetics.get_proteome(proteome=c1)

    p0 = proteins[0]
    assert p0.cds_start == 4
    assert p0.cds_end == 400
    assert p0.is_fwd is True
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].substrates == [_mb]
    assert p0.domains[0].products == [_ma]
    assert p0.domains[0].vmax == pytest.approx(1.1, abs=_TOLERANCE)
    assert p0.domains[0].km == pytest.approx(0.3, abs=_TOLERANCE)
    assert p0.domains[0].start == 6
    assert p0.domains[0].end == 60
    assert isinstance(p0.domains[1], CatalyticDomain)
    assert p0.domains[1].substrates == [_md]
    assert p0.domains[1].products == [_mb, _mc]
    assert p0.domains[1].vmax == pytest.approx(2.1, abs=_TOLERANCE)
    assert p0.domains[1].km == pytest.approx(1.4, abs=_TOLERANCE)
    assert p0.domains[1].start == 7
    assert p0.domains[1].end == 70

    p1 = proteins[1]
    assert p1.cds_start == 5
    assert p1.cds_end == 500
    assert p1.is_fwd is False
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].substrates == [_mb]
    assert p1.domains[0].products == [_mc]
    assert p1.domains[0].vmax == pytest.approx(1.9, abs=_TOLERANCE)
    assert p1.domains[0].km == pytest.approx(0.3, abs=_TOLERANCE)
    assert p1.domains[0].start == 8
    assert p1.domains[0].end == 80
    assert isinstance(p1.domains[1], CatalyticDomain)
    assert p1.domains[1].substrates == [_mb, _mc]
    assert p1.domains[1].products == [_md]
    assert p1.domains[1].vmax == pytest.approx(2.3, abs=_TOLERANCE)
    assert p1.domains[1].km == pytest.approx(1.7, abs=_TOLERANCE)
    assert p1.domains[1].start == 9
    assert p1.domains[1].end == 90


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
    Nf = torch.where(N < 0.0, -N, 0.0)
    Nb = torch.where(N > 0.0, N, 0.0)

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
    kinetics = _get_kinetics()
    kinetics.N = N
    kinetics.Nf = Nf
    kinetics.Nb = Nb
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    assert (Xd[0, 0] - dx_c0_a).abs() < _TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < _TOLERANCE
    assert (Xd[0, 2] - dx_c0_c).abs() < _TOLERANCE
    assert (Xd[0, 3] - dx_c0_d).abs() < _TOLERANCE

    assert (Xd[1, 0] - dx_c1_a).abs() < _TOLERANCE
    assert (Xd[1, 1] - dx_c1_b).abs() < _TOLERANCE
    assert (Xd[1, 2] - dx_c1_c).abs() < _TOLERANCE
    assert (Xd[1, 3] - dx_c1_d).abs() < _TOLERANCE


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
    Nf = torch.where(N < 0.0, -N, 0.0)
    Nb = torch.where(N > 0.0, N, 0.0)

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
    kinetics = _get_kinetics()
    kinetics.N = N
    kinetics.Nf = Nf
    kinetics.Nb = Nb
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    assert (Xd[0, 0] - dx_c0_a).abs() < _TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < _TOLERANCE
    assert (Xd[0, 2] - dx_c0_c).abs() < _TOLERANCE
    assert (Xd[0, 3] - dx_c0_d).abs() < _TOLERANCE

    assert (Xd[1, 0] - dx_c1_a).abs() < _TOLERANCE
    assert (Xd[1, 1] - dx_c1_b).abs() < _TOLERANCE
    assert (Xd[1, 2] - dx_c1_c).abs() < _TOLERANCE
    assert (Xd[1, 3] - dx_c1_d).abs() < _TOLERANCE


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
    Nf = torch.where(N < 0.0, -N, 0.0)
    Nb = torch.where(N > 0.0, N, 0.0)

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
    kinetics = _get_kinetics()
    kinetics.N = N
    kinetics.Nf = Nf
    kinetics.Nb = Nb
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    assert (Xd[0, 0] - dx_c0_a).abs() < _TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < _TOLERANCE
    assert (Xd[0, 2] - dx_c0_c).abs() < _TOLERANCE
    assert (Xd[0, 3] - dx_c0_d).abs() < _TOLERANCE

    assert (Xd[1, 0] - dx_c1_a).abs() < _TOLERANCE
    assert (Xd[1, 1] - dx_c1_b).abs() < _TOLERANCE
    assert (Xd[1, 2] - dx_c1_c).abs() < _TOLERANCE
    assert (Xd[1, 3] - dx_c1_d).abs() < _TOLERANCE


def test_mm_kinetic_with_cofactors():
    # 2 cell, 3 max proteins, 4 molecules (a, b, c, d)
    # N for a molecule might be 0 but it's still required
    # cell 0: P0: a -> b | b -> c
    # cell 0: P1: a + c -> b + c

    # fmt: off

    # concentrations (c, s)
    X0 = torch.tensor([
        [10.0, 0.1, 3.0, 0.8],
        [10.0, 3.0, 0.1, 0.0],
    ])

    # reactions (c, p, s)
    N = torch.tensor([
        [   [-1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]    ],
        [   [-1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])
    Nf = torch.tensor([
        [   [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]    ],
        [   [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])
    Nb = torch.tensor([
        [   [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]    ],
        [   [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]    ],
    ])

    # affinities (c, p)
    Kmf = torch.tensor([
        [2.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    Kmb = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    Kmr = torch.zeros(2, 3)

    # max velocities (c, p)
    Vmax = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    # allosterics (c, p, s)
    A = torch.zeros(2, 3, 4)

    # fmt: on

    def mm(s1, s2, p1, p2, kf, kb, v):
        vf = v * (s1 * s2 / kf - p1 * p2 / kb) / (1 + s1 * s2 / kf + p1 * p2 / kb)
        vb = v * (p1 * p2 / kb - s1 * s2 / kf) / (1 + s1 * s2 / kf + p1 * p2 / kb)
        return (vf - vb) / 2

    # expected outcome
    v_c0_0 = mm(
        X0[0, 0], X0[0, 1], X0[0, 1], X0[0, 2], Kmf[0, 0], Kmb[0, 0], Vmax[0, 0]
    )
    dx_c0_a = -v_c0_0
    dx_c0_b = 0.0
    dx_c0_c = v_c0_0
    dx_c0_d = 0.0

    v_c1_0 = mm(
        X0[1, 0], X0[1, 2], X0[1, 1], X0[1, 2], Kmf[1, 0], Kmb[1, 0], Vmax[1, 0]
    )
    dx_c1_a = -v_c1_0
    dx_c1_b = v_c1_0
    dx_c1_c = 0.0
    dx_c1_d = 0.0

    # test
    kinetics = _get_kinetics()
    kinetics.N = N
    kinetics.Nf = Nf
    kinetics.Nb = Nb
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    assert (Xd[0, 0] - dx_c0_a).abs() < _TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < _TOLERANCE
    assert (Xd[0, 2] - dx_c0_c).abs() < _TOLERANCE
    assert (Xd[0, 3] - dx_c0_d).abs() < _TOLERANCE

    assert (Xd[1, 0] - dx_c1_a).abs() < _TOLERANCE
    assert (Xd[1, 1] - dx_c1_b).abs() < _TOLERANCE
    assert (Xd[1, 2] - dx_c1_c).abs() < _TOLERANCE
    assert (Xd[1, 3] - dx_c1_d).abs() < _TOLERANCE


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
    Nf = torch.where(N < 0.0, -N, 0.0)
    Nb = torch.where(N > 0.0, N, 0.0)

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
    kinetics = _get_kinetics()
    kinetics.N = N
    kinetics.Nf = Nf
    kinetics.Nb = Nb
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    assert (Xd[0, 0] - dx_c0_a).abs() < _TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < _TOLERANCE
    assert (Xd[0, 2] - dx_c0_c).abs() < _TOLERANCE
    assert (Xd[0, 3] - dx_c0_d).abs() < _TOLERANCE

    assert (Xd[1, 0] - dx_c1_a).abs() < _TOLERANCE
    assert (Xd[1, 1] - dx_c1_b).abs() < _TOLERANCE
    assert (Xd[1, 2] - dx_c1_c).abs() < _TOLERANCE
    assert (Xd[1, 3] - dx_c1_d).abs() < _TOLERANCE


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
    Nf = torch.where(N < 0.0, -N, 0.0)
    Nb = torch.where(N > 0.0, N, 0.0)

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
    f = (X0[0, 0] - _EPS) / v_c0_0
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
    v_c1_0 = (X0[1, 2] - _EPS) / 2
    dx_c1_a = 0.0
    dx_c1_b = 0.0
    dx_c1_c = -2 * v_c1_0
    dx_c1_d = v_c1_0

    # test
    kinetics = _get_kinetics()
    kinetics.N = N
    kinetics.Nf = Nf
    kinetics.Nb = Nb
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    assert (Xd[0, 0] - dx_c0_a).abs() < _TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < _TOLERANCE
    assert (Xd[0, 2] - dx_c0_c).abs() < _TOLERANCE
    assert (Xd[0, 3] - dx_c0_d).abs() < _TOLERANCE

    assert (Xd[1, 0] - dx_c1_a).abs() < _TOLERANCE
    assert (Xd[1, 1] - dx_c1_b).abs() < _TOLERANCE
    assert (Xd[1, 2] - dx_c1_c).abs() < _TOLERANCE
    assert (Xd[1, 3] - dx_c1_d).abs() < _TOLERANCE

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
    Nf = torch.where(N < 0.0, -N, 0.0)
    Nb = torch.where(N > 0.0, N, 0.0)

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
    f = (X0[0, 0] - _EPS) / -naive_dx_c0_a
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
    kinetics = _get_kinetics()
    kinetics.N = N
    kinetics.Nf = Nf
    kinetics.Nb = Nb
    kinetics.Kmf = Kmf
    kinetics.Kmb = Kmb
    kinetics.Kmr = Kmr
    kinetics.Vmax = Vmax
    kinetics.A = A
    Xd = kinetics.integrate_signals(X=X0) - X0

    assert (Xd[0, 0] - dx_c0_a).abs() < _TOLERANCE
    assert (Xd[0, 1] - dx_c0_b).abs() < _TOLERANCE
    assert (Xd[0, 2] - dx_c0_c).abs() < _TOLERANCE
    assert (Xd[0, 3] - dx_c0_d).abs() < _TOLERANCE

    assert (Xd[1, 0] - dx_c1_a).abs() < _TOLERANCE
    assert (Xd[1, 1] - dx_c1_b).abs() < _TOLERANCE
    assert (Xd[1, 2] - dx_c1_c).abs() < _TOLERANCE
    assert (Xd[1, 3] - dx_c1_d).abs() < _TOLERANCE

    X1 = X0 + Xd
    assert not torch.any(X1 < 0.0)
