import math

import pytest
import torch
from magicsoup.constants import GAS_CONSTANT
from magicsoup.containers import (
    CatalyticDomain,
    Chemistry,
    Molecule,
    RegulatoryDomain,
    TransporterDomain,
)
from magicsoup.proteomics import Proteomics

_TOLERANCE = 1e-4
_FLOAT = torch.float32
_INT = torch.int8
_NAN = torch.nan


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

_CHEMISTRY = Chemistry(molecules=_MOLECULES, reactions=_REACTIONS)

# fmt: off
_KM_WEIGHTS = torch.tensor([
    _NAN, 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  # idxs 0-9
    1.0,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  # idxs 10-19
    2.0,  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  # idxs 20-29
])

_VMAX_WEIGHTS = torch.tensor([
    _NAN, 1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  # idxs 0-9
    2.0,  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  # idxs 10-19
])

_SIGNS = torch.tensor([0, 1, -1], dtype=_INT)  # idxs 0-2

_HILLS = torch.tensor([0, 1, 2, 3, 4, 5], dtype=_INT)  # idxs 0-5

_TRANSPORT_M = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0],  # idx 0: none
    [-1, 0, 0, 0, 1, 0, 0, 0], # idx 1: a in->out
    [0, -1, 0, 0, 0, 1, 0, 0], # idx 2: b in->out
    [0, 0, -1, 0, 0, 0, 1, 0], # idx 3: c in->out
    [0, 0, 0, -1, 0, 0, 0, 1], # idx 4: d in->out
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=_INT)

_EFFECTOR_M = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0], # idx 0: none
    [1, 0, 0, 0, 0, 0, 0, 0], # idx 1: a in
    [0, 1, 0, 0, 0, 0, 0, 0], # idx 2: b in
    [0, 0, 1, 0, 0, 0, 0, 0], # idx 3: c in
    [0, 0, 0, 1, 0, 0, 0, 0], # idx 4: d in
    [0, 0, 0, 0, 1, 0, 0, 0], # idx 5: a out
    [0, 0, 0, 0, 0, 1, 0, 0], # idx 6: b out
    [0, 0, 0, 0, 0, 0, 1, 0], # idx 7: c out
    [0, 0, 0, 0, 0, 0, 0, 1], # idx 8: d out
], dtype=_INT)

_REACTION_M = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0],   # idx 0: none
    [-1, 1, 0, 0, 0, 0, 0, 0],  # idx 1: a -> b
    [0, -1, 1, 0, 0, 0, 0, 0],  # idx 2: b -> c
    [0, -1, -1, 1, 0, 0, 0, 0], # idx 3: b,c -> d
    [0, 2, 0, -1, 0, 0, 0, 0],  # idx 4: d -> 2b
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=_INT)

# fmt: on


def _get_proteomics() -> Proteomics:
    protics = Proteomics(chemistry=_CHEMISTRY, abs_temp=310)
    protics.km_map.weights = _KM_WEIGHTS.clone()
    protics.vmax_map.weights = _VMAX_WEIGHTS.clone()
    protics.sign_map.signs = _SIGNS.clone()
    protics.transport_map.M = _TRANSPORT_M.clone()
    protics.effector_map.M = _EFFECTOR_M.clone()
    protics.reaction_map.M = _REACTION_M.clone()
    protics.hill_map.numbers = _HILLS.clone()
    return protics


def _ke(subs: list[Molecule], prods: list[Molecule]) -> float:
    e = sum(d.energy for d in prods) - sum(d.energy for d in subs)
    return math.exp(-e / 310 / GAS_CONSTANT)


def _avg(*x: float) -> float:
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
                    (2, 5, 5, 1, 1),  # transporter, v_max 1.5, Km 0.5, fwd, mol a
                    6, 27
                )
            ],
            13, 27, True
        )
        ,
        (
            [
                (
                    (2, 5, 5, 1, 1), # transporter, v_max 1.5, Km 0.5, fwd, mol a
                    5, 13
                ),
                (
                    (2, 1, 2, 2, 1),  # transporter, v_max 1.1, Km 0.2, bwd, mol a
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
                    (2, 5, 4, 1, 1), # transporter, v_max 1.5, Km 0.4, fwd, mol a
                    1, 10
                ),
                (
                    (2, 4, 5, 1, 1), # transporter, v_max 1.4, Km 0.5, fwd, mol a
                    2, 20
                ),
                (
                    (2, 3, 6, 1, 2), # transporter, v_max 1.3, Km 0.6, fwd, mol b
                    3, 30
                ),
                (
                    (2, 2, 7, 1, 3),  # transporter, v_max 1.2, Km 0.7, fwd, mol c
                    4, 40
                )
            ],
            91, 112, False
        ),
        (
            [
                (
                    (1, 10, 5, 1, 1), # catal, v_max 2.0, Km 0.5, fwd, a->b
                    5, 50
                ),
                (
                    (2, 5, 5, 1, 1),   # transporter, v_max 1.5, Km 0.5, fwd, mol a
                    6, 60
                )
            ],
            1, 10, False
        ),
    ]
    # fmt: on

    k_f = torch.zeros(2, 3, dtype=_FLOAT)
    k_b = torch.zeros(2, 3, dtype=_FLOAT)
    k_e = torch.zeros(2, 3, dtype=_FLOAT)
    K_r = torch.zeros(2, 3, 8, dtype=_FLOAT)
    v_max = torch.zeros(2, 3, dtype=_FLOAT)
    N = torch.zeros(2, 3, 8, dtype=_INT)
    N_f = torch.zeros(2, 3, 8, dtype=_INT)
    N_b = torch.zeros(2, 3, 8, dtype=_INT)
    N_h = torch.zeros(2, 3, 8, dtype=_INT)

    # test
    kinetics = _get_proteomics()
    kinetics.k_e = k_e
    kinetics.k_f = k_f
    kinetics.k_b = k_b
    kinetics.K_r = K_r
    kinetics.v_max = v_max
    kinetics.N = N
    kinetics.N_f = N_f
    kinetics.N_b = N_b
    kinetics.N_h = N_h
    proteomes = [c0, c1]
    kinetics.set_cell_params(cell_idxs=[0, 1], proteomes=proteomes)

    assert k_e[0, 0] == pytest.approx(1.0, abs=_TOLERANCE)
    assert k_f[0, 0] == pytest.approx(0.5, abs=_TOLERANCE)
    assert k_b[0, 0] == pytest.approx(0.5, abs=_TOLERANCE)
    assert k_e[0, 1] == pytest.approx(1.0, abs=_TOLERANCE)
    assert k_f[0, 1] == pytest.approx(_avg(0.5, 0.2), abs=_TOLERANCE)
    assert k_b[0, 1] == pytest.approx(_avg(0.5, 0.2), abs=_TOLERANCE)
    assert k_e[0, 1] == pytest.approx(1.0, abs=_TOLERANCE)
    assert k_f[0, 2] == pytest.approx(0.0, _TOLERANCE)
    assert k_b[0, 2] == pytest.approx(0.0, _TOLERANCE)

    ke_c1_1 = _ke([_ma], [_mb])
    assert k_e[1, 0] == pytest.approx(1.0, abs=_TOLERANCE)
    assert k_f[1, 0] == pytest.approx(_avg(0.4, 0.5, 0.6, 0.7), abs=_TOLERANCE)
    assert k_b[1, 0] == pytest.approx(_avg(0.4, 0.5, 0.6, 0.7), abs=_TOLERANCE)
    assert k_e[1, 1] == pytest.approx(ke_c1_1, abs=_TOLERANCE)
    assert k_f[1, 1] == pytest.approx(_avg(0.5, 0.5), abs=_TOLERANCE)
    assert k_b[1, 1] == pytest.approx(_avg(0.5, 0.5) * ke_c1_1, abs=_TOLERANCE)
    assert k_f[1, 2] == pytest.approx(0.0, _TOLERANCE)
    assert k_b[1, 2] == pytest.approx(0.0, _TOLERANCE)

    assert (K_r - 1.0 < _TOLERANCE).all()

    assert v_max[0, 0] == pytest.approx(1.5, abs=_TOLERANCE)
    assert v_max[0, 1] == pytest.approx(_avg(1.5, 1.1), abs=_TOLERANCE)
    assert v_max[0, 2] == 0.0

    assert v_max[1, 0] == pytest.approx(_avg(1.5, 1.4, 1.3, 1.2), abs=_TOLERANCE)
    assert v_max[1, 1] == pytest.approx(_avg(2.0, 1.5), abs=_TOLERANCE)
    assert v_max[1, 2] == 0.0

    assert N[0, 0, 0] == -1
    assert N[0, 0, 4] == 1
    assert (N[0, 0, [1, 2, 3, 5, 6, 7]] == 0).all()
    assert N_f[0, 0, 0] == 1
    assert (N_f[0, 0, [1, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert N_b[0, 1, 4] == 1
    assert (N_b[0, 0, [0, 1, 2, 3, 5, 6, 7]] == 0).all()

    assert N[1, 0, 0] == -2
    assert N[1, 0, 1] == -1
    assert N[1, 0, 2] == -1
    assert N[1, 0, 4] == 2
    assert N[1, 0, 5] == 1
    assert N[1, 0, 6] == 1
    assert (N[1, 0, [3, 7]] == 0).all()
    assert N_f[1, 0, 0] == 2
    assert N_f[1, 0, 1] == 1
    assert N_f[1, 0, 2] == 1
    assert (N_f[1, 0, [4, 5, 6, 3, 7]] == 0).all()
    assert N_b[1, 0, 4] == 2
    assert N_b[1, 0, 5] == 1
    assert N_b[1, 0, 6] == 1
    assert (N_b[1, 0, [0, 1, 2, 3, 7]] == 0).all()
    assert N[1, 1, 0] == -2
    assert N[1, 1, 1] == 1
    assert N[1, 1, 4] == 1
    assert (N[1, 1, [2, 3, 5, 6, 7]] == 0).all()
    assert N_f[1, 1, 0] == 2
    assert (N_f[1, 1, [1, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert N_b[1, 1, 1] == 1
    assert N_b[1, 1, 4] == 1
    assert (N_b[1, 1, [0, 2, 3, 5, 6, 7]] == 0).all()

    assert (N_h == 0).all()

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
                    (1, 10, 5, 1, 1), # catal, v_max 2.0, Km 0.5, fwd, a->b
                    1, 10
                ),
                (
                    (3, 1, 10, 1, 3), # reg, coeff 1, Km 1.0, cyto, act, c
                    2, 20
                ),
                (
                    (3, 2, 20, 2, 4), # reg, coeff 2, Km 2.0, cyto, inh, d
                    3, 30
                )
            ],
            1, 100, False
        ),
        (
            [
                (
                    (1, 10, 5, 1, 1), # catal, v_max 2.0, Km 0.5, fwd, a->b
                    4, 40
                ),
                (
                    (3, 1, 10, 1, 1), # reg, coeff 1, Km 1.0, cyto, act, a
                    5, 50
                ),
                (
                    (3, 3, 15, 1, 5), # reg, coeff 3, Km 1.5, transm, act, a
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
                    (1, 10, 5, 1, 1), # catal, v_max 2.0, Km 0.5, fwd, a->b
                    7, 70
                ),
                (
                    (3, 1, 10, 2, 2), # reg, coeff 3, Km 1.0, cyto, inh, b
                    8, 80
                ),
                (
                    (3, 3, 15, 2, 6), # reg, coeff 3, Km 1.5, transm, inh, b
                    9, 90
                )
            ],
            3, 300, False
        ),
        (
            [
                (
                    (1, 10, 5, 1, 1), # catal, v_max 2.0, Km 0.5, fwd, a->b
                    10, 100
                ),
                (
                    (3, 2, 10, 1, 4), # reg, coeff 2, Km 1.0, cyto, act, d
                    11, 110
                ),
                (
                    (3, 3, 15, 1, 4), # reg, coeff 3, Km 1.5, cyto, act, d
                    12, 120
                )
            ],
            4, 400, True
        )
    ]
    # fmt: on

    k_e = torch.zeros(2, 3, dtype=_FLOAT)
    k_f = torch.zeros(2, 3, dtype=_FLOAT)
    k_b = torch.zeros(2, 3, dtype=_FLOAT)
    K_r = torch.zeros(2, 3, 8, dtype=_FLOAT)
    v_max = torch.zeros(2, 3, dtype=_FLOAT)
    N = torch.zeros(2, 3, 8, dtype=_INT)
    N_f = torch.zeros(2, 3, 8, dtype=_INT)
    N_b = torch.zeros(2, 3, 8, dtype=_INT)
    N_h = torch.zeros(2, 3, 8, dtype=_INT)

    # test
    kinetics = _get_proteomics()
    kinetics.k_e = k_e
    kinetics.k_f = k_f
    kinetics.k_b = k_b
    kinetics.K_r = K_r
    kinetics.v_max = v_max
    kinetics.N = N
    kinetics.N_f = N_f
    kinetics.N_b = N_b
    kinetics.N_h = N_h
    proteomes = [c0, c1]
    kinetics.set_cell_params(cell_idxs=[0, 1], proteomes=proteomes)

    ke_a_b = _ke([_ma], [_mb])
    assert k_e[0, 0] == pytest.approx(ke_a_b, abs=_TOLERANCE)
    assert k_f[0, 0] == pytest.approx(0.5, abs=_TOLERANCE)
    assert k_b[0, 0] == pytest.approx(0.5 * ke_a_b, abs=_TOLERANCE)
    assert K_r[0, 0, 2] == pytest.approx(1.0, abs=_TOLERANCE)
    assert K_r[0, 0, 3] == pytest.approx(2.0 ** (-2), abs=_TOLERANCE)
    assert k_e[0, 1] == pytest.approx(ke_a_b, abs=_TOLERANCE)
    assert k_f[0, 1] == pytest.approx(0.5, abs=_TOLERANCE)
    assert k_b[0, 1] == pytest.approx(0.5 * ke_a_b, abs=_TOLERANCE)
    assert K_r[0, 1, 0] == pytest.approx(1.0, abs=_TOLERANCE)
    assert K_r[0, 1, 4] == pytest.approx(1.5**3, abs=_TOLERANCE)
    assert k_f[0, 2] == pytest.approx(0.0, _TOLERANCE)
    assert k_b[0, 2] == pytest.approx(0.0, _TOLERANCE)
    assert torch.all(K_r[0, 2] - 1.0 < _TOLERANCE)

    assert k_e[1, 0] == pytest.approx(ke_a_b, abs=_TOLERANCE)
    assert k_f[1, 0] == pytest.approx(0.5, abs=_TOLERANCE)
    assert k_b[1, 0] == pytest.approx(0.5 * ke_a_b, abs=_TOLERANCE)
    assert K_r[1, 0, 1] == pytest.approx(1.0, abs=_TOLERANCE)
    assert K_r[1, 0, 5] == pytest.approx(1.5 ** (-3), abs=_TOLERANCE)
    assert k_e[1, 1] == pytest.approx(ke_a_b, abs=_TOLERANCE)
    assert k_f[1, 1] == pytest.approx(0.5, abs=_TOLERANCE)
    assert k_b[1, 1] == pytest.approx(0.5 * ke_a_b, abs=_TOLERANCE)
    assert K_r[1, 1, 3] == pytest.approx(_avg(1.0, 1.5) ** 5, abs=_TOLERANCE)
    assert k_f[1, 2] == pytest.approx(0.0, _TOLERANCE)
    assert k_b[1, 2] == pytest.approx(0.0, _TOLERANCE)

    assert v_max[0, 0] == pytest.approx(2.0, abs=_TOLERANCE)
    assert v_max[0, 1] == pytest.approx(2.0, abs=_TOLERANCE)
    assert v_max[0, 2] == 0.0

    assert v_max[1, 0] == pytest.approx(2.0, abs=_TOLERANCE)
    assert v_max[1, 1] == pytest.approx(2.0, abs=_TOLERANCE)
    assert v_max[1, 2] == 0.0

    assert N[0, 0, 0] == -1
    assert N[0, 0, 1] == 1
    assert (N[0, 0, [2, 3, 4, 5, 6]] == 0).all()
    assert N_f[0, 0, 0] == 1
    assert (N_f[0, 0, [1, 2, 3, 4, 5, 6]] == 0).all()
    assert N_b[0, 0, 1] == 1
    assert (N_b[0, 0, [0, 2, 3, 4, 5, 6]] == 0).all()
    assert N[0, 1, 0] == -1
    assert N[0, 1, 1] == 1
    assert (N[0, 1, [2, 3, 4, 5, 6, 7]] == 0).all()
    assert N_f[0, 1, 0] == 1
    assert (N_f[0, 1, [1, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert N_b[0, 1, 1] == 1
    assert (N_b[0, 1, [0, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert (N[0, 2] == 0).all()
    assert (N_f[0, 2] == 0).all()
    assert (N_b[0, 2] == 0).all()

    assert N[1, 0, 0] == -1
    assert N[1, 0, 1] == 1
    assert (N[1, 0, [2, 3, 4, 5, 6, 7]] == 0).all()
    assert N_f[1, 0, 0] == 1
    assert (N_f[1, 0, [1, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert N_b[1, 0, 1] == 1
    assert (N_b[1, 0, [0, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert N[1, 1, 0] == -1
    assert N[1, 1, 1] == 1
    assert (N[1, 1, [2, 3, 4, 5, 6, 7]] == 0).all()
    assert N_f[1, 1, 0] == 1
    assert (N_f[1, 1, [1, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert N_b[1, 1, 1] == 1
    assert (N_b[1, 1, [0, 2, 3, 4, 5, 6, 7]] == 0).all()
    assert (N[1, 2] == 0).all()
    assert (N_f[1, 2] == 0).all()
    assert (N_b[1, 2] == 0).all()

    assert N_h[0, 0, 2] == 1
    assert N_h[0, 0, 3] == -2
    assert (N_h[0, 0, [0, 1, 4, 5, 6, 7]] == 0).all()
    assert N_h[0, 1, 0] == 1
    assert N_h[0, 1, 4] == 3
    assert (N_h[0, 1, [1, 2, 3, 5, 6, 7]] == 0).all()
    assert (N_h[0, 2] == 0).all()

    assert N_h[1, 0, 1] == -1
    assert N_h[1, 0, 5] == -3
    assert (N_h[1, 0, [0, 2, 3, 4, 6, 7]] == 0).all()
    assert N_h[1, 1, 0] == 0
    assert N_h[1, 1, 3] == 5
    assert (N_h[1, 1, [1, 2, 4, 5, 6, 7]] == 0).all()
    assert (N_h[1, 2] == 0).all()

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
    assert p0.domains[1].hill == 1
    assert p0.domains[1].start == 2
    assert p0.domains[1].end == 20
    assert isinstance(p0.domains[2], RegulatoryDomain)
    assert p0.domains[2].effector is _md
    assert p0.domains[2].is_inhibiting
    assert not p0.domains[2].is_transmembrane
    assert p0.domains[2].km == pytest.approx(2.0, abs=_TOLERANCE)
    assert p0.domains[2].hill == 2
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
    assert p1.domains[1].hill == 1
    assert p1.domains[1].start == 5
    assert p1.domains[1].end == 50
    assert isinstance(p1.domains[2], RegulatoryDomain)
    assert p1.domains[2].effector is _ma
    assert not p1.domains[2].is_inhibiting
    assert p1.domains[2].is_transmembrane
    assert p1.domains[2].km == pytest.approx(1.5, abs=_TOLERANCE)
    assert p1.domains[2].hill == 3
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
    assert p0.domains[1].hill == 1
    assert p0.domains[1].start == 8
    assert p0.domains[1].end == 80
    assert isinstance(p0.domains[2], RegulatoryDomain)
    assert p0.domains[2].effector is _mb
    assert p0.domains[2].is_inhibiting
    assert p0.domains[2].is_transmembrane
    assert p0.domains[2].km == pytest.approx(1.5, abs=_TOLERANCE)
    assert p0.domains[2].hill == 3
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
    assert p1.domains[1].hill == 2
    assert p1.domains[1].start == 11
    assert p1.domains[1].end == 110
    assert isinstance(p1.domains[2], RegulatoryDomain)
    assert p1.domains[2].effector is _md
    assert not p1.domains[2].is_inhibiting
    assert not p1.domains[2].is_transmembrane
    assert p1.domains[2].km == pytest.approx(1.5, abs=_TOLERANCE)
    assert p1.domains[2].hill == 3
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
                    (1, 1, 5, 1, 1), # catal, v_max 1.1, Km 0.5, fwd, a->b
                    1, 10
                ),
                (
                    (1, 2, 15, 2, 3), # catal, v_max 1.2, Km 1.5, bwd, bc->d
                    2, 20
                )
            ],
            1, 100, False
        ),
        (
            [
                (
                    (1, 10, 9, 1, 2), # catal, v_max 2.0, Km 0.9, fwd, b->c
                    3, 30
                ),
                (
                    (1, 3, 12, 2, 3), # catal, v_max 1.3, Km 1.2, bwd, bc->d
                    4, 40
                )
            ],
            2, 200, True
        ),
        (
            [
                (
                    (1, 19, 29, 1, 4), # catal, v_max 2.9, Km 2.9, fwd, d->bb
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
                    (1, 1, 3, 2, 1), # catal, v_max 1.1, Km 0.3, bwd, a->b
                    6, 60
                ),
                (
                    (1, 11, 14, 2, 3), # catal, v_max 2.1, Km 1.4, bwd, bc->d
                    7, 70
                )
            ],
            4, 400, True
        ),
        (
            [
                (
                    (1, 9, 3, 1, 2), # catal, v_max 1.9, Km 0.3, fwd, b->c
                    8, 80
                ),
                (
                    (1, 13, 17, 1, 3), # catal, v_max 2.3, Km 1.7, fwd, bc->d
                    9, 90
                )
            ],
            5, 500, False
        )
    ]
    # fmt: on

    k_e = torch.zeros(2, 3, dtype=_FLOAT)
    k_f = torch.zeros(2, 3, dtype=_FLOAT)
    k_b = torch.zeros(2, 3, dtype=_FLOAT)
    K_r = torch.zeros(2, 3, 8, dtype=_FLOAT)
    v_max = torch.zeros(2, 3, dtype=_FLOAT)
    N = torch.zeros(2, 3, 8, dtype=_INT)
    N_f = torch.zeros(2, 3, 8, dtype=_INT)
    N_b = torch.zeros(2, 3, 8, dtype=_INT)
    N_h = torch.zeros(2, 3, 8, dtype=_INT)

    # test
    kinetics = _get_proteomics()
    kinetics.k_e = k_e
    kinetics.k_f = k_f
    kinetics.k_b = k_b
    kinetics.K_r = K_r
    kinetics.v_max = v_max
    kinetics.N = N
    kinetics.N_f = N_f
    kinetics.N_b = N_b
    kinetics.N_h = N_h
    proteomes = [c0, c1]
    kinetics.set_cell_params(cell_idxs=[0, 1], proteomes=proteomes)

    ke_c0_0 = _ke([_ma, _md], [_mb, _mb, _mc])
    ke_c0_1 = _ke([_mb, _md], [_mc, _mb, _mc])
    ke_c0_2 = _ke([_md], [_mb, _mb])
    assert k_e[0, 0] == pytest.approx(ke_c0_0, abs=_TOLERANCE)
    assert k_f[0, 0] == pytest.approx(_avg(0.5, 1.5) / ke_c0_0, abs=_TOLERANCE)
    assert k_b[0, 0] == pytest.approx(_avg(0.5, 1.5), abs=_TOLERANCE)
    assert k_e[0, 1] == pytest.approx(ke_c0_1, abs=_TOLERANCE)
    assert k_f[0, 1] == pytest.approx(_avg(0.9, 1.2) / ke_c0_1, abs=_TOLERANCE)
    assert k_b[0, 1] == pytest.approx(_avg(0.9, 1.2), abs=_TOLERANCE)
    assert k_e[0, 2] == pytest.approx(ke_c0_2, abs=_TOLERANCE)
    assert k_f[0, 2] == pytest.approx(2.9 / ke_c0_2, abs=_TOLERANCE)
    assert k_b[0, 2] == pytest.approx(2.9, abs=_TOLERANCE)

    ke_c1_0 = _ke([_mb, _md], [_ma, _mb, _mc])
    ke_c1_1 = _ke([_mb, _mb, _mc], [_mc, _md])
    assert k_e[1, 0] == pytest.approx(ke_c1_0, abs=_TOLERANCE)
    assert k_f[1, 0] == pytest.approx(_avg(0.3, 1.4) / ke_c1_0, _TOLERANCE)
    assert k_b[1, 0] == pytest.approx(_avg(0.3, 1.4), _TOLERANCE)
    assert k_e[1, 1] == pytest.approx(ke_c1_1, abs=_TOLERANCE)
    assert k_f[1, 1] == pytest.approx(_avg(0.3, 1.7), _TOLERANCE)
    assert k_b[1, 1] == pytest.approx(_avg(0.3, 1.7) * ke_c1_1, _TOLERANCE)
    assert k_f[1, 2] == pytest.approx(0.0, _TOLERANCE)
    assert k_b[1, 2] == pytest.approx(0.0, _TOLERANCE)

    assert (K_r - 1.0 < _TOLERANCE).all()

    assert v_max[0, 0] == pytest.approx(_avg(1.1, 1.2), abs=_TOLERANCE)
    assert v_max[0, 1] == pytest.approx(_avg(2.0, 1.3), abs=_TOLERANCE)
    assert v_max[0, 2] == pytest.approx(2.9, abs=_TOLERANCE)

    assert v_max[1, 0] == pytest.approx(_avg(1.1, 2.1), abs=_TOLERANCE)
    assert v_max[1, 1] == pytest.approx(_avg(1.9, 2.3), abs=_TOLERANCE)
    assert v_max[1, 2] == 0.0

    assert N[0, 0, 0] == -1
    assert N[0, 0, 1] == 2
    assert N[0, 0, 2] == 1
    assert N[0, 0, 3] == -1
    assert (N[0, 0, [4, 5, 6, 7]] == 0).all()
    assert N_f[0, 0, 0] == 1
    assert N_f[0, 0, 3] == 1
    assert (N_f[0, 0, [1, 2, 4, 5, 6, 7]] == 0).all()
    assert N_b[0, 0, 1] == 2
    assert N_b[0, 0, 2] == 1
    assert (N_b[0, 0, [0, 3, 4, 5, 6, 7]] == 0).all()
    assert N[0, 1, 2] == 2
    assert N[0, 1, 3] == -1
    assert (N[0, 1, [0, 1, 4, 5, 6, 7]] == 0).all()
    assert N_f[0, 1, 1] == 1
    assert N_f[0, 1, 3] == 1
    assert (N_f[0, 1, [0, 2, 4, 5, 6, 7]] == 0).all()
    assert N_b[0, 1, 1] == 1
    assert N_b[0, 1, 2] == 2
    assert (N_b[0, 1, [0, 3, 4, 5, 6, 7]] == 0).all()
    assert N[0, 2, 0] == 0
    assert N[0, 2, 1] == 2
    assert N[0, 2, 2] == 0
    assert N[0, 2, 3] == -1
    assert (N[0, 2, [4, 5, 6, 7]] == 0).all()
    assert N_f[0, 2, 3] == 1
    assert (N_f[0, 2, [0, 1, 2, 4, 5, 6, 7]] == 0).all()
    assert N_b[0, 2, 1] == 2
    assert (N_b[0, 2, [0, 2, 3, 4, 5, 6, 7]] == 0).all()

    assert N[1, 0, 0] == 1
    assert N[1, 0, 1] == 0  # b is added and removed
    assert N[1, 0, 2] == 1
    assert N[1, 0, 3] == -1
    assert (N[1, 0, [4, 5, 6, 7]] == 0).all()
    assert N_f[1, 0, 1] == 1
    assert N_f[1, 0, 3] == 1
    assert (N_f[1, 0, [0, 2, 4, 5, 6, 7]] == 0).all()
    assert N_b[1, 0, 0] == 1
    assert N_b[1, 0, 1] == 1
    assert N_b[1, 0, 2] == 1
    assert (N_b[1, 0, [3, 4, 5, 6, 7]] == 0).all()
    assert N[1, 1, 0] == 0
    assert N[1, 1, 1] == -2
    assert N[1, 1, 2] == 0  # c is added and removed
    assert N[1, 1, 3] == 1
    assert (N[1, 1, [4, 5, 6, 7]] == 0).all()
    assert N_f[1, 1, 1] == 2
    assert N_f[1, 1, 2] == 1
    assert (N_f[1, 1, [0, 3, 4, 5, 6, 7]] == 0).all()
    assert N_b[1, 1, 2] == 1
    assert N_b[1, 1, 3] == 1
    assert (N_b[1, 1, [0, 1, 4, 5, 6, 7]] == 0).all()
    assert (N[1, 2] == 0).all()
    assert (N_f[1, 2] == 0).all()
    assert (N_b[1, 2] == 0).all()

    assert (N_h == 0).all()

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
