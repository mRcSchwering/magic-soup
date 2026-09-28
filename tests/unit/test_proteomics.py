import math
import random

import pytest
import torch
from magicsoup.constants import GAS_CONSTANT, DomainSpecType, ProteinSpecType
from magicsoup.containers import (
    CatalyticDomain,
    Chemistry,
    Molecule,
    RegulatoryDomain,
    TransporterDomain,
)
from magicsoup.proteomics import Proteomics

from tests.config import DEVICE

_TOL = 1e-4
_ATOL = 1e-4
_RTOL = 1e-4
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
_r_d_bb = ([_md], 2 * [_mb])
_REACTIONS = [_r_a_b, _r_b_c, _r_bc_d, _r_d_bb]

_CHEMISTRY = Chemistry(molecules=_MOLECULES, reactions=_REACTIONS)

# fmt: off
_KM_WEIGHTS = torch.tensor([
#   x0    x1    x2    x3    x4    x5    x6    x7    x8    x9
    _NAN, 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  # 0x
    1.0,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  # 1x
    2.0,  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  # 2x
])

_VMAX_WEIGHTS = torch.tensor([
#   x0    x1    x2    x3    x4    x5    x6    x7    x8    x9
    _NAN, 1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  # 0x
    2.0,  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  # 1x
])

#                      0   1   2
_SIGNS = torch.tensor([0,  1,  -1], dtype=_INT)

#                      0  1  2  3  4  5
_HILLS = torch.tensor([0, 1, 2, 3, 4, 5], dtype=_INT)

_TRANSPORT_M = torch.tensor([
    [ 0,  0,  0,  0,  0,  0,  0,  0], # 0: none
    [-1,  0,  0,  0,  1,  0,  0,  0], # 1: a intracellular -> extracellular
    [ 0, -1,  0,  0,  0,  1,  0,  0], # 2: b intracellular -> extracellular
    [ 0,  0, -1,  0,  0,  0,  1,  0], # 3: c intracellular -> extracellular
    [ 0,  0,  0, -1,  0,  0,  0,  1], # 4: d intracellular -> extracellular
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
], dtype=_INT)

_EFFECTOR_M = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 0], # 0: none
    [1, 0, 0, 0, 0, 0, 0, 0], # 1: a intracellular
    [0, 1, 0, 0, 0, 0, 0, 0], # 2: b intracellular
    [0, 0, 1, 0, 0, 0, 0, 0], # 3: c intracellular
    [0, 0, 0, 1, 0, 0, 0, 0], # 4: d intracellular
    [0, 0, 0, 0, 1, 0, 0, 0], # 5: a extracellular
    [0, 0, 0, 0, 0, 1, 0, 0], # 6: b extracellular
    [0, 0, 0, 0, 0, 0, 1, 0], # 7: c extracellular
    [0, 0, 0, 0, 0, 0, 0, 1], # 8: d extracellular
], dtype=_INT)

_REACTION_M = torch.tensor([
    [ 0,  0,  0,  0,  0,  0,  0,  0], # 0: none
    [-1,  1,  0,  0,  0,  0,  0,  0], # 1: a -> b
    [ 0, -1,  1,  0,  0,  0,  0,  0], # 2: b -> c
    [ 0, -1, -1,  1,  0,  0,  0,  0], # 3: b,c -> d
    [ 0,  2,  0, -1,  0,  0,  0,  0], # 4: d -> 2b
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
], dtype=_INT)

# fmt: on


def _get_proteomics() -> Proteomics:
    protics = Proteomics(chemistry=_CHEMISTRY, abs_temp=310, device=DEVICE)
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


def test_cell_params_with_transporter_domains() -> None:
    # Protein: (domains, cds_start, cds_end, is_fwd)
    # Domain: (domain_spec, dom_start, dom_end)
    # Domain spec indexes: (dom_types, reacts_trnspts_effctrs, Vmaxs, Kms, signs)
    # fmt: off
    c0: list[ProteinSpecType] = [
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
    c1: list[ProteinSpecType] = [
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

    # setup proteomics
    c = 2
    p = 3
    m = 8
    protics = _get_proteomics()
    protics.increase_cells(by_n=c)
    protics.increase_proteins(by_n=p)
    protics.set_cell_params(idx=torch.tensor([0, 1]), proteomes=[c0, c1])

    # expected cell params
    k_e_exp = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, _ke([_ma], [_mb]), 1.0],
        ],
        dtype=_FLOAT,
    )
    k_f_exp = torch.tensor(
        [
            [0.5, _avg(0.5, 0.2), 0.0],
            [_avg(0.4, 0.5, 0.6, 0.7), _avg(0.5, 0.5), 0.0],
        ],
        dtype=_FLOAT,
    )
    k_b_exp = torch.tensor(
        [
            [0.5, _avg(0.5, 0.2), 0.0],
            [_avg(0.4, 0.5, 0.6, 0.7), _avg(0.5, 0.5) * _ke([_ma], [_mb]), 0.0],
        ],
        dtype=_FLOAT,
    )
    K_r_exp = torch.zeros(2, 3, 8, dtype=_FLOAT)
    v_max_exp = torch.tensor(
        [
            [1.5, _avg(1.5, 1.1), 0.0],
            [_avg(1.5, 1.4, 1.3, 1.2), _avg(2.0, 1.5), 0.0],
        ],
        dtype=_FLOAT,
    )
    N_f_exp = torch.tensor(
        [
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [2, 1, 1, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ],
        dtype=_INT,
    )
    N_b_exp = torch.tensor(
        [
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 2, 1, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ],
        dtype=_INT,
    )
    N_h_exp = torch.zeros(c, p, m, dtype=_INT)

    # test cell params
    torch.testing.assert_close(protics.k_e, k_e_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.k_f, k_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.k_b, k_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.K_r, K_r_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.v_max, v_max_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.N, N_b_exp - N_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.N_f, N_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.N_b, N_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.N_h, N_h_exp, rtol=_RTOL, atol=_ATOL)

    # test proteome representation

    proteins = protics.get_proteome(proteome=c0)

    p0 = proteins[0]
    assert p0.cds_start == 13
    assert p0.cds_end == 27
    assert p0.is_fwd is True
    assert isinstance(p0.domains[0], TransporterDomain)
    assert p0.domains[0].molecule is _ma
    assert p0.domains[0].vmax == pytest.approx(1.5, abs=_TOL)
    assert p0.domains[0].km == pytest.approx(0.5, abs=_TOL)
    assert p0.domains[0].start == 6
    assert p0.domains[0].end == 27

    p1 = proteins[1]
    assert p1.cds_start == 36
    assert p1.cds_end == 74
    assert p1.is_fwd is False
    assert isinstance(p1.domains[0], TransporterDomain)
    assert p1.domains[0].molecule is _ma
    assert p1.domains[0].vmax == pytest.approx(1.5, abs=_TOL)
    assert p1.domains[0].km == pytest.approx(0.5, abs=_TOL)
    assert p1.domains[0].start == 5
    assert p1.domains[0].end == 13
    assert isinstance(p1.domains[1], TransporterDomain)
    assert p1.domains[1].molecule is _ma
    assert p1.domains[1].vmax == pytest.approx(1.1, abs=_TOL)
    assert p1.domains[1].km == pytest.approx(0.2, abs=_TOL)
    assert p1.domains[1].start == 7
    assert p1.domains[1].end == 12

    proteins = protics.get_proteome(proteome=c1)

    p0 = proteins[0]
    assert p0.cds_start == 91
    assert p0.cds_end == 112
    assert p0.is_fwd is False
    assert isinstance(p0.domains[0], TransporterDomain)
    assert p0.domains[0].molecule is _ma
    assert p0.domains[0].vmax == pytest.approx(1.5, abs=_TOL)
    assert p0.domains[0].km == pytest.approx(0.4, abs=_TOL)
    assert p0.domains[0].start == 1
    assert p0.domains[0].end == 10
    assert isinstance(p0.domains[1], TransporterDomain)
    assert p0.domains[1].molecule is _ma
    assert p0.domains[1].vmax == pytest.approx(1.4, abs=_TOL)
    assert p0.domains[1].km == pytest.approx(0.5, abs=_TOL)
    assert p0.domains[1].start == 2
    assert p0.domains[1].end == 20
    assert isinstance(p0.domains[2], TransporterDomain)
    assert p0.domains[2].molecule is _mb
    assert p0.domains[2].vmax == pytest.approx(1.3, abs=_TOL)
    assert p0.domains[2].km == pytest.approx(0.6, abs=_TOL)
    assert p0.domains[2].start == 3
    assert p0.domains[2].end == 30
    assert isinstance(p0.domains[3], TransporterDomain)
    assert p0.domains[3].molecule is _mc
    assert p0.domains[3].vmax == pytest.approx(1.2, abs=_TOL)
    assert p0.domains[3].km == pytest.approx(0.7, abs=_TOL)
    assert p0.domains[3].start == 4
    assert p0.domains[3].end == 40

    p1 = proteins[1]
    assert p1.cds_start == 1
    assert p1.cds_end == 10
    assert p1.is_fwd is False
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].substrates == [_ma]
    assert p1.domains[0].products == [_mb]
    assert p1.domains[0].vmax == pytest.approx(2.0, abs=_TOL)
    assert p1.domains[0].km == pytest.approx(0.5, abs=_TOL)
    assert p1.domains[0].start == 5
    assert p1.domains[0].end == 50
    assert isinstance(p1.domains[1], TransporterDomain)
    assert p1.domains[1].molecule is _ma
    assert p1.domains[1].vmax == pytest.approx(1.5, abs=_TOL)
    assert p1.domains[1].km == pytest.approx(0.5, abs=_TOL)
    assert p1.domains[1].start == 6
    assert p1.domains[1].end == 60


def test_cell_params_with_regulatory_domains() -> None:
    # Protein: (domains, cds_start, cds_end, is_fwd)
    # Domain: (domain_spec, dom_start, dom_end)
    # Domain spec indexes: (dom_types, reacts_trnspts_effctrs, Vmaxs, Kms, signs)
    # fmt: off
    c0: list[ProteinSpecType] = [
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

    c1: list[ProteinSpecType] = [
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

    # setup proteomics
    c = 2
    p = 3
    protics = _get_proteomics()
    protics.increase_cells(by_n=c)
    protics.increase_proteins(by_n=p)
    protics.set_cell_params(idx=torch.tensor([0, 1]), proteomes=[c0, c1])

    # expected cell params
    ke_a_b = _ke([_ma], [_mb])
    k_e_exp = torch.tensor(
        [
            [ke_a_b, ke_a_b, 1.0],
            [ke_a_b, ke_a_b, 1.0],
        ],
        dtype=_FLOAT,
    )
    k_f_exp = torch.tensor(
        [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=_FLOAT,
    )
    k_b_exp = torch.tensor(
        [
            [0.5 * ke_a_b, 0.5 * ke_a_b, 0.0],
            [0.5 * ke_a_b, 0.5 * ke_a_b, 0.0],
        ],
        dtype=_FLOAT,
    )
    K_r_exp = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.25, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=_FLOAT,
    )
    v_max_exp = torch.tensor(
        [
            [2.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=_FLOAT,
    )
    N_f_exp = torch.tensor(
        [
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ],
        dtype=_INT,
    )
    N_b_exp = torch.tensor(
        [
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ],
        dtype=_INT,
    )
    N_h_exp = torch.tensor(
        [
            [
                [0, 0, 1, -2, 0, 0, 0, 0],
                [1, 0, 0, 0, 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, -1, 0, 0, 0, -3, 0, 0],
                [0, 0, 0, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ],
        dtype=_INT,
    )

    # test cell params
    torch.testing.assert_close(protics.k_e, k_e_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.k_f, k_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.k_b, k_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.K_r, K_r_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.v_max, v_max_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.N, N_b_exp - N_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.N_f, N_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.N_b, N_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.N_h, N_h_exp, rtol=_RTOL, atol=_ATOL)

    # test protein representation

    proteins = protics.get_proteome(proteome=c0)

    p0 = proteins[0]
    assert p0.cds_start == 1
    assert p0.cds_end == 100
    assert p0.is_fwd is False
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].substrates == [_ma]
    assert p0.domains[0].products == [_mb]
    assert p0.domains[0].vmax == pytest.approx(2.0, abs=_TOL)
    assert p0.domains[0].km == pytest.approx(0.5, abs=_TOL)
    assert p0.domains[0].start == 1
    assert p0.domains[0].end == 10
    assert isinstance(p0.domains[1], RegulatoryDomain)
    assert p0.domains[1].effector is _mc
    assert not p0.domains[1].is_inhibiting
    assert not p0.domains[1].is_transmembrane
    assert p0.domains[1].km == pytest.approx(1.0, abs=_TOL)
    assert p0.domains[1].hill == 1
    assert p0.domains[1].start == 2
    assert p0.domains[1].end == 20
    assert isinstance(p0.domains[2], RegulatoryDomain)
    assert p0.domains[2].effector is _md
    assert p0.domains[2].is_inhibiting
    assert not p0.domains[2].is_transmembrane
    assert p0.domains[2].km == pytest.approx(2.0, abs=_TOL)
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
    assert p1.domains[0].vmax == pytest.approx(2.0, abs=_TOL)
    assert p1.domains[0].km == pytest.approx(0.5, abs=_TOL)
    assert p1.domains[0].start == 4
    assert p1.domains[0].end == 40
    assert isinstance(p1.domains[1], RegulatoryDomain)
    assert p1.domains[1].effector is _ma
    assert not p1.domains[1].is_inhibiting
    assert not p1.domains[1].is_transmembrane
    assert p1.domains[1].km == pytest.approx(1.0, abs=_TOL)
    assert p1.domains[1].hill == 1
    assert p1.domains[1].start == 5
    assert p1.domains[1].end == 50
    assert isinstance(p1.domains[2], RegulatoryDomain)
    assert p1.domains[2].effector is _ma
    assert not p1.domains[2].is_inhibiting
    assert p1.domains[2].is_transmembrane
    assert p1.domains[2].km == pytest.approx(1.5, abs=_TOL)
    assert p1.domains[2].hill == 3
    assert p1.domains[2].start == 6
    assert p1.domains[2].end == 60

    proteins = protics.get_proteome(proteome=c1)

    p0 = proteins[0]
    assert p0.cds_start == 3
    assert p0.cds_end == 300
    assert p0.is_fwd is False
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].substrates == [_ma]
    assert p0.domains[0].products == [_mb]
    assert p0.domains[0].vmax == pytest.approx(2.0, abs=_TOL)
    assert p0.domains[0].km == pytest.approx(0.5, abs=_TOL)
    assert p0.domains[0].start == 7
    assert p0.domains[0].end == 70
    assert isinstance(p0.domains[1], RegulatoryDomain)
    assert p0.domains[1].effector is _mb
    assert p0.domains[1].is_inhibiting
    assert not p0.domains[1].is_transmembrane
    assert p0.domains[1].km == pytest.approx(1.0, abs=_TOL)
    assert p0.domains[1].hill == 1
    assert p0.domains[1].start == 8
    assert p0.domains[1].end == 80
    assert isinstance(p0.domains[2], RegulatoryDomain)
    assert p0.domains[2].effector is _mb
    assert p0.domains[2].is_inhibiting
    assert p0.domains[2].is_transmembrane
    assert p0.domains[2].km == pytest.approx(1.5, abs=_TOL)
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
    assert p1.domains[0].vmax == pytest.approx(2.0, abs=_TOL)
    assert p1.domains[0].km == pytest.approx(0.5, abs=_TOL)
    assert p1.domains[0].start == 10
    assert p1.domains[0].end == 100
    assert isinstance(p1.domains[1], RegulatoryDomain)
    assert p1.domains[1].effector is _md
    assert not p1.domains[1].is_inhibiting
    assert not p1.domains[1].is_transmembrane
    assert p1.domains[1].km == pytest.approx(1.0, abs=_TOL)
    assert p1.domains[1].hill == 2
    assert p1.domains[1].start == 11
    assert p1.domains[1].end == 110
    assert isinstance(p1.domains[2], RegulatoryDomain)
    assert p1.domains[2].effector is _md
    assert not p1.domains[2].is_inhibiting
    assert not p1.domains[2].is_transmembrane
    assert p1.domains[2].km == pytest.approx(1.5, abs=_TOL)
    assert p1.domains[2].hill == 3
    assert p1.domains[2].start == 12
    assert p1.domains[2].end == 120


def test_cell_params_with_catalytic_domains() -> None:
    # Protein: (domains, cds_start, cds_end, is_fwd)
    # Domain: (domain_spec, dom_start, dom_end)
    # Domain spec indexes: (dom_types, reacts_trnspts_effctrs, Vmaxs, Kms, signs)
    # fmt: off
    c0: list[ProteinSpecType] = [
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
    c1: list[ProteinSpecType] = [
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

    # setup proteomics
    c = 2
    p = 3
    m = 8
    protics = _get_proteomics()
    protics.increase_cells(by_n=c)
    protics.increase_proteins(by_n=p)
    protics.set_cell_params(idx=torch.tensor([0, 1]), proteomes=[c0, c1])

    # expected cell params
    ke_c0_0 = _ke([_ma, _md], [_mb, _mb, _mc])
    ke_c0_1 = _ke([_mb, _md], [_mc, _mb, _mc])
    ke_c0_2 = _ke([_md], [_mb, _mb])
    ke_c1_0 = _ke([_mb, _md], [_ma, _mb, _mc])
    ke_c1_1 = _ke([_mb, _mb, _mc], [_mc, _md])
    k_e_exp = torch.tensor(
        [
            [ke_c0_0, ke_c0_1, ke_c0_2],
            [ke_c1_0, ke_c1_1, 1.0],
        ],
        dtype=_FLOAT,
    )
    k_f_exp = torch.tensor(
        [
            [_avg(0.5, 1.5) / ke_c0_0, _avg(0.9, 1.2) / ke_c0_1, 2.9 / ke_c0_2],
            [_avg(0.3, 1.4) / ke_c1_0, _avg(0.3, 1.7), 0.0],
        ],
        dtype=_FLOAT,
    )
    k_b_exp = torch.tensor(
        [
            [_avg(0.5, 1.5), _avg(0.9, 1.2), 2.9],
            [_avg(0.3, 1.4), _avg(0.3, 1.7) * ke_c1_1, 0.0],
        ]
    )
    K_r_exp = torch.zeros(c, p, m, dtype=_FLOAT)
    v_max_exp = torch.tensor(
        [
            [_avg(1.1, 1.2), _avg(2.0, 1.3), 2.9],
            [_avg(1.1, 2.1), _avg(1.9, 2.3), 0.0],
        ],
        dtype=_FLOAT,
    )
    N_f_exp = torch.tensor(
        [
            [
                [1, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            [
                [0, 1, 0, 1, 0, 0, 0, 0],
                [0, 2, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ],
        dtype=_INT,
    )
    N_b_exp = torch.tensor(
        [
            [
                [0, 2, 1, 0, 0, 0, 0, 0],
                [0, 1, 2, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ],
        dtype=_INT,
    )
    N_h_exp = torch.zeros(c, p, m, dtype=_INT)

    # test cell params
    torch.testing.assert_close(protics.k_e, k_e_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.k_f, k_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.k_b, k_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.K_r, K_r_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.v_max, v_max_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.N, N_b_exp - N_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.N_f, N_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.N_b, N_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(protics.N_h, N_h_exp, rtol=_RTOL, atol=_ATOL)

    # test protein representation

    proteins = protics.get_proteome(proteome=c0)

    p0 = proteins[0]
    assert p0.cds_start == 1
    assert p0.cds_end == 100
    assert p0.is_fwd is False
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].substrates == [_ma]
    assert p0.domains[0].products == [_mb]
    assert p0.domains[0].vmax == pytest.approx(1.1, abs=_TOL)
    assert p0.domains[0].km == pytest.approx(0.5, abs=_TOL)
    assert p0.domains[0].start == 1
    assert p0.domains[0].end == 10
    assert isinstance(p0.domains[1], CatalyticDomain)
    assert p0.domains[1].substrates == [_md]
    assert p0.domains[1].products == [_mb, _mc]
    assert p0.domains[1].vmax == pytest.approx(1.2, abs=_TOL)
    assert p0.domains[1].km == pytest.approx(1.5, abs=_TOL)
    assert p0.domains[1].start == 2
    assert p0.domains[1].end == 20

    p1 = proteins[1]
    assert p1.cds_start == 2
    assert p1.cds_end == 200
    assert p1.is_fwd is True
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].substrates == [_mb]
    assert p1.domains[0].products == [_mc]
    assert p1.domains[0].vmax == pytest.approx(2.0, abs=_TOL)
    assert p1.domains[0].km == pytest.approx(0.9, abs=_TOL)
    assert p1.domains[0].start == 3
    assert p1.domains[0].end == 30
    assert isinstance(p1.domains[1], CatalyticDomain)
    assert p1.domains[1].substrates == [_md]
    assert p1.domains[1].products == [_mb, _mc]
    assert p1.domains[1].vmax == pytest.approx(1.3, abs=_TOL)
    assert p1.domains[1].km == pytest.approx(1.2, abs=_TOL)
    assert p1.domains[1].start == 4
    assert p1.domains[1].end == 40

    p2 = proteins[2]
    assert p2.cds_start == 3
    assert p2.cds_end == 300
    assert p2.is_fwd is False
    assert isinstance(p2.domains[0], CatalyticDomain)
    assert p2.domains[0].substrates == [_md]
    assert p2.domains[0].products == [_mb, _mb]
    assert p2.domains[0].vmax == pytest.approx(2.9, abs=_TOL)
    assert p2.domains[0].km == pytest.approx(2.9, abs=_TOL)
    assert p2.domains[0].start == 5
    assert p2.domains[0].end == 50

    proteins = protics.get_proteome(proteome=c1)

    p0 = proteins[0]
    assert p0.cds_start == 4
    assert p0.cds_end == 400
    assert p0.is_fwd is True
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].substrates == [_mb]
    assert p0.domains[0].products == [_ma]
    assert p0.domains[0].vmax == pytest.approx(1.1, abs=_TOL)
    assert p0.domains[0].km == pytest.approx(0.3, abs=_TOL)
    assert p0.domains[0].start == 6
    assert p0.domains[0].end == 60
    assert isinstance(p0.domains[1], CatalyticDomain)
    assert p0.domains[1].substrates == [_md]
    assert p0.domains[1].products == [_mb, _mc]
    assert p0.domains[1].vmax == pytest.approx(2.1, abs=_TOL)
    assert p0.domains[1].km == pytest.approx(1.4, abs=_TOL)
    assert p0.domains[1].start == 7
    assert p0.domains[1].end == 70

    p1 = proteins[1]
    assert p1.cds_start == 5
    assert p1.cds_end == 500
    assert p1.is_fwd is False
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].substrates == [_mb]
    assert p1.domains[0].products == [_mc]
    assert p1.domains[0].vmax == pytest.approx(1.9, abs=_TOL)
    assert p1.domains[0].km == pytest.approx(0.3, abs=_TOL)
    assert p1.domains[0].start == 8
    assert p1.domains[0].end == 80
    assert isinstance(p1.domains[1], CatalyticDomain)
    assert p1.domains[1].substrates == [_mb, _mc]
    assert p1.domains[1].products == [_md]
    assert p1.domains[1].vmax == pytest.approx(2.3, abs=_TOL)
    assert p1.domains[1].km == pytest.approx(1.7, abs=_TOL)
    assert p1.domains[1].start == 9
    assert p1.domains[1].end == 90


def _get_random_proteomes(n_cells: int) -> tuple[list[list[ProteinSpecType]], int]:
    proteomes: list[list[ProteinSpecType]] = []
    p_max = 0

    for cell_i in range(n_cells):
        prots: list[ProteinSpecType] = []
        n_prots = random.choice(range(1, 10))

        for prot_i in range(n_prots):
            doms: list[DomainSpecType] = []
            n_doms = random.choice(range(1, 4))

            for dom_i in range(n_doms):
                domtype = random.choice([1, 2, 3])
                idx0 = random.choice(range(20 if domtype == 1 else 6))
                idx1 = random.choice(range(30))
                idx2 = random.choice(range(3))
                idx3 = random.choice(range(9))
                dom_start = random.choice(range(1, 100))
                dom_end = random.choice(range(dom_start, 200))
                doms.append(((domtype, idx0, idx1, idx2, idx3), dom_start, dom_end))

            dom_start = random.choice(range(1, 100))
            dom_end = random.choice(range(dom_start, 200))
            is_fwd = random.choice([True, False])
            prots.append((doms, dom_start, dom_end, is_fwd))

        p_max = max(p_max, len(prots))
        proteomes.append(prots)

    return proteomes, p_max


@pytest.mark.slow
def test_random_cell_params() -> None:
    for _ in range(100):
        m = len(_MOLECULES) * 2
        n_cells = random.choice(range(1, 10))
        proteomes, p_max = _get_random_proteomes(n_cells)

        # setup proteomics
        protics = _get_proteomics()
        protics.increase_cells(by_n=n_cells)
        protics.increase_proteins(by_n=p_max)
        idx = torch.tensor(range(n_cells))
        protics.set_cell_params(idx=idx, proteomes=proteomes)

        # test
        float_np = [protics.k_e, protics.k_f, protics.k_b, protics.v_max]
        float_npm = [protics.K_r]
        int_npm = [protics.N, protics.N_f, protics.N_b, protics.N_h]

        for t in float_np:
            assert t.dtype == _FLOAT
            assert t.shape == (n_cells, p_max)

        for t in float_npm:
            assert t.dtype == _FLOAT
            assert t.shape == (n_cells, p_max, m)

        for t in int_npm:
            assert t.dtype == _INT
            assert t.shape == (n_cells, p_max, m)

        for t in float_np + float_npm + int_npm:
            assert not torch.any(t.isnan())
            assert torch.all(t.isfinite())


# TODO: test adjusting Tensors with specific examples


def test_adjust_tensor_sizes_randomly() -> None:
    m = len(_MOLECULES) * 2
    c = 10
    proteomes, p = _get_random_proteomes(c)

    # setup proteomics
    protics = _get_proteomics()
    protics.increase_cells(by_n=c)
    protics.increase_proteins(by_n=p)
    idx = torch.tensor(range(c))
    protics.set_cell_params(idx=idx, proteomes=proteomes)

    for _ in range(100):
        dc = random.choice(range(-c, 10))
        dp = random.choice(range(-p, 10))
        if dc > 0:
            protics.increase_cells(by_n=dc)
        if dc < 0:
            keep = torch.tensor(random.sample(range(c), c + dc))
            protics.decrease_cells(keep_idx=keep.int())
        if dp > 0:
            protics.increase_proteins(by_n=dp)
        if dp < 0:
            protics.decrease_proteins(by_n=-dp)

        c += dc
        p += dp

        # test
        float_np = [protics.k_e, protics.k_f, protics.k_b, protics.v_max]
        float_npm = [protics.K_r]
        int_npm = [protics.N, protics.N_f, protics.N_b, protics.N_h]

        for t in float_np:
            assert t.dtype == _FLOAT
            assert t.shape == (c, p)

        for t in float_npm:
            assert t.dtype == _FLOAT
            assert t.shape == (c, p, m)

        for t in int_npm:
            assert t.dtype == _INT
            assert t.shape == (c, p, m)

        for t in float_np + float_npm + int_npm:
            assert not torch.any(t.isnan())
            assert torch.all(t.isfinite())
