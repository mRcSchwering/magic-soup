import math
import random
from typing import Any

import numpy as np
import pytest
import torch
from magicsoup.biology import CatalyticDomain, RegulatoryDomain, TransporterDomain
from magicsoup.chemistry import Chemistry, Molecule, Reaction
from magicsoup.config import GAS_CONSTANT
from magicsoup.genomics import DomainSpec, DomainTokens, ProteinSpec
from magicsoup.proteomics import (
    Proteomics,
    _EffectorMap,
    _LogNormMap,
    _MirrGeoMap,
    _ReactionMap,
    _TransporterMap,
)

from tests.util import FTYPE, ITYPE, ftensor, fzeros, itensor, izeros

_TOL = 1e-4
_ATOL = 1e-4
_RTOL = 1e-4


_ma = Molecule("a", energy=15)
_mb = Molecule("b", energy=10)
_mc = Molecule("c", energy=10)
_md = Molecule("d", energy=5)
_MOLECULES = [_ma, _mb, _mc, _md]

_r_a_b = Reaction([_ma], [_mb])
_r_b_c = Reaction([_mb], [_mc])
_r_bc_d = Reaction([_mb, _mc], [_md])
_r_d_bb = Reaction([_md], 2 * [_mb])
_REACTIONS = [_r_a_b, _r_b_c, _r_bc_d, _r_d_bb]

_CHEMISTRY = Chemistry(molecules=_MOLECULES, reactions=_REACTIONS)


# fmt: off

# token 0 always represents "no value"

_KMS = [
#   x0    x1    x2    x3    x4    x5    x6    x7    x8    x9
    0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  # 0x
    1.0,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  # 1x
    2.0,  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  # 2x
]

_VMAXS = [
#   x0    x1    x2    x3    x4    x5    x6    x7    x8    x9
    0.0,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  # 0x
    2.0,  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  # 1x
]

#         0   1   2
_SIGNS = [0,  1, -1]


_HILLS = [
#   x0   x1   x2   x3   x4   x5   x6  x7  x8  x9
    0,   1,   2,   3,   -1,  -2,  -3, 0,  0,  0,  # 0x
    0,   0,   0,   0,   0,   0,   0,  0,  0,  0,  # 1x
]

_TRANSPORT_M = [
    ( 0,  0,  0,  0,  0,  0,  0,  0 ), # 0: none
    (-1,  0,  0,  0,  1,  0,  0,  0 ), # 1: a intracellular -> extracellular
    ( 0, -1,  0,  0,  0,  1,  0,  0 ), # 2: b intracellular -> extracellular
    ( 0,  0, -1,  0,  0,  0,  1,  0 ), # 3: c intracellular -> extracellular
    ( 0,  0,  0, -1,  0,  0,  0,  1 ), # 4: d intracellular -> extracellular
    ( 0,  0,  0,  0,  0,  0,  0,  0 ),
    ( 0,  0,  0,  0,  0,  0,  0,  0 ),
    ( 0,  0,  0,  0,  0,  0,  0,  0 ),
    ( 0,  0,  0,  0,  0,  0,  0,  0 ),
]

_EFFECTOR_M = [
    ( 0, 0, 0, 0, 0, 0, 0, 0 ), # 0: none
    ( 1, 0, 0, 0, 0, 0, 0, 0 ), # 1: a intracellular
    ( 0, 1, 0, 0, 0, 0, 0, 0 ), # 2: b intracellular
    ( 0, 0, 1, 0, 0, 0, 0, 0 ), # 3: c intracellular
    ( 0, 0, 0, 1, 0, 0, 0, 0 ), # 4: d intracellular
    ( 0, 0, 0, 0, 1, 0, 0, 0 ), # 5: a extracellular
    ( 0, 0, 0, 0, 0, 1, 0, 0 ), # 6: b extracellular
    ( 0, 0, 0, 0, 0, 0, 1, 0 ), # 7: c extracellular
    ( 0, 0, 0, 0, 0, 0, 0, 1 ), # 8: d extracellular
]

_REACTION_M = [
    ( 0,  0,  0,  0,  0,  0,  0,  0 ), # 0: none
    (-1,  1,  0,  0,  0,  0,  0,  0 ), # 1: a -> b
    ( 0, -1,  1,  0,  0,  0,  0,  0 ), # 2: b -> c
    ( 0, -1, -1,  1,  0,  0,  0,  0 ), # 3: b,c -> d
    ( 0,  2,  0, -1,  0,  0,  0,  0 ), # 4: d -> 2b
    ( 0,  0,  0,  0,  0,  0,  0,  0 ),
    ( 0,  0,  0,  0,  0,  0,  0,  0 ),
    ( 0,  0,  0,  0,  0,  0,  0,  0 ),
    ( 0,  0,  0,  0,  0,  0,  0,  0 ),
]

_V_CELL = 1.0
_V_MED = 1000.0
_V_RATIOS = [
    1/_V_CELL,  1/_V_CELL,  1/_V_CELL,  1/_V_CELL,  
    1/_V_MED,   1/_V_MED,   1/_V_MED,   1/_V_MED
]

# fmt: on


def _get_proteomics() -> Proteomics:
    protics = Proteomics(chemistry=_CHEMISTRY, abs_temp=310)
    protics.km_map.token_2_scalar = {i: d for i, d in enumerate(_KMS)}
    protics.km_map.token_scalar = ftensor(_KMS)
    protics.vmax_map.token_2_scalar = {i: d for i, d in enumerate(_VMAXS)}
    protics.vmax_map.token_scalar = ftensor(_VMAXS)
    protics.sign_map.token_2_scalar = {i: d for i, d in enumerate(_SIGNS)}
    protics.sign_map.token_scalar = itensor(_SIGNS)
    protics.hill_map.token_2_scalar = {i: d for i, d in enumerate(_HILLS)}
    protics.hill_map.token_scalar = itensor(_HILLS)
    protics.transport_map.token_2_vector = {i: d for i, d in enumerate(_TRANSPORT_M)}
    protics.transport_map.token_vector = itensor(_TRANSPORT_M)
    protics.effector_map.token_2_vector = {i: d for i, d in enumerate(_EFFECTOR_M)}
    protics.effector_map.token_vector = itensor(_EFFECTOR_M)
    protics.reaction_map.token_2_vector = {i: d for i, d in enumerate(_REACTION_M)}
    N = itensor(_REACTION_M)
    protics.reaction_map.token_fwd_vector = torch.where(N < 0, -N, 0)
    protics.reaction_map.token_bwd_vector = torch.where(N > 0, N, 0)
    protics._setup_rs()
    return protics


def _ke(subs: list[Molecule], prods: list[Molecule]) -> float:
    e = sum(d.energy for d in prods) - sum(d.energy for d in subs)
    return math.exp(-e * 1e3 / 310 / GAS_CONSTANT)


def _avg(*x: float) -> float:
    return sum(x) / len(x)


def _flatten_proteomes(
    proteomes: list[list[ProteinSpec]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells: list[int] = []
    proteins: list[int] = []
    tokens: list[DomainTokens] = []
    for cell_i, cell_spec in enumerate(proteomes):
        for protein_i, protein_spec in enumerate(cell_spec):
            for domain_spec in protein_spec[0]:
                cells.append(cell_i)
                proteins.append(protein_i)
                tokens.append(domain_spec[0])

    return np.array(cells), np.array(proteins), np.array(tokens)


def test_cell_params_with_catalytic_domains() -> None:
    # Protein: (domains, cds_start, cds_end, is_fwd)
    # Domain: (domain_spec, dom_start, dom_end)
    # Domain spec indexes: (dom_types, reacts_trnspts_effctrs, Vmaxs, Kms, signs)
    # fmt: off
    c0: list[ProteinSpec] = [
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
    c1: list[ProteinSpec] = [
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
    # TODO: add a cofactor within one reaction definiton
    # fmt: on

    # setup proteomics
    c = 2  # 2 cells
    p = 3  # 2 max proteins
    m = 8  # 8 molecules
    protics = _get_proteomics()
    params = protics.get_cell_params(proteomes=[c0, c1])
    proteomes_flat = _flatten_proteomes(proteomes=[c0, c1])
    params2 = protics.get_cell_params2(*proteomes_flat, c=c, p=p)

    # expected cell params
    ke_c0_0 = _ke([_ma, _md], [_mb, _mb, _mc])
    ke_c0_1 = _ke([_mb, _md], [_mc, _mb, _mc])
    ke_c0_2 = _ke([_md], [_mb, _mb])
    ke_c1_0 = _ke([_mb, _md], [_ma, _mb, _mc])
    ke_c1_1 = _ke([_mb, _mb, _mc], [_mc, _md])
    k_f_exp = ftensor(
        [
            [_avg(0.5, 1.5) / ke_c0_0, _avg(0.9, 1.2) / ke_c0_1, 2.9 / ke_c0_2],
            [_avg(0.3, 1.4) / ke_c1_0, _avg(0.3, 1.7), 0.0],
        ]
    )
    k_b_exp = ftensor(
        [
            [_avg(0.5, 1.5), _avg(0.9, 1.2), 2.9],
            [_avg(0.3, 1.4), _avg(0.3, 1.7) * ke_c1_1, 0.0],
        ]
    )
    K_r_exp = fzeros(c, p, m)
    v_max_exp = ftensor(
        [
            [_avg(1.1, 1.2), _avg(2.0, 1.3), 2.9],
            [_avg(1.1, 2.1), _avg(1.9, 2.3), 0.0],
        ]
    )
    N_f_exp = itensor(
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
        ]
    )
    N_b_exp = itensor(
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
        ]
    )
    N_h_exp = izeros(c, p, m)

    N = N_b_exp - N_f_exp
    N_eff = N * ftensor(_V_RATIOS)

    # test cell params
    torch.testing.assert_close(params["k_f"], k_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["k_b"], k_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["K_r"], K_r_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["v_max"], v_max_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["N_e"], N_eff, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["N_f"], N_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["N_b"], N_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["N_h"], N_h_exp, rtol=_RTOL, atol=_ATOL)

    # test cell params2
    torch.testing.assert_close(params2["k_f"], k_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["k_b"], k_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["K_r"], K_r_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["v_max"], v_max_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["N_e"], N_eff, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["N_f"], N_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["N_b"], N_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["N_h"], N_h_exp, rtol=_RTOL, atol=_ATOL)

    # test protein representation

    proteins = protics.get_proteome(proteome=c0)

    p0 = proteins[0]
    assert p0.cds_start == 1
    assert p0.cds_end == 100
    assert p0.is_53 is False
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].reaction == _r_a_b
    assert p0.domains[0].is_fwd
    assert p0.domains[0].vmax == pytest.approx(1.1, abs=_TOL)
    assert p0.domains[0].km == pytest.approx(0.5, abs=_TOL)
    assert p0.domains[0].start == 1
    assert p0.domains[0].end == 10
    assert isinstance(p0.domains[1], CatalyticDomain)
    assert p0.domains[1].reaction == _r_bc_d
    assert not p0.domains[1].is_fwd
    assert p0.domains[1].vmax == pytest.approx(1.2, abs=_TOL)
    assert p0.domains[1].km == pytest.approx(1.5, abs=_TOL)
    assert p0.domains[1].start == 2
    assert p0.domains[1].end == 20

    p1 = proteins[1]
    assert p1.cds_start == 2
    assert p1.cds_end == 200
    assert p1.is_53 is True
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].reaction == _r_b_c
    assert p1.domains[0].is_fwd
    assert p1.domains[0].vmax == pytest.approx(2.0, abs=_TOL)
    assert p1.domains[0].km == pytest.approx(0.9, abs=_TOL)
    assert p1.domains[0].start == 3
    assert p1.domains[0].end == 30
    assert isinstance(p1.domains[1], CatalyticDomain)
    assert p1.domains[1].reaction == _r_bc_d
    assert not p1.domains[1].is_fwd
    assert p1.domains[1].vmax == pytest.approx(1.3, abs=_TOL)
    assert p1.domains[1].km == pytest.approx(1.2, abs=_TOL)
    assert p1.domains[1].start == 4
    assert p1.domains[1].end == 40

    p2 = proteins[2]
    assert p2.cds_start == 3
    assert p2.cds_end == 300
    assert p2.is_53 is False
    assert isinstance(p2.domains[0], CatalyticDomain)
    assert p2.domains[0].reaction == _r_d_bb
    assert p2.domains[0].is_fwd
    assert p2.domains[0].vmax == pytest.approx(2.9, abs=_TOL)
    assert p2.domains[0].km == pytest.approx(2.9, abs=_TOL)
    assert p2.domains[0].start == 5
    assert p2.domains[0].end == 50

    proteins = protics.get_proteome(proteome=c1)

    p0 = proteins[0]
    assert p0.cds_start == 4
    assert p0.cds_end == 400
    assert p0.is_53 is True
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].reaction == _r_a_b
    assert not p0.domains[0].is_fwd
    assert p0.domains[0].vmax == pytest.approx(1.1, abs=_TOL)
    assert p0.domains[0].km == pytest.approx(0.3, abs=_TOL)
    assert p0.domains[0].start == 6
    assert p0.domains[0].end == 60
    assert isinstance(p0.domains[1], CatalyticDomain)
    assert p0.domains[1].reaction == _r_bc_d
    assert not p0.domains[1].is_fwd
    assert p0.domains[1].vmax == pytest.approx(2.1, abs=_TOL)
    assert p0.domains[1].km == pytest.approx(1.4, abs=_TOL)
    assert p0.domains[1].start == 7
    assert p0.domains[1].end == 70

    p1 = proteins[1]
    assert p1.cds_start == 5
    assert p1.cds_end == 500
    assert p1.is_53 is False
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].reaction == _r_b_c
    assert p1.domains[0].is_fwd
    assert p1.domains[0].vmax == pytest.approx(1.9, abs=_TOL)
    assert p1.domains[0].km == pytest.approx(0.3, abs=_TOL)
    assert p1.domains[0].start == 8
    assert p1.domains[0].end == 80
    assert isinstance(p1.domains[1], CatalyticDomain)
    assert p1.domains[1].reaction == _r_bc_d
    assert p1.domains[1].is_fwd
    assert p1.domains[1].vmax == pytest.approx(2.3, abs=_TOL)
    assert p1.domains[1].km == pytest.approx(1.7, abs=_TOL)
    assert p1.domains[1].start == 9
    assert p1.domains[1].end == 90


def test_cell_params_with_transporter_domains() -> None:
    # Protein: (domains, cds_start, cds_end, is_fwd)
    # Domain: (domain_spec, dom_start, dom_end)
    # Domain spec indexes: (dom_types, reacts_trnspts_effctrs, Vmaxs, Kms, signs)
    # fmt: off
    c0: list[ProteinSpec] = [
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
    c1: list[ProteinSpec] = [
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
    c = 2  # 2 cells
    p = 2  # 2 max proteins
    m = 8  # 8 molecules
    protics = _get_proteomics()
    params = protics.get_cell_params(proteomes=[c0, c1])
    proteomes_flat = _flatten_proteomes(proteomes=[c0, c1])
    params2 = protics.get_cell_params2(*proteomes_flat, c=c, p=p)

    # expected cell params
    k_f_exp = ftensor(
        [
            [0.5, _avg(0.5, 0.2)],
            [_avg(0.4, 0.5, 0.6, 0.7), _avg(0.5, 0.5)],
        ]
    )
    k_b_exp = ftensor(
        [
            [0.5, _avg(0.5, 0.2)],
            [_avg(0.4, 0.5, 0.6, 0.7), _avg(0.5, 0.5) * _ke([_ma], [_mb])],
        ]
    )
    K_r_exp = fzeros(c, p, 8)
    v_max_exp = ftensor(
        [
            [1.5, _avg(1.5, 1.1)],
            [_avg(1.5, 1.4, 1.3, 1.2), _avg(2.0, 1.5)],
        ]
    )
    N_f_exp = itensor(
        [
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0],
            ],
            [
                [2, 1, 1, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    N_b_exp = itensor(
        [
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 2, 1, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 0],
            ],
        ]
    )
    N_h_exp = izeros(c, p, m)

    N = N_b_exp - N_f_exp
    N_eff = N * ftensor(_V_RATIOS)

    # test cell params
    torch.testing.assert_close(params["k_f"], k_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["k_b"], k_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["K_r"], K_r_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["v_max"], v_max_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["N_e"], N_eff, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["N_f"], N_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["N_b"], N_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["N_h"], N_h_exp, rtol=_RTOL, atol=_ATOL)

    # test cell params2
    torch.testing.assert_close(params2["k_f"], k_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["k_b"], k_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["K_r"], K_r_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["v_max"], v_max_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["N_e"], N_eff, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["N_f"], N_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["N_b"], N_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["N_h"], N_h_exp, rtol=_RTOL, atol=_ATOL)

    # test proteome representation

    proteins = protics.get_proteome(proteome=c0)

    p0 = proteins[0]
    assert p0.cds_start == 13
    assert p0.cds_end == 27
    assert p0.is_53 is True
    assert isinstance(p0.domains[0], TransporterDomain)
    assert p0.domains[0].molecule is _ma
    assert p0.domains[0].vmax == pytest.approx(1.5, abs=_TOL)
    assert p0.domains[0].km == pytest.approx(0.5, abs=_TOL)
    assert p0.domains[0].start == 6
    assert p0.domains[0].end == 27

    p1 = proteins[1]
    assert p1.cds_start == 36
    assert p1.cds_end == 74
    assert p1.is_53 is False
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
    assert p0.is_53 is False
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
    assert p1.is_53 is False
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].reaction == _r_a_b
    assert p1.domains[0].is_fwd
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
    c0: list[ProteinSpec] = [
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

    c1: list[ProteinSpec] = [
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
    c = 2  # 2 cells
    p = 2  # 2 max proteins
    protics = _get_proteomics()
    params = protics.get_cell_params(proteomes=[c0, c1])
    proteomes_flat = _flatten_proteomes(proteomes=[c0, c1])
    params2 = protics.get_cell_params2(*proteomes_flat, c=c, p=p)

    # expected cell params
    ke_a_b = _ke([_ma], [_mb])
    k_f_exp = ftensor(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )
    k_b_exp = ftensor(
        [
            [0.5 * ke_a_b, 0.5 * ke_a_b],
            [0.5 * ke_a_b, 0.5 * ke_a_b],
        ]
    )
    K_r_exp = ftensor(
        [
            [
                [0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.25, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    v_max_exp = ftensor(
        [
            [2.0, 2.0],
            [2.0, 2.0],
        ]
    )
    N_f_exp = itensor(
        [
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    N_b_exp = itensor(
        [
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
            ],
        ]
    )
    N_h_exp = itensor(
        [
            [
                [0, 0, 1, -2, 0, 0, 0, 0],
                [1, 0, 0, 0, 3, 0, 0, 0],
            ],
            [
                [0, -1, 0, 0, 0, -3, 0, 0],
                [0, 0, 0, 5, 0, 0, 0, 0],
            ],
        ]
    )

    N = N_b_exp - N_f_exp
    N_eff = N * ftensor(_V_RATIOS)

    # test cell params
    torch.testing.assert_close(params["k_f"], k_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["k_b"], k_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["K_r"], K_r_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["v_max"], v_max_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["N_e"], N_eff, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["N_f"], N_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["N_b"], N_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params["N_h"], N_h_exp, rtol=_RTOL, atol=_ATOL)

    print(K_r_exp)

    # test cell params2
    torch.testing.assert_close(params2["k_f"], k_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["k_b"], k_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["K_r"], K_r_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["v_max"], v_max_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["N_e"], N_eff, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["N_f"], N_f_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["N_b"], N_b_exp, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(params2["N_h"], N_h_exp, rtol=_RTOL, atol=_ATOL)

    # test protein representation

    proteins = protics.get_proteome(proteome=c0)

    p0 = proteins[0]
    assert p0.cds_start == 1
    assert p0.cds_end == 100
    assert p0.is_53 is False
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].reaction == _r_a_b
    assert p0.domains[0].is_fwd
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
    assert p1.is_53 is True
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].reaction == _r_a_b
    assert p0.domains[0].is_fwd
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
    assert p0.is_53 is False
    assert isinstance(p0.domains[0], CatalyticDomain)
    assert p0.domains[0].reaction == _r_a_b
    assert p0.domains[0].is_fwd
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
    assert p1.is_53 is True
    assert isinstance(p1.domains[0], CatalyticDomain)
    assert p1.domains[0].reaction == _r_a_b
    assert p0.domains[0].is_fwd
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


def _get_random_proteomes(n_cells: int) -> tuple[list[list[ProteinSpec]], int]:
    proteomes: list[list[ProteinSpec]] = []
    p_max = 0

    for _ in range(n_cells):
        prots: list[ProteinSpec] = []
        n_prots = random.choice(range(1, 10))

        for _ in range(n_prots):
            doms: list[DomainSpec] = []
            n_doms = random.choice(range(1, 4))

            for _ in range(n_doms):
                domtype = random.choice([1, 2, 3])
                idx0 = random.choice(range(1, 64 - 3))
                idx1 = random.choice(range(1, 64 - 3))
                idx2 = random.choice(range(1, 64 - 3))
                idx3 = random.choice(range(1, 4096 - 3 * 64))
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
        protics = Proteomics(chemistry=_CHEMISTRY)
        params = protics.get_cell_params(proteomes=proteomes)

        # test
        float_np = [params["k_f"], params["k_b"], params["v_max"]]
        float_npm = [params["K_r"], params["N_e"]]
        int_npm = [params["N_f"], params["N_b"], params["N_h"]]

        for t in float_np:
            assert t.dtype == FTYPE
            assert t.shape == (n_cells, p_max)

        for t in float_npm:
            assert t.dtype == FTYPE
            assert t.shape == (n_cells, p_max, m)

        for t in int_npm:
            assert t.dtype == ITYPE
            assert t.shape == (n_cells, p_max, m)

        for t in float_np + float_npm + int_npm:
            assert not torch.any(t.isnan())
            assert torch.all(t.isfinite())


def test_random_cell_reprs() -> None:
    proteomes, _ = _get_random_proteomes(1000)
    protics = Proteomics(chemistry=_CHEMISTRY)

    # TODO: check that the amount of proteins
    #       in the parameters reflects the number of proteins
    #       in the proteomes

    # test
    for proteome in proteomes:
        _ = protics.get_proteome(proteome=proteome)


def _assert_bijective(
    all_tokens: set[int], tokens_map: dict[Any, list[int]], value_map: dict[int, Any]
) -> None:
    # check that all tokens are defined
    scalar_tokens = {dd for d in tokens_map.values() for dd in d}
    tokens = set(value_map)
    assert all_tokens == tokens
    assert all_tokens == scalar_tokens

    # check that mapping is bijective
    assert len(scalar_tokens) == len([dd for d in tokens_map.values() for dd in d])


def test_log_norm_map() -> None:
    max_token = 100
    value_range = (1e-2, 1e2)
    mapper = _LogNormMap(max_token=max_token, value_range=value_range)
    all_tokens = set(range(max_token + 1))  # first token is none token
    _assert_bijective(all_tokens, mapper.scalar_2_tokens, mapper.token_2_scalar)

    # check inversion is correct
    for token in all_tokens:
        scalar = mapper.token_2_scalar[token]

        # none scalar
        if scalar == 0:
            assert token == 0
            assert token in mapper.scalar_2_tokens[None]
            continue

        # valued scalar
        assert token in mapper.scalar_2_tokens[scalar]

    # distribution fits
    values = list(mapper.token_2_scalar.values())[1:]
    lo, hi = value_range
    n = len(values)
    assert min(values) > lo / 100
    assert max(values) < hi * 100
    assert len([d for d in values if d < lo]) / n < 0.1
    assert len([d for d in values if d > hi]) / n < 0.1
    exp = math.exp((math.log(lo) + math.log(hi)) / 2)
    act = math.exp(sum(math.log(d) for d in values) / n)
    assert abs(exp - act) < 0.8  # very unstable


def test_mirr_geo_map() -> None:
    max_token = 100
    max_value = 4
    mapper = _MirrGeoMap(max_token=max_token, max_value=max_value)
    all_tokens = set(range(max_token + 1))  # first token is none token
    _assert_bijective(all_tokens, mapper.scalar_2_tokens, mapper.token_2_scalar)

    # check inversion is correct
    for token in all_tokens:
        scalar = mapper.token_2_scalar[token]

        # none scalar
        if scalar == 0:
            assert token == 0
            assert token in mapper.scalar_2_tokens[None]
            continue

        # valued scalar
        assert token in mapper.scalar_2_tokens[scalar]

    # distribution fits
    assert 0 not in mapper.scalar_2_tokens
    assert max_value + 1 not in mapper.scalar_2_tokens
    assert mapper.scalar_2_tokens[None] == [0]
    v = max_value
    p0l = len(mapper.scalar_2_tokens[-v]) / max_token
    p0r = len(mapper.scalar_2_tokens[v]) / max_token
    while True:
        v -= 1
        if v == 0:
            break

        p1l = len(mapper.scalar_2_tokens[-v]) / max_token
        p1r = len(mapper.scalar_2_tokens[v]) / max_token
        assert abs(p0l - 2 * p1l) < 0.7  # unstable
        assert abs(p0r - 2 * p1r) < 0.7  # unstable
        p0l = p1l
        p0r = p1r


def test_reaction_map() -> None:
    # it's possible to init without reactions
    chemistry = Chemistry(molecules=_MOLECULES, reactions=[])
    _ = _ReactionMap(chemistry=chemistry, max_token=20)

    # setup actual mapper
    chemistry = Chemistry(molecules=_MOLECULES, reactions=_REACTIONS)
    idx_2_mol = chemistry.idx_2_mol
    r = len(chemistry.reactions)

    max_token = 20
    mapper = _ReactionMap(chemistry=chemistry, max_token=max_token)
    all_tokens = set(range(max_token + 1))  # first token is none token
    _assert_bijective(all_tokens, mapper.react_2_tokens, mapper.token_2_vector)

    # check inversion is correct
    for token in all_tokens:
        vector = mapper.token_2_vector[token]
        subs = [[idx_2_mol[i]] * -d for i, d in enumerate(vector) if d < 0]
        prds = [[idx_2_mol[i]] * d for i, d in enumerate(vector) if d > 0]

        # none vector
        if len(subs) == 0 and len(prds) == 0:
            assert token == 0
            assert all(d == 0 for d in vector)
            assert token in mapper.react_2_tokens[None]
            continue

        # valued vector
        assert len(subs) > 0 and len(prds) > 0
        react = Reaction(
            substrates=[dd for d in subs for dd in d],
            products=[dd for d in prds for dd in d],
        )
        assert token in mapper.react_2_tokens[react]

    # check that all items are covered evenly
    n_exp = max_token // r
    assert mapper.react_2_tokens[None] == [0]
    for react in chemistry.reactions:
        assert len(mapper.react_2_tokens[react]) >= n_exp


def test_transporter_map() -> None:
    chemistry = Chemistry(molecules=_MOLECULES, reactions=[])
    mol_2_idx = chemistry.mol_2_idx
    m = len(mol_2_idx)

    idx_2_mol = {v: k for k, v in mol_2_idx.items()}
    idx_2_mol = {**idx_2_mol, **{k + m: v for k, v in idx_2_mol.items()}}

    max_token = 20
    mapper = _TransporterMap(chemistry=chemistry, max_token=max_token)
    all_tokens = set(range(max_token + 1))  # first token is none token
    _assert_bijective(all_tokens, mapper.mol_2_tokens, mapper.token_2_vector)

    # check inversion is correct
    for token in all_tokens:
        vector = mapper.token_2_vector[token]
        mols = [idx_2_mol[i] for i, d in enumerate(vector) if d != 0]

        # none vector
        if len(mols) == 0:
            assert token == 0
            assert all(d == 0 for d in vector)
            assert token in mapper.mol_2_tokens[None]
            continue

        # valued vector
        assert len(mols) == 2
        assert mols[0] is mols[1]
        assert token in mapper.mol_2_tokens[mols[0]]

    # check that all items are covered evenly
    n_exp = max_token // m
    assert mapper.mol_2_tokens[None] == [0]
    for mol in mol_2_idx:
        assert len(mapper.mol_2_tokens[mol]) >= n_exp


def test_effector_map() -> None:
    chemistry = Chemistry(molecules=_MOLECULES, reactions=[])
    mol_2_idx = chemistry.mol_2_idx
    m = len(mol_2_idx)
    m2 = 2 * m

    idx_2_mol = {v: k for k, v in mol_2_idx.items()}
    idx_2_mol = {**idx_2_mol, **{k + m: v for k, v in idx_2_mol.items()}}

    max_token = 20
    mapper = _EffectorMap(chemistry=chemistry, max_token=max_token)
    all_tokens = set(range(max_token + 1))  # first token is none token
    _assert_bijective(all_tokens, mapper.mol_2_tokens, mapper.token_2_vector)

    # check inversion is correct
    for token in all_tokens:
        vector = mapper.token_2_vector[token]
        mols = [idx_2_mol[i] for i, d in enumerate(vector) if d != 0]

        # none vector
        if len(mols) == 0:
            assert token == 0
            assert all(d == 0 for d in vector)
            assert token in mapper.mol_2_tokens[None]
            continue

        # valued vector
        assert len(mols) == 1
        assert token in mapper.mol_2_tokens[mols[0]]

    # check that all items are covered evenly
    n_exp = max_token // m2
    assert mapper.mol_2_tokens[None] == [0]
    for mol in mol_2_idx:
        assert len(mapper.mol_2_tokens[mol]) >= n_exp
