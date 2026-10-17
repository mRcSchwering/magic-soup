import math
import random

import pytest
import torch
from magicsoup.physics import Degradation, Diffusion, Permeation

from tests.config import DEVICE, FTYPE
from tests.util import frand, ftensor, fzeros

_ATOL = 1e-4
_RTOL = 1e-4


def test_diffusion():
    molecule_map = ftensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    exp = ftensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [9.6098e-01, 9.6094e-03, 4.9151e-05, 1.4459e-04, 9.6103e-03],
                [9.6098e-03, 9.6565e-05, 9.7047e-05, 9.6580e-03, 1.9266e-04],
                [9.6893e-05, 4.9156e-05, 9.7054e-03, 9.7059e-01, 9.7059e-03],
                [9.6893e-05, 4.9156e-05, 9.7054e-03, 9.7059e-01, 9.7059e-03],
                [9.6098e-03, 9.6565e-05, 9.7047e-05, 9.6580e-03, 1.9266e-04],
            ],
            [
                [7.1383e-03, 6.8744e-02, 1.7052e-02, 6.8744e-02, 7.1383e-03],
                [7.1574e-02, 6.8769e-01, 1.3960e-01, 6.8769e-01, 7.1574e-02],
                [7.4696e-03, 7.5166e-02, 8.1588e-02, 7.5166e-02, 7.4695e-03],
                [3.8769e-03, 7.1574e-02, 6.8453e-01, 7.1574e-02, 3.8769e-03],
                [7.1649e-04, 1.0299e-02, 6.8744e-02, 1.0299e-02, 7.1650e-04],
            ],
        ]
    )

    coeffs = [0.0, 10 * 1e-8, 100 * 1e-8]  # in cm²/s
    diffusion = Diffusion(
        diff_coeffs=coeffs,
        dx=10.0,  # target molecule map voxel side length
        dt=0.1,  # target diffusion time step
        map_size=molecule_map.shape[1],
        ftype=FTYPE,
        device=DEVICE,
    )

    molecule_map = diffusion.diffuse(molecule_map=molecule_map)
    torch.testing.assert_close(molecule_map, exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
def test_diffusion_randomly() -> None:
    m = 10
    N = 100
    coeffs = [random.uniform(0, 10) for _ in range(m)]
    diffusion = Diffusion(diff_coeffs=coeffs, map_size=N, ftype=FTYPE, device=DEVICE)
    molecule_map = frand(m, N, N) * 100

    totals_0 = molecule_map.sum(dim=(1, 2))

    for _ in range(10000):
        molecule_map = diffusion.diffuse(molecule_map=molecule_map)
        assert (molecule_map > 0.0).all()

    totals_1 = molecule_map.sum(dim=(1, 2))
    torch.testing.assert_close(totals_0, totals_1, atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize(
    "cellmols, expmols",
    [
        # no cells
        (
            [],
            [],
        ),
        # some permeation
        (
            [
                [10.0, 0.0, 0.0, 10.0],
                [0.0, 10.0, 10.0, 0.0],
                [5.0, 10.0, 15.0, 20.0],
            ],
            [
                [7.6966e-02, 9.9900e00, 9.9230e-03, 9.9900e00],
                [9.9230e00, 9.9900e-03, 9.9901e00, 9.9900e-03],
                [1.4923e01, 1.9990e01, 1.4990e01, 1.9990e01],
            ],
        ),
    ],
)
def test_permeation(cellmols: list, expmols: list):
    cell_molecules = ftensor(cellmols) if cellmols else fzeros(0, 4)
    exp = ftensor(expmols) if expmols else fzeros(0, 4)

    permeation = Permeation(perm_coeffs=[0.001, 0.1], ftype=FTYPE, device=DEVICE)

    cell_molecules = permeation.permeate(x=cell_molecules)
    torch.testing.assert_close(cell_molecules, exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
def test_permeation_randomly() -> None:
    M = 10
    coeffs = [random.uniform(0, 1) for _ in range(M)]
    permeation = Permeation(perm_coeffs=coeffs, ftype=FTYPE, device=DEVICE)
    cell_molecules = frand(1000, M * 2) * 100
    int_mols = slice(0, M)
    ext_mols = slice(M, 2 * M)

    totals_int_0 = cell_molecules[:, int_mols]
    totals_ext_0 = cell_molecules[:, ext_mols]
    totals_0 = totals_int_0 + 1000 * totals_ext_0

    for _ in range(10000):
        molecule_map = permeation.permeate(x=cell_molecules)
        assert (molecule_map > 0.0).all()

    totals_int_1 = cell_molecules[:, int_mols]
    totals_ext_1 = cell_molecules[:, ext_mols]
    totals_1 = totals_int_1 + 1000 * totals_ext_1

    torch.testing.assert_close(totals_0, totals_1, atol=_ATOL, rtol=_RTOL)


def _degrade(d: float, t: float) -> float:
    # degradation in days for half-life t in days
    return d * math.exp(-math.log(2) / t)


@pytest.mark.slow
@pytest.mark.parametrize(
    "x_, exp_",
    [
        (  # mx2x2 molecule map degradation
            [
                [
                    [0.0, 10],
                    [1, 1e3],
                ],
                [
                    [0.0, 10],
                    [1, 1e3],
                ],
                [
                    [0.0, 10],
                    [1, 1e3],
                ],
            ],
            [
                [
                    [0.0, 10],
                    [1, 1e3],
                ],
                [
                    [0.0, _degrade(10, 0.1)],
                    [_degrade(1, 0.1), _degrade(1e3, 0.1)],
                ],
                [
                    [0.0, _degrade(10, 1)],
                    [_degrade(1, 1), _degrade(1e3, 1)],
                ],
            ],
        ),
        (
            # 4xm cell molecules degradation
            [
                [1e3, 1e3, 1e3],
                [1e2, 1e2, 1e2],
                [10, 10, 10],
                [0, 0, 0],
            ],
            [
                [1e3, _degrade(1e3, 0.1), _degrade(1e3, 1.0)],
                [1e2, _degrade(1e2, 0.1), _degrade(1e2, 1.0)],
                [10, _degrade(10, 0.1), _degrade(10, 1.0)],
                [0, 0, 0],
            ],
        ),
    ],
)
def test_degradation(x_: list, exp_: list) -> None:
    x = ftensor(x_)
    exp = ftensor(exp_)

    # use m=3 for this test
    m_dim = x.shape.index(3)

    # half lifes in days
    degradation = Degradation(
        half_lifes=[-1, 0.1, 1.0],
        dt=1.0,  # 1 second
        ftype=FTYPE,
        device=DEVICE,
    )

    # degrade for a day
    for _ in range(60 * 60 * 24):
        x = degradation.degrade(x=x, m_dim=m_dim)

    # accumulates float errors over many time steps
    torch.testing.assert_close(x, exp, atol=1e-2, rtol=1e-2)
