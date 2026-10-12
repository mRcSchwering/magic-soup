import random

import pytest
import torch
from magicsoup.physics import Diffusion, FftDiffusion, Permeation

from tests.config import DEVICE, FTYPE
from tests.util import frand

_ATOL = 1e-4
_RTOL = 1e-4


def test_diffusion():
    molecule_map = torch.tensor(
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
    exp = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.9600, 0.0100, 0.0000, 0.0000, 0.0100],
                [0.0100, 0.0000, 0.0000, 0.0100, 0.0000],
                [0.0000, 0.0000, 0.0100, 0.9700, 0.0100],
                [0.0000, 0.0000, 0.0100, 0.9700, 0.0100],
                [0.0100, 0.0000, 0.0000, 0.0100, 0.0000],
            ],
            [
                [0.0000, 0.1000, 0.0000, 0.1000, 0.0000],
                [0.1000, 0.6000, 0.2000, 0.6000, 0.1000],
                [0.0000, 0.1000, 0.1000, 0.1000, 0.0000],
                [0.0000, 0.1000, 0.6000, 0.1000, 0.0000],
                [0.0000, 0.0000, 0.1000, 0.0000, 0.0000],
            ],
        ]
    )

    diffusion = Diffusion(
        diff_coeffs=[0, 10, 100],
        dx=10.0,  # target molecule map voxel side length
        dt=0.1,  # target diffusion time step
        ftype=FTYPE,
        device=DEVICE,
    )

    molecule_map = diffusion.diffuse(molecule_map=molecule_map)
    torch.testing.assert_close(molecule_map, exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
def test_random_diffusion() -> None:
    M = 10
    L = 100
    coeffs = [random.uniform(0, 1000) for _ in range(M)]
    diffusion = Diffusion(
        diff_coeffs=coeffs, dx=10.0, dt=0.1, ftype=FTYPE, device=DEVICE
    )
    molecule_map = frand(M, L, L) * 100

    totals_0 = molecule_map.sum(dim=(1, 2))

    for _ in range(1000):
        molecule_map = diffusion.diffuse(molecule_map=molecule_map)
        assert (molecule_map > 0.0).all()

    totals_1 = molecule_map.sum(dim=(1, 2))
    torch.testing.assert_close(totals_0, totals_1, atol=_ATOL, rtol=_RTOL)


def test_fft_diffusion():
    molecule_map = torch.tensor(
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
    exp = torch.tensor(
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

    coeffs = [0, 10, 100]
    diffusion = FftDiffusion(
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
def test_random_fft_diffusion() -> None:
    M = 10
    L = 100
    coeffs = [random.uniform(0, 1000) for _ in range(M)]
    diffusion = FftDiffusion(
        diff_coeffs=coeffs, map_size=L, dx=10.0, dt=0.1, ftype=FTYPE, device=DEVICE
    )
    molecule_map = frand(M, L, L) * 100

    totals_0 = molecule_map.sum(dim=(1, 2))

    for _ in range(10000):
        molecule_map = diffusion.diffuse(molecule_map=molecule_map)
        assert (molecule_map > 0.0).all()

    totals_1 = molecule_map.sum(dim=(1, 2))
    torch.testing.assert_close(totals_0, totals_1, atol=_ATOL, rtol=_RTOL)


def test_permeation():
    cell_molecules = torch.tensor(
        [
            [10.0, 0.0, 0.0, 10.0],
            [0.0, 10.0, 10.0, 0.0],
            [5.0, 10.0, 15.0, 20.0],
        ]
    )
    exp = torch.tensor(
        [
            [7.6966e-02, 9.9900e00, 9.9230e-03, 9.9900e00],
            [9.9230e00, 9.9900e-03, 9.9901e00, 9.9900e-03],
            [1.4923e01, 1.9990e01, 1.4990e01, 1.9990e01],
        ]
    )

    permeation = Permeation(permeabilities=[0.001, 0.1], ftype=FTYPE, device=DEVICE)

    cell_molecules = permeation.permeate(x=cell_molecules)
    torch.testing.assert_close(cell_molecules, exp, atol=_ATOL, rtol=_RTOL)


@pytest.mark.slow
def test_permeation_randomly() -> None:
    M = 10
    coeffs = [random.uniform(0, 1) for _ in range(M)]
    permeation = Permeation(permeabilities=coeffs, ftype=FTYPE, device=DEVICE)
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
