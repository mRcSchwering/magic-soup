import math
from typing import TypedDict

import torch

# pylint: disable=not-callable
from torch.fft import fftfreq, irfft2, rfft2, rfftfreq

from magicsoup.util import TensorClass, TensorKwargs


class DiffusionKwargs(TypedDict, total=False):
    dx: float


class Diffusion(TensorClass):
    r"""
    Diffusion model for periodic square N x N grid of 3D voxels

    Arguments:
        diff_coeffs: diffusion coefficients in cm²/s (square centi meter per second)
            for each molecule species as in the molecule map
        N: side length of molecule map
        dx: side length of a voxel on the molecule map in µm (micro meter)
        dt: time step in s (seconds)
        tensor_kwargs: Kwargs passed to [TensorClass][magicsoup.util.TensorClass]

    Assumes constant anisotropic diffusion coefficients.
    Solves diffusion using discrete fourier transform of concentration field
    and eigenvalues of 2D graph Laplacian for N x N periodic grid.

    During the simulation all proteomes are represented as tensors of dimensions `c`, `p`, `m`.
    Tensor dimension `c` describes each cell, `p` each protein, `m` each molecule species.
    """

    def __init__(
        self,
        diff_coeffs: list[float],  # in cm²/s
        N: int,
        dx: float = 10.0,  # in µm
        dt: float = 1.0,  # in s
        tensor_kwargs: TensorKwargs | None = None,
    ) -> None:
        super().__init__(**(tensor_kwargs or {}))

        self.N = N

        # diffusion coefficients cm²/s -> µm²/s
        D = self.ftensor(diff_coeffs) * 1e8

        # allowed frequencies in radians per pixel
        kx = 2 * math.pi * fftfreq(self.N, dtype=self.ftype, device=self.device)
        ky = 2 * math.pi * rfftfreq(self.N, dtype=self.ftype, device=self.device)

        # eigenvalues for discrete Laplacian
        lam = (
            (2 - 2 * torch.cos(kx))[:, None] + (2 - 2 * torch.cos(ky))[None, :]
        ) / dx**2  # (m, N, N//2 + 1)

        # decay factor for ĉ(t) = ĉ(0) exp(-D λ dt)
        self.decay = torch.exp(-D[:, None, None] * dt * lam)  # (m, N, N//2 + 1)

    def diffuse(self, molecule_map: torch.Tensor) -> torch.Tensor:
        """
        Diffuse molecule map

        Arguments:
            molecule_map: (m, N, N) float tensor of concentrations

        Returns:
            molecule_map (m, N, N) float tensor of diffused concentrations
        """
        N = self.N

        # accurately track molecule totals before
        before = molecule_map.sum(dim=(1, 2), dtype=torch.float64)

        # diffuse molecules for dt along dx
        c_hat: torch.Tensor = rfft2(molecule_map)
        out: torch.Tensor = irfft2(c_hat * self.decay, s=(N, N))
        out = out.clamp(min=0)

        # accurately track molecule totals after
        after = out.sum(dim=(1, 2), dtype=torch.float64)

        # scale molecule map to preserve molecule totals
        scale = torch.where(after > 0, before / after, torch.ones_like(after))
        out *= scale.to(out.dtype)[:, None, None]
        return out


class PermeationKwargs(TypedDict, total=False):
    a_cell: float


class Permeation(TensorClass):
    """
    Cell-membrane molecule permeation model

    Arguments:
        perm_coeffs: Permeability coefficients in cm/s (centi meter per second)
            for each molecule species as in the molecule map.
        dt: time step in s (seconds)
        v_cell: volume of a cell in µm³ (cubic micro meter)
        v_med: volume of a cell culture voxel in µm³ (cubic micro meter)
        a_cell: cell membrane area in µm² (square micro meter)
        tensor_kwargs: Kwargs passed to [TensorClass][magicsoup.util.TensorClass]

    The defaults approximate a cell as 1 fL (femto litre) sphere
    swimming in a 1 pL (pico litre) cell culture voxel.

    During the simulation all proteomes are represented as tensors of dimensions `c`, `p`, `m`.
    Tensor dimension `c` describes each cell, `p` each protein, `m` each molecule species.
    In tensors intracellular molecules are represented first, then extracellular ones, hence `2m`.
    """

    def __init__(
        self,
        perm_coeffs: list[float],  # cm/s
        dt: float = 0.1,  # time step in s
        v_cell: float = 1.0,  # cell volume in µm³
        v_vox: float = 1000.0,  # medium voxel volume in µm³
        a_cell: float = 5.0,  # cell membrane area in µm² (~= sphere)
        tensor_kwargs: TensorKwargs | None = None,
    ) -> None:
        super().__init__(**(tensor_kwargs or {}))

        n_molecules = len(perm_coeffs)
        self.int_mols = slice(0, n_molecules)
        self.ext_mols = slice(n_molecules, n_molecules * 2)
        self.v_cell = v_cell
        self.v_vox = v_vox
        self.v_total = v_cell + v_vox

        P = self.ftensor([d for d in perm_coeffs]) * 1e4  # (m,) in µm/s
        k = P * a_cell * (1 / v_cell + 1 / v_vox)  # (m,) exchange rate in 1/s
        self.decay = torch.exp(-k * dt)  # (m,) remaining fraction per step

    def permeate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Permeate and return cell molecule concentrations (c, m)

        Arguments:
            x: (c, 2m) float tensor of molecule concentrations

        Returns:
            x (c, 2m) float tensor of permeated molecule concentrations
        """
        xi = x[:, self.int_mols]  # (c, m) intracellular concentrations
        xe = x[:, self.ext_mols]  # (c, m) extracellular concentrations
        c_eq = (self.v_cell * xi + self.v_vox * xe) / self.v_total
        x[:, self.int_mols] = c_eq + (xi - c_eq) * self.decay
        x[:, self.ext_mols] = c_eq + (xe - c_eq) * self.decay
        return x  # (c, 2m)


class Degradation(TensorClass):
    """
    First-order molecule degradation model

    Arguments:
        half_lifes: Half-life in days for each molecule species as in the molecule map.
        dt: time step in s (seconds)
        tensor_kwargs: Kwargs passed to [TensorClass][magicsoup.util.TensorClass]

    Assumes concentration-independent degradation.
    Set negative half-life for no degradation at all.

    During the simulation all proteomes are represented as tensors of dimensions `c`, `p`, `m`.
    Tensor dimension `c` describes each cell, `p` each protein, `m` each molecule species.
    """

    def __init__(
        self,
        half_lifes: list[float],
        dt: float = 0.1,
        tensor_kwargs: TensorKwargs | None = None,
    ) -> None:
        super().__init__(**(tensor_kwargs or {}))

        t_half = self.ftensor(half_lifes) * 24 * 60 * 60  # in s
        k = torch.log(self.ftensor(2)) / t_half  # (m,)
        self.decay = torch.where(t_half > 0, torch.exp(-k * dt), 1.0)  # (m,)

    def degrade(self, x: torch.Tensor, m_dim: int) -> torch.Tensor:
        """
        Degrade molecules.

        Arguments:
            x: float tensor of concentrations with at least one dimension `m`
            m_dim: index of dimension `m`

        Returns:
            x float tensor of degraded concentrations
        """
        shape = [1] * x.ndim
        shape[m_dim] = -1
        return x * self.decay.view(shape)
