import math
from typing import TypedDict

import torch
import torch.nn.functional as F

# pylint: disable=not-callable
from torch.fft import fftfreq, irfft2, rfft2, rfftfreq

from magicsoup.util import TensorClass

# TODO: check numbers again (molecule energies, volume of cell, diffusivity, concentrations)
#       1 cell with 1um side length is 1um^3 = 1e-5 L
#       intracellular molecule concentrations usually are in 0.01 to 100 mM range
#       Energies are often given in J/mole with avogardo constant N_A = 6e23 / mol
#       so 1 mM in a cell would be 1e-3 mol/L * 1e-5 L * 6e23 / mol = 6e5 (molecules) = 1e-8 mol
#
#       since volumes are constant, I can choose any number as concentration
#       Problem: with realistic rates, the pixel is depleted within 1ms
#       usually cells even at 90% confluency have 1000x more medium
#       so I could e.g. model a pixel as 10 x 1ß x 10 µm and a cell as 1 x 1 x 1 µm
#       but then I have to take that into account in the transporters
#       From then on a transporter should do 1 A <-> 1000 A:
#
#       ratio = V_cell / V_medium (e.g. 1/1000)
#       N_eff = N.clone()
#       N_eff[..., n_intracellular:, :] *= ratio  # if intracellular first, then extracellular
#       N_eff wird nur für das update mit x0/x1 benutzt
#       N wird für die MM kinetics (rates) benutzt
#       Gibbs Energie wird weiterhin ohne ratio berechnet
#
#       ich müsste das gleich auch bei permability beachten (* ratio)
#
#       zusätzlich noch activity + diffusion in 0.1 time steps (sollte dann ganz gut sein)

# TODO: absolute temperature to room temp? easier diffusivity numbers


class Diffusion(TensorClass):
    r"""
    Diffusion model for 2D grid of 3D voxels

    Arguments:
        diff_coeffs: diffusion coefficients for each of the M molecules in the molecule map
        dx: side length of a voxel on the molecule map in micro meter
        dt: time of a diffusion step in seconds
        device: Device to use for tensors
            (see [pytorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)).
            This has to be aligned with [World][magicsoup.world.World].
        ftype: Dtype of float tensors used.
            This has to be aligned with [World][magicsoup.world.World].

    Models diffusion as 5-point Laplacian and approximates it using 2D convolution with kernel
    ```
        [ 0      a      0 ]
        [ a    1 - 4a   a ]
        [ 0      a      0 ] , a = D * h / dx^2
    ```
    for a <= 0.25 with diffusion coefficients D in µm^2/s.
    """

    def __init__(
        self,
        diff_coeffs: list[float],
        dx: float = 10.0,
        dt: float = 0.1,
        ftype: torch.dtype = torch.float32,
        device: str = "cuda",
    ) -> None:
        super().__init__(device=device, ftype=ftype)

        D = self.ftensor(diff_coeffs)
        a = D * dt / dx**2
        self.n_sub = max(1, math.ceil(float(a.max()) / 0.25))  # stability: a <= 0.25
        a = a / self.n_sub

        self.M = D.numel()
        ker = self.fzeros(self.M, 1, 3, 3)
        ker[:, 0, 0, 1] = a
        ker[:, 0, 2, 1] = a
        ker[:, 0, 1, 0] = a
        ker[:, 0, 1, 2] = a
        ker[:, 0, 1, 1] = 1 - 4 * a
        self.kernel = ker

    def diffuse(self, molecule_map: torch.Tensor) -> torch.Tensor:
        """Diffuse and return molecule_map (M, L, L)"""

        # accurately track molecule totals before
        before = molecule_map.sum(dim=(1, 2), dtype=torch.float64)

        x = molecule_map[None]  # (1, M, L, L) for batch
        for _ in range(self.n_sub):
            x = F.pad(x, (1, 1, 1, 1), mode="circular")  # torus wrap
            x = F.conv2d(x, self.kernel, groups=self.M)  # pylint: disable=not-callable

        out = x[0]

        # accurately track molecule totals after
        after = out.sum(dim=(1, 2), dtype=torch.float64)

        # scale molecule map to preserve molecule totals
        scale = torch.where(after > 0, before / after, torch.ones_like(after))
        out *= scale.to(out.dtype)[:, None, None]
        return out


class DiffusionKwargs(TypedDict, total=False):
    dx: float


class FftDiffusion(TensorClass):
    r"""
    FFT Diffusion model for 2D grid of 3D voxels

    Arguments:
        diff_coeffs: diffusion coefficients for each of the M molecules in the molecule map
        map_size: side length of molecule map
        dx: side length of a voxel on the molecule map in micro meter
        dt: time of a diffusion step in seconds
        device: Device to use for tensors
            (see [pytorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)).
            This has to be aligned with [World][magicsoup.world.World].
        ftype: Dtype of float tensors used.
            This has to be aligned with [World][magicsoup.world.World].

    Models diffusion and solves it as 5-point Laplacian using FFT.
    For each mode diffusion on lattice is
    ```
        dc/dt = -D · λ · c
        c(t) = c(0) · exp(-D · λ · t)
    ```
    where D is the diffusion coefficient, dt is the time step, and dc the concentration change.
    `exp(-D · λ · t)` is the decay factor.
    λ is the Eigenvalue of the sums of a mode for y (rows) and x (columns) directions:

        λ = -(2 - 2·cos(kx) + 2 - 2·cos(ky)) / dx²

    k is the frequency in radians per pixel.
    There are `k = 2 pi j / L` with `j = 0...L-1` allowed frequencies for map size `L`.
    """

    def __init__(
        self,
        diff_coeffs: list[float],
        map_size: int,
        dx: float = 10.0,
        dt: float = 1.0,
        ftype: torch.dtype = torch.float32,
        device: str = "cuda",
    ) -> None:
        super().__init__(device=device, ftype=ftype)

        self.L = map_size
        self.M = len(diff_coeffs)
        D = self.ftensor(diff_coeffs)

        # allowed frequencies k = 2 pi j / L with j = 0...L-1
        kx = 2 * math.pi * fftfreq(self.L, dtype=self.ftype, device=self.device)
        ky = 2 * math.pi * rfftfreq(self.L, dtype=self.ftype, device=self.device)

        # positive eigenvalues of the discrete 5-point Laplacian (not the continuous k²)
        # with lam = (2 - 2 cos(kx) + 2 - 2 cos(ky)) / dx
        sx = 2 - 2 * torch.cos(kx)
        sy = 2 - 2 * torch.cos(ky)
        lam = (sx[:, None] + sy[None, :]) / dx**2

        # decay factor for c(t) = c(0) exp(-D lam t)
        self.decay = torch.exp(-D[:, None, None] * dt * lam)  # (M, L, L//2+1)

    def diffuse(self, molecule_map: torch.Tensor) -> torch.Tensor:
        """Diffuse and return molecule_map (M, L, L)"""
        L = self.L

        # accurately track molecule totals before
        before = molecule_map.sum(dim=(1, 2), dtype=torch.float64)

        # diffuse molecules for dt along dx
        f: torch.Tensor = rfft2(molecule_map)
        out: torch.Tensor = irfft2(f * self.decay, s=(L, L))
        out = out.clamp(min=0)

        # accurately track molecule totals after
        after = out.sum(dim=(1, 2), dtype=torch.float64)

        # scale molecule map to preserve molecule totals
        scale = torch.where(after > 0, before / after, torch.ones_like(after))
        out *= scale.to(out.dtype)[:, None, None]
        return out


class PermeationKwargs(TypedDict, total=False):
    A_cell: float


class Permeation(TensorClass):
    """
    Cell-membrane molecule permeation model

    Arguments:
        permeabilities: Permeabilities in cm/s for each molecule species
            in the same order as used by the molecule map.
        dt: time of a permeation step in seconds
        V_cell: volume of a cell in cubic micro meter
        V_med: volume of a cell culture voxel in cubic micro meter
        A_cell: cell membrane area in square micro meter
        device: Device to use for tensors
            (see [pytorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)).
            This has to be aligned with [World][magicsoup.world.World].
        ftype: Dtype of float tensors used.
            This has to be aligned with [World][magicsoup.world.World].

    The defaults approximate a cell as sphere swimming in a cell culture voxel.
    Flux `J = P (c_ext - c_int)` with permeability P creates
    ```
        dc_int/dt = P · A / V_cell · (c_ext - c_int)
        dc_ext/dt = -P · A / V_medium · (c_ext - c_int)
    ```
    Total amount `V_cell · c_int + V_medium · c_ext` stays constant
    and the concentration difference decays exponentially with the rate

        k = P · A · (1/V_cell + 1/V_medium)

    Both concentrations approach the equilibrium concentration

        c_eq = (V_cell · c_int + V_medium · c_ext) / (V_cell + V_medium)
    """

    def __init__(
        self,
        permeabilities: list[float],
        dt: float = 0.1,  # time step in seconds
        V_cell: float = 1.0,  # cell volume in µm³
        V_med: float = 1000.0,  # medium voxel volume in µm³
        A_cell: float = 5.0,  # cell membrane area in µm² (~= sphere)
        ftype: torch.dtype = torch.float32,
        device: str = "cuda",
    ) -> None:

        super().__init__(device=device, ftype=ftype)

        n_molecules = len(permeabilities)
        self.int_mols = slice(0, n_molecules)
        self.ext_mols = slice(n_molecules, n_molecules * 2)
        self.A_cell = A_cell
        self.V_cell = V_cell
        self.V_total = V_cell + V_med
        self.V_med = V_med

        P = self.ftensor([d * 10_000 for d in permeabilities])  # (M,) in µm/s
        k = P * A_cell * (1 / V_cell + 1 / V_med)  # (M,) exchange rate in 1/s
        self.f = torch.exp(-k * dt)  # (S,) remaining fraction per step

    def permeate(self, x: torch.Tensor) -> torch.Tensor:
        """Permeate and returncell molecule concentrations (C, 2M)"""
        xi = x[:, self.int_mols]  # (C, M) intracellular concentrations
        xe = x[:, self.ext_mols]  # (C, M) extracellular concentrations
        c_eq = (self.V_cell * xi + self.V_med * xe) / self.V_total
        x[:, self.int_mols] = c_eq + (xi - c_eq) * self.f
        x[:, self.ext_mols] = c_eq + (xe - c_eq) * self.f
        return x
