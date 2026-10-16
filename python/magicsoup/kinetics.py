import logging
from typing import TypedDict

import torch
import torch.nn.functional as F

from magicsoup.util import TensorClass

_log = logging.getLogger(__name__)


class KineticsKwargs(TypedDict, total=False):
    eps: float
    n_max_sweeps: int
    n_max_bisect: int
    xi_conv_tol: float


class Kinetics(TensorClass):
    """
    Solving ODE system that describes reaction kinetics

    Arguments:
        dt: time interval of a step in s (seconds)
        n_max_sweeps: maximum number of block Jacobi sweeps to perform when solving ODEs
        n_max_bisect: maximum number of bisections when solving for xi
        xi_conv_tol: xi convergence tolerance when solving ODEs.
            Bisection steps and Jacobi sweeps stop once xi doesn't change for than this tolerance.
        eps: Smallest possible non-zero positive value
        device: Device to use for tensors
            (see [pytorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)).
            This has to be aligned with [World][magicsoup.world.World].
        ftype: Dtype of float tensors used.
            This has to be aligned with [World][magicsoup.world.World].
        ftype: Dtype of integer tensors used.
            This has to be aligned with [World][magicsoup.world.World].

    Reaction kinetics are modelled as reversible Michaelis-Menten (MM) kinetics.
    Parameters were optimized for realistic values when assuming the following units.
    All molecule concentrations are given in milli molar,
    enzyme concentrations in the cell are constant at 1 micro molar,
    and maximum velocities of enzymes are given in milli molar per second.
    """

    def __init__(
        self,
        dt: float = 0.1,
        n_max_sweeps: int = 3,
        n_max_bisect: int = 20,
        xi_conv_tol: float = 1e-4,
        eps: float = 1e-40,
        device: str = "cpu",
        ftype: torch.dtype = torch.float32,
        itype: torch.dtype = torch.int8,
    ):
        super().__init__(device=device, itype=itype, ftype=ftype)
        self.eps = eps
        self.dt = dt
        self.n_max_sweeps = n_max_sweeps
        self.n_max_bisect = n_max_bisect
        self.xi_conv_tol = xi_conv_tol

    def get_x1(
        self,
        x0: torch.Tensor,  # (c, m)
        N_e: torch.Tensor,  # (c, p, m)
        N_h: torch.Tensor,  # (c, p, m)
        N_f: torch.Tensor,  # (c, p, m)
        N_b: torch.Tensor,  # (c, p, m)
        k_f: torch.Tensor,  # (c, p)
        k_b: torch.Tensor,  # (c, p)
        K_r: torch.Tensor,  # (c, p, m)
        v_max: torch.Tensor,  # (c, p)
    ) -> torch.Tensor:
        """
        Calculate concentrations after one time step

        Arguments:
            x0: (c, m) float tensor of initial concentrations in mM (milli molar)
            N_e: (c, p, m) float tensor of effective stoichiometric numbers for calculating concentration changes
            N_h: (c, p, m) integer tensor of hill coefficients of regulatory effectors
            N_f: (c, p, m) integer tensor of stoichiometric coefficients of substrates for calculating MM kinetics
            N_b: (c, p, m) integer tensor of stoichiometric coefficients of products for calculating MM kinetics
            k_f: (c, p) float tensor of reactant affinity in forward direction
            k_b: (c, p) float tensor of reactant affinity in backward direction
            K_r: (c, p, m) float tensor of half occupancy concentrations of regulatory domains
            v_max: (c, p) float tensor maximum velocity of assuming 1 µM (micro molar) of protein in mM/s (milli molar per second)

        Tensor dimension `c` describes each cell, `p` each protein, `m` each molecule species.
        `m` includes both all intracellular molecule species, then all extracellular molecule species.

        Effective stoichiometric numbers means stoichiometric numbers adjusted to volume differences between cell and voxel.
        `N_f` and `N_b` is used to calculate catalytic activity from concentrations,
        `N_e` is used to calculate the resulting concentrations.
        """
        c, p = v_max.shape
        dt = self.dt
        n_max_sweeps = self.n_max_sweeps
        n_max_bisect = self.n_max_bisect
        xi_conv_tol = self.xi_conv_tol

        # initial xi
        xi = self.fzeros(c, p)  # (c, p)

        for _ in range(n_max_sweeps):
            xi_prev = xi.clone()

            # concentration delta
            dx = torch.einsum("cpm,cp->cm", N_e, xi)  # (c, m)

            # per cell dampening helps background convergence here
            xi, dx = self._dampen_cells(x0=x0, dx=dx, xi=xi)

            dx_ = dx.unsqueeze(1)  # (c, 1, m)
            xi_ = xi.unsqueeze(-1)  # (c, p, 1)
            x0_ = x0.unsqueeze(1)  # (c, 1, m)

            # substract contribution of each protein from total concentration change
            B = x0_ + (dx_ - N_e * xi_)  # (c, p, m)

            self._check_nonfinite(B, "B during Jacobi sweep")

            xi_lo, xi_hi = self._get_bounds(
                B=B, N_e=N_e, v_max=v_max, dt=dt
            )  # (c, p), (c, p)
            xi = self._bisect_xi(
                dt=dt,
                n_max_iters=n_max_bisect,
                xi_conv_tol=xi_conv_tol,
                B=B,
                xi_lo=xi_lo,
                xi_hi=xi_hi,
                N=N_e,
                N_h=N_h,
                N_f=N_f,
                N_b=N_b,
                k_f=k_f,
                k_b=k_b,
                K_r=K_r,
                v_max=v_max,
            )  # (c, p)

            if (xi - xi_prev).abs().max() < xi_conv_tol:
                break

        # concentration delta
        dx = torch.einsum("cpm,cp->cm", N_e, xi)  # (c, m)

        # per cell dampening for final result
        xi, dx = self._dampen_cells(x0=x0, dx=dx, xi=xi)

        x1 = x0 + dx  # (c, m)

        x1 = self._fix_nonfinite(x1, "x1 after Jacobi sweep")
        x1 = self._fix_negative(x1, "x1 after Jacobi sweep")

        if x1.min() < 0.0:
            raise ValueError("Fix didnt work")

        return x1.clamp(min=0.0)  # (c, m)

    def _bisect_xi(
        self,
        n_max_iters: int,
        xi_conv_tol: float,
        dt: float,
        B: torch.Tensor,  # (c, p, m)
        xi_lo: torch.Tensor,  # (c, p)
        xi_hi: torch.Tensor,  # (c, p)
        N: torch.Tensor,  # (c, p, m)
        N_h: torch.Tensor,  # (c, p, m)
        N_f: torch.Tensor,  # (c, p, m)
        N_b: torch.Tensor,  # (c, p, m)
        k_f: torch.Tensor,  # (c, p)
        k_b: torch.Tensor,  # (c, p)
        K_r: torch.Tensor,  # (c, p, m)
        v_max: torch.Tensor,  # (c, p)
    ) -> torch.Tensor:
        # solve for xi per protein
        # - B must be concentrations of all other activity at initial xi
        # - xi_lo, xi_hi must ensure concentration positivity
        # - N defines stoichiometric numbers
        # - xi defines current reaction extend
        # - v_max must be >= 0 for all elements where N != 0
        #
        # use bisection method so that:
        #
        #   0 = g(xi) = xi - dt * v(N * xi + B)
        #
        # xi is chosen as midpoint between lo and hi
        # lo,hi is updated each iteration based on sign of g(xi)
        # velocity is calculated given concentrations from background activity:
        #
        #   v(x) = v_max * alpha_cat(x) * alpha_reg(x)
        #   dx = N * xi
        #
        lo = xi_lo.clone()  # (c, p)
        hi = xi_hi.clone()  # (c, p)

        for _ in range(n_max_iters):
            xi = 0.5 * (lo + hi)  # (c, p)

            # bounds can be non-finite for N=0
            xi = xi.nan_to_num(0.0)

            # broadcast into a per-reaction shifted concentration
            x1 = B + N * xi.unsqueeze(-1)  # (c, p, m)

            # only for numerical safety, not for mass bookkeeping
            x1 = x1.clamp(min=0.0)  # (c, p, m)

            # calculate per protein catalytic activity
            a_cat = self._get_alpha_cat(x=x1, N_f=N_f, N_b=N_b, k_f=k_f, k_b=k_b)

            # calculate per protein regulatory activity
            a_reg = self._get_alpha_reg(x=x1, N_h=N_h, K_r=K_r)

            v = v_max * a_cat * a_reg  # (c, p)

            too_high = xi - dt * v > 0  # (c, p)
            hi = torch.where(too_high, xi, hi)  # (c, p)
            lo = torch.where(too_high, lo, xi)  # (c, p)

            if (hi - lo).max() < xi_conv_tol:
                break

        return 0.5 * (lo + hi)  # (c, p)

    def _get_alpha_reg(
        self,
        x: torch.Tensor,  # (c, p, m)
        N_h: torch.Tensor,  # (c, p, m)
        K_r: torch.Tensor,  # (c, p, m)
    ) -> torch.Tensor:
        # calculate per protein regulatory activity
        # - N_h defines hill coefficients
        # - x must be >= 0 for all elements where N_h > 0
        # - K_r must be > 0 for all elements where N_h > 0
        # - K_r must be 0 for all elements where N_h = 0
        #
        # for each protein calculate alpha_reg:
        #
        #   alpha = prod(x_i^n_i / (k_i^n_i + x_i^n_i))  i in 1..m
        #
        # for better effective dynamic range as:
        #
        #   z_i = n_i * (log(k_i) - log(x_i))
        #   alpha = exp(-sum(softplus(z_i)))  i in 1..m
        #
        eps = self.eps
        z = N_h * (torch.log(K_r) - torch.log(x + eps))  # (c, p, m)
        s = F.softplus(z, beta=1.0, threshold=20.0)  # pylint: disable=E1102
        alpha = torch.exp(-s.nansum(dim=2))  # (c, p)
        return alpha  # (c, p)

    def _get_alpha_cat(
        self,
        x: torch.Tensor,  # (c, p, m)
        N_f: torch.Tensor,  # (c, p, m)
        N_b: torch.Tensor,  # (c, p, m)
        k_f: torch.Tensor,  # (c, p)
        k_b: torch.Tensor,  # (c, p)
    ) -> torch.Tensor:
        # calculate per protein catalytic activity
        # - N_f defines forward stoichiometric coefficients
        # - N_b defines backward stoichiometric coefficients
        # - x must be >= 0 for all elements where N_f, N_b > 0
        # - k_f, K_b must be > 0 for all elements where N_f, N_b > 0
        # - k_f, k_b must be 0 for all elements where N_f, N_b = 0
        #
        # for each protein calculate alpha_cat:
        #
        #   a_f = 1/k_f * product(x_i^n_f_i)  i in 1..m
        #   a_b = 1/k_b * product(x_i^n_b_i)  i in 1..m
        #   alpha = (a_f - a_b) / (1 + a_f + a_b)
        #
        # for better effective dynamic range as:
        #
        #   l_f = -log(k_f) + sum(n_f_i * log(x_i))  i in 1..m
        #   l_b = -log(k_b) + sum(n_b_i * log(x_i))  i in 1..m
        #   l_max = max(l_f, l_b, 0)
        #   alpha  = (exp(l_f - l_max) - exp(l_b - l_max)) / (exp(-l_max) + exp(l_f - l_max) + exp(l_b - l_max))
        #
        eps = self.eps
        l_f = -torch.log(k_f + eps) + (N_f * torch.log(x + eps)).sum(dim=2)  # (c, p)
        l_b = -torch.log(k_b + eps) + (N_b * torch.log(x + eps)).sum(dim=2)  # (c, p)
        l_max = torch.maximum(l_f, l_b).clamp(min=0.0)  # (c, p)
        e_f = torch.exp(l_f - l_max)  # (c, p)
        e_b = torch.exp(l_b - l_max)  # (c, p)
        e_max = torch.exp(-l_max)  # (c, p)
        alpha = (e_f - e_b) / (e_max + e_f + e_b)  # (c, p)
        return alpha.nan_to_num(0.0)  # (c, p)

    def _dampen_cells(
        self, x0: torch.Tensor, dx: torch.Tensor, xi: torch.Tensor, pad: float = 1e-3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Rescale xi and x0 to ensure positive concentrations
        #
        # Rescaling happens per cell (whole cell is slowed down) but is quick
        # Jacobi iterations should still converge to correct solution
        #
        # find scaling factor as largest λ∈(0,1] with:
        #
        #   x + λ * dx >= 0
        #
        inf = float("inf")
        ratio = torch.where(dx < 0, x0 / (-dx + pad), torch.full_like(dx, inf))
        lam = ratio.min(dim=1, keepdim=True).values.clamp(max=1.0)  # (c, 1)
        return xi * lam, dx * lam  # (c, p), (c, m)

    def _fix_negative(self, t: torch.Tensor, where: str) -> torch.Tensor:
        is_neg = t < -self.eps  # error < eps is negligible
        if not is_neg.any():
            return t.clamp(min=0.0)

        bad = t[is_neg].flatten()[:5]
        _log.warning("negative values in %s, cells/molecules: %r", where, bad)
        _log.warning("negative values in %s replaced with 0.0", where)
        return t.clamp(min=0.0)

    def _fix_nonfinite(self, t: torch.Tensor, where: str) -> torch.Tensor:
        is_finite = torch.isfinite(t)
        if is_finite.all():
            return t

        bad = t[~is_finite].flatten()[:5]
        _log.warning("non-finite values in %s, cells/proteins: %r", where, bad)
        _log.warning("non-finite values in %s replaced with 0.0", where)
        t[~is_finite] = 0.0
        return t

    def _check_nonfinite(self, t: torch.Tensor, where: str) -> None:
        is_finite = torch.isfinite(t)
        if not is_finite.all():
            bad = t[~is_finite].flatten()[:5]
            _log.warning("non-finite values in %s, cells/proteins: %r", where, bad)

    @classmethod
    def _get_bounds(
        cls,
        B: torch.Tensor,  # (c, p, m)
        N_e: torch.Tensor,  # (c, p, m)
        v_max: torch.Tensor,  # (c, p)
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # calculate xi bounds for bisection
        # - B represents molecule concentration per protein if all other proteins were active
        # - N represents stoichiometric numbers per protein
        #
        # bounds are chosen so that concentrations cannot become negative:
        #
        #   lo = max(-B / N)  for N > 0
        #   hi = min(B / -N)  for N < 0
        #
        inf = float("inf")

        ratio_lo = torch.where(N_e > 0, -B / N_e, -inf)
        xi_lo = ratio_lo.max(dim=2).values  # (c, p)

        ratio_hi = torch.where(N_e < 0, B / -N_e, inf)
        xi_hi = ratio_hi.min(dim=2).values  # (c, p)

        # guard against infeasible per-protein brackets
        # could happen with B < 0 elements
        # which means one background protein is producing substrate
        xi_lo = torch.minimum(xi_lo, xi_hi)  # (c, p)

        # since always xi <= dt * v_max it can be used as limiting factor
        max_rate = dt * v_max  # (c, p)
        xi_lo = torch.maximum(xi_lo, -max_rate)
        xi_hi = torch.minimum(xi_hi, max_rate)

        return xi_lo, xi_hi  # (c, p), (c, p)
