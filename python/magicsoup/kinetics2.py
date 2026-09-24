import logging

import torch
import torch.nn.functional as F

_log = logging.getLogger(__name__)


class Kinetics:

    def __init__(
        self,
        device: str = "cpu",
        itype: torch.dtype = torch.int8,
        ftype: torch.dtype = torch.float32,
        eps: float = 1e-40,
    ):
        self.device = device
        self.itype = itype
        self.ftype = ftype
        self.eps = eps

    def _check_nonfinite(self, t: torch.Tensor, where: str) -> None:
        is_finite = torch.isfinite(t)
        if not is_finite.all():
            bad = (~is_finite).nonzero()
            _log.warning("non-finite values in %s, cells/proteins: %r", where, bad[:5])

    def _fix_nonfinite(self, t: torch.Tensor, where: str) -> None:
        is_finite = torch.isfinite(t)
        if not is_finite.all():
            bad = (~is_finite).nonzero()
            _log.warning("non-finite values in %s, cells/proteins: %r", where, bad[:5])
            t[~is_finite] = 0.0
            _log.warning("non-finite values in %s replaced with 0.0", where)

    def _fix_negative(self, t: torch.Tensor, where: str) -> None:
        is_neg = t < 0.0
        if is_neg.any():
            bad = (~is_neg).nonzero()
            _log.warning("negative values in %s, cells/molecules: %r", where, bad[:5])
            t[~is_neg] = 0.0
            _log.warning("negative values in %s replaced with 0.0", where)

    def _dampen_cells(
        self, x0: torch.Tensor, dx: torch.Tensor, xi: torch.Tensor
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
        eps = self.eps
        inf = float("inf")
        ratio = torch.where(dx < 0, x0 / (-dx + eps), torch.full_like(dx, inf))
        lam = ratio.min(dim=1, keepdim=True).values.clamp(max=1.0)  # (c, 1)
        return xi * lam, dx * lam  # (c, p), (c, m)

    @classmethod
    def _get_bounds(
        cls,
        B: torch.Tensor,  # (c, p, m)
        N: torch.Tensor,  # (c, p, m)
        v_max: torch.Tensor,  # (c, p)
        h: float,
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

        ratio_lo = torch.where(N > 0, -B / N, -inf)
        xi_lo = ratio_lo.max(dim=2).values  # (c, p)

        ratio_hi = torch.where(N < 0, B / -N, inf)
        xi_hi = ratio_hi.min(dim=2).values  # (c, p)

        # guard against infeasible per-protein brackets
        # could happen with B < 0 elements
        # which means one background protein is producing substrate
        xi_lo = torch.minimum(xi_lo, xi_hi)  # (c, p)

        # since always xi <= h * v_max it can be used as limiting factor
        max_rate = h * v_max  # (c, p)
        xi_lo = torch.maximum(xi_lo, -max_rate)
        xi_hi = torch.minimum(xi_hi, max_rate)

        return xi_lo, xi_hi  # (c, p), (c, p)

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
        #   alpha = (a_f - b_b) / (1 + a_f + b_b)
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

    def _bisect_xi(
        self,
        n_max_iters: int,
        xi_conv_tol: float,
        h: float,
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
        #   0 = g(xi) = xi - h * v(N * xi + B)
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

            too_high = xi - h * v > 0  # (c, p)
            hi = torch.where(too_high, xi, hi)  # (c, p)
            lo = torch.where(too_high, lo, xi)  # (c, p)

            if (hi - lo).max() < xi_conv_tol:
                break

        return 0.5 * (lo + hi)  # (c, p)

    def step_protein_activity(
        self,
        x0: torch.Tensor,  # (c, m)
        N_h: torch.Tensor,  # (c, p, m)
        N_f: torch.Tensor,  # (c, p, m)
        N_b: torch.Tensor,  # (c, p, m)
        k_f: torch.Tensor,  # (c, p)
        k_b: torch.Tensor,  # (c, p)
        K_r: torch.Tensor,  # (c, p, m)
        v_max: torch.Tensor,  # (c, p)
        h: float = 1.0,
        n_max_sweeps: int = 3,
        n_max_bisect: int = 20,
        xi_conv_tol: float = 1e-4,
    ) -> torch.Tensor:
        c, p = v_max.shape
        N = N_b - N_f.float()

        # initial xi
        xi = torch.zeros((c, p), device=self.device, dtype=self.ftype)  # (c, p)

        for _ in range(n_max_sweeps):
            xi_prev = xi.clone()
            dx = torch.einsum("cpm,cp->cm", N, xi)  # (c, m)

            # per cell dampening helps background convergence here
            xi, dx = self._dampen_cells(x0=x0, dx=dx, xi=xi)

            dx_ = dx.unsqueeze(1)  # (c, 1, m)
            xi_ = xi.unsqueeze(-1)  # (c, p, 1)
            x0_ = x0.unsqueeze(1)  # (c, 1, m)

            # substract contribution of each protein from total concentration change
            B = x0_ + (dx_ - N * xi_)  # (c, p, m)

            self._check_nonfinite(B, "B during Jacobi sweep")

            xi_lo, xi_hi = self._get_bounds(
                B=B, N=N, v_max=v_max, h=h
            )  # (c, p), (c, p)
            xi = self._bisect_xi(
                h=h,
                n_max_iters=n_max_bisect,
                xi_conv_tol=xi_conv_tol,
                B=B,
                xi_lo=xi_lo,
                xi_hi=xi_hi,
                N=N,
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

        dx = torch.einsum("cpm,cp->cm", N, xi)  # (c, m)

        # per cell dampening for final result
        xi, dx = self._dampen_cells(x0=x0, dx=dx, xi=xi)

        x1 = x0 + dx  # (c, m)

        self._fix_nonfinite(x1, "x1 after Jacobi sweep")
        self._fix_negative(x1, "x1 after Jacobi sweep")

        return x1.clamp(min=0.0)  # (c, m)
