import torch
import torch.nn.functional as F

# 100x below f32 limits
_EPS = 1e-43  # n < -45 -> 0.0
_MAX = 1e36  # n > 38 -> inf
_MIN = -1e36  # n > 38 -> inf
_INF = float("inf")


class Kinetics:

    def __init__(self, device: str = "cpu"):
        self.device = device

    @classmethod
    def _get_bounds(
        cls,
        B: torch.Tensor,  # f32 (c, p, m)
        N: torch.Tensor,  # f32 (c, p, m)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ratio_lo = torch.where(N > 0, -B / N, -_INF)
        xi_lo = ratio_lo.max(dim=2).values  # f32 (c, p)

        ratio_hi = torch.where(N < 0, B / -N, _INF)
        xi_hi = ratio_hi.min(dim=2).values  # f32 (c, p)

        return xi_lo, xi_hi  # f32 (c, p), f32 (c, p)

    def _get_alpha_cat(
        self,
        x: torch.Tensor,  # f32 (c, p, m)
        N_f: torch.Tensor,  # i32 (c, p, m)
        N_b: torch.Tensor,  # i32 (c, p, m)
        k_f: torch.Tensor,  # f32 (c, p)
        k_b: torch.Tensor,  # f32 (c, p)
    ) -> torch.Tensor:
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
        l_f = -torch.log(k_f) + (N_f * torch.log(x + _EPS)).sum(dim=2)  # f32 (c, p)
        l_b = -torch.log(k_b) + (N_b * torch.log(x + _EPS)).sum(dim=2)  # f32 (c, p)
        l_max = torch.maximum(l_f, l_b).clamp(min=0.0)  # f32 (c, p)
        e_f = torch.exp(l_f - l_max)  # f32 (c, p)
        e_b = torch.exp(l_b - l_max)  # f32 (c, p)
        e_max = torch.exp(-l_max)  # f32 (c, p)
        alpha = (e_f - e_b) / (e_max + e_f + e_b)  # f32 (c, p)
        return alpha.nan_to_num(0.0)  # f32 (c, p)

    def _get_alpha_reg(
        self,
        x: torch.Tensor,  # f32 (c, p, m)
        N_h: torch.Tensor,  # i32 (c, p, m)
        K_r: torch.Tensor,  # f32 (c, p, m)
    ) -> torch.Tensor:
        # for each protein calculate alpha_reg:
        #
        #   alpha = prod(x_i^n_i / (k_i^n_i + x_i^n_i))  i in 1..m
        #
        # for better effective dynamic range as:
        #
        #   z_i = n_i * (log(k_i) - log(x_i))
        #   alpha = exp(-sum(softplus(z_i)))  i in 1..m
        #
        z = N_h * (torch.log(K_r) - torch.log(x))  # f32 (c, p, m)
        s = F.softplus(z, beta=1.0, threshold=20.0)  # pylint: disable=E1102
        alpha = torch.exp(-s.sum(dim=2))  # f32 (c, p)
        return alpha  # f32 (c, p)

    def _get_velocity(
        self,
        xi: torch.Tensor,  # f32 (c, p)
        B: torch.Tensor,  # f32 (c, p, m)
        N: torch.Tensor,  # f32 (c, p, m)
        N_r: torch.Tensor,  # i32 (c, p, m)
        N_f: torch.Tensor,  # i32 (c, p, m)
        N_b: torch.Tensor,  # i32 (c, p, m)
        k_f: torch.Tensor,  # f32 (c, p)
        k_b: torch.Tensor,  # f32 (c, p)
        K_r: torch.Tensor,  # f32 (c, p, m)
        v_max: torch.Tensor,  # f32 (c, p)
    ) -> torch.Tensor:

        # broadcast into a per-reaction shifted concentration
        x1 = B + N * xi.unsqueeze(-1)  # f32 (c, p, m)

        # TODO: shouldn't be necessary
        if x1.min() < 0.0:
            raise ValueError("negative concentration")

        # calculate per protein catalytic activity
        alpha_cat = self._get_alpha_cat(
            x=x1, N_f=N_f, N_b=N_b, k_f=k_f, k_b=k_b
        )  # f32 (c, p)

        # calculate per protein regulatory activity
        alpha_reg = self._get_alpha_reg(x=x1, N_h=N_r, K_r=K_r)  # f32 (c, p)

        return v_max * alpha_cat * alpha_reg  # f32 (c, p)

    def _bisect_xi(
        self,
        n_iters: int,
        h: float,
        B: torch.Tensor,  # f32 (c, p, m)
        xi_lo: torch.Tensor,  # f32 (c, p)
        xi_hi: torch.Tensor,  # f32 (c, p)
        N: torch.Tensor,  # i32 (c, p, m)
        N_r: torch.Tensor,  # i32 (c, p, m)
        N_f: torch.Tensor,  # i32 (c, p, m)
        N_b: torch.Tensor,  # i32 (c, p, m)
        k_f: torch.Tensor,  # f32 (c, p)
        k_b: torch.Tensor,  # f32 (c, p)
        K_r: torch.Tensor,  # f32 (c, p, m)
        v_max: torch.Tensor,  # f32 (c, p)
    ) -> torch.Tensor:
        lo = xi_lo.clone()  # f32 (c, p)
        hi = xi_hi.clone()  # f32 (c, p)

        for _ in range(n_iters):
            mid = 0.5 * (lo + hi)  # f32 (c, p)
            v = self._get_velocity(
                xi=mid,
                B=B,
                N=N,
                N_r=N_r,
                N_f=N_f,
                N_b=N_b,
                k_f=k_f,
                k_b=k_b,
                K_r=K_r,
                v_max=v_max,
            )  # f32 (c, p)

            too_high = mid - h * v > 0  # bool (c, p)
            hi = torch.where(too_high, mid, hi)  # f32 (c, p)
            lo = torch.where(too_high, lo, mid)  # f32 (c, p)

        return 0.5 * (lo + hi)  # f32 (c, p)

    def step_protein_activity(
        self,
        h: float,
        x0: torch.Tensor,  # f32 (c, m)
        N_r: torch.Tensor,  # i32 (c, p, m)
        N_f: torch.Tensor,  # i32 (c, p, m)
        N_b: torch.Tensor,  # i32 (c, p, m)
        k_f: torch.Tensor,  # f32 (c, p)
        k_b: torch.Tensor,  # f32 (c, p)
        K_r: torch.Tensor,  # f32 (c, p, m)
        v_max: torch.Tensor,  # f32 (c, p)
        n_sweeps: int = 3,
        n_bisect: int = 10,
    ) -> torch.Tensor:
        c, p = v_max.shape
        N = (N_b - N_f).float()

        # initial xi
        xi = torch.zeros((c, p), device=self.device, dtype=torch.float32)  # f32 (c, p)

        for _ in range(n_sweeps):
            dx = torch.einsum("cpm,cp->cm", N, xi)  # (c, m)
            dx_ = dx.unsqueeze(1)  # (c, 1, m)
            xi_ = xi.unsqueeze(-1)  # (c, p, 1)
            x0_ = x0.unsqueeze(1)  # (c, 1, m)

            # substract contribution of each protein from total concentration change
            B = x0_ + (dx_ - N * xi_)  # (c, p, m)

            xi_lo, xi_hi = self._get_bounds(B=B, N=N)  # (c, p), (c, p)
            xi = self._bisect_xi(
                h=h,
                n_iters=n_bisect,
                B=B,
                xi_lo=xi_lo,
                xi_hi=xi_hi,
                N=N,
                N_r=N_r,
                N_f=N_f,
                N_b=N_b,
                k_f=k_f,
                k_b=k_b,
                K_r=K_r,
                v_max=v_max,
            )  # (c, p)

        x1 = x0 + torch.einsum("cpm,cp->cm", N, xi)
        return x1
