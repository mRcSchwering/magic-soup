import logging
import torch
from multiprocessing import Pool
from magicsoup.constants import EPS, GAS_CONSTANT
from magicsoup.containers import Protein

_log = logging.getLogger(__name__)

# TODO: a molecule can be created by one domain and at the same time deconstructed
#       by another. Currently, this results in N = 0 for that molecule. With the current
#       implementation, the molecule will not be part of X_sub and X_prod, i.e. not part
#       of the MM equiation. Does this make sense???
#       Shouldn't that molecule then act more like an allosteric activator?

# TODO: Currently, I cant handle high Vmax. Even with Vmax >= 50, if there are proteins
#       with many substrates and/or effectors, the effectors will multiply to Inf, which
#       in turn will create NaNs down the line. These NaNs then propagate. The problem
#       is X.pow(N).prod(2) which can explode. This doesn't seem to happen with small Vmax.
#       One explanation might be my heuristic that is supposed to prevent negative concentrations.
#       This heuristic becomes more unstable with large Vmax, and it might thereby create
#       more possibilities for proteins to create more of a molecule that physically possible.


# TODO: proper docstring
class Kinetics:
    """
    Class holding logic for simulating protein work. There are `c` cells, `p` proteins, `s` signals.
    Signals are basically molecules, but we have to differentiate between the molecule species
    inside and outside of the cell. So, there are twice as many signals as molecule species.

    - `X` Signal/molecule concentrations (c, s). Must all be >= 0.0.
    - `Km` Domain affinities of all proteins for each molecule (c, p, s). Must all be >= 0.0.
    - `Vmax` Maximum velocities of all proteins (c, p). Must all be >= 0.0.
    - `Ke` Equilibrium constants of all proteins (c, p). If the current reaction quotient of a protein
    exceeds its equilibrium constant, it will work in the other direction. Must all be >= 0.0.
    - `N` Reaction stoichiometry of all proteins for each molecule (c, p, s). Numbers < 0.0 indicate this
    amount of this molecule is being used up by the protein. Numbers > 0.0 indicate this amount of this
    molecule is being created by the protein. 0.0 means this molecule is not part of the reaction of
    this protein.
    - `A` allosteric control of all proteins for each molecule (c, p, s). Numbers > 0.0 mean these molecules
    act as activating effectors, numbers < 0.0 mean these molecules act as inhibiting effectors.
    
    Everything is based on Michaelis Menten kinetics where protein velocity
    depends on substrate concentration:

    ```
        v = Vmax * x / (Km + x)
    ```

    `Vmax` is the maximum velocity of the protein and `Km` the substrate affinity. Multiple
    substrates create interaction terms such as:

    ```
        v = Vmax * x1 * / (Km1 + x1) * x2 / (Km2 + x2)
    ```

    Allosteric effectors work non-competitively such that they reduce or raise `Vmax` but
    leave any `Km` unaffected. Effectors themself also use Michaelis Menten kinteics.
    Here is substrate `x` with inhibitor `i`:
    
    ```
        v = Vmax * x / (Kmx + x) * (1 - Vi)
        Vi = i / (Kmi + i)
    ```

    Activating effectors effectively make the protein dependent on that activator:

    ```
        v = Vmax * x / (Kmx + x) * Va
        Va = a / (Kma + a)
    ```

    Multiple inhibitors and activators are each integrated with each other as if they
    were interacting with each other. I.e. they are multiplied when calculating their
    activity. Each term is clamped to `[0; 1]` before it is multiplied with `Vmax`.
    So a protein can never exceed `Vmax`.

    ```
        v = Vmax * x / (Km + x) * Va * (1 - Vi)
        Va = a1 * a2 / ((Kma1 + a1) * (Kma2 + a2))
        Vi = i1 * i2 / ((Kmi1 + i1) * (Kmi2 + i2))
    ```

    Equilibrium constants `Ke` define whether the reaction of a protein (as defined in `N`)
    can take place. If the reaction quotient (the combined concentrations of all products
    devided by all substrates) is smaller than its `Ke` the reaction proceeds. If it is
    greater than `Ke` the reverse reaction will take place (`N * -1.0` for this protein).
    The idea is to link signal integration to energy conversion.

    ```
        -dG0 = R * T * ln(Ke)
        Ke = exp(-dG0 / R / T)
    ```

    where `dG0` is the standard Gibb's free energy of this reaction, `R` is gas constant,
    `T` is absolute temperature.

    While the reaction quotient `Q` is well below _Ke_ the protein will proceed with the reaction
    in forward direction with a velocity as defined by the above equiations in each time step.
    However, as `Q` approaches `Ke` there is a chance that in one time step `Q` surpasses `Ke`.
    In the next time step the reaction would turn around and `Q` might possibly jump again below
    `Ke`. Thus, in a sensitive protein (low `Km`, high `Vmax`, no inhibition) `Q` might jump
    above and below `Ke` with every time step, i.e. always overshootig the equilibrium state.
    
    One would have to solve for `v` given `Ke` and the stoichiometry in order to limit `v`
    to reach the exact equilibrium state. The equiation has this form:

    ```
        ([P] + m * v)^m / ([S] + n * v)^n = Ke
    ```

    Where `[P]` and `[S]` are product and substrate concentrations and `m` and `n` are their
    stoichiometric coefficients (extends as product with multiple substrate and product species).

    I can't solve this in general. Thus, to limit the extend of such a flickering around the
    equilibrium state `Km`s of the backwards reaction should the inverse of the forward reaction.
    E.g. if the forward reaction was very sensitive and tends to overshoot the equilibrium state,
    the backward reaction will be very unsensitive and only slowly approach the equilibrium state
    from the other side. As transporters also work with this mechanism, they tend to have a
    direction. So, if a transporter has a high affinity in one direction, it will have a low
    affinity in the other.

    Somehwat related: As reactions can overshoot their equilibriums state, they can also
    overshoot the actual substrate concentrations. I.e. it can happen that a protein tries to
    react with more substrate than actually available (e.g. very low `Km`, high `Vmax`). In
    addition, each protein in the cell is not aware of the other proteins. So, 2 proteins could
    both try to react with substrate `S`. And while each of them would have had enough `S`, both
    of them together actually reduce `S` to a negative concentration.

    To avoid this there is a correction term that limits the velocity of any protein to the actual
    abundance of the substrates. If there are multiple proteins in a cell all using up the same
    substrate, these limiting substrate abundances are shared equally among these proteins.
    Furthermore, instead of allowing the proteins to reduce substrates to exactly `0.0`, they will
    actually only be allowed to reduce any substrate to a small number `eps > 0.0`. This is to
    avoid creating infinite reaction quotients and it is another mitigation for the problem of
    overshooting the equilibirum state.

    Limitations:
    - all based on Michaelis-Menten kinetics, no cooperativity
    - all allosteric control is non-competitive (activating or inhibiting)
    - there are substrate-substrate interactions but no interactions among effectors
    - 1 protein can have multiple substrates and products but there is only 1 Km for each type of molecule
    - there can only be 1 effector per molecule (e.g. not 2 different allosteric controls for the same type of molecule)
    - proteins can catalyze reactions in a way that they overshoot their equilibirum state (heuristics try to avoid that, see text above)
    """

    def __init__(
        self,
        n_signals: int,
        abs_temp=310.0,
        dtype=torch.float,
        device="cpu",
        n_workers=4,
    ):
        self.n_signals = n_signals
        self.abs_temp = abs_temp
        self.dtype = dtype
        self.device = device
        self.n_workers = n_workers
        self.torch_kwargs = {"dtype": dtype, "device": device}

        self.Km = self._tensor(0, 0, n_signals)
        self.Vmax = self._tensor(0, 0)
        self.E = self._tensor(0, 0)
        self.N = self._tensor(0, 0, n_signals)
        self.A = self._tensor(0, 0, n_signals)

    def unset_cell_params(self, cell_prots: list[tuple[int, int]]):
        """Set cell params for these proteins to 0.0"""
        if len(cell_prots) == 0:
            return
        cells, prots = list(map(list, zip(*cell_prots)))
        self.E[cells, prots] = 0.0
        self.Vmax[cells, prots] = 0.0
        self.Km[cells, prots] = 0.0
        self.A[cells, prots] = 0.0
        self.N[cells, prots] = 0.0

    def set_cell_params(self, cell_prots: list[tuple[int, int, Protein]]):
        """Set cell params for these proteins accordingly"""
        if len(cell_prots) == 0:
            return

        with Pool(self.n_workers) as pool:
            args = [(d[0], d[1], self.n_signals, d[2]) for d in cell_prots]
            res = pool.starmap(Kinetics._get_protein_params, args)

        cis = []
        pis = []
        E = []
        Km = []
        Vmax = []
        A = []
        N = []
        for ci, pi, e, k, v, a, n in res:
            cis.append(ci)
            pis.append(pi)
            E.append(e)
            Km.append(k)
            Vmax.append(v)
            A.append(a)
            N.append(n)
        self.E[cis, pis] = torch.tensor(E)
        self.Km[cis, pis] = torch.tensor(Km)
        self.Vmax[cis, pis] = torch.tensor(Vmax)
        self.A[cis, pis] = torch.tensor(A)
        self.N[cis, pis] = torch.tensor(N)

    def integrate_signals(self, X: torch.Tensor) -> torch.Tensor:
        """
        Simulate protein work by integrating all signals.

        - `X` Signal/molecule concentrations (c, s). Must all be >= 0.0.
        
        Returns `delta X`, the molecules deconstructed or constructed during
        this time step.
        """
        _log.debug(
            "Integrate signals with (c, p, s)=(%i, %i, %i)", *list(self.Km.shape),
        )

        inh_V = self._get_inhibitor_activity(X=X)  # (c, p)
        act_V = self._get_activator_activity(X=X)  # (c, p)

        # adjust direction
        Ke = torch.exp(-self.E / self.abs_temp / GAS_CONSTANT)
        Q = self._get_reaction_quotients(X=X)  # (c, p)
        adj_N = torch.where(Q > Ke, -1.0, 1.0)  # (c, p)
        N_adj = torch.einsum("cps,cp->cps", self.N, adj_N)

        prot_V = self._get_protein_activity(X=X, N=N_adj)  # (c, p)
        V = self.Vmax * prot_V * (1 - inh_V) * act_V  # (c, p)

        # get limiting velocities (substrate would be empty)
        X_max = torch.einsum("cs,cps->cps", -X + EPS, 1 / N_adj)
        V_limit = X_max.max(dim=2).values.clamp(0.0)

        Xd = torch.einsum("cps,cp->cs", N_adj, V.clamp(max=V_limit).nan_to_num())

        # low float precision can still drive X below 0
        # final lazy correction
        Xd = Xd.clamp(-X + EPS)

        return Xd

    def copy_cell_params(self, from_idxs: list[int], to_idxs: list[int]):
        """Copy paremeters from a list of cells to another list of cells"""
        self.Km[to_idxs] = self.Km[from_idxs]
        self.Vmax[to_idxs] = self.Vmax[from_idxs]
        self.E[to_idxs] = self.E[from_idxs]
        self.N[to_idxs] = self.N[from_idxs]
        self.A[to_idxs] = self.A[from_idxs]

    def remove_cell_params(self, keep: torch.Tensor):
        """Filter cell params for cells in `keep`"""
        self.Km = self.Km[keep]
        self.Vmax = self.Vmax[keep]
        self.E = self.E[keep]
        self.N = self.N[keep]
        self.A = self.A[keep]

    def increase_max_cells(self, by_n: int):
        """Increase the cell dimension by `by_n`"""
        self.Km = self._expand(t=self.Km, n=by_n, d=0)
        self.Vmax = self._expand(t=self.Vmax, n=by_n, d=0)
        self.E = self._expand(t=self.E, n=by_n, d=0)
        self.N = self._expand(t=self.N, n=by_n, d=0)
        self.A = self._expand(t=self.A, n=by_n, d=0)

    def increase_max_proteins(self, max_n: int):
        """Increase the protein dimension to `max_n`"""
        n_prots = int(self.Km.shape[1])
        if max_n > n_prots:
            by_n = max_n - n_prots
            self.Km = self._expand(t=self.Km, n=by_n, d=1)
            self.Vmax = self._expand(t=self.Vmax, n=by_n, d=1)
            self.E = self._expand(t=self.E, n=by_n, d=1)
            self.N = self._expand(t=self.N, n=by_n, d=1)
            self.A = self._expand(t=self.A, n=by_n, d=1)

    def _get_reaction_quotients(self, X: torch.Tensor) -> torch.Tensor:
        # substrates
        sub_mask = self.N < 0.0  # (c, p, s)
        sub_X = torch.einsum("cps,cs->cps", sub_mask, X)  # (c, p, s)
        sub_N = sub_mask * -self.N  # (c, p, s)

        # products
        pro_mask = self.N > 0.0  # (c, p s)
        pro_X = torch.einsum("cps,cs->cps", pro_mask, X)  # (c, p, s)
        pro_N = pro_mask * self.N  # (c, p, s)

        # quotients
        nom = torch.pow(pro_X, pro_N).prod(2)  # (c, p)
        denom = torch.pow(sub_X, sub_N).prod(2)  # (c, p)
        return nom / denom  # (c, p)

    def _get_protein_activity(self, X: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        mask = N < 0.0  # (c, p s)
        sub_X = torch.einsum("cps,cs->cps", mask, X)  # (c, p, s)
        sub_N = mask * -N  # (c, p, s)
        return torch.pow(sub_X / (self.Km + sub_X), sub_N).prod(2)  # (c, p)

    def _get_inhibitor_activity(self, X: torch.Tensor) -> torch.Tensor:
        inh_N = (-self.A).clamp(0)  # (c, p, s)
        inh_X = torch.einsum("cps,cs->cps", inh_N, X)  # (c, p, s)
        V = torch.pow(inh_X / (self.Km + inh_X), inh_N).prod(2)  # (c, p)
        return torch.where(torch.any(self.A < 0.0, dim=2), V, 0.0)  # (c, p)

    def _get_activator_activity(self, X: torch.Tensor) -> torch.Tensor:
        act_N = self.A.clamp(0)  # (c, p, s)
        act_X = torch.einsum("cps,cs->cps", act_N, X)  # (c, p, s)
        V = torch.pow(act_X / (self.Km + act_X), act_N).prod(2)  # (c, p)
        return torch.where(torch.any(self.A > 0.0, dim=2), V, 1.0)  # (c, p)

    def _expand(self, t: torch.Tensor, n: int, d: int) -> torch.Tensor:
        pre = t.shape[slice(d)]
        post = t.shape[slice(d + 1, t.dim())]
        zeros = self._tensor(*pre, n, *post)
        return torch.cat([t, zeros], dim=d)

    def _tensor(self, *args, **kwargs) -> torch.Tensor:
        return torch.zeros(*args, **{**self.torch_kwargs, **kwargs})

    @staticmethod
    def _get_protein_params(
        cell_i: int, prot_i: int, n_signals: int, protein: Protein
    ) -> tuple[int, int, float, list[float], float, list[float], list[float]]:
        energy = 0.0
        Km: list[list[float]] = [[] for _ in range(n_signals)]
        Vmax: list[float] = []
        A: list[float] = [0.0 for _ in range(n_signals)]
        N: list[float] = [0.0 for _ in range(n_signals)]

        for dom in protein.domains:

            if dom.is_allosteric:
                mol = dom.substrates[0]
                if dom.is_transmembrane:
                    mol_i = mol.ext_idx
                else:
                    mol_i = mol.int_idx
                Km[mol_i].append(dom.affinity)
                A[mol_i] += -1.0 if dom.is_inhibiting else 1.0

            if dom.is_transporter:
                Vmax.append(dom.velocity)
                mol = dom.substrates[0]

                if dom.is_bkwd:
                    sub_i = mol.ext_idx
                    prod_i = mol.int_idx
                else:
                    sub_i = mol.int_idx
                    prod_i = mol.ext_idx

                Km[sub_i].append(dom.affinity)
                N[sub_i] -= 1.0

                Km[prod_i].append(1 / dom.affinity)
                N[prod_i] += 1.0

            if dom.is_catalytic:
                Vmax.append(dom.velocity)

                if dom.is_bkwd:
                    subs = dom.products
                    prods = dom.substrates
                else:
                    subs = dom.substrates
                    prods = dom.products

                for mol in subs:
                    energy -= mol.energy
                    mol_i = mol.int_idx
                    Km[mol_i].append(dom.affinity)
                    N[mol_i] -= 1.0

                for mol in prods:
                    energy += mol.energy
                    mol_i = mol.int_idx
                    Km[mol_i].append(1 / dom.affinity)
                    N[mol_i] += 1.0

        v = sum(Vmax) / len(Vmax) if len(Vmax) > 0 else 0.0
        ks = [sum(d) / len(d) if len(d) > 0 else 0.0 for d in Km]
        return cell_i, prot_i, energy, ks, v, A, N

