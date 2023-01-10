import torch
from magicsoup.constants import GAS_CONSTANT
from magicsoup.containers import Protein, Molecule


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
    - `A` regulatory control of all proteins for each molecule (c, p, s). Numbers > 0.0 mean these molecules
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

    Limitations:
    - all based on Michaelis-Menten kinetics, no cooperativity
    - all allosteric control is non-competitive (activating or inhibiting)
    - there are substrate-substrate interactions but no interactions among effectors
    - 1 protein can have multiple substrates and products but there is only 1 Km for each type of molecule
    - there can only be 1 effector per molecule (e.g. not 2 different allosteric controls for the same type of molecule)
    - proteins can catalyze reactions in a way that they overshoot their equilibirum state (heuristics try to avoid that, see text above)
    """

    # TODO: use torch.amax everywhere....

    def __init__(
        self,
        molecules: list[Molecule],
        abs_temp=310.0,
        dtype=torch.float,
        device="cpu",
    ):
        n = len(molecules)
        self.n_signals = 2 * n
        self.int_mol_map = {d.name: i for i, d in enumerate(molecules)}
        self.ext_mol_map = {d.name: i + n for i, d in enumerate(molecules)}

        self.abs_temp = abs_temp
        self.dtype = dtype
        self.device = device

        self.Km = self._tensor(0, 0, self.n_signals)
        self.Vmax = self._tensor(0, 0)
        self.E = self._tensor(0, 0)
        self.N = self._tensor(0, 0, self.n_signals)
        self.A = self._tensor(0, 0, self.n_signals)

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

        cis = []
        pis = []
        E = []
        Km = []
        Vmax = []
        A = []
        N = []
        for ci, pi, prot in cell_prots:
            e, k, v, a, n = self._get_protein_params(protein=prot)
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
        inh_V = self._get_inhibitor_activity(X=X)  # (c, p)
        act_V = self._get_activator_activity(X=X)  # (c, p)

        # adjust direction
        Ke = torch.exp(-self.E / self.abs_temp / GAS_CONSTANT)
        Q = self._get_reaction_quotients(X=X)  # (c, p)
        adj_N = torch.where(Q > Ke, -1.0, 1.0)  # (c, p)
        N_adj = torch.einsum("cps,cp->cps", self.N, adj_N)

        prot_V = self._get_protein_activity(X=X, N=N_adj)  # (c, p)
        V = self.Vmax * prot_V * (1 - inh_V) * act_V  # (c, p)
        Xd = torch.einsum("cps,cp->cs", N_adj, V)  # (c, s)

        # if I want to reduce specific protein velocities only
        # I also have to account for the interaction of protein:
        # e.g. I might reduce protein 0 which destroyed too much molecule A
        # but it also produced molecule B, which is now produced less
        # and now molecule B destroying protein 1, which was not an issue
        # before, now also has to be reduced
        # here I am taking the easy route and just reduce all proteins in the cell
        # by the same value
        fact = torch.where(X + Xd < 0.0, X * 0.99 / -Xd, 1.0)  # (c, s)
        Xd = torch.einsum("cs,c->cs", Xd, fact.amin(1))

        # # limit velocities
        # # TODO: theoretically this should reduce V enough to not produce
        # #       zeros, but it seems due to floating point errors, Xd still goes below 0
        # sig_fact = torch.where(
        #     X + Xd < 0.0, (X - 0.1).clamp(min=0.0) * 0.9 / -Xd - 0.1, 1.0
        # )  # (c, s)
        # sig_fact[sig_fact < 0.1] = 0.0
        # prot_fact = torch.einsum(
        #     "cs,cps->cps", sig_fact, torch.ones(N_adj.size())
        # )  # (c, p, s)
        # prot_fact[N_adj >= 0.0] = 1.0
        # # sig_fact = (X * 0.5 / -Xd).nan_to_num(1.0, 1.0, 1.0)
        # # prot_fact = torch.einsum("cs,cps->cps", sig_fact, N_adj < 0.0)
        # # prot_fact[prot_fact <= 0.0] = 1.0
        # v_fact = prot_fact.amin(2)
        # V = v_fact.clamp(max=1.0) * V  # (c, p)
        # Xd = torch.einsum("cps,cp->cs", N_adj, V)

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

    def _get_protein_params(
        self, protein: Protein
    ) -> tuple[float, list[float], float, list[float], list[float]]:
        energy = 0.0
        Km: list[list[float]] = [[] for _ in range(self.n_signals)]
        Vmax: list[float] = []
        A: list[float] = [0.0 for _ in range(self.n_signals)]
        N: list[float] = [0.0 for _ in range(self.n_signals)]

        for dom in protein.domains:

            if dom.is_regulatory:
                mol = dom.substrates[0]
                if dom.is_transmembrane:
                    mol_i = self.ext_mol_map[mol.name]
                else:
                    mol_i = self.int_mol_map[mol.name]
                Km[mol_i].append(dom.affinity)
                A[mol_i] += -1.0 if dom.is_inhibiting else 1.0

            if dom.is_transporter:
                Vmax.append(dom.velocity)
                mol = dom.substrates[0]

                if dom.is_bkwd:
                    sub_i = self.ext_mol_map[mol.name]
                    prod_i = self.int_mol_map[mol.name]
                else:
                    sub_i = self.int_mol_map[mol.name]
                    prod_i = self.ext_mol_map[mol.name]

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
                    mol_i = self.int_mol_map[mol.name]
                    Km[mol_i].append(dom.affinity)
                    N[mol_i] -= 1.0

                for mol in prods:
                    energy += mol.energy
                    mol_i = self.int_mol_map[mol.name]
                    Km[mol_i].append(1 / dom.affinity)
                    N[mol_i] += 1.0

        v = sum(Vmax) / len(Vmax) if len(Vmax) > 0 else 0.0
        ks = [sum(d) / len(d) if len(d) > 0 else 0.0 for d in Km]
        return energy, ks, v, A, N

    def _expand(self, t: torch.Tensor, n: int, d: int) -> torch.Tensor:
        pre = t.shape[slice(d)]
        post = t.shape[slice(d + 1, t.dim())]
        zeros = self._tensor(*pre, n, *post)
        return torch.cat([t, zeros], dim=d)

    def _tensor(self, *args) -> torch.Tensor:
        return torch.zeros(*args, dtype=self.dtype).to(self.device)


# TODO: negative concentrations
#       need to avoid without clamping
"""
import torch
from magicsoup.constants import GAS_CONSTANT

# fmt: off
Km = torch.tensor([[[1.0852, 0.4830, 0.4702, 0.6562, 1.4267, 0.4033, 0.4366, 0.7077],
         [0.3672, 0.9985, 0.3935, 1.4161, 0.2647, 1.7397, 1.3877, 0.9762],
         [1.3251, 2.3163, 1.2171, 0.8589, 1.5683, 1.9996, 0.4057, 0.0198],
         [0.6043, 0.3691, 1.5378, 0.4824, 0.4397, 0.9057, 0.3077, 1.8780]],

        [[0.8746, 0.7680, 1.2370, 1.2017, 0.8733, 0.2967, 0.5571, 1.2091],
         [1.6995, 0.4635, 1.6632, 1.2919, 0.2911, 1.8839, 0.8749, 0.1237],
         [0.6656, 0.1083, 1.1277, 0.5231, 1.2678, 1.1612, 0.7818, 1.0274],
         [0.7885, 2.5267, 1.1163, 0.7970, 0.5118, 0.0632, 1.5325, 0.3691]],

        [[1.0664, 0.7554, 0.4640, 1.0391, 0.4447, 0.4967, 1.6364, 0.8434],
         [1.8537, 0.8482, 0.2371, 0.1783, 0.0927, 0.0691, 0.4484, 0.6307],
         [0.4507, 1.4069, 1.3742, 0.7561, 0.7389, 0.0769, 0.0123, 0.3157],
         [2.3395, 1.3122, 1.2445, 1.0203, 0.9379, 0.4009, 0.7176, 1.3371]],

        [[0.4613, 0.0276, 0.5041, 0.6459, 0.4302, 0.8680, 0.1937, 0.6582],
         [1.7052, 0.2276, 0.7294, 0.4220, 0.3188, 0.3629, 0.3002, 1.2323],
         [0.3939, 0.0136, 0.8156, 0.6482, 1.0903, 0.5469, 1.2396, 0.1570],
         [0.4027, 0.4840, 0.3973, 0.8541, 0.3751, 0.2451, 0.0575, 0.5700]],

        [[0.4412, 0.4627, 1.1986, 1.5639, 1.1717, 1.9279, 1.1178, 0.0885],
         [0.0602, 2.7953, 0.4728, 0.3091, 0.2077, 1.5234, 0.5015, 1.2487],
         [0.7083, 0.7809, 0.0904, 0.2434, 0.8732, 0.6628, 0.1446, 2.0374],
         [1.3990, 0.4092, 0.9279, 0.2084, 0.9799, 1.0271, 0.8243, 0.6731]],

        [[0.2859, 1.9411, 1.8658, 0.1039, 0.0157, 0.6261, 0.1579, 0.3080],
         [1.1289, 0.6053, 0.7617, 0.8440, 1.1069, 0.6247, 0.5879, 0.1840],
         [1.0592, 0.1933, 0.3453, 0.7428, 0.2039, 1.0766, 0.0993, 0.4631],
         [1.2444, 0.9293, 0.5483, 0.0071, 0.8766, 0.2344, 1.6868, 0.3444]],

        [[0.9438, 0.6309, 0.0291, 0.1800, 0.2914, 0.7978, 1.2495, 0.4345],
         [0.3208, 0.1701, 3.0364, 0.0984, 0.8076, 1.1575, 0.9068, 0.5950],
         [1.3799, 0.3265, 1.0578, 0.4502, 0.3099, 1.1174, 0.0662, 1.1941],
         [0.0678, 1.3197, 1.1246, 0.4922, 0.9860, 1.5518, 0.0495, 1.3284]],

        [[0.1626, 0.0677, 1.7732, 1.3239, 0.1464, 1.3834, 0.2641, 1.5463],
         [0.3072, 0.1059, 1.7039, 1.3110, 0.6068, 1.3085, 1.5805, 0.0125],
         [1.1665, 1.2570, 0.1207, 0.1562, 0.8201, 0.3521, 0.2984, 1.1118],
         [0.2678, 0.3048, 0.4131, 1.7119, 1.9506, 0.8307, 0.1419, 1.7283]],

        [[0.7859, 0.4019, 0.1755, 0.4489, 1.6371, 0.8230, 0.8785, 0.5640],
         [0.7410, 0.1100, 0.6758, 0.1036, 0.4442, 0.6524, 0.4032, 0.6894],
         [0.0937, 0.4635, 0.3012, 0.4719, 0.5802, 2.0570, 0.9934, 0.6013],
         [0.8506, 0.6473, 0.0668, 0.2984, 0.1502, 0.0073, 1.8854, 0.4886]],

        [[0.9932, 0.8448, 0.5871, 0.2526, 1.8674, 0.9538, 2.5900, 0.1951],
         [0.0996, 1.5111, 0.3476, 0.3965, 1.5644, 0.5263, 0.4086, 0.3109],
         [1.1890, 1.0443, 0.7408, 0.0052, 0.2673, 0.4953, 0.0975, 0.8945],
         [0.5243, 1.0628, 0.0854, 1.1276, 1.0119, 1.4205, 1.1484, 1.5693]]])
E = torch.tensor([[-0.4679, -1.5542, -0.9607, -0.0587],
        [-0.8109,  1.3131, -0.2389,  1.3911],
        [-2.0653, -1.8964,  0.6599, -0.9933],
        [ 0.3705, -0.3155,  0.4594, -2.3917],
        [-0.8002,  1.1552, -1.4458, -0.1827],
        [-0.7070, -0.2683,  2.1252, -0.2263],
        [ 0.0360,  1.8811,  2.0211, -0.6860],
        [-1.4400, -0.8365,  0.1922,  0.0547],
        [ 0.7447, -0.7142, -0.0717, -0.0670],
        [ 0.5654, -1.4396,  2.2156, -0.1697]])
N = torch.tensor([[[ 2, -2,  0, -2,  2,  2,  0,  3],
         [-1, -3,  1, -2,  2,  0, -3, -2],
         [ 3,  1, -3,  2, -2, -3, -1,  3],
         [ 3,  0, -2, -3, -3,  0, -3, -3]],

        [[-3,  0,  2,  0,  2,  1,  0,  0],
         [-2, -3, -1,  0,  2, -2, -1, -2],
         [-3, -1, -1,  3, -3,  0, -1, -2],
         [-1, -1,  0,  1, -1,  3,  2,  2]],

        [[ 2, -1,  0, -1,  1,  2,  1,  3],
         [ 0, -2,  3,  1,  2,  0,  3, -3],
         [ 0,  3, -1,  0, -3,  2, -1,  3],
         [-3, -1, -3,  3,  0, -1, -1,  0]],

        [[-3, -3, -3,  3, -1,  1,  2,  3],
         [ 0,  3, -3, -1,  3,  2, -1,  1],
         [-3,  0,  3,  3,  0,  1,  0, -2],
         [ 0,  2, -3, -2, -1,  1, -2, -1]],

        [[ 3,  0,  2,  2, -2,  2,  2,  0],
         [ 0,  2, -2,  0,  3,  3, -3,  0],
         [ 1, -1, -3, -2,  3, -3,  3,  2],
         [ 3,  0, -2,  1, -3,  0,  0,  3]],

        [[ 0,  1, -1, -3,  2, -1, -2, -1],
         [ 3, -1, -1,  2,  3,  1,  2,  2],
         [ 0,  0,  0,  0,  3,  3,  3, -2],
         [-3, -1,  1,  2,  0, -2,  0,  0]],

        [[ 2, -3,  2, -3,  1, -2,  3,  2],
         [ 1,  1,  1,  3, -2, -2, -3,  3],
         [-1, -2,  2,  3,  2,  0, -1, -3],
         [-1, -1,  2, -1, -1, -3, -2, -2]],

        [[ 1,  2,  0, -1,  0,  1, -3,  3],
         [ 2,  2, -3,  2,  3, -3, -1, -2],
         [ 2,  1, -2,  1, -1,  2, -1,  0],
         [-1, -3,  3,  2,  1, -1, -3,  3]],

        [[ 0, -2,  2, -2, -2, -2,  0, -3],
         [-3, -1, -1,  3,  0,  0, -3,  0],
         [ 3,  3,  0,  3, -1, -2, -2,  2],
         [-2,  1,  2,  2, -1, -1, -2, -2]],

        [[ 1, -2,  1, -1,  2, -3, -3, -3],
         [ 2,  3,  0,  1,  3, -1,  0, -1],
         [ 0,  0, -2,  2, -1, -3,  1,  1],
         [ 2,  0, -1,  3,  3,  0,  1, -1]]])
A = torch.tensor([[[ 1, -1, -1,  2, -1, -2, -2,  2],
         [ 1,  2, -1, -2, -1,  0, -1,  0],
         [-1,  2,  1, -1,  1, -1, -1,  1],
         [ 2, -2,  2,  0,  0, -2,  0,  0]],

        [[ 2,  2,  2, -2, -1,  0,  0,  2],
         [ 1, -1,  2,  2,  0, -1, -1,  2],
         [ 1,  0,  1,  1,  2, -2, -2,  1],
         [ 1,  0, -1, -1,  1, -1, -1, -1]],

        [[ 0,  1,  2, -1, -2,  2,  0, -2],
         [-1, -1,  2,  0,  0, -1,  2,  2],
         [-1, -2, -1, -1,  2, -2, -2, -2],
         [-2,  1,  0,  0, -1,  2, -2, -2]],

        [[ 1,  0, -1, -1,  0,  0, -1, -2],
         [-1,  0,  0, -2,  1,  2,  1,  0],
         [-2, -2,  0, -1,  1,  0, -2,  0],
         [-2, -1, -1,  1, -1, -2,  0,  0]],

        [[ 2,  1,  2, -1,  1, -2, -1,  0],
         [-1, -1,  2, -2,  0, -2, -1,  0],
         [ 2,  2,  0,  1,  1, -1,  1,  1],
         [-2, -2,  1, -1, -1, -1,  2, -2]],

        [[ 0, -2,  0,  0, -2,  1,  2,  0],
         [ 0,  0, -1,  0,  1,  2,  0, -2],
         [ 2,  1,  2,  2, -2,  1,  1,  1],
         [-2, -1, -2, -1,  2,  2,  2,  1]],

        [[ 1,  0,  1,  1, -1,  2, -2,  0],
         [ 0,  2,  1,  0,  0,  2, -1, -1],
         [ 0,  0, -2, -1,  2, -2,  0,  1],
         [ 2, -2, -2,  2,  0,  2,  0,  1]],

        [[ 1, -1, -2,  1,  1, -1,  0, -1],
         [ 2, -1,  0, -1,  0, -2,  2, -2],
         [ 0, -1,  2,  0,  1, -1,  0,  0],
         [ 0,  2, -1,  2, -2, -2,  1,  1]],

        [[ 0, -1,  0, -2,  0, -1,  0, -2],
         [ 1,  0,  1,  1, -1,  0,  2,  0],
         [ 2, -1, -1,  1, -1, -2,  2, -1],
         [ 0,  1,  0,  2, -1,  1, -2,  2]],

        [[-2,  1,  2,  1, -2,  1, -1, -2],
         [ 2,  1, -2, -1,  0,  1,  2,  2],
         [ 1,  1, -1, -2, -2,  2, -1, -2],
         [-2, -2, -1,  1, -1, -1, -2,  1]]])
Vmax = torch.tensor([[10.9343, 16.7540, 11.0745,  1.1482],
        [ 2.2539,  2.3776, 10.2157,  7.8683],
        [ 0.8693,  3.0228,  9.7119,  1.2059],
        [11.4163,  5.2745,  8.7387, 13.7377],
        [ 7.3843,  9.0372, 11.6707,  3.2419],
        [ 0.9470,  5.0971,  6.9764,  3.6665],
        [ 4.6183,  4.0886,  3.8202,  1.3474],
        [ 0.9529,  0.9190, 11.6070, 11.5748],
        [13.6915,  1.9257,  8.4612,  4.8804],
        [ 0.2800, 20.3668,  4.5160,  3.9846]])
X = torch.tensor([[1.3967, 0.5859, 1.6266, 1.8686, 0.8111, 0.9110, 2.7816, 2.1297],
        [0.3937, 0.5790, 0.6452, 0.4828, 0.3758, 0.9087, 1.1773, 0.6520],
        [0.9615, 0.4406, 1.5278, 0.2525, 2.3162, 1.0911, 0.1839, 0.6617],
        [0.5031, 0.7432, 1.8240, 0.3737, 1.0587, 0.5245, 0.9836, 1.7745],
        [0.6006, 0.6160, 1.1453, 0.8749, 0.4503, 2.3886, 0.5489, 0.5941],
        [1.6985, 1.1436, 1.2108, 1.0949, 0.9687, 0.6723, 1.1374, 0.6414],
        [0.4885, 0.1102, 0.6269, 0.3732, 1.0762, 1.9160, 0.5519, 2.0911],
        [0.3598, 0.8746, 0.6879, 0.4730, 1.3414, 0.1371, 1.1396, 0.4291],
        [0.8262, 0.7453, 0.5895, 0.3090, 0.1471, 0.0649, 0.5566, 0.3462],
        [0.7070, 1.4802, 1.8659, 1.4581, 2.4362, 2.0599, 0.8083, 1.4367]])

# fmt: on


def _get_reaction_quotients(X: torch.Tensor) -> torch.Tensor:
    # substrates
    sub_mask = N < 0.0  # (c, p, s)
    sub_X = torch.einsum("cps,cs->cps", sub_mask, X)  # (c, p, s)
    sub_N = sub_mask * -N  # (c, p, s)

    # products
    pro_mask = N > 0.0  # (c, p s)
    pro_X = torch.einsum("cps,cs->cps", pro_mask, X)  # (c, p, s)
    pro_N = pro_mask * N  # (c, p, s)

    # quotients
    nom = torch.pow(pro_X, pro_N).prod(2)  # (c, p)
    denom = torch.pow(sub_X, sub_N).prod(2)  # (c, p)
    return nom / denom  # (c, p)


def _get_protein_activity(X: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
    mask = N < 0.0  # (c, p s)
    sub_X = torch.einsum("cps,cs->cps", mask, X)  # (c, p, s)
    sub_N = mask * -N  # (c, p, s)
    return torch.pow(sub_X / (Km + sub_X), sub_N).prod(2)  # (c, p)


def _get_inhibitor_activity(X: torch.Tensor) -> torch.Tensor:
    inh_N = (-A).clamp(0)  # (c, p, s)
    inh_X = torch.einsum("cps,cs->cps", inh_N, X)  # (c, p, s)
    V = torch.pow(inh_X / (Km + inh_X), inh_N).prod(2)  # (c, p)
    return torch.where(torch.any(A < 0.0, dim=2), V, 0.0)  # (c, p)


def _get_activator_activity(X: torch.Tensor) -> torch.Tensor:
    act_N = A.clamp(0)  # (c, p, s)
    act_X = torch.einsum("cps,cs->cps", act_N, X)  # (c, p, s)
    V = torch.pow(act_X / (Km + act_X), act_N).prod(2)  # (c, p)
    return torch.where(torch.any(A > 0.0, dim=2), V, 1.0)  # (c, p)


def f(X):
    inh_V = _get_inhibitor_activity(X=X)  # (c, p)
    act_V = _get_activator_activity(X=X)  # (c, p)

    # adjust direction
    Ke = torch.exp(-E / 310 / GAS_CONSTANT)
    Q = _get_reaction_quotients(X=X)  # (c, p)
    adj_N = torch.where(Q > Ke, -1.0, 1.0)  # (c, p)
    N_adj = torch.einsum("cps,cp->cps", N, adj_N)

    prot_V = _get_protein_activity(X=X, N=N_adj)  # (c, p)
    V = Vmax * prot_V * (1 - inh_V) * act_V  # (c, p)

    # get limiting velocities (substrate would be empty)
    # TODO: found Problem
    # cell 1 for signal 7: protein 0 reduces signal by 3
    #                      but netto signal production is +2
    #                      so the XV entries are 0.0 (!)
    # Error: N needs to be adjusted with V!!!!
    Xd_naive = torch.einsum("cps,cp->cs", N_adj, V)  # (c, s)
    sig_fact = (X / -Xd_naive).nan_to_num(1.0, 1.0, 1.0)  # (c, s)
    prot_fact = torch.einsum("cs,cps->cps", sig_fact, N_adj < 0.0)  # (c, p, s)
    prot_fact[prot_fact <= 0.0] = 1.0
    V_adj = prot_fact.amin(2).clamp(max=1.0) * V  # (c, p)
    # X = 1.1367
    # N = -3
    # V = 1.9636
    # Xd_naive = -5.8651 = -3 * 1.9636
    # zu viel: -5.8651 + 1.1367 = -4.7284
    # muss reduzieren 1.1367 / 5.8651 = 0.1938
    # reduziert: Xd = -1.14163704 = -3 * 1.9636 * 0.1938

    # X_max = -X / N_adj.sum(1)  # (c, s)
    # XV = torch.einsum("cs,cps->cps", X_max, N_adj < 0.0)  # (c, p, s)
    # XV[XV.isnan()] = torch.inf
    # XV[XV <= 0.0] = torch.inf
    # V_limit = XV.min(2).values  # (c, p)

    Xd = torch.einsum("cps,cp->cs", N_adj, V_adj)

    return Xd


def f2(X):
    inh_V = _get_inhibitor_activity(X=X)  # (c, p)
    act_V = _get_activator_activity(X=X)  # (c, p)

    # adjust direction
    Ke = torch.exp(-E / 310 / GAS_CONSTANT)
    Q = _get_reaction_quotients(X=X)  # (c, p)
    adj_N = torch.where(Q > Ke, -1.0, 1.0)  # (c, p)
    N_adj = torch.einsum("cps,cp->cps", N, adj_N)

    prot_V = _get_protein_activity(X=X, N=N_adj)  # (c, p)
    V = Vmax * prot_V * (1 - inh_V) * act_V  # (c, p)
    Xd = torch.einsum("cps,cp->cs", N_adj, V)  # (c, s)

    # if I want to reduce specific protein velocities only
    # I also have to account for the interaction of protein:
    # e.g. I might reduce protein 0 which destroyed too much molecule A
    # but it also produced molecule B, which is now produced less
    # and now molecule B destroying protein 1, which was not an issue
    # before, now also has to be reduced
    # here I am taking the easy route and just reduce all proteins in the cell
    # by the same value
    fact = torch.where(X + Xd < 0.0, X / -Xd, 1.0)  # (c, s)
    Xd = torch.einsum("cs,c->cs", Xd, fact.amin(1))

    return Xd


torch.any(X + f2(X) < 0.0)
# cell 3, signal 3
"""
