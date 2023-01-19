import torch
from magicsoup.constants import GAS_CONSTANT
from magicsoup.containers import Protein, Molecule

_EPS = 1e-5


class Kinetics:
    """
    Class holding logic for simulating protein work.
    Usually this class is instantiated automatically when initializing `world`.
    You can access it on `world.kinetics`.

    - `molecules` List of molecule species.
      They have to be in the same order as they are on `chemistry.molecules`.
    - `abs_temp` Absolute temperature in Kelvin will influence the free Gibbs energy calculation of reactions.
      Higher temperature will give the reaction quotient term higher importance.
    - `device` Device to use for tensors
      (see [pytorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)).
      This has to be the same device that is used by `world`.
    
    There are `c` cells, `p` proteins, `s` signals.
    Signals are basically molecule species, but we have to differentiate between intra- and extracellular molecule species.
    So, there are twice as many signals as molecule species.
    The order of molecule species is always the same as in `chemistry.molecules`.
    First, all intracellular molecule species are listed, then all extracellular.
    The order of cells is always the same as in `world.cells` and the order of proteins
    for every cell is always the same as the order of proteins in a cell object `cell.proteome`.
    
    Attributes on this class describe cell parameters:

    - `Km` Affinities to every signal that is processed by each protein in every cell (c, p, s).
    - `Vmax` Maximum velocities of every protein in every cell (c, p).
    - `E` Standard reaction Gibbs free energy of every protein in every cell (c, p).
    - `N` Stoichiometric number for every signal that is processed by each protein in every cell (c, p, s).
    - `A` Regulatory effect for each signal in every protein in every cell (c, p, s).
      This is looks similar to a stoichiometric number. Numbers > 0.0 mean these molecules
      act as activating effectors, numbers < 0.0 mean these molecules act as inhibiting effectors.
    
    The main method is `kinetics.integrate_signals()`.
    When calling `world.enzymatic_activity()`, a matrix `X` of signals (c, s) is prepared
    and then `kinetics.integrate_signals(X)` is called.
    Updated signals are returned and `world` writes them back to `world.cell_molecules` and `world.molecule_map`.

    Another method, which ended up here, is `kinetics.set_cell_params()` (and `kinetics.unset_cell_params()`)
    which reads proteomes and updates cell parameters accordingly.
    This is called whenever the proteomes of some cells changed.
    Currently, this is also the main bottleneck in performance.
    """

    # TODO: I could use some native torch functions in some places, e.g. ReLU
    #       might be faster than matrix multiplications

    # TODO: are the molecule maps faster if molecule2idx (instead of molecule name)?

    def __init__(
        self, molecules: list[Molecule], abs_temp=310.0, device="cpu",
    ):
        n = len(molecules)
        self.n_signals = 2 * n
        self.int_mol_map = {d.name: i for i, d in enumerate(molecules)}
        self.ext_mol_map = {d.name: i + n for i, d in enumerate(molecules)}

        self.abs_temp = abs_temp
        self.device = device

        self.Km = self._tensor(0, 0, self.n_signals)
        self.Vmax = self._tensor(0, 0)
        self.E = self._tensor(0, 0)
        self.N = self._tensor(0, 0, self.n_signals)
        self.A = self._tensor(0, 0, self.n_signals)

    def unset_cell_params(self, cell_prots: list[tuple[int, int]]):
        """
        Set cell params for these proteins to 0.0
        
        - `cell_prots` list of tuples of cell indexes and protein indexes

        This is useful for cells that lost some of their proteins.
        """
        if len(cell_prots) == 0:
            return
        cells, prots = list(map(list, zip(*cell_prots)))
        self.E[cells, prots] = 0.0
        self.Vmax[cells, prots] = 0.0
        self.Km[cells, prots] = 0.0
        self.A[cells, prots] = 0.0
        self.N[cells, prots] = 0.0

    def set_cell_params(self, cell_prots: list[tuple[int, int, Protein]]):
        """
        Set cell params for these proteins accordingly
        
        - `cell_prots` list of tuples of cell indexes, protein indexes, and the protein itself

        You can compare proteins within a cell and only update the ones that changed.
        The comparison (`protein0 == protein1`) will note a difference in any of the proteins attributes.

        Indexes for proteins are the same as in a cell's object `cell.proteome`
        and indexes for cells are the same as in `world.cells` or `cell.idx`.
        """
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

        - `X` Tensor of every signal in every cell (c, s). Must all be >= 0.0.
        
        Returns a new tensor of the same shape which represents the updated signals for every cell.

        The order of cells in `X` is the same as in `world.cells`
        The order of signals is first all intracellular molecule species in the same order as `chemistry.molecules`,
        then again all molecule species in the same order but this time describing extracellular molecule species.
        The number of intracellular molecules comes from `world.cell_molecules` for any particular cell.
        The number of extracellular molecules comes from `world.molecule_map` from the pixel the particular cell currently lives on.
        """
        # adjust direction
        lKe = -self.E / self.abs_temp / GAS_CONSTANT  # (c, p)
        lQ = self._get_ln_reaction_quotients(X=X)  # (c, p)
        adj_N = torch.where(lQ > lKe, -1.0, 1.0)  # (c, p)
        N_adj = torch.einsum("cps,cp->cps", self.N, adj_N)

        # activations
        a_inh = self._get_inhibitor_activity(X=X)  # (c, p)
        a_act = self._get_activator_activity(X=X)  # (c, p)
        a_cat = self._get_catalytic_activity(X=X, N=N_adj)  # (c, p)

        # trim velocities when approaching equilibrium Q -> Ke
        # and stop reaction if Q ~= Ke
        # f(x) = {1.0 if x > 1.0; 0.0 if x <= 0.1; x otherwise}
        # with x being the abs(ln(Q) - ln(Ke))
        trim = (lQ - lKe).abs().clamp(max=1.0)  # (c, p)
        trim[trim.abs() <= 0.1] = 0.0

        # velocity and naive Xd
        V = self.Vmax * a_cat * (1 - a_inh) * a_act * trim  # (c, p)
        Xd = torch.einsum("cps,cp->cs", N_adj, V)  # (c, s)

        # proteins can deconstruct more of a molecule than available in a cell
        # Here, I am reducing the velocity of all proteins in a cell by the same
        # amount to just not deconstruct more of any molecule species than
        # available in this cell
        # One can also reduce only the velocity of the proteins which are part
        # of deconstructing the molecule species that is becomming the problem
        # but these reduced protein speeds could also have a downstream effect
        # on the construction of another molecule species, which could then
        # create the same problem again.
        # By reducing all proteins by the same factor, this cannot happen.
        fact = torch.where(X + Xd < 0.0, X * 0.99 / -Xd, 1.0)  # (c, s)
        Xd = torch.einsum("cs,c->cs", Xd, fact.amin(1))

        return (X + Xd).clamp(min=0.0)

    def copy_cell_params(self, from_idxs: list[int], to_idxs: list[int]):
        """
        Copy paremeters from a list of cells to another list of cells
        
        - `from_idxs` list of cell indexes to copy from
        - `to_idxs` list of cell indexes to copy to
        
        `from_idxs` and `to_idxs` must have the same length.
        They refer to the same cell indexes as in `world.cells`.
        """
        self.Km[to_idxs] = self.Km[from_idxs]
        self.Vmax[to_idxs] = self.Vmax[from_idxs]
        self.E[to_idxs] = self.E[from_idxs]
        self.N[to_idxs] = self.N[from_idxs]
        self.A[to_idxs] = self.A[from_idxs]

    def remove_cell_params(self, keep: torch.Tensor):
        """
        Remove cells from cell params

        - `keep` Bool tensor (c,) which is true for every cell that should not be removed
          and false for every cell that should be removed.
        
        `keep` must have the same length as `world.cells`.
        The indexes on `keep` reflect the indexes in `world.cells`.
        """
        self.Km = self.Km[keep]
        self.Vmax = self.Vmax[keep]
        self.E = self.E[keep]
        self.N = self.N[keep]
        self.A = self.A[keep]

    def increase_max_cells(self, by_n: int):
        """
        Increase the cell dimension of all cell parameters

        - `by_n` By how many rows to increase the cell dimension
        """
        self.Km = self._expand(t=self.Km, n=by_n, d=0)
        self.Vmax = self._expand(t=self.Vmax, n=by_n, d=0)
        self.E = self._expand(t=self.E, n=by_n, d=0)
        self.N = self._expand(t=self.N, n=by_n, d=0)
        self.A = self._expand(t=self.A, n=by_n, d=0)

    def increase_max_proteins(self, max_n: int):
        """
        Increase the protein dimension of all cell parameters

        - `max_n` The maximum number of rows required in the protein dimension
        """
        n_prots = int(self.Km.shape[1])
        if max_n > n_prots:
            by_n = max_n - n_prots
            self.Km = self._expand(t=self.Km, n=by_n, d=1)
            self.Vmax = self._expand(t=self.Vmax, n=by_n, d=1)
            self.E = self._expand(t=self.E, n=by_n, d=1)
            self.N = self._expand(t=self.N, n=by_n, d=1)
            self.A = self._expand(t=self.A, n=by_n, d=1)

    def _get_ln_reaction_quotients(self, X: torch.Tensor) -> torch.Tensor:
        # substrates
        smask = self.N < 0.0
        sub_X = torch.einsum("cps,cs->cps", smask, X)  # (c, p, s)
        sub_N = smask * -self.N  # (c, p, s)

        # products
        pmask = self.N > 0.0
        pro_X = torch.einsum("cps,cs->cps", pmask, X)  # (c, p, s)
        pro_N = pmask * self.N  # (c, p, s)

        # quotients
        prods = (torch.log(pro_X + _EPS) * pro_N).sum(2)  # (c, p)
        subs = (torch.log(sub_X + _EPS) * sub_N).sum(2)  # (c, p)
        return prods - subs  # (c, p)

    def _get_catalytic_activity(self, X: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        mask = N < 0.0
        sub_X = torch.einsum("cps,cs->cps", mask, X)  # (c, p, s)
        sub_N = mask * -N  # (c, p, s)
        lact = torch.log(sub_X + _EPS) - torch.log(self.Km + sub_X + _EPS)  # (c, p, s)
        return (lact * sub_N).sum(2).exp()

    def _get_inhibitor_activity(self, X: torch.Tensor) -> torch.Tensor:
        inh_N = (-self.A).clamp(0)  # (c, p, s)
        inh_X = torch.einsum("cps,cs->cps", inh_N, X)  # (c, p, s)
        lact = torch.log(inh_X + _EPS) - torch.log(self.Km + inh_X + _EPS)  # (c, p, s)
        V = (lact * inh_N).sum(2).exp()  # (c, p)
        return torch.where(torch.any(self.A < 0.0, dim=2), V, 0.0)  # (c, p)

    def _get_activator_activity(self, X: torch.Tensor) -> torch.Tensor:
        act_N = self.A.clamp(0)  # (c, p, s)
        act_X = torch.einsum("cps,cs->cps", act_N, X)  # (c, p, s)
        lact = torch.log(act_X + _EPS) - torch.log(self.Km + act_X + _EPS)  # (c, p, s)
        V = (lact * act_N).sum(2).exp()  # (c, p)
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
        return torch.zeros(*args).to(self.device)

