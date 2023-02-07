from typing import Optional, Any
import math
import random
import torch
from magicsoup.constants import GAS_CONSTANT, ALL_CODONS
from magicsoup.containers import (
    Molecule,
    Protein,
    CatalyticDomain,
    TransporterDomain,
    RegulatoryDomain,
    Domain,
)


_EPS = 1e-5


class _LogWeightMapFact:
    """
    Creates model that maps domains to a float
    which is sampled from a log uniform distribution.
    """

    def __init__(
        self,
        max_token: int,
        weight_range: tuple[float, float],
        device: str = "cpu",
        zero_value: float = torch.nan,
    ):
        l_min_w = math.log(min(weight_range))
        l_max_w = math.log(max(weight_range))
        weights = torch.tensor(
            [zero_value]
            + [math.exp(random.uniform(l_min_w, l_max_w)) for _ in range(max_token)]
        )
        self.weights = weights.to(device)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """t: long (c, p, d)"""
        return self.weights[t]


class _SignMapFact:
    """
    Creates a model that maps domains to 1.0 or -1.0
    with 50% probability of each being mapped.
    """

    def __init__(self, max_token: int, device: str = "cpu", zero_value: float = 0.0):
        choices = [1.0, -1.0]
        signs = torch.tensor([zero_value] + random.choices(choices, k=max_token))
        self.signs = signs.to(device)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """t: long (c, p, d)"""
        return self.signs[t]


class _VectorMapFact:
    """
    Create model that maps one-hot encoded domains
    to a list of vectors. Each vector will be mapped with
    the same frequency.
    """

    def __init__(
        self,
        max_token: int,
        vectors: list[list[float]],
        device: str = "cpu",
        zero_value: float = 0.0,
    ):
        n_vectors = len(vectors)
        vector_size = len(vectors[0])

        if not all(len(d) == vector_size for d in vectors):
            raise ValueError("Not all vectors have the same length")

        if n_vectors > max_token:
            raise ValueError(
                f"There are max_token={max_token} and {n_vectors} vectors."
                " It is not possible to map all vectors"
            )

        idxs = random.choices(list(range(n_vectors)), k=max_token)
        M = torch.full((max_token + 1, vector_size), fill_value=zero_value)
        for row_i, idx in enumerate(idxs):
            M[row_i + 1] = torch.tensor(vectors[idx])
        self.M = M.to(device)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """t: long (c, p, d)"""
        return self.M[t]


class _ReactionMapFact(_VectorMapFact):
    def __init__(
        self,
        molmap: dict[Molecule, int],
        reactions: list[tuple[list[Molecule], list[Molecule]]],
        max_token: int,
        device: str = "cpu",
        zero_value: float = 0.0,
    ):
        n_signals = 2 * len(molmap)
        n_reacts = len(reactions)

        # careful, only copy [0] to avoid having references to the same list
        vectors = [[0.0] * n_signals for _ in range(n_reacts + 1)]
        for ri, (lft, rgt) in enumerate(reactions):
            for mol in lft:
                mol_i = molmap[mol]
                vectors[ri + 1][mol_i] -= 1
            for mol in rgt:
                mol_i = molmap[mol]
                vectors[ri + 1][mol_i] += 1

        super().__init__(
            vectors=vectors, max_token=max_token, device=device, zero_value=zero_value
        )


class _TransporterMapFact(_VectorMapFact):
    def __init__(
        self,
        n_molecules: int,
        max_token: int,
        device: str = "cpu",
        zero_value: float = 0.0,
    ):
        n_signals = 2 * n_molecules

        # careful, only copy [0] to avoid having references to the same list
        vectors = [[0.0] * n_signals for _ in range(n_signals + 1)]
        for mi in range(n_molecules):
            vectors[mi + 1][mi] = 1
            vectors[mi + 1][mi + n_molecules] = -1
            vectors[mi + n_molecules + 1][mi] = -1
            vectors[mi + n_molecules + 1][mi + n_molecules] = 1

        super().__init__(
            vectors=vectors, max_token=max_token, device=device, zero_value=zero_value
        )


class _RegulatoryMapFact(_VectorMapFact):
    def __init__(
        self,
        n_molecules: int,
        max_token: int,
        device: str = "cpu",
        zero_value: float = 0.0,
    ):
        n_signals = 2 * n_molecules

        # careful, only copy [0] to avoid having references to the same list
        vectors = [[0.0] * n_signals for _ in range(n_signals + 1)]
        for mi in range(n_signals):
            vectors[mi + 1][mi] = 1

        super().__init__(
            vectors=vectors, max_token=max_token, device=device, zero_value=zero_value
        )


class Kinetics:
    """
    Class holding logic for simulating protein work.
    Usually this class is instantiated automatically when initializing [World][magicsoup.world.World].
    You can access it on `world.kinetics`.

    Arguments:
        molecules: List of molecule species.
            They have to be in the same order as they are on `chemistry.molecules`.
        abs_temp: Absolute temperature in Kelvin will influence the free Gibbs energy calculation of reactions.
            Higher temperature will give the reaction quotient term higher importance.
        device: Device to use for tensors
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

    The main method is [integrate_signals()][magicsoup.kinetics.Kinetics].
    When calling [enzymatic_activity()][magicsoup.world.World.enzymatic_activity], a matrix `X` of signals (c, s) is prepared
    and then [integrate_signals(X)][magicsoup.kinetics.Kinetics] is called.
    Updated signals are returned and [World][magicsoup.world.World] writes them back to `world.cell_molecules` and `world.molecule_map`.

    Another method, which ended up here, is [set_cell_params()][magicsoup.kinetics.Kinetics.set_cell_params]
    (and [unset_cell_params()][magicsoup.kinetics.Kinetics.unset_cell_params])
    which reads proteomes and updates cell parameters accordingly.
    This is called whenever the proteomes of some cells changed.
    Currently, this is also the main bottleneck in performance.
    """

    # TODO: I could use functions that map directly to C. E.g. replace torch.einsum
    #       with the correct matmul

    # TODO: I could also try to make the cell dimension variable. E.g. in nn.Models
    #       dim=0 is always kept variable

    def __init__(
        self,
        molecules: list[Molecule],
        reactions: list[tuple[list[Molecule], list[Molecule]]],
        abs_temp: float = 310.0,
        km_range: tuple[float, float] = (1e-5, 1.0),
        vmax_range: tuple[float, float] = (0.01, 10.0),
        device: str = "cpu",
        workers: int = 2,
    ):
        self.abs_temp = abs_temp
        self.device = device
        self.workers = workers

        self.mol_energies = self._tensor_from([d.energy for d in molecules] * 2)
        self.mol_2_mi = {d: i for i, d in enumerate(molecules)}
        self.mi_2_mol = {v: k for k, v in self.mol_2_mi.items()}

        # working cell params
        n_signals = 2 * len(molecules)
        self.Km = self._tensor(0, 0, n_signals)
        self.Vmax = self._tensor(0, 0)
        self.E = self._tensor(0, 0)
        self.N = self._tensor(0, 0, n_signals)
        self.A = self._tensor(0, 0, n_signals)

        # the domain specifications return 4 indexes
        # idx0 is a 2-codon idx for big mappings (4096)
        # idx1-3 are 1-codon mappings for floats (64)
        one_codon_size = len(ALL_CODONS)
        two_codon_size = one_codon_size**2

        # TODO: these maps need to be saved

        self.affinity_map = _LogWeightMapFact(
            max_token=one_codon_size,
            weight_range=km_range,
            device=device,
        )

        self.velocity_map = _LogWeightMapFact(
            max_token=one_codon_size,
            weight_range=vmax_range,
            device=device,
        )

        self.orient_map = _SignMapFact(max_token=one_codon_size, device=device)

        self.reaction_map = _ReactionMapFact(
            molmap=self.mol_2_mi,
            reactions=reactions,
            max_token=two_codon_size,
            device=device,
        )

        self.trnsp_mol_map = _TransporterMapFact(
            n_molecules=len(molecules), max_token=two_codon_size, device=device
        )

        self.reg_mol_map = _RegulatoryMapFact(
            n_molecules=len(molecules), max_token=two_codon_size, device=device
        )

    def unset_cell_params(self, cell_prots: list[tuple[int, int]]):
        """
        Set cell params for these proteins to 0.0.
        This is useful for cells that lost some of their proteins.

        Arguments:
            cell_prots: List of tuples of cell indexes and protein indexes
        """
        if len(cell_prots) == 0:
            return
        cells, prots = list(map(list, zip(*cell_prots)))
        self.E[cells, prots] = 0.0
        self.Vmax[cells, prots] = 0.0
        self.Km[cells, prots] = 0.0
        self.A[cells, prots] = 0.0
        self.N[cells, prots] = 0.0

    def get_proteome(
        self, proteome: list[list[tuple[int, int, int, int, int]]]
    ) -> list[Protein]:
        """
        Calculate cell parameters for a single proteome and return it as
        a list of proteins

        Arguments:
            proteome: proteome which should be calculated

        Retruns:
            A list of objects that represent proteins with domains and their
            specifications.
        """
        N_d, A_d, Km_d, Vmax_d, dom_types = self._get_proteome_tensors(
            proteomes=[proteome]
        )

        prots: list[Protein] = []
        for pi in range(dom_types.size(1)):
            doms: list[Domain] = []
            for di in range(dom_types.size(2)):

                if dom_types[0][pi][di].item() == 1:
                    lfts: list[Molecule] = []
                    rgts: list[Molecule] = []
                    for mi, n in enumerate(N_d[0][pi][di].tolist()):
                        if n >= 1:
                            rgts.extend(([self.mi_2_mol[mi]] * int(n)))
                        elif n <= -1:
                            lfts.extend(([self.mi_2_mol[mi]] * -int(n)))
                    doms.append(
                        CatalyticDomain(
                            reaction=(lfts, rgts),
                            affinity=Km_d[0][pi][di].amin().item(),
                            velocity=Vmax_d[0][pi][di].item(),
                        )
                    )

                if dom_types[0][pi][di].item() == 2:
                    mi = int(torch.argwhere(N_d[0][pi][di] != 0).min().item())
                    doms.append(
                        TransporterDomain(
                            molecule=self.mi_2_mol[mi],
                            affinity=Km_d[0][pi][di].amin().item(),
                            velocity=Vmax_d[0][pi][di].item(),
                            is_bkwd=bool((N_d[0][pi][di][mi] < 0).item()),
                        )
                    )

                if dom_types[0][pi][di].item() == 3:
                    mi = int(torch.argwhere(A_d[0][pi][di] != 0)[0].item())
                    if mi in self.mi_2_mol:
                        is_trnsm = False
                        mol = self.mi_2_mol[mi]
                    else:
                        is_trnsm = True
                        mol = self.mi_2_mol[mi - len(self.mi_2_mol)]
                    doms.append(
                        RegulatoryDomain(
                            effector=mol,
                            affinity=Km_d[0][pi][di].amin().item(),
                            is_inhibiting=bool((A_d[0][pi][di][mi] == -1).item()),
                            is_transmembrane=is_trnsm,
                        )
                    )

            prots.append(Protein(domains=doms, label=f"P{pi}"))

        return prots

    def set_cell_params(
        self,
        cell_idxs: list[int],
        proteomes: list[list[list[tuple[int, int, int, int, int]]]],
    ):
        """
        Calculate and set cell parameters for new proteomes

        Arguments:
            cell_idxs: Indexes of cells which proteomes belong to
            proteomes: list of proteomes which should be calculated and set
        """
        N_d, A_d, Km_d, Vmax_d, _ = self._get_proteome_tensors(proteomes=proteomes)

        # N (c, p, d, s)
        N = N_d.sum(dim=2)
        self.N[cell_idxs] = N

        # A (c, p, d, s)
        A = A_d.sum(dim=2)
        self.A[cell_idxs] = A

        # Km (c, p, d, s)
        Km = Km_d.nanmean(dim=2).nan_to_num(0.0)
        self.Km[cell_idxs] = Km

        # Vmax_d (c, p, d)
        Vmax = Vmax_d.nanmean(dim=2).nan_to_num(0.0)
        self.Vmax[cell_idxs] = Vmax

        # N (c, p, s)
        E = torch.einsum("cps,s->cp", N, self.mol_energies)
        self.E[cell_idxs] = E

    def integrate_signals(self, X: torch.Tensor) -> torch.Tensor:
        """
        Simulate protein work by integrating all signals.

        Arguments:
            X: Tensor of every signal in every cell (c, s). Must all be >= 0.0.

        Returns:
            New tensor of the same shape which represents the updated signals for every cell.

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

        Arguments:
            from_idxs: List of cell indexes to copy from
            to_idxs: List of cell indexes to copy to

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

        Arguments:
            keep: Bool tensor (c,) which is true for every cell that should not be removed
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

        Arguments:
            by_n: By how many rows to increase the cell dimension
        """
        # TODO: I could use a scaling method like for proteins
        #       but then I would have to keep cell idxs, and also remember
        #       rows which are currently free
        # TODO: I could also try to express all calculations with a free batch
        #       dimension. E.g. nn.Models are always implemented in a way that
        #       dim=0 can be variable
        self.Km = self._expand(t=self.Km, n=by_n, d=0)
        self.Vmax = self._expand(t=self.Vmax, n=by_n, d=0)
        self.E = self._expand(t=self.E, n=by_n, d=0)
        self.N = self._expand(t=self.N, n=by_n, d=0)
        self.A = self._expand(t=self.A, n=by_n, d=0)

    def increase_max_proteins(self, max_n: int):
        """
        Increase the protein dimension of all cell parameters

        Arguments:
            max_n: The maximum number of rows required in the protein dimension
        """
        # TODO: the number of upscales could be reduced by upscaling to e.g. the next
        #       higher 10s
        # TODO: I could also downscale proteins: E.g. if 10x in a row max_n was always
        #       more than 10 below the current n_prots, I can downscale.
        n_prots = int(self.Km.shape[1])
        if max_n > n_prots:
            by_n = max_n - n_prots
            self.Km = self._expand(t=self.Km, n=by_n, d=1)
            self.Vmax = self._expand(t=self.Vmax, n=by_n, d=1)
            self.E = self._expand(t=self.E, n=by_n, d=1)
            self.N = self._expand(t=self.N, n=by_n, d=1)
            self.A = self._expand(t=self.A, n=by_n, d=1)

    def _get_proteome_tensors(
        self, proteomes: list[list[list[tuple[int, int, int, int, int]]]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # separate domain specifications into different tensors
        dom_types, idxs0, idxs1, idxs2, idxs3 = self._collect_proteome_idxs(
            proteomes=proteomes
        )

        # identify domain types
        # 1=catalytic, 2=transporter, 3=regulatory
        catal_mask = (dom_types == 1).float()
        trnsp_mask = (dom_types == 2).float()
        reg_mask = (dom_types == 3).float()
        catal_trnsp_mask = ((dom_types == 1) | (dom_types == 2)).float()

        # map indices of domain specifications to concrete values
        # idx0 is a 2-codon index specific for every domain type (n=4096)
        # idx1-3 are 1-codon used for the floats (n=64)
        reacts = self.reaction_map(idxs0)  # float (c, p, d, s)
        trnsp_mols = self.trnsp_mol_map(idxs0)  # float (c, p, d, s)
        reg_mols = self.reg_mol_map(idxs0)  # float (c, p, d, s)

        velo = self.velocity_map(idxs1)  # float (c, p, d)
        aff = self.affinity_map(idxs2)  # float (c, p, d)
        orients = self.orient_map(idxs3)  # float (c, p, d)

        # N (c, p, d, s)
        N_r = torch.einsum("cpds,cpd->cpds", reacts, catal_mask)
        N_t = torch.einsum("cpds,cpd->cpds", trnsp_mols, trnsp_mask)
        N_d = torch.einsum("cpds,cpd->cpds", (N_r + N_t), orients)

        # A (c, p, d, s)
        A_r = torch.einsum("cpds,cpd->cpds", reg_mols, reg_mask)
        A_d = torch.einsum("cpds,cpd->cpds", A_r, orients)

        # Km (c, p, d, s)
        lft_mols = ((A_d > 0.0) | (N_d > 0.0)).float()
        rgt_mols = (N_d < 0.0).float()
        Km_l = torch.einsum("cpds,cpd->cpds", lft_mols, aff)
        Km_r = torch.einsum("cpds,cpd->cpds", rgt_mols, 1 / aff)
        Km_d = Km_l + Km_r

        # Vmax_d (c, p, d)
        Vmax_d = velo * catal_trnsp_mask

        return N_d, A_d, Km_d, Vmax_d, dom_types

    def _collect_proteome_idxs(
        self, proteomes: list[list[list[tuple[int, int, int, int, int]]]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,]:
        n_prots = self.N.size(1)
        n_doms = max(len(dd) for d in proteomes for dd in d)
        empty_seq = [0] * n_doms

        c_dts = []
        c_idxs0 = []
        c_idxs1 = []
        c_idxs2 = []
        c_idxs3 = []
        for proteins in proteomes:
            p_dts = []
            p_idxs0 = []
            p_idxs1 = []
            p_idxs2 = []
            p_idxs3 = []
            for doms in proteins:
                d_dts = []
                d_idxs0 = []
                d_idxs1 = []
                d_idxs2 = []
                d_idxs3 = []
                for dt, i0, i1, i2, i3 in doms:
                    d_dts.append(dt)
                    d_idxs0.append(i0)
                    d_idxs1.append(i1)
                    d_idxs2.append(i2)
                    d_idxs3.append(i3)
                d_pad = n_doms - len(d_idxs0)
                p_dts.append(d_dts + [0] * d_pad)
                p_idxs0.append(d_idxs0 + [0] * d_pad)
                p_idxs1.append(d_idxs1 + [0] * d_pad)
                p_idxs2.append(d_idxs2 + [0] * d_pad)
                p_idxs3.append(d_idxs3 + [0] * d_pad)

            p_pad = n_prots - len(p_idxs0)
            c_dts.append(p_dts + [empty_seq] * p_pad)
            c_idxs0.append(p_idxs0 + [empty_seq] * p_pad)
            c_idxs1.append(p_idxs1 + [empty_seq] * p_pad)
            c_idxs2.append(p_idxs2 + [empty_seq] * p_pad)
            c_idxs3.append(p_idxs3 + [empty_seq] * p_pad)

        dom_types = self._tensor_from(c_dts)  # long (c, p, d)
        idxs0 = self._tensor_from(c_idxs0)  # long (c, p, d)
        idxs1 = self._tensor_from(c_idxs1)  # long (c, p, d)
        idxs2 = self._tensor_from(c_idxs2)  # long (c, p, d)
        idxs3 = self._tensor_from(c_idxs3)  # long (c, p, d)
        return dom_types, idxs0, idxs1, idxs2, idxs3

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

    def _expand(self, t: torch.Tensor, n: int, d: int) -> torch.Tensor:
        pre = t.shape[slice(d)]
        post = t.shape[slice(d + 1, t.dim())]
        zeros = self._tensor(*pre, n, *post)
        return torch.cat([t, zeros], dim=d)

    def _tensor(self, *args, d: Optional[float] = None) -> torch.Tensor:
        if d is None:
            return torch.zeros(*args).to(self.device)
        return torch.full(tuple(args), d).to(self.device)

    def _tensor_from(self, d: Any) -> torch.Tensor:
        return torch.tensor(d).to(self.device)
