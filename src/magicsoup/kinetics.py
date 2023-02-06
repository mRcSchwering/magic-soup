from typing import Optional
from itertools import product
import math
import random
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from magicsoup.constants import GAS_CONSTANT, CODON_TABLE, CODON_SIZE
from magicsoup.containers import Protein, Molecule

_EPS = 1e-5


def _dom_to_tokens(seq: str) -> list[int]:
    s = CODON_SIZE
    n = len(seq)
    ijs = [(i, i + s) for i in range(0, n + 1 - s, s)]
    return [CODON_TABLE[seq[i:j]] for i, j in ijs]


def _get_one_hot_conversion_matrix(
    oh_size: int, len_size: int, enc_idxs: list[int]
) -> torch.Tensor:
    enc_size = len(enc_idxs)
    all_oh_encs = list(range(oh_size))
    out_oh_encs = list(product(*[all_oh_encs] * enc_size))
    out_size = len(out_oh_encs)

    w = torch.zeros(out_size, len_size, oh_size)
    for i, combi in enumerate(out_oh_encs):
        for j, oh in zip(enc_idxs, combi):
            w[i, j, oh] = 1.0

    return w / enc_size


class _OneHot:
    """
    Creates a model for one-hot encoding tokens
    and vice versa.
    """

    def __init__(self, enc_size: int, device: str = "cpu"):
        self.w = torch.cat(
            [torch.zeros(1, enc_size), torch.diag(torch.ones(enc_size))]
        ).to(device)

    def encode(self, t: torch.Tensor) -> torch.Tensor:
        """size: (..., )"""
        return self.w[t]

    def decode(self, t: torch.Tensor) -> torch.Tensor:
        """size: (..., one-hot) must be last dim"""
        return t.argmax(dim=-1) + t.sum(dim=-1)


class _LogWeightMapFact:
    """
    Creates model that maps domains to a float
    which is sampled from a log uniform distribution.
    """

    def __init__(
        self,
        enc_size: int,
        dom_size: int,
        enc_idxs: list[int],
        weight_range: tuple[float, float],
        device: str = "cpu",
    ):
        l_min_w = math.log(min(weight_range))
        l_max_w = math.log(max(weight_range))
        self.w0 = _get_one_hot_conversion_matrix(
            oh_size=enc_size, len_size=dom_size, enc_idxs=enc_idxs
        ).to(device)
        out_size = self.w0.size(0)
        weights = [math.exp(random.uniform(l_min_w, l_max_w)) for _ in range(out_size)]
        self.w1 = torch.tensor(weights).to(device)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """size: (..., domain, one-hot)"""
        enc = (torch.einsum("odi,...di->...o", self.w0, t) == 1.0).float()
        out = enc @ self.w1
        return out


class _SignMapFact:
    """
    Creates a model that maps domains to 1.0 or -1.0
    with 50% probability of each being mapped.
    """

    def __init__(
        self, enc_size: int, dom_size: int, enc_idxs: list[int], device: str = "cpu"
    ):
        choices = [1.0, -1.0]
        self.w0 = _get_one_hot_conversion_matrix(
            oh_size=enc_size, len_size=dom_size, enc_idxs=enc_idxs
        ).to(device)
        out_size = self.w0.size(0)
        self.w1 = torch.tensor(random.choices(choices, k=out_size)).to(device)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """size: (..., domain, one-hot)"""
        enc = (torch.einsum("odi,...di->...o", self.w0, t) == 1.0).float()
        out = enc @ self.w1
        return out


class _VectorMapFact:
    """
    Create model that maps one-hot encoded domains
    to a list of vectors. Each vector will be mapped with
    the same frequency.
    """

    def __init__(
        self,
        enc_size: int,
        dom_size: int,
        enc_idxs: list[int],
        vectors: list[list[float]],
        device: str = "cpu",
    ):
        n_vectors = len(vectors)
        vector_size = len(vectors[0])
        if not all(len(d) == vector_size for d in vectors):
            raise ValueError("Not all vectors have the same length")

        self.w0 = _get_one_hot_conversion_matrix(
            oh_size=enc_size, len_size=dom_size, enc_idxs=enc_idxs
        ).to(device)
        out_size = self.w0.size(0)
        idxs = random.choices(list(range(n_vectors)), k=out_size)
        w1 = torch.zeros(out_size, vector_size)
        for row_i, idx in enumerate(idxs):
            w1[row_i] = torch.tensor(vectors[idx])
        self.w1 = w1.to(device)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """size: (..., domain, one-hot)"""
        enc = (torch.einsum("odi,...di->...o", self.w0, t) == 1.0).float()
        out = torch.einsum("...o,ov->...v", enc, self.w1)
        return out


def _convert_dom_seqs(
    proteins: list[list[tuple[tuple[bool, bool, bool], str]]],
    n_prots: int,
    n_doms: int,
    n_dom_codons: int,
) -> tuple[list[list[bool]], list[list[bool]], list[list[bool]], list[list[list[int]]]]:
    empty_prot_bool = [False] * n_doms
    empty_dom_seq = [0] * n_dom_codons
    empty_prot_seq = [empty_dom_seq] * n_doms

    p_catals = []
    p_trnsps = []
    p_regs = []
    p_tokens = []
    for doms in proteins:
        d_catals = []
        d_trnsps = []
        d_regs = []
        d_tokens = []
        for (catal, trnsp, reg), seq in doms:
            d_catals.append(catal)
            d_trnsps.append(trnsp)
            d_regs.append(reg)
            d_tokens.append(_dom_to_tokens(seq))
        d_pad = n_doms - len(d_tokens)
        p_catals.append(d_catals + [False] * d_pad)
        p_trnsps.append(d_trnsps + [False] * d_pad)
        p_regs.append(d_regs + [False] * d_pad)
        p_tokens.append(d_tokens + [empty_dom_seq] * d_pad)
    p_pad = n_prots - len(p_tokens)

    catals = p_catals + [empty_prot_bool] * p_pad
    trnsps = p_trnsps + [empty_prot_bool] * p_pad
    regs = p_regs + [empty_prot_bool] * p_pad
    dom_specs = p_tokens + [empty_prot_seq] * p_pad
    return catals, trnsps, regs, dom_specs


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

    # TODO: I could use some native torch functions in some places, e.g. ReLU
    #       might be faster than matrix multiplications

    # TODO: are the molecule maps faster if molecule2idx (instead of molecule name)?

    def __init__(
        self,
        molecules: list[Molecule],
        reactions: list[tuple[list[Molecule], list[Molecule]]],
        n_dom_codons: int,
        abs_temp: float = 310.0,
        km_range: tuple[float, float] = (1e-5, 1.0),
        vmax_range: tuple[float, float] = (0.01, 10.0),
        device: str = "cpu",
        workers: int = 2,
    ):
        n = len(molecules)
        self.n_molecules = n
        self.n_signals = 2 * n
        self.int_mol_map = {d.name: i for i, d in enumerate(molecules)}
        self.ext_mol_map = {d.name: i + n for i, d in enumerate(molecules)}

        self.mol_energies = torch.tensor([d.energy for d in molecules] * 2)
        self.n_dom_codons = n_dom_codons

        oh_enc_size = max(CODON_TABLE.values())
        self.one_hot = _OneHot(enc_size=oh_enc_size, device=device)

        # Catalytic: Km (1), Vmax (1), orientation (1), reaction vectors (2)
        # Transporter: Km(1), Vmax (1), orientation (1), molecule vectors for transporters (2)
        # Regulatory: Km(1), effect (1), molecule vectors for regulatory (2)

        self.affinity_map = _LogWeightMapFact(
            enc_idxs=[0],
            weight_range=km_range,
            enc_size=oh_enc_size,
            dom_size=n_dom_codons,
            device=device,
        )

        self.velocity_map = _LogWeightMapFact(
            enc_idxs=[1],
            weight_range=vmax_range,
            enc_size=oh_enc_size,
            dom_size=n_dom_codons,
            device=device,
        )

        self.orient_map = _SignMapFact(
            enc_idxs=[2], enc_size=oh_enc_size, dom_size=n_dom_codons, device=device
        )

        # careful, only copy [0] to avoid having references to the same list
        react_vectors = [[0.0] * self.n_signals for _ in range(len(reactions))]
        for ri, (lft, rgt) in enumerate(reactions):
            for mol in lft:
                mol_i = self.int_mol_map[mol.name]
                react_vectors[ri][mol_i] -= 1
            for mol in rgt:
                mol_i = self.int_mol_map[mol.name]
                react_vectors[ri][mol_i] += 1

        self.reaction_map = _VectorMapFact(
            enc_idxs=[3, 4],
            vectors=react_vectors,
            enc_size=oh_enc_size,
            dom_size=n_dom_codons,
            device=device,
        )

        trnsp_mol_vectors = [[0.0] * self.n_signals for _ in range(self.n_signals)]
        for mi in range(self.n_molecules):
            trnsp_mol_vectors[mi][mi] = 1
            trnsp_mol_vectors[mi][mi + self.n_molecules] = -1
            trnsp_mol_vectors[mi + self.n_molecules][mi] = -1
            trnsp_mol_vectors[mi + self.n_molecules][mi + self.n_molecules] = 1

        self.trnsp_mol_map = _VectorMapFact(
            enc_idxs=[3, 4],
            vectors=trnsp_mol_vectors,
            enc_size=oh_enc_size,
            dom_size=n_dom_codons,
            device=device,
        )

        reg_mol_vectors = [[0.0] * self.n_signals for _ in range(self.n_signals)]
        for mi in range(self.n_signals):
            reg_mol_vectors[mi][mi] = 1

        self.reg_mol_map = _VectorMapFact(
            enc_idxs=[3, 4],
            vectors=reg_mol_vectors,
            enc_size=oh_enc_size,
            dom_size=n_dom_codons,
            device=device,
        )

        self.abs_temp = abs_temp
        self.device = device
        self.workers = workers

        # working cell params
        self.Km = self._tensor(0, 0, self.n_signals)
        self.Vmax = self._tensor(0, 0)
        self.E = self._tensor(0, 0)
        self.N = self._tensor(0, 0, self.n_signals)
        self.A = self._tensor(0, 0, self.n_signals)

        # containers for translation
        self.N_d = self._tensor(0, 0, 0, self.n_signals).share_memory_()
        self.A_d = self._tensor(0, 0, 0, self.n_signals).share_memory_()
        self.Km_d = self._tensor(0, 0, 0, self.n_signals, d=torch.nan).share_memory_()
        self.Vmax_d = self._tensor(0, 0, 0, d=torch.nan).share_memory_()

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

    def set_cell_params(
        self,
        cell_idxs: list[int],
        dom_seqs_lst: list[list[list[tuple[tuple[bool, bool, bool], str]]]],
    ):
        print("starting cell params")
        n_prots = self.N.size(1)
        n_doms = max(len(dd) for d in dom_seqs_lst for dd in d)

        args = []
        for proteins in dom_seqs_lst:
            args.append((proteins, n_prots, n_doms, self.n_dom_codons))

        # TODO: pool necessary?
        # TODO: fill-up pre-built tensors?
        print("starting cell pool")
        with mp.Pool(self.workers) as pool:
            res = pool.starmap(_convert_dom_seqs, args)

        print("creating tensors")
        c_catals = []
        c_trnsps = []
        c_regs = []
        c_doms = []
        for catals, trnsps, regs, doms in res:
            c_catals.append(catals)
            c_trnsps.append(trnsps)
            c_regs.append(regs)
            c_doms.append(doms)

        # c cells, p proteins, d domains, n domain len, h one-hot enc len
        catal_mask = torch.tensor(c_catals)  # bool (c, p, d)
        trnsp_mask = torch.tensor(c_trnsps)  # bool (c, p, d)
        reg_mask = torch.tensor(c_regs)  # bool (c, p, d)
        dom_specs = self.one_hot.encode(torch.tensor(c_doms))  # float (c, p, d, n, h)

        print("mapping")
        velo = self.velocity_map(dom_specs)  # float (c, p, d)
        aff = self.affinity_map(dom_specs)  # float (c, p, d)
        orients = self.orient_map(dom_specs)  # float (c, p, d)
        reacts = self.reaction_map(dom_specs)  # float (c, p, d, s)
        trnsp_mols = self.trnsp_mol_map(dom_specs)  # float (c, p, d, s)
        reg_mols = self.reg_mol_map(dom_specs)  # float (c, p, d, s)

        print("calculating params")

        # N (c, p, d, s)
        N_r = torch.einsum("cpds,cpd->cpds", reacts, catal_mask.float())
        N_t = torch.einsum("cpds,cpd->cpds", trnsp_mols, trnsp_mask.float())
        N_d = torch.einsum("cpds,cpd->cpds", (N_r + N_t), orients)
        N = N_d.sum(dim=2)
        self.N[cell_idxs] = N

        # A (c, p, d, s)
        A_r = torch.einsum("cpds,cpd->cpds", reg_mols, reg_mask.float())
        A_d = torch.einsum("cpds,cpd->cpds", A_r, orients)
        A = A_d.sum(dim=2)
        self.A[cell_idxs] = A

        # Km (c, p, d, s)
        lft_mols = ((A_d > 0.0) | (N_d > 0.0)).float()
        rgt_mols = (N_d < 0.0).float()
        Km_l = torch.einsum("cpds,cpd->cpds", lft_mols, aff)
        Km_r = torch.einsum("cpds,cpd->cpds", rgt_mols, 1 / aff)
        Km_d = Km_l + Km_r
        Km_d[Km_d == 0.0] = torch.nan
        Km = Km_d.nanmean(dim=2).nan_to_num(0.0)
        self.Km[cell_idxs] = Km

        # Vmax_d (c, p, d)
        Vmax_d = velo * (catal_mask | trnsp_mask).float()
        Vmax_d[Vmax_d == 0.0] = torch.nan
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
        self.Km = self._expand(t=self.Km, n=by_n, d=0)
        self.Vmax = self._expand(t=self.Vmax, n=by_n, d=0)
        self.E = self._expand(t=self.E, n=by_n, d=0)
        self.N = self._expand(t=self.N, n=by_n, d=0)
        self.A = self._expand(t=self.A, n=by_n, d=0)

    def increase_max_cells_d(self, max_n: int):
        n_cells = int(self.N.size(0))
        if max_n > n_cells:
            by_n = max_n - n_cells
            self.Km_d = self._expand(t=self.Km_d, n=by_n, d=0)
            self.Vmax_d = self._expand(t=self.Vmax_d, n=by_n, d=0)
            self.N_d = self._expand(t=self.N_d, n=by_n, d=0)
            self.A_d = self._expand(t=self.A_d, n=by_n, d=0)

    def increase_max_proteins(self, max_n: int):
        """
        Increase the protein dimension of all cell parameters

        Arguments:
            max_n: The maximum number of rows required in the protein dimension
        """
        n_prots = int(self.Km.shape[1])
        if max_n > n_prots:
            by_n = max_n - n_prots
            self.Km = self._expand(t=self.Km, n=by_n, d=1)
            self.Vmax = self._expand(t=self.Vmax, n=by_n, d=1)
            self.E = self._expand(t=self.E, n=by_n, d=1)
            self.N = self._expand(t=self.N, n=by_n, d=1)
            self.A = self._expand(t=self.A, n=by_n, d=1)

    def increase_max_proteins_d(self, max_n: int):
        n_prots = int(self.N_d.size(1))
        if max_n > n_prots:
            by_n = max_n - n_prots
            self.Km_d = self._expand(t=self.Km_d, n=by_n, d=1)
            self.Vmax_d = self._expand(t=self.Vmax_d, n=by_n, d=1)
            self.N_d = self._expand(t=self.N_d, n=by_n, d=1)
            self.A_d = self._expand(t=self.A_d, n=by_n, d=1)

    def increase_max_domains_d(self, max_n: int):
        n_doms = int(self.N_d.size(2))
        if max_n > n_doms:
            by_n = max_n - n_doms
            self.Km_d = self._expand(t=self.Km_d, n=by_n, d=2)
            self.Vmax_d = self._expand(t=self.Vmax_d, n=by_n, d=2)
            self.N_d = self._expand(t=self.N_d, n=by_n, d=2)
            self.A_d = self._expand(t=self.A_d, n=by_n, d=2)

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

    def _tensor(self, *args, d: Optional[float] = None) -> torch.Tensor:
        if d is None:
            return torch.zeros(*args).to(self.device)
        return torch.full(tuple(args), d).to(self.device)
