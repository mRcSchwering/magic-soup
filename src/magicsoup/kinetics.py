from typing import Any, Optional
from itertools import product
import math
import random
import torch
import torch.multiprocessing as mp
from magicsoup.constants import GAS_CONSTANT, CODON_TABLE, CODON_SIZE
from magicsoup.containers import Protein, Molecule

_EPS = 1e-5

# TODO: conversion matrix for each domain type
# domain seq is one-hot encoded
# 1 multiplication maps to different one hot encoding
#
# reactions: vector (s,) with signed Integers
# orientation: +1 or -1 will be multiplied with reactions vector
# affinity: float, will be multiplied vector derived from reaction vector
#           that is 1.0 for all places where reaction != 0.0
#
# brauche 3 Masken, eine für jeden Domain Type
# dann führe ich Multiplikation for jeden Domain Type mit der entsprechenden
# conversion Matrix aux
#
# genetics translation könnte zu jeder domain sequenz zusätzlich tuple[bool, bool, bool]
# zurück geben und vielleicht zusätzlich auch schon die strs in tokens übersetzt haben
#
# aus den list[list[list[tuple[bool, bool, bool]]]] kann man sich dann 3 masken bauen
# jede dieser Masken wird mit list[list[list[list[int]]]] multipliziert (bzw mit der one-hot encodetedn version davon)
# und dann mit den entsprechenden conversion matrizen
#
# dann werden die resultate aggregiert (addieren für N, average oder so für Vmax, Km)
# das wird dann auf die leeren Stellen der self.N, ... gesetzt
#
# Ich brauche diese Conversion Matrizen:
# 1. one-hot -> 1.0 or -1.0 (1:1 prob)
# 2. one-hot -> float (log-distributed)
# 3. one-hot -> one-hot (categorical)


def dom_seq_to_tokens(seq: str) -> torch.Tensor:
    s = CODON_SIZE
    n = len(seq)
    ijs = [(i, i + s) for i in range(0, n + 1 - s, s)]
    tokens = [CODON_TABLE[seq[i:j]] for i, j in ijs]
    return torch.tensor(tokens)


def get_one_hot_conversion_matrix(
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


class OneHot:
    """
    Creates a model for one-hot encoding tokens
    and vice versa.
    """

    def __init__(self, enc_size: int):
        self.w = torch.cat([torch.zeros(1, enc_size), torch.diag(torch.ones(enc_size))])

    def encode(self, t: torch.Tensor) -> torch.Tensor:
        """size: (..., )"""
        return self.w[t]

    def decode(self, t: torch.Tensor) -> torch.Tensor:
        """size: (..., one-hot) must be last dim"""
        return t.argmax(dim=-1) + t.sum(dim=-1)


class DomainToLogWeight:
    """
    Creates model that maps domains to a float
    which is sampled from a log uniform distribution.
    """

    def __init__(
        self,
        enc_size: int,
        dom_size: int,
        enc_idxs: list[int],
        min_w: float,
        max_w: float,
    ):
        l_min_w = math.log(min_w)
        l_max_w = math.log(max_w)
        self.w0 = get_one_hot_conversion_matrix(
            oh_size=enc_size, len_size=dom_size, enc_idxs=enc_idxs
        )
        out_size = self.w0.size(0)
        weights = [math.exp(random.uniform(l_min_w, l_max_w)) for _ in range(out_size)]
        self.w1 = torch.tensor(weights)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """size: (..., domain, one-hot)"""
        enc = (torch.einsum("odi,...di->...o", self.w0, t) == 1.0).float()
        out = enc @ self.w1
        return out


class DomainToSign:
    """
    Creates a model that maps domains to 1.0 or -1.0
    with 50% probability of each being mapped.
    """

    def __init__(
        self,
        enc_size: int,
        dom_size: int,
        enc_idxs: list[int],
    ):
        choices = [1.0, -1.0]
        self.w0 = get_one_hot_conversion_matrix(
            oh_size=enc_size, len_size=dom_size, enc_idxs=enc_idxs
        )
        out_size = self.w0.size(0)
        self.w1 = torch.tensor(random.choices(choices, k=out_size))

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """size: (..., domain, one-hot)"""
        enc = (torch.einsum("odi,...di->...o", self.w0, t) == 1.0).float()
        out = enc @ self.w1
        return out


class DomainToVector:
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
    ):
        vector_size = len(vectors[0])
        assert all(len(d) == vector_size for d in vectors)

        self.w0 = get_one_hot_conversion_matrix(
            oh_size=enc_size, len_size=dom_size, enc_idxs=enc_idxs
        )
        out_size = self.w0.size(0)
        idxs = random.choices(list(range(len(vectors))), k=out_size)
        self.w1 = torch.zeros(out_size, vector_size)
        for row_i, idx in enumerate(idxs):
            self.w1[row_i] = torch.tensor(vectors[idx])

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """size: (..., domain, one-hot)"""
        enc = (torch.einsum("odi,...di->...o", self.w0, t) == 1.0).float()
        out = torch.einsum("...o,ov->...v", enc, self.w1)
        return out


vectors = [
    [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
    [0.0, 2.0, 0.0, 0.0, 0.0, -2.0],
    [0.0, 1.0, 1.0, 0.0, -1.0, 0.0],
    [0.0, -2.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
]

model = DomainToVector(
    enc_size=len(CODON_TABLE), dom_size=5, enc_idxs=[0, 1], vectors=vectors
)


one_hot = OneHot(enc_size=len(CODON_TABLE))

model(one_hot.encode(dom_seq_to_tokens("AAAAGATACAAGAAA")))

p0 = torch.stack(
    [
        dom_seq_to_tokens("AATGATTACAAGAAA"),
        dom_seq_to_tokens("TATGATTACAAGAAA"),
        dom_seq_to_tokens("ATTGATTACAAGAAA"),
        dom_seq_to_tokens("AGGGATTACAAGAAT"),
    ]
)
p1 = torch.stack(
    [
        dom_seq_to_tokens("AATGATTACAAGAAA"),
        dom_seq_to_tokens("TATGATTACATTAAG"),
    ]
)
proteins = [p0, p1]
max_doms = max(d.size(0) for d in proteins)
cell = torch.stack(
    [
        torch.nn.functional.pad(
            d, pad=(0, 0, 0, max_doms - d.size(0)), mode="constant", value=0
        )
        for d in proteins
    ]
)

d = one_hot.encode(cell)
d.size()
one_hot.decode(d)
r = model(d)
r.sum(dim=1)


# TODO: tryout
def set_domain_params(
    cell_i: int,
    prot_i: int,
    dom_i: int,
    seq: str,
    domain_map: dict[str, tuple[bool, bool, bool]],
    molecule_map: dict[str, int],
    reaction_map: dict[str, tuple[list[int], list[int]]],
    affinity_map: dict[str, float],
    velocity_map: dict[str, float],
    bool_map: dict[str, bool],
    n_nts: dict[str, int],
    n_molecules: int,
    N_d: torch.Tensor,
    A_d: torch.Tensor,
    Km_d: torch.Tensor,
    Vmax_d: torch.Tensor,
):
    n_dom_type_nts = n_nts["n_dom_type_nts"]
    n_reaction_nts = n_nts["n_reaction_nts"]
    n_molecule_nts = n_nts["n_molecule_nts"]
    n_affinity_nts = n_nts["n_affinity_nts"]
    n_velocity_nts = n_nts["n_velocity_nts"]
    n_bool_nts = n_nts["n_bool_nts"]

    i = n_dom_type_nts
    is_catal, is_trnsp, is_reg = domain_map[seq[:n_dom_type_nts]]

    if is_catal:
        lft, rgt = reaction_map[seq[i : i + n_reaction_nts]]
        i += n_reaction_nts
        aff = affinity_map[seq[i : i + n_affinity_nts]]
        i += n_affinity_nts
        velo = velocity_map[seq[i : i + n_velocity_nts]]
        i += n_velocity_nts
        orient = bool_map[seq[i : i + n_bool_nts]]

        Vmax_d[cell_i, prot_i, dom_i] = velo

        if orient:
            prods = lft
            subs = rgt
        else:
            subs = lft
            prods = rgt

        for mol_i in subs:
            N_d[cell_i, prot_i, dom_i, mol_i] = -1.0
            Km_d[cell_i, prot_i, dom_i, mol_i] = aff

        for mol_i in prods:
            N_d[cell_i, prot_i, dom_i, mol_i] = 1.0
            Km_d[cell_i, prot_i, dom_i, mol_i] = 1 / aff

    if is_trnsp:
        mol = molecule_map[seq[i : i + n_molecule_nts]]
        i += n_molecule_nts
        aff = affinity_map[seq[i : i + n_affinity_nts]]
        i += n_affinity_nts
        velo = velocity_map[seq[i : i + n_velocity_nts]]
        i += n_velocity_nts
        orient = bool_map[seq[i : i + n_bool_nts]]

        Vmax_d[cell_i, prot_i, dom_i] = velo

        if orient:
            sub_i = mol + n_molecules
            prod_i = mol
        else:
            sub_i = mol
            prod_i = mol + n_molecules

        N_d[cell_i, prot_i, dom_i, sub_i] = -1.0
        N_d[cell_i, prot_i, dom_i, prod_i] = 1.0
        Km_d[cell_i, prot_i, dom_i, sub_i] = aff
        Km_d[cell_i, prot_i, dom_i, prod_i] = 1 / aff

    if is_reg:
        mol = molecule_map[seq[i : i + n_molecule_nts]]
        i += n_molecule_nts
        aff = affinity_map[seq[i : i + n_affinity_nts]]
        i += n_affinity_nts
        transm = bool_map[seq[i : i + n_bool_nts]]
        i += n_bool_nts
        inh = bool_map[seq[i : i + n_bool_nts]]

        mol_i = mol + n_molecules if transm else mol
        A_d[cell_i, prot_i, dom_i, mol_i] = -1.0 if inh else 1.0
        Km_d[cell_i, prot_i, dom_i, mol_i] = aff


# TODO: tryout
def translate_dom_seqs(
    cell_i: int,
    cdss: list[str],
    maps: dict,
    n_nts: dict[str, int],
    n_molecules: int,
    N_d: torch.Tensor,
    A_d: torch.Tensor,
    Km_d: torch.Tensor,
    Vmax_d: torch.Tensor,
):
    domain_map = maps["domain_map"]
    molecule_map = maps["molecule_map"]
    reaction_map = maps["reaction_map"]
    affinity_map = maps["affinity_map"]
    velocity_map = maps["velocity_map"]
    bool_map = maps["bool_map"]

    kwargs = {
        "cell_i": cell_i,
        "domain_map": domain_map,
        "molecule_map": molecule_map,
        "reaction_map": reaction_map,
        "affinity_map": affinity_map,
        "velocity_map": velocity_map,
        "bool_map": bool_map,
        "n_nts": n_nts,
        "n_molecules": n_molecules,
        "N_d": N_d,
        "A_d": A_d,
        "Km_d": Km_d,
        "Vmax_d": Vmax_d,
    }

    for prot_i, dom_seqs in enumerate(cdss):
        for dom_i, seq in enumerate(dom_seqs):
            set_domain_params(prot_i=prot_i, dom_i=dom_i, seq=seq, **kwargs)  # type: ignore


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
        abs_temp: float = 310.0,
        device: str = "cpu",
        workers: int = 2,
    ):
        n = len(molecules)
        self.n_molecules = n
        self.n_signals = 2 * n
        self.int_mol_map = {d.name: i for i, d in enumerate(molecules)}
        self.ext_mol_map = {d.name: i + n for i, d in enumerate(molecules)}

        # TODO: tryout
        self.ext_offset = n
        self.mol_energies = torch.tensor([d.energy for d in molecules] * 2)

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

    # TODO: tryout
    def get_all_proteomes(
        self,
        dom_seqs_lst: list[list[list[str]]],
        mappings: dict[str, Any],
        region_lens: dict[str, int],
    ) -> list[int]:
        args = []
        for ci, cdss in enumerate(dom_seqs_lst):
            args.append(
                (
                    ci,
                    cdss,
                    mappings,
                    region_lens,
                    self.n_molecules,
                    self.N_d,
                    self.A_d,
                    self.Km_d,
                    self.Vmax_d,
                )
            )

        with mp.Pool(self.workers) as pool:
            pool.starmap(translate_dom_seqs, args)

        # N (c, p, d, s)
        # all 0s N ^= no domain or regulatory domains only
        keep = (self.N_d != 0).any(dim=3).any(dim=2).any(dim=1)
        keep_idxs = keep.argwhere().flatten().tolist()

        return keep_idxs

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

    def set_cell_params(self, cell_prots: list[tuple[int, int, Protein]]):
        """
        Set cell params for these proteins accordingly
        You can compare proteins within a cell and only update the ones that changed.
        The comparison (`protein0 == protein1`) will note a difference in any of the proteins attributes.

        Arguments:
            cell_prots: List of tuples of cell indexes, protein indexes, and the protein itself

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
        self.E[cis, pis] = torch.tensor(E).to(self.device)
        self.Km[cis, pis] = torch.tensor(Km).to(self.device)
        self.Vmax[cis, pis] = torch.tensor(Vmax).to(self.device)
        self.A[cis, pis] = torch.tensor(A).to(self.device)
        self.N[cis, pis] = torch.tensor(N).to(self.device)

    # TODO: tryout
    def set_cell_params_new(self, cell_idxs: list[int], container_idxs: list[int]):
        # N (c, p, d, s)
        N = self.N_d[container_idxs].sum(dim=2)
        self.N[cell_idxs] = N

        # A (c, p, d, s)
        A = self.A_d[container_idxs].sum(dim=2)
        self.A[cell_idxs] = A

        # Km (c, p, d, s)
        Km = self.Km_d[container_idxs].nanmean(dim=2).nan_to_num(0.0)
        self.Km[cell_idxs] = Km

        # Vmax_d (c, p, d)
        Vmax = self.Vmax_d[container_idxs].nanmean(dim=2).nan_to_num(0.0)
        self.Vmax[cell_idxs] = Vmax

        # TODO: check if thats correct, should multiply energies with N
        #       for each signal, then take sum over these signal-energies
        # N (c, p, s)
        self.E[cell_idxs] = torch.einsum("cps,s->cp", N, self.mol_energies)

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
