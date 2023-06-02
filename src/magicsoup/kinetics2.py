from typing import Any
import math
import random
import torch
from magicsoup.constants import GAS_CONSTANT
from magicsoup.containers import (
    Molecule,
    Protein,
    CatalyticDomain,
    TransporterDomain,
    RegulatoryDomain,
    DomainType,
)

_EPS = 1e-8


class _LogNormWeightMapFact:
    """
    Creates an object that maps tokens to a float
    which is sampled from a log normal distribution.
    """

    def __init__(
        self,
        max_token: int,
        weight_range: tuple[float, float],
        device: str = "cpu",
        zero_value: float = torch.nan,
    ):
        min_w = min(weight_range)
        max_w = max(weight_range)
        l_min_w = math.log(min_w)
        l_max_w = math.log(max_w)
        mu = (l_min_w + l_max_w) / 2
        sig = l_max_w - l_min_w
        non_zero_weights: list[float] = []
        for _ in range(max_token):
            sample = math.exp(random.gauss(mu, sig))
            while not min_w <= sample <= max_w:
                sample = math.exp(random.gauss(mu, sig))
            non_zero_weights.append(sample)
        weights = torch.tensor([zero_value] + non_zero_weights)
        self.weights = weights.to(device)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """t: long (c, p, d)"""
        return self.weights[t]


class _SignMapFact:
    """
    Creates an object that maps tokens to 1.0 or -1.0
    with 50% probability of each being mapped.
    """

    def __init__(self, max_token: int, device: str = "cpu", zero_value: float = 0.0):
        choices = [1.0, -1.0]
        signs = torch.tensor([zero_value] + random.choices(choices, k=max_token))
        self.signs = signs.to(device)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """t: long (c, p, d)"""
        return self.signs[t]

    def inverse(self) -> dict[bool, list[int]]:
        sign_map = {}
        M = self.signs
        sign_map[True] = torch.argwhere(M == 1.0).flatten().tolist()
        sign_map[False] = torch.argwhere(M == -1.0).flatten().tolist()
        return sign_map


class _VectorMapFact:
    """
    Create an object that maps tokens
    to a list of vectors. Each vector will be mapped with
    the same frequency.
    """

    def __init__(
        self,
        max_token: int,
        n_signals: int,
        vectors: list[list[float]],
        device: str = "cpu",
        zero_value: float = 0.0,
    ):
        n_vectors = len(vectors)
        M = torch.full((max_token + 1, n_signals), fill_value=zero_value)

        if n_vectors == 0:
            self.M = M.to(device)
            return

        if not all(len(d) == n_signals for d in vectors):
            raise ValueError(f"Not all vectors have length of signal_size={n_signals}")

        if n_vectors > max_token:
            raise ValueError(
                f"There are max_token={max_token} and {n_vectors} vectors."
                " It is not possible to map all vectors"
            )

        for vector in vectors:
            if all(d == 0.0 for d in vector):
                raise ValueError(
                    "At least one vector includes only zeros."
                    " Each vector should contain at least one non-zero value."
                )

        idxs = random.choices(list(range(n_vectors)), k=max_token)
        for row_i, idx in enumerate(idxs):
            M[row_i + 1] = torch.tensor(vectors[idx])
        self.M = M.to(device)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """t: long (c, p, d)"""
        return self.M[t]


class _ReactionMapFact(_VectorMapFact):
    """
    Create an object that maps tokens to vectors.
    Each vector has signals length and represents the
    stoichiometry of a reaction.
    """

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
        vectors = [[0.0] * n_signals for _ in range(n_reacts)]
        for ri, (lft, rgt) in enumerate(reactions):
            for mol in lft:
                mol_i = molmap[mol]
                vectors[ri][mol_i] -= 1
            for mol in rgt:
                mol_i = molmap[mol]
                vectors[ri][mol_i] += 1

        super().__init__(
            vectors=vectors,
            n_signals=n_signals,
            max_token=max_token,
            device=device,
            zero_value=zero_value,
        )

    def inverse(
        self,
        molmap: dict[Molecule, int],
        reactions: list[tuple[list[Molecule], list[Molecule]]],
        n_signals: int,
    ) -> dict[tuple[tuple[Molecule, ...], tuple[Molecule, ...]], list[int]]:
        react_map = {}
        for subs, prods in reactions:
            t = torch.zeros(n_signals)
            for sub in subs:
                t[molmap[sub]] -= 1
            for prod in prods:
                t[molmap[prod]] += 1
            M = self.M
            idxs = torch.argwhere((M == t).all(dim=1)).flatten().tolist()
            react_map[(tuple(subs), tuple(prods))] = idxs
        return react_map


class _TransporterMapFact(_VectorMapFact):
    """
    Create an object that maps tokens to vectors.
    Each vector has signals length and represents the
    stoichiometry of a molecule transport into or out of the cell.
    """

    def __init__(
        self,
        n_molecules: int,
        max_token: int,
        device: str = "cpu",
        zero_value: float = 0.0,
    ):
        n_signals = 2 * n_molecules

        # careful, only copy [0] to avoid having references to the same list
        vectors = [[0.0] * n_signals for _ in range(n_signals)]
        for mi in range(n_molecules):
            vectors[mi][mi] = 1
            vectors[mi][mi + n_molecules] = -1
            vectors[mi + n_molecules][mi] = -1
            vectors[mi + n_molecules][mi + n_molecules] = 1

        super().__init__(
            vectors=vectors,
            n_signals=n_signals,
            max_token=max_token,
            device=device,
            zero_value=zero_value,
        )

    def inverse(self, molecules: list[Molecule]) -> dict[Molecule, list[int]]:
        trnsp_map = {}
        for mi, mol in enumerate(molecules):
            M = self.M
            idxs = torch.argwhere(M[:, mi] != 0).flatten().tolist()
            trnsp_map[mol] = idxs
        return trnsp_map


class _RegulatoryMapFact(_VectorMapFact):
    """
    Create an object that maps tokens to vectors.
    Each vector has signals length and represents the
    either activating (+1) or inhibiting (-1) effect
    of an effector molecule.
    """

    def __init__(
        self,
        n_molecules: int,
        max_token: int,
        device: str = "cpu",
        zero_value: float = 0.0,
    ):
        n_signals = 2 * n_molecules

        # careful, only copy [0] to avoid having references to the same list
        vectors = [[0.0] * n_signals for _ in range(n_signals)]
        for mi in range(n_signals):
            vectors[mi][mi] = 1

        super().__init__(
            vectors=vectors,
            n_signals=n_signals,
            max_token=max_token,
            device=device,
            zero_value=zero_value,
        )

    def inverse(self, molecules: list[Molecule]) -> dict[Molecule, list[int]]:
        reg_map = {}
        for mi, mol in enumerate(molecules):
            M = self.M
            idxs = torch.argwhere(M[:, mi] != 0).flatten().tolist()
            reg_map[mol] = idxs
        return reg_map


class Kinetics:
    """
    Class holding logic for simulating protein work.
    Usually this class is instantiated automatically when initializing [World][magicsoup.world.World].
    You can access it on `world.kinetics`.

    Arguments:
        molecules: List of molecule species.
            They have to be in the same order as they are on `chemistry.molecules`.
        reactions: List of all possible reactions in this simulation as a list of tuples: `(substrates, products)`.
            All reactions can happen in both directions (left to right or vice versa).
        abs_temp: Absolute temperature in Kelvin will influence the free Gibbs energy calculation of reactions.
            Higher temperature will give the reaction quotient term higher importance.
        km_range: The range from which to sample Michaelis Menten constants for domains (in mM).
            They are sampled from a lognormal distribution, so all values must be > 0.
        vmax_range: The range from which to sample maximum velocities for domains (in mM/s).
            They are sampled from a lognormal distribution, so all values must be > 0.
        device: Device to use for tensors
            (see [pytorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)).
            This has to be the same device that is used by `world`.
        scalar_enc_size: Number of tokens that can be used to encode the scalars Vmax, Km, and sign.
            This should be the output of `max(genetics.one_codon_map.values())`.
        vector_enc_size: Number of tokens that can be used to encode the vectors for reactions and molecules.
            This should be the output of `max(genetics.two_codon_map.values())`.

    There are `c` cells, `p` proteins, `s` signals.
    Signals are basically molecule species, but we have to differentiate between intra- and extracellular molecule species.
    So, there are twice as many signals as molecule species.
    The order of molecule species is always the same as in `chemistry.molecules`.
    First, all intracellular molecule species are listed, then all extracellular.
    The order of cells is always the same as in `world.genomes` and the order of proteins
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
    which reads proteomes and updates cell parameters accordingly.
    This is called whenever the proteomes of some cells changed.
    Currently, this is also the main bottleneck in performance.

    When this class is initialized it generates the mappings from nucleotide sequences to domains by random sampling.
    These mappings are then used throughout the simulation.
    If you initialize this class again, these mappings will be different.
    Initializing [World][magicsoup.world.World] will also create one `Kinetics` instance. It is on `world.kinetics`.
    If you want to access nucleotide to domain mappings of your simulation, you should use `world.kinetics`.

    As all reactions are calculated step by step, there are some corrections for spcial cases.
    E.g. some corrections make sure that a very sensitive, high-velocity enzyme doesn't accidentally
    deconstruct more substrate than available. To make kinetic calculations smoother, one could reduce
    the maximum velocity by e.g. 10 (`vmax_range=(1e-4, 10.0)`).
    [enzymatic_activity][magicsoup.world.World.enzymatic_activity] would then represent 0.1s instead of 1s.
    The same can be done for diffusion and degradations by lowering the [Molecule][magicsoup.containers.Molecule]
    parameters. I haven't tried around with this a lot, but on a GPU diffusion and enzymatic activity are very
    efficient. So, calling them 10x instead of only once only slightly increases the computation time per step.
    """

    def __init__(
        self,
        molecules: list[Molecule],
        reactions: list[tuple[list[Molecule], list[Molecule]]],
        abs_temp: float = 310.0,
        km_range: tuple[float, float] = (1e-2, 100.0),
        vmax_range: tuple[float, float] = (1e-3, 100.0),
        device: str = "cpu",
        scalar_enc_size: int = 64 - 3,
        vector_enc_size: int = 4096 - 3 * 64,
        n_computations: int = 10,
    ):
        self.abs_temp = abs_temp
        self.device = device
        self.abs_temp = abs_temp
        self.n_computations = n_computations

        self.mol_energies = self._tensor_from([d.energy for d in molecules] * 2)
        self.mol_2_mi = {d: i for i, d in enumerate(molecules)}
        self.mi_2_mol = {v: k for k, v in self.mol_2_mi.items()}

        # working cell params
        n_signals = 2 * len(molecules)
        self.Kmf = self._tensor(0, 0)
        self.Kmb = self._tensor(0, 0)
        self.Vmax = self._tensor(0, 0)
        self.N = self._tensor(0, 0, n_signals)
        self.A = self._tensor(0, 0, n_signals)

        # the domain specifications return 4 indexes
        # idx 0-2 are 1-codon idxs for scalars (n=64)
        # idx3 is a 2-codon idx for vetors (n=4096)

        self.km_map = _LogNormWeightMapFact(
            max_token=scalar_enc_size,
            weight_range=km_range,
            device=device,
        )

        self.vmax_map = _LogNormWeightMapFact(
            max_token=scalar_enc_size,
            weight_range=vmax_range,
            device=device,
        )

        self.sign_map = _SignMapFact(max_token=scalar_enc_size, device=device)

        self.reaction_map = _ReactionMapFact(
            molmap=self.mol_2_mi,
            reactions=reactions,
            max_token=vector_enc_size,
            device=device,
        )

        self.transport_map = _TransporterMapFact(
            n_molecules=len(molecules), max_token=vector_enc_size, device=device
        )

        self.effector_map = _RegulatoryMapFact(
            n_molecules=len(molecules), max_token=vector_enc_size, device=device
        )

        # derive inverse maps for genome generation
        self.sign_2_idxs = self.sign_map.inverse()
        self.trnsp_2_idxs = self.transport_map.inverse(molecules=molecules)
        self.regul_2_idxs = self.effector_map.inverse(molecules=molecules)
        self.catal_2_idxs = self.reaction_map.inverse(
            molmap=self.mol_2_mi, reactions=reactions, n_signals=n_signals
        )

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
            is_useful = False

            doms: list[DomainType] = []
            for di in range(dom_types.size(2)):
                # catalytic domain (N has positive and negative integers)
                if dom_types[0][pi][di].item() == 1:
                    lfts: list[Molecule] = []
                    rgts: list[Molecule] = []
                    for mi, n in enumerate(N_d[0][pi][di].tolist()):
                        if n >= 1:
                            rgts.extend(([self.mi_2_mol[mi]] * int(n)))
                        elif n <= -1:
                            lfts.extend(([self.mi_2_mol[mi]] * -int(n)))
                    if len(lfts) > 0:
                        mi = self.mol_2_mi[lfts[0]]
                        doms.append(
                            CatalyticDomain(
                                reaction=(lfts, rgts),
                                km=Km_d[0][pi][di][mi].item(),
                                vmax=Vmax_d[0][pi][di].item(),
                            )
                        )
                        is_useful = True

                # transporter domain (N has one +1 and one -1)
                if dom_types[0][pi][di].item() == 2:
                    lft = int(torch.argwhere(N_d[0][pi][di] == -1)[0].item())
                    rgt = int(torch.argwhere(N_d[0][pi][di] == 1)[0].item())
                    mi = lft if lft in self.mi_2_mol else rgt
                    doms.append(
                        TransporterDomain(
                            molecule=self.mi_2_mol[mi],
                            km=Km_d[0][pi][di][mi].item(),
                            vmax=Vmax_d[0][pi][di].item(),
                        )
                    )
                    is_useful = True

                # regulatory domain (A has values != 0)
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
                            km=Km_d[0][pi][di][mi].item(),
                            is_inhibiting=bool((A_d[0][pi][di][mi] == -1).item()),
                            is_transmembrane=is_trnsm,
                        )
                    )

            # ignore proteins without a non-regulatory domain
            if is_useful:
                prots.append(Protein(domains=doms))

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

        Proteomes must be represented as a list (proteomes) of lists (proteins)
        of lists (domains) which each carry tuples. These tuples are domain specifications
        that are derived by [Genetics][magicsoup.genetics.Genetics].
        These are indices which will be mapped to concrete values
        (molecule species, Km, Vmax, reactions, ...).
        """
        # get proteome tensors

        dom_types, idxs0, idxs1, idxs2, idxs3 = self._collect_proteome_idxs(
            proteomes=proteomes
        )

        # identify domain types
        # 1=catalytic, 2=transporter, 3=regulatory
        catal_mask = (dom_types == 1).float()  # (c, p, d)
        trnsp_mask = (dom_types == 2).float()  # (c, p, d)
        reg_mask = (dom_types == 3).float()  # (c, p, d)
        catal_trnsp_mask = ((dom_types == 1) | (dom_types == 2)).float()

        # map indices of domain specifications to concrete values
        # idx0 is a 2-codon index specific for every domain type (n=4096)
        # idx1-3 are 1-codon used for the floats (n=64)

        # map indices of domain specifications to scalars/vectors
        # idxs 0-2 are 1-codon indexes used for scalars (n=64 (- stop codons))
        # idx3 is a 2-codon index used for vectors (n=4096 (- stop codons))
        Vmaxs = self.vmax_map(idxs0)  # float (c, p, d)
        Kms = self.km_map(idxs1)  # float (c, p, d)
        signs = self.sign_map(idxs2)  # float (c, p, d)

        reacts = self.reaction_map(idxs3)  # float (c, p, d, s)
        trnspts = self.transport_map(idxs3)  # float (c, p, d, s)
        effectors = self.effector_map(idxs3)  # float (c, p, d, s)

        # N (c, p, d, s)
        N_r = torch.einsum("cpds,cpd->cpds", reacts, catal_mask)
        N_t = torch.einsum("cpds,cpd->cpds", trnspts, trnsp_mask)
        N_d = torch.einsum("cpds,cpd->cpds", (N_r + N_t), signs)

        # A (c, p, d, s)
        A_r = torch.einsum("cpds,cpd->cpds", effectors, reg_mask)
        A_d = torch.einsum("cpds,cpd->cpds", A_r, signs)

        # Km (c, p, d)
        Km_d = Kms

        # Vmax_d (c, p, d)
        Vmax_d = Vmaxs * catal_trnsp_mask

        # aggregations

        # N_d (c, p, d, s)
        N = N_d.sum(dim=2)
        self.N[cell_idxs] = N

        # A_d (c, p, d, s)
        A = A_d.sum(dim=2)
        # TODO: is there a more logical way of doing this?
        # reactants must be added as effectors
        # if their stoichiometric number became 0 during summing up
        # this is a ill-posed problem, e.g.: dom0 d <-> 2b, dom1 b + c <-> d
        # should it be d + c <-> b + d (d is a cofactor for c <-> b)?
        # or b + c <-> 2b (b is a cofactor for c <-> b)?
        # I decided to always keep the substrates of the first domain in A
        delta_N = N_d[:, :, 0].clamp(max=0.0) - N.clamp(max=0.0)
        # A += -1.0 * delta_N.clamp(max=0.0)
        A = A - delta_N.clamp(max=0.0)
        self.A[cell_idxs] = A

        # Km_d (c, p, d)
        # make sure the sampled Km is the smaller one while keeping Ke the same
        E = torch.einsum("cps,s->cp", N, self.mol_energies)
        Ke = torch.exp(-E / self.abs_temp / GAS_CONSTANT)
        ke_ge_1 = Ke >= 1.0
        Km = Km_d.nanmean(dim=2).nan_to_num(0.0)
        self.Kmf[cell_idxs] = Km.clone()
        self.Kmb[cell_idxs] = Km.clone()
        self.Kmf[~ke_ge_1] /= Ke
        self.Kmb[ke_ge_1] *= Ke

        # Vmax_d (c, p, d)
        Vmax = Vmax_d.nanmean(dim=2).nan_to_num(0.0)
        self.Vmax[cell_idxs] = Vmax

    def unset_cell_params(self, cell_idxs: list[int]):
        """
        Unset cell parameters (Vmax, Km, ...) for cells with empty
        or non-viable proteomes.

        Arguments:
            cell_idxs: Indexes of cells
        """
        self.N[cell_idxs] = 0.0
        self.A[cell_idxs] = 0.0
        self.Kmf[cell_idxs] = 0.0
        self.Kmb[cell_idxs] = 0.0
        self.Vmax[cell_idxs] = 0.0

    def integrate_signals(self, X: torch.Tensor) -> torch.Tensor:
        """
        Simulate protein work by integrating all signals.

        Arguments:
            X: Tensor of every signal in every cell (c, s). Must all be >= 0.0.

        Returns:
            New tensor of the same shape which represents the updated signals for every cell.

        The order of cells in `X` is the same as in `world.genomes`
        The order of signals is first all intracellular molecule species in the same order as `chemistry.molecules`,
        then again all molecule species in the same order but this time describing extracellular molecule species.
        The number of intracellular molecules comes from `world.cell_molecules` for any particular cell.
        The number of extracellular molecules comes from `world.molecule_map` from the pixel the particular cell currently lives on.
        """
        # TODO: can be pre-calculated:
        # - masks for substrates and products
        # - Ns for substrates and products
        # - Ns for activators and inhibitors

        Vmax = self.Vmax / self.n_computations
        fwd_mask = torch.full_like(X, 0.0).bool()
        incr_trim = torch.full_like(X, 1.0)

        for i in range(self.n_computations):
            # catalytic activity

            # signals are aggregated for forward and backward reaction
            # proteins that had no involved catalytic region should not be active
            xxf, f_prots = self._aggregate_signals(X=X, mask=self.N < 0.0, N=-self.N)
            kf = xxf / self.Kmf
            kf[~f_prots] = 0.0

            xxb, b_prots = self._aggregate_signals(X=X, mask=self.N > 0.0, N=self.N)
            kb = xxb / self.Kmb
            kb[~b_prots] = 0.0

            # the correct formula yields 2 * (ks - kp) / (1 + ks + kp)
            # but then there can be a maximum activity of 200%
            # while regulatory regions can only go up to 100%
            # thus (ks - kp) / (1 + ks + kp)
            a_cat = (kf - kb) / (1 + kf + kb)  # (c, p)

            # NaNs could have been propagated so far by stray NaN Kms
            # they should represent 0 velocity
            a_cat = a_cat.nan_to_num(0.0)

            # inhibitor activity
            xxi, i_prots = self._aggregate_signals(X=X, mask=self.A < 0.0, N=-self.A)
            a_inh = xxi / (xxi + self.Kmf)
            a_inh[~i_prots] = 0.0  # proteins without inhibitor should be active

            # activator activity
            xxa, a_prots = self._aggregate_signals(X=X, mask=self.A > 0.0, N=self.A)
            a_act = xxa / (xxa + self.Kmf)
            a_act[~a_prots] = 1.0  # proteins without activator should be active

            # velocity and naive Xd
            V = Vmax * a_cat * (1 - a_inh) * a_act  # (c, p)
            Xd = torch.einsum("cps,cp->cs", self.N, V)  # (c, s)

            # if direction of Xd switches back and forth it might be that a protein
            # in the cell is constantly overshooting the equilibrium state
            # with each repetition the cells Xd gets reduced more and more
            # like above this is done for the whole cell in order to avoid creating
            # downstream conflicts with other proteins/signals
            if i == 0:
                fwd_mask[Xd > 0.0] = True
            else:
                new_fwd_mask = Xd > 0.0
                incr_trim[fwd_mask != new_fwd_mask] *= 0.5
                Xd = torch.einsum("cs,c->cs", Xd, incr_trim.amin(1))
                fwd_mask = new_fwd_mask

            # proteins can deconstruct more of a molecule than available in a cell
            # but I can't just clip X at 0 because then reactions would not adhere
            # to their stoichiometry anymore
            # instead I need to reduce protein velocity
            # However, a cell's Xd is the sum of all its protein's activities
            # if I reduce the velocity of 1 protein, it could create a new
            # below-zero situation for another one
            # One would have to repeat this action until all conflicts are satisfied
            # Here, I am instead reducing all the cell's proteins by the same factor
            # that way these secondary below-zero situations cannot appear
            trim_to_zero = torch.where(X + Xd < 0.0, (X - _EPS) / -Xd, 1.0)  # (c, s)
            Xd = torch.einsum("cs,c->cs", Xd, trim_to_zero.amin(1))

            # should not be necessary but floating point precision
            # can still create very small negative values (<1e-7)
            # these are so small that they should not matter in the overall simulation
            X = (X + Xd).clamp(min=0.0)

        return X

    def copy_cell_params(self, from_idxs: list[int], to_idxs: list[int]):
        """
        Copy paremeters from a list of cells to another list of cells

        Arguments:
            from_idxs: List of cell indexes to copy from
            to_idxs: List of cell indexes to copy to

        `from_idxs` and `to_idxs` must have the same length.
        They refer to the same cell indexes as in `world.genomes`.
        """
        self.Kmf[to_idxs] = self.Kmf[from_idxs]
        self.Kmb[to_idxs] = self.Kmb[from_idxs]
        self.Vmax[to_idxs] = self.Vmax[from_idxs]
        self.N[to_idxs] = self.N[from_idxs]
        self.A[to_idxs] = self.A[from_idxs]

    def remove_cell_params(self, keep: torch.Tensor):
        """
        Remove cells from cell params

        Arguments:
            keep: Bool tensor (c,) which is true for every cell that should not be removed
                and false for every cell that should be removed.

        `keep` must have the same length as `world.genomes`.
        The indexes on `keep` reflect the indexes in `world.genomes`.
        """
        self.Kmf = self.Kmf[keep]
        self.Kmb = self.Kmb[keep]
        self.Vmax = self.Vmax[keep]
        self.N = self.N[keep]
        self.A = self.A[keep]

    def increase_max_cells(self, by_n: int):
        """
        Increase the cell dimension of all cell parameters

        Arguments:
            by_n: By how many rows to increase the cell dimension
        """
        self.Kmf = self._expand_c(t=self.Kmf, n=by_n)
        self.Kmb = self._expand_c(t=self.Kmb, n=by_n)
        self.Vmax = self._expand_c(t=self.Vmax, n=by_n)
        self.N = self._expand_c(t=self.N, n=by_n)
        self.A = self._expand_c(t=self.A, n=by_n)

    def increase_max_proteins(self, max_n: int):
        """
        Increase the protein dimension of all cell parameters

        Arguments:
            max_n: The maximum number of rows required in the protein dimension
        """
        n_prots = int(self.N.size(1))
        if max_n > n_prots:
            by_n = max_n - n_prots
            self.Kmf = self._expand_p(t=self.Kmf, n=by_n)
            self.Kmb = self._expand_p(t=self.Kmb, n=by_n)
            self.Vmax = self._expand_p(t=self.Vmax, n=by_n)
            self.N = self._expand_p(t=self.N, n=by_n)
            self.A = self._expand_p(t=self.A, n=by_n)

    def _aggregate_signals(
        self, X: torch.Tensor, mask: torch.Tensor, N: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # consider:
        # - a) some proteins are not involved at all (their Ns are all 0)
        # - b) some are involved but the signal 0 (required X is 0)
        # - c) there can be multiple signals (Xs must be multiplied)
        # - d) each signal has a stoichiometric number (X must be raised)

        # signals are prepared as c,p,s
        # all non-involved signals (or involved but 0)
        # are set to NaN so that it is possible
        # to log them later without need of EPS addition
        x = torch.einsum("cps,cs->cps", mask, X)  # (c, p, s)
        x[x == 0.0] = torch.nan

        # stoichiometric numbers are prepared
        # all non-involved fields are 0
        n = mask * N  # (c, p, s)

        # proteins which have at least 1 non-zero stoichiometric number
        involved_prots = n.sum(2) != 0.0

        # proteins which have all zero signals or which
        # don't have at least 1 non-zero stoichiometric number
        zero_prots = x.isnan().all(2)

        # (1) raise each signal to its stoichiometric number
        # then (2) multiply all over protein
        # while ignoring non-involved signals
        # (1) is done in log-space, all NaNs stay NaN, then
        # (2) can be done as nansum (treats NaNs as 0) so
        # if a protein had only 0 stoichiometric numbers or
        # all involved signals were 0 it will be 0
        # after exp it will become 1
        xx = torch.exp((torch.log(x) * n).nansum(2))

        # proteins that actually had a stoichiometric number
        # but its signal was 0 became 1, but they should be 0
        # I have to identify them with the masks calculated earlier
        # because in xx a signal could theoretically become 1
        # from a legitimate stoichiometric number and non-zero signal
        xx[involved_prots & zero_prots] = 0.0

        return xx, involved_prots

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

        # map indices of domain specifications to scalars/vectors
        # idxs 0-2 are 1-codon indexes used for scalars (n=64 (- stop codons))
        # idx3 is a 2-codon index used for vectors (n=4096 (- stop codons))
        Vmaxs = self.vmax_map(idxs0)  # float (c, p, d)
        Kms = self.km_map(idxs1)  # float (c, p, d)
        signs = self.sign_map(idxs2)  # float (c, p, d)

        reacts = self.reaction_map(idxs3)  # float (c, p, d, s)
        trnspts = self.transport_map(idxs3)  # float (c, p, d, s)
        effectors = self.effector_map(idxs3)  # float (c, p, d, s)

        # N (c, p, d, s)
        N_r = torch.einsum("cpds,cpd->cpds", reacts, catal_mask)
        N_t = torch.einsum("cpds,cpd->cpds", trnspts, trnsp_mask)
        N_d = torch.einsum("cpds,cpd->cpds", (N_r + N_t), signs)

        # A (c, p, d, s)
        A_r = torch.einsum("cpds,cpd->cpds", effectors, reg_mask)
        A_d = torch.einsum("cpds,cpd->cpds", A_r, signs)

        # Km (c, p, d)

        # Vmax_d (c, p, d)
        Vmax_d = Vmaxs * catal_trnsp_mask

        return N_d, A_d, Kms, Vmax_d, dom_types

    def _collect_proteome_idxs(
        self, proteomes: list[list[list[tuple[int, int, int, int, int]]]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _expand_c(self, t: torch.Tensor, n: int) -> torch.Tensor:
        size = t.size()
        zeros = self._tensor(n, *size[1:])
        return torch.cat([t, zeros], dim=0)

    def _expand_p(self, t: torch.Tensor, n: int) -> torch.Tensor:
        size = t.size()
        zeros = self._tensor(size[0], n, *size[2:])
        return torch.cat([t, zeros], dim=1)

    def _tensor(self, *args) -> torch.Tensor:
        return torch.zeros(*args).to(self.device)

    def _tensor_from(self, d: Any) -> torch.Tensor:
        return torch.tensor(d).to(self.device)
