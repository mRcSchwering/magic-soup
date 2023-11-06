from typing import Any
import math
import random
import torch
from magicsoup.constants import GAS_CONSTANT, DomainSpecType, ProteinSpecType
from magicsoup.containers import (
    Molecule,
    Protein,
    CatalyticDomain,
    TransporterDomain,
    RegulatoryDomain,
    Domain,
)

_EPS = 1e-7
_MIN = 1e-45
_MAX = 1e38


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
        M = self.signs.to("cpu")
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
            M = self.M.to("cpu")
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
            M = self.M.to("cpu")
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
            M = self.M.to("cpu")
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
            All reactions are reversible and happen in both directions (left to right or vice versa).
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
        n_computations: In how many computations to integrate signals. With a higher number chemical equilibriums
            are reached smoother but it takes more time to compute (see details below about computation).
        alpha: A trimming factor used to trim velocity at each step during signal integration. This number
            can be adjusted if `n_computations` is changed and reactions don't reach their equilibrium state
            anymore (see details below about computation).

    There are `c` cells, `p` proteins, `s` signals.
    Signals are basically molecule species, but we have to differentiate between intra- and extracellular molecule species.
    So, there are twice as many signals as molecule species.
    The order of molecule species is always the same as in `chemistry.molecules`.
    First, all intracellular molecule species are listed, then all extracellular.
    The order of cells is always the same as in `world.cell_genomes` and the order of proteins
    for every cell is always the same as the order of proteins in a cell object `cell.get_proteome(world=world)`.

    Attributes on this class describe cell parameters:

    - `Kmf`, `Kmb`, `Kmr` Affinities to every signal that is processed by each protein in every cell (c, p, s).
      There are affinities for (f)orward and (b)ackward reactions, and for (r)egulatory domains.
    - `Vmax` Maximum velocities of every protein in every cell (c, p).
    - `E` Standard reaction Gibbs free energy of every protein in every cell (c, p).
    - `N` Stoichiometric number for every signal that is processed by each protein in every cell (c, p, s).
      Additionally, there are `Nf` and `Nb` which describe only forward and backward stoichiometric numbers.
      This is needed in addition to `N` to properly describe reactions that involve molecules
      which are not changed, _i.e._ where `n=0`.
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
    Currently, this is also the main performance bottleneck.

    When this class is initialized it generates the mappings from nucleotide sequences to domains by random sampling.
    These mappings are then used throughout the simulation.
    If you initialize this class again, these mappings will be different.
    Initializing [World][magicsoup.world.World] will also create one `Kinetics` instance. It is on `world.kinetics`.
    If you want to access nucleotide to domain mappings of your simulation, you should use `world.kinetics`.

    Note: All default values are based on the assumption that energies are in J, a time step represents 1s,
    and molecule numbers are in mM
    If you change the defaults, you might reconsider how these numbers should be interpreted.
    If you lower the minimum `Km` or raise the maximum `Vmax`, please also note the comments about
    `n_computation` and `alpha` below.

    The kinetics used here can never create negative molecule concentrations and make the reaction quotient move
    towards to equilibrium constant at all times.
    However, this simulation computes these things one step at a time
    (with the premise that one step should be something similar to 1s).
    This and the numerical limits of data types used here can create edge cases that need to be dealth with:
    reaction quotients overshooting the equilibrium state and negative concentrations.
    Both are caused by high $V_{max}$ with low $K_m$ values.

    If an enzyme is far away from its equilibrium state $K_e$ and substrate concentrations are far above $K_M$ it
    will progress its reaction at full speed $V_{max}$. This rate can be so high that, within one step, $Q$ surpasses
    $K_E$. In the next step it will move at full speed into the opposite direction, overshooting $K_E$ again, and so on.
    Reactions with high stoichiometric numbers are more prone to this as their rate functions are sharper.
    To combat this one can divide the whole computation into many steps.
    This is what `n_computations` is for. $V_{max}$ is decreased appropriately.
    However, with this alone one would have to perform over 100 computations per step in order to make some
    extreme reactions reach their equilibrium state.
    Thus, with each computation in `n_computations` the reaction rate is exponentially decayed.
    This has the effect that, while in the first few computations a reaction might still overshoot $K_E$,
    during the last computations rates are so low, that even extremely volatile reactions can come close
    to their intended equilibrium. The factor of this exponential decay is `alpha`.
    For $V_{max} \le 100$ and $K_M \ge 0.01$ `n_computations=11` and `alpha=0.6` seem to work very well
    (few computations, but stable equilibriums). You can always increase `n_computations`, but if you
    decrease it, you might also want to try out different `alpha` values.

    With the `n_computation` and `alpha` trick above, chances of enzymes trying to deconstruct more
    substrate than available is very low. However, it can still happen. Sometimes it is also floating point
    inaccuracies that can tip a near-zero value below zero. Thus, before the updated signals tensor `X`
    gets returned, it is checked for negative concentrations.
    Then, protein velocities for all cells with below zero concentrations is reduced and `X` is calculated again.
    They are reduced by a factor that will create $X = \epsilon \gt 0$ for the signals that
    were negative in the first calculation.
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
        n_computations: int = 11,
        alpha: float = 0.6,
    ):
        self.abs_temp = abs_temp
        self.device = device
        self.abs_temp = abs_temp

        # calculate signal integration in n_computation steps
        # to reach equilibrium Ke faster, steps get increasingly slower
        # trim(i) = trim0 * alpha^i and sum(trim0 * alpha^i) = 1 over all i
        self.n_computations = n_computations
        self.alpha = alpha
        self.n_comp_trim = 1 / sum(self.alpha**d for d in range(self.n_computations))

        self.mol_energies = self._tensor_from([d.energy for d in molecules] * 2)
        self.mol_2_mi = {d: i for i, d in enumerate(molecules)}
        self.mi_2_mol = {v: k for k, v in self.mol_2_mi.items()}
        self.molecules = molecules

        # working cell params
        n_signals = 2 * len(molecules)
        self.Kmf = self._tensor(0, 0)
        self.Kmb = self._tensor(0, 0)
        self.Kmr = self._tensor(0, 0)
        self.Vmax = self._tensor(0, 0)
        self.N = self._tensor(0, 0, n_signals)
        self.Nf = self._tensor(0, 0, n_signals)
        self.Nb = self._tensor(0, 0, n_signals)
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
        self,
        proteome: list[ProteinSpecType],
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
            proteomes=[[d[0] for d in proteome]]
        )

        Nf_d = torch.where(N_d < 0.0, -N_d, 0.0)
        Nb_d = torch.where(N_d > 0.0, N_d, 0.0)
        mols = self.molecules
        n_mols = len(mols)

        # looping through list of lists while accessing tensors
        # should be ok because in _get_proteome_tensors none are filtered
        prots: list[Protein] = []
        for pi, protein_spec in enumerate(proteome):
            doms: list[Domain] = []
            for di, dom_spec in enumerate(protein_spec[0]):
                kwargs = {
                    "start": dom_spec[1],
                    "end": dom_spec[2],
                }

                # catalytic domain (N has positive and negative integers)
                if dom_types[0][pi][di].item() == 1:
                    lft_ns = Nf_d[0][pi][di].int().tolist()
                    rgt_ns = Nb_d[0][pi][di].int().tolist()
                    lfts = [m for m, n in zip(mols, lft_ns) for _ in range(n)]
                    rgts = [m for m, n in zip(mols, rgt_ns) for _ in range(n)]
                    if len(lfts) > 0:
                        doms.append(
                            CatalyticDomain(
                                reaction=(lfts, rgts),
                                km=Km_d[0][pi][di].item(),
                                vmax=Vmax_d[0][pi][di].item(),
                                **kwargs,
                            )
                        )

                # transporter domain (N has one +1 and one -1)
                if dom_types[0][pi][di].item() == 2:
                    mis = torch.argwhere(N_d[0][pi][di] != 0).int().flatten().tolist()
                    doms.append(
                        TransporterDomain(
                            molecule=self.mi_2_mol[min(mis)],
                            km=Km_d[0][pi][di].item(),
                            vmax=Vmax_d[0][pi][di].item(),
                            **kwargs,
                        )
                    )

                # regulatory domain (A has one +1 or -1)
                if dom_types[0][pi][di].item() == 3:
                    mi = int(torch.argwhere(A_d[0][pi][di] != 0)[0].item())
                    if mi in self.mi_2_mol:
                        is_trnsm = False
                        mol = self.mi_2_mol[mi]
                    else:
                        is_trnsm = True
                        mol = self.mi_2_mol[mi - n_mols]
                    doms.append(
                        RegulatoryDomain(
                            effector=mol,
                            km=Km_d[0][pi][di].item(),
                            is_inhibiting=bool((A_d[0][pi][di][mi] == -1).item()),
                            is_transmembrane=is_trnsm,
                            **kwargs,
                        )
                    )

            if len(doms) > 0:
                prots.append(
                    Protein(
                        domains=doms,
                        cds_start=proteome[pi][1],
                        cds_end=proteome[pi][2],
                        is_fwd=proteome[pi][3],
                    )
                )

        return prots

    def set_cell_params(
        self,
        cell_idxs: list[int],
        proteomes: list[list[list[DomainSpecType]]],
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
        is_catal = dom_types == 1  # (c, p, d)
        is_trnsp = dom_types == 2  # (c, p, d)
        is_reg = dom_types == 3  # (c, p, d)

        # map indices of domain specifications to concrete values
        # idx0 is a 2-codon index specific for every domain type (n=4096)
        # idx1-3 are 1-codon used for the floats (n=64)
        # some values are not defined for certain domain types
        # setting their indices to 0 lets them map to empty values (0-vector, NaN)
        catal_lng = (is_catal).long()
        trnsp_lng = (is_trnsp).long()
        reg_lng = (is_reg).long()
        not_reg_lng = (~is_reg).long()

        # idxs 0-2 are 1-codon indexes used for scalars (n=64 (- stop codons))
        Vmaxs = self.vmax_map(idxs0 * not_reg_lng)  # float (c, p, d)
        Kms = self.km_map(idxs1)  # float (c, p, d)
        signs = self.sign_map(idxs2)  # float (c, p, d)

        # idx3 is a 2-codon index used for vectors (n=4096 (- stop codons))
        reacts = self.reaction_map(idxs3 * catal_lng)  # float (c, p, d, s)
        trnspts = self.transport_map(idxs3 * trnsp_lng)  # float (c, p, d, s)
        effectors = self.effector_map(idxs3 * reg_lng)  # float (c, p, d, s)

        # Vmax are averaged over domains
        # undefined Vmax enries are NaN and are ignored by nanmean
        self.Vmax[cell_idxs] = Vmaxs.nanmean(dim=2).nan_to_num(0.0)

        # effector vectors are summed up over domains
        # undefined vectors are all 0s
        self.A[cell_idxs] = torch.einsum("cpds,cpd->cpds", effectors, signs).sum(dim=2)

        # Kms of regulatory domains are aggregated for effectors
        # Kms from other domains are ignored using NaNs and nanmean
        self.Kmr[cell_idxs] = (
            torch.where(is_reg, Kms, torch.nan).nanmean(dim=2).nan_to_num(0.0)
        )

        # reaction stoichiometry N is derived from transporter and catalytic vectors
        # vectors for regulatory domains or emptpy proteins are all 0s
        N_d = torch.einsum("cpds,cpd->cpds", (reacts + trnspts), signs)
        N = N_d.sum(dim=2)
        self.N[cell_idxs] = N

        # N for forward and backward reactions is distinguished
        # to not loose molecules like co-facors whose net N would become 0
        self.Nf[cell_idxs] = torch.where(N_d < 0.0, -N_d, 0.0).sum(dim=2)
        self.Nb[cell_idxs] = torch.where(N_d > 0.0, N_d, 0.0).sum(dim=2)

        # Kms of catalytic and transporter domains are aggregated
        # Kms from other domains are ignored using NaNs and nanmean
        Kmn = torch.where(~is_reg, Kms, torch.nan).nanmean(dim=2).nan_to_num(0.0)

        # energies define Ke which defines Ke = Kmf/Kmb
        # extreme energies can create Inf or 0.0, avoid them with clamp
        E = torch.einsum("cps,s->cp", N, self.mol_energies)
        Ke = torch.exp(-E / self.abs_temp / GAS_CONSTANT).clamp(_MIN, _MAX)

        # Km is sampled between a defined range
        # exessively small Km can create numerical instability
        # thus, sampled Km should define the smaller Km of Ke = Kmf/Kmb
        # Ke>=1  => Kmf=Km,         Kmb=Ke*Km
        # Ke<1   => Kmf=Km/Ke,      Kmb=Km
        # this operation can create again Inf or 0.0, avoided with clamp
        # this effectively limits Ke around 1e38
        is_fwd = Ke >= 1.0
        self.Kmf[cell_idxs] = torch.where(is_fwd, Kmn, Kmn / Ke).clamp(_MIN, _MAX)
        self.Kmb[cell_idxs] = torch.where(is_fwd, Kmn * Ke, Kmn).clamp(_MIN, _MAX)

    def unset_cell_params(self, cell_idxs: list[int]):
        """
        Unset cell parameters (Vmax, Km, ...) for cells with empty
        or non-viable proteomes.

        Arguments:
            cell_idxs: Indexes of cells
        """
        self.N[cell_idxs] = 0.0
        self.Nf[cell_idxs] = 0.0
        self.Nb[cell_idxs] = 0.0
        self.A[cell_idxs] = 0.0
        self.Kmf[cell_idxs] = 0.0
        self.Kmb[cell_idxs] = 0.0
        self.Kmr[cell_idxs] = 0.0
        self.Vmax[cell_idxs] = 0.0

    def integrate_signals(self, X: torch.Tensor) -> torch.Tensor:
        """
        Simulate protein work by integrating all signals.

        Arguments:
            X: Tensor of every signal in every cell (c, s). Must all be >= 0.0.

        Returns:
            New tensor of the same shape which represents the updated signals for every cell.

        The order of cells in `X` is the same as in `world.cell_genomes`
        The order of signals is first all intracellular molecule species in the same order as `chemistry.molecules`,
        then again all molecule species in the same order but this time describing extracellular molecule species.
        The number of intracellular molecules comes from `world.cell_molecules` for any particular cell.
        The number of extracellular molecules comes from `world.molecule_map` from the pixel the particular cell currently lives on.
        """
        is_fwd = self.Nf > 0.0
        is_bwd = self.Nb > 0.0
        is_inh = self.A < 0.0
        is_act = self.A > 0.0
        Vmax_adj = self.Vmax * self.n_comp_trim

        for i in range(self.n_computations):
            # catalytic activity

            # signals are aggregated for forward and backward reaction
            # proteins that had no involved catalytic region should not be active
            xxf, f_prots = self._aggregate_signals(X=X, mask=is_fwd, N=self.Nf)
            kf = xxf / self.Kmf
            kf[~f_prots] = 0.0  # rm artifacts created by ln

            xxb, b_prots = self._aggregate_signals(X=X, mask=is_bwd, N=self.Nb)
            kb = xxb / self.Kmb
            kb[~b_prots] = 0.0  # rm artifacts created by ln

            # custom reversible MM equation
            a_cat = (kf - kb) / (1 + kf + kb)  # (c, p)

            # NaNs could have been propagated so far by stray NaN Kms
            # they should represent 0 velocity
            a_cat = a_cat.nan_to_num(0.0)

            # inhibitor activity
            xxi, i_prots = self._aggregate_signals(X=X, mask=is_inh, N=-self.A)
            a_inh = xxi / (xxi + self.Kmr)
            a_inh[~i_prots] = 0.0  # proteins without inhibitor should be active

            # activator activity
            xxa, a_prots = self._aggregate_signals(X=X, mask=is_act, N=self.A)
            a_act = xxa / (xxa + self.Kmr)
            a_act[~a_prots] = 1.0  # proteins without activator should be active

            # velocity from activities
            # as well as trimming factor of n_computations
            V = Vmax_adj * a_cat * (1 - a_inh) * a_act * self.alpha**i  # (c, p)

            # Kms can be close to Inf, they can create Infs
            # thus result velocity should be clamped again
            V = V.clamp(max=_MAX)

            # naive Xd, might produce negative X
            Xd = torch.einsum("cps,cp->cs", self.N, V)  # (c, s)

            # proteins can deconstruct more of a molecule than available in a cell
            # this happens mainly with multiple proteins with low Km trying to deconstruct
            # the same molecule species in one cell
            # reaction stoichiometry must be obeyed, so mere X.clamp(0.0) is not allowed
            # One could factor out which proteins must be slowed down by how much,
            # and selectively only slow down these proteins by a certain trimming factor.
            # However, this could create a follow-up X<0.0 situation with one of the other proteins
            # which were not slowed down. To solve this, one would have to construct a
            # dependency graph of the protein network, or just iterate this process often enough.
            # Here, I am instead reducing all the cell's proteins by the same factor
            # that way these follow-up X<0.0 situations cannot appear
            # due to floating point precision I need to lift the cutoff from 0 to EPS
            trim_factors = (X / -Xd - _EPS).clamp(0.0)
            cell_trims = torch.where(X + Xd < 0.0, trim_factors, 1.0).amin(1)  # (c,)
            Xd = torch.einsum("cs,c->cs", Xd, cell_trims)

            # update signals, this time no negative X
            X = X + Xd

            # above was tested without clamp(0.0) and it never failed
            # However, if due to some floating point inaccuracies, a negative X slips through
            # it could create either NaNs or extremely large values (it would ruin a simulation)
            # Thus, as a final measure clamp is used (it should never actually do anything)
            X = X.clamp(0.0)

        # NaNs can be created when overflow creates Infs (most likely in aggregate_signals)
        # with kinetics default values I have not been able to achieve this (>100 testruns)
        # however non-default values (e.g. large Vmax) might achieve that
        # once a 1 NaN is generated, it will spread over the whole simulation
        # this is a last effort in avoiding that
        X[X.isnan()] = 0.0

        return X

    def copy_cell_params(self, from_idxs: list[int], to_idxs: list[int]):
        """
        Copy paremeters from a list of cells to another list of cells

        Arguments:
            from_idxs: List of cell indexes to copy from
            to_idxs: List of cell indexes to copy to

        `from_idxs` and `to_idxs` must have the same length.
        They refer to the same cell indexes as in `world.cell_genomes`.
        """
        self.Kmf[to_idxs] = self.Kmf[from_idxs]
        self.Kmb[to_idxs] = self.Kmb[from_idxs]
        self.Kmr[to_idxs] = self.Kmr[from_idxs]
        self.Vmax[to_idxs] = self.Vmax[from_idxs]
        self.N[to_idxs] = self.N[from_idxs]
        self.Nf[to_idxs] = self.Nf[from_idxs]
        self.Nb[to_idxs] = self.Nb[from_idxs]
        self.A[to_idxs] = self.A[from_idxs]

    def remove_cell_params(self, keep: torch.Tensor):
        """
        Remove cells from cell params

        Arguments:
            keep: Bool tensor (c,) which is true for every cell that should not be removed
                and false for every cell that should be removed.

        `keep` must have the same length as `world.cell_genomes`.
        The indexes on `keep` reflect the indexes in `world.cell_genomes`.
        """
        self.Kmf = self.Kmf[keep]
        self.Kmb = self.Kmb[keep]
        self.Kmr = self.Kmr[keep]
        self.Vmax = self.Vmax[keep]
        self.N = self.N[keep]
        self.Nf = self.Nf[keep]
        self.Nb = self.Nb[keep]
        self.A = self.A[keep]

    def increase_max_cells(self, by_n: int):
        """
        Increase the cell dimension of all cell parameters

        Arguments:
            by_n: By how many rows to increase the cell dimension
        """
        self.Kmf = self._expand_c(t=self.Kmf, n=by_n)
        self.Kmb = self._expand_c(t=self.Kmb, n=by_n)
        self.Kmr = self._expand_c(t=self.Kmr, n=by_n)
        self.Vmax = self._expand_c(t=self.Vmax, n=by_n)
        self.N = self._expand_c(t=self.N, n=by_n)
        self.Nf = self._expand_c(t=self.Nf, n=by_n)
        self.Nb = self._expand_c(t=self.Nb, n=by_n)
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
            self.Kmr = self._expand_p(t=self.Kmr, n=by_n)
            self.Vmax = self._expand_p(t=self.Vmax, n=by_n)
            self.N = self._expand_p(t=self.N, n=by_n)
            self.Nf = self._expand_p(t=self.Nf, n=by_n)
            self.Nb = self._expand_p(t=self.Nb, n=by_n)
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
        self,
        proteomes: list[list[list[DomainSpecType]]],
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
        self,
        proteomes: list[list[list[DomainSpecType]]],
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
                for (dt, i0, i1, i2, i3), *_ in doms:
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
