from typing import Any
import math
import random
import torch
from magicsoup.constants import GAS_CONSTANT, ProteinSpecType
from magicsoup.containers import Molecule, Protein, Chemistry
from magicsoup import _lib  # type: ignore

# MAX,MIN should be at least x100 away from inf
# EPS should be 1/MAX
_EPS = 1e-36
_MAX = 1e36
_MIN = -1e36


class _HillMapFact:
    """
    Creates an object that maps tokens to 1, 2, 3, 4, 5
    with chances 52%, 26%, 13%, 6%, 3% respectively.
    """

    def __init__(self, max_token: int, device: str = "cpu", zero_value: int = 0):
        choices = [5] + 2 * [4] + 4 * [3] + 8 * [2] + 16 * [1]
        numbers = torch.tensor([zero_value] + random.choices(choices, k=max_token))
        self.numbers = numbers.to(device=device, dtype=torch.int32)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """t: long (c, p, d)"""
        return self.numbers[t]

    def inverse(self) -> dict[int, list[int]]:
        numbers_map = {}
        M = self.numbers.to("cpu")
        numbers_map[1] = torch.argwhere(M == 1.0).flatten().tolist()
        numbers_map[3] = torch.argwhere(M == 3.0).flatten().tolist()
        numbers_map[5] = torch.argwhere(M == 5.0).flatten().tolist()
        return numbers_map


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
        self.weights = weights.to(device=device, dtype=torch.float32)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """t: long (c, p, d)"""
        return self.weights[t]

    def inverse(self) -> dict[float, list[int]]:
        flt_map: dict[float, list[int]] = {}
        M = self.weights.to("cpu")
        for i in range(1, M.size(0)):
            v = M[i].item()
            if v not in flt_map:
                flt_map[v] = []
            flt_map[v].append(i)
        return flt_map


class _SignMapFact:
    """
    Creates an object that maps tokens to 1 or -1
    with 50% probability of each being mapped.
    """

    def __init__(self, max_token: int, device: str = "cpu", zero_value: int = 0):
        choices = [1, -1]
        signs = torch.tensor([zero_value] + random.choices(choices, k=max_token))
        self.signs = signs.to(device=device, dtype=torch.int32)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """t: long (c, p, d)"""
        return self.signs[t]

    def inverse(self) -> dict[bool, list[int]]:
        sign_map = {}
        M = self.signs.to("cpu")
        sign_map[True] = torch.argwhere(M == 1).flatten().tolist()
        sign_map[False] = torch.argwhere(M == -1).flatten().tolist()
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
        vectors: list[list[int]],
        device: str = "cpu",
        zero_value: int = 0,
    ):
        n_vectors = len(vectors)
        M = torch.full((max_token + 1, n_signals), fill_value=zero_value)

        if n_vectors == 0:
            self.M = M.to(device=device, dtype=torch.int32)
            return

        if not all(len(d) == n_signals for d in vectors):
            raise ValueError(f"Not all vectors have length of signal_size={n_signals}")

        if n_vectors > max_token:
            raise ValueError(
                f"There are max_token={max_token} and {n_vectors} vectors."
                " It is not possible to map all vectors"
            )

        for vector in vectors:
            if all(d == 0 for d in vector):
                raise ValueError(
                    "At least one vector includes only zeros."
                    " Each vector should contain at least one non-zero value."
                )

        idxs = random.choices(list(range(n_vectors)), k=max_token)
        for row_i, idx in enumerate(idxs):
            M[row_i + 1] = torch.tensor(vectors[idx])
        self.M = M.to(device=device, dtype=torch.int32)

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
        zero_value: int = 0,
    ):
        n_signals = 2 * len(molmap)
        n_reacts = len(reactions)

        # careful, only copy [0] to avoid having references to the same list
        vectors = [[0] * n_signals for _ in range(n_reacts)]
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
        M = self.M.to("cpu")
        for subs, prods in reactions:
            t = torch.zeros(n_signals)
            for sub in subs:
                t[molmap[sub]] -= 1
            for prod in prods:
                t[molmap[prod]] += 1
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
        zero_value: int = 0,
    ):
        n_signals = 2 * n_molecules

        # careful, only copy [0] to avoid having references to the same list
        vectors = [[0] * n_signals for _ in range(n_molecules)]
        for mi in range(n_molecules):
            vectors[mi][mi] = -1
            vectors[mi][mi + n_molecules] = 1

        super().__init__(
            vectors=vectors,
            n_signals=n_signals,
            max_token=max_token,
            device=device,
            zero_value=zero_value,
        )

    def inverse(self, molecules: list[Molecule]) -> dict[Molecule, list[int]]:
        trnsp_map = {}
        M = self.M.to("cpu")
        for mi, mol in enumerate(molecules):
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
        zero_value: int = 0,
    ):
        n_signals = 2 * n_molecules

        # careful, only copy [0] to avoid having references to the same list
        vectors = [[0] * n_signals for _ in range(n_signals)]
        for mi in range(n_signals):
            vectors[mi][mi] = 1

        super().__init__(
            vectors=vectors,
            n_signals=n_signals,
            max_token=max_token,
            device=device,
            zero_value=zero_value,
        )

    def inverse(
        self, molecules: list[Molecule]
    ) -> dict[tuple[Molecule, bool], list[int]]:
        n = len(molecules)
        reg_map = {}
        M = self.M.to("cpu")
        for mi, mol in enumerate(molecules):
            idxs_int = torch.argwhere(M[:, mi] != 0).flatten().tolist()
            idxs_ext = torch.argwhere(M[:, mi + n] != 0).flatten().tolist()
            reg_map[(mol, False)] = idxs_int
            reg_map[(mol, True)] = idxs_ext
        return reg_map


class Kinetics:
    """
    Class holding logic for simulating protein work.
    Usually this class is instantiated automatically when initializing [World][magicsoup.world.World].
    You can access it on `world.kinetics`.

    Parameters:
        chemistry: Simulation [Chemistry][magicsoup.containers.Chemistry]
        abs_temp: Absolute temperature in Kelvin. Will influence reaction equilibriums.
            Higher temperatures give concentration gradients higher importance.
        km_range: The range from which to sample Michaelis Menten constants for domains (in mM).
            They are sampled from a uniform distribution with its reciprocal. All values must be > 0.
        vmax_range: The range from which to sample maximum velocities for domains (in mM/s).
            They are sampled from a lognormal distribution, so all values must be > 0.
        device: Device to use for tensors
            (see [pytorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)).
            This has to be the same device that is used by [World][magicsoup.world.World].
        scalar_enc_size: Number of tokens that can be used to encode the scalars Vmax, Km, and sign.
            This should be the output of `max(genetics.one_codon_map.values())` ([Genetics][magicsoup.genetics.Genetics]).
        vector_enc_size: Number of tokens that can be used to encode the vectors for reactions and molecules.
            This should be the output of `max(genetics.two_codon_map.values())` ([Genetics][magicsoup.genetics.Genetics]).

    There are `c` cells, `p` proteins, `s` signals.
    Signals are basically [Molecules][magicsoup.containers.Molecule],
    but we have to differentiate between intra- and extracellular molecules.
    So, there are twice as many signals as [Molecules][magicsoup.containers.Molecule].
    Molecules are always ordered as in [Chemistry][magicsoup.containers.Chemistry]:
    first, the intracellular ones, then the extracellular ones.
    Cells are ordered as in [World][magicsoup.world.World]. Proteins are ordered
    how they were found by [Genetics.translate_genomes()][magicsoup.genetics.Genetics.translate_genomes] in each cell.

    Attributes on an object of this class describe cell parameters:

    - `Kmf`, `Kmb` Affinities to all signals processed by each protein in each cell (c, p).
      Ratios of (f)orward and (b)ackward reaction affinities are their equilibrium constants.
    - `Kmr` Affinity of each regulating signal for each protein and each cell (c, p, s)
        exponentiated by their hill coefficients.
    - `Vmax` Maximum velocities of each protein in each cell (c, p).
    - `Ke` Equilibrium constants derived from standard reaction free energy for each protein in each cell (c, p).
    - `N` Stoichiometric number for each signal that is processed by each protein in each cell (c, p, s).
      Additionally, there are `Nf` and `Nb` which describe only forward and backward stoichiometric coefficients.
      This is needed in addition to `N` to properly describe reactions that involve co-factors (_i.e._ `n=0`).
    - `A` Allosteric modulation for each signal in each protein in each cell (c, p, s).
      The number represents the hill coefficient. Positive coefficients have an activating effect,
      negative have an inhibiting effect.

    The main method is [integrate_signals()][magicsoup.kinetics.Kinetics].
    When calling [World.enzymatic_activity()][magicsoup.world.World.enzymatic_activity],
    a matrix `X` of signals (c, s) is prepared and then [integrate_signals(X)][magicsoup.kinetics.Kinetics] is called.
    Updated signals are returned and [World.enzymatic_activity()][magicsoup.world.World.enzymatic_activity]
    writes them back to `world.cell_molecules` and `world.molecule_map`.

    Another method, which ended up here, is [set_cell_params()][magicsoup.kinetics.Kinetics.set_cell_params]
    which reads proteomes and updates cell parameters accordingly.
    This is called whenever the genomes of some cells have changed.
    Currently, this is also the main performance bottleneck.

    When this class is initialized it generates the mappings from indices to domain parameters by random sampling.
    These mappings are then used throughout the simulation.
    If you initialize this class again, these mappings will be different.
    Initializing [World][magicsoup.world.World] will also create one `Kinetics` instance on `world.kinetics`.
    If you want to access indices to domain mappings in your simulation, you should use `world.kinetics`.

    Note: All default values are based on the assumption that energies are in J, a time step represents 1s,
    and molecule numbers are in mM.
    If you change the defaults, you need to reconsider how these numbers should be interpreted.

    The kinetics used here can theoretically never create negative molecule concentrations or
    make a reaction overshoot its rquilibrium state.
    However, this simulation computes these things one step at a time.
    This and the numerical limits of data types used here can create edge cases that need to be dealth with:
    reaction quotients overshooting the equilibrium state and negative concentrations.
    Both are caused by high $V_{max}$ with low $K_m$ values.
    With the default $V_{max}$ and $K_m$ ranges and stoichiometric coefficients below 100,
    the heuristics in [integrate_signals()][magicsoup.kinetics.Kinetics.integrate_signals]
    are close enough for a meaningful simulation.

    If an enzyme is far away from its equilibrium state $K_e$ and substrate concentrations are far above $K_m$ it
    will progress its reaction at full speed $V_{max}$. This rate can be so high that, within one step, $Q$ surpasses
    $K_e$. In the next step it will move at full speed into the opposite direction, overshooting $K_e$ again, and so on.
    Reactions with high stoichiometric coefficients are more prone to this as their rate functions are sharper.
    To combat this [integrate_signals()][magicsoup.kinetics.Kinetics] works by iteratively approaching
    $V_{max}$ on multiple levels of the computation.
    The approach was tuned to compute fast and give satifisfying reults with certain conditions in mind.
    $V_{max} \le 100$, $K_m \ge 0.01$, and concentrations $X$ generally not much higher than 100.
    Violating these assumptions could lead to reaction quotients constantly overshooting their equilibrium.
    """

    def __init__(
        self,
        chemistry: Chemistry,
        abs_temp: float = 310.0,
        km_range: tuple[float, float] = (1e-2, 100.0),
        vmax_range: tuple[float, float] = (1e-3, 100.0),
        device: str = "cpu",
        scalar_enc_size: int = 64 - 3,
        vector_enc_size: int = 4096 - 3 * 64,
    ):
        self.abs_temp = abs_temp
        self.device = device
        self.abs_temp = abs_temp

        self.mol_names = [d.name for d in chemistry.molecules]
        self.mol_energies = self._f32_tensor(
            [d.energy for d in chemistry.molecules] * 2
        )

        # working cell params
        n_signals = 2 * len(chemistry.molecules)
        self.Ke = self._zeros_f32_tensor(0, 0)
        self.Kmf = self._zeros_f32_tensor(0, 0)
        self.Kmb = self._zeros_f32_tensor(0, 0)
        self.Kmr = self._zeros_f32_tensor(0, 0, n_signals)
        self.Vmax = self._zeros_f32_tensor(0, 0)
        self.N = self._zeros_i32_tensor(0, 0, n_signals)
        self.Nf = self._zeros_i32_tensor(0, 0, n_signals)
        self.Nb = self._zeros_i32_tensor(0, 0, n_signals)
        self.A = self._zeros_i32_tensor(0, 0, n_signals)

        # the domain specifications return 4 indexes
        # idx 0-2 are 1-codon idxs for scalars (n=64)
        # idx3 is a 2-codon idx for vetors (n=4096)
        mol_2_mi = {d: i for i, d in enumerate(chemistry.molecules)}

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

        self.hill_map = _HillMapFact(max_token=scalar_enc_size, device=device)

        self.reaction_map = _ReactionMapFact(
            molmap=mol_2_mi,
            reactions=chemistry.reactions,
            max_token=vector_enc_size,
            device=device,
        )

        self.transport_map = _TransporterMapFact(
            n_molecules=len(chemistry.molecules),
            max_token=vector_enc_size,
            device=device,
        )

        self.effector_map = _RegulatoryMapFact(
            n_molecules=len(chemistry.molecules),
            max_token=vector_enc_size,
            device=device,
        )

        # derive inverse maps for genome generation
        self.km_2_idxs = self.km_map.inverse()
        self.vmax_2_idxs = self.vmax_map.inverse()
        self.sign_2_idxs = self.sign_map.inverse()
        self.hill_2_idxs = self.hill_map.inverse()
        self.trnsp_2_idxs = self.transport_map.inverse(molecules=chemistry.molecules)
        self.regul_2_idxs = self.effector_map.inverse(molecules=chemistry.molecules)
        self.catal_2_idxs = self.reaction_map.inverse(
            molmap=mol_2_mi, reactions=chemistry.reactions, n_signals=n_signals
        )

    def get_proteome(
        self,
        proteome: list[ProteinSpecType],
    ) -> list[Protein]:
        """
        Translate and return cell parameters for a single proteome

        Parameters:
            proteome: proteome which should be translated and returned

        Retruns:
            List [Proteins][magicsoup.containers.Protein] that describe
            the cell's proteome.
        """
        # get proteome tensors
        dom_types, idxs0, idxs1, idxs2, idxs3 = self._collect_proteome_idxs(
            proteomes=[proteome]
        )

        # identify domain types
        # 1=catalytic, 2=transporter, 3=regulatory
        is_catal = dom_types == 1  # (c,p,d)
        is_trnsp = dom_types == 2  # (c,p,d)
        is_reg = dom_types == 3  # (c,p,d)

        # map indices of domain specifications to concrete values
        # idx0 is a 2-codon index specific for every domain type (n=4096)
        # idx1-3 are 1-codon used for the floats (n=64)
        # some values are not defined for certain domain types
        # setting their indices to 0 lets them map to empty values (0-vector, NaN)
        catal_lng = (is_catal).int()
        trnsp_lng = (is_trnsp).int()
        reg_lng = (is_reg).int()
        not_reg_lng = (~is_reg).int()

        # idxs 0-2 are 1-codon indexes used for scalars (n=64 (- stop codons))
        Vmaxs = self.vmax_map(idxs0 * not_reg_lng)  # f32 (c,p,d)
        Hills = self.hill_map(idxs0 * reg_lng)  # i32 (c,p,d)
        Kms = self.km_map(idxs1)  # f32 (c,p,d)
        signs = self.sign_map(idxs2)  # i32 (c,p,d)

        # idx3 is a 2-codon index used for vectors (n=4096 (- stop codons))
        reacts = self.reaction_map(idxs3 * catal_lng)  # i32 (c,p,d,s)
        trnspts = self.transport_map(idxs3 * trnsp_lng)  # i32 (c,p,d,s)
        effectors = self.effector_map(idxs3 * reg_lng)  # i32 (c,p,d,s)

        proteome_kwargs = _lib.get_proteome(
            proteome,
            Vmaxs[0].tolist(),
            Kms[0].tolist(),
            Hills[0].tolist(),
            signs[0].tolist(),
            reacts[0].tolist(),
            trnspts[0].tolist(),
            effectors[0].tolist(),
            self.mol_names,
        )
        return [Protein.from_dict(d) for d in proteome_kwargs]

    def set_cell_params(
        self,
        cell_idxs: list[int],
        proteomes: list[list[ProteinSpecType]],
    ):
        """
        Translate and set cell parameters for proteomes

        Parameters:
            cell_idxs: Indexes of cells which proteomes belong to
            proteomes: List of proteomes which should translated and set

        `proteomes` is a a list (proteomes) of lists (proteins) of domain specifications.
        It is derived from [Genetics.translate_genomes()][magicsoup.genetics.Genetics].
        The domain specification themself are tuples which are mapped to concrete values
        (molecule species, Km, Vmax, reactions, ...).
        `cell_idxs` refer to the [World's][magicsoup.world.World] cell indexes.
        """
        # get proteome tensors
        dom_types, idxs0, idxs1, idxs2, idxs3 = self._collect_proteome_idxs(
            proteomes=proteomes
        )

        # identify domain types
        # 1=catalytic, 2=transporter, 3=regulatory
        is_catal = dom_types == 1  # (c,p,d)
        is_trnsp = dom_types == 2  # (c,p,d)
        is_reg = dom_types == 3  # (c,p,d)

        # map indices of domain specifications to concrete values
        # idx0 is a 2-codon index specific for every domain type (n=4096)
        # idx1-3 are 1-codon used for the floats (n=64)
        # some values are not defined for certain domain types
        # setting their indices to 0 lets them map to empty values (0-vector, NaN)
        catal_lng = (is_catal).int()
        trnsp_lng = (is_trnsp).int()
        reg_lng = (is_reg).int()
        not_reg_lng = (~is_reg).int()

        # idxs 0-2 are 1-codon indexes used for scalars (n=64 (- stop codons))
        Vmaxs = self.vmax_map(idxs0 * not_reg_lng)  # f32 (c,p,d)
        Hills = self.hill_map(idxs0 * reg_lng)  # i32 (c,p,d)
        Kms = self.km_map(idxs1)  # f32 (c,p,d)
        signs = self.sign_map(idxs2)  # i32 (c,p,d)

        # idx3 is a 2-codon index used for vectors (n=4096 (- stop codons))
        reacts = self.reaction_map(idxs3 * catal_lng)  # i32 (c,p,d,s)
        trnspts = self.transport_map(idxs3 * trnsp_lng)  # i32 (c,p,d,s)
        effectors = self.effector_map(idxs3 * reg_lng)  # i32 (c,p,d,s)

        # Vmax are averaged over domains
        # undefined Vmax enries are NaN and are ignored by nanmean
        self.Vmax[cell_idxs] = Vmaxs.nanmean(dim=2).nan_to_num(0.0)

        # effector vectors are multiplied with signs and hill coefficients
        # and summed up over domains
        # no Int matmul impl in torch CUDA (dimenion is not shared)
        A = torch.einsum(
            "cpds,cpd->cps", effectors.float(), (signs * Hills).float()
        ).int()

        # Kms from other domains are ignored using NaNs
        # their Kms must be seperated for each signal
        Kmr_d = torch.where(is_reg, Kms, torch.nan)  # (c,p,d)
        Kmr_ds = torch.einsum("cpds,cpd->cpds", effectors, Kmr_d)

        # average Kmrs, ignored unused with nanmean
        Kmr_ds[Kmr_ds == 0.0] = torch.nan  # effectors introduce 0s
        Kmr = Kmr_ds.nanmean(dim=2).nan_to_num(0.0)  # (c,p,s)

        # Kms of regulatory domains are already exponentiated with Hill coefficients
        self.Kmr[cell_idxs] = torch.pow(Kmr, A)
        self.A[cell_idxs] = A

        # reaction stoichiometry N is derived from transporter and catalytic vectors
        # vectors for regulatory domains or emptpy proteins are all 0s
        N_d = torch.einsum("cpds,cpd->cpds", (reacts + trnspts), signs)
        N = N_d.sum(dim=2, dtype=torch.int32)
        self.N[cell_idxs] = N

        # N for forward and backward reactions is distinguished
        # to not loose molecules like co-facors whose net N would become 0
        self.Nf[cell_idxs] = torch.where(N_d < 0, -N_d, 0).sum(dim=2, dtype=torch.int32)
        self.Nb[cell_idxs] = torch.where(N_d > 0, N_d, 0).sum(dim=2, dtype=torch.int32)

        # Kms of catalytic and transporter domains are aggregated
        # Kms from other domains are ignored using NaNs and nanmean
        Kmn = torch.where(~is_reg, Kms, torch.nan).nanmean(dim=2).nan_to_num(0.0)

        # energies define Ke which defines Ke = Kmf/Kmb
        # extreme energies can create Inf or 0.0, avoid them with clamp
        E = torch.einsum("cps,s->cp", N.float(), self.mol_energies)
        Ke = torch.exp(-E / self.abs_temp / GAS_CONSTANT).clamp(_EPS, _MAX)
        self.Ke[cell_idxs] = Ke

        # Km is sampled between a defined range
        # exessively small Km can create numerical instability
        # thus, sampled Km should define the smaller Km of Ke = Kmf/Kmb
        # Ke>=1  => Kmf=Km,         Kmb=Ke*Km
        # Ke<1   => Kmf=Km/Ke,      Kmb=Km
        # this operation can create again Inf or 0.0, avoided with clamp
        # this effectively limits Ke around 1e38
        is_fwd = Ke >= 1.0
        self.Kmf[cell_idxs] = torch.where(is_fwd, Kmn, Kmn / Ke).clamp(_EPS, _MAX)
        self.Kmb[cell_idxs] = torch.where(is_fwd, Kmn * Ke, Kmn).clamp(_EPS, _MAX)

    def unset_cell_params(self, cell_idxs: list[int]):
        """
        Set cell parameters for cells to zero.

        Parameters:
            cell_idxs: Indexes of cells

        Indexes refer to the [World's][magicsoup.world.World] cell indexes.
        """
        self.N[cell_idxs] = 0
        self.Nf[cell_idxs] = 0
        self.Nb[cell_idxs] = 0
        self.A[cell_idxs] = 0
        self.Ke[cell_idxs] = 0.0
        self.Kmf[cell_idxs] = 0.0
        self.Kmb[cell_idxs] = 0.0
        self.Kmr[cell_idxs] = 0.0
        self.Vmax[cell_idxs] = 0.0

    def copy_cell_params(self, from_idxs: list[int], to_idxs: list[int]):
        """
        Copy paremeters from a list of cells to another list of cells

        Parameters:
            from_idxs: List of cell indexes to copy from
            to_idxs: List of cell indexes to copy to

        `from_idxs` and `to_idxs` must have the same length.
        Indexes refer to the [World's][magicsoup.world.World] cell indexes.
        """
        self.Ke[to_idxs] = self.Ke[from_idxs]
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

        Parameters:
            keep: Bool tensor (c,) which is true for every cell that should not be removed
                and false for every cell that should be removed.

        `keep` must have length `world.n_cells`.
        Indexes of `keep` refer to the [World's][magicsoup.world.World] cell indexes.
        """
        self.Ke = self.Ke[keep]
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

        Parameters:
            by_n: By how many rows to increase the cell dimension
        """
        self.Ke = self._expand_c(t=self.Ke, n=by_n)
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

        Parameters:
            max_n: The maximum number of rows required in the protein dimension
        """
        n_prots = int(self.N.size(1))
        if max_n > n_prots:
            by_n = max_n - n_prots
            self.Ke = self._expand_p(t=self.Ke, n=by_n)
            self.Kmf = self._expand_p(t=self.Kmf, n=by_n)
            self.Kmb = self._expand_p(t=self.Kmb, n=by_n)
            self.Kmr = self._expand_p(t=self.Kmr, n=by_n)
            self.Vmax = self._expand_p(t=self.Vmax, n=by_n)
            self.N = self._expand_p(t=self.N, n=by_n)
            self.Nf = self._expand_p(t=self.Nf, n=by_n)
            self.Nb = self._expand_p(t=self.Nb, n=by_n)
            self.A = self._expand_p(t=self.A, n=by_n)

    def integrate_signals(self, X: torch.Tensor) -> torch.Tensor:
        """
        Simulate protein work by integrating all signals.

        Parameters:
            X: 2D tensor of every signal in every cell (c,s). Must all be >= 0.0.

        Returns:
            New tensor of the same shape which represents the updated signals for every cell.

        Dimension 0 indexes of `X` refer to the [World's][magicsoup.world.World] cell indexes.
        Dimension 1 indexes of `X` refer to [Molecule][magicsoup.containers.Molecule] indexes
        on [Chemistry][magicsoup.containers.Chemistry].
        The order of signals is first all intracellular molecule species in the same order as `chemistry.molecules`,
        then again all molecule species in the same order but this time describing extracellular molecule species.
        The number of intracellular molecules comes from `world.cell_molecules` for any particular cell.
        The number of extracellular molecules comes from `world.molecule_map` from the pixel the particular cell currently lives on.
        """
        # calculate in multiple parts while reducing velocity
        # trim factors get increasingly smaller to make it more likely to approach the
        # reaction equilibrium if it was overshot in the previous part
        trim_factors = (0.7, 0.2, 0.1)
        for trim in trim_factors:
            X = self._integrate_signals_part(
                adj_vmax=(self.Vmax * trim).clamp(0.0), X0=X
            )
        return X

    def _integrate_signals_part(
        self, adj_vmax: torch.Tensor, X0: torch.Tensor
    ) -> torch.Tensor:
        V = self._get_velocities(X=X0, Vmax=adj_vmax)  # (c,p)

        # NV to keep synthesis and deconstruction separate
        NV = torch.einsum("cps,cp->cps", self.N, V)  # (c,p,s)

        # adjust NV downward to avoid negative concentrations
        NV_adj = self._get_negative_adjusted_nv(NV=NV, X=X0)  # (c,p,s)
        X1 = X0 + NV_adj.sum(1)  # (c,s)
        X1[X1 < 0.0] = 0.0  # small floating point errors can lead to -1e-7

        # adjust X downward to avoid Q overshooting Ke
        X1_adj = self._get_equilibrium_adjusted_x(X0=X0, X1=X1, NV=NV_adj, V=V)

        return X1_adj

    def _get_velocities(self, X: torch.Tensor, Vmax: torch.Tensor) -> torch.Tensor:
        # catalytic activity X^N / Km
        # signals are aggregated for forward and backward reaction
        # proteins that had no involved catalytic region should not be active
        kf, f_prots = self._multiply_signals(X=X, N=self.Nf)  # (c,p)
        kf /= self.Kmf
        kf[~f_prots] = 0.0  # non-involved proteins are not active
        kf[kf.isinf()] = _MAX  # possibly Inf after Km division

        kb, b_prots = self._multiply_signals(X=X, N=self.Nb)  # (c,p)
        kb /= self.Kmb
        kb[~b_prots] = 0.0  # non-involved proteins are not active
        kb[kb.isinf()] = _MAX  # possibly Inf after Km division

        # reversible MM equation
        a_cat = (kf - kb) / (1 + kf + kb)  # (c,p)

        # allosteric modulation X^A / (X^A + K^A)
        # A=1 => a=x/(x+k) is activating [0;1]
        # A=-1 => a=k/(k+x)=1-x/(x+k) is deactivating (0;1] (X=0 is undefined)
        # A=0 => a=1/2, so they have to be set back to 1.0
        # here each signal is considered with seperate Km
        is_reg = self.A != 0
        x_reg = torch.einsum("cps,cs->cps", is_reg.float(), X)
        a_reg_s = torch.pow(x_reg, self.A)  # (c,p,s)
        a_reg_s = a_reg_s / (a_reg_s + self.Kmr)  # Kmr already has power
        a_reg_s[a_reg_s.isnan()] = 1.0  # A<0,X=0 => inhibitor not present, so active
        a_reg_s[~is_reg] = 1.0  # set all uninvolved to 1
        a_reg = torch.prod(a_reg_s, 2)  # (c,p)
        a_reg[a_reg.isinf()] = _MAX  # possibly Inf divisions, multiplications

        # velocity from activities
        V = a_cat * Vmax * a_reg  # (c,p)

        # activities close to Inf could create Inf in multiplication
        return V.clamp(_MIN, _MAX)

    def _get_equilibrium_adjusted_x(
        self,
        X0: torch.Tensor,
        X1: torch.Tensor,
        NV: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        # adjust NV downwards so Q does not overshoot Ke
        # must calculate X1 and Q1 to do adjustments, X1 is returned
        # all proteins influence each others Q, so iterative heuristic
        # increments to an adjustment factor in steps
        increments = (0.5, 0.25, 0.125, 0.0625)

        # stop adjusting if Ke ~= Q
        upper_thresh = 1.5
        lower_thresh = 1 / 1.5

        has_impact = V.abs() > 0.1
        is_fwd = V > 0.0

        # running factors for adjusting NV
        F = torch.ones_like(V)  # (c,p)
        for increment in increments:
            # calculate expected quotient Q1 and compare to Ke (t1)
            Q1 = self._get_quotient(X=X1)  # (c,p)
            QKe = Q1 / self.Ke

            # fwd reaction: Q -> Ke from below, QKe > 1 is overshoot
            # bwd reaction: Q -> Ke from above, QKe < 1 is overshoot
            # reactions cant be adjusted higher than 1.0

            v_too_low = torch.where(is_fwd, QKe < lower_thresh, QKe > upper_thresh)
            v_too_low[is_fwd & (F == 1.0)] = False

            v_too_high = torch.where(is_fwd, QKe > upper_thresh, QKe < lower_thresh)
            v_too_high[~is_fwd & (F == 0.0)] = False

            # mask for which velocities can be adjusted
            if not torch.any((v_too_low | v_too_high) & has_impact):
                return X1

            # adjust F upwards/downwards
            F[v_too_high] -= increment
            F[v_too_low] += increment
            F[F > 1.0] = 1.0
            F[F < 0.0] = 0.0

            # calculate new X1
            X1 = X0 + torch.einsum("cps,cp->cs", NV, F)  # (c,s)
            X1[X1 < 0.0] = 0.0

        return X1

    def _get_negative_adjusted_nv(
        self, NV: torch.Tensor, X: torch.Tensor
    ) -> torch.Tensor:
        # how does each protein have to be slowed down for each signal
        # so not more signal than available is removed
        F = X / (-NV).clamp(min=0.0).sum(1)  # (c,s)
        F[F > 1.0] = 1.0

        # which signals are removed by each protein
        M_rm = NV < 0.0  # (c,p,s)

        # factor for slowing down each protein for each signal
        F_prots = torch.einsum("cps,cs->cps", M_rm.float(), F)
        F_prots[~M_rm] = 1.0  # (c,p,s)

        # how much can each protein be active at most
        F_min = F_prots.min(dim=2).values  # (c,p)

        return torch.einsum("cps,cp->cps", NV, F_min)

    def _get_quotient(self, X: torch.Tensor) -> torch.Tensor:
        xx_prod, prod_prots = self._multiply_signals(X=X, N=self.Nb)
        xx_prod[~prod_prots] = 0.0
        xx_prod[xx_prod.isinf()] = _MAX

        xx_subs, subs_prots = self._multiply_signals(X=X, N=self.Nf)
        xx_subs[~subs_prots] = 0.0
        xx_subs[xx_subs.isinf()] = _MAX

        # note: x/0=Inf, 0/x=0, 0/0=nan
        # 0.0 and Inf is avoided as it is in Ke calculation
        return (xx_prod / xx_subs).clamp(min=_EPS, max=_MAX).nan_to_num(1.0)

    def _multiply_signals(
        self, X: torch.Tensor, N: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # calculate torch.prod(torch.pow(x, n), 2)
        # consider:
        # - some proteins are not involved at all (their Ns are all 0)
        # - some are involved but the signal 0 (required X is 0)
        # - 0^n=0 but x^0=1
        M = N > 0

        # expand to p while keeping uninvolved 0
        x = torch.einsum("cps,cs->cps", M, X)

        # stray 1s created here
        xx = torch.prod(torch.pow(x, N), 2)

        # Infs could have been created, MAX is still >e2 away from Inf
        # also I'm not sure how but somehow small negative values
        # (<1e-7) can be produced (I suspect floating point errors)
        xx[xx.isnan()] = 0.0
        xx[xx < 0.0] = 0.0
        xx[xx.isinf()] = _MAX

        # mask (c,p) which proteins are active must be returned
        return xx, M.sum(2, dtype=torch.int32) > 0

    def _collect_proteome_idxs(
        self,
        proteomes: list[list[ProteinSpecType]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_prots = self.N.size(1)
        n_doms = max(len(dd[0]) for d in proteomes for dd in d)
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
            for doms, *_ in proteins:
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

        dom_types = self._i32_tensor(c_dts)  # (c,p,d)
        idxs0 = self._i32_tensor(c_idxs0)  # (c,p,d)
        idxs1 = self._i32_tensor(c_idxs1)  # (c,p,d)
        idxs2 = self._i32_tensor(c_idxs2)  # (c,p,d)
        idxs3 = self._i32_tensor(c_idxs3)  # (c,p,d)
        return dom_types, idxs0, idxs1, idxs2, idxs3

    def _expand_c(self, t: torch.Tensor, n: int) -> torch.Tensor:
        size = t.size()
        zeros = torch.zeros(n, *size[1:], device=self.device, dtype=t.dtype)
        return torch.cat([t, zeros], dim=0)

    def _expand_p(self, t: torch.Tensor, n: int) -> torch.Tensor:
        size = t.size()
        zeros = torch.zeros(size[0], n, *size[2:], device=self.device, dtype=t.dtype)
        return torch.cat([t, zeros], dim=1)

    def _zeros_i32_tensor(self, *args) -> torch.Tensor:
        return torch.zeros(*args, device=self.device, dtype=torch.int32)

    def _zeros_f32_tensor(self, *args) -> torch.Tensor:
        return torch.zeros(*args, device=self.device, dtype=torch.float32)

    def _i32_tensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=torch.int32)

    def _f32_tensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=torch.float32)
