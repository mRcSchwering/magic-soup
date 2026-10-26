import math
import random
from collections import defaultdict
from typing import TYPE_CHECKING, TypedDict

import torch

from magicsoup import rs
from magicsoup.biology import Protein
from magicsoup.chemistry import Chemistry, Molecule, Reaction
from magicsoup.constants import GAS_CONSTANT, ROOM_TEMP
from magicsoup.util import TensorClass, TensorKwargs

if TYPE_CHECKING:
    from magicsoup.genomics import ProteinSpec


def _get_vector_map(
    max_token: int, vectors: list[list[int]], none_value: int = 0
) -> dict[int, tuple[int, ...]]:
    """
    Get a map that maps tokens to a list of vectors.
    Each vector will be mapped with the same frequency.
    """
    n_vectors = len(vectors)
    if n_vectors == 0:
        # TODO: should also work without reactions
        # raise ValueError("No vectors provided for mapping.")
        return {}

    if n_vectors > max_token:
        raise ValueError(
            f"There are max_token={max_token} and {n_vectors} vectors."
            " It is not possible to map all vectors"
        )

    vec_len = len(vectors[0])
    if not all(len(d) == vec_len for d in vectors):
        raise ValueError("Not all vectors have the same length")

    for vector in vectors:
        if all(d == 0 for d in vector):
            raise ValueError(
                "At least one vector includes only zeros."
                " Each vector should contain at least one non-zero value."
            )

    idxs = random.choices(list(range(n_vectors)), k=max_token)
    none_vector = tuple([none_value] * vec_len)
    all_vectors = [none_vector] + [tuple(vectors[idx]) for idx in idxs]
    return {i: v for i, v in enumerate(all_vectors)}


def _get_inverse[T](m: dict[int, T]) -> dict[T, list[int]]:
    inv = defaultdict(list)
    for k, v in m.items():
        inv[v].append(k)
    return inv


class _LogNormMap:
    """
    Get a map that maps tokens to a float
    which is sampled from a log normal distribution.
    """

    def __init__(
        self, max_token: int, weight_range: tuple[float, float], none_value: float = 0.0
    ) -> None:

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

        weights = [none_value] + non_zero_weights
        self.token_2_scalar = {i: w for i, w in enumerate(weights)}
        self.scalar_2_tokens = _get_inverse(self.token_2_scalar)


class _SignMap:
    """
    Get a map that maps tokens to 1 or -1
    with 50% probability of each being mapped.
    """

    def __init__(self, max_token: int, none_value: int = 0) -> None:
        choices = [1, -1]
        signs = [none_value] + random.choices(choices, k=max_token)
        self.token_2_scalar = {i: s for i, s in enumerate(signs)}
        self.scalar_2_tokens = _get_inverse(self.token_2_scalar)


class _HillMap:
    """
    Get a map that maps tokens to 1, 2, 3, 4, 5
    with chances 52%, 26%, 13%, 6%, 3% respectively.
    """

    def __init__(self, max_token: int, none_value: int = 0) -> None:
        choices = [5] + 2 * [4] + 4 * [3] + 8 * [2] + 16 * [1]
        numbers = [none_value] + random.choices(choices, k=max_token)
        self.token_2_scalar = {i: n for i, n in enumerate(numbers)}
        self.scalar_2_tokens = _get_inverse(self.token_2_scalar)


# TODO: test these maps (also with no reactions/molecules)


class _ReactionMap:
    """
    Get a map that maps tokens to a list of vectors.
    Each vector will be mapped with the same frequency.
    Each vector has number of molecules length and represents the
    stoichiometry of a reaction.
    """

    def __init__(
        self,
        molmap: dict[Molecule, int],
        reactions: list[Reaction],
        max_token: int,
        none_value: int = 0,
    ) -> None:
        n_mols = 2 * len(molmap)

        # careful, only copy [0] to avoid having references to the same list
        vectors = [[0] * n_mols for _ in range(len(reactions))]
        for ri, react in enumerate(reactions):
            for mol in react.substrates:
                mol_i = molmap[mol]
                vectors[ri][mol_i] -= 1
            for mol in react.products:
                mol_i = molmap[mol]
                vectors[ri][mol_i] += 1

        self.token_2_vector = _get_vector_map(
            max_token=max_token, vectors=vectors, none_value=none_value
        )

        vector_2_tokens = _get_inverse(self.token_2_vector)
        molmap_i = {i: m for m, i in molmap.items()}

        self.react_2_tokens: dict[Reaction, list[int]] = {}
        for vec, tokens in vector_2_tokens.items():
            subs = [dd for i, n in enumerate(vec) if n < 0 for dd in [molmap_i[i]] * -n]
            prods = [dd for i, n in enumerate(vec) if n > 0 for dd in [molmap_i[i]] * n]
            self.react_2_tokens[Reaction(substrates=subs, products=prods)] = tokens


# TODO: Co-factors are regarded in kinetics, but not in token mapping
#       tokens map to vectors which are rows in N (so Co-factors are 0)
#       Should separate forward and reverse

# TODO: I always need 2 mapping:
#       1. token -> (vector, scalar) for translation
#       2. (scalar, Molecule, Reaction) -> token for synthesis


class _TransporterMap:
    """
    Get a map that maps tokens to a list of vectors.
    Each vector has number of molecules length and represents the
    stoichiometry of a molecule transport into or out of the cell.
    """

    def __init__(
        self,
        molmap: dict[Molecule, int],
        max_token: int,
        none_value: int = 0,
    ) -> None:

        n_mols = len(molmap)

        # careful, only copy [0] to avoid having references to the same list
        vectors = [[0] * n_mols * 2 for _ in range(n_mols)]
        for mi in range(n_mols):
            vectors[mi][mi] = -1
            vectors[mi][mi + n_mols] = 1

        self.token_2_vector = _get_vector_map(
            max_token=max_token, vectors=vectors, none_value=none_value
        )

        vector_2_tokens = _get_inverse(self.token_2_vector)
        molmap_i = {i: m for m, i in molmap.items()}

        # all vectors have 1/-1 for intra- and extra cellular molecules
        self.mol_2_tokens: dict[Molecule, list[int]] = {}
        for vec, tokens in vector_2_tokens.items():
            non_zeros = [i for i, d in enumerate(vec) if d != 0]
            if len(non_zeros) == 0:
                continue

            self.mol_2_tokens[molmap_i[non_zeros[0]]] = tokens


class _EffectorMap:
    """
    Get a map that maps tokens to a list of vectors.
    Each vector has number of molecules length and represents the
    either activating (+1) or inhibiting (-1) effect
    of an effector molecule.
    """

    def __init__(
        self,
        molmap: dict[Molecule, int],
        max_token: int,
        none_value: int = 0,
    ) -> None:

        m = len(molmap) * 2

        # careful, only copy [0] to avoid having references to the same list
        vectors = [[0] * m for _ in range(m)]
        for mi in range(m):
            vectors[mi][mi] = 1

        self.token_2_vector = _get_vector_map(
            max_token=max_token, vectors=vectors, none_value=none_value
        )

        vector_2_tokens = _get_inverse(self.token_2_vector)
        molmap_i = {i: m for m, i in molmap.items()}

        # vectors have a 1 for each mol (intra and extra cellular)
        self.mol_2_tokens: dict[tuple[Molecule, bool], list[int]] = {}
        for vec, tokens in vector_2_tokens.items():
            non_zeros = [i for i, d in enumerate(vec) if d != 0]
            if len(non_zeros) == 0:
                continue

            idx = non_zeros[0]
            is_int = idx < len(molmap)
            mol_i = idx if is_int else idx - len(molmap)
            self.mol_2_tokens[(molmap_i[mol_i], is_int)] = tokens


class ProteomicsKwargs(TypedDict, total=False):
    abs_temp: float
    km_range: tuple[float, float]
    vmax_range: tuple[float, float]
    max_k: float
    eps: float


class Proteomics(TensorClass):
    """
    Manage parameters that make up proteomes of all cells

    Arguments:
        chemistry: simulation [Chemistry][magicsoup.chemistry.Chemistry]
        cv_ratio: ratio of cell volume to surrounding voxel volume
        abs_temp: temperature in K (Kelvin)
        km_range: Sampling range for enzyme affinity constants in mM (milli molar)
        vmax_range: Sampling range for enzyme maximum velocities in mM/s (milli molar per second)
            assuming 1 µM (micro molar) protein concentration
        scalar_enc_size: Number of unique codons encoding scalar protein domain specifications
        vector_enc_size: Number of unique codons encoding vector protein domain specifications
        max_k: Upper bound for k_f and k_b (effectively limiting equilibrium constant K_e)
        eps: Smallest possible non-zero positive value
        tensor_kwargs: Kwargs passed to [TensorClass][magicsoup.util.TensorClass]

    Constant intracellular protein concentration of 1 µM (micro molar) is assumed for each protein.
    Affinities and maximum velocities are sampled from log-normal distributions,
    all others from uniform distributions.

    During the simulation all proteomes are represented as tensors of dimensions `c`, `p`, `2m`.
    Tensor dimension `c` describes each cell, `p` each protein, `m` each molecule species.
    In tensors intracellular molecules are represented first, then extracellular ones, hence `2m`.
    To get a more interpretable representation of proteomes use [get_proteome()][.get_proteome].
    """

    def __init__(
        self,
        chemistry: Chemistry,
        cv_ratio: float = 1e-3,
        abs_temp: float = ROOM_TEMP,  # in K
        km_range: tuple[float, float] = (1e-3, 10.0),  # in mM
        vmax_range: tuple[float, float] = (1e-3, 1e4),  # in mM/s
        scalar_enc_size: int = 64 - 3,  # 1 codon w/o stop
        vector_enc_size: int = 4096 - 3 * 64,  # 2 codons w/o stop in 1st
        max_k: float = 1e36,
        eps: float = 1e-40,
        tensor_kwargs: TensorKwargs | None = None,
    ) -> None:
        super().__init__(**(tensor_kwargs or {}))

        self.abs_temp = abs_temp
        self.max_k = max_k
        self.eps = eps
        self.chemistry = chemistry

        molecules = chemistry.molecules
        reactions = chemistry.reactions
        n_mols = len(molecules)
        self.m = 2 * n_mols
        mol_2_mi = {d: i for i, d in enumerate(molecules)}

        # cell-volume-to-voxel-volume ratios for ext mols (2m,)
        self.cv_ratio = self.ftensor([1] * n_mols + [cv_ratio] * n_mols)

        # energies (kJ/mol -> J/mol) for all mols (2m,)
        self.mol_energies = self.ftensor([d.energy for d in molecules] * 2) * 1e3

        # TODO: Km range: log normal
        #       median at 0.1 mM, 0.003 mM 5th pctile, 3 mM 95th pctile
        # TODO: Vmax range: log-normal sigma=10
        #       median 1e-2 mM/s, 1e-4 mM/s 5th pctile, 1 mM/s 95th pctile
        self.km_map = _LogNormMap(max_token=scalar_enc_size, weight_range=km_range)
        self.vmax_map = _LogNormMap(max_token=scalar_enc_size, weight_range=vmax_range)
        self.sign_map = _SignMap(max_token=scalar_enc_size)
        self.hill_map = _HillMap(max_token=scalar_enc_size)
        self.transport_map = _TransporterMap(molmap=mol_2_mi, max_token=vector_enc_size)
        self.effector_map = _EffectorMap(molmap=mol_2_mi, max_token=vector_enc_size)
        self.reaction_map = _ReactionMap(
            reactions=reactions, molmap=mol_2_mi, max_token=vector_enc_size
        )

        # init rust class
        self._setup_rs()

    def get_proteome(self, proteome: "list[ProteinSpec]") -> list[Protein]:
        """
        Translate proteome of a single cell

        Arguments:
            proteome: list of proteins which should be translated and returned

        Retruns:
            List of [Proteins][magicsoup.containers.Protein] that describe
            the cell's proteome.

        The returned list of proteins is purely for inspecting a cell.
        The simulation uses [get_cell_params()][.get_cell_params].
        """
        proteome_kwargs = self.rs.get_proteome_dict(
            proteome=proteome,
            molecules=[d.name for d in self.chemistry.molecules],
        )
        return [
            Protein.from_dict(d, molmap=self.chemistry.molname_2_mol)
            for d in proteome_kwargs
        ]

    def get_cell_params(
        self, proteomes: "list[list[ProteinSpec]]"
    ) -> dict[str, torch.Tensor]:
        """
        Translate and return cell parameters for many proteomes

        Arguments:
            proteomes: nested list of proteome specifications for each cell
                where each proteome specification is a list of protein specifications

        Returns:
            Dict of all tensors necessary to describe cell proteomes.

        This function is used by the simulation.
        To get a more interpretable representation of proteomes
        use [get_proteome()][.get_proteome].
        """
        v_max_, k_m_, K_r_, N_f_, N_b_, N_h_ = self.rs.get_proteome_params(proteomes)
        v_max = self.ftensor(v_max_)
        k_m = self.ftensor(k_m_)
        K_r = self.ftensor(K_r_)
        N_f = self.itensor(N_f_)
        N_b = self.itensor(N_b_)
        N_h = self.itensor(N_h_)

        N = self.derive_stoichiometry(N_f=N_f, N_b=N_b)
        k_f, k_b = self.derive_rates(N=N, k_m=k_m)
        N_e = self.derive_effective_stoichiometry(N=N)

        return {
            "N_e": N_e,
            "N_f": N_f,
            "N_b": N_b,
            "N_h": N_h,
            "k_f": k_f,
            "k_b": k_b,
            "K_r": K_r,
            "v_max": v_max,
        }

    def derive_stoichiometry(
        self, N_f: torch.Tensor, N_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Get stoichiometric numbers from stoichiometric coeffcients.

        Arguments:
            N_f: (c,p,2m) integer tensor of stoichiometric coefficients of substrates
            N_b: (c,p,2m) integer tensor of stoichiometric coefficients of products

        Returns:
            (c,p,2m) float tensor of stoichiometric numbers

        Signs of stoichiometric numbers and coefficients are viewed from forward
        direction as of protein specifications.
        """
        # N_b and N_f is kept separate for MM-kinetics
        # otherwise co-factors (with N=0) would be ignored
        return (N_b - N_f).to(self.ftype)  # (c,p,2m)

    def derive_rates(
        self, N: torch.Tensor, k_m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get reaction rates from stoichiometric numbers and affinities

        Arguments:
            N: (c, p, m) float tensor of stoichiometric numbers
            k_m: (c, p) float tensor of reaction affinities

        Returns:
            Tuple (k_f, k_b) each a (c,p) float tensor describing reaction rates.
        """
        eps = self.eps
        max_k = self.max_k

        # TODO: N could be an empty tensor
        #       in that case einsum cpm,m->cp below fails
        #       should find the right place to protect against empty tensors
        #       more likely to be triggered when spawning 1 cell

        # energies define k_e which defines k_e = k_f/k_b
        e = torch.einsum("cpm,m->cp", N, self.mol_energies)
        k_e = torch.exp(-e / self.abs_temp / GAS_CONSTANT).clamp(eps, max_k)

        # TODO: come up with a better way of deriving kf/kb from km
        #       basically I am limiting km

        # Km is sampled between a defined range
        # exessively small Km can create numerical instability
        # thus, sampled Km should define the smaller Km of k_e = k_f/k_b
        # k_e>=1  => k_f=Km,         k_b=k_e*Km
        # k_e<1   => k_f=Km/k_e,      k_b=Km
        # this operation can create again Inf or 0.0, avoided with clamp, limits K_e
        is_fwd = k_e >= 1.0
        k_f = torch.where(is_fwd, k_m, k_m / k_e)  # .clamp(max=max_k)
        k_b = torch.where(is_fwd, k_m * k_e, k_m)  # .clamp(max=max_k)
        return k_f, k_b

    def derive_effective_stoichiometry(self, N: torch.Tensor) -> torch.Tensor:
        """
        Get effective stoichiometric numbers from stoichiometric numbers.

        Arguments:
            N: (c,p,2m) float tensor of stoichiometric numbers

        Returns:
            (c,p,2m) float tensor of effective stoichiometric numbers

        Effective stoichiometric numbers means stoichiometric numbers adjusted to volume differences between cell and voxel.
        """
        # effective stoichiometry considering volume difference of cell and medium
        return N * self.cv_ratio

    def _setup_rs(self) -> None:
        self.rs = rs.Proteomics(
            km_map=self.km_map.token_2_scalar,
            vmax_map=self.vmax_map.token_2_scalar,
            sign_map=self.sign_map.token_2_scalar,
            hill_map=self.hill_map.token_2_scalar,
            reaction_map=self.reaction_map.token_2_vector,
            transporter_map=self.transport_map.token_2_vector,
            effector_map=self.effector_map.token_2_vector,
            m=self.m,
        )
