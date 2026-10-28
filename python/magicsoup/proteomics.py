import math
import random
from collections import defaultdict
from statistics import NormalDist
from typing import TYPE_CHECKING, TypedDict

import torch

from magicsoup import rs
from magicsoup.biology import Protein
from magicsoup.chemistry import Chemistry, Molecule, Reaction
from magicsoup.config import GAS_CONSTANT, ROOM_TEMP
from magicsoup.util import TensorClass, TensorKwargs

if TYPE_CHECKING:
    from magicsoup.genomics import ProteinSpec


class _LogNormMap:
    """
    Get a map that maps tokens to floats
    which is sampled from a log normal distribution.
    """

    def __init__(self, max_token: int, value_range: tuple[float, float]) -> None:
        lo, hi = value_range
        z95 = NormalDist().inv_cdf(0.975)
        mu = (math.log(lo) + math.log(hi)) / 2
        sig = (math.log(hi) - math.log(lo)) / (2 * z95)

        self.token_2_scalar: dict[int, float] = {}
        self.scalar_2_tokens: dict[float | None, list[int]] = defaultdict(list)

        # none value on 0 token
        self.token_2_scalar[0] = 0.0
        self.scalar_2_tokens[None] = [0]

        # assign tokens
        for token in range(1, max_token + 1):
            value = random.lognormvariate(mu, sig)
            self.token_2_scalar[token] = value
            self.scalar_2_tokens[value].append(token)


class _MirrGeoMap:
    """
    Get a map that maps tokens to ints
    which are sampled from a geometric distribution with p=1/2, p0=0 mirrored around 0.
    """

    def __init__(self, max_token: int, max_value: int) -> None:
        n = 1
        pop: list[int] = []
        for d in range(max_value, 0, -1):
            pop.extend(n * [d])
            pop.extend(n * [-d])
            n *= 2

        self.token_2_scalar: dict[int, int] = {}
        self.scalar_2_tokens: dict[int | None, list[int]] = defaultdict(list)

        # none value on 0 token
        self.token_2_scalar[0] = 0
        self.scalar_2_tokens[None] = [0]

        # assign tokens
        for token in range(1, max_token + 1):
            value = random.choice(pop)
            self.token_2_scalar[token] = value
            self.scalar_2_tokens[value].append(token)


class _ReactionMap:
    """
    Get a map that maps tokens to a list of vectors.
    Each vector will be mapped with the same frequency.
    Each vector has number of molecules length and represents the
    stoichiometry of a reaction.
    """

    def __init__(self, chemistry: Chemistry, max_token: int) -> None:
        r = len(chemistry.reactions)
        m = len(chemistry.molecules)
        m2 = 2 * m
        if r > max_token:
            raise ValueError(
                f"There are max_token={max_token} and {r} reactions."
                " It is not possible to create an enzyme for all reactions"
            )

        self.token_2_vector: dict[int, tuple[int, ...]] = {}
        self.react_2_tokens: dict[Reaction | None, list[int]] = defaultdict(list)
        if len(chemistry.reactions) == 0:
            return

        # only copy [0] to avoid having references to the same list
        vectors = [[0] * m2 for _ in range(r)]
        mol_2_idx = chemistry.mol_2_idx
        ri_2_react: dict[int, Reaction] = {}
        for ri, react in enumerate(chemistry.reactions):
            ri_2_react[ri] = react
            for mol in react.substrates:
                mi = mol_2_idx[mol]
                vectors[ri][mi] -= 1
            for mol in react.products:
                mi = mol_2_idx[mol]
                vectors[ri][mi] += 1

        # none vector on 0 token
        none_vector = [0] * m2
        self.token_2_vector[0] = tuple(none_vector)
        self.react_2_tokens[None] = [0]

        # assign tokens round-robin
        for token in range(1, max_token + 1):
            ri = token % r
            self.token_2_vector[token] = tuple(vectors[ri])
            self.react_2_tokens[ri_2_react[ri]].append(token)


# TODO: Co-factors are regarded in kinetics, but not in token mapping
#       tokens map to vectors which are rows in N (so Co-factors are 0)
#       Should separate forward and reverse


class _TransporterMap:
    """
    Get a map that maps tokens to a list of vectors.
    Each vector has number of molecules length and represents the
    stoichiometry of a molecule transport into or out of the cell.
    """

    def __init__(self, chemistry: Chemistry, max_token: int) -> None:
        m = len(chemistry.molecules)
        m2 = m * 2
        if m > max_token:
            raise ValueError(
                f"There are max_token={max_token} and {m} molecules."
                " It is not possible to create a transporter for all molecules"
            )

        self.token_2_vector: dict[int, tuple[int, ...]] = {}
        self.mol_2_tokens: dict[Molecule | None, list[int]] = defaultdict(list)
        idx_2_mol = chemistry.idx_2_mol

        # only copy [0] to avoid having references to the same list
        vectors = [[0] * m2 for _ in range(m)]
        for mi in range(m):
            vectors[mi][mi] = -1
            vectors[mi][mi + m] = 1

        # none vector on 0 token
        none_vector = [0] * m2
        self.token_2_vector[0] = tuple(none_vector)
        self.mol_2_tokens[None] = [0]

        # assign tokens round-robin
        for token in range(1, max_token + 1):
            mi = token % m
            self.token_2_vector[token] = tuple(vectors[mi])
            self.mol_2_tokens[idx_2_mol[mi]].append(token)


class _EffectorMap:
    """
    Get a map that maps tokens to a list of vectors.
    Each vector has number of molecules length and represents the
    either activating (+1) or inhibiting (-1) effect
    of an effector molecule.
    """

    def __init__(self, chemistry: Chemistry, max_token: int) -> None:
        m = len(chemistry.molecules)
        m2 = m * 2
        if m2 > max_token:
            raise ValueError(
                f"There are max_token={max_token} and {m} molecules."
                " It is not possible to create a receptor for all molecules"
            )

        self.token_2_vector: dict[int, tuple[int, ...]] = {}
        self.mol_2_tokens: dict[Molecule | None, list[int]] = defaultdict(list)
        idx_2_mol = chemistry.idx_2_mol

        #  only copy [0] to avoid having references to the same list
        vectors = [[0] * m2 for _ in range(m2)]
        for mi in range(m):
            vectors[mi][mi] = 1
            vectors[mi + m][mi + m] = 1

        # none vector on 0 token
        none_vector = [0] * m2
        self.token_2_vector[0] = tuple(none_vector)
        self.mol_2_tokens[None] = [0]

        # assign tokens round-robin
        for token in range(1, max_token + 1):
            self.token_2_vector[token] = tuple(vectors[token % m2])
            self.mol_2_tokens[idx_2_mol[token % m]].append(token)


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
        hill_max: int = 3,
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
        n_mols = len(molecules)
        self.m = 2 * n_mols

        # cell-volume-to-voxel-volume ratios for ext mols (2m,)
        self.cv_ratio = self.ftensor([1] * n_mols + [cv_ratio] * n_mols)

        # energies (kJ/mol -> J/mol) for all mols (2m,)
        self.mol_energies = self.ftensor([d.energy for d in molecules] * 2) * 1e3

        self.km_map = _LogNormMap(max_token=scalar_enc_size, value_range=km_range)
        self.vmax_map = _LogNormMap(max_token=scalar_enc_size, value_range=vmax_range)
        self.sign_map = _MirrGeoMap(max_token=scalar_enc_size, max_value=1)
        self.hill_map = _MirrGeoMap(max_token=scalar_enc_size, max_value=hill_max)
        self.transport_map = _TransporterMap(chemistry, max_token=vector_enc_size)
        self.effector_map = _EffectorMap(chemistry, max_token=vector_enc_size)
        self.reaction_map = _ReactionMap(chemistry, max_token=vector_enc_size)

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
