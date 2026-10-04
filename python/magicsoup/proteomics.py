import math
import random
from collections import defaultdict

import torch

from magicsoup import rs
from magicsoup.util import TensorClass

from .cellular import Protein
from .chemistry import Chemistry, Molecule
from .constants import GAS_CONSTANT, ProteinSpecType


def _get_hill_map(max_token: int, none_value: int = 0) -> dict[int, int]:
    """
    Get a map that maps tokens to 1, 2, 3, 4, 5
    with chances 52%, 26%, 13%, 6%, 3% respectively.
    """
    choices = [5] + 2 * [4] + 4 * [3] + 8 * [2] + 16 * [1]
    numbers = [none_value] + random.choices(choices, k=max_token)
    return {i: n for i, n in enumerate(numbers)}


def _get_log_norm_weight_map(
    max_token: int, weight_range: tuple[float, float], none_value: float = 0.0
) -> dict[int, float]:
    """
    Get a map that maps tokens to a float
    which is sampled from a log normal distribution.
    """
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
    return {i: w for i, w in enumerate(weights)}


def _get_sign_map(max_token: int, none_value: int = 0) -> dict[int, int]:
    """
    Get a map that maps tokens to 1 or -1
    with 50% probability of each being mapped.
    """
    choices = [1, -1]
    signs = [none_value] + random.choices(choices, k=max_token)
    return {i: s for i, s in enumerate(signs)}


def _get_vector_map(
    max_token: int, vectors: list[list[int]], none_value: int = 0
) -> dict[int, tuple[int, ...]]:
    """
    Get a map that maps tokens to a list of vectors.
    Each vector will be mapped with the same frequency.
    """
    n_vectors = len(vectors)
    if n_vectors == 0:
        raise ValueError("No vectors provided for mapping.")

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


def _get_reaction_map(
    molmap: dict[Molecule, int],
    reactions: list[tuple[list[Molecule], list[Molecule]]],
    max_token: int,
    none_value: int = 0,
) -> dict[int, tuple[int, ...]]:
    """
    Get a map that maps tokens to a list of vectors.
    Each vector will be mapped with the same frequency.
    Each vector has number of molecules length and represents the
    stoichiometry of a reaction.
    """
    n_mols = 2 * len(molmap)

    # careful, only copy [0] to avoid having references to the same list
    vectors = [[0] * n_mols for _ in range(len(reactions))]
    for ri, (lft, rgt) in enumerate(reactions):
        for mol in lft:
            mol_i = molmap[mol]
            vectors[ri][mol_i] -= 1
        for mol in rgt:
            mol_i = molmap[mol]
            vectors[ri][mol_i] += 1

    return _get_vector_map(max_token=max_token, vectors=vectors, none_value=none_value)


def _get_transporter_map(
    n_molecules: int, max_token: int, none_value: int = 0
) -> dict[int, tuple[int, ...]]:
    """
    Get a map that maps tokens to a list of vectors.
    Each vector has number of molecules length and represents the
    stoichiometry of a molecule transport into or out of the cell.
    """
    n_mols = 2 * n_molecules

    # careful, only copy [0] to avoid having references to the same list
    vectors = [[0] * n_mols for _ in range(n_molecules)]
    for mi in range(n_molecules):
        vectors[mi][mi] = -1
        vectors[mi][mi + n_molecules] = 1

    return _get_vector_map(max_token=max_token, vectors=vectors, none_value=none_value)


def _get_regulatory_map(
    n_molecules: int, max_token: int, none_value: int = 0
) -> dict[int, tuple[int, ...]]:
    """
    Get a map that maps tokens to a list of vectors.
    Each vector has number of molecules length and represents the
    either activating (+1) or inhibiting (-1) effect
    of an effector molecule.
    """
    n_mols = 2 * n_molecules

    # careful, only copy [0] to avoid having references to the same list
    vectors = [[0] * n_mols for _ in range(n_mols)]
    for mi in range(n_mols):
        vectors[mi][mi] = 1

    return _get_vector_map(max_token=max_token, vectors=vectors, none_value=none_value)


def _get_inverse[T](m: dict[int, T]) -> dict[T, list[int]]:
    inv = defaultdict(list)
    for k, v in m.items():
        inv[v].append(k)
    return inv


class Proteomics(TensorClass):

    def __init__(
        self,
        chemistry: Chemistry,
        abs_temp: float = 310.0,
        km_range: tuple[float, float] = (1e-2, 100.0),
        vmax_range: tuple[float, float] = (1e-3, 100.0),
        device: str = "cpu",
        itype: torch.dtype = torch.int8,
        ftype: torch.dtype = torch.float32,
        scalar_enc_size: int = 64 - 3,  # 1 codon w/o stop
        vector_enc_size: int = 4096 - 3 * 64,  # 2 codons w/o stop in 1st
        max_k: float = 1e36,
        eps: float = 1e-40,
    ) -> None:
        super().__init__(device=device, itype=itype, ftype=ftype)

        self.abs_temp = abs_temp
        self.max_k = max_k
        self.eps = eps

        self.mol_names = [d.name for d in chemistry.molecules]
        self.mol_energies = self.ftensor([d.energy for d in chemistry.molecules] * 2)

        self.m = 2 * len(chemistry.molecules)
        mol_2_mi = {d: i for i, d in enumerate(chemistry.molecules)}

        self.km_map = _get_log_norm_weight_map(
            max_token=scalar_enc_size, weight_range=km_range
        )
        self.vmax_map = _get_log_norm_weight_map(
            max_token=scalar_enc_size, weight_range=vmax_range
        )
        self.sign_map = _get_sign_map(max_token=scalar_enc_size)
        self.hill_map = _get_hill_map(max_token=scalar_enc_size)
        self.reaction_map = _get_reaction_map(
            reactions=chemistry.reactions, molmap=mol_2_mi, max_token=vector_enc_size
        )
        self.transport_map = _get_transporter_map(
            n_molecules=len(chemistry.molecules), max_token=vector_enc_size
        )
        self.effector_map = _get_regulatory_map(
            n_molecules=len(chemistry.molecules), max_token=vector_enc_size
        )

        # derive inverse maps for genome generation
        self.km_2_idxs = _get_inverse(self.km_map)
        self.vmax_2_idxs = _get_inverse(self.vmax_map)
        self.sign_2_idxs = _get_inverse(self.sign_map)
        self.hill_2_idxs = _get_inverse(self.hill_map)
        self.trnsp_2_idxs = _get_inverse(self.transport_map)
        self.regul_2_idxs = _get_inverse(self.effector_map)
        self.catal_2_idxs = _get_inverse(self.reaction_map)

        # init rust class
        self._setup_rs()

    def _setup_rs(self) -> None:
        self.rs = rs.Proteomics(
            km_map=self.km_map,
            vmax_map=self.vmax_map,
            sign_map=self.sign_map,
            hill_map=self.hill_map,
            reaction_map=self.reaction_map,
            transporter_map=self.transport_map,
            effector_map=self.effector_map,
            m=self.m,
        )

    def get_proteome(self, proteome: list[ProteinSpecType]) -> list[Protein]:
        """
        Translate and return cell parameters for a single proteome

        Parameters:
            proteome: proteome which should be translated and returned

        Retruns:
            List [Proteins][magicsoup.containers.Protein] that describe
            the cell's proteome.
        """
        proteome_kwargs = self.rs.get_proteome_dict(
            proteome=proteome, molecules=self.mol_names
        )
        return [Protein.from_dict(d) for d in proteome_kwargs]

    def get_cell_params(
        self, proteomes: list[list[ProteinSpecType]]
    ) -> dict[str, torch.Tensor]:
        v_max_, k_m_, K_r_, N_f_, N_b_, N_h_ = self.rs.get_proteome_params(proteomes)
        v_max = self.ftensor(v_max_)
        k_m = self.ftensor(k_m_)
        K_r = self.ftensor(K_r_)
        N_f = self.itensor(N_f_)
        N_b = self.itensor(N_b_)
        N_h = self.itensor(N_h_)

        N = self.derive_stoichiometry(N_f=N_f, N_b=N_b)
        k_f, k_b = self.derive_rates(N=N, k_m=k_m)

        return {
            "N": N,
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
        return N_b - N_f  # (c,p,m)

    def derive_rates(
        self, N: torch.Tensor, k_m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = self.eps
        max_k = self.max_k

        # energies define k_e which defines k_e = k_f/k_b
        # extreme energies can create Inf or 0.0, avoid them with clamp
        e = torch.einsum("cpm,m->cp", N.to(self.ftype), self.mol_energies)
        k_e = torch.exp(-e / self.abs_temp / GAS_CONSTANT).clamp(eps, max_k)

        # Km is sampled between a defined range
        # exessively small Km can create numerical instability
        # thus, sampled Km should define the smaller Km of k_e = k_f/k_b
        # k_e>=1  => k_f=Km,         k_b=k_e*Km
        # k_e<1   => k_f=Km/k_e,      k_b=Km
        # this operation can create again Inf or 0.0, avoided with clamp, limits K_e
        is_fwd = k_e >= 1.0
        k_f = torch.where(is_fwd, k_m, k_m / k_e).clamp(eps, max_k)
        k_b = torch.where(is_fwd, k_m * k_e, k_m).clamp(eps, max_k)
        return k_f, k_b
