import math
import random
from typing import Any

import torch

from magicsoup import _lib  # type: ignore

from .cellular import Protein
from .chemistry import Chemistry, Molecule
from .constants import GAS_CONSTANT, ProteinSpecType

# TODO: Maps umbauen -> einfach Python Maps


class _HillMapFact:
    """
    Creates an object that maps tokens to 1, 2, 3, 4, 5
    with chances 52%, 26%, 13%, 6%, 3% respectively.
    """

    def __init__(
        self,
        max_token: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.int8,
        zero_value: int = 0,
    ):
        choices = [5] + 2 * [4] + 4 * [3] + 8 * [2] + 16 * [1]
        numbers = torch.tensor([zero_value] + random.choices(choices, k=max_token))
        self.numbers = numbers.to(device=device, dtype=dtype)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.numbers[t]  # (t.shape) values of numbers

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
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        zero_value: float = 0.0,
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
        self.weights = weights.to(device=device, dtype=dtype)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.weights[t]  # (t.shape) values of weights

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

    def __init__(
        self,
        max_token: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.int8,
        zero_value: int = 0,
    ):
        choices = [1, -1]
        signs = torch.tensor([zero_value] + random.choices(choices, k=max_token))
        self.signs = signs.to(device=device, dtype=dtype)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.signs[t]  # (t.shape) values of weights

    def inverse(self) -> dict[bool, list[int]]:
        sign_map = {}
        M = self.signs.to("cpu")
        sign_map[True] = torch.argwhere(M == 1).flatten().tolist()
        sign_map[False] = torch.argwhere(M == -1).flatten().tolist()
        return sign_map


class _VectorMapFact:
    """
    Create an object that maps tokens to a list of vectors.
    Each vector will be mapped with the same frequency.
    """

    def __init__(
        self,
        max_token: int,
        vec_len: int,
        vectors: list[list[int]],
        device: str = "cpu",
        dtype: torch.dtype = torch.int16,
        zero_value: int = 0,
    ):
        n_vectors = len(vectors)
        M = torch.full((max_token + 1, vec_len), fill_value=zero_value)

        if n_vectors == 0:
            self.M = M.to(device=device, dtype=dtype)
            return

        if not all(len(d) == vec_len for d in vectors):
            raise ValueError(f"Not all vectors have length of vec_len={vec_len}")

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

        self.M = M.to(device=device, dtype=dtype)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.M[t]  # (t.shape, vec_len) values of vectors


class _ReactionMapFact(_VectorMapFact):
    """
    Create an object that maps tokens to vectors.
    Each vector has number of molecules length and represents the
    stoichiometry of a reaction.
    """

    def __init__(
        self,
        molmap: dict[Molecule, int],
        reactions: list[tuple[list[Molecule], list[Molecule]]],
        max_token: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.int8,
        zero_value: int = 0,
    ):
        n_mols = 2 * len(molmap)
        n_reacts = len(reactions)

        # careful, only copy [0] to avoid having references to the same list
        vectors = [[0] * n_mols for _ in range(n_reacts)]
        for ri, (lft, rgt) in enumerate(reactions):
            for mol in lft:
                mol_i = molmap[mol]
                vectors[ri][mol_i] -= 1
            for mol in rgt:
                mol_i = molmap[mol]
                vectors[ri][mol_i] += 1

        super().__init__(
            vectors=vectors,
            vec_len=n_mols,
            max_token=max_token,
            device=device,
            dtype=dtype,
            zero_value=zero_value,
        )

    def inverse(
        self,
        molmap: dict[Molecule, int],
        reactions: list[tuple[list[Molecule], list[Molecule]]],
        n_mols: int,
    ) -> dict[tuple[tuple[Molecule, ...], tuple[Molecule, ...]], list[int]]:
        react_map = {}
        M = self.M.to("cpu")

        for subs, prods in reactions:
            t = torch.zeros(n_mols)
            for sub in subs:
                t[molmap[sub]] -= 1

            for prod in prods:
                t[molmap[prod]] += 1

            idxs = torch.argwhere((t == M).all(dim=1)).flatten().tolist()
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
        dtype: torch.dtype = torch.int8,
        zero_value: int = 0,
    ):
        n_mols = 2 * n_molecules

        # careful, only copy [0] to avoid having references to the same list
        vectors = [[0] * n_mols for _ in range(n_molecules)]
        for mi in range(n_molecules):
            vectors[mi][mi] = -1
            vectors[mi][mi + n_molecules] = 1

        super().__init__(
            vectors=vectors,
            vec_len=n_mols,
            max_token=max_token,
            device=device,
            dtype=dtype,
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
        dtype: torch.dtype = torch.int8,
        zero_value: int = 0,
    ):
        n_mols = 2 * n_molecules

        # careful, only copy [0] to avoid having references to the same list
        vectors = [[0] * n_mols for _ in range(n_mols)]
        for mi in range(n_mols):
            vectors[mi][mi] = 1

        super().__init__(
            vectors=vectors,
            vec_len=n_mols,
            max_token=max_token,
            device=device,
            dtype=dtype,
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


class Proteomics:

    def __init__(
        self,
        chemistry: Chemistry,
        abs_temp: float = 310.0,
        km_range: tuple[float, float] = (1e-2, 100.0),
        vmax_range: tuple[float, float] = (1e-3, 100.0),
        device: str = "cpu",
        itype: torch.dtype = torch.int8,
        ftype: torch.dtype = torch.float32,
        scalar_enc_size: int = 64 - 3,
        vector_enc_size: int = 4096 - 3 * 64,
        max_k: float = 1e36,
        eps: float = 1e-40,
    ) -> None:
        self.abs_temp = abs_temp
        self.device = device
        self.itype = itype
        self.ftype = ftype
        self.max_k = max_k
        self.eps = eps

        self.mol_names = [d.name for d in chemistry.molecules]
        self.mol_energies = self._ftensor([d.energy for d in chemistry.molecules] * 2)

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
        self.m = 2 * len(chemistry.molecules)
        self.km_2_idxs = self.km_map.inverse()
        self.vmax_2_idxs = self.vmax_map.inverse()
        self.sign_2_idxs = self.sign_map.inverse()
        self.hill_2_idxs = self.hill_map.inverse()
        self.trnsp_2_idxs = self.transport_map.inverse(molecules=chemistry.molecules)
        self.regul_2_idxs = self.effector_map.inverse(molecules=chemistry.molecules)
        self.catal_2_idxs = self.reaction_map.inverse(
            molmap=mol_2_mi, reactions=chemistry.reactions, n_mols=self.m
        )

        # init rust class
        self._setup_proteomics()

    def _setup_proteomics(self) -> None:
        km_map = {
            i: d for i, d in enumerate(self.km_map.weights.to("cpu").numpy().tolist())
        }
        vmax_map = {
            i: d for i, d in enumerate(self.vmax_map.weights.to("cpu").numpy().tolist())
        }
        sign_map = {
            i: d > 0
            for i, d in enumerate(self.sign_map.signs.to("cpu").numpy().tolist())
        }
        hill_map = {
            i: d for i, d in enumerate(self.hill_map.numbers.to("cpu").numpy().tolist())
        }
        reaction_map = {
            i: d for i, d in enumerate(self.reaction_map.M.to("cpu").numpy().tolist())
        }
        transport_map = {
            i: d for i, d in enumerate(self.transport_map.M.to("cpu").numpy().tolist())
        }
        effector_map = {
            i: d for i, d in enumerate(self.effector_map.M.to("cpu").numpy().tolist())
        }
        self._proteomics = _lib.Proteomics(
            km_map,
            vmax_map,
            sign_map,
            hill_map,
            reaction_map,
            transport_map,
            effector_map,
            self.m,
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
        proteome_kwargs = self._proteomics.get_proteome_repr(proteome, self.mol_names)
        return [Protein.from_dict(d) for d in proteome_kwargs]

    def get_cell_params(
        self, proteomes: list[list[ProteinSpecType]], p: int
    ) -> dict[str, torch.Tensor]:
        eps = self.eps
        max_k = self.max_k

        params = self._proteomics.get_proteome_params(proteomes)
        v_max, k_m, K_r, N_f, N_b, N_h = self._collect_proteome_params(
            params=params, p=p, m=self.m
        )

        N = N_b - N_f  # (c,p,m)

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

        return {
            "N": N,
            "N_f": N_f,
            "N_b": N_b,
            "N_h": N_h,
            "k_e": k_e,
            "k_f": k_f,
            "k_b": k_b,
            "K_r": K_r,
            "v_max": v_max,
        }

    def _collect_proteome_params(
        self,
        params: list[
            list[
                tuple[float, float, list[float], list[float], list[float], list[float]]
            ]
        ],
        p: int,
        m: int,
    ):
        zero_vector = [0.0] * m

        c_v_max = []
        c_k_m = []
        c_k_r = []
        c_n_f = []
        c_n_b = []
        c_n_h = []
        for proteome_params in params:
            p_v_max = []
            p_k_m = []
            p_k_r = []
            p_n_f = []
            p_n_b = []
            p_n_h = []
            for v_max_, k_m_, k_r_, n_f_, n_b_, n_h_ in proteome_params:
                p_v_max.append(v_max_)
                p_k_m.append(k_m_)
                p_k_r.append(k_r_)
                p_n_f.append(n_f_)
                p_n_b.append(n_b_)
                p_n_h.append(n_h_)

            p_pad = p - len(p_v_max)
            c_v_max.append(p_v_max + [0.0] * p_pad)
            c_k_m.append(p_k_m + [0.0] * p_pad)
            c_k_r.append(p_k_r + [zero_vector] * p_pad)
            c_n_f.append(p_n_f + [zero_vector] * p_pad)
            c_n_b.append(p_n_b + [zero_vector] * p_pad)
            c_n_h.append(p_n_h + [zero_vector] * p_pad)

        v_max = self._ftensor(c_v_max)  # (c,p)
        k_m = self._ftensor(c_k_m)  # (c,p)
        k_r = self._ftensor(c_k_r)  # (c,p,m)
        n_f = self._itensor(c_n_f)  # (c,p,m)
        n_b = self._itensor(c_n_b)  # (c,p,m)
        n_h = self._itensor(c_n_h)  # (c,p)

        return v_max, k_m, k_r, n_f, n_b, n_h

    def _collect_proteome_idxs(
        self, proteomes: list[list[ProteinSpecType]], p: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

            # TODO: couldn't I do the padding without p?
            p_pad = p - len(p_idxs0)
            c_dts.append(p_dts + [empty_seq] * p_pad)
            c_idxs0.append(p_idxs0 + [empty_seq] * p_pad)
            c_idxs1.append(p_idxs1 + [empty_seq] * p_pad)
            c_idxs2.append(p_idxs2 + [empty_seq] * p_pad)
            c_idxs3.append(p_idxs3 + [empty_seq] * p_pad)

        dom_types = self._idxtensor(c_dts)  # (c,p,d)
        idxs0 = self._idxtensor(c_idxs0)  # (c,p,d)
        idxs1 = self._idxtensor(c_idxs1)  # (c,p,d)
        idxs2 = self._idxtensor(c_idxs2)  # (c,p,d)
        idxs3 = self._idxtensor(c_idxs3)  # (c,p,d)
        return dom_types, idxs0, idxs1, idxs2, idxs3

    def _idxtensor(self, d: Any) -> torch.Tensor:
        # indexing Tensors must be at least int32
        return torch.tensor(d, device=self.device, dtype=torch.int32)

    def _ftensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=self.ftype)

    def _itensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=self.itype)
