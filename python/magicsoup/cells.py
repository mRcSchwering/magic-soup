import json
import logging
import math
from pathlib import Path
from typing import Annotated, TypedDict, overload

import torch

from magicsoup.chemistry import Chemistry
from magicsoup.core.strarray import StrArray
from magicsoup.core.tensorclass import TensorClass, TensorKwargs
from magicsoup.core.util import read_fasta, write_fasta
from magicsoup.genomics import Genomics, GenomicsKwargs
from magicsoup.proteomics import Proteomics, ProteomicsKwargs

_log = logging.getLogger(__name__)


A = Annotated[StrArray, "cpu"]
T = Annotated[torch.Tensor, "cuda"]


class CellsKwargs(TypedDict, total=False):
    dim_scaling: float
    p_max: int


class Cells(TensorClass):
    """
    Manage all cell states throughout the simulation

    Arguments:
        chemistry: simulation [Chemistry][magicsoup.chemistry.Chemistry]
        genomics: simulation [Genomics][magicsoup.genomics.Genomics]
        proteomics: simulation [Proteomics][magicsoup.proteomics.Proteomics]
        dim_scaling: Scaling hysteresis used for dimensions `c` and `p` in all cell states.
            Increase to reduce number of memory allocations,
            reduce to reduce memory and compute during simulation.
        c_max: Maximum possible number of cells (number of voxels on cell map)
        p_max: Maximum possible number of proteins (pruned if more)
        tensor_kwargs: Kwargs passed to [TensorClass][magicsoup.util.TensorClass]

    During the simulation all proteomes are represented as tensors of dimensions `c`, `p`, `2m`.
    Tensor dimension `c` describes each cell, `p` each protein, `m` each molecule species.
    In tensors intracellular molecules are represented first, then extracellular ones, hence `2m`.

    Dimensions `c` and `p` can change during the simulation.
    If their sizes have to be increased or decreased use [update_c()][.update_c()] and [update_p()][.update_p].
    According to `dim_scaling` the dimension can increase more or decrease less than the provided number.
    This also means some element in dimension `c` can be empty.
    Thus, the size of `c` is not the number of cells (use [get_n_cells()][.get_n_cells] for that).
    Use [get_cell_idxs()][.get_cell_idxs] to get all indices of `c` which currently contain cells.
    And use [get_free_idxs()][.get_free_idxs] to get all indices of `c` which currently don't contain cells.
    """

    def __init__(
        self,
        chemistry: Chemistry,
        genomics: Genomics | None = None,
        proteomics: Proteomics | None = None,
        dim_scaling: float = 0.2,
        c_max: int = 100_000,
        p_max: int = 1_000,
        genomics_kwargs: GenomicsKwargs | None = None,
        proteomics_kwargs: ProteomicsKwargs | None = None,
        tensor_kwargs: TensorKwargs | None = None,
    ):
        super().__init__(**(tensor_kwargs or {}))
        self.chemistry = chemistry
        self.dim_scaling = dim_scaling
        self.c_max = c_max
        self.p_max = p_max

        # without reactions, catalytic domains are not allowed
        genomics_kwargs = genomics_kwargs or {}
        if len(chemistry.reactions) == 0:
            genomics_kwargs["p_catal_dom"] = 0

        self.genomics = genomics or Genomics(**genomics_kwargs)
        self.proteomics = proteomics or Proteomics(
            **(proteomics_kwargs or {}),
            chemistry=chemistry,
            scalar_enc_size=self.genomics.one_codon_map.encoding_size,
            vector_enc_size=self.genomics.two_codon_map.encoding_size,
            tensor_kwargs=tensor_kwargs,
        )

        molecules = chemistry.molecules
        n_molecules = len(molecules)
        self.c = 0  # dim 0
        self.p = 0  # dim 1
        self.m = 2 * n_molecules  # dim2

        # cellular parameters
        self.genomes: A = StrArray.empty(0)  # cell genomes
        self.lineages: A = StrArray.empty(0)  # cell lineages
        self.alive: T = self.idxzeros(0).bool()  # which ones are alive
        self.ages: T = self.fzeros(0)  # time since last division
        self.generations: T = self.idxzeros(0)  # number of divisions
        self.molecules: T = self.fzeros(0, n_molecules)  # intracellular concentrations

        # kinetics parameters
        self.k_f: T = self.fzeros(0, 0)  # forward reaction rates
        self.k_b: T = self.fzeros(0, 0)  # backward reaction rates
        self.k_r: T = self.fzeros(0, 0)  # effector affinities
        self.v_max: T = self.fzeros(0, 0)  # maximum reaction rates
        self.N_e: T = self.fzeros(0, 0, self.m)  # effective stoichiometric numbers
        self.N_f: T = self.izeros(0, 0, self.m)  # forward stoichiometric coefficients
        self.N_b: T = self.izeros(0, 0, self.m)  # backward stoichiometric coefficients
        self.N_h: T = self.izeros(0, 0, self.m)  # Hill coefficients

    def get_n_cells(self) -> int:
        """Get number of currently living cells"""
        return int(self.alive.sum())  # sync with host

    def get_cell_idxs(self) -> T:
        """Get (n,) cell index tensor of currently living cells"""
        return self.alive.nonzero(as_tuple=True)[0].int()  # sync with host

    def get_free_idxs(self) -> T:
        """Get (k,) cell index tensor of unused cell indices"""
        return (~self.alive).nonzero(as_tuple=True)[0].int()  # sync with host

    def update_genomes(self, genomes: StrArray, idxs: T) -> None:
        """
        Update cell genomes and proteomes

        Arguments:
            genomes: list of k new cell genomes
            idxs: (k,) cell index tensor of respective cells to be updated

        Position `i` of `genomes` relates to position `i` of `idxs`.
        Genomes are updated and translated. New proteomes are immediately active.
        """
        c = len(genomes)
        if c == 0:
            return

        cell, prot, tok, p = self.genomics.encode_proteomes(genomes=genomes)
        if p == 0:
            return

        params = self.proteomics.get_cell_params(
            cells=cell, proteins=prot, tokens=tok, p=p, c=c
        )

        v_max = params["v_max"]
        k_f = params["k_f"]
        k_b = params["k_b"]
        k_r = params["k_r"]
        N_e = params["N_e"]
        N_f = params["N_f"]
        N_b = params["N_b"]
        N_h = params["N_h"]

        # new proteomes may have different p
        self.update_p(p_req=p)

        # p have could been adjusted upwards or p was already < self.p
        if self.p > p:
            p_diff = self.p - p
            v_max = self._grow(v_max, dim=1, by_n=p_diff)
            k_f = self._grow(k_f, dim=1, by_n=p_diff)
            k_b = self._grow(k_b, dim=1, by_n=p_diff)
            k_r = self._grow(k_r, dim=1, by_n=p_diff)
            N_e = self._grow(N_e, dim=1, by_n=p_diff)
            N_f = self._grow(N_f, dim=1, by_n=p_diff)
            N_b = self._grow(N_b, dim=1, by_n=p_diff)
            N_h = self._grow(N_h, dim=1, by_n=p_diff)

        # p could have been > p_max
        if self.p < p:
            N_e = N_e[:, : self.p]
            N_f = N_f[:, : self.p]
            N_b = N_b[:, : self.p]
            N_h = N_h[:, : self.p]
            _log.warning("Proteome size %d has to be reduced to %d", p, self.p_max)

        # update kinetics parameters
        self.N_e[idxs] = N_e
        self.N_f[idxs] = N_f
        self.N_b[idxs] = N_b
        self.N_h[idxs] = N_h
        self.k_f[idxs] = k_f
        self.k_b[idxs] = k_b
        self.k_r[idxs] = k_r
        self.v_max[idxs] = v_max

        # sync with host
        idxs = idxs.cpu()

        # update genomes
        self.genomes[idxs.numpy()] = genomes
        if self.genomes.utilization <= 0.5:
            self.genomes.compact()

    def copy_proteomes(self, from_idxs: T, to_idxs: T) -> None:
        """
        Copy cell proteomes

        Arguments:
            from_idxs: (k,) cell index tensor of cells to copy from
            to_idxs: (k,) cell index tensor of cells to copy to

        Position `i` of `genomes` relates to position `i` of `idxs`.
        New proteomes are immediately active.
        """
        # update kinetics parameters
        self.N_e[to_idxs] = self.N_e[from_idxs]
        self.N_f[to_idxs] = self.N_f[from_idxs]
        self.N_b[to_idxs] = self.N_b[from_idxs]
        self.N_h[to_idxs] = self.N_h[from_idxs]
        self.k_f[to_idxs] = self.k_f[from_idxs]
        self.k_b[to_idxs] = self.k_b[from_idxs]
        self.k_r[to_idxs] = self.k_r[from_idxs]
        self.v_max[to_idxs] = self.v_max[from_idxs]

    def update_c(self, c_req: int) -> None:
        """Update `c` dimension by providing a new required size"""
        mar = math.ceil(c_req * self.dim_scaling)
        to_c = min(c_req + mar, self.c_max)
        if to_c < self.c:
            self._shrink_c(by_n=self.c - to_c)
        if c_req > self.c:
            self._grow_c(by_n=to_c - self.c)

    def update_p(self, p_req: int) -> None:
        """Update `p` dimension by providing a new required size"""
        mar = math.ceil(p_req * self.dim_scaling)
        to_p = min(p_req + mar, self.p_max)
        if to_p < self.p:
            self._shrink_p(by_n=self.p - to_p)
        if p_req > self.p:
            self._grow_p(by_n=to_p - self.p)

    def save_state(self, statedir: Path) -> None:
        """
        Save current state

        Arguments:
            statedir: Directory to store files in

        Only attributes which update during the simulation are saved.
        """
        statedir = statedir / type(self).__name__
        statedir.mkdir(parents=True, exist_ok=True)

        shape = {"c": self.c, "p": self.p, "m": self.m}
        (statedir / "shape.json").write_text(json.dumps(shape), encoding="utf-8")

        torch.save(self.alive, statedir / "alive.pt")
        torch.save(self.ages, statedir / "ages.pt")
        torch.save(self.generations, statedir / "generations.pt")
        torch.save(self.molecules, statedir / "molecules.pt")

        l2g = {
            f"{i} {l}": g
            for i, (l, g) in enumerate(
                zip(self.lineages.tolist(), self.genomes.tolist())
            )
        }
        write_fasta(file=statedir / "genomes.fasta", name_2_seq=l2g)

    def load_state(self, statedir: Path) -> None:
        """
        Load a saved state.
        The state had to be saved with [save_state()][.save_state] previously.

        Arguments:
            statedir: Directory that contains all files of that state
        """
        statedir = statedir / type(self).__name__

        shape = json.loads((statedir / "shape.json").read_text(encoding="utf-8"))
        self.c = shape["c"]
        self.p = shape["p"]
        self.m = shape["m"]

        self.alive = torch.load(statedir / "alive.pt", map_location=self.device)
        self.ages = torch.load(statedir / "ages.pt", map_location=self.device)
        self.generations = torch.load(
            statedir / "generations.pt", map_location=self.device
        )
        self.molecules = torch.load(statedir / "molecules.pt", map_location=self.device)

        l2g = read_fasta(file=statedir / "genomes.fasta")
        self.lineages = StrArray.from_strings([d.partition(" ")[2] for d in l2g])
        self.genomes = StrArray.from_strings(list(l2g.values()))

        # setup kinetics parameters again
        self.v_max = self.fzeros(self.c, self.p)
        self.k_f = self.fzeros(self.c, self.p)
        self.k_b = self.fzeros(self.c, self.p)
        self.k_r = self.fzeros(self.c, self.p)
        self.N_f = self.izeros(self.c, self.p, self.m)
        self.N_b = self.izeros(self.c, self.p, self.m)
        self.N_h = self.izeros(self.c, self.p, self.m)
        self.N_e = self.fzeros(self.c, self.p, self.m)

        # derive kinetics parameters from genomes
        idxs = self.idxarange(self.genomes.shape[0])
        cpu_idxs = idxs.cpu().numpy()  # sync with host
        self.update_genomes(genomes=self.genomes[cpu_idxs], idxs=idxs)

    def _grow_c(self, by_n: int) -> None:
        # cell paramaters
        self.genomes = self._grow(self.genomes, dim=0, by_n=by_n)
        self.lineages = self._grow(self.lineages, dim=0, by_n=by_n)
        self.alive = self._grow(self.alive, dim=0, by_n=by_n)
        self.ages = self._grow(self.ages, dim=0, by_n=by_n)
        self.generations = self._grow(self.generations, dim=0, by_n=by_n)
        self.molecules = self._grow(self.molecules, dim=0, by_n=by_n)

        # kinetics parameters
        self.v_max = self._grow(self.v_max, dim=0, by_n=by_n)
        self.k_f = self._grow(self.k_f, dim=0, by_n=by_n)
        self.k_b = self._grow(self.k_b, dim=0, by_n=by_n)
        self.k_r = self._grow(self.k_r, dim=0, by_n=by_n)
        self.N_f = self._grow(self.N_f, dim=0, by_n=by_n)
        self.N_b = self._grow(self.N_b, dim=0, by_n=by_n)
        self.N_h = self._grow(self.N_h, dim=0, by_n=by_n)
        self.N_e = self._grow(self.N_e, dim=0, by_n=by_n)

        # set new c
        self.c = len(self.genomes)

    def _shrink_c(self, by_n: int) -> None:
        avail = self.get_free_idxs()  # sync with host
        n_avail = avail.numel()
        if n_avail < by_n:
            raise ValueError(f"Cannot decrease c by {by_n}, only {n_avail} available")

        # sync with host
        avail = avail[:by_n]
        keep = torch.ones_like(self.alive, dtype=torch.bool)
        keep[avail] = False
        cpu_keep = keep.cpu().numpy()

        # cell parameters
        self.genomes = self.genomes[cpu_keep]
        self.lineages = self.lineages[cpu_keep]
        self.alive = self.alive[keep]
        self.ages = self.ages[keep]
        self.generations = self.generations[keep]
        self.molecules = self.molecules[keep]

        # kinetics parameters
        self.v_max = self.v_max[keep]
        self.k_f = self.k_f[keep]
        self.k_b = self.k_b[keep]
        self.k_r = self.k_r[keep]
        self.N_f = self.N_f[keep]
        self.N_b = self.N_b[keep]
        self.N_h = self.N_h[keep]
        self.N_e = self.N_e[keep]

        # set new c
        self.c = len(self.genomes)

    def _grow_p(self, by_n: int) -> None:
        # kinetics parameters
        self.v_max = self._grow(self.v_max, dim=1, by_n=by_n)
        self.k_f = self._grow(self.k_f, dim=1, by_n=by_n)
        self.k_b = self._grow(self.k_b, dim=1, by_n=by_n)
        self.k_r = self._grow(self.k_r, dim=1, by_n=by_n)
        self.N_f = self._grow(self.N_f, dim=1, by_n=by_n)
        self.N_b = self._grow(self.N_b, dim=1, by_n=by_n)
        self.N_h = self._grow(self.N_h, dim=1, by_n=by_n)
        self.N_e = self._grow(self.N_e, dim=1, by_n=by_n)

        # set new p
        self.p = self.v_max.shape[1]

    def _shrink_p(self, by_n: int) -> None:
        # kinetics parameters
        self.v_max = self.v_max[:, :-by_n]
        self.k_f = self.k_f[:, :-by_n]
        self.k_b = self.k_b[:, :-by_n]
        self.k_r = self.k_r[:, :-by_n]
        self.N_f = self.N_f[:, :-by_n]
        self.N_b = self.N_b[:, :-by_n]
        self.N_h = self.N_h[:, :-by_n]
        self.N_e = self.N_e[:, :-by_n]

        # set new p
        self.p = self.v_max.shape[1]

    @overload
    def _grow(self, o: A, dim: int, by_n: int) -> A: ...
    @overload
    def _grow(self, o: T, dim: int, by_n: int) -> T: ...
    def _grow(self, o: T | A, dim: int, by_n: int) -> T | A:
        shape = o.shape
        size = (*shape[:dim], by_n, *shape[dim + 1 :])
        if isinstance(o, torch.Tensor):
            empty_tens = torch.full(size, 0, device=o.device, dtype=o.dtype)
            return torch.cat([o, empty_tens], dim=dim)
        elif isinstance(o, StrArray):
            empty_arr = StrArray.empty(by_n)
            return StrArray.concat([o, empty_arr])
        raise ValueError(f"Unknown type {type(o)}")


# TODO: add n_proteins or similar as tensor
#       to easily get the number fo proteins of each cell
#       required to charge for protein expression
