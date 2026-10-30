import json
import logging
import math
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from magicsoup.chemistry import Chemistry
from magicsoup.genomics import Genomics, GenomicsKwargs
from magicsoup.proteomics import Proteomics, ProteomicsKwargs
from magicsoup.util import TensorClass, TensorKwargs, read_fasta, write_fasta

_log = logging.getLogger(__name__)


# TODO: have all genomes as str array
#       move between rust and python using str arrays


class CellsKwargs(TypedDict, total=False):
    dim_scaling: float
    p_max: int


# TODO: k_r -> k_r


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
        self.genomes = self.strempty(0)  # cell genomes
        self.lineages = self.strempty(0)  # cell lineages
        self.alive = self.idxzeros(0).bool()  # which ones are alive
        self.ages = self.fzeros(0)  # time since last division
        self.generations = self.idxzeros(0)  # number of divisions
        self.molecules = self.fzeros(0, n_molecules)  # intracellular concentrations

        # kinetics parameters
        self.k_f = self.fzeros(0, 0)  # forward reaction rates
        self.k_b = self.fzeros(0, 0)  # backward reaction rates
        self.k_r = self.fzeros(0, 0)  # effector affinities
        self.v_max = self.fzeros(0, 0)  # maximum reaction rates
        self.N_e = self.fzeros(0, 0, self.m)  # effective stoichiometric numbers
        self.N_f = self.izeros(0, 0, self.m)  # forward stoichiometric coefficients
        self.N_b = self.izeros(0, 0, self.m)  # backward stoichiometric coefficients
        self.N_h = self.izeros(0, 0, self.m)  # Hill coefficients

    def get_n_cells(self) -> int:
        """Get number of currently living cells"""
        return int(self.alive.sum().item())

    def get_cell_idxs(self) -> torch.Tensor:
        """Get (n,) cell index tensor of currently living cells"""
        return self.alive.nonzero(as_tuple=True)[0].int()

    def get_free_idxs(self) -> torch.Tensor:
        """Get (k,) cell index tensor of unused cell indices"""
        return (~self.alive).nonzero(as_tuple=True)[0].int()

    def update_genomes(self, genomes: list[str], idxs: torch.Tensor) -> None:
        """
        Update cell genomes and proteomes

        Arguments:
            genomes: list of k new cell genomes
            idxs: (k,) cell index tensor of respective cells to be updated

        Position `i` of `genomes` relates to position `i` of `idxs`.
        Genomes are updated and translated.
        New proteomes are immediately active.
        Prefer [copy_genomes()][.copy_genomes] over [update_genomes()][.update_genomes]
        if possible as it's much faster.
        """
        c = len(genomes)
        if c == 0:
            return

        cell, prot, tok, p = self.genomics.translate_genomes_arr(genomes=genomes)
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
            v_max = self._exp_tens_p(v_max, by_n=p_diff)
            k_f = self._exp_tens_p(k_f, by_n=p_diff)
            k_b = self._exp_tens_p(k_b, by_n=p_diff)
            k_r = self._exp_tens_p(k_r, by_n=p_diff)
            N_e = self._exp_tens_p(N_e, by_n=p_diff)
            N_f = self._exp_tens_p(N_f, by_n=p_diff)
            N_b = self._exp_tens_p(N_b, by_n=p_diff)
            N_h = self._exp_tens_p(N_h, by_n=p_diff)

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

        # update cellular parameters
        self.genomes[idxs] = genomes

    def copy_genomes(self, from_idxs: torch.Tensor, to_idxs: torch.Tensor) -> None:
        """
        Copy cell genomes and proteomes

        Arguments:
            from_idxs: (k,) cell index tensor of cells to copy from
            to_idxs: (k,) cell index tensor of cells to copy to

        Position `i` of `genomes` relates to position `i` of `idxs`.
        New proteomes are immediately active.
        Prefer [copy_genomes()][.copy_genomes] over [update_genomes()][.update_genomes]
        if possible as it's much faster.
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

        # update cellular parameters
        self.genomes[to_idxs] = self.genomes[from_idxs]

    def update_c(self, c_req: int) -> None:
        """Update `c` dimension by providing a new required size"""
        mar = math.ceil(c_req * self.dim_scaling)
        to_c = min(c_req + mar, self.c_max)
        if to_c < self.c:
            self._decrease_c(by_n=self.c - to_c)
        if c_req > self.c:
            self._increase_c(by_n=to_c - self.c)

    def update_p(self, p_req: int) -> None:
        """Update `p` dimension by providing a new required size"""
        mar = math.ceil(p_req * self.dim_scaling)
        to_p = min(p_req + mar, self.p_max)
        if to_p < self.p:
            self._decrease_p(by_n=self.p - to_p)
        if p_req > self.p:
            self._increase_p(by_n=to_p - self.p)

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

        lineages_2_genomes = {
            f"{i} {l}": g
            for i, (l, g) in enumerate(
                zip(self.lineages.tolist(), self.genomes.tolist())
            )
        }
        write_fasta(file=statedir / "genomes.fasta", name_2_seq=lineages_2_genomes)

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

        lineage_2_genome = read_fasta(file=statedir / "genomes.fasta")
        self.lineages = self.strarray([d.partition(" ")[2] for d in lineage_2_genome])
        self.genomes = self.strarray(list(lineage_2_genome.values()))

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
        self.update_genomes(genomes=self.genomes[idxs], idxs=idxs)

    def _increase_c(self, by_n: int) -> None:
        # cell paramaters
        self.genomes = self._exp_arr_c(a=self.genomes, by_n=by_n)
        self.lineages = self._exp_arr_c(a=self.lineages, by_n=by_n)
        self.alive = self._exp_tens_c(t=self.alive, by_n=by_n)
        self.ages = self._exp_tens_c(t=self.ages, by_n=by_n)
        self.generations = self._exp_tens_c(t=self.generations, by_n=by_n)
        self.molecules = self._exp_tens_c(t=self.molecules, by_n=by_n)

        # kinetics parameters
        self.v_max = self._exp_tens_c(t=self.v_max, by_n=by_n)
        self.k_f = self._exp_tens_c(t=self.k_f, by_n=by_n)
        self.k_b = self._exp_tens_c(t=self.k_b, by_n=by_n)
        self.k_r = self._exp_tens_c(t=self.k_r, by_n=by_n)
        self.N_f = self._exp_tens_c(t=self.N_f, by_n=by_n)
        self.N_b = self._exp_tens_c(t=self.N_b, by_n=by_n)
        self.N_h = self._exp_tens_c(t=self.N_h, by_n=by_n)
        self.N_e = self._exp_tens_c(t=self.N_e, by_n=by_n)

        # set new c
        self.c = self.genomes.shape[0]

    def _decrease_c(self, by_n: int) -> None:
        avail = self.get_free_idxs()
        n_avail = avail.numel()
        if n_avail < by_n:
            raise ValueError(f"Cannot decrease c by {by_n}, only {n_avail} available")

        avail = avail[:by_n]
        keep = torch.ones_like(self.alive, dtype=torch.bool)
        keep[avail] = False

        # cell parameters
        self.genomes = self.genomes[keep]
        self.lineages = self.lineages[keep]
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
        self.c = self.genomes.shape[0]

    def _increase_p(self, by_n: int) -> None:
        # kinetics parameters
        self.v_max = self._exp_tens_p(t=self.v_max, by_n=by_n)
        self.k_f = self._exp_tens_p(t=self.k_f, by_n=by_n)
        self.k_b = self._exp_tens_p(t=self.k_b, by_n=by_n)
        self.k_r = self._exp_tens_p(t=self.k_r, by_n=by_n)
        self.N_f = self._exp_tens_p(t=self.N_f, by_n=by_n)
        self.N_b = self._exp_tens_p(t=self.N_b, by_n=by_n)
        self.N_h = self._exp_tens_p(t=self.N_h, by_n=by_n)
        self.N_e = self._exp_tens_p(t=self.N_e, by_n=by_n)

        # set new p
        self.p = self.v_max.shape[1]

    def _decrease_p(self, by_n: int) -> None:
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

    def _exp_tens_c(self, t: torch.Tensor, by_n: int, value: int = 0) -> torch.Tensor:
        size = t.size()
        shape = (by_n, *size[1:])
        zeros = torch.full(shape, value, device=t.device, dtype=t.dtype)
        return torch.cat([t, zeros], dim=0)

    def _exp_arr_c(self, a: np.ndarray, by_n: int, value: str = "") -> np.ndarray:
        shape = a.shape
        shape = (by_n, *shape[1:])
        zeros = np.full(shape, value, dtype=a.dtype)
        return np.concatenate([a, zeros], axis=0)

    def _exp_tens_p(self, t: torch.Tensor, by_n: int, value: int = 0) -> torch.Tensor:
        size = t.size()
        shape = (size[0], by_n, *size[2:])
        zeros = torch.full(shape, value, device=t.device, dtype=t.dtype)
        return torch.cat([t, zeros + value], dim=1)
