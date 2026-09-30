import pickle
import random
from io import BytesIO
from pathlib import Path
from typing import Any

import torch

from magicsoup import _lib  # type: ignore
from magicsoup.cellular import Cell, Cells
from magicsoup.chemistry import Chemistry
from magicsoup.constants import ProteinSpecType
from magicsoup.genomics import Genomics
from magicsoup.kinetics2 import Kinetics
from magicsoup.map import Map
from magicsoup.mutations import point_mutations, recombinations
from magicsoup.proteomics import Proteomics
from magicsoup.util import randstr


def _torch_load(map_loc: str | None = None):
    # Closure rather than a lambda to preserve map_loc
    return lambda b: torch.load(BytesIO(b), map_location=map_loc)


class _CPU_Unpickler(pickle.Unpickler):
    """Inject map_location when unpickling tensor objects"""

    def __init__(self, *args, map_location: str | None = None, **kwargs):
        self._map_location = map_location
        super().__init__(*args, **kwargs)

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return _torch_load(map_loc=self._map_location)
        else:
            return super().find_class(module, name)


class World:

    def __init__(
        self,
        chemistry: Chemistry,
        map_size: int = 128,
        abs_temp: float = 310.0,
        mol_map_init: str = "zeros",
        time_step: float = 1.0,
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        device: str = "cpu",
        itype: torch.dtype = torch.int8,
        ftype: torch.dtype = torch.float32,
    ):
        self.time_step = time_step
        self.device = device
        self.itype = itype
        self.ftype = ftype
        self.abs_temp = abs_temp
        self.chemistry = chemistry

        self.map = Map(
            mol_init=mol_map_init,
            size=map_size,
            chemistry=chemistry,
            device=device,
            ftype=ftype,
        )
        self.cells = Cells(chemistry=chemistry, device=device, itype=itype, ftype=ftype)
        self.genomics = Genomics(start_codons=start_codons, stop_codons=stop_codons)
        self.kinetics = Kinetics(device=device)
        self.proteomics = Proteomics(
            chemistry=chemistry,
            device=device,
            abs_temp=abs_temp,
            scalar_enc_size=max(self.genomics.one_codon_map.values()),
            vector_enc_size=max(self.genomics.two_codon_map.values()),
        )

        n_molecules = len(chemistry.molecules)
        self._int_mol_idxs = list(range(n_molecules))
        self._ext_mol_idxs = list(range(n_molecules, n_molecules * 2))
        self.n_cells = 0

    def get_cell(
        self,
        by_idx: int | None = None,
        by_position: tuple[int, int] | None = None,
    ) -> "Cell":
        idx = -1
        if by_idx is not None:
            idx = by_idx
        if by_position is not None:
            pos = self._i32_tensor(by_position)
            mask = (self.cells.positions == pos).all(dim=1)
            idxs = torch.argwhere(mask).flatten().tolist()
            if len(idxs) == 0:
                raise ValueError(f"Cell at {by_position} not found")
            idx = idxs[0]

        return Cell(
            world=self,
            idx=idx,
            genome=self.cells.genomes[idx],
            position=tuple(self.cells.positions[idx].tolist()),  # type: ignore
            label=self.cells.labels[idx],
            age=int(self.cells.ages[idx].item()),
            generation=int(self.cells.generations[idx].item()),
        )

    def get_neighbors(
        self, cell_idxs: list[int], nghbr_idxs: list[int] | None = None
    ) -> list[tuple[int, int]]:
        if len(cell_idxs) == 0:
            return []

        from_idxs = list(set(cell_idxs))
        if nghbr_idxs is None:
            to_idxs = list(set(cell_idxs))
        else:
            to_idxs = list(set(nghbr_idxs))

        if to_idxs == 0:
            return []

        xs = self.cells.positions[:, 0].tolist()
        ys = self.cells.positions[:, 1].tolist()
        positions = [(x, y) for x, y in zip(xs, ys)]
        nghbrs = _lib.get_neighbors(from_idxs, to_idxs, positions, self.map.size)
        return nghbrs

    def spawn_cells(self, genomes: list[str]) -> list[int]:
        n_new_cells = len(genomes)
        if n_new_cells == 0:
            return []

        free_pos = self.map.find_free_random_positions(n=n_new_cells)
        n_avail_pos = free_pos.size(0)
        if n_avail_pos == 0:
            return []

        if n_avail_pos < n_new_cells:
            n_new_cells = n_avail_pos
            random.shuffle(genomes)
            # TODO: log warning or info
            genomes = genomes[:n_new_cells]

        n_cells = self.cells.get_alive_cells()
        self.cells.set_c(n_cells + n_new_cells)
        free_idxs = self.cells.get_available_idxs()
        new_idxs = free_idxs[:n_new_cells].tolist()

        for genome, idx in zip(genomes, new_idxs):
            self.cells.genomes[idx] = genome
            self.cells.labels[idx] = randstr(n=12)

        # occupy positions
        new_pos = free_pos[:n_new_cells]
        xs = new_pos[:, 0]
        ys = new_pos[:, 1]
        self.map.cells[xs, ys] = True
        self.cells.positions[new_idxs] = new_pos

        # cell is picking up half the molecules of the pxl it is born on
        pickup = self.map.molecules[:, xs, ys] * 0.5
        self.cells.x_i[new_idxs, :] += pickup.T
        self.map.molecules[:, xs, ys] -= pickup

        self._update_cell_params(genomes=genomes, idxs=new_idxs)
        return new_idxs

    def add_cells(self, cells: list["Cell"]) -> list[int]:
        n_new_cells = len(cells)
        if n_new_cells == 0:
            return []

        free_pos = self.map.find_free_random_positions(n=n_new_cells)
        n_avail_pos = free_pos.size(0)
        if n_avail_pos == 0:
            return []

        if n_avail_pos < n_new_cells:
            n_new_cells = n_avail_pos
            random.shuffle(cells)
            # TODO: log warning or info
            cells = cells[:n_new_cells]

        n_cells = self.cells.get_alive_cells()
        self.cells.set_c(n_cells + n_new_cells)
        free_idxs = self.cells.get_available_idxs()
        new_idxs = free_idxs[:n_new_cells].tolist()

        for cell, idx in zip(cells, new_idxs):
            self.cells.genomes[idx] = cell.genome
            self.cells.labels[idx] = cell.label

        # occupy positions
        new_pos = free_pos[:n_new_cells]
        xs = new_pos[:, 0]
        ys = new_pos[:, 1]
        self.map.cells[xs, ys] = True
        self.cells.positions[new_idxs] = new_pos

        # previous molecules, ages, divisions are transfered
        int_mols = [d.int_molecules for d in cells]
        ages = [d.age for d in cells]
        generations = [d.generation for d in cells]
        genomes = [d.genome for d in cells]

        self.cells.x_i[new_idxs, :] = self._f32_tensor(torch.stack(int_mols))
        self.cells.ages[new_idxs] = self._i32_tensor(ages)
        self.cells.generations[new_idxs] = self._i32_tensor(generations)

        self._update_cell_params(genomes=genomes, idxs=new_idxs)
        return new_idxs

    def divide_cells(self, cell_idxs: list[int]) -> list[tuple[int, int]]:
        if len(cell_idxs) == 0:
            return []

        # duplicates could lead to unexpected results
        cell_idxs = list(set(cell_idxs))

        n_cells = self.cells.get_alive_cells()

        xs = self.cells.positions[:, 0].tolist()
        ys = self.cells.positions[:, 1].tolist()
        occupied_positions = [(x, y) for x, y in zip(xs, ys)]
        (parent_idxs, child_idxs, child_pos) = _lib.divide_cells_if_possible(
            cell_idxs, occupied_positions, n_cells, self.map.size
        )

        n_new_cells = len(child_idxs)
        if n_new_cells == 0:
            return []

        self.cells.set_c(n_cells + n_new_cells)

        # transfer genomes, labels
        for child_idx, parent_idx in zip(child_idxs, parent_idxs):
            self.cells.genomes[child_idx] = self.cells.genomes[parent_idx]
            self.cells.labels[child_idx] = self.cells.labels[parent_idx]

        self.cells.k_e[parent_idxs] = self.cells.k_e[child_idxs]
        self.cells.k_f[parent_idxs] = self.cells.k_f[child_idxs]
        self.cells.k_b[parent_idxs] = self.cells.k_b[child_idxs]
        self.cells.K_r[parent_idxs] = self.cells.K_r[child_idxs]
        self.cells.v_max[parent_idxs] = self.cells.v_max[child_idxs]
        self.cells.N[parent_idxs] = self.cells.N[child_idxs]
        self.cells.N_f[parent_idxs] = self.cells.N_f[child_idxs]
        self.cells.N_b[parent_idxs] = self.cells.N_b[child_idxs]
        self.cells.N_h[parent_idxs] = self.cells.N_h[child_idxs]

        # position new cells
        child_pos = self._idxtensor(child_pos)
        self.map.cells[child_pos[:, 0], child_pos[:, 1]] = True
        self.cells.positions[child_idxs] = child_pos

        # cells share molecules, increment generations, reset lifetimes
        descendant_idxs = parent_idxs + child_idxs
        self.cells.x_i[child_idxs] = self.cells.x_i[parent_idxs]
        self.cells.x_i[descendant_idxs] *= 0.5
        self.cells.generations[child_idxs] = self.cells.generations[parent_idxs]
        self.cells.generations[descendant_idxs] += 1
        self.cells.ages[descendant_idxs] = 0

        return list(zip(parent_idxs, child_idxs))

    def update_cells(self, genome_idx_pairs: list[tuple[str, int]]) -> None:
        if len(genome_idx_pairs) == 0:
            return

        for genome, idx in genome_idx_pairs:
            self.cells.genomes[idx] = genome

        genomes, idxs = list(map(list, zip(*genome_idx_pairs)))
        self._update_cell_params(genomes=genomes, idxs=idxs)  # type: ignore

    def kill_cells(self, cell_idxs: list[int] | None = None) -> None:
        if cell_idxs is None:
            cell_idxs = list(range(self.n_cells))

        if len(cell_idxs) == 0:
            return

        # duplicates could raise error later
        cell_idxs = list(set(cell_idxs))

        # free up map
        xs = self.cells.positions[cell_idxs, 0]
        ys = self.cells.positions[cell_idxs, 1]
        self.map.cells[xs, ys] = False

        # spill out molecules
        spillout = self.cells.x_i[cell_idxs, :]
        self.map.molecules[:, xs, ys] += spillout.T

        self.cells.ages[cell_idxs] = 0.0
        self.cells.positions[cell_idxs] = 0
        self.cells.generations[cell_idxs] = 0
        self.cells.x_i[cell_idxs] = 0.0
        self.cells.N[cell_idxs] = 0
        self.cells.N_f[cell_idxs] = 0
        self.cells.N_b[cell_idxs] = 0
        self.cells.N_h[cell_idxs] = 0
        self.cells.k_e[cell_idxs] = 0.0
        self.cells.k_f[cell_idxs] = 0.0
        self.cells.k_b[cell_idxs] = 0.0
        self.cells.K_r[cell_idxs] = 0.0
        self.cells.v_max[cell_idxs] = 0.0

        for idx in cell_idxs:
            self.cells.genomes[idx] = ""
            self.cells.labels[idx] = ""

    def migrate_cells(self, cell_idxs: list[int] | None = None):
        if cell_idxs is None:
            cell_idxs = list(range(self.n_cells))

        if len(cell_idxs) == 0:
            return

        # duplicates could lead to unexpected results
        cell_idxs = list(set(cell_idxs))

        xs = self.cells.positions[:, 0].tolist()
        ys = self.cells.positions[:, 1].tolist()
        positions = [(x, y) for x, y in zip(xs, ys)]
        new_pos, moved_idxs = _lib.move_cells(cell_idxs, positions, self.map.size)

        # reposition cells
        old_pos = self.cells.positions[moved_idxs]
        self.map.cells[old_pos[:, 0], old_pos[:, 1]] = False
        new_pos = self._idxtensor(new_pos)
        self.map.cells[new_pos[:, 0], new_pos[:, 1]] = True
        self.cells.positions[moved_idxs] = new_pos

    def resuspend_cells(self, cell_idxs: list[int] | None = None):
        if cell_idxs is None:
            cell_idxs = list(range(self.n_cells))

        if len(cell_idxs) == 0:
            return

        # duplicates could lead to unexpected results
        cell_idxs = list(set(cell_idxs))

        # unoccupy current positions
        old_xs = self.cells.positions[cell_idxs, 0]
        old_ys = self.cells.positions[cell_idxs, 1]
        self.map.cells[old_xs, old_ys] = False

        # find new unoccupied positions
        new_pos = self.map.find_free_random_positions(n=len(cell_idxs))
        new_xs = new_pos[:, 0]
        new_ys = new_pos[:, 1]

        self.map.cells[new_xs, new_ys] = True
        self.cells.positions[cell_idxs] = new_pos

    def enzymatic_activity(self):
        if self.n_cells == 0:
            return

        alive = self.cells.alive
        xs = self.cells.positions[alive, 0]
        ys = self.cells.positions[alive, 1]
        x0 = torch.cat(
            [self.cells.x_i[alive], self.map.molecules[alive, xs, ys].T], dim=1
        )
        x1 = self.kinetics.step_protein_activity(
            h=self.time_step,
            x0=x0,
            N=self.cells.N[alive],
            N_f=self.cells.N_f[alive],
            N_b=self.cells.N_b[alive],
            N_h=self.cells.N_h[alive],
            k_f=self.cells.k_f[alive],
            k_b=self.cells.k_b[alive],
            K_r=self.cells.K_r[alive],
            v_max=self.cells.v_max[alive],
        )

        self.map.molecules[alive, xs, ys] = x1[:, self._ext_mol_idxs].T
        self.cells.x_i[alive] = x1[:, self._int_mol_idxs]

    def age_cells(self):
        self.cells.ages += self.time_step

    def mutate_cells(
        self,
        cell_idxs: list[int] | None = None,
        p: float = 1e-6,
        p_indel: float = 0.4,
        p_del: float = 0.66,
    ):
        idxs = (
            self.cells.get_available_idxs().tolist() if cell_idxs is None else cell_idxs
        )
        seqs = [self.cells.genomes[d] for d in idxs]
        mutated = point_mutations(seqs=seqs, p=p, p_indel=p_indel, p_del=p_del)
        pairs = [(d, idxs[i]) for d, i in mutated]
        self.update_cells(genome_idx_pairs=pairs)

    def recombinate_cells(self, cell_idxs: list[int] | None = None, p: float = 1e-7):
        idxs = (
            self.cells.get_available_idxs().tolist() if cell_idxs is None else cell_idxs
        )
        nghbrs = self.get_neighbors(cell_idxs=idxs)
        pairs = [(self.cells.genomes[a], self.cells.genomes[b]) for a, b in nghbrs]
        mutated = recombinations(seq_pairs=pairs, p=p)

        genome_idx_pairs = []
        for c0, c1, idx in mutated:
            c0_i, c1_i = nghbrs[idx]
            genome_idx_pairs.append((c0, c0_i))
            genome_idx_pairs.append((c1, c1_i))
        self.update_cells(genome_idx_pairs=genome_idx_pairs)

    def save(self, rundir: Path, name: str = "world.pkl"):
        rundir.mkdir(parents=True, exist_ok=True)
        with open(rundir / name, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def from_file(
        self,
        rundir: Path,
        name: str = "world.pkl",
        device: str | None = None,
    ) -> "World":
        with open(rundir / name, "rb") as fh:
            unpickler = _CPU_Unpickler(fh, map_location=device)
            obj: World = unpickler.load()

        if device is not None:
            obj.device = device
            obj.kinetics.device = device

        return obj

    def save_state(self, statedir: Path):
        statedir.mkdir(parents=True, exist_ok=True)
        self.map.save_state(statedir=statedir)
        self.cells.save_state(statedir=statedir)

    def load_state(self, statedir: Path):
        self.map.load_state(statedir=statedir)
        self.cells.load_state(statedir=statedir)

    def _update_cell_params(self, genomes: list[str], idxs: list[int]) -> None:
        proteomes = self.genomics.translate_genomes(genomes=genomes)

        max_prots: int = 0
        set_idxs: list[int] = []
        unset_idxs: list[int] = []
        set_proteomes: list[list[ProteinSpecType]] = []
        for idx, proteome in zip(idxs, proteomes):
            n_prots = len(proteome)
            if n_prots > 0:
                set_idxs.append(idx)
                set_proteomes.append(proteome)
                max_prots = max(max_prots, n_prots)
            else:
                unset_idxs.append(idx)

        self.cells.N[unset_idxs] = 0
        self.cells.N_f[unset_idxs] = 0
        self.cells.N_b[unset_idxs] = 0
        self.cells.N_h[unset_idxs] = 0
        self.cells.k_e[unset_idxs] = 0.0
        self.cells.k_f[unset_idxs] = 0.0
        self.cells.k_b[unset_idxs] = 0.0
        self.cells.K_r[unset_idxs] = 0.0
        self.cells.v_max[unset_idxs] = 0.0

        if max_prots == 0:
            return

        self.cells.set_p(max(max_prots - self.cells.p, 0))

        # TODO: implement batch updates
        N, N_f, N_b, N_h, k_e, k_f, k_b, K_r, v_max = self.proteomics.get_cell_params(
            proteomes=set_proteomes, p=self.cells.p
        )
        self.cells.N[set_idxs] = N
        self.cells.N_f[set_idxs] = N_f
        self.cells.N_b[set_idxs] = N_b
        self.cells.N_h[set_idxs] = N_h
        self.cells.k_e[set_idxs] = k_e
        self.cells.k_f[set_idxs] = k_f
        self.cells.k_b[set_idxs] = k_b
        self.cells.K_r[set_idxs] = K_r
        self.cells.v_max[set_idxs] = v_max

    def _get_permeate(self, mol_perm_rate: float) -> float:
        if mol_perm_rate < 0.0:
            mol_perm_rate = -mol_perm_rate

        mol_perm_rate = min(mol_perm_rate, 1.0)

        if mol_perm_rate == 0.0:
            return 0.0

        d = 1 / mol_perm_rate
        return 1 / (d + 1)

    def _i32_tensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=torch.int32)

    def _f32_tensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=torch.float32)

    def _expand_c(self, t: torch.Tensor, by_n: int) -> torch.Tensor:
        size = t.size()
        zeros = torch.zeros(by_n, *size[1:], device=self.device, dtype=t.dtype)
        return torch.cat([t, zeros], dim=0)

    def _expand_p(self, t: torch.Tensor, by_n: int) -> torch.Tensor:
        size = t.size()
        zeros = torch.zeros(size[0], by_n, *size[2:], device=self.device, dtype=t.dtype)
        return torch.cat([t, zeros], dim=1)

    def _izeros(self, *args) -> torch.Tensor:
        return torch.zeros(*args, device=self.device, dtype=self.itype)

    def _fzeros(self, *args) -> torch.Tensor:
        return torch.zeros(*args, device=self.device, dtype=self.ftype)

    def _idxtensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=torch.float32)

    def __repr__(self) -> str:
        kwargs = {
            "map_size": self.map.size,
            "abs_temp": self.abs_temp,
            "device": self.device,
        }
        args = [f"{k}:{d!r}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"
