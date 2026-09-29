import math
import pickle
import random
from io import BytesIO
from pathlib import Path
from typing import Any

import torch

from magicsoup import _lib  # type: ignore
from magicsoup.constants import ProteinSpecType
from magicsoup.containers import Cell, Chemistry
from magicsoup.genetics import Genetics
from magicsoup.kinetics2 import Kinetics
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
        mol_map_init: str = "randn",
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        device: str = "cpu",
        itype: torch.dtype = torch.int8,
        ftype: torch.dtype = torch.float32,
        batch_size: int | None = None,
    ):
        if not torch.cuda.is_available():
            device = "cpu"

        self.device = device
        self.itype = itype
        self.ftype = ftype
        self.batch_size = batch_size
        self.map_size = map_size
        self.abs_temp = abs_temp
        self.chemistry = chemistry

        self.genetics = Genetics(start_codons=start_codons, stop_codons=stop_codons)
        self.kinetics = Kinetics(device=device)
        self.proteomics = Proteomics(
            chemistry=chemistry,
            device=device,
            abs_temp=abs_temp,
            scalar_enc_size=max(self.genetics.one_codon_map.values()),
            vector_enc_size=max(self.genetics.two_codon_map.values()),
        )

        mol_degrads: list[float] = []
        diffusion: list[torch.nn.Conv2d] = []
        permeation: list[float] = []
        for mol in chemistry.molecules:
            mol_degrads.append(math.exp(-math.log(2) / mol.half_life))
            diffusion.append(self._get_diffuse(mol_diff_rate=mol.diffusivity))
            permeation.append(self._get_permeate(mol_perm_rate=mol.permeability))

        self.n_molecules = len(chemistry.molecules)
        self._int_mol_idxs = list(range(self.n_molecules))
        self._ext_mol_idxs = list(range(self.n_molecules, self.n_molecules * 2))
        self._mol_degrads = mol_degrads
        self._diffusion = diffusion
        self._permeation = permeation

        # working params
        m = 2 * len(chemistry.molecules)
        self.k_e = self._fzeros(0, 0)
        self.k_f = self._fzeros(0, 0)
        self.k_b = self._fzeros(0, 0)
        self.K_r = self._fzeros(0, 0, m)
        self.v_max = self._fzeros(0, 0)
        self.N = self._izeros(0, 0, m)
        self.N_f = self._izeros(0, 0, m)
        self.N_b = self._izeros(0, 0, m)
        self.N_h = self._izeros(0, 0, m)

        self.n_cells = 0
        self.cell_genomes: list[str] = []
        self.cell_labels: list[str] = []
        self.cell_map: torch.Tensor = torch.zeros(map_size, map_size).to(device).bool()
        self.cell_positions: torch.Tensor = torch.zeros(0, 2).to(device).int()
        self.cell_lifetimes: torch.Tensor = torch.zeros(0).to(device).int()
        self.cell_divisions: torch.Tensor = torch.zeros(0).to(device).int()
        self.cell_molecules: torch.Tensor = (
            torch.zeros(0, self.n_molecules).to(device).float()
        )
        self.molecule_map: torch.Tensor = self._get_molecule_map(
            n=self.n_molecules, size=map_size, init=mol_map_init
        )

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
            mask = (self.cell_positions == pos).all(dim=1)
            idxs = torch.argwhere(mask).flatten().tolist()
            if len(idxs) == 0:
                raise ValueError(f"Cell at {by_position} not found")
            idx = idxs[0]

        return Cell(
            world=self,
            idx=idx,
            genome=self.cell_genomes[idx],
            position=tuple(self.cell_positions[idx].tolist()),  # type: ignore
            label=self.cell_labels[idx],
            n_steps_alive=int(self.cell_lifetimes[idx].item()),
            n_divisions=int(self.cell_divisions[idx].item()),
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

        xs = self.cell_positions[:, 0].tolist()
        ys = self.cell_positions[:, 1].tolist()
        positions = [(x, y) for x, y in zip(xs, ys)]
        nghbrs = _lib.get_neighbors(from_idxs, to_idxs, positions, self.map_size)
        return nghbrs

    def spawn_cells(self, genomes: list[str]) -> list[int]:
        n_new_cells = len(genomes)
        if n_new_cells == 0:
            return []

        free_pos = self._find_free_random_positions(n_cells=n_new_cells)
        n_avail_pos = free_pos.size(0)
        if n_avail_pos == 0:
            return []

        if n_avail_pos < n_new_cells:
            n_new_cells = n_avail_pos
            random.shuffle(genomes)
            genomes = genomes[:n_new_cells]

        new_pos = free_pos[:n_new_cells]
        new_idxs = list(range(self.n_cells, self.n_cells + n_new_cells))
        self.n_cells += n_new_cells
        self.cell_genomes.extend(genomes)
        self.cell_labels.extend(randstr(n=12) for _ in range(n_new_cells))
        self._increase_cells(by_n=n_new_cells)

        # occupy positions
        xs = new_pos[:, 0]
        ys = new_pos[:, 1]
        self.cell_map[xs, ys] = True
        self.cell_positions[new_idxs] = new_pos

        # cell is picking up half the molecules of the pxl it is born on
        pickup = self.molecule_map[:, xs, ys] * 0.5
        self.cell_molecules[new_idxs, :] += pickup.T
        self.molecule_map[:, xs, ys] -= pickup

        self._update_cell_params(genomes=genomes, idxs=new_idxs)
        return new_idxs

    def add_cells(self, cells: list["Cell"]) -> list[int]:
        n_new_cells = len(cells)
        if n_new_cells == 0:
            return []

        free_pos = self._find_free_random_positions(n_cells=n_new_cells)
        n_avail_pos = free_pos.size(0)
        if n_avail_pos == 0:
            return []

        if n_avail_pos < n_new_cells:
            n_new_cells = n_avail_pos
            random.shuffle(cells)
            cells = cells[:n_new_cells]

        new_pos = free_pos[:n_new_cells]
        new_idxs = list(range(self.n_cells, self.n_cells + n_new_cells))
        self.n_cells += n_new_cells
        for cell in cells:
            self.cell_genomes.append(cell.genome)
            self.cell_labels.append(cell.label)

        self._increase_cells(by_n=n_new_cells)

        # occupy positions
        xs = new_pos[:, 0]
        ys = new_pos[:, 1]
        self.cell_map[xs, ys] = True
        self.cell_positions[new_idxs] = new_pos

        # previous molecules, lifetimes, divisions are transfered
        int_mols = [d.int_molecules for d in cells]
        lifetimes = [d.n_steps_alive for d in cells]
        divisions = [d.n_divisions for d in cells]
        genomes = [d.genome for d in cells]

        self.cell_molecules[new_idxs, :] = torch.stack(int_mols).to(self.device).float()
        self.cell_lifetimes[new_idxs] = self._i32_tensor(lifetimes)
        self.cell_divisions[new_idxs] = self._i32_tensor(divisions)

        self._update_cell_params(genomes=genomes, idxs=new_idxs)

        return new_idxs

    def divide_cells(self, cell_idxs: list[int]) -> list[tuple[int, int]]:
        if len(cell_idxs) == 0:
            return []

        # duplicates could lead to unexpected results
        cell_idxs = list(set(cell_idxs))

        xs = self.cell_positions[:, 0].tolist()
        ys = self.cell_positions[:, 1].tolist()
        occupied_positions = [(x, y) for x, y in zip(xs, ys)]
        (parent_idxs, child_idxs, child_pos) = _lib.divide_cells_if_possible(
            cell_idxs, occupied_positions, self.n_cells, self.map_size
        )

        n_new_cells = len(child_idxs)
        if n_new_cells == 0:
            return []

        # increment cells, genomes, labels
        self.n_cells += n_new_cells
        self.cell_genomes.extend([self.cell_genomes[d] for d in parent_idxs])
        self.cell_labels.extend([self.cell_labels[d] for d in parent_idxs])

        self._increase_cells(by_n=n_new_cells)
        self._copy_cell_params(from_idx=parent_idxs, to_idx=child_idxs)

        # position new cells
        child_pos = self._i32_tensor(child_pos)
        self.cell_map[child_pos[:, 0], child_pos[:, 1]] = True
        self.cell_positions[child_idxs] = child_pos

        # cells share molecules and increment cell divisions
        descendant_idxs = parent_idxs + child_idxs
        self.cell_molecules[child_idxs] = self.cell_molecules[parent_idxs]
        self.cell_molecules[descendant_idxs] *= 0.5
        self.cell_divisions[child_idxs] = self.cell_divisions[parent_idxs]
        self.cell_divisions[descendant_idxs] += 1
        self.cell_lifetimes[descendant_idxs] = 0

        return list(zip(parent_idxs, child_idxs))

    def update_cells(self, genome_idx_pairs: list[tuple[str, int]]) -> None:
        if len(genome_idx_pairs) == 0:
            return

        for genome, idx in genome_idx_pairs:
            self.cell_genomes[idx] = genome

        genomes, idxs = list(map(list, zip(*genome_idx_pairs)))
        self._update_cell_params(genomes=genomes, idxs=idxs)  # type: ignore

    def kill_cells(self, cell_idxs: list[int] | None = None) -> None:
        if cell_idxs is None:
            cell_idxs = list(range(self.n_cells))

        if len(cell_idxs) == 0:
            return

        # duplicates could raise error later
        cell_idxs = list(set(cell_idxs))

        xs = self.cell_positions[cell_idxs, 0]
        ys = self.cell_positions[cell_idxs, 1]
        self.cell_map[xs, ys] = False

        spillout = self.cell_molecules[cell_idxs, :]
        self.molecule_map[:, xs, ys] += spillout.T

        n_cells = self.cell_lifetimes.size(0)
        keep = torch.ones(n_cells, dtype=torch.bool, device=self.device)
        keep[cell_idxs] = False
        self.cell_lifetimes = self.cell_lifetimes[keep]
        self.cell_positions = self.cell_positions[keep]
        self.cell_divisions = self.cell_divisions[keep]
        self.cell_molecules = self.cell_molecules[keep]
        self._unset_cell_params(idx=~keep)
        self._decrease_cells(keep_idx=keep)

        for idx in sorted(cell_idxs, reverse=True):
            self.cell_genomes.pop(idx)
            self.cell_labels.pop(idx)

        self.n_cells -= len(cell_idxs)

    def migrate_cells(self, cell_idxs: list[int] | None = None):
        if cell_idxs is None:
            cell_idxs = list(range(self.n_cells))

        if len(cell_idxs) == 0:
            return

        # duplicates could lead to unexpected results
        cell_idxs = list(set(cell_idxs))

        xs = self.cell_positions[:, 0].tolist()
        ys = self.cell_positions[:, 1].tolist()
        positions = [(x, y) for x, y in zip(xs, ys)]
        new_pos, moved_idxs = _lib.move_cells(cell_idxs, positions, self.map_size)

        # reposition cells
        old_pos = self.cell_positions[moved_idxs]
        self.cell_map[old_pos[:, 0], old_pos[:, 1]] = False
        new_pos = self._i32_tensor(new_pos)
        self.cell_map[new_pos[:, 0], new_pos[:, 1]] = True
        self.cell_positions[moved_idxs] = new_pos

    def resuspend_cells(self, cell_idxs: list[int] | None = None):
        if cell_idxs is None:
            cell_idxs = list(range(self.n_cells))

        if len(cell_idxs) == 0:
            return

        # duplicates could lead to unexpected results
        cell_idxs = list(set(cell_idxs))

        # unoccupy current positions
        old_xs = self.cell_positions[cell_idxs, 0]
        old_ys = self.cell_positions[cell_idxs, 1]
        self.cell_map[old_xs, old_ys] = False

        # find new unoccupied positions
        new_pos = self._find_free_random_positions(n_cells=len(cell_idxs))
        new_xs = new_pos[:, 0]
        new_ys = new_pos[:, 1]

        self.cell_map[new_xs, new_ys] = True
        self.cell_positions[cell_idxs] = new_pos

    def enzymatic_activity(self):
        if self.n_cells == 0:
            return

        xs = self.cell_positions[:, 0]
        ys = self.cell_positions[:, 1]
        X0 = torch.cat([self.cell_molecules, self.molecule_map[:, xs, ys].T], dim=1)
        X1 = self.kinetics.step_protein_activity(
            h=1,
            x0=X0,
            N=self.N,
            N_f=self.N_f,
            N_b=self.N_b,
            N_h=self.N_h,
            k_f=self.k_f,
            k_b=self.k_b,
            K_r=self.K_r,
            v_max=self.v_max,
        )

        self.molecule_map[:, xs, ys] = X1[:, self._ext_mol_idxs].T
        self.cell_molecules = X1[:, self._int_mol_idxs]

    @torch.no_grad()
    def diffuse_molecules(self):
        n_pxls = self.map_size**2
        for mol_i, diffuse in enumerate(self._diffusion):
            total_before = self.molecule_map[mol_i].sum()
            before = self.molecule_map[mol_i].unsqueeze(0).unsqueeze(1)
            after = diffuse(before)
            self.molecule_map[mol_i] = torch.squeeze(after, 0).squeeze(0)
            total_after = self.molecule_map[mol_i].sum()

            # attempt to fix the problem that convolusion makes a small amount of
            # molecules appear or disappear (I think because floating point)
            self.molecule_map[mol_i] += (total_before - total_after) / n_pxls
            self.molecule_map[mol_i] = self.molecule_map[mol_i].clamp(0.0)

        if self.n_cells == 0:
            return

        xs = self.cell_positions[:, 0]
        ys = self.cell_positions[:, 1]
        X = torch.cat([self.cell_molecules, self.molecule_map[:, xs, ys].T], dim=1)

        for mol_i, permeate in enumerate(self._permeation):
            d_int = X[:, mol_i] * permeate
            d_ext = X[:, mol_i + self.n_molecules] * permeate
            X[:, mol_i] += d_ext - d_int
            X[:, mol_i + self.n_molecules] += d_int - d_ext

        self.molecule_map[:, xs, ys] = X[:, self._ext_mol_idxs].T
        self.cell_molecules = X[:, self._int_mol_idxs]

    def increment_cell_lifetimes(self):
        self.cell_lifetimes += 1

    def mutate_cells(
        self,
        cell_idxs: list[int] | None = None,
        p: float = 1e-6,
        p_indel: float = 0.4,
        p_del: float = 0.66,
    ):
        if cell_idxs is None:
            seqs = self.cell_genomes
            mutated = point_mutations(seqs=seqs, p=p, p_indel=p_indel, p_del=p_del)
            self.update_cells(genome_idx_pairs=mutated)
        else:
            seqs = [self.cell_genomes[d] for d in cell_idxs]
            mutated = point_mutations(seqs=seqs, p=p, p_indel=p_indel, p_del=p_del)
            pairs = [(d, cell_idxs[i]) for d, i in mutated]
            self.update_cells(genome_idx_pairs=pairs)

    def recombinate_cells(self, cell_idxs: list[int] | None = None, p: float = 1e-7):
        idxs = list(range(self.n_cells)) if cell_idxs is None else cell_idxs
        nghbrs = self.get_neighbors(cell_idxs=idxs)
        pairs = [(self.cell_genomes[a], self.cell_genomes[b]) for a, b in nghbrs]
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
        torch.save(self.cell_molecules, statedir / "cell_molecules.pt")
        torch.save(self.cell_map, statedir / "cell_map.pt")
        torch.save(self.molecule_map, statedir / "molecule_map.pt")
        torch.save(self.cell_lifetimes, statedir / "cell_lifetimes.pt")
        torch.save(self.cell_positions, statedir / "cell_positions.pt")
        torch.save(self.cell_divisions, statedir / "cell_divisions.pt")

        lines: list[str] = []
        for idx, (genome, label) in enumerate(zip(self.cell_genomes, self.cell_labels)):
            lines.append(f">{idx} {label}\n{genome}")

        with open(statedir / "cells.fasta", "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def load_state(self, statedir: Path, ignore_cell_params: bool = False):
        if not ignore_cell_params:
            self.kill_cells(cell_idxs=list(range(self.n_cells)))

        # TODO: load after calling increase cells
        #       then fill tensors with their values
        cell_molecules = torch.load(
            statedir / "cell_molecules.pt", map_location=self.device
        ).float()
        cell_map = torch.load(statedir / "cell_map.pt", map_location=self.device).bool()
        molecule_map = torch.load(
            statedir / "molecule_map.pt", map_location=self.device
        )
        cell_lifetimes = torch.load(
            statedir / "cell_lifetimes.pt", map_location=self.device
        ).int()
        cell_positions = torch.load(
            statedir / "cell_positions.pt", map_location=self.device
        ).int()
        cell_divisions = torch.load(
            statedir / "cell_divisions.pt", map_location=self.device
        ).int()

        with open(statedir / "cells.fasta", encoding="utf-8") as fh:
            text: str = fh.read()
            entries = [d.strip() for d in text.split(">") if len(d.strip()) > 0]

        self.cell_labels = []
        self.cell_genomes = []
        genome_idx_pairs: list[tuple[str, int]] = []
        for idx, entry in enumerate(entries):
            parts = entry.split("\n")
            descr = parts[0]
            seq = "" if len(parts) < 2 else parts[1]
            names = descr.split()
            label = names[1].strip() if len(names) > 1 else ""
            self.cell_genomes.append(seq)
            self.cell_labels.append(label)
            genome_idx_pairs.append((seq, idx))

        self.n_cells = len(genome_idx_pairs)

        self._increase_cells(by_n=self.n_cells)
        self.cell_molecules[:] = cell_molecules
        self.cell_map[:] = cell_map
        self.molecule_map[:] = molecule_map
        self.cell_lifetimes[:] = cell_lifetimes
        self.cell_positions[:] = cell_positions
        self.cell_divisions[:] = cell_divisions

        if not ignore_cell_params:
            self.update_cells(genome_idx_pairs=genome_idx_pairs)

    def _update_cell_params(self, genomes: list[str], idxs: list[int]):
        proteomes = self.genetics.translate_genomes(genomes=genomes)

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

        self._unset_cell_params(idx=unset_idxs)
        if max_prots == 0:
            return

        p = self.N.size(1)
        self._increase_proteins(by_n=max(max_prots - p, 0))

        n = len(set_proteomes)
        s = n  # TODO: should be configurable
        for a in range(0, n, s):
            b = a + s
            N, N_f, N_b, N_h, k_e, k_f, k_b, K_r, v_max = (
                self.proteomics.set_cell_params(proteomes=set_proteomes[a:b], p=p)
            )
            self.N[set_idxs[a:b]] = N
            self.N_f[set_idxs[a:b]] = N_f
            self.N_b[set_idxs[a:b]] = N_b
            self.N_h[set_idxs[a:b]] = N_h
            self.k_e[set_idxs[a:b]] = k_e
            self.k_f[set_idxs[a:b]] = k_f
            self.k_b[set_idxs[a:b]] = k_b
            self.K_r[set_idxs[a:b]] = K_r
            self.v_max[set_idxs[a:b]] = v_max

    def _find_free_random_positions(self, n_cells: int) -> torch.Tensor:
        # available spots on map
        pxls = torch.nonzero(~self.cell_map).int()
        n_pxls = pxls.size(0)
        n_cells = min(n_cells, n_pxls)

        # place cells on map
        idxs = random.sample(range(n_pxls), k=n_cells)
        chosen = pxls[idxs]
        return chosen

    def _get_molecule_map(self, n: int, size: int, init: str) -> torch.Tensor:
        args = [n, size, size]
        if init == "zeros":
            return torch.zeros(*args, device=self.device, dtype=torch.float32)
        if init == "randn":
            return (
                torch.randn(*args, dtype=torch.float32, device=self.device) + 10.0
            ).abs()
        raise ValueError(
            f"Didnt recognize mol_map_init={init}."
            " Should be one of: 'zeros', 'randn'."
        )

    def _get_permeate(self, mol_perm_rate: float) -> float:
        if mol_perm_rate < 0.0:
            mol_perm_rate = -mol_perm_rate

        mol_perm_rate = min(mol_perm_rate, 1.0)

        if mol_perm_rate == 0.0:
            return 0.0

        d = 1 / mol_perm_rate
        return 1 / (d + 1)

    def _get_diffuse(self, mol_diff_rate: float) -> torch.nn.Conv2d:
        if mol_diff_rate < 0.0:
            mol_diff_rate = -mol_diff_rate

        # mol_diff_rate > 1.0 could also mean expanding the kernel
        # so that molecules can diffuse more than just 1 pxl per round
        mol_diff_rate = min(mol_diff_rate, 1.0)

        if mol_diff_rate == 0.0:
            a = 0.0
            b = 1.0
        else:
            d = 1 / mol_diff_rate
            a = 1 / (d + 8)
            b = d * a
            b = b + 1.0 - (8 * a + b)  # try correcting inaccuracy

        # fmt: off
        kernel = self._f32_tensor([[[
            [a, a, a],
            [a, b, a],
            [a, a, a],
        ]]])
        # fmt: on

        conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
            device=self.device,
        )
        conv.weight = torch.nn.Parameter(kernel, requires_grad=False)
        return conv

    def _i32_tensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=torch.int32)

    def _f32_tensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=torch.float32)

    def _copy_cell_params(
        self, from_idx: torch.Tensor | list[int], to_idx: torch.Tensor | list[int]
    ) -> None:
        self.k_e[to_idx] = self.k_e[from_idx]
        self.k_f[to_idx] = self.k_f[from_idx]
        self.k_b[to_idx] = self.k_b[from_idx]
        self.K_r[to_idx] = self.K_r[from_idx]
        self.v_max[to_idx] = self.v_max[from_idx]
        self.N[to_idx] = self.N[from_idx]
        self.N_f[to_idx] = self.N_f[from_idx]
        self.N_b[to_idx] = self.N_b[from_idx]
        self.N_h[to_idx] = self.N_h[from_idx]

    def _unset_cell_params(self, idx: torch.Tensor | list[int]) -> None:
        self.N[idx] = 0
        self.N_f[idx] = 0
        self.N_b[idx] = 0
        self.N_h[idx] = 0
        self.k_e[idx] = 0.0
        self.k_f[idx] = 0.0
        self.k_b[idx] = 0.0
        self.K_r[idx] = 0.0
        self.v_max[idx] = 0.0

    def _increase_cells(self, by_n: int) -> None:
        self.cell_lifetimes = self._expand_c(t=self.cell_lifetimes, by_n=by_n)
        self.cell_positions = self._expand_c(t=self.cell_positions, by_n=by_n)
        self.cell_divisions = self._expand_c(t=self.cell_divisions, by_n=by_n)
        self.cell_molecules = self._expand_c(t=self.cell_molecules, by_n=by_n)
        self.k_e = self._expand_c(t=self.k_e, by_n=by_n)
        self.k_f = self._expand_c(t=self.k_f, by_n=by_n)
        self.k_b = self._expand_c(t=self.k_b, by_n=by_n)
        self.K_r = self._expand_c(t=self.K_r, by_n=by_n)
        self.v_max = self._expand_c(t=self.v_max, by_n=by_n)
        self.N = self._expand_c(t=self.N, by_n=by_n)
        self.N_f = self._expand_c(t=self.N_f, by_n=by_n)
        self.N_b = self._expand_c(t=self.N_b, by_n=by_n)
        self.N_h = self._expand_c(t=self.N_h, by_n=by_n)

    def _decrease_cells(self, keep_idx: torch.Tensor | list[int]) -> None:
        self.k_e = self.k_e[keep_idx]
        self.k_f = self.k_f[keep_idx]
        self.k_b = self.k_b[keep_idx]
        self.K_r = self.K_r[keep_idx]
        self.v_max = self.v_max[keep_idx]
        self.N = self.N[keep_idx]
        self.N_f = self.N_f[keep_idx]
        self.N_b = self.N_b[keep_idx]
        self.N_h = self.N_h[keep_idx]

    def _increase_proteins(self, by_n: int) -> None:
        self.k_e = self._expand_p(t=self.k_e, by_n=by_n)
        self.k_f = self._expand_p(t=self.k_f, by_n=by_n)
        self.k_b = self._expand_p(t=self.k_b, by_n=by_n)
        self.K_r = self._expand_p(t=self.K_r, by_n=by_n)
        self.v_max = self._expand_p(t=self.v_max, by_n=by_n)
        self.N = self._expand_p(t=self.N, by_n=by_n)
        self.N_f = self._expand_p(t=self.N_f, by_n=by_n)
        self.N_b = self._expand_p(t=self.N_b, by_n=by_n)
        self.N_h = self._expand_p(t=self.N_h, by_n=by_n)

    def _decrease_proteins(self, by_n: int) -> None:
        self.k_e = self.k_e[:, :-by_n]
        self.k_f = self.k_f[:, :-by_n]
        self.k_b = self.k_b[:, :-by_n]
        self.K_r = self.K_r[:, :-by_n]
        self.v_max = self.v_max[:, :-by_n]
        self.N = self.N[:, :-by_n]
        self.N_f = self.N_f[:, :-by_n]
        self.N_b = self.N_b[:, :-by_n]
        self.N_h = self.N_h[:, :-by_n]

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

    def __repr__(self) -> str:
        kwargs = {
            "map_size": self.map_size,
            "abs_temp": self.abs_temp,
            "device": self.device,
        }
        args = [f"{k}:{d!r}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"
