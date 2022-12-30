from typing import Optional, Any
import random
from itertools import product
import math
import pickle
from pathlib import Path
import torch
from magicsoup.constants import EPS
from magicsoup.containers import Cell, Protein, Molecule, _DomainFact
from magicsoup.util import moore_nghbrhd
from magicsoup.kinetics import Kinetics
from magicsoup.genetics import Genetics


class World:
    """
    This object holds all information about the world the cells live in including
    the map, molecular diffusion and half-life, protein kinetic parameters, how genes
    are defined, and what molecules and reactions exist.

    - `domain_facts` dict mapping available domain factories to all possible nucleotide sequences
      by which they are encoded. During translation if any such nucleotide sequence appears
      (in-frame) in the coding sequence it will create the mapped domain.
    - `molecules` list of all molecule species that are part of this simulation
    - `map_size` size of world map as number of pixels in x- and y-direction
    - `mol_halflife` Half life of all molecules. Assuming one time step represents one second, values around 1e5
      would be reasonable for small molecules. Must be > 0.0.
    - `mol_diff_coef` Diffusion coefficient for all molecules. Assuming one time step is one second and one pixel
      has a size of 10um, values around 1e-8 are reasonable for small molecules. 0.0 would mean no diffusion at all.
      Diffusion rate reaches its maximum at around 1e-6 where all molecules are equally spread out around the pixels' Moor's neighborhood with
      each time step.
    - `abs_temp` Absolute temperature (K). Affects entropy term of reaction energy.
    - `mol_map_init` How to initialize molecule maps (`randn` or `zeros`). `zeros` are not actually 0.0 but a small positive
      value epsilon instead. 
    - `start_codons` start codons which start a coding sequence (translation only happens within coding sequences)
    - `stop_codons` stop codons which stop a coding sequence (translation only happens within coding sequences)
    - `dtype` pytorch dtype to use for (most) tensors. Use `float16` to speed up calculation at the expense of accuracy
      (if your hardware supports it). Beware of higher chance of overflow when using single precision.
    - `device` pytorch device to use for tensors. E.g. `cuda` to use CUDA on GPU.
      See https://pytorch.org/docs/stable/notes/cuda.html for pytorch CUDA semantics.

    The world map (for cells and molecules) is a square map with `map_size` pixels in both
    directions. But it is "circular" in that when you move over the border to the left, you re-appear
    on the right and vice versa (same for top and bottom). Each cell occupies one pixel. The cell
    experiences molecules on on that pixel as extracellular molecules. Additionally,
    the cell has its intracellular molecules.

    This object holds several tensors representing the environment of the world and its cells:

    - `cell_map` Map referencing which pixels are occupied by a cell (bool 2d tensor)
    - `molecule_map` Map referencing molecule abundances (3d float tensor with molecules in dimension 0)
    - `cell_molecules` Intracellular molecule abundances (2d float tensor with cells in dimension 0, molecules in dimension 1)
    - `cell_survival` Number of survived time steps for every living cell (1d float tensor)

    Furthermore, there is a list `cells` which holds all currently living cells with their position, genome,
    proteome and more. To gather all information for a particular cell use `get_cell`. Some attributes like
    intra- and extracellular molecule abundances will be added to the cell object when using `get_cell`.
    """

    def __init__(
        self,
        domain_facts: dict[_DomainFact, list[str]],
        molecules: list[Molecule],
        map_size=128,
        mol_halflife=100_000,
        mol_diff_coef=1e-8,
        abs_temp=310.0,
        mol_map_init="randn",
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        device="cpu",
        dtype=torch.float,
    ):
        self.map_size = map_size
        self.abs_temp = abs_temp
        self.mol_halflife = mol_halflife
        self.mol_degrad = math.exp(-math.log(2) / mol_halflife)
        self.mol_diff_coef = mol_diff_coef
        self.dtype = dtype

        if not torch.cuda.is_available():
            device = "cpu"

        self.device = device

        self.genetics = Genetics(
            domain_facts=domain_facts,
            molecules=molecules,
            start_codons=start_codons,
            stop_codons=stop_codons,
        )

        self.n_molecules = len(self.genetics.molecules)
        for idx, mol in enumerate(self.genetics.molecules):
            mol.idx = idx
            mol.idx_ext = self.n_molecules + idx
        self._int_mol_idxs = list(range(self.n_molecules))
        self._ext_mol_idxs = list(range(self.n_molecules, self.n_molecules * 2))

        self._diffuse = self._get_conv(mol_diff_rate=mol_diff_coef * 1e6)
        self._nghbrhd_map = {
            (x, y): torch.tensor(moore_nghbrhd(x, y, map_size))
            for x, y in product(range(map_size), range(map_size))
        }

        self.cells: list[Cell] = []
        self.kinetics = Kinetics(
            n_signals=2 * self.n_molecules,
            abs_temp=abs_temp,
            device=self.device,
            dtype=self.dtype,
        )

        self.cell_map = torch.zeros(map_size, map_size, dtype=torch.bool).to(device)
        self.cell_survival = torch.zeros(0, dtype=dtype).to(device)
        self.cell_molecules = torch.zeros(0, self.n_molecules, dtype=dtype).to(device)
        self.molecule_map = self._get_molecule_map(
            n=self.n_molecules, size=self.map_size, init=mol_map_init
        )

    def get_cell(
        self, by_idx: Optional[int] = None, by_label: Optional[str] = None
    ) -> Cell:
        """
        Get a cell with information about its current environment
        
        - `by_idx` get cell by cell index (`cell.idx`)
        - `by_label` get cell by cell label (`cell.label`)

        When using `get_cell` instead of accessing `cells` directly the returned cell
        object contains more information. E.g. extra- and intracellular molecule
        abundances are added.
        """
        if by_idx is not None:
            idx = by_idx
        if by_label is not None:
            cell_labels = [d.label for d in self.cells]
            idx = cell_labels.index(by_label)
            if idx < 0:
                raise ValueError(f"Cell {by_label} not found")

        cell = self.cells[idx]
        cell.int_molecules = self.cell_molecules[idx, :]
        cell.ext_molecules = self.molecule_map[:, cell.position[0], cell.position[1]].T
        cell.n_survived_steps = int(self.cell_survival[idx].item())

        return cell

    def add_random_cells(self, genomes: list[str]):
        """
        Create new cells from `genomes` to be placed randomly on the map.

        Each cell will be placed randomly on the map and receive the molecule
        abundances of the pixel where it was added.

        If there are less pixels left on the cell map than cells you want to add,
        only the remaining pixels will be filled with new cells.
        """
        if len(genomes) == 0:
            return

        new_positions = self._place_new_cells_in_random_positions(n_cells=len(genomes))
        n_new_cells = len(new_positions)

        if n_new_cells == 0:
            return

        prot_lens = []
        new_idxs = []
        new_params: list[tuple[int, int, Protein]] = []
        run_idx = len(self.cells)
        for genome, pos in zip(genomes, new_positions):
            proteome = self.genetics.get_proteome(seq=genome)
            prot_lens.append(len(proteome))
            new_idxs.append(run_idx)
            cell = Cell(idx=run_idx, genome=genome, proteome=proteome, position=pos)
            self.cells.append(cell)
            for prot_i, prot in enumerate(proteome):
                new_params.append((run_idx, prot_i, prot))
            run_idx += 1

        self.cell_survival = self._expand(t=self.cell_survival, n=n_new_cells, d=0)
        self.cell_molecules = self._expand(t=self.cell_molecules, n=n_new_cells, d=0)
        self.kinetics.increase_max_cells(by_n=n_new_cells)
        self.kinetics.increase_max_proteins(max_n=max(prot_lens))
        self.kinetics.set_cell_params(cell_prots=new_params)

        # TODO: half the concentrations
        # cell is supposed to have the same molecules as the pxl it lives on
        xs, ys = list(map(list, zip(*new_positions)))
        self.cell_molecules[new_idxs, :] = self.molecule_map[:, xs, ys].T

    def replicate_cells(self, parent_idxs: list[int]) -> tuple[list[int], list[int]]:
        """
        Replicate existing cells by their parents cell indexes (`cell.idx`).

        Each cell will be placed randomly next to its parent (Moore's neighborhood).
        It shares all molecules with its parent. So both will have have the molecule
        abundances the parent had before replicating.

        If every pixel in the cells' Moore neighborhood is taken
        the cell will not replicate.
        """
        if len(parent_idxs) == 0:
            return [], []

        succ_parent_idxs, child_idxs = self._replicate_cells_as_possible(
            parent_idxs=parent_idxs
        )

        n_new_cells = len(child_idxs)
        if n_new_cells == 0:
            return [], []

        self.cell_survival = self._expand(t=self.cell_survival, n=n_new_cells, d=0)
        self.cell_molecules = self._expand(t=self.cell_molecules, n=n_new_cells, d=0)
        self.kinetics.increase_max_cells(by_n=n_new_cells)
        self.kinetics.copy_cell_params(from_idxs=succ_parent_idxs, to_idxs=child_idxs)

        # TODO: check if correct
        # cell shares molecules with parent
        self.cell_molecules[child_idxs, :] = (
            self.cell_molecules[succ_parent_idxs, :] / 2
        )
        self.cell_molecules[succ_parent_idxs, :] = (
            self.cell_molecules[succ_parent_idxs, :] / 2
        )

        return succ_parent_idxs, child_idxs

    def update_cells(self, genomes: list[str], idxs: list[int]):
        """
        Update existing cells with new genomes.

        - `genomes` (new) genomes of cells to be updated
        - `idcs` cell indexes (`cell.idx`) of cells to be updated
        """
        if len(genomes) != len(idxs):
            raise ValueError(
                "Genomes and idxs represent the same list of cells."
                f" But now they don't have the same length. genomes: {len(genomes)}, idxs: {len(idxs)}."
            )

        if len(genomes) == 0:
            return

        prot_lens: list[int] = []
        set_params: list[tuple[int, int, Protein]] = []
        unset_params: list[tuple[int, int]] = []
        for idx, genome in zip(idxs, genomes):
            cell = self.cells[idx]
            newprot = self.genetics.get_proteome(seq=genome)
            oldprot = cell.proteome

            n_new = len(newprot)
            n_old = len(oldprot)
            n = min(n_old, n_new)
            for pi, (np, op) in enumerate(zip(newprot[:n], oldprot[:n])):
                if np != op:
                    set_params.append((idx, pi, np))
            if n_old > n_new:
                unset_params.extend((idx, i) for i in range(n, n_old))

            cell.proteome = newprot
            cell.genome = genome
            prot_lens.append(n_new)

        self.kinetics.increase_max_proteins(max_n=max(prot_lens))
        self.kinetics.set_cell_params(cell_prots=set_params)
        self.kinetics.unset_cell_params(cell_prots=unset_params)

    def kill_cells(self, cell_idxs: list[int]):
        """
        Remove cells by their indexes (`cell.idx`) and spill out their molecule
        contents onto the pixels they used to live on.
        """
        if len(cell_idxs) == 0:
            return

        xs, ys = list(map(list, zip(*[self.cells[i].position for i in cell_idxs])))
        self.cell_map[xs, ys] = False

        spillout = self.cell_molecules[cell_idxs, :]
        self.molecule_map[:, xs, ys] += spillout.T

        keep = torch.ones(self.cell_survival.shape[0], dtype=torch.bool)
        keep[cell_idxs] = False
        self.cell_survival = self.cell_survival[keep]
        self.cell_molecules = self.cell_molecules[keep]
        self.kinetics.remove_cell_params(keep=keep)

        new_idx = 0
        new_cells = []
        for old_idx, cell in enumerate(self.cells):
            if old_idx not in cell_idxs:
                cell.idx = new_idx
                new_idx += 1
                new_cells.append(cell)
        self.cells = new_cells

    def move_cells(self, cell_idxs: list[int]):
        """
        Move cells to a random position in their Moore neighborhood

        - `cell_idxs` cell indexes (`cell.idx`)
        
        If every pixel in the cells' Moore neighborhood is taken
        the cell will not be moved.
        """
        if len(cell_idxs) == 0:
            return

        cells = [self.cells[i] for i in cell_idxs]
        self._randomly_move_cells(cells=cells)

    @torch.no_grad()
    def diffuse_molecules(self):
        """Let molecules in world map diffuse by 1 time step"""
        before = self.molecule_map.unsqueeze(1)
        after = self._diffuse(before)
        self.molecule_map = torch.squeeze(after, 1)

    def degrade_molecules(self):
        """Degrade molecules in world map and cells by 1 time step"""
        self.molecule_map = self.molecule_map * self.mol_degrad
        self.cell_molecules = self.cell_molecules * self.mol_degrad

    def increment_cell_survival(self):
        """Increment number of currently living cells' time steps by 1"""
        idxs = list(range(len(self.cells)))
        self.cell_survival[idxs] += 1

    def enzymatic_activity(self):
        """Catalyze reactions for 1 time step and update all molecule abudances"""
        if len(self.cells) == 0:
            return

        xs, ys = list(map(list, zip(*[d.position for d in self.cells])))
        X = torch.cat([self.cell_molecules, self.molecule_map[:, xs, ys].T], dim=1)

        Xd = self.kinetics.integrate_signals(X=X)
        X += Xd

        self.molecule_map[:, xs, ys] = X[:, self._ext_mol_idxs].T
        self.cell_molecules = X[:, self._int_mol_idxs]

    def save(self, outdir: Path, name="world.pkl"):
        """Write whole world object to pickle file"""
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / name, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def from_file(self, filepath: Path) -> "World":
        """Restore previously saved world object from pickle file"""
        with open(filepath, "rb") as fh:
            return pickle.load(fh)

    def save_state(self, statedir: Path):
        """
        Save current state only. Will write a few files.
        
        Faster and more lightweight than `save`.
        Only saves the variable parts of `world`.
        You need one `save` to restore the whole `world` object though.
        """
        statedir.mkdir(parents=True, exist_ok=True)
        torch.save(self.cell_molecules, statedir / "cell_molecules.pt")
        torch.save(self.cell_map, statedir / "cell_map.pt")
        torch.save(self.molecule_map, statedir / "molecule_map.pt")
        with open(statedir / "cells.txt", "w") as fh:
            lines = [
                f"{d.idx}({d.position[0]},{d.position[1]}): {d.genome}"
                for d in self.cells
            ]
            fh.write("\n".join(lines))

    def load_state(self, statedir: Path):
        """Restore `world` to a state previously saved with `save_state`"""
        cell_molecules: torch.Tensor = torch.load(statedir / "cell_molecules.pt")
        self.cell_molecules = cell_molecules.to(self.dtype).to(self.device)
        cell_map: torch.Tensor = torch.load(statedir / "cell_map.pt")
        self.cell_map = cell_map.to(torch.bool).to(self.device)
        molecule_map: torch.Tensor = torch.load(statedir / "molecule_map.pt")
        self.molecule_map = molecule_map.to(self.dtype).to(self.device)

        with open(statedir / "cells.txt", "r") as fh:
            lines = [d for d in fh.read().split("\n") if len(d) > 0]

        genomes = []
        idxs = []
        self.cells = []
        for line in lines:
            prefix, genome = line.split("): ")
            idx, position = prefix.split("(")
            x, y = position.split(",")
            self.cells.append(Cell(genome="", proteome=[], position=(int(x), int(y))))
            genomes.append(genome)
            idxs.append(int(idx))

        self.update_cells(genomes=genomes, idxs=idxs)

    def summary(self, as_dict=False) -> Optional[dict]:
        """Summarize the current world setup with cells, molecules, and what kind of genomes to expect"""
        genetics_res = self.genetics.summary(as_dict=False)

        n_cells = len(self.cells)
        g_lens = [len(d.genome) for d in self.cells]
        pxls = self.map_size * self.map_size

        cells: dict[str, Any] = {}
        cells["nCells"] = n_cells
        cells["pctMapOccupied"] = n_cells / pxls * 100
        cells["maxSurvival"] = self.cell_survival.max().item() if n_cells > 0 else 0.0
        cells["avgSurvival"] = self.cell_survival.mean().item() if n_cells > 0 else 0.0
        cells["maxGenomeSize"] = max(g_lens) if n_cells > 0 else 0.0
        cells["avgGenomeSize"] = sum(g_lens) / n_cells if n_cells > 0 else 0.0

        mols: dict[str, Any] = {}
        for idx, mol in enumerate(self.genetics.molecules):
            mols[mol.name] = {
                "avgExtConc": self.molecule_map[idx].mean().item(),
                "avgIntConc": self.cell_molecules[:, idx].mean().item(),
            }

        if as_dict:
            kwargs = genetics_res or {}
            return {"molecules": mols, "cells": cells, **kwargs}

        # fmt: off
        print("\nCurrently living cells:")
        print(f"- {n_cells} cells occupying {cells['pctMapOccupied']:.0f}% of the map")
        print(f"- {cells['avgSurvival']:.0f} average cell survival, the oldest cell is {cells['maxSurvival']:.0f} old")
        print(f"- {cells['avgGenomeSize']:.0f} average genome size, the largest genome is {cells['maxGenomeSize']:.0f} large")

        print("\nCurrent average molecule concentrations:")
        for name, item in mols.items():
            print(f"- {name}: {item['avgIntConc']:.2f} intracellular, {item['avgExtConc']:.2f} extracellular")
        # fmt: on

        print("")
        return None

    def _randomly_move_cells(self, cells: list[Cell]):
        for cell in cells:
            x, y = cell.position
            nghbrhd = self._nghbrhd_map[(x, y)]
            pxls = nghbrhd[~self.cell_map[nghbrhd[:, 0], nghbrhd[:, 1]]]
            n = len(pxls)

            if n == 0:
                continue

            # move cells
            new_x, new_y = pxls[random.randint(0, n - 1)].tolist()
            self.cell_map[x, y] = False
            self.cell_map[new_x, new_y] = True
            cell.position = (new_x, new_y)

    def _replicate_cells_as_possible(
        self, parent_idxs: list[int]
    ) -> tuple[list[int], list[int]]:
        idx = 0
        child_idxs = []
        successful_parent_idxs = []
        for parent_idx in parent_idxs:
            parent = self.cells[parent_idx]

            x, y = parent.position
            nghbrhd = self._nghbrhd_map[(x, y)]
            pxls = nghbrhd[~self.cell_map[nghbrhd[:, 0], nghbrhd[:, 1]]]
            n = len(pxls)

            if n == 0:
                continue

            new_x, new_y = pxls[random.randint(0, n - 1)].tolist()
            self.cell_map[new_x, new_y] = True

            child = parent.copy(idx=idx, position=(new_x, new_y))
            successful_parent_idxs.append(parent_idx)
            child_idxs.append(idx)
            self.cells.append(child)
            idx += 1

        return successful_parent_idxs, child_idxs

    def _place_new_cells_in_random_positions(
        self, n_cells: int
    ) -> list[tuple[int, int]]:
        # available spots on map
        pxls = torch.argwhere(~self.cell_map)
        if n_cells > len(pxls):
            n_cells = len(pxls)

        # place cells on map
        idxs = random.sample(range(len(pxls)), k=n_cells)
        chosen = pxls[idxs]
        xs, ys = chosen.T.tolist()
        self.cell_map[xs, ys] = True
        return [(int(d[0]), int(d[1])) for d in chosen.tolist()]

    def _get_molecule_map(self, n: int, size: int, init: str) -> torch.Tensor:
        args = [n, size, size]
        if init == "zeros":
            return (torch.zeros(*args, dtype=self.dtype) + EPS).to(self.device)
        if init == "randn":
            return torch.randn(*args, dtype=self.dtype).abs().clamp(EPS).to(self.device)
        raise ValueError(
            f"Didnt recognize mol_map_init={init}."
            " Should be one of: 'zeros', 'randn'."
        )

    def _get_conv(self, mol_diff_rate: float) -> torch.nn.Conv2d:
        if mol_diff_rate < 0.0:
            mol_diff_rate = -mol_diff_rate

        if mol_diff_rate > 1.0:
            mol_diff_rate = 1.0

        if mol_diff_rate == 0.0:
            a = 0.0
            b = 1.0
        else:
            d = 1 / mol_diff_rate
            a = 1 / (d + 8)
            b = d * a

        # fmt: off
        kernel = torch.tensor([[[
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
            dtype=self.dtype,
            device=self.device,
        )
        conv.weight = torch.nn.Parameter(kernel, requires_grad=False)
        return conv

    def _expand(self, t: torch.Tensor, n: int, d: int) -> torch.Tensor:
        pre = t.shape[slice(d)]
        post = t.shape[slice(d + 1, t.dim())]
        zeros = torch.zeros(*pre, n, *post, dtype=self.dtype).to(self.device)
        return torch.cat([t, zeros], dim=d)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(map_size=%r,mol_halflife=%r,mol_diff_coef=%r,abs_temp=%r)" % (
            clsname,
            self.map_size,
            self.mol_halflife,
            self.mol_diff_coef,
            self.abs_temp,
        )

