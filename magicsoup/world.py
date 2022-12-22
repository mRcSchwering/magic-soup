from typing import Optional, Any
import logging
import random
from itertools import product
import math
import torch
from magicsoup.constants import EPS
from magicsoup.containers import Molecule, Cell, Protein
from magicsoup.util import moore_nghbrhd
from magicsoup.kinetics import Kinetics
from magicsoup.genetics import Genetics


_log = logging.getLogger(__name__)


class World:
    """
    World definition with map that stores molecule/signal concentrations and cell
    positions. It also defines how all proteins of all cells integrate signals
    of their respective environments.

    - `molecules` list of all molecules that are part of this simulation
    - `map_size` number of pixels in x- and y-direction
    - `mol_halflife` Half life of all molecules. Assuming each time step is a second, values around 1e5 are realistic.
      Must be > 0.0.
    - `mol_diff_coef` Diffusion coefficient for all molecules. Assuming one time step is a second and one pixel
      has a size of 10um, values around 1e-8 are realistic. 0.0 would mean no diffusion at all. Diffusion rate reaches
      its maximum at around 1e-6 where all molecules are equally spread out around the pixels' Moor's neighborhood with
      each time step.
    - `mol_map_init` How to initialize molecule maps (`randn` or `zeros`). `zeros` are not actually 0.0 but a small positive
      value instead.
    - `abs_temp` Absolute temperature (K). Affects entropy term of reaction energy (i.e. in which direction a reaction will occur)
    - `dtype` pytorch dtype to use for tensors (see pytorch docs)
    - `device` pytorch device to use for tensors (e.g. `cpu` or `gpu`, see pytorch docs)

    The world map (for cells and molecules) is a square map with `map_size` pixels in both
    directions. But it is "circular" in that when you move over the border to the left, you re-appear
    on the right and vice versa (same for top and bottom). Each cell occupies one pixel. The cell
    experiences molecule concentrations on that pixel as external molecule concentrations. Additionally,
    it has its internal molecule concentrations.

    There are different tensors representing the environment of the world and its cells:

    - `cell_map` Map referencing which pixels are occupied by a cell (bool 2d tensor)
    - `molecule_map` Map referencing molecule concentrations (3d tensor with molecules in dimension 0)
    - `cell_molecules` Intracellular molecule concentrations (2d tensor with cells in dimension 0, molecules in dimension 1)
    - `cell_survival` Numbe of survived time steps for every living cell (1d tensor)

    To gather all information for a particular cell use `get_cell` or `get_cells`.

    For `dtype` and `device` see pytorch's documentation. You can speed up computation by moving calculations
    to a GPU. If available you can use `dtype=torch.float16` which can speed up computations further.
    """

    def __init__(
        self,
        genetics: Genetics,
        map_size=128,
        mol_halflife=100_000,
        mol_diff_coef=1e-8,
        mol_map_init="randn",
        abs_temp=310.0,
        device="cpu",
        dtype=torch.float,
    ):
        self.genetics = genetics
        self.map_size = map_size
        self.abs_temp = abs_temp
        self.mol_halflife = mol_halflife
        self.mol_degrad = math.exp(-math.log(2) / mol_halflife)
        self.mol_diff_coef = mol_diff_coef
        self.dtype = dtype
        self.device = device
        self.torch_kwargs = {"dtype": dtype, "device": device}

        self.n_molecules = len(self.genetics.molecules)
        for idx, mol in enumerate(self.genetics.molecules):
            mol.int_idx = idx
            mol.ext_idx = self.n_molecules + idx
        self._int_mol_idxs = list(range(self.n_molecules))
        self._ext_mol_idxs = list(range(self.n_molecules, self.n_molecules * 2))

        self._diffuse = self._get_conv(mol_diff_rate=mol_diff_coef * 1e6)
        self._nghbrhd_map = {
            (x, y): torch.tensor(moore_nghbrhd(x, y, map_size))
            for x, y in product(range(map_size), range(map_size))
        }

        self.cells: list[Cell] = []
        self.kinetics = Kinetics(n_signals=2 * self.n_molecules)

        self.cell_map = self._tensor(self.map_size, self.map_size, dtype=torch.bool)
        self.cell_survival = self._tensor(0)
        self.cell_molecules = self._tensor(0, self.n_molecules)
        self.molecule_map = self._get_molecule_map(
            n=self.n_molecules, size=self.map_size, init=mol_map_init
        )

        _log.info("Major world tensors are running on %s as %s", device, dtype)
        _log.debug(
            "Instantiating world with size %i and %i molecule species",
            map_size,
            self.n_molecules,
        )

    def get_cell(
        self, by_idx: Optional[int] = None, by_label: Optional[str] = None
    ) -> Cell:
        """Get a cell with information about its current environment"""
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

        return cell

    def add_random_cells(self, genomes: list[str]):
        """
        Create new cells to be placed randomly on the map.

        Each cell will be placed randomly on the map and receive the molecule
        concentration of the pixel where it was added.

        If there are less pixels left on the cell map than cells you want to add,
        only the remaining pixels will be filled with new cells.
        """
        _log.debug("Adding %i random cells", len(genomes))
        if len(genomes) == 0:
            return

        new_positions = self._place_new_cells_in_random_positions(n_cells=len(genomes))
        n_new_cells = len(new_positions)

        if n_new_cells == 0:
            return

        cells: list[Cell] = []
        prot_lens = []
        new_idxs = []
        new_params: list[tuple[int, int, Protein]] = []
        run_idx = len(self.cells)
        for genome, pos in zip(genomes, new_positions):
            proteome = self.genetics.get_proteome(seq=genome)
            prot_lens.append(len(proteome))
            new_idxs.append(run_idx)
            cell = Cell(idx=run_idx, genome=genome, proteome=proteome, position=pos)
            cells.append(cell)
            for prot_i, prot in enumerate(proteome):
                new_params.append((run_idx, prot_i, prot))
            run_idx += 1

        self.cell_survival = self._expand(t=self.cell_survival, n=n_new_cells, d=0)
        self.cell_molecules = self._expand(t=self.cell_molecules, n=n_new_cells, d=0)
        self.kinetics.increase_max_cells(by_n=n_new_cells)
        self.kinetics.increase_max_proteins(max_n=max(prot_lens))
        self.kinetics.set_cell_params(cell_prots=new_params)

        # cell is supposed to have the same concentrations as the pxl it lives on
        xs, ys = list(map(list, zip(*new_positions)))
        self.cell_molecules[new_idxs, :] = self.molecule_map[:, xs, ys].T

    def replicate_cells(self, parent_idxs: list[int]) -> tuple[list[int], list[int]]:
        """
        Replicate existing cells.

        Each cell will be placed randomly next to its parent (Moore's neighborhood)
        and will receive the same molecule concentrations as its parent.

        If every pixel in the cells' Moore neighborhood is taken
        the cell will not replicate.
        """
        _log.debug("Replicating %i cells", len(parent_idxs))
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

        # cell is supposed to have the same concentrations as the parent had
        self.cell_molecules[child_idxs, :] = self.cell_molecules[succ_parent_idxs, :]

        return succ_parent_idxs, child_idxs

    def update_cells(self, genomes: list[str], idxs: list[int]):
        """
        Update existing cells with new genomes and proteomes.

        - `cells` cells with that need to be updated
        """
        _log.debug("Updating %i cells", len(genomes))
        if len(genomes) != len(idxs):
            raise ValueError(
                "Genomes and idxs represent the same list of cells."
                f" But now they don't have the same length. genomes: {len(genomes)}, idxs: {len(idxs)}."
            )

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
        Remove cells and spill out their molecule contents onto the pixels
        they used to live on.
        """
        _log.debug("Killing %i cells", len(cell_idxs))
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
        
        If every pixel in the cells' Moore neighborhood is taken
        the cell will not be moved.
        """
        _log.debug("Moving %i cells", len(cell_idxs))

        if len(cell_idxs) == 0:
            return

        cells = [self.cells[i] for i in cell_idxs]
        self._randomly_move_cells(cells=cells)

    @torch.no_grad()
    def diffuse_molecules(self):
        """Let molecules in world map diffuse by 1 time step"""
        _log.debug("Diffusing %i molecule species", int(self.molecule_map.shape[0]))

        before = self.molecule_map.unsqueeze(1)
        after = self._diffuse(before)
        self.molecule_map = torch.squeeze(after, 1)

    def degrade_molecules(self):
        """Degrade molecules in world map and cells by 1 time step"""
        _log.debug("Degrading %i molecule species", int(self.molecule_map.shape[0]))

        self.molecule_map = self.molecule_map * self.mol_degrad
        self.cell_molecules = self.cell_molecules * self.mol_degrad

    def increment_cell_survival(self):
        """Increment number of currently living cells' time steps by 1"""
        _log.debug("Incrementing cell survival of %i cells", len(self.cells))

        idxs = list(range(len(self.cells)))
        self.cell_survival[idxs] += 1

    def enzymatic_activity(self):
        """Catalyze reactions for 1 time step and update molecule concentrations"""
        _log.debug("Run enzymatic activity with %i cells", len(self.cells))

        xs, ys = list(map(list, zip(*[d.position for d in self.cells])))
        X = torch.cat([self.cell_molecules, self.molecule_map[:, xs, ys].T], dim=1)

        Xd = self.kinetics.integrate_signals(X=X)
        X += Xd

        self.molecule_map[:, xs, ys] = X[:, self._ext_mol_idxs].T
        self.cell_molecules = X[:, self._int_mol_idxs]

    def summary(self, as_dict=False) -> Optional[dict]:
        """Get current world summary"""
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
            return {"molecules": mols, "cells": cells}

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
                _log.info(
                    "Wanted to move cell at %i, %i"
                    " but no pixel in the Moore neighborhood was free."
                    " So, cell wasn't moved.",
                    x,
                    y,
                )
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
                _log.info(
                    "Wanted to replicate cell next to %i, %i"
                    " but no pixel in the neighborhood was available."
                    " So, cell wasn't able to replicate.",
                    x,
                    y,
                )
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
            _log.info(
                "Wanted to add %i new random cells"
                " but only %i pixels left on map."
                " So, only %i cells were added.",
                n_cells,
                len(pxls),
                len(pxls),
            )
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
            return self._tensor(*args) + EPS
        if init == "randn":
            return torch.randn(*args, **self.torch_kwargs).abs().clamp(EPS)
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
            **self.torch_kwargs,
        )
        conv.weight = torch.nn.Parameter(kernel, requires_grad=False)
        return conv

    def _expand(self, t: torch.Tensor, n: int, d: int) -> torch.Tensor:
        pre = t.shape[slice(d)]
        post = t.shape[slice(d + 1, t.dim())]
        zeros = self._tensor(*pre, n, *post)
        return torch.cat([t, zeros], dim=d)

    def _tensor(self, *args, **kwargs) -> torch.Tensor:
        return torch.zeros(*args, **{**self.torch_kwargs, **kwargs})

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(map_size=%r,mol_halflife=%r,mol_diff_coef=%r,abs_temp=%r,device=%r,dtype=%r)"
            % (
                clsname,
                self.map_size,
                self.mol_halflife,
                self.mol_diff_coef,
                self.abs_temp,
                self.device,
                self.dtype,
            )
        )
