from typing import Optional, Any
import logging
import random
from itertools import product
import math
import torch
from magicsoup.constants import GAS_CONSTANT, EPS
from magicsoup.containers import Molecule, Cell, Protein
from magicsoup.util import moore_nghbrhd
from magicsoup.kinetics import integrate_signals, calc_cell_params


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
        molecules: list[Molecule],
        map_size=128,
        mol_halflife=100_000,
        mol_diff_coef=1e-8,
        mol_map_init="randn",
        abs_temp=310.0,
        device="cpu",
        dtype=torch.float,
        n_workers=4,
    ):
        self.molecules = molecules
        self.map_size = map_size
        self.abs_temp = abs_temp
        self.mol_halflife = mol_halflife
        self.mol_degrad = math.exp(-math.log(2) / mol_halflife)
        self.mol_diff_coef = mol_diff_coef
        self.dtype = dtype
        self.device = device
        self.n_workers = n_workers
        self.torch_kwargs = {"dtype": dtype, "device": device}

        self.n_molecules = len(molecules)
        for idx, mol in enumerate(molecules):
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

        self.cell_map = self._tensor(self.map_size, self.map_size, dtype=torch.bool)
        self.cell_survival = self._tensor(0)
        self.cell_molecules = self._tensor(0, self.n_molecules)
        self.molecule_map = self._get_molecule_map(
            n=self.n_molecules, size=self.map_size, init=mol_map_init
        )

        self.affinities = self._tensor(0, 0, 2 * self.n_molecules)
        self.velocities = self._tensor(0, 0)
        self.energies = self._tensor(0, 0)
        self.stoichiometry = self._tensor(0, 0, 2 * self.n_molecules)
        self.effectors = self._tensor(0, 0, 2 * self.n_molecules)

        _log.info("Major world tensors are running on %s as %s", device, dtype)
        _log.debug(
            "Instantiating world with size %i and %i molecule species",
            map_size,
            len(molecules),
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
        cell.ext_molecules = self._get_cell_ext_molecules(cells=[cell])[0, :]
        return cell

    def get_cells(self, by_idxs: list[int]) -> list[Cell]:
        """More performant than calling `get_cell` multiple times"""
        _log.debug("Getting %i cells by indices", len(by_idxs))

        if len(by_idxs) == 0:
            return []

        cells = [self.cells[i] for i in by_idxs]
        ext_molecules = self._get_cell_ext_molecules(cells=cells)
        for ci, cell in enumerate(cells):
            cell.int_molecules = self.cell_molecules[cell.idx, :]
            cell.ext_molecules = ext_molecules[ci, :]
        return cells

    def add_random_cells(self, cells: list[Cell]):
        """
        Add new random cells with new genomes and proteomes.

        - `cells` new cells with new geneomes and proteomes

        Each cell will be placed randomly on the map and receive the molecule
        concentration of the pixel where it was added.

        If there are less pixels left on the cell map than cells you want to add,
        only the remaining pixels will be filled with new cells.
        """
        _log.debug("Adding %i random cells", len(cells))

        if len(cells) == 0:
            return

        new_idxs = self._place_new_cells_in_random_positions(cells=cells)

        n_new_cells = len(new_idxs)
        if n_new_cells == 0:
            return

        self._expand_max_cells(by_n=n_new_cells)

        new_cells = [self.cells[i] for i in new_idxs]
        self._expand_max_proteins(max_n=max(len(d.proteome) for d in new_cells))

        # cell is supposed to have the same concentrations as the pxl it lives on
        new_positions = [self.cells[i].position for i in new_idxs]
        xs = []
        ys = []
        for x, y in new_positions:
            xs.append(x)
            ys.append(y)
        self.cell_molecules[new_idxs, :] = self.molecule_map[:, xs, ys].T

        cell_prots: list[tuple[int, int, Optional[Protein]]] = []
        for cell in new_cells:
            for prot_i, prot in enumerate(cell.proteome):
                cell_prots.append((cell.idx, prot_i, prot))

        calc_cell_params(
            cell_prots=cell_prots,
            n_signals=2 * self.n_molecules,
            Km=self.affinities,
            Vmax=self.velocities,
            E=self.energies,
            N=self.stoichiometry,
            A=self.effectors,
        )

    def replicate_cells(self, parent_idxs) -> tuple[list[int], list[int]]:
        """
        Replicate existing cells with new genomes and proteomes.

        - `cells` new cells with new geneomes and proteomes, but with parent
          position and idexes.

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

        self._expand_max_cells(by_n=n_new_cells)

        # cell is supposed to have the same concentrations as the parent had
        self.cell_molecules[child_idxs, :] = self.cell_molecules[succ_parent_idxs, :]

        self.affinities[child_idxs] = self.affinities[succ_parent_idxs]
        self.velocities[child_idxs] = self.velocities[succ_parent_idxs]
        self.energies[child_idxs] = self.energies[succ_parent_idxs]
        self.stoichiometry[child_idxs] = self.stoichiometry[succ_parent_idxs]
        self.effectors[child_idxs] = self.effectors[succ_parent_idxs]

        return succ_parent_idxs, child_idxs

    def update_cells(self, cells: list[Cell]):
        """
        Update existing cells with new genomes and proteomes.

        - `cells` cells with that need to be updated
        """
        _log.debug("Updating %i cells", len(cells))

        cell_prots: list[tuple[int, int, Optional[Protein]]] = []
        for new_cell in cells:
            ci = new_cell.idx
            oldprot = self.cells[ci].proteome
            newprot = new_cell.proteome
            n_new = len(newprot)
            n_old = len(oldprot)
            n = min(n_old, n_new)
            for pi, (np, op) in enumerate(zip(newprot[:n], oldprot[:n])):
                if np != op:
                    cell_prots.append((ci, pi, np))
            if n_old > n_new:
                cell_prots.extend((ci, i, None) for i in range(n, n_old))

        self._expand_max_proteins(max_n=max(len(d.proteome) for d in cells))

        calc_cell_params(
            cell_prots=cell_prots,
            n_signals=2 * self.n_molecules,
            Km=self.affinities,
            Vmax=self.velocities,
            E=self.energies,
            N=self.stoichiometry,
            A=self.effectors,
        )

    def kill_cells(self, cell_idxs: list[int]):
        """
        Remove cells and spill out their molecule contents onto the pixels
        they used to live on.
        """
        _log.debug("Killing %i cells", len(cell_idxs))

        if len(cell_idxs) == 0:
            return

        cells = [self.cells[i] for i in cell_idxs]

        xs = []
        ys = []
        for cell in cells:
            x, y = cell.position
            xs.append(x)
            ys.append(y)
        self.cell_map[xs, ys] = False

        spillout = self.cell_molecules[cell_idxs, :]
        self._add_cell_ext_molecules(cells=cells, molecules=spillout)
        self._remove_cell_rows(idxs=cell_idxs)

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

        ext_molecules = self._get_cell_ext_molecules(cells=self.cells)
        X = torch.cat([self.cell_molecules, ext_molecules], dim=1)

        _log.debug(
            "Integrate signals with (c, p, s)=(%i, %i, %i)",
            int(X.shape[0]),
            int(self.velocities.shape[1]),
            int(X.shape[1]),
        )

        Xd = integrate_signals(
            X=X,
            Km=self.affinities,
            Vmax=self.velocities,
            Ke=torch.exp(-self.energies / self.abs_temp / GAS_CONSTANT),
            N=self.stoichiometry,
            A=self.effectors,
        )
        X += Xd

        self.cell_molecules = X[:, self._int_mol_idxs]
        self._put_cell_ext_molecules(
            cells=self.cells, molecules=X[:, self._ext_mol_idxs]
        )

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
        for idx, mol in enumerate(self.molecules):
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

    def _place_new_cells_in_random_positions(self, cells: list[Cell]) -> list[int]:
        # available spots on map
        pxls = torch.argwhere(~self.cell_map)
        n_cells = len(cells)
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
        xs, ys = pxls[idxs].T.tolist()
        self.cell_map[xs, ys] = True

        new_idx = len(self.cells)
        new_idxs: list[int] = []
        for x, y, cell in zip(xs, ys, cells):

            # set new cell position
            cell.position = (x, y)

            # set new cell index
            cell.idx = new_idx
            self.cells.append(cell)
            new_idxs.append(new_idx)
            new_idx += 1

        return new_idxs

    def _get_cell_ext_molecules(self, cells: list[Cell]) -> torch.Tensor:
        xs = []
        ys = []
        for cell in cells:
            x, y = cell.position
            xs.append(x)
            ys.append(y)
        return self.molecule_map[:, xs, ys].T

    def _add_cell_ext_molecules(self, cells: list[Cell], molecules: torch.Tensor):
        xs = []
        ys = []
        for cell in cells:
            x, y = cell.position
            xs.append(x)
            ys.append(y)
        self.molecule_map[:, xs, ys] += molecules.T

    def _put_cell_ext_molecules(self, cells: list[Cell], molecules: torch.Tensor):
        xs = []
        ys = []
        for cell in cells:
            x, y = cell.position
            xs.append(x)
            ys.append(y)
        self.molecule_map[:, xs, ys] = molecules.T

    def _get_molecule_map(self, n: int, size: int, init: str) -> torch.Tensor:
        args = [n, size, size]
        if init == "zeros":
            return self._tensor(*args) + EPS
        if init == "randn":
            return torch.randn(*args, **self.torch_kwargs).abs() + EPS
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

    def _expand_max_cells(self, by_n: int):
        self.cell_survival = self._expand(t=self.cell_survival, n=by_n, d=0)
        self.cell_molecules = self._expand(t=self.cell_molecules, n=by_n, d=0)
        self.affinities = self._expand(t=self.affinities, n=by_n, d=0)
        self.velocities = self._expand(t=self.velocities, n=by_n, d=0)
        self.energies = self._expand(t=self.energies, n=by_n, d=0)
        self.stoichiometry = self._expand(t=self.stoichiometry, n=by_n, d=0)
        self.effectors = self._expand(t=self.effectors, n=by_n, d=0)

    def _remove_cell_rows(self, idxs: list[int]):
        keep = torch.ones(self.cell_survival.shape[0], dtype=torch.bool)
        keep[idxs] = False
        self.cell_survival = self.cell_survival[keep]
        self.cell_molecules = self.cell_molecules[keep]
        self.affinities = self.affinities[keep]
        self.velocities = self.velocities[keep]
        self.energies = self.energies[keep]
        self.stoichiometry = self.stoichiometry[keep]
        self.effectors = self.effectors[keep]

    def _expand_max_proteins(self, max_n: int):
        n_prots = int(self.affinities.shape[1])
        if max_n > n_prots:
            by_n = max_n - n_prots
            self.affinities = self._expand(t=self.affinities, n=by_n, d=1)
            self.velocities = self._expand(t=self.velocities, n=by_n, d=1)
            self.energies = self._expand(t=self.energies, n=by_n, d=1)
            self.stoichiometry = self._expand(t=self.stoichiometry, n=by_n, d=1)
            self.effectors = self._expand(t=self.effectors, n=by_n, d=1)

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
            "%s(molecules=%r,map_size=%r,mol_halflife=%r,mol_diff_coef=%r,abs_temp=%r,device=%r,dtype=%r)"
            % (
                clsname,
                self.molecules,
                self.map_size,
                self.mol_halflife,
                self.mol_diff_coef,
                self.abs_temp,
                self.device,
                self.dtype,
            )
        )
