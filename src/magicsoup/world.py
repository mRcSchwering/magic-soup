from typing import Optional
import random
from itertools import product
import math
import pickle
from pathlib import Path
import torch
from magicsoup.containers import Cell, Protein, Chemistry
from magicsoup.util import moore_nghbrhd
from magicsoup.kinetics import Kinetics
from magicsoup.genetics import Genetics


class World:
    """
    This is the main object for running the simulation.
    It holds all information and includes all methods for advancing the simulation.

    - `chemistry` The chemistry object that defines molecule species and reactions for this simulation.
    - `map_size` Size of world map as number of pixels in x- and y-direction
    - `abs_temp` Absolute temperature in Kelvin will influence the free Gibbs energy calculation of reactions.
      Higher temperature will give the reaction quotient term higher importance.
    - `mol_map_init` How to initialize molecule maps (`randn` or `zeros`).
      `randn` is normally distributed N(10, 1), `zeros` is all zeros.
    - `start_codons` start codons which start a coding sequence (translation only happens within coding sequences)
    - `stop_codons` stop codons which stop a coding sequence (translation only happens within coding sequences)
    - `device` Device to use for tensors (see [pytorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)).
      This has to be the same device that is used by `world`.

    Most attributes on this class describe the current state of molecules and cells.
    Whenever molecules are listed or represented in one dimension, they are ordered the same way as in `chemistry.molecules`.
    Likewise, cells are always ordered the same way as in `world.cells` (see below).
    The index of a certain cell is the index of that cell in `world.cells`.
    It is the same index as `cell.idx` of a cell object you retrieved with `world.get_cell()`.
    But whenever an operation modifies the number of cells (like `world.kill_cells()` or `world.replicate_cells()`),
    cells get new indexes. Here are the most important attributes:

    - `cells` A list of cell objects. These cell objects hold e.g. each cell's genome and proteome.
    - `cell_map` Boolean 2D tensor referencing which pixels are occupied by a cell.
      Dimension 0 represents the x, dimension 1 y.
    - `molecule_map` Float 3D tensor describing how many molecules (in mol) of each molecule species exist on every pixel in this world.
      Dimension 0 describes the molecule species. They are in the same order as `chemistry.molecules`.
      Dimension 1 represents x, dimension 2 y.
      So, `world.molecule_map[0, 1, 2]` is number of molecules of the 0th molecule species on pixel 1, 2.
    - `cell_molecules` Float 2D tensor describing the number of molecules (in mol) for each molecule species in each cell.
      Dimension 0 is the cell index. It is the same as in `world.cells` and the same as on a cell object (`cell.idx`).
      Dimension 1 describes the molecule species. They are in the same order as `chemistry.molecules`.
      So, `world.cell_molecules[0, 1]` represents how many mol of the 1st molecule species the 0th cell contains.
    - `cell_survival` Integer 1D tensor describing how many time steps each cell survived.
      Cells are in the same as in `world.cells` and the same as on a cell object (`cell.idx`).
    - `cell_divisions` Integer 1D tensor describing how many times each cell replicated.
      Cells are in the same as in `world.cells` and the same as on a cell object (`cell.idx`).

    Then, there are methods for advancing the simulation:

    - `add_random_cells` add new cells and place them randomly on the map
    - `replicate_cells` replicate existing cells
    - `update_cells` update existing cells if their genome has changed
    - `kill_cells` kill existing cells
    - `move_cells` move existing cells to a random position in their Moore's neighborhood
    - `diffuse_molecules` let molecules diffuse and permeate by one time step
    - `degrade_molecules` let molecules degrade by one time step
    - `increment_cell_survival` increment `world.cell_survival` by 1
    - `enzymatic_activity` let cell proteins work for one time step
      
    If you want to get a cell with all information about its contents and its current environment use `world.get_cell()`.
    During the simulation you should however work directly with the tensors mentioned above for performance reasons.

    Furthermore, there are methods for saving and loading a simulation.
    For any new simulation use `world.save()` once to save the whole world object (with chemistry, genetics, kinetics)
    to a pickle file. You can restore it with `World.from_file()` later on.
    Then, to save the world's state you can use `world.save_state()`.
    This is a quick, lightweight save, but it only saves things that change during the simulation.
    Use `world.load_state()` to re-load a certain state.

    Finally, the `world` object carries `world.genetics`, `world.kinetics`, and `world.genetics.chemistry`
    (which is just a reference to the chemistry object that was used when initializing `world`).
    Usually, you don't need to touch them.
    But, if you want to override them or look into some details, see the docstrings of their classes for more information.
    """

    def __init__(
        self,
        chemistry: Chemistry,
        map_size=128,
        abs_temp=310.0,
        mol_map_init="randn",
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        device="cpu",
    ):
        if not torch.cuda.is_available():
            device = "cpu"

        self.device = device
        self.map_size = map_size
        self.abs_temp = abs_temp

        self.genetics = Genetics(
            chemistry=chemistry, start_codons=start_codons, stop_codons=stop_codons,
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

        self._nghbrhd_map = {
            (x, y): torch.tensor(moore_nghbrhd(x, y, map_size))
            for x, y in product(range(map_size), range(map_size))
        }

        self.cells: list[Cell] = []
        self.kinetics = Kinetics(
            molecules=chemistry.molecules, abs_temp=abs_temp, device=self.device,
        )

        self.cell_map = torch.zeros(map_size, map_size).to(device).bool()
        self.cell_survival = torch.zeros(0).to(device).int()
        self.cell_divisions = torch.zeros(0).to(device).int()
        self.cell_molecules = torch.zeros(0, self.n_molecules).to(device)
        self.molecule_map = self._get_molecule_map(
            n=self.n_molecules, size=self.map_size, init=mol_map_init
        ).to(self.device)

    # TODO: does it make sense to keep cell indexes constant (even after deleting cells)
    #       and only adding new indexes?

    def get_cell(
        self,
        by_idx: Optional[int] = None,
        by_label: Optional[str] = None,
        by_position: Optional[tuple[int, int]] = None,
    ) -> Cell:
        """
        Get a cell with information about its current environment
        
        - `by_idx` get cell by cell index (`cell.idx`)
        - `by_label` get cell by cell label (`cell.label`)
        - `by_position` get cell by position (x, y)

        When accessing `world.cells` directly, the cell object will not have all information.
        For performance reasons most cell attributes are maintained in tensors during the simulation.
        Only genome and proteome are maintained during the simulation.
        When you call `world.get_cell()` all missing information will be added to the object.
        """
        if by_idx is not None:
            idx = by_idx
        if by_label is not None:
            cell_labels = [d.label for d in self.cells]
            try:
                idx = cell_labels.index(by_label)
            except ValueError as err:
                raise ValueError(f"Cell {by_label} not found") from err
        if by_position is not None:
            cell_positions = [d.position for d in self.cells]
            try:
                idx = cell_positions.index(by_position)
            except ValueError as err:
                raise ValueError(f"Cell at {by_position} not found") from err

        cell = self.cells[idx]
        cell.int_molecules = self.cell_molecules[idx, :]
        cell.ext_molecules = self.molecule_map[:, cell.position[0], cell.position[1]]
        cell.n_survived_steps = self.cell_survival[idx].item()
        cell.n_replications = self.cell_divisions[idx].item()
        return cell

    def add_random_cells(self, genomes: list[str]) -> list[int]:
        """
        Create new cells and place them on randomly on the map.

        - `genomes` list of genomes of the newly added cells

        Returns the indexes of successfully added cells.

        Each cell will be placed randomly on the map and receive half the molecules of the pixel where it was added.
        If there are less pixels left on the cell map than cells you want to add,
        only the remaining pixels will be filled with new cells.

        All lists and tensors that reference cells will be updated.
        """
        genomes = [d for d in genomes if len(d) > 0]
        if len(genomes) == 0:
            return []

        xs, ys = self._find_free_random_positions(n_cells=len(genomes))
        if len(xs) == 0:
            return []

        prot_lens = []
        new_idxs = []
        new_params: list[tuple[int, int, Protein]] = []
        next_idx = len(self.cells)
        n_new_cells = 0
        new_xs = []
        new_ys = []
        for genome in genomes:
            proteome = self.genetics.get_proteome(seq=genome)
            n_proteins = len(proteome)
            if n_proteins == 0:
                continue

            prot_lens.append(n_proteins)
            new_idxs.append(next_idx)
            for prot_i, prot in enumerate(proteome):
                new_params.append((next_idx, prot_i, prot))

            x = xs[n_new_cells]
            y = ys[n_new_cells]
            new_xs.append(x)
            new_ys.append(y)

            cell = Cell(idx=next_idx, genome=genome, proteome=proteome, position=(x, y))
            self.cells.append(cell)

            next_idx += 1
            n_new_cells += 1

        if n_new_cells == 0:
            return []

        self.cell_survival = self._expand(t=self.cell_survival, n=n_new_cells, d=0)
        self.cell_divisions = self._expand(t=self.cell_divisions, n=n_new_cells, d=0)
        self.cell_molecules = self._expand(t=self.cell_molecules, n=n_new_cells, d=0)
        self.kinetics.increase_max_cells(by_n=n_new_cells)
        self.kinetics.increase_max_proteins(max_n=max(prot_lens))
        self.kinetics.set_cell_params(cell_prots=new_params)

        # occupy positions
        self.cell_map[new_xs, new_ys] = True

        # cell is picking up half the molecules of the pxl it is born on
        pickup = self.molecule_map[:, new_xs, new_ys] * 0.5
        self.cell_molecules[new_idxs, :] += pickup.T
        self.molecule_map[:, new_xs, new_ys] -= pickup

        return new_idxs

    def replicate_cells(self, parent_idxs: list[int]) -> list[tuple[int, int]]:
        """
        Bring cells to replicate

        - `parent_idxs` Cell indexes of the cells that should replicate.

        Returns a list of tuples of parent and child cell indexes for all successfully replicated cells.

        In this simulation the cell that was brought to replicate is the parent.
        It still lives on the same pixel.
        The child is the cell that was newly added next to the parent.
        The child will start with 0 survived steps and 0 replications,
        while the parent keeps its number of survived steps and increments its number of replications.
        Half the parent's molecules will be given to the child.

        Each child will be placed randomly next to its parent (Moore's neighborhood).
        If every pixel in the parent's Moore's neighborhood is taken the cell will not replicate.

        All lists and tensors that reference cells will be updated.
        """
        if len(parent_idxs) == 0:
            return []

        succ_parent_idxs, child_idxs = self._replicate_cells_as_possible(
            parent_idxs=parent_idxs
        )
        self.cell_divisions[succ_parent_idxs] += 1

        n_new_cells = len(child_idxs)
        if n_new_cells == 0:
            return []

        self.cell_survival = self._expand(t=self.cell_survival, n=n_new_cells, d=0)
        self.cell_divisions = self._expand(t=self.cell_divisions, n=n_new_cells, d=0)
        self.cell_molecules = self._expand(t=self.cell_molecules, n=n_new_cells, d=0)
        self.kinetics.increase_max_cells(by_n=n_new_cells)
        self.kinetics.copy_cell_params(from_idxs=succ_parent_idxs, to_idxs=child_idxs)

        # cell shares molecules with parent
        self.cell_molecules[child_idxs] = self.cell_molecules[succ_parent_idxs]
        self.cell_molecules[child_idxs + succ_parent_idxs] *= 0.5

        return list(zip(succ_parent_idxs, child_idxs))

    def update_cells(self, genome_idx_pairs: list[tuple[str, int]]):
        """
        Update existing cells with new genomes.

        - `genome_idx_pairs` list of tuples of genomes and cell indexes

        The indexes refer to the index of each cell that is changed.
        The genomes refer to the genome of each cell that is changed.
        `world.cells` will be updated with new genomes and proteomes.
        """
        # TODO: 3 steps where only 1 proc is 100% active total about 25s
        #       then 1 step where all procs are 50% active (probably enzymatic_activity())
        #       total about 7k cells with avg genome size 4000
        #       needs to speed up....

        if len(genome_idx_pairs) == 0:
            return

        kill_idxs: list[int] = []
        prot_lens: list[int] = []
        set_params: list[tuple[int, int, Protein]] = []
        unset_params: list[tuple[int, int]] = []
        for genome, idx in genome_idx_pairs:
            if len(genome) == 0:
                kill_idxs.append(idx)
                continue

            cell = self.cells[idx]
            newprot = self.genetics.get_proteome(seq=genome)
            if len(newprot) == 0:
                kill_idxs.append(idx)
                continue

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
        self.kill_cells(cell_idxs=kill_idxs)

    def kill_cells(self, cell_idxs: list[int]):
        """
        Remove existing cells
        
        - `cell_idxs` indexes of the cells that should die
        
        Cells that are killed dump their molecule contents onto the pixel they used to live on.
        
        Cells will be removed from all lists and tensors that reference cells.
        Thus, after killing cells the index of some living cells will be updated.
        E.g. if there are 10 cells and you kill the cell with index 8 (the 9th cell),
        the cell that used to have index 9 (10th cell) will now have index 9.
        """
        if len(cell_idxs) == 0:
            return

        xs, ys = list(map(list, zip(*[self.cells[i].position for i in cell_idxs])))
        self.cell_map[xs, ys] = False

        spillout = self.cell_molecules[cell_idxs, :]
        self.molecule_map[:, xs, ys] += spillout.T

        keep = torch.ones(self.cell_survival.size(0), dtype=torch.bool)
        keep[cell_idxs] = False
        self.cell_survival = self.cell_survival[keep]
        self.cell_divisions = self.cell_divisions[keep]
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
        Move cells to a random position in their Moore's neighborhood

        - `cell_idxs` indexes of cells that should be moved
        
        If every pixel in the cells' Moore neighborhood is taken the cell will not be moved.
        `world.cell_map` will be updated.
        """
        if len(cell_idxs) == 0:
            return

        cells = [self.cells[i] for i in cell_idxs]
        self._randomly_move_cells(cells=cells)

    def enzymatic_activity(self):
        """
        Catalyze reactions for one time step

        This includes molecule transport into or out of the cell.
        `world.molecule_map` and `cell_molecules` are updated.
        """
        if len(self.cells) == 0:
            return

        xs, ys = list(map(list, zip(*[d.position for d in self.cells])))
        X = torch.cat([self.cell_molecules, self.molecule_map[:, xs, ys].T], dim=1)
        X = self.kinetics.integrate_signals(X=X)

        self.molecule_map[:, xs, ys] = X[:, self._ext_mol_idxs].T
        self.cell_molecules = X[:, self._int_mol_idxs]

    @torch.no_grad()
    def diffuse_molecules(self):
        """
        Let molecules in molecule map diffuse and permeate through cell membranes
        by one time step.

        By how much each molcule species diffuses around the world map and permeates
        into or out of cells if defined on the `Molecule` objects itself.
        See `Molecule` for more information.

        `world.molecule_map` and `cell_molecules` are updated.
        """
        for mol_i, diffuse in enumerate(self._diffusion):
            before = self.molecule_map[mol_i].unsqueeze(0).unsqueeze(1)
            after = diffuse(before)
            self.molecule_map[mol_i] = torch.squeeze(after, 0).squeeze(0)

        if len(self.cells) == 0:
            return

        xs, ys = list(map(list, zip(*[d.position for d in self.cells])))
        X = torch.cat([self.cell_molecules, self.molecule_map[:, xs, ys].T], dim=1)

        for mol_i, permeate in enumerate(self._permeation):
            d_int = X[:, mol_i] * permeate
            d_ext = X[:, mol_i + self.n_molecules] * permeate
            X[:, mol_i] += d_ext - d_int
            X[:, mol_i + self.n_molecules] += d_int - d_ext

        self.molecule_map[:, xs, ys] = X[:, self._ext_mol_idxs].T
        self.cell_molecules = X[:, self._int_mol_idxs]

    def degrade_molecules(self):
        """
        Degrade molecules in world map and cells by one time step.
        
        How quickly each molecule species degrades depends on its half life
        which is defined on the `Molecule` object of that species.
        
        `world.molecule_map` and `cell_molecules` are updated.
        """
        for mol_i, degrad in enumerate(self._mol_degrads):
            self.molecule_map[mol_i] *= degrad
            self.cell_molecules[:, mol_i] *= degrad

    def increment_cell_survival(self):
        """
        Increment `world.cell_survival` by 1.
        """
        idxs = list(range(len(self.cells)))
        self.cell_survival[idxs] += 1

    def save(self, rundir: Path, name="world.pkl"):
        """
        Write whole world object to pickle file

        - `rundir` directory of the pickle file
        - `name` name of the pickle file
        
        This is a big and slow save.
        It saves everything needed to restore the world with its chemistry and genetics.
        Use `World.from_file()` to restore it.
        For a small and quick save use `world.save_state()`.
        """
        # TODO: make this JSON, txt, (except for tensors), to make it possible
        #       to continue using a different language
        #       would need to organize domain factories differently to do that
        rundir.mkdir(parents=True, exist_ok=True)
        with open(rundir / name, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def from_file(self, rundir: Path, name="world.pkl") -> "World":
        """
        Restore previously saved world from pickle file.

        - `rundir` directory of the pickle file
        - `name` name of the pickle file

        The file had to be saved with `world.save()`.
        """
        with open(rundir / name, "rb") as fh:
            return pickle.load(fh)

    def save_state(self, statedir: Path):
        """
        Save current state only

        - `statedir` directory to store files in (there are multiple files per state)
        
        This is a small and quick save.
        It only saves things which change during the simulation.
        Restore a certain state with `world.load_state()`.
        
        To restore a world object you need to save it at least once with `world.save()`.
        Then, `world.save_state()` can be used to save different states of that world object.
        """
        statedir.mkdir(parents=True, exist_ok=True)
        torch.save(self.cell_molecules, statedir / "cell_molecules.pt")
        torch.save(self.cell_map, statedir / "cell_map.pt")
        torch.save(self.molecule_map, statedir / "molecule_map.pt")
        torch.save(self.cell_survival, statedir / "cell_survival.pt")
        torch.save(self.cell_divisions, statedir / "cell_divisions.pt")

        with open(statedir / "cells.fasta", "w") as fh:
            lines = [
                f">{d.idx} {d.label} ({d.position[0]},{d.position[1]})\n{d.genome}"
                for d in self.cells
            ]
            fh.write("\n".join(lines))

    def load_state(self, statedir: Path, update_cell_params=True):
        """
        Load a saved world state.

        - `statedir` directory that contains all files of that state
        - `update_cell_params` whether to update cell parameters as well.
          If you are only interested in the cells' genomes and molecules
          you can set this to `False` to make loading states faster.

        The state had to be saved with `world.save_state()` previously.
        """
        cell_molecules: torch.Tensor = torch.load(statedir / "cell_molecules.pt")
        self.cell_molecules = cell_molecules.to(self.device)
        cell_map: torch.Tensor = torch.load(statedir / "cell_map.pt")
        self.cell_map = cell_map.to(torch.bool).to(self.device)
        molecule_map: torch.Tensor = torch.load(statedir / "molecule_map.pt")
        self.molecule_map = molecule_map.to(self.device)
        cell_survival: torch.Tensor = torch.load(statedir / "cell_survival.pt")
        self.cell_survival = cell_survival.to(self.device).int()
        cell_divisions: torch.Tensor = torch.load(statedir / "cell_divisions.pt")
        self.cell_divisions = cell_divisions.to(self.device).int()

        with open(statedir / "cells.fasta", "r") as fh:
            text: str = fh.read()
            entries = [d.strip() for d in text.split(">") if len(d.strip()) > 0]

        genome_idx_pairs: list[tuple[str, int]] = []
        self.cells = []
        for entry in entries:
            descr, seq = entry.split("\n")
            names_part, pos_part = descr.split("(")
            x, y = pos_part.split(")")[0].split(",")
            names = names_part.split()
            idx = int(names[0].strip())
            pos = (int(x.strip()), int(y.strip()))
            label = names[1].strip() if len(names) > 1 else ""
            cell = Cell(idx=idx, label=label, genome=seq, proteome=[], position=pos)
            self.cells.append(cell)
            genome_idx_pairs.append((seq, idx))

        if update_cell_params:
            self.update_cells(genome_idx_pairs=genome_idx_pairs)

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
        idx = len(self.cells)
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

            child = parent.copy(idx=idx, position=(new_x, new_y), n_survived_steps=0)
            successful_parent_idxs.append(parent_idx)
            child_idxs.append(idx)
            self.cells.append(child)
            idx += 1

        return successful_parent_idxs, child_idxs

    def _find_free_random_positions(self, n_cells: int) -> tuple[list[int], list[int]]:
        # available spots on map
        pxls = torch.argwhere(~self.cell_map)
        if n_cells > len(pxls):
            n_cells = len(pxls)

        # place cells on map
        idxs = random.sample(range(len(pxls)), k=n_cells)
        chosen = pxls[idxs]
        xs, ys = chosen.T.tolist()
        return xs, ys

    def _get_molecule_map(self, n: int, size: int, init: str) -> torch.Tensor:
        args = [n, size, size]
        if init == "zeros":
            return torch.zeros(*args)
        if init == "randn":
            return torch.abs(torch.randn(*args) + 10.0)
        raise ValueError(
            f"Didnt recognize mol_map_init={init}."
            " Should be one of: 'zeros', 'randn'."
        )

    def _get_permeate(self, mol_perm_rate: float) -> float:
        if mol_perm_rate < 0.0:
            mol_perm_rate = -mol_perm_rate

        if mol_perm_rate > 1.0:
            mol_perm_rate = 1.0

        if mol_perm_rate == 0.0:
            return 0.0

        d = 1 / mol_perm_rate
        return 1 / (d + 1)

    def _get_diffuse(self, mol_diff_rate: float) -> torch.nn.Conv2d:
        if mol_diff_rate < 0.0:
            mol_diff_rate = -mol_diff_rate

        # TODO: mol_diff_rate > 1.0 could also mean expanding the kernel
        #       so that molecules can diffuse more than just 1 pxl per round
        if mol_diff_rate > 1.0:
            mol_diff_rate = 1.0

        if mol_diff_rate == 0.0:
            a = 0.0
            b = 1.0
        else:
            d = 1 / mol_diff_rate
            a = 1 / (d + 8)
            b = d * a
            b = b + 1.0 - (8 * a + b)  # try correcting inaccuracy

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
            device=self.device,
        )
        conv.weight = torch.nn.Parameter(kernel, requires_grad=False)
        return conv

    def _expand(self, t: torch.Tensor, n: int, d: int) -> torch.Tensor:
        pre = t.shape[slice(d)]
        post = t.shape[slice(d + 1, t.dim())]
        zeros = torch.zeros(*pre, n, *post, dtype=t.dtype).to(self.device)
        return torch.cat([t, zeros], dim=d)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(map_size=%r,abs_temp=%r)" % (clsname, self.map_size, self.abs_temp,)

