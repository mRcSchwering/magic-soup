import random
import math
from typing import Any
from io import BytesIO
import pickle
from pathlib import Path
import torch
from magicsoup.constants import ProteinSpecType
from magicsoup.util import randstr
from magicsoup.containers import Chemistry, Cell
from magicsoup.kinetics import Kinetics
from magicsoup.genetics import Genetics
from magicsoup.mutations import point_mutations, recombinations
from magicsoup import _lib  # type: ignore


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
    """
    The object of this class is the main API for running the simulation.
    It holds the state of a simulation and offers methods for advancing it.

    Parameters:
        chemistry: [Chemistry][magicsoup.containers.Chemistry] with [Molecules][magicsoup.containers.Molecule] and reactions.
        map_size: Size of the 2D world map as number of pixels in x- and y-direction.
        abs_temp: Absolute temperature in Kelvin. Will influence reaction equilibriums.
            Higher temperatures give concentration gradients higher importance.
        mol_map_init: How to give initial molecule concentrations to molecule maps (`randn` or `zeros`).
            `randn` is normally distributed N(10, 1), `zeros` is all zeros.
        start_codons: Codons which start a coding sequence
        stop_codons: Codons which stop a coding sequence
        device: Device to use for tensors (see [pytorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)).
            Use this to move most calculations to a GPU.
        batch_size: Optional parameter for reducing memory peaks when cell parameters are updated.
            By iteratively calculating cell paremeters of maximum `batch_size` cells at once,
            one can reduce memory peaks during functions that update cell parameters.

    Most attributes on this class describe the current state of molecules and cells.
    Both cells and molecules are referred to by index.
    Whenever cells are listed or represented in one dimension they are ordered consistently.
    _E.g._ at any point in the simulation `world.cell_genomes[i]` and `world.cell_labels[i]`
    refer to the genome and label of cell with index `i`.
    As cell number grows and shrinks cell indexes are not constant throughout the simulation.
    Whenever an operation modifies the number of cells
    (like [kill_cells()][magicsoup.world.World.kill_cells] or [divide_cells()][magicsoup.world.World.divide_cells])
    cell indexes are updated.
    Likewise, molecules are always ordered as in [Chemistry][magicsoup.containers.Chemistry].
    _E.g._ `world.chemistry.molecules[i]` refers to the ith [Molecule][magicsoup.containers.Molecule] object
    and `world.molecule_map[i]` refers to a 2D map describing molecule concentrations of the ith molecule.
    Molecule indexes stay constant throughout a simulation.

    The simulation works with lists and [PyTorch Tensor](https://pytorch.org/docs/stable/tensors.html)
    representations of cell proteomes and molecule concentrations.
    You can get a [Cell][magicsoup.containers.Cell] object from [get_cell()][magicsoup.world.World.get_cell]
    with easy-to-interpret attributes (see [Cell][magicsoup.containers.Cell] docs).
    However, for performance reasons only the lists and tensor representations are updated during a simulation.
    [get_cell()][magicsoup.world.World.get_cell] is intended for analyzing simulation results.

    For customizing a simulation one can interact with the working lists and tensors of the simulation.
    They could be used for reading cell and molecule information, or for modifying it.
    The following attributes are always updated during the simulation:

    - `cell_genomes`: A list of strings representing cell genomes ordered by current cell index.
    - `labels`: List of strings representing cell labels ordered by current cell index.
        Labels are strings that can be used to track cell origins.
        When spawning new cells ([spawn_cell()][magicsoup.world.World.spawn_cells]) a random label is assigned to each cell.
        If a cell divides, its descendants get the same label.
    - `cell_map`: Boolean 2D tensor describing which pixels on the map are occupied by a cell.
        Dimension 0 represents the x, dimension 1 the y direction.
    - `molecule_map`: Float 3D tensor describing concentrations (in mM by default) for each [Molecule][magicsoup.containers.Molecule].
        Dimension 0 represents the [Molecule][magicsoup.containers.Molecule] ordered as in [Chemistry][magicsoup.containers.Chemistry].
        Dimension 1 represents the x, dimension 2 the y direction.
        So, `world.molecule_map[0, 1, 2]` is the concentration of the 0th [Molecule][magicsoup.containers.Molecule]
        on pixel $(1, 2)$ of the world map.
    - `cell_molecules`: Float 2D tensor describing intracellular concentrations (in mM by default) for each [Molecule][magicsoup.containers.Molecule].
        Dimension 0 represents the cell ordered by current cell index.
        Dimension 1 represents the [Molecule][magicsoup.containers.Molecule] ordered as in [Chemistry][magicsoup.containers.Chemistry].
        So, `world.cell_molecules[0, 1]` represents the intracellular concentration in mM
        of the 1st [Molecule][magicsoup.containers.Molecule] in the cell which currently has index 0.
    - `cell_lifetimes`: Integer 1D tensor describing how many time steps each cell survived since it was spawned
        or since its last division ordered by current cell index.
    - `cell_divisions`: Integer 1D tensor describing how many times each cell's ancestors divided ordered by current cell index.

    Some methods are provided to advance the simulation and interact with it:

    - [spawn_cells()][magicsoup.world.World.spawn_cells] spawn new cells from genomes and place them randomly on the map
    - [add_cells()][magicsoup.world.World.add_cells] add new cells from [Cell][magicsoup.containers.Cell] objects and place them randomly on the map
    - [divide_cells()][magicsoup.world.World.divide_cells] divide existing cells
    - [update_cells()][magicsoup.world.World.update_cells] update existing cells with a new genome and proteome
    - [kill_cells()][magicsoup.world.World.kill_cells] kill existing cells
    - [move_cells()][magicsoup.world.World.move_cells] move existing cells to a random position in their Moore's neighborhood
    - [diffuse_molecules()][magicsoup.world.World.diffuse_molecules] let molecules diffuse and permeate by one time step
    - [degrade_molecules()][magicsoup.world.World.degrade_molecules] let molecules degrade by one time step
    - [increment_cell_lifetimes()][magicsoup.world.World.increment_cell_lifetimes] increment `world.cell_lifetimes` by 1
    - [enzymatic_activity()][magicsoup.world.World.enzymatic_activity] let cell proteins work for one time step
    - [get_cell()][magicsoup.world.World.get_cell] get a cell representation as a [Cell][magicsoup.containers.Cell] object

    Furthermore, there are methods for saving and loading a simulation.
    For any new simulation use [save()][magicsoup.world.World.save] once to save the whole world object to a pickle file
    (including its [Chemistry][magicsoup.containers.Chemistry], [Genetics][magicsoup.genetics.Genetics], [Kinetics][magicsoup.kinetics.Kinetics]).
    You can restore it with [from_file()][magicsoup.world.World.from_file] later on.
    Then, to save the world's state you can use [save_state()][magicsoup.world.World.save_state].
    This is a quick, lightweight save, but it only saves things that change during the simulation.
    Use [load_state()][magicsoup.world.World.load_state] to re-load a certain state.

    [PyTorch](https://pytorch.org/) is heavily used in this simulation.
    Some important attributes are [PyTorch Tensors](https://pytorch.org/docs/stable/tensors.html) (see above).
    By setting the `device` parameter most calculations can be moved to a GPU.
    This heavily increases performance and is recommended.
    When working with tensors one should try to use [tensor methods](https://pytorch.org/docs/stable/torch.html).
    That way computations and data stay on the GPU.
    [Here](https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html) is a tutorial for working with tensors.
    Try to avoid sending data back and forth between CPU and GPU.

    [World][magicsoup.world.World] object carries a [Genetics][magicsoup.genetics.Genetics] instance on `world.genetics`,
    a [Kinetics][magicsoup.kinetics.Kinetics] instance on `world.kinetics`,
    and a [Chemistry][magicsoup.containers.Chemistry] instance on `world.chemistry`
    (which is just a reference to the [Chemistry][magicsoup.containers.Chemistry] object that was used when initializing [World][magicsoup.world.World]).
    Usually, you don't need to touch them.
    But, if you want to override them or look into some details, see the docstrings of their classes for more information.

    By default in this simulation molecule numbers can be thought of as being in mM, time steps in s, energies J/mol.
    Eventually, they are just numbers and can be interpreted as anything.
    However, with the default parameters in [Kinetics][magicsoup.kinetics.Kinetics]
    and the way [Molecule][magicsoup.containers.Molecule] objects are defined
    it makes sense to interpret them in mM, s, and J.
    """

    def __init__(
        self,
        chemistry: Chemistry,
        map_size: int = 128,
        abs_temp: float = 310.0,
        mol_map_init: str = "randn",
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        device: str = "cpu",
        batch_size: int | None = None,
    ):
        if not torch.cuda.is_available():
            device = "cpu"

        self.device = device
        self.batch_size = batch_size
        self.map_size = map_size
        self.abs_temp = abs_temp
        self.chemistry = chemistry

        self.genetics = Genetics(start_codons=start_codons, stop_codons=stop_codons)

        self.kinetics: Kinetics = Kinetics(
            chemistry=chemistry,
            abs_temp=abs_temp,
            device=self.device,
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
        """
        Get a [Cell][magicsoup.containers.Cell] object with information about its current environment.
        Raises `ValueError` if cell was not found.

        Parameters:
            by_idx: Get cell by current cell index in [World][magicsoup.world.World]
            by_position: Get cell by position `(x, y)` on the world map.

        Returns:
            The searched cell described as [Cell][magicsoup.containers.Cell] object.

        The simulation works with tensor representations of cell proteomes and molecule concentrations.
        The return value of this function is a helper which makes interpreting a cell easier.
        See the documentation of [Cell][magicsoup.containers.Cell] for more details.
        """
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
        """
        For each cell from a list of cell indexes find all other cells that
        are in the Moore's neighborhood of this cell.

        Arguments:
            cell_idxs: Indexes of cells for which to find neighbors
            nghbr_idxs: Optional list of cells regarded as neighbors.
                If `None` (default) each cell in `cell_idxs` can form a pair
                with any other neighboring cell in `cell_idxs`. With `nghbr_idxs` provided
                each cell in `cell_idxs` can only form a pair with a neighboring
                cell that is in `nghbr_idxs`.

        Returns:
            List of tuples of cell indexes for each unique neighboring pair.

        Returned neighbors are unique.
        So, _e.g._ return value `[(1, 4)]` describes the neighbors of a cell
        with index 1 and a cell with index 4. `(4, 1)` is not returned.
        """
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
        """
        Create new cells and place them randomly on the map.
        All lists and tensors that reference cells will be updated.

        Parameters:
            genomes: List of genome strings for newly added cells

        Returns:
            Cell indexes of successfully added cells.

        Each cell is placed randomly on the map and receives half the molecules of the pixel where it is added.
        If too few unoccupied pixels are left on the cell map only the remaining unoccupied pixels will be filled with new cells.
        For each new cell all cell parameters are updated.
        Each new cell recieves a lifetime of 0, 0 divisions, and a random label.
        """
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

        self.cell_lifetimes = self._expand_c(t=self.cell_lifetimes, n=n_new_cells)
        self.cell_positions = self._expand_c(t=self.cell_positions, n=n_new_cells)
        self.cell_divisions = self._expand_c(t=self.cell_divisions, n=n_new_cells)
        self.cell_molecules = self._expand_c(t=self.cell_molecules, n=n_new_cells)
        self.kinetics.increase_max_cells(by_n=n_new_cells)

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
        """
        Place [Cells][magicsoup.containers.Cell] randomly on the map.
        All lists and tensors that reference cells will be updated.

        Parameters:
            cells: List of [Cell][magicsoup.containers.Cell] objects to be added

        Returns:
            The indexes of successfully added cells.

        Each cell will be placed randomly on the map.
        If too few unoccupied pixels are left on the cell map only the remaining unoccupied pixels will be filled with cells.
        Added cells keep their genomes, proteomes, intracellular molecules, lifetimes, divisions, labels.
        Their indexes and positions are updated.
        """
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

        self.cell_lifetimes = self._expand_c(t=self.cell_lifetimes, n=n_new_cells)
        self.cell_positions = self._expand_c(t=self.cell_positions, n=n_new_cells)
        self.cell_divisions = self._expand_c(t=self.cell_divisions, n=n_new_cells)
        self.cell_molecules = self._expand_c(t=self.cell_molecules, n=n_new_cells)
        self.kinetics.increase_max_cells(by_n=n_new_cells)

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
        """
        Bring cells to divide.
        All lists and tensors that reference cells will be updated.

        Parameters:
            cell_idxs: Cell indexes of the cells that should divide.

        Returns:
            List of tuples `(anc_idx, new_idx)` of successfully divided cells
            where `anc_idx` is the ancestor cell index and `new_idx`
            is the newly created cell index.

        Each cell divides by creating a clone of itself on a random pixel next to itself (Moore's neighborhood).
        If every pixel in its Moore's neighborhood is already occupied, it will not be able to divide.
        Both the original ancestor cell and the newly placed cell will become the descendants.
        They will have the same genome, proteome, and label.
        Molecule of the ancestor cell are shared evenly among descendants.

        Both descendants get the ancestor's number of divisions + 1.
        So, _e.g._ if a cell with `n_divisions=2` divides, its descendants both have `n_divisions=3`.
        Both descendants recieve a new lifetime of 0.

        Of the list of descendant index tuples,
        the first descendant in each tuple is the cell that still lives on the same pixel.
        The second descendant in that tuple is the cell that was newly created.
        """
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

        self.cell_lifetimes = self._expand_c(t=self.cell_lifetimes, n=n_new_cells)
        self.cell_positions = self._expand_c(t=self.cell_positions, n=n_new_cells)
        self.cell_divisions = self._expand_c(t=self.cell_divisions, n=n_new_cells)
        self.cell_molecules = self._expand_c(t=self.cell_molecules, n=n_new_cells)
        self.kinetics.increase_max_cells(by_n=n_new_cells)
        self.kinetics.copy_cell_params(from_idxs=parent_idxs, to_idxs=child_idxs)

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

    def update_cells(self, genome_idx_pairs: list[tuple[str, int]]):
        """
        Update existing cells with new genomes and derive cell
        proteomes for them.

        Parameters:
            genome_idx_pairs: List of tuples `(genome, idx)` where
                `genome` is the cells genome and `idx` is its current index.
        """
        if len(genome_idx_pairs) == 0:
            return

        for genome, idx in genome_idx_pairs:
            self.cell_genomes[idx] = genome

        genomes, idxs = list(map(list, zip(*genome_idx_pairs)))
        self._update_cell_params(genomes=genomes, idxs=idxs)  # type: ignore

    def kill_cells(self, cell_idxs: list[int] | None = None):
        """
        Remove existing cells.
        All lists and tensors referencing cells are updated.

        Parameters:
            cell_idxs: Indexes of the cells that should die.
                `None` to kill all cells.

        Cells that are killed dump their molecule contents onto the pixel they used to live on.

        Cells are removed from all lists and tensors that reference cells.
        Thus, after killing cells the index of some living cells are updated.
        E.g. if there are 10 cells and you kill the cell with index 8 (the 9th cell),
        the cell that used to have index 9 (10th cell) will now have index 9.
        """
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
        self.kinetics.remove_cell_params(keep=keep)

        for idx in sorted(cell_idxs, reverse=True):
            self.cell_genomes.pop(idx)
            self.cell_labels.pop(idx)

        self.n_cells -= len(cell_idxs)

    def move_cells(self, cell_idxs: list[int] | None = None):
        """
        Move cells to a random position in their Moore's neighborhood.
        If every pixel in the cells' Moore neighborhood is occupied already
        the cell will not be moved.

        Parameters:
            cell_idxs: Indexes of cells that should be moved.
                `None` to move all cells.

        `world.cell_positions` and `world.cell_map` are updated.
        """
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

    def reposition_cells(self, cell_idxs: list[int] | None = None):
        """
        Reposition cells randomly on cell map without changing them.

        Parameters:
            cell_idxs: Indexes of cells that should be repositioned.
                `None` to reposition all cells.

        Cells are removed from their current pixels on the cell map
        and repositioned randomly on any free pixel on the cell map.
        `world.cell_positions` and `world.cell_map` are updated.
        _I.e._ genomes, proteomes, cell molecules stay the same.
        """
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
        """
        Catalyze reactions for one time step (1s by default).
        This includes molecule transport into or out of the cell.
        `world.molecule_map` and `world.cell_molecules` are updated.
        """
        if self.n_cells == 0:
            return

        xs = self.cell_positions[:, 0]
        ys = self.cell_positions[:, 1]
        X0 = torch.cat([self.cell_molecules, self.molecule_map[:, xs, ys].T], dim=1)
        X1 = self.kinetics.integrate_signals(X=X0)

        self.molecule_map[:, xs, ys] = X1[:, self._ext_mol_idxs].T
        self.cell_molecules = X1[:, self._int_mol_idxs]

    @torch.no_grad()
    def diffuse_molecules(self):
        """
        Let molecules in molecule map diffuse and permeate through cell membranes
        by one time step (1s by default).
        `world.molecule_map` and `world.cell_molecules` are updated.

        By how much each molcule species diffuses around the world map and permeates
        into or out of cells is defined on [Molecule][magicsoup.containers.Molecule]
        in [Chemistry][magicsoup.containers.Chemistry].
        """
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

    def degrade_molecules(self):
        """
        Degrade molecules in world map and cells by one time step.
        `world.molecule_map` and `world.cell_molecules` are updated.

        How quickly each molecule species degrades depends on its half life
        which is defined on [Molecule][magicsoup.containers.Molecule]
        in [Chemistry][magicsoup.containers.Chemistry].
        """
        for mol_i, degrad in enumerate(self._mol_degrads):
            self.molecule_map[mol_i] *= degrad
            self.cell_molecules[:, mol_i] *= degrad

    def increment_cell_lifetimes(self):
        """
        Increment `world.cell_lifetimes` by 1.
        Cell lifetimes can be used to filter cells but,
        apart from what you implement,
        they don't have any effect in the simulation.
        """
        self.cell_lifetimes += 1

    def mutate_cells(
        self,
        cell_idxs: list[int] | None = None,
        p: float = 1e-6,
        p_indel: float = 0.4,
        p_del: float = 0.66,
    ):
        """
        Mutate cells with point mutations

        Arguments:
            cell_idxs: Indexes of cells which are allowed to mutate.
                Leave `None` to allow all cells to mutate.
            p: Probability of a mutation per base pair
            p_indel: Probability of any point mutation being an indel
                    (inverse probability of it being a substitution)
            p_del: Probability of any indel being a deletion
                (inverse probability of it being an insertion)

        Convenience function that uses [mutations.point_mutations()][magicsoup.mutations.point_mutations]
        to mutate genomes, then updates cell parameters for cells whose genomes were changed.
        """
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
        """
        Recombinate neighbourig cells

        Arguments:
            cell_idxs: Indexes of cells which are allowed to recombinate.
                Leave `None` to allow all cells to recombinate.
            p: Probability of a strand break per base pair during recombinataion.

        Convenience function that uses [mutations.recombinations()][magicsoup.mutations.recombinations]
        to recombinate genomes of neighbouring cells, then updates cell parameters for cells
        whose genomes were changed.
        [get_neighbors()][magicsoup.world.World.get_neighbors] is used to identify neighbors.
        """
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
        """
        Write `world` object to pickle file

        Parameters:
            rundir: Directory of the pickle file
            name: Name of the pickle file

        This is a big and slow save.
        It saves everything needed to restore `world` with its chemistry and genetics.
        Use [from_file()][magicsoup.world.World.from_file] to restore it.
        For a small and quick save use [save_state()][magicsoup.world.World.save_state].
        """
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
        """
        Restore previously saved `world` object from pickle file.
        It had to be saved with [save()][magicsoup.world.World.save].

        Parameters:
            rundir: Directory of the pickle file
            name: Name of the pickle file
            device: Optionally set device to which tensors should be loaded.
                Default is the device they had when they were saved.
                (see [pytorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html))

        Returns:
            A new [World][magicsoup.world.World] instance.
        """
        with open(rundir / name, "rb") as fh:
            unpickler = _CPU_Unpickler(fh, map_location=device)
            obj: "World" = unpickler.load()

        if device is not None:
            obj.device = device
            obj.kinetics.device = device

        return obj

    def save_state(self, statedir: Path):
        """
        Save current state

        Parameters:
            statedir: Directory to store files in (there are multiple files per state)

        This is a small and quick save.
        It only saves things which change during the simulation.
        Restore a state with [load_state()][magicsoup.world.World.load_state].

        To restore a whole `world` object you need to save it once with [save()][magicsoup.world.World.save].
        Then, `world.save_state()` can be used to save different states of that `world` object.
        """
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
        """
        Load a saved world state.
        The state had to be saved with [save_state()][magicsoup.world.World.save_state] previously.

        Parameters:
            statedir: Directory that contains all files of that state
            ignore_cell_params: Whether to not update cell parameters as well.
                If you are only interested in the cells' genomes, molecules, or labels
                you can set this to `True` to make loading a lot faster.
        """
        if not ignore_cell_params:
            self.kill_cells(cell_idxs=list(range(self.n_cells)))

        self.cell_molecules = torch.load(
            statedir / "cell_molecules.pt", map_location=self.device
        ).float()
        self.cell_map = torch.load(
            statedir / "cell_map.pt", map_location=self.device
        ).bool()
        self.molecule_map = torch.load(
            statedir / "molecule_map.pt", map_location=self.device
        )
        self.cell_lifetimes = torch.load(
            statedir / "cell_lifetimes.pt", map_location=self.device
        ).int()
        self.cell_positions = torch.load(
            statedir / "cell_positions.pt", map_location=self.device
        ).int()
        self.cell_divisions = torch.load(
            statedir / "cell_divisions.pt", map_location=self.device
        ).int()

        with open(statedir / "cells.fasta", "r", encoding="utf-8") as fh:
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

        self.kinetics.increase_max_cells(by_n=self.n_cells)
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

        self.kinetics.unset_cell_params(cell_idxs=unset_idxs)
        if max_prots == 0:
            return

        self.kinetics.increase_max_proteins(max_n=max_prots)

        n = len(set_proteomes)
        s = n if self.batch_size is None else n
        for a in range(0, n, s):
            b = a + s
            self.kinetics.set_cell_params(
                cell_idxs=set_idxs[a:b], proteomes=set_proteomes[a:b]
            )

    def _find_free_random_positions(self, n_cells: int) -> torch.Tensor:
        # available spots on map
        pxls = torch.nonzero(~self.cell_map).int()
        n_pxls = pxls.size(0)
        if n_cells > n_pxls:
            n_cells = n_pxls

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

        if mol_perm_rate > 1.0:
            mol_perm_rate = 1.0

        if mol_perm_rate == 0.0:
            return 0.0

        d = 1 / mol_perm_rate
        return 1 / (d + 1)

    def _get_diffuse(self, mol_diff_rate: float) -> torch.nn.Conv2d:
        if mol_diff_rate < 0.0:
            mol_diff_rate = -mol_diff_rate

        # mol_diff_rate > 1.0 could also mean expanding the kernel
        # so that molecules can diffuse more than just 1 pxl per round
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

    def _expand_c(self, t: torch.Tensor, n: int) -> torch.Tensor:
        size = t.size()
        zeros = torch.zeros(n, *size[1:], dtype=t.dtype, device=self.device)
        return torch.cat([t, zeros], dim=0)

    def _i32_tensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=torch.int32)

    def _f32_tensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=torch.float32)

    def __repr__(self) -> str:
        kwargs = {
            "map_size": self.map_size,
            "abs_temp": self.abs_temp,
            "device": self.device,
        }
        args = [f"{k}:{repr(d)}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"
