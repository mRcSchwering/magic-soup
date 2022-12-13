from typing import Optional
import logging
import random
import torch
from .constants import GAS_CONSTANT
from .containers import Molecule, Protein, Cell
from .util import cpad2d, pad_2_true_idx, padded_indices, pad_map, unpad_map
from .kinetics import integrate_signals, calc_cell_params


# TODO: fit diffusion rate to natural diffusion rate of small molecules in cell
# TODO: summary()
# TODO: have Molecules carry their own idx?

_log = logging.getLogger(__name__)


class World:
    """
    World definition with map that stores molecule/signal concentrations and cell
    positions. It also defines how all proteins of all cells integrate signals
    of their respective environments.

    - `molecules` list of all molecules that are part of this simulation
    - `map_size` number of pixels in x- and y-direction
    - `mol_degrad` Factor by which molecules are degraded per time step (`1.0` for no degradation)
    - `mol_diff_rate` Factor by which molecules diffuse per time step (`0.0` for no diffusion)
    - `mol_map_init` How to initialize molecule maps (`randn` or `zeros`)
    - `n_max_proteins` how many proteins any single cell is expected to have at maximum at any point during the simulation
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
        mol_degrad=0.99,
        mol_diff_rate=1e-2,
        mol_map_init="randn",
        abs_temp=310.0,
        device="cpu",
        dtype=torch.float,
    ):
        self.molecules = molecules
        self.map_size = map_size
        self.abs_temp = abs_temp
        self.mol_degrad = mol_degrad
        self.mol_diff_rate = mol_diff_rate
        self.dtype = dtype
        self.device = device
        self.torch_kwargs = {"dtype": dtype, "device": device}

        self.n_molecules = len(molecules)
        self._mol_2_idx = self._get_mol_2_idx_map(molecules=molecules)
        self._int_mol_idxs = list(range(self.n_molecules))
        self._ext_mol_idxs = list(range(self.n_molecules, self.n_molecules * 2))

        self._diffuse = self._get_conv(mol_diff_rate=mol_diff_rate)
        self._pad_2_true_idx = self._get_pad_2_true_idx_map(size=self.map_size)

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
        self.regulators = self._tensor(0, 0, 2 * self.n_molecules)

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

    def get_intracellular_molecule_idxs(self, molecules: list[Molecule]) -> list[int]:
        """
        Get indexes for intracellular molecule concentrations.

        Intracellular molecule concentrations are referenced on `cell_molecules`.
        This is a tensor with cells in dimension 0 and molecules in dimension 1.
        Use like:

        ```
            idxs = world.get_molecule_idxs(molecules=[ATP, ADP])
            X = world.cell_molecules[:, idxs]
        ```

        It gets updated whenever some action that changes the amount of cells
        is performed. So, you should always access it directly on `world.cell_molecules`.
        """
        return [self._mol_2_idx[(d, False)] for d in molecules]

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

        new_cells = [ſelf.cells[i] for i in new_idxs]
        self._expand_max_proteins(max_n=max(len(d.proteome) for d in new_cells))

        # cell is supposed to have the same concentrations as the pxl it lives on
        new_positions = [self.cells[i].position for i in new_idxs]
        xs = []
        ys = []
        for x, y in new_positions:
            xs.append(x)
            ys.append(y)
        self.cell_molecules[new_idxs, :] = self.molecule_map[:, xs, ys].T

        self._add_new_cells_to_proteome_params(
            proteomes=[d.proteome for d in new_cells], cell_idxs=new_idxs
        )

    def replicate_cells(self, cells: list[Cell]):
        """
        Replicate existing cells with new genomes and proteomes.

        - `cells` new cells with new geneomes and proteomes, but with parent
          position and idexes.

        Each cell will be placed randomly next to its parent (Moore's neighborhood)
        and will receive the same molecule concentrations as its parent.

        If every pixel in the cells' Moore neighborhood is taken
        the cell will not replicate.
        """
        _log.debug("Replicating %i cells", len(cells))

        if len(cells) == 0:
            return

        old_idxs, new_idxs = self._place_replicated_cells_near_parents(cells=cells)

        n_new_cells = len(new_idxs)
        if n_new_cells == 0:
            return

        self._expand_max_cells(by_n=n_new_cells)

        new_cells = [self.cells[i] for i in new_idxs]
        self._expand_max_proteins(max_n=max(len(d.proteome) for d in new_cells))

        # cell is supposed to have the same concentrations as the parent had
        self.cell_molecules[new_idxs, :] = self.cell_molecules[old_idxs, :]

        self._add_new_cells_to_proteome_params(
            proteomes=[d.proteome for d in new_cells], cell_idxs=new_idxs
        )

    def update_cells(self, cells: list[Cell]):
        """
        Update existing cells with new genomes and proteomes.

        - `cells` cells with that need to be updated
        """
        _log.debug("Updating %i cells", len(cells))

        for cell in cells:
            self.cells[cell.idx] = cell

        self._expand_max_proteins(max_n=max(len(d.proteome) for d in cells))
        self._add_new_cells_to_proteome_params(
            proteomes=[d.proteome for d in cells], cell_idxs=[d.idx for d in cells]
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
            A=self.regulators.clamp(-1.0, 1.0),
        )
        X += Xd

        self.cell_molecules = X[:, self._int_mol_idxs]
        self._put_cell_ext_molecules(
            cells=self.cells, molecules=X[:, self._ext_mol_idxs]
        )

    def _randomly_move_cells(self, cells: list[Cell]):
        size = self.map_size + 2
        padded_map = pad_map(self.cell_map)

        for cell in cells:
            old_x, old_y = cell.position
            old_xp = old_x + 1
            old_yp = old_y + 1
            new_pos = self._find_open_spot_in_neighborhood(
                padded_map=padded_map, x=old_xp, y=old_yp, size=size
            )

            if new_pos is None:
                _log.info(
                    "Wanted to move cell at %i, %i"
                    " but no pixel in the Moore neighborhood was free."
                    " So, cell wasn't moved.",
                    old_x,
                    old_y,
                )
                continue

            # move cells
            old_pos = padded_indices(x=old_xp, y=old_yp, s=size)
            padded_map[old_pos] = False

            padded_map[new_pos] = True
            xs, ys = new_pos
            cell.position = (
                pad_2_true_idx(idx=xs[0], size=self.map_size),
                pad_2_true_idx(idx=ys[0], size=self.map_size),
            )

        self.cell_map = unpad_map(padded_map)

    def _place_replicated_cells_near_parents(
        self, cells: list[Cell]
    ) -> tuple[list[int], list[int]]:
        size = self.map_size + 2
        padded_map = pad_map(self.cell_map)

        new_idx = len(self.cells)
        new_idxs: list[int] = []
        old_idxs: list[int] = []

        for cell in cells:
            old_x, old_y = cell.position
            old_xp = old_x + 1
            old_yp = old_y + 1
            new_pos = self._find_open_spot_in_neighborhood(
                padded_map=padded_map, x=old_xp, y=old_yp, size=size
            )

            if new_pos is None:
                _log.info(
                    "Wanted to replicate cell next to %i, %i"
                    " but no pixel in the Moore neighborhood was free."
                    " So, cell wasn't able to replicate.",
                    old_x,
                    old_y,
                )
                continue

            # place cell in position
            padded_map[new_pos] = True
            xs, ys = new_pos
            cell.position = (
                pad_2_true_idx(idx=xs[0], size=self.map_size),
                pad_2_true_idx(idx=ys[0], size=self.map_size),
            )

            # set new cell idx
            old_idxs.append(cell.idx)
            cell.idx = new_idx
            self.cells.append(cell)
            new_idxs.append(new_idx)
            new_idx += 1

        self.cell_map = unpad_map(padded_map)

        return old_idxs, new_idxs

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

    def _add_new_cells_to_proteome_params(
        self, proteomes: list[list[Protein]], cell_idxs: list[int]
    ):
        calc_cell_params(
            proteomes=proteomes,
            n_signals=2 * self.n_molecules,
            cell_idxs=cell_idxs,
            mol_2_idx=self._mol_2_idx,
            Km=self.affinities,
            Vmax=self.velocities,
            E=self.energies,
            N=self.stoichiometry,
            A=self.regulators,
        )

    def _find_open_spot_in_neighborhood(
        self, padded_map: torch.Tensor, x: int, y: int, size: int
    ) -> Optional[tuple[list[int], list[int]]]:
        xps = slice(x - 1, x + 2)
        yps = slice(y - 1, y + 2)
        pxls = torch.argwhere(~padded_map[xps, yps])
        if len(pxls) == 0:
            return None

        offset_x, offset_y = random.choice(pxls.tolist())
        new_x = offset_x + x - 1
        new_y = offset_y + y - 1
        return padded_indices(x=new_x, y=new_y, s=size)

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
            return self._tensor(*args)
        if init == "randn":
            return torch.randn(*args, **self.torch_kwargs).abs()
        raise ValueError(
            f"Didnt recognize mol_map_init={init}."
            " Should be one of: 'zeros', 'randn'."
        )

    def _get_conv(self, mol_diff_rate: float) -> torch.nn.Conv2d:
        if not 0.0 <= mol_diff_rate <= 1.0:
            raise ValueError(
                "Diffusion rate must be between 0 and 1."
                f" Now it's mol_diff_rate={mol_diff_rate}"
            )

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
        self.regulators = self._expand(t=self.regulators, n=by_n, d=0)

    def _remove_cell_rows(self, idxs: list[int]):
        keep = torch.ones(self.cell_survival.shape[0], dtype=torch.bool)
        keep[idxs] = False
        self.cell_survival = self.cell_survival[keep]
        self.cell_molecules = self.cell_molecules[keep]
        self.affinities = self.affinities[keep]
        self.velocities = self.velocities[keep]
        self.energies = self.energies[keep]
        self.stoichiometry = self.stoichiometry[keep]
        self.regulators = self.regulators[keep]

    def _expand_max_proteins(self, max_n: int):
        n_prots = int(self.affinities.shape[1])
        if max_n > n_prots:
            by_n = max_n - n_prots
            self.affinities = self._expand(t=self.affinities, n=by_n, d=1)
            self.velocities = self._expand(t=self.velocities, n=by_n, d=1)
            self.energies = self._expand(t=self.energies, n=by_n, d=1)
            self.stoichiometry = self._expand(t=self.stoichiometry, n=by_n, d=1)
            self.regulators = self._expand(t=self.regulators, n=by_n, d=1)

    def _expand(self, t: torch.Tensor, n: int, d: int) -> torch.Tensor:
        pre = t.shape[slice(d)]
        post = t.shape[slice(d + 1, t.dim())]
        zeros = self._tensor(*pre, n, *post)
        return torch.cat([t, zeros], dim=d)

    def _tensor(self, *args, **kwargs) -> torch.Tensor:
        return torch.zeros(*args, **{**self.torch_kwargs, **kwargs})

    def _get_mol_2_idx_map(
        self, molecules: list[Molecule]
    ) -> dict[tuple[Molecule, bool], int]:
        n_molecules = len(molecules)
        mol_2_idx = {}
        for mol_i, mol in enumerate(molecules):
            mol_2_idx[(mol, False)] = mol_i
            mol_2_idx[(mol, True)] = mol_i + n_molecules
        return mol_2_idx

    def _get_pad_2_true_idx_map(self, size: int) -> dict[int, int]:
        return {i: pad_2_true_idx(idx=i, size=size) for i in range(size + 2)}

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(molecules=%r,map_size=%r,mol_degrad=%r,mol_diff_rate=%r,abs_temp=%r,device=%r,dtype=%r)"
            % (
                clsname,
                self.molecules,
                self.map_size,
                self.mol_degrad,
                self.mol_diff_rate,
                self.abs_temp,
                self.device,
                self.dtype,
            )
        )
