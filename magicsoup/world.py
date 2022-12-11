from typing import Optional
import warnings
import random
import torch
from .proteins import Molecule, Protein
from .util import GAS_CONSTANT, cpad2d, pad_2_true_idx
from .kinetics import integrate_signals, calc_cell_params

# TODO: refactor API:
#       - adding new cells
#       - getting a cell with all details
#       - recalculating certain cells after they were mutated
#       - dividing a cell
#       - killing a cell
#       - moving a cell

# TODO: get kill and replicate going to see implications


# TODO: fit diffusion rate to natural diffusion rate of small molecules in cell
# TODO: summary()


class Cell:
    """
    Container for information about a cell
    """

    def __init__(
        self,
        genome: str,
        proteome: list[Protein],
        position: tuple[int, int] = (-1, -1),
        idx=-1,
        label: Optional[str] = None,
        n_survived_steps=0,
    ):
        self.genome = genome
        self.proteome = proteome
        self.label = label
        self.position = position
        self.idx = idx
        self.n_survived_steps = n_survived_steps
        self.int_molecules: Optional[torch.Tensor] = None
        self.ext_molecules: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(genome=%r,proteome=%r,position=%r,idx=%r,label=%r,n_survived_steps=%r)"
            % (
                clsname,
                self.genome,
                self.proteome,
                self.position,
                self.idx,
                self.label,
                self.n_survived_steps,
            )
        )


class World:
    """
    World definition with map that stores molecule/signal concentrations and cell
    positions. It also defines how all proteins of all cells integrate signals
    of their respective environments.

    - `molecules` list of all possible molecules that can be encoutered
    - `actions` list of all possible actions that can be encountered
    - `map_size` number of pixel in both directions
    - `mol_degrad` by how much molecules should be degraded each step
    - `mol_diff_rate` how much molecules diffuse at each step
    - `mol_map_init` how to initialize molecule maps (`randn` or `zeros`)
    - `n_max_proteins` how many proteins any single cell is expected to have at maximum at any point during the simulation
    - `trunc_n_decs` by how many decimals to truncate values after each computation
    - `dtype` pytorch dtype to use for tensors (see pytorch docs)
    - `device` pytorch device to use for tensors (e.g. `cpu` or `gpu`, see pytorch docs)

    The world map (for cells and molecules) is a square map with `map_size` pixels in both
    directions. But it is "circular" in that when you move over the border to the left, you re-appear
    on the right and vice versa (same for up and down). Each cell occupies one pixel. The cell
    experiences molecule concentrations on that pixel as external molecule concentrations. Additionally,
    it has its internal molecule concentrations.

    It is faster to setup a large tensor (filled mostly with `0.0`) than to readjust the size of the tensor
    during every round (according to how many proteins the largest cell has at the moment). Better set
    `n_max_proteins` a bit higher than expected. The simulation would raise an exception and stop
    if a cell at some point would have more proteins than `n_max_proteins`.

    Reducing `trunc_n_decs` has 2 effects. It speeds up calculations but also makes them less accurate.
    Another side effect is that certain values become zero more quickly. This could be a desired effect.
    E.g. the degradation of molecules would asymptotically approach 0. With `trunc_n_decs=4` any concentration
    at `0.0001` will become `0.0000` after the next time step.

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
        trunc_n_decs=4,
        device="cpu",
        dtype=torch.float,
    ):
        self.map_size = map_size
        self.abs_temp = abs_temp
        self.mol_degrad = mol_degrad
        self.mol_diff_rate = mol_diff_rate
        self.trunc_n_decs = trunc_n_decs
        self.dtype = dtype
        self.device = device
        self.torch_kwargs = {"dtype": dtype, "device": device}

        self.n_molecules = len(molecules)
        self.mol_2_idx = self._get_mol_2_idx_map(molecules=molecules)
        self.int_mol_idxs = list(range(self.n_molecules))
        self.ext_mol_idxs = list(range(self.n_molecules, self.n_molecules * 2))

        self.molecule_map = self._get_molecule_map(mol_map_init=mol_map_init)
        self.cell_map = self._tensor(map_size, map_size, dtype=torch.bool)
        self.conv113 = self._get_conv(mol_diff_rate=mol_diff_rate)
        self._pad_2_true_idx = self._get_pad_2_true_idx_map(size=map_size)

        self.cells: list[Cell] = []

        self.cell_survival = self._tensor(0)
        self.cell_molecules = self._tensor(0, self.n_molecules)
        self.affinities = self._tensor(0, 0, 2 * self.n_molecules)
        self.velocities = self._tensor(0, 0)
        self.energies = self._tensor(0, 0)
        self.stoichiometry = self._tensor(0, 0, 2 * self.n_molecules)
        self.regulators = self._tensor(0, 0, 2 * self.n_molecules)

    def get_cell(
        self, by_idx: Optional[int] = None, by_label: Optional[str] = None
    ) -> Cell:
        """Get a cell with all its information by name or index"""
        if by_idx is not None:
            idx = by_idx
        if by_label is not None:
            cell_labels = [d.label for d in self.cells]
            idx = cell_labels.index(by_label)
            if idx < 0:
                raise ValueError(f"Cell {by_label} not found")

        cell = self.cells[idx]
        cell.int_molecules = self.cell_molecules[idx, :]
        cell.ext_molecules = self._get_cell_ext_molecules(cell_idxs=[idx])[0, :]
        return cell

    def add_random_cells(self, cells: list[Cell]):
        """
        Randomly place cells on cell map and fill them with the molecules that
        were present at their respective pixels.
        """
        n_new_cells = len(cells)
        self._expand_max_cells(by_n=n_new_cells)
        self._expand_max_proteins(max_n=max(len(d.proteome) for d in cells))

        cell_idxs = list(range(len(self.cells), len(self.cells) + n_new_cells))
        self._place_new_cells_in_random_positions(new_idxs=cell_idxs, new_cells=cells)
        self._add_new_cells_to_proteome_params(
            proteomes=[d.proteome for d in cells], cell_idxs=cell_idxs
        )

    def replicate_cells(self, cells: list[Cell]):
        n_new_cells = len(cells)
        self._expand_max_cells(by_n=n_new_cells)
        self._expand_max_proteins(max_n=max(len(d.proteome) for d in cells))

        cell_idxs = list(range(len(self.cells), len(self.cells) + n_new_cells))
        self._place_replicated_cells_near_parents(new_idxs=cell_idxs, new_cells=cells)
        self._add_new_cells_to_proteome_params(
            proteomes=[d.proteome for d in cells], cell_idxs=cell_idxs
        )

    def kill_cells(self, cell_idxs: list[int]):
        spillout = self.cell_molecules[cell_idxs, :]
        self._add_cell_ext_molecules(cell_idxs=cell_idxs, molecules=spillout)
        self._remove_cell_rows(idxs=cell_idxs)

        new_idx = 0
        new_cells = []
        for old_idx, cell in enumerate(self.cells):
            if old_idx not in cell_idxs:
                cell.idx = new_idx
                new_idx += 1
                new_cells.append(cell)
        self.cells = new_cells

    @torch.no_grad()
    def diffuse_molecules(self):
        """Let molecules in world map diffuse by 1 time step"""
        before = self.molecule_map.unsqueeze(1)
        after = self.conv113(before)
        self.molecule_map = torch.squeeze(after, 1)

    def degrade_molecules(self):
        """Degrade molecules in world map and cells by 1 time step"""
        self.molecule_map = self.molecule_map * self.mol_degrad
        self.cell_molecules = self.cell_molecules * self.mol_degrad

    def increment_cell_survival(self):
        """Increment number of current cells' time steps by 1"""
        idxs = list(range(len(self.cells)))
        self.cell_survival[idxs] += 1

    def integrate_signals(self):
        """Integrate signals and update molecule maps"""
        ext_molecules = self._get_all_cell_ext_molecules()
        X = torch.cat([self.cell_molecules, ext_molecules], dim=1)

        Xd = integrate_signals(
            X=X,
            Km=self.affinities,
            Vmax=self.velocities,
            Ke=torch.exp(-self.energies / self.abs_temp / GAS_CONSTANT),
            N=self.stoichiometry,
            A=self.regulators.clamp(-1.0, 1.0),
        )
        X += Xd

        self.cell_molecules = X[:, self.int_mol_idxs]
        self._put_all_cell_ext_molecules(molecules=X[:, self.ext_mol_idxs])

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

    def _place_replicated_cells_near_parents(
        self, new_cells: list[Cell], new_idxs: list[int]
    ):
        # TODO: should figure out indexes itself, might not be able to replicate all cells
        #       if not enough space.
        #       1. try to replicate each cell, not index of each replicated cell
        #       2. expand tensors, by number of actually added cells
        #       3. add cell molecules to tensor, update cell parameters
        padded_map = cpad2d(self.cell_map.to(torch.float)).to(torch.bool)

        xs = []
        ys = []
        for idx, cell in zip(new_idxs, new_cells):
            old_x, old_y = cell.position
            pad_x = old_x + 1
            pad_y = old_y + 1
            pxls = torch.argwhere(
                ~padded_map[pad_x - 1 : pad_x + 2, pad_y - 1 : pad_y + 2]
            )

            try:
                idxs = random.sample(range(len(pxls)), k=1)
            except ValueError:
                warnings.warn(
                    f"Wanted to add a cell next to {old_x}, {old_y} no pixel was left on map"
                )
                continue

            new_pad_x, new_pad_y = pxls[idxs[0]].tolist()
            new_x = self._pad_2_true_idx[new_pad_x]
            new_y = self._pad_2_true_idx[new_pad_y]
            cell.position = new_x, new_y
            cell.idx = idx
            xs.append(new_x)
            ys.append(new_y)
            self.cells.append(cell)

        self.cell_map[xs, ys] = True

        # cell is supposed to have the same concentrations as the parent had
        old_idxs = [d.idx for d in new_cells]
        self.cell_molecules[new_idxs, :] = self.cell_molecules[old_idxs, :]

    def _place_new_cells_in_random_positions(
        self, new_idxs: list[int], new_cells: list[Cell]
    ):
        n_cells = len(new_idxs)
        pxls = torch.argwhere(~self.cell_map)

        try:
            idxs = random.sample(range(len(pxls)), k=n_cells)
        except ValueError:
            warnings.warn(
                f"Wanted to add {n_cells} cells but only {len(pxls)} pixels left on map"
            )
            return

        xs, ys = pxls[idxs].T.tolist()
        self.cell_map[xs, ys] = True

        for idx, x, y, cell in zip(new_idxs, xs, ys, new_cells):
            cell.idx = idx
            cell.position = (x, y)
            self.cells.append(cell)

        # cell is supposed to have the same concentrations as the pxl it lives on
        self.cell_molecules[new_idxs, :] = self.molecule_map[:, xs, ys].T

    def _add_new_cells_to_proteome_params(
        self, proteomes: list[list[Protein]], cell_idxs: list[int]
    ):
        calc_cell_params(
            proteomes=proteomes,
            n_signals=2 * self.n_molecules,
            cell_idxs=cell_idxs,
            mol_2_idx=self.mol_2_idx,
            Km=self.affinities,
            Vmax=self.velocities,
            E=self.energies,
            N=self.stoichiometry,
            A=self.regulators,
        )

    def _get_cell_ext_molecules(self, cell_idxs: list[int]) -> torch.Tensor:
        cells = [self.cells[i] for i in cell_idxs]
        xs = []
        ys = []
        for cell in cells:
            x, y = cell.position
            xs.append(x)
            ys.append(y)
        return self.molecule_map[:, xs, ys].T

    def _get_all_cell_ext_molecules(self) -> torch.Tensor:
        xs = []
        ys = []
        for cell in self.cells:
            x, y = cell.position
            xs.append(x)
            ys.append(y)
        return self.molecule_map[:, xs, ys].T

    def _add_cell_ext_molecules(self, cell_idxs: list[int], molecules: torch.Tensor):
        cells = [self.cells[i] for i in cell_idxs]
        xs = []
        ys = []
        for cell in cells:
            x, y = cell.position
            xs.append(x)
            ys.append(y)
        self.molecule_map[:, xs, ys] += molecules.T

    def _put_all_cell_ext_molecules(self, molecules: torch.Tensor):
        xs = []
        ys = []
        for cell in self.cells:
            x, y = cell.position
            xs.append(x)
            ys.append(y)
        self.molecule_map[:, xs, ys] = molecules.T

    def _get_molecule_map(self, mol_map_init: str) -> torch.Tensor:
        args = [self.n_molecules, self.map_size, self.map_size]
        if mol_map_init == "zeros":
            return self._tensor(*args)
        if mol_map_init == "randn":
            return torch.randn(*args, **self.torch_kwargs).abs()
        raise ValueError(f"Didnt recognize mol_map_init={mol_map_init}")

    def _get_conv(self, mol_diff_rate: float) -> torch.nn.Conv2d:
        if not 0.0 <= mol_diff_rate <= 1.0:
            raise ValueError(
                "Diffusion rate must be between 0 and 1. "
                f"Now it's mol_diff_rate={mol_diff_rate}"
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
        mol_2_idx = {}
        for mol_i, mol in enumerate(molecules):
            mol_2_idx[(mol, False)] = mol_i
            mol_2_idx[(mol, True)] = mol_i + self.n_molecules
        return mol_2_idx

    def _get_pad_2_true_idx_map(self, size: int) -> dict[int, int]:
        return {pad_2_true_idx(idx=i, size=size) for i in range(size + 2)}

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(map_size=%r,abs_temp=%r,mol_degrad=%r,mol_diff_rate=%r,trunc_n_decs=%r,n_cells=%s)"
            % (
                clsname,
                self.map_size,
                self.abs_temp,
                self.mol_degrad,
                self.mol_diff_rate,
                self.trunc_n_decs,
                len(self.cells),
            )
        )
