from typing import Optional
import random
import torch
from .proteins import Molecule, Protein
from .util import trunc, randstr, GAS_CONSTANT
from .kinetics import integrate_signals, get_cell_params

# TODO: refactor API:
#       - adding new cells
#       - getting a cell with all details
#       - recalculating certain cells after they were mutated
#       - get updated molecule map
#       - dividing a cell
#       - killing a cell
#       - moving a cell

# TODO: max_n_cells (avoid torch.cat)


# TODO: fit diffusion rate to natural diffusion rate of small molecules in cell
# TODO: summary()


class Cell:
    """
    Container for information about a cell
    """

    def __init__(self, genome: str, name: Optional[str] = None):
        self.genome = genome
        self.name = name or randstr()
        self.position: Optional[tuple[int, int]] = None
        self.n_survived_steps: Optional[int] = None
        self.molecules: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(genome=%r,name=%r)" % (clsname, self.genome, self.name)

    def __str__(self) -> str:
        return self.name


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
        n_max_proteins=1000,
        trunc_n_decs=4,
        device="cpu",
        dtype=torch.float,
    ):
        self.map_size = map_size
        self.abs_temp = abs_temp
        self.n_max_proteins = n_max_proteins
        self.mol_degrad = mol_degrad
        self.mol_diff_rate = mol_diff_rate
        self.trunc_n_decs = trunc_n_decs
        self.dtype = dtype
        self.device = device
        self.torch_kwargs = {"dtype": dtype, "device": device}

        self.molecules = molecules
        self.n_molecules = len(molecules)
        self.n_signals = 2 * self.n_molecules

        # ordering of signals: internal molecules, external molecules
        self.mol_2_idx = self._get_mol_2_idx_map(molecules=molecules)

        self.molecule_map = self._get_molecule_map(mol_map_init=mol_map_init)
        self.cell_map = torch.zeros(map_size, map_size, device=device, dtype=torch.bool)
        self.conv113 = self._get_conv(mol_diff_rate=mol_diff_rate)

        self.cell_molecules = self._tensor(0, self.n_molecules)

        self.cells: list[Cell] = []
        self.cell_survival = self._tensor(0)
        self.cell_positions: list[tuple[int, int]] = []

        self.X = self._tensor(0, self.n_signals)
        self.Km = self._tensor(0, self.n_max_proteins, self.n_signals)
        self.Vmax = self._tensor(0, self.n_max_proteins)
        self.Ke = self._tensor(0, self.n_max_proteins)
        self.N = self._tensor(0, self.n_max_proteins, self.n_signals)
        self.A = self._tensor(0, self.n_max_proteins, self.n_signals)

    def get_cell(
        self, by_idx: Optional[int] = None, by_name: Optional[str] = None
    ) -> Cell:
        """Get a cell with all its information by name or index"""
        if by_idx is not None:
            idx = by_idx
        if by_name is not None:
            cell_names = [d.name for d in self.cells]
            idx = cell_names.index(by_name)
            if idx < 0:
                raise ValueError(f"Cell {by_name} not found")

        cell = self.cells[idx]
        cell.position = self.cell_positions[idx]
        cell.n_survived_steps = int(self.cell_survival[idx].item())
        cell.molecules = self.cell_molecules[idx]
        return cell

    def add_random_cells(self, genomes: list[str], proteomes: list[list[Protein]]):
        """
        Randomly place cells on cell map and fill them with the molecules that
        were present at their respective pixels.
        """
        if len(genomes) != len(proteomes):
            raise ValueError(
                "Genomes and proteomes should be of equal length. "
                "They both represent the same list of cells. "
                f"Now, there are {len(genomes)} genomes and {len(proteomes)} proteomes."
            )

        n_cells = len(genomes)
        self.cells.extend(Cell(genome=d) for d in genomes)
        positions = self._place_new_cells_in_random_positions(n_cells=n_cells)

        self._add_new_cells_to_proteome_params(proteomes=proteomes)
        self._add_new_cells_to_x(positions=positions)

        cell_survival = torch.zeros(n_cells, dtype=self.dtype, device=self.device)
        self.cell_survival = torch.cat([self.cell_survival, cell_survival], dim=0)

    @torch.no_grad()
    def diffuse_molecules(self):
        """Let molecules in world map diffuse by 1 time step"""
        before = self.molecule_map.unsqueeze(1)
        after = trunc(tens=self.conv113(before), n_decs=self.trunc_n_decs)
        self.molecule_map = torch.squeeze(after, 1)

    def degrade_molecules(self):
        """Degrade molecules in world map and cells by 1 time step"""
        self.molecule_map = trunc(
            tens=self.molecule_map * self.mol_degrad, n_decs=self.trunc_n_decs
        )
        self.cell_molecules = trunc(
            tens=self.cell_molecules * self.mol_degrad, n_decs=self.trunc_n_decs
        )

    def increment_cell_survival(self):
        """Increment number of current cells' time steps by 1"""
        self.cell_survival = self.cell_survival + 1

    def integrate_signals(self):
        """Integrate signals and update molecule maps"""
        self._send_molecules_from_world_to_x()
        Xd = integrate_signals(
            X=self.X, Km=self.Km, Vmax=self.Vmax, Ke=self.Ke, N=self.N, A=self.A
        )
        self.X += trunc(Xd, n_decs=self.trunc_n_decs)
        self._send_molecules_from_x_to_world()

    def _place_new_cells_in_random_positions(
        self, n_cells: int
    ) -> list[tuple[int, int]]:
        pxls = torch.argwhere(~self.cell_map)

        try:
            idxs = random.sample(range(len(pxls)), k=n_cells)
        except ValueError as err:
            raise ValueError(
                f"Wanted to add {n_cells} cells but only {len(pxls)} pixels left on map"
            ) from err

        # TODO: doublecheck this
        xs = [d[0] for d in pxls]
        ys = [d[1] for d in pxls]
        cell_molecules = self.molecule_map[:, xs, ys].T
        self.cell_molecules = torch.cat([self.cell_molecules, cell_molecules], dim=0)

        positions: list[tuple[int, int]] = [tuple(d.tolist()) for d in pxls[idxs]]  # type: ignore
        self.cell_positions.extend(positions)

        return positions

    def _add_new_cells_to_x(self, positions: list[tuple[int, int]]):
        X = torch.zeros(len(positions), self.n_signals, dtype=self.dtype)
        for cell_i, (x, y) in enumerate(positions):
            for mol_i in range(self.n_molecules):
                X[cell_i, mol_i] = self.cell_molecules[cell_i, mol_i]
                X[cell_i, mol_i + self.n_molecules] = self.molecule_map[mol_i, x, y]
        self.X = torch.cat([self.X, X.to(self.device)], dim=0)

    def _add_new_cells_to_proteome_params(self, proteomes: list[list[Protein]]):
        n_cells = len(proteomes)
        kwargs = {"dtype": self.dtype, "device": self.device}

        Km = torch.zeros(n_cells, self.n_max_proteins, self.n_signals, **kwargs)
        Vmax = torch.zeros(n_cells, self.n_max_proteins, **kwargs)
        E = torch.zeros(n_cells, self.n_max_proteins, **kwargs)
        N = torch.zeros(n_cells, self.n_max_proteins, self.n_signals, **kwargs)
        A = torch.zeros(n_cells, self.n_max_proteins, self.n_signals, **kwargs)

        Km, Vmax, E, N, A = get_cell_params(
            proteomes=proteomes,
            n_signals=self.n_signals,
            mol_2_idx=self.mol_2_idx,
            Km=Km,
            Vmax=Vmax,
            E=E,
            N=N,
            A=A,
        )

        Ke = torch.exp(-E / self.abs_temp / GAS_CONSTANT)

        self.Km = torch.cat([self.Km, Km], dim=0)
        self.Vmax = torch.cat([self.Vmax, Vmax], dim=0)
        self.Ke = torch.cat([self.Ke, Ke], dim=0)
        self.N = torch.cat([self.N, N], dim=0)
        self.A = torch.cat([self.A, A], dim=0)

    def _send_molecules_from_world_to_x(self):
        for ci, (x, y) in enumerate(self.cell_positions):
            for mi in range(self.n_molecules):
                self.X[ci, mi] = self.cell_molecules[ci, mi]
                self.X[ci, mi + self.n_molecules] = self.molecule_map[mi, x, y]

    def _send_molecules_from_x_to_world(self):
        for ci, (x, y) in enumerate(self.cell_positions):
            for mi in range(self.n_molecules):
                self.cell_molecules[ci, mi] = self.X[ci, mi]
                self.molecule_map[mi, x, y] = self.X[ci, mi + self.n_molecules]

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

    def _tensor(self, *args) -> torch.Tensor:
        return torch.zeros(*args, **self.torch_kwargs)

    def _get_mol_2_idx_map(
        self, molecules: list[Molecule]
    ) -> dict[tuple[Molecule, bool], int]:
        n_mols = len(molecules)
        mol_2_idx = {}
        for mol_i, mol in enumerate(molecules):
            mol_2_idx[(mol, False)] = mol_i
            mol_2_idx[(mol, True)] = mol_i + n_mols
        return mol_2_idx

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(molecules=%r,map_size=%r,abs_temp=%r,mol_degrad=%r,mol_diff_rate=%r,n_max_proteins=%r,trunc_n_decs=%r,n_cells=%s)"
            % (
                clsname,
                self.molecules,
                self.map_size,
                self.abs_temp,
                self.mol_degrad,
                self.mol_diff_rate,
                self.n_max_proteins,
                self.trunc_n_decs,
                len(self.cells),
            )
        )
