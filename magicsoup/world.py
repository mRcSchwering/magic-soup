from typing import Optional
import random
import torch
from .proteins import Molecule, Protein
from .util import randstr, GAS_CONSTANT
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

    def __init__(self, genome: str, name: Optional[str] = None):
        self.genome = genome
        self.name = name or randstr()
        self.position: Optional[tuple[int, int]] = None
        self.n_survived_steps: Optional[int] = None
        self.int_molecules: Optional[torch.Tensor] = None
        self.ext_molecules: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(genome=%r,name=%r)" % (clsname, self.genome, self.name)

    def __str__(self) -> str:
        return str(self.int_molecules)


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

        # ordering of signals: internal molecules, external molecules
        self.mol_2_idx = self._get_mol_2_idx_map(molecules=molecules)
        self.int_mol_idxs = list(range(len(molecules)))
        self.ext_mol_idxs = list(range(len(molecules), len(molecules) * 2))
        self.n_molecules = len(molecules)  # number of molecule types
        self.n_signals = 2 * self.n_molecules  # int + ext molecules

        self.molecule_map = self._get_molecule_map(mol_map_init=mol_map_init)
        self.cell_map = self._tensor(map_size, map_size, dtype=torch.bool)
        self.conv113 = self._get_conv(mol_diff_rate=mol_diff_rate)

        # self._cell_molecules = self._tensor(0, self.n_molecules)

        self.cells: list[Cell] = []
        self.cell_survival = self._tensor(0)
        self.cell_positions: list[tuple[int, int]] = []

        self.X = self._tensor(0, self.n_signals)
        self.affinities = self._tensor(0, 0, self.n_signals)
        self.velocities = self._tensor(0, 0)
        self.energies = self._tensor(0, 0)
        self.stoichiometry = self._tensor(0, 0, self.n_signals)
        self.regulators = self._tensor(0, 0, self.n_signals)

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
        cell.int_molecules = self.X[idx, self.int_mol_idxs]
        cell.ext_molecules = self.X[idx, self.ext_mol_idxs]
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

        n_new_cells = len(genomes)
        self._expand_max_cells(by_n=n_new_cells)

        n_prots = int(self.affinities.shape[1])
        n_max_prots = max(len(d) for d in proteomes)
        if n_max_prots > n_prots:
            self._expand_max_proteins(by_n=n_max_prots - n_prots)

        cell_idxs = list(range(len(self.cells), len(self.cells) + n_new_cells))
        self.cells.extend(Cell(genome=d) for d in genomes)
        self._place_new_cells_in_random_positions(cell_idxs=cell_idxs)

        self._add_new_cells_to_proteome_params(proteomes=proteomes, cell_idxs=cell_idxs)

    @torch.no_grad()
    def diffuse_molecules(self):
        """Let molecules in world map diffuse by 1 time step"""
        before = self.molecule_map.unsqueeze(1)
        after = self.conv113(before)
        self.molecule_map = torch.squeeze(after, 1)

    def degrade_molecules(self):
        """Degrade molecules in world map and cells by 1 time step"""
        self.molecule_map = self.molecule_map * self.mol_degrad
        self.X = self.X * self.mol_degrad

    def increment_cell_survival(self):
        """Increment number of current cells' time steps by 1"""
        idxs = list(range(len(self.cells)))
        self.cell_survival[idxs] += 1

    def integrate_signals(self):
        """Integrate signals and update molecule maps"""
        self._send_molecules_from_world_to_x()
        Xd = integrate_signals(
            X=self.X,
            Km=self.affinities,
            Vmax=self.velocities,
            Ke=torch.exp(-self.energies / self.abs_temp / GAS_CONSTANT),
            N=self.stoichiometry,
            A=self.regulators.clamp(-1.0, 1.0),
        )
        self.X += Xd
        self._send_molecules_from_x_to_world()

    def _expand_max_cells(self, by_n: int):
        self.cell_survival = self._expand(t=self.cell_survival, n=by_n, d=0)
        self.X = self._expand(t=self.X, n=by_n, d=0)
        self.affinities = self._expand(t=self.affinities, n=by_n, d=0)
        self.velocities = self._expand(t=self.velocities, n=by_n, d=0)
        self.energies = self._expand(t=self.energies, n=by_n, d=0)
        self.stoichiometry = self._expand(t=self.stoichiometry, n=by_n, d=0)
        self.regulators = self._expand(t=self.regulators, n=by_n, d=0)

    def _expand_max_proteins(self, by_n: int):
        self.affinities = self._expand(t=self.affinities, n=by_n, d=1)
        self.velocities = self._expand(t=self.velocities, n=by_n, d=1)
        self.energies = self._expand(t=self.energies, n=by_n, d=1)
        self.stoichiometry = self._expand(t=self.stoichiometry, n=by_n, d=1)
        self.regulators = self._expand(t=self.regulators, n=by_n, d=1)

    def _place_new_cells_in_random_positions(self, cell_idxs: list[int]):
        n_cells = len(cell_idxs)
        pxls = torch.argwhere(~self.cell_map)

        try:
            idxs = random.sample(range(len(pxls)), k=n_cells)
        except ValueError as err:
            raise ValueError(
                f"Wanted to add {n_cells} cells but only {len(pxls)} pixels left on map"
            ) from err

        positions: list[tuple[int, int]] = [tuple(d.tolist()) for d in pxls[idxs]]  # type: ignore
        self.cell_positions.extend(positions)

        xs = [d[0] for d in positions]
        ys = [d[1] for d in positions]

        # cell is supposed to have the same concentrations as the pxl it lives on
        # so I can just repeat the external concentrations for internal
        self.X[cell_idxs, :] = self.molecule_map[:, xs, ys].T.repeat(1, 2)

    def _add_new_cells_to_proteome_params(
        self, proteomes: list[list[Protein]], cell_idxs: list[int]
    ):
        calc_cell_params(
            proteomes=proteomes,
            n_signals=self.n_signals,
            cell_idxs=cell_idxs,
            mol_2_idx=self.mol_2_idx,
            Km=self.affinities,
            Vmax=self.velocities,
            E=self.energies,
            N=self.stoichiometry,
            A=self.regulators,
        )

    def _send_molecules_from_world_to_x(self):
        xs = [d[0] for d in self.cell_positions]
        ys = [d[1] for d in self.cell_positions]
        self.X[:, self.ext_mol_idxs] = self.molecule_map[:, xs, ys].T

    def _send_molecules_from_x_to_world(self):
        xs = [d[0] for d in self.cell_positions]
        ys = [d[1] for d in self.cell_positions]
        self.molecule_map[:, xs, ys] = self.X[:, self.ext_mol_idxs].T

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
        n_mols = len(molecules)
        mol_2_idx = {}
        for mol_i, mol in enumerate(molecules):
            mol_2_idx[(mol, False)] = mol_i
            mol_2_idx[(mol, True)] = mol_i + n_mols
        return mol_2_idx

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
