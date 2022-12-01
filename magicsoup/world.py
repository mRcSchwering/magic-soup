from typing import Optional
import random
import torch
from .genetics import Signal, Protein
from .util import trunc, randstr


# TODO: fit diffusion rate to natural diffusion rate of small molecules in cell
# TODO: use Nernst equation in Z matrix


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
        self.actions: Optional[torch.Tensor] = None


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
        molecules: list[Signal],
        actions: list[Signal],
        map_size=128,
        mol_degrad=0.9,
        mol_diff_rate=1.0,
        mol_map_init="randn",
        n_max_proteins=1000,
        trunc_n_decs=4,
        device="cpu",
        dtype=torch.float,
    ):
        self.map_size = map_size
        self.n_max_proteins = n_max_proteins
        self.mol_degrad = mol_degrad
        self.trunc_n_decs = trunc_n_decs
        self.torch_kwargs = {"dtype": dtype, "device": device}

        self.molecules = molecules
        self.actions = actions
        self.n_molecules = len(molecules)
        self.n_actions = len(actions)
        self.n_signals = 2 * self.n_molecules + self.n_actions

        # ordering of signals:
        # internal molecules, internal actions, external molecules
        self.in_mol_pad = 0
        self.in_act_pad = self.in_mol_pad + self.n_molecules
        self.ex_mol_pad = self.in_act_pad + self.n_actions

        self.molecule_map = self._get_molecule_map(mol_map_init=mol_map_init)
        self.cell_map = torch.zeros(map_size, map_size, device=device, dtype=torch.bool)
        self.conv113 = self._get_conv(mol_diff_rate=mol_diff_rate)

        self.cell_molecules = self._tensor(0, self.n_molecules)
        self.cell_actions = self._tensor(0, self.n_actions)

        self.cells: list[Cell] = []
        self.cell_survival = self._tensor(0)
        self.cell_positions: list[tuple[int, int]] = []

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
        cell.actions = self.cell_actions[idx]
        return cell

    def add_cells(self, genomes: list[str]):
        """
        Randomly place cells on cell map and fill them with the molecules that
        were present at their respective pixels.
        """
        n_cells = len(genomes)
        pxls = torch.argwhere(~self.cell_map)

        try:
            idxs = random.sample(range(len(pxls)), k=n_cells)
        except ValueError as err:
            raise ValueError(
                f"Wanted to add {n_cells} cells but only {len(pxls)} pixels left on map"
            ) from err

        self.cell_positions.extend(tuple(d.tolist()) for d in pxls[idxs])  # type: ignore
        self.cells.extend(Cell(genome=d) for d in genomes)

        # TODO: get parent actions, molecules (for duplication events only)
        #       replication action should be set to 0 first

        # TODO: also doublecheck this
        xs = [d[0] for d in pxls]
        ys = [d[1] for d in pxls]
        cell_molecules = self.molecule_map[:, xs, ys].T

        cell_actions = self._tensor(n_cells, self.n_actions)
        cell_survival = self._tensor(n_cells)
        self.cell_molecules = torch.concat([self.cell_molecules, cell_molecules], dim=0)
        self.cell_actions = torch.concat([self.cell_actions, cell_actions], dim=0)
        self.cell_survival = torch.concat([self.cell_survival, cell_survival], dim=0)

    @torch.no_grad()
    def diffuse_molecules(self):
        """Let molecules in world map diffuse by 1 time step"""
        before = self.molecule_map.unsqueeze(1)
        after = trunc(tens=self.conv113(before), n_decs=self.trunc_n_decs)
        self.molecule_map = torch.squeeze(after, 1)

    def degrade_molecules(self):
        """Truncate molecules in world map and cells by 1 time step"""
        self.molecule_map = trunc(
            tens=self.molecule_map * self.mol_degrad, n_decs=self.trunc_n_decs
        )
        self.cell_molecules = trunc(
            tens=self.cell_molecules * self.mol_degrad, n_decs=self.trunc_n_decs
        )

    def increment_cell_survival(self):
        """Increment number of current cells' time steps by 1"""
        self.cell_survival = self.cell_survival + 1

    def get_signals(self) -> torch.Tensor:
        """
        Create signals tensor for all cells from all sources of signals
        """
        X = self._tensor(len(self.cells), self.n_signals)
        for cell_i, (x, y) in enumerate(self.cell_positions):
            for mol_i in range(len(self.molecules)):
                X[cell_i, mol_i + self.in_mol_pad] = self.cell_molecules[cell_i, mol_i]
                X[cell_i, mol_i + self.ex_mol_pad] = self.molecule_map[mol_i, x, y]
            for act_i in range(len(self.actions)):
                X[cell_i, act_i + self.in_act_pad] = self.cell_actions[cell_i, act_i]
        return X

    def update_signals(self, X: torch.Tensor):
        """
        Update all sources of signals for all cells with signals tensor `X`

        - `X` new signals tensor of shape `(c, s)` (`c` cells, `s` signals)

        This method relies on the ordering of cells and signals to reduce computation.
        If cells or signals are somehow shuffled this method will silently return
        wrong results.
        """
        for cell_i, (x, y) in enumerate(self.cell_positions):
            for mol_i in range(len(self.molecules)):
                self.cell_molecules[cell_i, mol_i] = X[cell_i, mol_i + self.in_mol_pad]
                self.molecule_map[mol_i, x, y] = X[cell_i, mol_i + self.ex_mol_pad]
            for act_i in range(len(self.actions)):
                self.cell_actions[cell_i, act_i] = X[cell_i, act_i + self.in_act_pad]

    def get_cell_params(
        self, proteomes: list[list[Protein]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate tensors A, B, Z from cell proteomes

        Returns tuple (A, B, Z):

        - `A` parameters defining how incomming signals map to proteins, `(c, s, p)`
        - `B` parameters defining how much proteins can produce output signals, `(c, s, p)`
        - `Z` binary matrix defining which protein can produce output at all, `(c, p)`
        
        where `s` is the number of signals, `p` is the number of proteins,
        and `c` is the number of cells.
        """
        n_cells = len(proteomes)
        A = self._tensor(n_cells, self.n_signals, self.n_max_proteins)
        B = self._tensor(n_cells, self.n_signals, self.n_max_proteins)
        Z = self._tensor(n_cells, self.n_max_proteins)
        for cell_i, cell in enumerate(proteomes):
            for prot_i, protein in enumerate(cell):
                net_energy = 0.0
                for dom in protein.domains:
                    net_energy += dom.energy
                    if dom.signal.is_molecule:
                        idx = self.molecules.index(dom.signal)
                        if dom.is_transmembrane:
                            offset = self.ex_mol_pad
                        else:
                            offset = self.in_mol_pad
                    else:
                        idx = self.actions.index(dom.signal)
                        offset = self.in_act_pad
                    if dom.is_receptor:
                        A[cell_i, idx + offset, prot_i] = dom.weight
                    else:
                        B[cell_i, idx + offset, prot_i] = dom.weight
                if net_energy <= 0:
                    Z[cell_i, prot_i] = 1.0
        return (A, B, Z)

    def integrate_signals(
        self, X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, Z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate new signals (molecules/actions) created by all proteins of all cells
        after 1 timestep. Returns the change in signals in shape `(c, s)`.

        - `X` initial signals tensor, `(c, s)`
        - `A` parameters defining how incomming signals map to proteins, `(c, s, p)`
        - `B` parameters defining how much proteins can produce output signals, `(c, s, p)`
        - `Z` binary matrix defining which protein can produce output at all, `(c, p)`

        where `s` signals, `p` proteins, `c` cells.
        Proteins' output is defined by:
        
        ```
            f(X, A, B, Z) = sig(X * A) * (B * Z)
            
            sig(x) = max(1, min(0, 1 - exp(-x ** 3)))
        ```

        There's currently a chance that proteins in a cell can deconstruct more
        of a molecule than available. This would lead to a negative concentration
        in the resulting signals `X`. To avoid this, there is a correction heuristic
        which will downregulate these proteins by so much, that the molecule will
        only be reduced to 0.
        """
        # protein activity, matrix (c x p)
        P_act = torch.einsum("ij,ijk->ik", X, A)
        P_act = torch.clamp(1 - torch.exp(-(P_act ** 3)), min=0, max=1)

        # protein output potentials, matrix (c x s x p)
        B_adj = torch.einsum("ijk,ik->ijk", B, Z)

        # protein output, matrix (c x s)
        X_delta = torch.einsum("ij,ikj->ik", P_act, B_adj)

        # check for possible negative concentrations
        X_1 = X + X_delta
        if torch.any(X_1 < 0):

            # which signals will be negative (c, s)
            X_mask = torch.where(X_1 < 0, 1.0, 0.0)

            # which values in B could be responsible (c, s, p)
            B_mask = torch.where(B_adj < 0.0, 1.0, 0.0)

            # which proteins need to be down-regulated (c, p)
            BX_mask = torch.einsum("ijk,ij->ijk", B_mask, X_mask)
            Z_mask = BX_mask.max(dim=1).values

            # what are the correction factors (c,)
            correct = torch.where(X_mask > 0.0, -X / X_delta, 1.0).min(dim=1).values

            # correction matrix for B (c, p)
            Z_adj = torch.einsum("ij,i->ij", Z_mask, correct)
            Z_adj = torch.where(Z_adj == 0.0, 1.0, Z_adj)

            # new protein output, matrix (c x s)
            B_adj = torch.einsum("ijk,ik->ijk", B_adj, Z_adj)
            X_delta = torch.einsum("ij,ikj->ik", P_act, B_adj)

        return trunc(tens=X_delta, n_decs=self.trunc_n_decs)

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

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(map_size=%r,n_cells=%s,molecules=%r,actions=%r)" % (
            clsname,
            self.map_size,
            len(self.cells),
            self.molecules,
            self.actions,
        )

