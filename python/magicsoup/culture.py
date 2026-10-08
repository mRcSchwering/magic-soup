from pathlib import Path
from typing import TypedDict

import torch

from magicsoup import rs
from magicsoup.chemistry import Chemistry
from magicsoup.util import TensorClass


class CultureKwargs(TypedDict, total=False):
    map_size: int


class Culture(TensorClass):
    r"""
    Represents a cell culture.
    Usually this class is instantiated automatically when initializing [World][magicsoup.world.World].
    You can access it on `world.culture`.

    Arguments:
        chemistry: Simulation [Chemistry][magicsoup.containers.Chemistry]
        map_size: Side length of the square 2D world map in pixels.
        device: Device to use for tensors
            (see [pytorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)).
            This has to be aligned with [World][magicsoup.world.World].
        ftype: Dtype of float tensors used.
            This has to be aligned with [World][magicsoup.world.World].
        itype: Dtype of integer tensors used.
            This has to be aligned with [World][magicsoup.world.World].

    Cells are living on the `cell_map` which is a (L, L) integer tensor of indices with L being `map_size`.
    All empty pixels in this map have value -1.
    Each pixel occupied by a cell has its cell index as value.
    The map is square and has a side length of `map_size` pixels.
    Thus, the maximum number of cells possible is `map_size * map_size`.

    Molecule concentrations are managed on `molecule_map` which is a (M, L, L) float tensor withh M being the number of molecules.
    Molecule concentrations are ordered as the molecules in the provided [Chemistry][magicsoup.containers.Chemistry] object.
    _I.e._ molecule i on the `molecule_map` represents molecule i in `chemistry.molecules`.

    If a cell occupies a pixel on the `cell_map` it can interact with the molecules of the same pixel in the `molecule_map`.
    _I.e._ a cell occupying (i, j) on the `cell_map` can interact with the molecules of (i, j) on the `molecule_map`.

    Diffusion only moves molecules on the `molecule_map`.
    """

    def __init__(
        self,
        chemistry: Chemistry,
        map_size: int = 128,
        device: str = "cpu",
        ftype: torch.dtype = torch.float32,
        itype: torch.dtype = torch.int8,
    ):
        super().__init__(device=device, ftype=ftype, itype=itype)

        molecules = chemistry.molecules
        self.map_size = map_size
        self.pixels = map_size**2

        # describes all coordinates
        self.coord_map = self._get_coord_map()

        # holds cell idxs, -1 for empty pixels
        self.cell_map: torch.Tensor = self.idxfull(map_size, map_size, value=-1)

        # flat representation of cell idxs (shares memory)
        self.cell_map_flat = self.cell_map.view(-1)  # (s * s,)

        # holds molecule concentrations
        self.molecule_map: torch.Tensor = self.fzeros(
            len(molecules), map_size, map_size
        )

        self._diffusion_funs: list[torch.nn.Conv2d] = [
            self._get_diffuse(mol_diff_rate=m.diffusivity) for m in molecules
        ]
        self._nb_offs = self._get_moores_neighborhood()
        self._hlf_nb_offs = self._get_half_moores_neighborhood()

        # setup rust class
        self._setup_rs()

    @torch.no_grad()
    def diffuse_molecules(self):
        # TODO: calibrate diffusion to physics
        for mol_i, diffuse in enumerate(self._diffusion_funs):
            before = self.molecule_map[mol_i].unsqueeze(0).unsqueeze(1)
            after = diffuse(before)
            self.molecule_map[mol_i] = torch.squeeze(after, 0).squeeze(0)

    def cell_neighbor_pairs(self) -> torch.Tensor:
        """
        Get all unique pairs of neighbouring cells.

        Returns:
            (k, 2) tensor of indices of k unique cell neighbors

        Cells are neighbors if they are within each others Moore's neighbourhood.
        Note: With map size smaller 3 duplicate neighbors can be returned.
        """
        L = self.map_size
        N = L * L

        pos = self.get_cell_positions()  # (N, 2) with (-1, -1) for empty
        valid = pos[:, 0] >= 0  # N rows which are not empty

        # look up half cell neighborhoods to only get unique pairs (N, 4, 2)
        nbs = self._get_neighborhoods(coords=pos, offs=self._hlf_nb_offs)
        flat_nbs = self._flatten_coordinates(nbs)  # (N, 4)

        nb_cells = self.cell_map_flat[flat_nbs]  # (N, 4)
        src = self.idxarange(N)[:, None].expand_as(nb_cells)

        mask = valid[:, None] & (nb_cells >= 0)  # (N, 4) with K true
        return torch.stack((src[mask], nb_cells[mask]), dim=1)  # (K, 2)

    def migrate_cells(
        self,
        cell_idxs: torch.Tensor,  # i32 (n,) indices of cells that want to migrate
        n_max_rounds: int = 3,
    ) -> torch.Tensor:
        """
        Let cells migrate and return indices of successfully migrated cells.

        Arguments:
            cell_idxs: (n,) tensor of indices of n cells that want to migrate
            n_max_rounds: how many rounds to resolve possible conflicts among cells (3 is decent)

        Returns:
            (k,) tensor of indices of k successfully migrated cells

        A cell can only migrate if it finds a free spot in its Moore's neighbourhood.
        """
        # For each cell:
        # - get free Moore's neighborhood
        # - randomly choose a target pixel in neighborhood
        # Gather all chosen pixels (let one cell win if clashes)
        # Commit winners
        # Repeat with left overs

        # snapshot of current positions, read-only below
        pos = self.get_cell_positions()

        # generate shuffled indices for cell_idxs
        n = cell_idxs.numel()
        pending = self.idxrandperm(n)

        # target pixel indices (flattened) of successful cells
        success = self.idxfull(n, value=-1)

        for _ in range(n_max_rounds):
            p = pending.numel()  # number of cells still trying to act
            if p == 0:
                break

            # cell idxs of pending cells
            idxs_pend = cell_idxs[pending]

            # 8 neighbourhoods of each pending cells (p, 8, 2)
            nbs = self._get_neighborhoods(coords=pos[idxs_pend], offs=self._nb_offs)

            # generate target neighbourhood and choose who gets it (p,)
            target_nbs, winners = self._choose_neighbourhood(nbs=nbs, p=p)

            # commit winners
            w_idxs = pending[winners]  # indices for idx
            w_tgts = target_nbs[winners]  # indices for target pixel

            # set cell idx on pixel (also affects 2d map)
            self.cell_map_flat[w_tgts] = cell_idxs[w_idxs]

            success[w_idxs] = w_tgts  # add target pixel index to result

            # migrants vacate their old pixel
            old = pos[cell_idxs[w_idxs]]
            self.cell_map_flat[self._flatten_coordinates(old)] = -1

            # cells with no free neighbor stay too, migrate may vacate nearby
            pending = pending[~winners]  # (p,)

        return cell_idxs[success >= 0]

    def divide_cells(
        self,
        parent_idxs: torch.Tensor,  # i32 (n,) indices of cells that want to divide
        child_idxs: torch.Tensor,  # i32 (n,) reserved indices for child cells
        n_max_rounds: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Let cells divide and return parent and child indices of successfully divided cells.

        Arguments:
            parent_idxs: (n,) tensor of indices of n cells that want to divide
            child_idxs: (n,) tensor of prepared indices for new cells to assume if division is successful
            n_max_rounds: how many rounds to resolve possible conflicts among cells (3 is decent)

        Returns:
            Tuple `parent_idxs, child_idxs`
            - `parent_idxs`: (k,) tensor of indices of k successfully divided cells
            - `child_idxs`: (k,) tensor of indices of respective new cells of successfully divided cells

        Position `i` of `parent_idxs` always relates to position `i` of `child_indxs` in both inputs and outputs.
        _I.e._ if `parent_idxs=[1, 2], child_idxs=[3, 4]` is provided but only cell `2` can divide,
        the returned values will be `parent_idxs=[2], child_idxs=[4]`.
        A cell can only divide if it finds a free spot in its Moore's neighbourhood.
        """
        # For each cell:
        # - get free Moore's neighborhood
        # - randomly choose a target pixel in neighborhood
        # Gather all chosen pixels (let one cell win if clashes)
        # Commit winners
        # Repeat with left overs

        # snapshot of current positions, read-only below
        pos = self.get_cell_positions()

        # generate shuffled indices for cell_idxs
        n = parent_idxs.numel()
        pending = self.idxrandperm(n)

        # target pixel indices (flattened) of successful cells
        success = self.idxfull(n, value=-1)

        for _ in range(n_max_rounds):
            p = pending.numel()  # number of cells still trying to act
            if p == 0:
                break

            # cell idxs of pending cells
            idxs_pend = parent_idxs[pending]

            # 8 neighbourhoods of each pending cells (p, 8, 2)
            nbs = self._get_neighborhoods(coords=pos[idxs_pend], offs=self._nb_offs)

            # generate target neighbourhood and choose who gets it (p,)
            target_nbs, winners = self._choose_neighbourhood(nbs=nbs, p=p)

            # commit winners
            w_idxs = pending[winners]  # indices for idx
            w_tgts = target_nbs[winners]  # indices for target pixel

            # occupy pixel (also affects 2d cells)
            self.cell_map_flat[w_tgts] = child_idxs[w_idxs]

            success[w_idxs] = w_tgts  # add target pixel index to result

            # cells with no free neighbor stay too, migrate may vacate nearby
            pending = pending[~winners]  # (p,)

        ok = success >= 0
        return parent_idxs[ok], child_idxs[ok]

    def get_cell_positions(self) -> torch.Tensor:
        """
        Get current coordinates of all cells on cell map.

        Returns:
            (n, 2) tensor of n coordinates representing the cell map

        All possible `n` coordinates are presented in the returned tensor.
        Each cell is contained with its x and y coordinates at the position of its index.
        _E.g._ if cell 5 is at (3, 0), then the tensor at index 5 contains values 3 and 0.
        All empty/missing cells contain coordinates (-1, -1).
        _E.g._ if there is no cell 4, then the tensor at index 4 contains values -1 and -1.
        """
        # build (N, 2) cell x-y cell positions from current map
        # row is cell index, empty rows are (-1, -1)
        N = self.map_size * self.map_size
        flat = self.cell_map_flat

        # map empty pixels into dummy row N + 1
        pos = self.idxfull(N + 1, 2, value=-1)  # (N + 1, 2)
        rows = torch.where(flat >= 0, flat, N)  # (N + 1, 2)
        pos[rows] = self.coord_map
        return pos[:N]  # (N, 2) remove dummy row

    def random_free_positions(self, n: int) -> torch.Tensor:
        """
        Get `n` available random positions on cell map.
        Returns fewer if less are avilable.

        Arguments:
            n: Desired number of random positions

        Returns:
            (k, 2) tensor of available coordinates with `k = min(n, number of free pixels)`
        """
        free = self.cell_map_flat < 0
        score = self.frand(free.shape).masked_fill(~free, -1.0)
        sel = score.topk(n).indices  # n distinct flat pixel indices
        sel = sel[free[sel]]  # drop occupied ones if fewer than n were free
        return self._unflatten_coordinates(sel)

    def save_state(self, statedir: Path) -> None:
        """
        Save current state

        Arguments:
            statedir: Directory to store files in

        Only attributes which update during the simulation are saved.
        """
        statedir = statedir / type(self).__name__
        statedir.mkdir(parents=True, exist_ok=True)
        torch.save(self.cell_map, statedir / "cell_map.pt")
        torch.save(self.molecule_map, statedir / "molecule_map.pt")

    def load_state(self, statedir: Path) -> None:
        """
        Load a saved world state.
        The state had to be saved with [save_state()][magicsoup.culture.Culture.save_state] previously.

        Arguments:
            statedir: Directory that contains all files of that state
        """
        statedir = statedir / type(self).__name__
        self.cell_map[:] = torch.load(
            statedir / "cell_map.pt", map_location=self.device
        )
        self.molecule_map[:] = torch.load(
            statedir / "molecule_map.pt", map_location=self.device
        )

    def _choose_neighbourhood(
        self, nbs: torch.Tensor, p: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_nbs = self._flatten_coordinates(nbs)  # (p, 8)

        # find free indexes on flat map for each neighborhood
        free_nbs = self.cell_map_flat[flat_nbs] < 0  # (p, 8)
        can_act_mask = free_nbs.any(1)  # (p,)

        # propose a target pixel on each flattened neighborhood
        nb_scores = self.frand(free_nbs.shape)  # (p, 8)
        nb_scores = nb_scores.masked_fill(~free_nbs, -1.0)  # (p, 8)
        nb_choices = nb_scores.argmax(1, keepdim=True).int()  # (p, 1)
        target_nbs = flat_nbs.gather(1, nb_choices).int().squeeze(1)  # (p,)

        # generate unique random priorities
        prios = self.idxrandperm(p)

        # if same target, higher priority wins
        best = self.idxfull(self.pixels, value=-1)
        best.scatter_reduce_(
            src=prios[can_act_mask],  # reduce from src
            index=target_nbs[can_act_mask],  # to indices index
            reduce="amax",  # via reduce argument (on src)
            dim=0,  # for dimension dim
        )  # (p,)
        winners = can_act_mask & (best[target_nbs] == prios)  # (p,)

        return target_nbs, winners  # int (p,), bool (p,)

    def _get_neighborhoods(
        self, coords: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        return (coords[:, None, :] + offs[None]) % self.map_size  # torus wrap

    def _get_diffuse(self, mol_diff_rate: float) -> torch.nn.Conv2d:
        if mol_diff_rate < 0.0:
            mol_diff_rate = -mol_diff_rate

        # mol_diff_rate > 1.0 could also mean expanding the kernel
        # so that molecules can diffuse more than just 1 pxl per round
        mol_diff_rate = min(mol_diff_rate, 1.0)

        if mol_diff_rate == 0.0:
            a = 0.0
            b = 1.0
        else:
            d = 1 / mol_diff_rate
            a = 1 / (d + 8)
            b = d * a
            b = b + 1.0 - (8 * a + b)  # try correcting inaccuracy

        # fmt: off
        kernel = self.ftensor([[[
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

    def _flatten_coordinates(self, xy: torch.Tensor) -> torch.Tensor:
        return xy[..., 0] * self.map_size + xy[..., 1]

    def _unflatten_coordinates(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack((z // self.map_size, z % self.map_size), dim=1)

    def _get_coord_map(self) -> torch.Tensor:
        L = self.map_size
        xs, ys = torch.meshgrid(self.idxarange(L), self.idxarange(L), indexing="ij")
        return torch.stack((xs, ys), dim=-1).view(-1, 2)  # (L*L, 2)

    def _get_moores_neighborhood(self) -> torch.Tensor:
        # fmt: off
        return self.idxtensor(
            [
                [-1, -1], [-1, 0], [-1, 1],
                [ 0, -1],          [ 0, 1],
                [ 1, -1], [ 1, 0], [ 1, 1],
            ]
        )
        # fmt: on

    def _get_half_moores_neighborhood(self) -> torch.Tensor:
        # fmt: off
        return self.idxtensor(
            [

                                   [ 0, 1],
                [ 1, -1], [ 1, 0], [ 1, 1],
            ]
        )
        # fmt: on

    def _setup_rs(self) -> None:
        self.rs = rs.Culture(size=self.map_size)
