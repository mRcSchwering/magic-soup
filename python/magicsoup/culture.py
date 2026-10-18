from pathlib import Path
from typing import TypedDict

import torch

from magicsoup.chemistry import Chemistry
from magicsoup.util import TensorClass, TensorKwargs


class MooreNeighborhood(TensorClass):
    """
    Describe Moore neighborhood on periodic N x N grid

    Arguments:
        N: side length of grid
        tensor_kwargs: Kwargs passed to [TensorClass][magicsoup.util.TensorClass]
    """

    def __init__(self, N: int, tensor_kwargs: TensorKwargs | None = None) -> None:
        super().__init__(**(tensor_kwargs or {}))
        self.N = N

        # fmt: off
        self.full = self.idxtensor(
            [
                [-1, -1], [-1, 0], [-1, 1],
                [ 0, -1],          [ 0, 1],
                [ 1, -1], [ 1, 0], [ 1, 1],
            ]
        )
        self.half = self.idxtensor(
            [

                                   [ 0, 1],
                [ 1, -1], [ 1, 0], [ 1, 1],
            ]
        )
        # fmt: on

    def get_neighborhoods(
        self, coords: torch.Tensor, half: bool = False
    ) -> torch.Tensor:
        """
        Get all Moore neighborhoods for provided coordinates

        Arguments:
            coords: (k, 2) index tensor for positions on N x N grid
            half: whether to use only half Moore neighborhood (e.g. to get unique paris)

        Returns:
            (k, 8, 2) or index tensor of all Moore neighborhoods for each provided coordinate,
                (k, 4, 2) if `half`
        """
        offs = self.half if half else self.full
        return (coords[:, None, :] + offs[None]) % self.N  # torus wrap


class CultureKwargs(TypedDict, total=False):
    map_size: int


class Culture(TensorClass):
    r"""
    Represents a cell culture on a periodic 2D map.

    Arguments:
        chemistry: Simulation [Chemistry][magicsoup.containers.Chemistry]
        map_size: Number of voxels along one side of the 2D map
        tensor_kwargs: Kwargs passed to [TensorClass][magicsoup.util.TensorClass]

    The cell culture is a periodic 2D grid of voxels represented by tensors.
    Each voxel can contain molecules and it can contain at most one cell.

    `cell_map` manages this cell culture as N x N index tensor where N is the `map_size`.
    Each cell can be identified by a cell index.
    All occupied voxels contain the index of the occupying cell.
    _E.g._ if cell 3 occupies (1, 2), then `cell_map[1, 2]` is `3`.
    All empty voxels in this map have value -1.

    Molecule concentrations are managed on `molecule_map` which is a (m, N, N) float tensor, `m` representing molecule species.
    Molecule species are ordered as in [Chemistry][magicsoup.containers.Chemistry].
    _E.g._ `molecule_map[1]` is the concentration field of `chemistry.molecules[1]`.
    Molecules can diffuse across the molecule map.

    A cell can interact with it the molecules of its voxel.
    _E.g._ if there is a cell in `cell_map[3, 4]`, it interacts with all molecules in `molecule_map[:, 3, 4]`.
    Molecules can permeate through the cell membrane or be transported across it by the cell itself.
    """

    def __init__(
        self,
        chemistry: Chemistry,
        map_size: int = 128,
        tensor_kwargs: TensorKwargs | None = None,
    ):
        super().__init__(**(tensor_kwargs or {}))
        molecules = chemistry.molecules
        self.map_size = map_size
        self.n_molecules = len(molecules)

        # describes all coordinates
        self.coord_map = self._get_coord_map()  # (N², 2)

        # holds cell idxs, -1 for empty voxels
        self.cell_map: torch.Tensor = self.idxfull(map_size, map_size, value=-1)

        # flat representation of cell idxs (shares memory)
        self.cell_map_flat = self.cell_map.view(-1)  # (N²,)

        # holds molecule concentrations
        self.molecule_map: torch.Tensor = self.fzeros(
            len(molecules), map_size, map_size
        )  # (m, N, N)

        self._nbh = MooreNeighborhood(N=self.map_size, tensor_kwargs=tensor_kwargs)

    def get_cell_neighbor_pairs(self) -> torch.Tensor:
        """
        Get all unique pairs of neighbouring cells.

        Returns:
            (k, 2) cell index tensor of all k unique cell neighbors

        Cells are neighbors if they are within each others Moore neighbourhood.
        Note: With map size smaller 3 duplicate neighbors can be returned.
        """
        L = self.map_size
        N2 = L * L

        pos = self.get_cell_positions()  # (N², 2) with (-1, -1) for empty
        valid = pos[:, 0] >= 0  # rows which are not empty

        # look up half cell neighborhoods to only get unique pairs (N², 4, 2)
        nbs = self._nbh.get_neighborhoods(coords=pos, half=True)
        flat_nbs = self._flatten_positions(nbs)  # (N², 4)

        nb_cells = self.cell_map_flat[flat_nbs]  # (N², 4)
        src = self.idxarange(N2)[:, None].expand_as(nb_cells)

        mask = valid[:, None] & (nb_cells >= 0)  # (N², 4) with k true
        return torch.stack((src[mask], nb_cells[mask]), dim=1)  # (k, 2)

    def get_cell_positions(self) -> torch.Tensor:
        """
        Get current positions of all cells on cell map.

        Returns:
            (N², 2) position index tensor encoding all cells with positions

        All possible N² positions are presented in the returned tensor.
        Each cell is contained with its row and column indexes at the position of its cell index.
        _E.g._ if cell 5 is at (3, 0), then the tensor at index 5 contains values 3 and 0.
        Unused cell indices contain (-1, -1).
        _E.g._ if there is no cell with index 4, then the tensor at index 4 contains values -1 and -1.
        """
        # build (N², 2) cell x-y cell positions from current map
        # row is cell index, empty rows are (-1, -1)
        N2 = self.map_size * self.map_size
        flat = self.cell_map_flat

        # map empty voxels into dummy row N² + 1
        pos = self.idxfull(N2 + 1, 2, value=-1)  # (N²+1, 2)
        rows = torch.where(flat >= 0, flat, N2)  # (N²+1, 2)
        pos[rows] = self.coord_map
        return pos[:N2]  # (N², 2) remove dummy row

    def get_free_positions(self, n: int) -> torch.Tensor:
        """
        Get `n` available random positions on cell map.
        Returns fewer if fewer are avilable.

        Arguments:
            n: Desired number of random positions

        Returns:
            (k, 2) position index tensor of available positions with `k = min(n, number of free voxels)`
        """
        free = self.cell_map_flat < 0
        score = self.frand(free.shape).masked_fill(~free, -1.0)
        sel = score.topk(n).indices  # n distinct flat voxel indices
        sel = sel[free[sel]]  # drop occupied ones if fewer than n were free
        return self._unflatten_positions(sel)

    def undirected_cell_migration(
        self,
        cell_idxs: torch.Tensor,  # i32 (n,) indices of cells that want to migrate
        n_max_rounds: int = 3,
    ) -> torch.Tensor:
        """
        Let cells migrate randomly around their Moore neighbourhood

        Arguments:
            cell_idxs: (n,) cell index tensor for n cells that want to migrate
            n_max_rounds: how many rounds to resolve possible conflicts among cells

        Returns:
            (k,) cell index tensor of k successfully migrated cells

        A cell can only migrate if it finds a free voxel in its Moore neighbourhood.
        """
        # For each cell:
        # - get free Moore neighborhood
        # - randomly choose a target voxel in neighborhood
        # Gather all chosen voxels (let one cell win in case of collisions)
        # Commit winners
        # Repeat with left-overs

        # snapshot of current positions, read-only below
        pos = self.get_cell_positions()

        # generate shuffled indices for cell_idxs
        n = cell_idxs.numel()
        remaining = self.idxrandperm(n)

        # target voxel indices (flattened) of successful cells
        success = self.idxfull(n, value=-1)

        for _ in range(n_max_rounds):
            r = remaining.numel()  # number of cells still trying to act
            if r == 0:
                break

            # cell idxs of remaining cells
            idxs_pend = cell_idxs[remaining]

            # 8 neighbourhoods of each remaining cells (r, 8, 2)
            nbs = self._nbh.get_neighborhoods(coords=pos[idxs_pend])

            # generate target neighbourhood and choose who gets it (r,)
            target_nbs, winners = self._choose_neighbourhood(nbs=nbs, r=r)

            # commit winners
            w_idxs = remaining[winners]  # indices for idx
            w_tgts = target_nbs[winners]  # indices for target voxel

            # set cell idx on voxel (also affects 2d map)
            self.cell_map_flat[w_tgts] = cell_idxs[w_idxs]

            success[w_idxs] = w_tgts  # add target voxel index to result

            # migrants vacate their old voxel
            old = pos[cell_idxs[w_idxs]]
            self.cell_map_flat[self._flatten_positions(old)] = -1

            # cells with no free neighbor stay too, migrate may vacate nearby
            remaining = remaining[~winners]  # (r,)

        return cell_idxs[success >= 0]

    def cell_division(
        self,
        parent_idxs: torch.Tensor,  # i32 (n,) indices of cells that want to divide
        child_idxs: torch.Tensor,  # i32 (n,) reserved indices for child cells
        n_max_rounds: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Attempt to divide cells

        Arguments:
            parent_idxs: (n,) cell index tensor of n cells that want to divide
            child_idxs: (n,) cell index tensor for new cells to assume if division is successful
            n_max_rounds: how many rounds to resolve possible conflicts among cells

        Returns:
            Tuple `parent_idxs, child_idxs`
            - `parent_idxs`: (k,) cell index tensor of k successfully divided cells
            - `child_idxs`: (k,) cell index tensor of respective descendant cells

        A cell can only divide if it finds a free voxel in its Moore neighbourhood.
        Position `i` of `parent_idxs` relates to position `i` of `child_indxs` in both inputs and outputs.
        _I.e._ if `parent_idxs=[1, 2]` and `child_idxs=[3, 4]` is provided but only cell `2` can divide,
        the returned values will be `parent_idxs=[2]` and `child_idxs=[4]`.
        """
        # For each cell:
        # - get free Moore neighborhood
        # - randomly choose a target voxel in neighborhood
        # Gather all chosen voxels (let one cell win in case of collisions)
        # Commit winners
        # Repeat with left overs

        # snapshot of current positions, read-only below
        pos = self.get_cell_positions()

        # generate shuffled indices for cell_idxs
        n = parent_idxs.numel()
        remaining = self.idxrandperm(n)

        # target voxel indices (flattened) of successful cells
        success = self.idxfull(n, value=-1)

        for _ in range(n_max_rounds):
            r = remaining.numel()  # number of cells still trying to act
            if r == 0:
                break

            # cell idxs of remaining cells
            idxs_pend = parent_idxs[remaining]

            # 8 neighbourhoods of each remaining cells (r, 8, 2)
            nbs = self._nbh.get_neighborhoods(coords=pos[idxs_pend])

            # generate target neighbourhood and choose who gets it (r,)
            target_nbs, winners = self._choose_neighbourhood(nbs=nbs, r=r)

            # commit winners
            w_idxs = remaining[winners]  # indices for idx
            w_tgts = target_nbs[winners]  # indices for target voxel

            # occupy voxel (also affects 2d cells)
            self.cell_map_flat[w_tgts] = child_idxs[w_idxs]

            success[w_idxs] = w_tgts  # add target voxel index to result

            # cells with no free neighbor stay too, migrate may vacate nearby
            remaining = remaining[~winners]  # (r,)

        ok = success >= 0
        return parent_idxs[ok], child_idxs[ok]

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
        Load a saved state.
        The state had to be saved with [save_state()][.save_state] previously.

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
        self, nbs: torch.Tensor, r: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N2 = self.map_size * self.map_size
        flat_nbs = self._flatten_positions(nbs)  # (r, 8)

        # find free indexes on flat map for each neighborhood
        free_nbs = self.cell_map_flat[flat_nbs] < 0  # (r, 8)
        can_act_mask = free_nbs.any(1)  # (r,)

        # propose a target voxel on each flattened neighborhood
        nb_scores = self.frand(free_nbs.shape)  # (r, 8)
        nb_scores = nb_scores.masked_fill(~free_nbs, -1.0)  # (r, 8)
        nb_choices = nb_scores.argmax(1, keepdim=True).int()  # (r, 1)
        target_nbs = flat_nbs.gather(1, nb_choices).int().squeeze(1)  # (r,)

        # generate unique random priorities
        prios = self.idxrandperm(r)

        # if same target, higher priority wins
        best = self.idxfull(N2, value=-1)
        best.scatter_reduce_(
            src=prios[can_act_mask],  # reduce from src
            index=target_nbs[can_act_mask],  # to indices index
            reduce="amax",  # via reduce argument (on src)
            dim=0,  # for dimension dim
        )  # (r,)
        winners = can_act_mask & (best[target_nbs] == prios)  # (r,)

        return target_nbs, winners  # int (r,), bool (r,)

    def _flatten_positions(self, xy: torch.Tensor) -> torch.Tensor:
        return xy[..., 0] * self.map_size + xy[..., 1]

    def _unflatten_positions(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack((z // self.map_size, z % self.map_size), dim=1)

    def _get_coord_map(self) -> torch.Tensor:
        L = self.map_size
        xs, ys = torch.meshgrid(self.idxarange(L), self.idxarange(L), indexing="ij")
        return torch.stack((xs, ys), dim=-1).view(-1, 2)  # (N², 2)
