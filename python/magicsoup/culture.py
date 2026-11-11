from pathlib import Path
from typing import TypedDict

import torch

from magicsoup.chemistry import Chemistry
from magicsoup.core.tensorclass import T, TensorClass, TensorKwargs


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

    def get_neighborhoods(self, coords: T, half: bool = False) -> T:
        """
        Get all Moore neighborhoods for provided coordinates

        Arguments:
            coords: (k, 2) index tensor for (row, column) coordinates on N x N grid
            half: whether to use only half Moore neighborhood (e.g. to get unique paris)

        Returns:
            (k, 8, 2) index tensor of all Moore neighborhoods for each provided coordinate
                ((k, 4, 2) if `half`)
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
        map_size: Number of voxels along one side of the square map
        tensor_kwargs: Kwargs passed to [TensorClass][magicsoup.util.TensorClass]

    The cell culture is a periodic (torus wrap) 2D grid of voxels represented by tensors.
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

    N x N maps (`cell_map` and `molecule_map`) can be represented 2D with rows and columns
    to identify items. They can also be represented 1D with flat `row_i * N + col_i` indices.
    To highlight which representation a method requires or returns, the 2D representation is
    refered to as *coordinates* while the 1D representation is refered to as *voxels*.
    """

    def __init__(
        self,
        chemistry: Chemistry,
        map_size: int = 128,
        tensor_kwargs: TensorKwargs | None = None,
    ):
        super().__init__(**(tensor_kwargs or {}))
        N = map_size
        self.N2: int = N**2
        self.map_size: int = N

        m = len(chemistry.molecules)
        self.n_molecules: int = m

        # all possible voxel positions (row, col)
        self.all_positions: T = self._get_all_positions()  # (N², 2)

        # all N x N voxels with cell idxs, -1 for empty voxels (N, N)
        self.cell_map: T = self.idxfull(N, N, value=-1)

        # all N x N voxels with molecule concentrations (m, N, N)
        self.molecule_map: T = self.fzeros(m, N, N)

        # manage Moore Neighborhoods
        self._nbh = MooreNeighborhood(N=N, tensor_kwargs=tensor_kwargs)

    def get_cell_neighbor_pairs(self) -> T:
        """
        Get all unique pairs of neighbouring cells.

        Returns:
            (k, 2) cell index tensor of all k unique cell neighbors

        Cells are neighbors if they are within each others Moore neighbourhood.
        Note: With map size smaller 3 duplicate neighbors can be returned.
        """
        cell_coords = self.get_cell_coords()  # (N², 2) with (-1, -1) for empty
        valid = cell_coords[:, 0] >= 0  # rows which are not empty
        flat_cell_map = self.cell_map.view(-1)  # (N²,) view

        # look up half cell neighborhoods to only get unique pairs (N², 4, 2)
        nbs = self._nbh.get_neighborhoods(coords=cell_coords, half=True)

        # represent neighborhoods as flat indices of the N x N map
        nbh_idxs = self.coords_to_voxels(nbs)  # (N², 4)

        # select neighbor voxels on flat cell idx map
        nb_voxels = flat_cell_map[nbh_idxs]  # (N², 4)

        # create mask of k cells that do have a neighbour
        mask = valid[:, None] & (nb_voxels >= 0)  # (N², 4)

        # generate (N², 4) source cell indices for each neighbor voxel
        src = self.idxarange(self.N2)[:, None].expand_as(nb_voxels)

        # return only valid pairs of cell indices (k, 2)
        return torch.stack((src[mask], nb_voxels[mask]), dim=1)  # (k, 2)

    def get_cell_coords(self) -> T:
        """
        Get current coordinates of all cells on cell map.

        Returns:
            (N², 2) index coordinates tensor encoding all cells with row and column (<= N)

        All possible N² coordinates are presented in the returned tensor.
        Each cell is contained with its row and column indexes at the position of its cell index.
        _E.g._ if cell 5 is at (3, 0), then the tensor at index 5 contains values 3 and 0.
        Unused cell indices contain (-1, -1).
        _E.g._ if there is no cell with index 4, then the tensor at index 4 contains values -1 and -1.
        """
        # build (N², 2) cell x-y cell positions from current map
        # row is cell index, empty rows are (-1, -1)
        flat_cell_map = self.cell_map.view(-1)  # (N²,) view

        # map empty voxels into dummy row N² + 1
        pos = self.idxfull(self.N2 + 1, 2, value=-1)  # (N²+1, 2)
        rows = torch.where(flat_cell_map >= 0, flat_cell_map, self.N2)  # (N²+1, 2)
        pos[rows] = self.all_positions
        return pos[: self.N2]  # (N², 2) remove dummy row

    def get_random_free_voxels(self, mask: T | None = None) -> T:
        """
        Get unoccupied random voxels

        Arguments:
            mask: (c,) bool tensor of slots to get free voxels for;
                None for all free voxels in random order (c <= N²)

        Returns:
            (c,) index tensor assigning each True in `mask` a free voxel
                (<= N²) on the flat cell map. -1 otherwise.

        Returned indices are -1 if `mask` was False for the slot or if
        no more voxels were available on the cell map.
        """
        flat_cell_map = self.cell_map.view(-1)  # (N²,) view
        n2 = self.N2
        if mask is None:
            mask = self.idxfull(n2, value=1).bool()

        # rank now-free voxels in random order
        c = mask.shape[0]
        free = flat_cell_map < 0  # (N²,)
        scores = self.frand(n2).masked_fill(~free, float("inf"))  # (N²,) occupied last

        # the c smallest keys are a uniform random sample of free slots (no replacement)
        vals, idxs = torch.topk(scores, k=c, largest=False)  # (c,)
        is_free = torch.isfinite(vals)  # (c,)

        # assign the i-th selected cell the i-th free voxel
        rank = (mask.cumsum(0, dtype=self.idxtype) - 1).clamp(0)  # (c,)
        select = idxs[rank]  # gather (c,) idxs
        valid = is_free[rank]  # gather (c,) flags
        voxels = torch.where(mask, select, -1)  # hide unrequested voxels
        voxels[~valid] = -1  # hide occupied voxels
        return voxels

    def update_cell_map(self, idx_map: T) -> None:
        """
        Update indices of existing cells on the cell map

        Arguments:
            idx_map: (c,) index tensor mapping each old cell index to its new index,
                with -1 for dropped slots (c <= N²)
        """
        flat_cell_map = self.cell_map.view(-1)  # (N²,) view
        tmp_map = idx_map[flat_cell_map.clamp(0)]
        flat_cell_map[:] = torch.where(flat_cell_map >= 0, tmp_map, flat_cell_map)

    def place_cells(self, voxels: T) -> None:
        """
        Place cells on the cell map

        Arguments:
            `voxels`: (c,) index tensor (c <= N²) containing `voxel[i]` for cell `i`, -1 for unused.

        This method doesn't guarantee that a cell or voxel will be in use twice.
        """
        flat_cell_map = self.cell_map.view(-1)  # (N²,) view
        flat_cell_map[:] = self.idxmap_scatter(tgt=flat_cell_map, src=voxels)

    def migrate_cells_randomly(self, mask: T, n_rounds: int = 3) -> T:
        """
        Let cells migrate randomly around their Moore neighbourhood

        Arguments:
            mask: (c,) bool tensor, which cells attempt to migrate (c <= N²)
            n_rounds: how many rounds to resolve possible conflicts among cells

        Returns:
            (c,) bool tensor of cells that successfully migrated (c <= N²)

        Updates the cell map and returns successfully migrated cells.
        A cell can only migrate if it finds a free voxel in its Moore neighbourhood.
        """
        # For each cell still trying:
        # - get free Moore neighborhood, randomly choose a target voxel
        # - let one cell win each contested voxel
        # Commit winners (vacate old voxel, occupy target), repeat with left-overs
        #
        # Everything is kept full-width (c,) so no boolean-mask gather
        # or nonzero is used and no host synchronization happens.
        c = mask.shape[0]
        dump = self.idxtensor([-1])  # throwaway voxel slot for non-winners
        i_dump = self.N2  # index of throwaway voxel
        flat_cell_map = self.cell_map.view(-1)  # (N²,) view
        cell_coords = self.get_cell_coords()[:c]  # (c, 2) row = cell idx -> position
        old_flat = self.coords_to_voxels(cell_coords)  # (c,) current voxel
        cell_idxs = self.idxarange(c)  # (c,)
        still = mask.clone()  # (c,) cells still trying

        for _ in range(n_rounds):
            tgt, win = self._choose_neighbourhood(coords=cell_coords, acting=still)

            # commit winners: vacate old voxel, occupy target (losers -> dump slot)
            padded = torch.cat([flat_cell_map, dump])  # (N²+1,)
            vacate = torch.where(win, old_flat, i_dump)  # (c,)
            place = torch.where(win, tgt, i_dump)  # (c,)
            padded[vacate] = -1
            padded[place] = cell_idxs
            flat_cell_map[:] = padded[: self.N2]

            still &= ~win

        return mask & ~still  # (c,) successfully migrated

    def divide_cells(self, mask: T, n_rounds: int = 3) -> tuple[T, T]:
        """
        Resolve free-voxel contention for dividing cells.

        Arguments:
            mask: (c,) bool tensor, which cells attempt to divide (c <= N²)
            n_rounds: how many rounds to resolve possible conflicts among cells

        Returns:
            tuple (success, won_voxel):
            - `success`: (c,) bool tensor of parents that won a free neighbour voxel (c <= N²)
            - `won_voxel`: (c,) int tensor of the claimed voxel (c <= N²) per parent (-1 if none)

        A cell can only divide if it finds a free voxel in its Moore neighbourhood.
        Each won voxel is reserved on the map with the parent's idx as a placeholder.
        The caller must overwrite those reservations with the real child idxs.
        """
        # For each parent still trying: get a free Moore neighbour voxel and let
        # one parent win each contested voxel. Reserve won voxels (so later rounds
        # see them taken), then repeat with the left-overs. Everything stays
        # full-width (c,) so no boolean-mask gather or nonzero is used.
        c = mask.shape[0]
        dump = self.idxtensor([-1])  # throwaway voxel slot for non-winners
        i_dump = self.N2  # index of throwaway voxel
        flat_cell_map = self.cell_map.view(-1)  # (N²,) view
        cell_coords = self.get_cell_coords()[:c]  # (c, 2) row = cell idx -> position
        cell_idxs = self.idxarange(c)  # (c,)
        still = mask.clone()  # (c,)
        won_voxel = self.idxfull(c, value=-1)  # (c,) parent -> claimed voxel

        for _ in range(n_rounds):
            tgt, win = self._choose_neighbourhood(coords=cell_coords, acting=still)

            # reserve won voxels with the parent idx so later rounds see them taken
            padded = torch.cat([flat_cell_map, dump])  # (N²+1,)
            place = torch.where(win, tgt, i_dump)  # (c,)
            padded[place] = cell_idxs
            flat_cell_map[:] = padded[:i_dump]

            won_voxel = torch.where(win, tgt, won_voxel)
            still &= ~win

        return (won_voxel >= 0), won_voxel  # bool (c,), idx (c,)

    def voxels_to_coords(self, voxels: T) -> T:
        """Convert (N²,) voxel indices to (N, N) coordinates"""
        n = self.map_size
        return torch.stack((voxels // n, voxels % n), dim=1)

    def coords_to_voxels(self, coords: T) -> T:
        """Convert (N, N) voxel indices to (N²)"""
        has_cell = coords[..., 0] > 0
        n = self.map_size
        return torch.where(has_cell, coords[..., 0] * n, 0) + coords[..., 1]

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
        self.cell_map[:] = torch.load(statedir / "cell_map.pt", map_location=self.device)
        self.molecule_map[:] = torch.load(statedir / "molecule_map.pt", map_location=self.device)

    def _choose_neighbourhood(self, coords: T, acting: T) -> tuple[T, T]:
        # coords: (c, 2) current row-col position of each cell slot
        # acting: (c,) bool, which cell slots attempt to claim a free neighbour voxel
        #
        # Returns (target, winners): (c,) flat target voxel idx per cell and (c,)
        # bool of which cells won their contested target.
        #
        # Operates over all c cell slots; non-acting cells are routed to a
        # throwaway voxel (index N2) so no boolean-mask gather / nonzero is needed
        # and the whole method performs no host synchronization.
        n2 = self.N2
        flat_cell_map = self.cell_map.view(-1)  # (N²,) view

        # 8 row-col neighbourhoods per cell, as flat idxs of the N x N map
        nbs = self._nbh.get_neighborhoods(coords=coords)  # (c, 8, 2)
        flat_nbs = self.coords_to_voxels(nbs)  # (c, 8)

        # find free voxels in each neighbourhood
        free_nbs = flat_cell_map[flat_nbs] < 0  # (c, 8)
        can_act = free_nbs.any(1) & acting  # (c,)

        # propose a random free target voxel per cell
        nb_scores = self.frand(free_nbs.shape).masked_fill(~free_nbs, -1.0)  # (c, 8)
        nb_choices = nb_scores.argmax(1, keepdim=True).int()  # (c, 1)
        target = flat_nbs.gather(1, nb_choices).int().squeeze(1)  # (c,)

        # unique random priorities; highest priority wins a contested voxel
        prios = self.idxrandperm(coords.shape[0])  # (c,)
        best = self.idxfull(n2 + 1, value=-1)  # extra dump slot at index n2
        scatter_idx = torch.where(can_act, target, self.idxtensor(n2))  # (c,)
        best.scatter_reduce_(0, scatter_idx, prios, reduce="amax")  # (n2 + 1,)
        winners = can_act & (best[target] == prios)  # (c,)

        return target, winners  # idx (c,), bool (c,)

    def _get_all_positions(self) -> T:
        L = self.map_size
        xs, ys = torch.meshgrid(self.idxarange(L), self.idxarange(L), indexing="ij")
        return torch.stack((xs, ys), dim=-1).view(-1, 2)  # (N², 2)
