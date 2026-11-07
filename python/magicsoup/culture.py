from pathlib import Path
from typing import TypedDict

import torch

from magicsoup.chemistry import Chemistry
from magicsoup.core.tensorclass import TensorClass, TensorKwargs

type T = torch.Tensor


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
            coords: (k, 2) index tensor for positions on N x N grid
            half: whether to use only half Moore neighborhood (e.g. to get unique paris)

        Returns:
            (k, 8, 2) or index tensor of all Moore neighborhoods for each provided position
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
        N = map_size
        self.N2: int = N**2
        self.map_size: int = N

        m = len(chemistry.molecules)
        self.n_molecules: int = m

        # all possible voxel positions (row, col)
        self.all_positions: T = self._get_all_positions()  # (N², 2)

        # all N x N voxels with cell idxs, -1 for empty voxels (N, N)
        self.cell_map: T = self.idxfull(N, N, value=-1)

        # flat representation of cell idxs in N x N map (shares memory)
        self.cell_map_flat: T = self.cell_map.view(-1)  # (N²,)

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
        pos = self.get_cell_positions()  # (N², 2) with (-1, -1) for empty
        valid = pos[:, 0] >= 0  # rows which are not empty

        # look up half cell neighborhoods to only get unique pairs (N², 4, 2)
        nbs = self._nbh.get_neighborhoods(coords=pos, half=True)

        # represent neighborhoods as flat indices of the N x N map
        nbh_idxs = self._2d_to_1d(nbs)  # (N², 4)

        # select neighbor voxels on flat cell idx map
        nb_voxels = self.cell_map_flat[nbh_idxs]  # (N², 4)

        # create mask of k cells that do have a neighbour
        mask = valid[:, None] & (nb_voxels >= 0)  # (N², 4)

        # generate (N², 4) source cell indices for each neighbor voxel
        src = self.idxarange(self.N2)[:, None].expand_as(nb_voxels)

        # return only valid pairs of cell indices (k, 2)
        return torch.stack((src[mask], nb_voxels[mask]), dim=1)  # (k, 2)

    def get_cell_positions(self) -> T:
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
        flat = self.cell_map_flat

        # map empty voxels into dummy row N² + 1
        pos = self.idxfull(self.N2 + 1, 2, value=-1)  # (N²+1, 2)
        rows = torch.where(flat >= 0, flat, self.N2)  # (N²+1, 2)
        pos[rows] = self.all_positions
        return pos[: self.N2]  # (N², 2) remove dummy row

    def get_cell_positions_flat(self) -> T:
        """
        Get current positions of all cells on cell map as flat representation.

        Returns:
            (N²,) position index tensor encoding all cells with 1D positions

        All possible 1D positions on the flat map are presented in the returned tensor.
        Each cell is contained with its 1D position of its cell index.
        _E.g._ if cell 5 is at (372,), then the tensor at index 5 contains 372.
        Unused cell indices contain (-1,).
        _E.g._ if there is no cell with index 4, then the tensor at index 4 contains -1.
        """
        pos = self.get_cell_positions()
        return self._2d_to_1d(pos)

    def get_free_positions(self, n: int) -> T:
        """
        Get `n` available random positions on cell map.
        Returns fewer if fewer are avilable.

        Arguments:
            n: Desired number of random positions

        Returns:
            (k, 2) position index tensor of available positions with `k = min(n, number of free voxels)`
        """
        free = self.cell_map_flat < 0  # idxs of free voxels
        score = self.frand(free.shape).masked_fill(~free, -1.0)  # score free foxels
        sel = score.topk(n).indices  # n distinct flat voxel idxs (prefer free ones)
        sel = sel[free[sel]]  # drop occupied ones if fewer than n were free
        return self._1d_to_2d(sel)  # convert into row-col (k, 2)

    def occupy_cell_map_randomly(self, mask: T) -> None:
        flat = self.cell_map_flat
        n2 = self.N2
        c = mask.shape[0]
        cell_idxs = self.idxarange(c)  # (c,)

        # rank now-free voxels in random order; assign the r-th selected cell the
        # r-th free voxel (there are always >= as many free voxels as selected
        # cells, since each selected cell just vacated one)
        free = flat < 0  # (N²,)
        scores = self.frand(n2).masked_fill(~free, 2.0)  # free in [0, 1), occupied 2
        order = scores.argsort()  # (N²,) free voxels first, in random order
        sel_rank = mask.cumsum(0, dtype=self.idxtype) - 1  # (c,)
        target = order[sel_rank.clamp(min=0)].int()  # (c,) target voxel per cell

        # place the selected cells on their target voxels
        padded = torch.cat([flat, self.idxtensor([-1])])
        place = torch.where(mask, target, self.idxtensor(n2))  # (c,)
        padded.scatter_(0, place, cell_idxs)
        flat[:] = padded[:n2]

    def occupy_cell_map(self, mask: T, flat_voxel: T, idx: T) -> None:
        """
        Arguments:
            - `mask`: (c,) bool tensor of provided slots to place
            - `flat_voxel`: (c,) index tensor of the claimed flat voxel idx per slot
            - `idx`: (c,) index tensor of cell index to place on voxel per slot
        """
        flat = self.cell_map_flat
        n2 = self.N2

        # flat map with extra -1 voxel at the end
        padded = torch.cat([flat, self.idxtensor([-1])])

        # map unused to slots to -1 voxel
        place = torch.where(mask, flat_voxel, self.idxtensor(n2))  # (c,)

        # TODO: necessary? the scatter would place ~mask to -1 anyway?!
        cell_idxs = torch.where(mask, idx, self.idxtensor(0))  # (c,)

        # place cell idxs on flat cell map
        padded.scatter_(dim=0, index=place, src=cell_idxs)
        flat[:] = padded[:n2]  # drop -1 voxel again

    def undirected_cell_migration(self, mask: T, n_rounds: int = 3) -> T:
        """
        Let cells migrate randomly around their Moore neighbourhood

        Arguments:
            mask: (c,) bool tensor, which cell slots attempt to migrate
            n_rounds: how many rounds to resolve possible conflicts among cells

        Returns:
            (c,) bool tensor of cells that successfully migrated

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
        pos = self.get_cell_positions()  # (N², 2) row = cell idx -> position
        c = mask.shape[0]
        coords = pos[:c]  # (c, 2) fixed snapshot of each cell's position
        old_flat = self._2d_to_1d(coords)  # (c,) current voxel
        cell_idxs = self.idxarange(c)  # (c,)
        dump = self.N2  # throwaway voxel slot for non-winners
        still = mask.clone()  # (c,) cells still trying

        for _ in range(n_rounds):
            target, winners = self._choose_neighbourhood(coords=coords, acting=still)

            # commit winners: vacate old voxel, occupy target (losers -> dump slot)
            padded = torch.cat([self.cell_map_flat, self.idxtensor([-1])])  # (N²+1,)
            vacate = torch.where(winners, old_flat, self.idxtensor(dump))  # (c,)
            place = torch.where(winners, target, self.idxtensor(dump))  # (c,)
            padded.scatter_(0, vacate, self.idxfull(c, value=-1))
            padded.scatter_(0, place, cell_idxs)
            self.cell_map_flat[:] = padded[: self.N2]

            still &= ~winners

        return mask & ~still  # (c,) successfully migrated

    def cell_division(self, mask: T, n_rounds: int = 3) -> tuple[T, T]:
        """
        Resolve free-voxel contention for dividing cells.

        Arguments:
            mask: (c,) bool tensor, which cell slots attempt to divide
            n_rounds: how many rounds to resolve possible conflicts among cells

        Returns:
            Tuple `(success, won_voxel)`
            - `success`: (c,) bool tensor of parents that won a free neighbour voxel
            - `won_voxel`: (c,) int tensor of the claimed flat voxel idx per parent (-1 if none)

        A cell can only divide if it finds a free voxel in its Moore neighbourhood.
        Each won voxel is reserved on the map with the parent's idx as a placeholder.
        The caller must overwrite those reservations with the real child idxs.
        """
        # For each parent still trying: get a free Moore neighbour voxel and let
        # one parent win each contested voxel. Reserve won voxels (so later rounds
        # see them taken), then repeat with the left-overs. Everything stays
        # full-width (c,) so no boolean-mask gather or nonzero is used.
        n2 = self.N2
        c = mask.shape[0]
        coords = self.get_cell_positions()[:c]  # (c, 2) row = cell idx -> position
        cell_idxs = self.idxarange(c)  # (c,)
        still = mask.clone()  # (c,)
        won_voxel = self.idxfull(c, value=-1)  # (c,) parent -> claimed voxel

        for _ in range(n_rounds):
            target, winners = self._choose_neighbourhood(coords=coords, acting=still)

            # reserve won voxels with the parent idx so later rounds see them taken
            padded = torch.cat([self.cell_map_flat, self.idxtensor([-1])])  # (N²+1,)
            place = torch.where(winners, target, self.idxtensor(n2))  # (c,)
            padded.scatter_(0, place, cell_idxs)
            self.cell_map_flat[:] = padded[: self.N2]

            won_voxel = torch.where(winners, target, won_voxel)
            still &= ~winners

        return (won_voxel >= 0), won_voxel  # bool (c,), idx (c,)

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

        # 8 row-col neighbourhoods per cell, as flat idxs of the N x N map
        nbs = self._nbh.get_neighborhoods(coords=coords)  # (c, 8, 2)
        flat_nbs = self._2d_to_1d(nbs)  # (c, 8)

        # find free voxels in each neighbourhood
        free_nbs = self.cell_map_flat[flat_nbs] < 0  # (c, 8)
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

    def _2d_to_1d(self, rc: T) -> T:
        # represent (n, 2) positions on N x N map as (n,) flat indices of the map
        has_cell = rc[..., 0] > 0
        return torch.where(has_cell, rc[..., 0] * self.map_size, 0) + rc[..., 1]

    def _1d_to_2d(self, z: T) -> T:
        # represent flat indices of N x N map as (n, 2) positions on the map
        return torch.stack((z // self.map_size, z % self.map_size), dim=1)

    def _get_all_positions(self) -> T:
        L = self.map_size
        xs, ys = torch.meshgrid(self.idxarange(L), self.idxarange(L), indexing="ij")
        return torch.stack((xs, ys), dim=-1).view(-1, 2)  # (N², 2)
