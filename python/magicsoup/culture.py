from pathlib import Path
from typing import TypedDict

import torch

from magicsoup.chemistry import Chemistry
from magicsoup.core.tensorclass import TensorClass, TensorKwargs


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
        self.N2 = N**2
        self.map_size = N

        m = len(chemistry.molecules)
        self.n_molecules = m

        # all possible voxel positions (row, col)
        self.all_positions = self._get_all_positions()  # (N², 2)

        # all N x N voxels with cell idxs, -1 for empty voxels (N, N)
        self.cell_map: torch.Tensor = self.idxfull(N, N, value=-1)

        # flat representation of cell idxs in N x N map (shares memory)
        self.cell_map_flat = self.cell_map.view(-1)  # (N²,)

        # all N x N voxels with molecule concentrations (m, N, N)
        self.molecule_map: torch.Tensor = self.fzeros(m, N, N)

        # manage Moore Neighborhoods
        self._nbh = MooreNeighborhood(N=N, tensor_kwargs=tensor_kwargs)

    def get_cell_neighbor_pairs(self) -> torch.Tensor:
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
        nbh_idxs = self._positions_to_idxs(nbs)  # (N², 4)

        # select neighbor voxels on flat cell idx map
        nb_voxels = self.cell_map_flat[nbh_idxs]  # (N², 4)

        # create mask of k cells that do have a neighbour
        mask = valid[:, None] & (nb_voxels >= 0)  # (N², 4)

        # generate (N², 4) source cell indices for each neighbor voxel
        src = self.idxarange(self.N2)[:, None].expand_as(nb_voxels)

        # return only valid pairs of cell indices (k, 2)
        return torch.stack((src[mask], nb_voxels[mask]), dim=1)  # (k, 2)

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
        flat = self.cell_map_flat

        # map empty voxels into dummy row N² + 1
        pos = self.idxfull(self.N2 + 1, 2, value=-1)  # (N²+1, 2)
        rows = torch.where(flat >= 0, flat, self.N2)  # (N²+1, 2)
        pos[rows] = self.all_positions
        return pos[: self.N2]  # (N², 2) remove dummy row

    def get_free_positions(self, n: int) -> torch.Tensor:
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
        return self._idxs_to_positions(sel)  # convert into row-col (k, 2)

    def undirected_cell_migration(
        self,
        mask: torch.Tensor,  # bool (c,) cells that want to migrate
        n_max_rounds: int = 3,
    ) -> torch.Tensor:
        """
        Let cells migrate randomly around their Moore neighbourhood

        Arguments:
            mask: (c,) bool tensor, which cell slots attempt to migrate
            n_max_rounds: how many rounds to resolve possible conflicts among cells

        Returns:
            (c,) bool tensor of cells that successfully migrated

        A cell can only migrate if it finds a free voxel in its Moore neighbourhood.
        Operates over all cell slots using masks and performs no host synchronization.
        """
        # For each cell still trying:
        # - get free Moore neighborhood, randomly choose a target voxel
        # - let one cell win each contested voxel
        # Commit winners (vacate old voxel, occupy target), repeat with left-overs
        #
        # Everything is kept full-width (c,): winners commit via scatter and
        # non-winners are routed to a throwaway voxel, so no boolean-mask gather
        # or nonzero is used and no host synchronization happens.

        pos = self.get_cell_positions()  # (N², 2) row = cell idx -> position
        c = mask.shape[0]
        coords = pos[:c]  # (c, 2) fixed snapshot of each cell's position
        old_flat = self._positions_to_idxs(coords)  # (c,) current voxel (unused if dead)
        cell_ids = self.idxarange(c)  # (c,)
        dump = self.N2  # throwaway voxel slot for non-winners
        still = mask.clone()  # (c,) cells still trying

        for _ in range(n_max_rounds):
            target, winners = self._choose_neighbourhood(coords=coords, acting=still)

            # commit winners: vacate old voxel, occupy target (losers -> dump slot)
            padded = torch.cat([self.cell_map_flat, self.idxtensor([-1])])  # (N²+1,)
            vacate = torch.where(winners, old_flat, self.idxtensor(dump))  # (c,)
            place = torch.where(winners, target, self.idxtensor(dump))  # (c,)
            padded.scatter_(0, vacate, self.idxfull(c, value=-1))
            padded.scatter_(0, place, cell_ids)
            self.cell_map_flat[:] = padded[: self.N2]

            still &= ~winners

        return mask & ~still  # (c,) successfully migrated

    def cell_division(
        self,
        mask: torch.Tensor,  # bool (c,) parents that want to divide
        n_max_rounds: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Resolve free-voxel contention for dividing cells.

        Arguments:
            mask: (c,) bool tensor, which cell slots attempt to divide
            n_max_rounds: how many rounds to resolve possible conflicts among cells

        Returns:
            Tuple `(success, won_voxel)`
            - `success`: (c,) bool tensor of parents that won a free neighbour voxel
            - `won_voxel`: (c,) int tensor of the claimed flat voxel idx per parent (-1 if none)

        A cell can only divide if it finds a free voxel in its Moore neighbourhood.
        Each won voxel is reserved on the map with the parent's id as a placeholder
        so that later rounds treat it as occupied; the caller must overwrite those
        reservations with the real child ids. Operates over all cell slots using
        masks and performs no host synchronization.
        """
        # For each parent still trying: get a free Moore neighbour voxel and let
        # one parent win each contested voxel. Reserve won voxels (so later rounds
        # see them taken), then repeat with the left-overs. Everything stays
        # full-width (c,): winners commit via scatter and non-winners are routed to
        # a throwaway voxel, so no boolean-mask gather or nonzero is used.
        pos = self.get_cell_positions()  # (N², 2) row = cell idx -> position
        c = mask.shape[0]
        coords = pos[:c]  # (c, 2) parent positions (read-only snapshot)
        cell_ids = self.idxarange(c)  # (c,)
        dump = self.N2  # throwaway voxel slot for non-winners
        still = mask.clone()  # (c,)
        won_voxel = self.idxfull(c, value=-1)  # (c,) parent -> claimed voxel

        for _ in range(n_max_rounds):
            target, winners = self._choose_neighbourhood(coords=coords, acting=still)

            # reserve won voxels with the parent id so later rounds see them taken
            padded = torch.cat([self.cell_map_flat, self.idxtensor([-1])])  # (N²+1,)
            place = torch.where(winners, target, self.idxtensor(dump))  # (c,)
            padded.scatter_(0, place, cell_ids)
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

    def _choose_neighbourhood(
        self, coords: torch.Tensor, acting: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        flat_nbs = self._positions_to_idxs(nbs)  # (c, 8)

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

    def _positions_to_idxs(self, rc: torch.Tensor) -> torch.Tensor:
        # represent (n, 2) positions on N x N map as (n,) flat indices of the map
        return rc[..., 0] * self.map_size + rc[..., 1]

    def _idxs_to_positions(self, z: torch.Tensor) -> torch.Tensor:
        # represent flat indices of N x N map as (n, 2) positions on the map
        return torch.stack((z // self.map_size, z % self.map_size), dim=1)

    def _get_all_positions(self) -> torch.Tensor:
        L = self.map_size
        xs, ys = torch.meshgrid(self.idxarange(L), self.idxarange(L), indexing="ij")
        return torch.stack((xs, ys), dim=-1).view(-1, 2)  # (N², 2)
