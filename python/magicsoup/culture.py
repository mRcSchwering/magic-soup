import random
from pathlib import Path
from typing import TypedDict

import torch

from magicsoup import rs
from magicsoup.chemistry import Chemistry
from magicsoup.util import TensorClass


class CultureKwargs(TypedDict, total=False):
    map_size: int


class Culture(TensorClass):

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
        self.cell_map: torch.Tensor = self.izeros(map_size, map_size).bool()
        self.molecule_map: torch.Tensor = self.fzeros(
            len(molecules), map_size, map_size
        )
        self._diffusion_funs: list[torch.nn.Conv2d] = [
            self._get_diffuse(mol_diff_rate=m.diffusivity) for m in molecules
        ]
        self._nb_offs = self._get_moores_neighborhood()

        # setup rust class
        self._setup_rs()

    @torch.no_grad()
    def diffuse_molecules(self):
        n_pxls = self.map_size**2
        for mol_i, diffuse in enumerate(self._diffusion_funs):
            total_before = self.molecule_map[mol_i].sum()
            before = self.molecule_map[mol_i].unsqueeze(0).unsqueeze(1)
            after = diffuse(before)
            self.molecule_map[mol_i] = torch.squeeze(after, 0).squeeze(0)
            total_after = self.molecule_map[mol_i].sum()

            # attempt to fix the problem that convolusion makes a small amount of
            # molecules appear or disappear (I think because floating point)
            self.molecule_map[mol_i] += (total_before - total_after) / n_pxls
            self.molecule_map[mol_i] = self.molecule_map[mol_i].clamp(0.0)

    def get_neighbors(
        self, from_idxs: list[int], to_idxs: list[int], positions: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        # TODO: refactor onto GPU
        return self.rs.get_neighbors(from_idxs, to_idxs, positions)

    def find_free_random_positions(self, n: int) -> torch.Tensor:
        # available spots on map
        pxls = torch.nonzero(~self.cell_map).int()
        n_pxls = pxls.size(0)
        n = min(n, n_pxls)

        # place cells on map
        idxs = random.sample(range(n_pxls), k=n)
        chosen = pxls[idxs]
        return chosen

    def divide_cells(
        self, cell_idxs: torch.Tensor, cell_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parent_idxs, child_positions = self._resolve_placements(
            cell_idxs=cell_idxs, cell_positions=cell_positions, is_migration=False
        )
        return parent_idxs, child_positions

    def migrate_cells(
        self, cell_idxs: torch.Tensor, cell_positions: torch.Tensor
    ) -> None:
        self._resolve_placements(
            cell_idxs=cell_idxs, cell_positions=cell_positions, is_migration=True
        )

    def save_state(self, statedir: Path) -> None:
        statedir = statedir / type(self).__name__
        statedir.mkdir(parents=True, exist_ok=True)
        torch.save(self.cell_map, statedir / "cell_map.pt")
        torch.save(self.molecule_map, statedir / "molecule_map.pt")

    def load_state(self, statedir: Path) -> None:
        statedir = statedir / type(self).__name__
        self.cell_map[:] = torch.load(
            statedir / "cell_map.pt", map_location=self.device
        )
        self.molecule_map[:] = torch.load(
            statedir / "molecule_map.pt", map_location=self.device
        )

    def _resolve_placements(
        self,
        cell_idxs: torch.Tensor,  # i32 (n,) indices of cells that want to act
        cell_positions: torch.Tensor,  # i32 (n,2) cell positions updated in palce for migrating cells
        is_migration: bool,  # division otherwise
        n_max_rounds: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # for cells that want to divide or migrate:
        #
        # generate a random priority ranking over cells
        # for each cell:
        # - get free Moore's neighborhood
        # - randomly choose a target pixel in neighborhood
        # gather all chosen pixels
        # - cell with higher priority wins in case of clashes
        # commit winners
        # repeat with left overs
        # - until no left overs left or some number of rounds is reached

        # flat representation of cells (shares memory)
        cell_map_flat = self.cell_map.view(-1)  # (s * s,)

        # generate shuffled indices for cell_idxs
        n_pending = cell_idxs.numel()
        pending = self.idxrandperm(n_pending)

        # target pixel indices (flattened) of successful cells
        success = self.idxfull(n_pending, value=-1)

        for _ in range(n_max_rounds):
            p = pending.numel()  # number of cells still trying to act
            if p == 0:
                break

            # cell idxs of pending cells
            idxs_pend = cell_idxs[pending]

            # 8 neighbourhoods of each pending cells
            nbs = (
                # (p, 1, 2) + (1, 8, 2) -> (p, 8, 2)
                (cell_positions[idxs_pend][:, None, :] + self._nb_offs[None])
                % self.map_size  # torus wrap
            )

            # flattened neighborhood coordinates x,y -> x * s + y
            flat_nbs = self._flatten_coordinates(nbs)  # (p, 8)

            # find free indexes on flat map for each neighborhood
            free_nbs = ~cell_map_flat[flat_nbs]  # (p, 8)
            can_act_mask = free_nbs.sum(1) > 0  # (p,)

            # propose a target pixel on each flattened neighborhood
            nb_scores = torch.rand(free_nbs.shape, device=self.device)  # (p, 8)
            nb_scores = nb_scores.masked_fill(~free_nbs, -1.0)  # (p, 8)
            nb_choices = nb_scores.argmax(1, keepdim=True).int()  # (p, 1)
            target_nbs = flat_nbs.gather(1, nb_choices).int().squeeze(1)  # (p,)

            # generate unique random priorities
            prios = self.idxrandperm(p)
            prios = torch.where(can_act_mask, prios, torch.full_like(prios, -1))

            # if same target, higher priority wins
            best = self.idxfull(self.pixels, value=-1)
            best.scatter_reduce_(
                src=prios[can_act_mask],  # reduce from src
                index=target_nbs[can_act_mask],  # to indices index
                reduce="amax",  # via reduce argument (on src)
                dim=0,  # for dimension dim
            )  # (p,)
            winners = can_act_mask & (best[target_nbs] == prios)  # (p,)

            # commit winners
            w_idxs = pending[winners]  # indices for idx
            w_tgts = target_nbs[winners]  # indices for target pixel
            cell_map_flat[w_tgts] = True  # occupy pixel (also affects 2d cells)
            success[w_idxs] = w_tgts  # add target pixel index to result

            # migration
            if is_migration:
                mcells = cell_idxs[w_idxs]  # successful cell indices
                old = cell_positions[mcells]  # their x-y coordinates
                self.cell_map[old[:, 0], old[:, 1]] = False  # unoccupy old position

                # convert target pixel back to 2d representation and update positions
                cell_positions[mcells] = self._unflatten_coordinates(w_tgts)  # (p, 2)

            # cells with no free neighbor stay too, migrate may vacate nearby
            pending = pending[~winners]  # (p,)

        # extract idx and new x-y pos of divided cells
        ok = success >= 0
        ok_idxs = cell_idxs[ok]
        new_pos = self._unflatten_coordinates(success[ok])

        return ok_idxs, new_pos

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

    def _setup_rs(self) -> None:
        self.rs = rs.Culture(size=self.map_size)
