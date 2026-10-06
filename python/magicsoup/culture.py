import random
from pathlib import Path
from typing import TypedDict

import torch

from magicsoup import rs
from magicsoup.chemistry import Chemistry
from magicsoup.util import TensorClass


class CultureKwargs(TypedDict, total=False):
    size: int


class Culture(TensorClass):

    def __init__(
        self,
        chemistry: Chemistry,
        size: int = 128,
        device: str = "cpu",
        ftype: torch.dtype = torch.float32,
        itype: torch.dtype = torch.int8,
    ):
        super().__init__(device=device, ftype=ftype, itype=itype)

        molecules = chemistry.molecules
        self.size = size
        self.pixels = size**2
        self.cells: torch.Tensor = self.izeros(size, size).bool()
        self.molecules: torch.Tensor = self.fzeros(len(molecules), size, size)
        self._diffusion_funs: list[torch.nn.Conv2d] = [
            self._get_diffuse(mol_diff_rate=m.diffusivity) for m in molecules
        ]

        # setup rust class
        self._setup_rs()

    @torch.no_grad()
    def diffuse_molecules(self):
        n_pxls = self.size**2
        for mol_i, diffuse in enumerate(self._diffusion_funs):
            total_before = self.molecules[mol_i].sum()
            before = self.molecules[mol_i].unsqueeze(0).unsqueeze(1)
            after = diffuse(before)
            self.molecules[mol_i] = torch.squeeze(after, 0).squeeze(0)
            total_after = self.molecules[mol_i].sum()

            # attempt to fix the problem that convolusion makes a small amount of
            # molecules appear or disappear (I think because floating point)
            self.molecules[mol_i] += (total_before - total_after) / n_pxls
            self.molecules[mol_i] = self.molecules[mol_i].clamp(0.0)

    def move_cells(
        self, cell_idxs: list[int], positions: list[tuple[int, int]]
    ) -> tuple[list[tuple[int, int]], list[int]]:
        return self.rs.move_cells(cell_idxs=cell_idxs, positions=positions)

    def free_moores_nghbhd(
        self, x: int, y: int, positions: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        return self.rs.free_moores_nghbhd(x, y, positions)

    def get_neighbors(
        self, from_idxs: list[int], to_idxs: list[int], positions: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        return self.rs.get_neighbors(from_idxs, to_idxs, positions)

    def divide_cells_if_possible(
        self, cell_idxs: list[int], positions: list[tuple[int, int]], n_cells: int
    ) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        """Returns divided cell idxs, their child idx, their child positions"""
        return self.rs.divide_cells_if_possible(cell_idxs, positions, n_cells)

    def save_state(self, statedir: Path) -> None:
        statedir = statedir / type(self).__name__
        statedir.mkdir(parents=True, exist_ok=True)
        torch.save(self.cells, statedir / "cells.pt")
        torch.save(self.molecules, statedir / "molecules.pt")

    def load_state(self, statedir: Path) -> None:
        statedir = statedir / type(self).__name__
        self.cells[:] = torch.load(statedir / "cells.pt", map_location=self.device)
        self.molecules[:] = torch.load(
            statedir / "molecules.pt", map_location=self.device
        )

    def find_free_random_positions(self, n: int) -> torch.Tensor:
        # available spots on map
        pxls = torch.nonzero(~self.cells).int()
        n_pxls = pxls.size(0)
        n = min(n, n_pxls)

        # place cells on map
        idxs = random.sample(range(n_pxls), k=n)
        chosen = pxls[idxs]
        return chosen

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

    def _setup_rs(self) -> None:
        self.rs = rs.Culture(size=self.size)
