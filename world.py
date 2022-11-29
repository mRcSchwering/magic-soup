from typing import Optional
from itertools import product
import random
import torch


class World:
    """
    World definition with `map` that stores molecule/signal concentrations and cell
    positions, and methods `diffuse` and `degrade` to increment world by one time step.
    """

    def __init__(
        self,
        size=128,
        n_molecules=4,
        mol_degrad=0.9,
        device="cpu",
        dtype=torch.float,
        mol_diff_kernel: Optional[torch.Tensor] = None,
        map_init: Optional[str] = None,
        trunc_n_decs=4,
    ):
        self.size = size
        self.n_molecules = n_molecules
        self.mol_degrad = mol_degrad
        self.torch_kwargs = {"dtype": dtype, "device": device}
        self.trunc_n_decs = trunc_n_decs
        self.molecule_map = self._init_map(map_init=map_init)
        self.cell_map = self._init_cell_map()
        self.conv113 = self._init_conv(kernel=mol_diff_kernel)

    def position_cells(self, n_cells: int) -> list[tuple[int, int]]:
        """
        Randomly place `n_cells` on `world.cell_map` and return
        their positions.
        """
        perms = product(range(self.size), range(self.size))
        positions = list(random.sample(list(perms), k=n_cells))
        for pos in positions:
            self.cell_map[pos] = True
        return positions

    def _init_cell_map(self) -> torch.Tensor:
        kwargs = {**self.torch_kwargs, "dtype": torch.bool}
        return torch.zeros(self.size, self.size, **kwargs)  # type: ignore

    def _init_map(self, map_init: Optional[str] = None) -> torch.Tensor:
        if map_init is None or map_init == "zeros":
            return torch.zeros(
                self.n_molecules, 1, self.size, self.size, **self.torch_kwargs
            )
        if map_init == "randn":
            return torch.randn(
                self.n_molecules, 1, self.size, self.size, **self.torch_kwargs
            ).abs()
        raise ValueError(f"Didnt recognize map_init={map_init}")

    def _init_conv(self, kernel: Optional[torch.Tensor] = None) -> torch.nn.Conv2d:
        if kernel is None:
            # fmt: off
            kernel = torch.tensor([[[
                [0.1, 0.1, 0.1],
                [0.1, 0.2, 0.1],
                [0.1, 0.1, 0.1],
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

    @torch.no_grad()
    def diffuse_molecules(self):
        self.molecule_map = self.truncate(self.conv113(self.molecule_map))

    def degrade_molecules(self):
        self.molecule_map = self.truncate(self.molecule_map * self.mol_degrad)

    def truncate(self, tens: torch.Tensor) -> torch.Tensor:
        return torch.round(tens * 10 ** self.trunc_n_decs) / (10 ** self.trunc_n_decs)
