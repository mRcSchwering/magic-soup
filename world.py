from typing import Optional
from itertools import product
import random
import torch


class World:
    """
    World definition with `map` that stores molecule/signal concentrations and cell
    positions, and methods `diffuse` and `degrade` to increment world by one time step.

    - `size` number of pixels in x and y direction for world map
    - `layers` number of layers/channels the world has (for each type of molecule)
    - `degrad` amount of degradation that happens on every step (1=no degradation)
    - `n_decs` number of decimals when rounding (this will lead to small numbers becomming zero)
    - `device` optionally use GPU
    - `dtype` optionally use larger or smaller floating point numbers (float16 only with GPU)

    > 24.11.22 around 0.38s for 1000 steps of `WorldTorch(size=128)` and a `randn` map.
    """

    def __init__(
        self,
        size=128,
        layers=4,
        degrad=0.8,
        n_decs=2,
        device="cpu",
        dtype=torch.float,
        kernel: Optional[torch.Tensor] = None,
        map_init: Optional[str] = None,
    ):
        self.size = size
        self.layers = layers
        self.n_decs = n_decs
        self.degrad = degrad
        self.device = device
        self.dtype = dtype
        self.map = self._init_map(map_init=map_init)
        self._conv = self._init_conv(kernel=kernel)
        self.cell_map = self._init_cell_map()

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
        return torch.zeros(self.size, self.size, device=self.device, dtype=torch.bool)

    def _init_map(self, map_init: Optional[str] = None) -> torch.Tensor:
        if map_init is None or map_init == "zeros":
            return torch.zeros(
                self.layers,
                1,
                self.size,
                self.size,
                device=self.device,
                dtype=self.dtype,
            )
        if map_init == "randn":
            return torch.randn(
                self.layers,
                1,
                self.size,
                self.size,
                device=self.device,
                dtype=self.dtype,
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
            dtype=self.dtype,
            device=self.device,
        )
        conv.weight = torch.nn.Parameter(kernel, requires_grad=False)
        return conv

    @torch.no_grad()
    def diffuse(self):
        """Let molecules diffuse for 1 time step"""
        self.map = self._conv(self.map)

    @torch.no_grad()
    def degrade(self):
        """Let molecules degrade for 1 time step"""
        self.map = torch.trunc(self.map * 10 ** self.n_decs) / (10 ** self.n_decs)
