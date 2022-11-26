from typing import Optional
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
        self._init_map(map_init=map_init)
        self._init_conv(kernel=kernel)

    def _init_map(self, map_init: Optional[str] = None):
        if map_init is None or map_init == "zeros":
            self.map = torch.zeros(
                self.layers, 1, self.size, self.size, device=self.device, dtype=self.dtype
            )
        elif map_init == "randn":
            self.map = torch.zeros(
                self.layers, 1, self.size, self.size, device=self.device, dtype=self.dtype
            )
        

    def _init_conv(self, kernel: Optional[torch.Tensor] = None):
        if kernel is None:
            # fmt: off
            kernel = torch.tensor([[[
                [0.1, 0.1, 0.1],
                [0.1, 0.2, 0.1],
                [0.1, 0.1, 0.1],
            ]]])
            # fmt: on

        self._conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
            dtype=self.dtype,
            device=self.device,
        )
        self._conv.weight = torch.nn.Parameter(kernel, requires_grad=False)

    @torch.no_grad()
    def diffuse(self):
        """Let molecules diffuse for 1 time step"""
        self.map = self._conv(self.map)

    @torch.no_grad()
    def degrade(self):
        """Let molecules degrade for 1 time step"""
        self.map = torch.trunc(self.map * 10 ** self.n_decs) / (10 ** self.n_decs)
