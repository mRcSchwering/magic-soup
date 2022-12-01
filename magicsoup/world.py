import random
from util import trunc
import torch


# TODO: fit diffusion rate to natural diffusion rate of small molecules in cell


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
        mol_diff_rate=1.0,
        mol_map_init="randn",
        trunc_n_decs=4,
    ):
        self.size = size
        self.n_molecules = n_molecules
        self.mol_degrad = mol_degrad
        self.torch_kwargs = {"dtype": dtype, "device": device}
        self.trunc_n_decs = trunc_n_decs

        self.molecule_map = self._get_molecule_map(mol_map_init=mol_map_init)
        self.cell_map = self._get_cell_map()
        self.conv113 = self._get_conv(mol_diff_rate=mol_diff_rate)

    def add_cells(self, n_cells: int) -> list[tuple[int, int]]:
        """
        Randomly place `n_cells` on `world.cell_map` and return
        their positions.
        """
        pxls = torch.argwhere(~self.cell_map)

        try:
            idxs = random.sample(range(len(pxls)), k=n_cells)
        except ValueError as err:
            raise ValueError(
                f"Wanted to add {n_cells} cells but only {len(pxls)} pixels left on map"
            ) from err

        return [tuple(d.tolist()) for d in pxls[idxs]]  # type: ignore

    @torch.no_grad()
    def diffuse_molecules(self):
        self.molecule_map = trunc(
            tens=self.conv113(self.molecule_map), n_decs=self.trunc_n_decs
        )

    def degrade_molecules(self):
        self.molecule_map = trunc(
            tens=self.molecule_map * self.mol_degrad, n_decs=self.trunc_n_decs
        )

    def _get_cell_map(self) -> torch.Tensor:
        kwargs = {**self.torch_kwargs, "dtype": torch.bool}
        return torch.zeros(self.size, self.size, **kwargs)  # type: ignore

    def _get_molecule_map(self, mol_map_init: str) -> torch.Tensor:
        args = [self.n_molecules, 1, self.size, self.size]
        if mol_map_init == "zeros":
            return torch.zeros(*args, **self.torch_kwargs)
        if mol_map_init == "randn":
            return torch.randn(*args, **self.torch_kwargs).abs()
        raise ValueError(f"Didnt recognize mol_map_init={mol_map_init}")

    def _get_conv(self, mol_diff_rate: float) -> torch.nn.Conv2d:
        if not 0.0 <= mol_diff_rate <= 1.0:
            raise ValueError(
                "Diffusion rate must be between 0 and 1. "
                f"Now it's mol_diff_rate={mol_diff_rate}"
            )

        if mol_diff_rate == 0.0:
            a = 0.0
            b = 1.0
        else:
            d = 1 / mol_diff_rate
            a = 1 / (d + 8)
            b = d * a

        # fmt: off
        kernel = torch.tensor([[[
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
            **self.torch_kwargs,
        )
        conv.weight = torch.nn.Parameter(kernel, requires_grad=False)
        return conv

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(size=%r,n_molecules=%r)" % (clsname, self.size, self.n_molecules)
