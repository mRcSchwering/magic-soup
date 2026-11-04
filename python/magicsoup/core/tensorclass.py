from typing import Any, TypedDict

import torch

from magicsoup.core.config import DEFAULT_DEVICE, FTYPE, IDXTYPE, ITYPE


class TensorKwargs(TypedDict, total=False):
    device: torch.device | str


class TensorClass:
    """
    Setup pytorch tensors

    Arguments:
        device: Device to use for torch tensors
            (see [pytorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)).

    Device have to be aligned between the different classes used in the simulation.
    """

    def __init__(self, device: torch.device | str = DEFAULT_DEVICE) -> None:
        self.device = device
        self.itype = ITYPE
        self.ftype = FTYPE
        self.idxtype = IDXTYPE

    def itensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=ITYPE)

    def izeros(self, *args) -> torch.Tensor:
        return torch.zeros(*args, device=self.device, dtype=ITYPE)

    def ftensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=FTYPE)

    def fzeros(self, *args) -> torch.Tensor:
        return torch.zeros(*args, device=self.device, dtype=FTYPE)

    def frand(self, *args) -> torch.Tensor:
        return torch.rand(*args, device=self.device, dtype=FTYPE)

    def idxtensor(self, d: Any) -> torch.Tensor:
        return torch.tensor(d, device=self.device, dtype=IDXTYPE)

    def idxfull(self, *args, value: int) -> torch.Tensor:
        return torch.full(args, fill_value=value, device=self.device, dtype=IDXTYPE)

    def idxzeros(self, *args) -> torch.Tensor:
        return torch.zeros(*args, device=self.device, dtype=IDXTYPE)

    def idxrandperm(self, n: int) -> torch.Tensor:
        return torch.randperm(n, device=self.device, dtype=IDXTYPE)

    def idxarange(self, n: int) -> torch.Tensor:
        return torch.arange(n, device=self.device, dtype=IDXTYPE)
