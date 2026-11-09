from typing import Any, TypedDict

import torch

from magicsoup.core.config import DEFAULT_DEVICE, FTYPE, IDXTYPE, ITYPE, T


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

    def remap_gather(self, t: T, idx_map: T, na: int = -1) -> T:
        """With index(i) -> j convert values from values(i) to values(j)"""
        out = t.index_select(0, idx_map.clamp(0))
        out.masked_fill_(idx_map < 0, na)
        return out

    def invert_idx_map(self, t: T, na: int = -1) -> T:
        """Convert index(i) -> j to index(j) -> i"""
        c = t.shape[0]
        buf = self.idxfull(c + 1, value=na)
        idx = torch.where(t >= 0, t, c)  # c -> na
        buf[idx] = self.idxarange(c)
        return buf[:c]

    def itensor(self, d: Any) -> T:
        return torch.tensor(d, device=self.device, dtype=ITYPE)

    def izeros(self, *args) -> T:
        return torch.zeros(*args, device=self.device, dtype=ITYPE)

    def ftensor(self, d: Any) -> T:
        return torch.tensor(d, device=self.device, dtype=FTYPE)

    def fzeros(self, *args) -> T:
        return torch.zeros(*args, device=self.device, dtype=FTYPE)

    def frand(self, *args) -> T:
        return torch.rand(*args, device=self.device, dtype=FTYPE)

    def idxtensor(self, d: Any) -> T:
        return torch.tensor(d, device=self.device, dtype=IDXTYPE)

    def idxfull(self, *args, value: int) -> T:
        return torch.full(args, fill_value=value, device=self.device, dtype=IDXTYPE)

    def idxzeros(self, *args) -> T:
        return torch.zeros(*args, device=self.device, dtype=IDXTYPE)

    def idxrandperm(self, n: int) -> T:
        return torch.randperm(n, device=self.device, dtype=IDXTYPE)

    def idxarange(self, n: int) -> T:
        return torch.arange(n, device=self.device, dtype=IDXTYPE)

    def idxempty(self, *args) -> T:
        return torch.empty(*args, device=self.device, dtype=IDXTYPE)
