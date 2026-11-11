from typing import Annotated, Any, TypedDict

import torch

from magicsoup.core.config import DEFAULT_DEVICE, FTYPE, IDXTYPE, ITYPE

T = Annotated[torch.Tensor, "cuda"]


class TensorKwargs(TypedDict, total=False):
    device: torch.device | str


class TensorClass:
    """Consolidates tensor device and dtypes

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
        self.limits = (-1e38, 1e38)

    def idxmap_scatter(self, src: T, tgt: T) -> T:
        """Scatter 1D elements from src to tgt so that tgt[src[i]] = i

        Index of src becomes value in tgt at index of value of src.
        Elements < 0 are considered empty and are not copied.
        All tensors must be of index type and tgt.shape[0] >= src.shape[0].
        """
        c = src.shape[0]
        n = tgt.shape[0]
        idx = self.idxarange(c)
        out = self.idxempty(n + 1)
        out[:n].copy_(tgt)
        out.scatter_(0, torch.where(src >= 0, src, n), idx)
        return out[:n]

    def remap_gather(self, tgt: T, src: T, na: int = -1) -> T:
        """Gather 1D elements from tgt using src so that tgt[src[i]] = tgt[i]

        Converts tgt[i] -> tgt[j] using src[i] -> j.
        Elements < 0 are considered empty and are filled with na.
        src must be of index type and tgt.shape[0] >= src.shape[0].
        """
        out = tgt.index_select(0, src.clamp(0))
        out.masked_fill_(src < 0, na)
        return out

    def invert_idxmap(self, t: T, na: int = -1) -> T:
        """Invert 1D elements of t so that t[t[i]] = i

        Converts index map t[i] -> j to t[j] -> i.
        Elements < 0 are considered empty and are filled with na.
        t must be of index type.
        """
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
