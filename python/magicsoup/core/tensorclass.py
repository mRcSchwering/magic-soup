from collections.abc import Generator
from typing import Annotated, Any, TypedDict

import torch

from magicsoup.core.config import DEFAULT_DEVICE

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
        """torch device in use"""

        self.itype = torch.int8
        """torch dtype used for small integers"""

        self.ftype = torch.float32
        """torch dtype used for floats"""

        self.idxtype = torch.int32
        """torch dtype used for indexing integers"""

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
        """Get an integer tensor on device"""
        return torch.tensor(d, device=self.device, dtype=self.itype)

    def izeros(self, *args) -> T:
        """Get an integer tensor of zeros on device"""
        return torch.zeros(*args, device=self.device, dtype=self.itype)

    def ftensor(self, d: Any) -> T:
        """Get a float tensor on device"""
        return torch.tensor(d, device=self.device, dtype=self.ftype)

    def fzeros(self, *args) -> T:
        """Get a float tensor of zeros on device"""
        return torch.zeros(*args, device=self.device, dtype=self.ftype)

    def frand(self, *args) -> T:
        """Get a 0 to 1 uniform distribution float tensor on device"""
        return torch.rand(*args, device=self.device, dtype=self.ftype)

    def idxtensor(self, d: Any) -> T:
        """Get a index tensor on device"""
        return torch.tensor(d, device=self.device, dtype=self.idxtype)

    def idxfull(self, *args, value: int) -> T:
        """Get a index tensor filled with value on device"""
        return torch.full(args, fill_value=value, device=self.device, dtype=self.idxtype)

    def idxzeros(self, *args) -> T:
        """Get a index tensor of zeros on device"""
        return torch.zeros(*args, device=self.device, dtype=self.idxtype)

    def idxrandperm(self, n: int) -> T:
        """Get a 0 to n-1 random permutation index tensor on device"""
        return torch.randperm(n, device=self.device, dtype=self.idxtype)

    def idxarange(self, n: int) -> T:
        """Get a 0 to n-1 arranged index tensor on device"""
        return torch.arange(n, device=self.device, dtype=self.idxtype)

    def idxempty(self, *args) -> T:
        """Get an empty index tensor on device"""
        return torch.empty(*args, device=self.device, dtype=self.idxtype)


def iter_tensorclasses(root: object) -> Generator[TensorClass]:
    """Yield every [TensorClass][magicsoup.core.tensorclass.TensorClass] reachable from `root` (incl. `root`).

    Descends only through `TensorClass` instances; within a `World` they all
    reference each other (`World` -> `cells`/`culture`/..., `cells` -> `proteomics`
    -> maps, `culture` -> neighbourhood), so this reaches every one.
    """
    seen: set[int] = set()
    stack: list[object] = [root]
    while stack:
        obj = stack.pop()
        if not isinstance(obj, TensorClass) or id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        stack.extend(vars(obj).values())
