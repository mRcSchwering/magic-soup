import pytest
import torch
from magicsoup.core.tensorclass import TensorClass

from tests.util import idxtensor


@pytest.fixture
def tenscls() -> TensorClass:
    return TensorClass()


@pytest.mark.parametrize(
    "fwd, bwd",
    [
        ([], []),
        ([-1, 2, -1, 1], [-1, 3, 1, -1]),
        ([-1, -1], [-1, -1]),
        ([2, 0, 1], [1, 2, 0]),
    ],
)
def test_invert_idxmap(tenscls: TensorClass, fwd: list, bwd: list) -> None:
    idx_map = idxtensor(fwd)
    exp_inv = idxtensor(bwd)
    res = tenscls.invert_idxmap(idx_map)
    assert torch.equal(res, exp_inv)


@pytest.mark.parametrize(
    "idxmap, vals, exp",
    [
        ([], [], []),
        ([-1, 2, -1, 1], [1, 2, 3, 4], [-1, 3, -1, 2]),
        ([-1, -1], [1, 2], [-1, -1]),
        ([2, 0, 1], [5, 2, 7], [7, 5, 2]),
    ],
)
def test_remap_gather(tenscls: TensorClass, idxmap: list, vals: list, exp: list) -> None:
    idx_map = idxtensor(idxmap)
    values = idxtensor(vals)
    res = tenscls.remap_gather(tgt=values, src=idx_map)
    assert torch.equal(res, idxtensor(exp))


@pytest.mark.parametrize(
    "src, tgt, exp",
    [
        ([], [], []),
        ([-1, -1, -1, 0, 2], [-1, 2, -1, 1, 0], [3, 2, 4, 1, 0]),
        ([-1, -1, -1], [-1, -1], [-1, -1]),
        ([-1, 1, 0, -1, 3], [0, 1, 2, -1, -1], [2, 1, 2, 4, -1]),
    ],
)
def test_idxmap_scatter(tenscls: TensorClass, src: list, tgt: list, exp: list) -> None:
    place = idxtensor(src)
    rec = idxtensor(tgt)
    res = tenscls.idxmap_scatter(src=place, tgt=rec)
    assert torch.equal(res, idxtensor(exp))
