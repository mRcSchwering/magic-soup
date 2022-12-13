import torch
import magicsoup as ms


def test_cpad2d():
    # fmt: off
    true_map = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]
    padded_map = [
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 1.0],
    ]
    # fmt: on

    res = ms.cpad2d(torch.tensor(true_map), n=1)
    assert torch.all(res == torch.tensor(padded_map))


# 0,0 -> 0:3,0:3
# 0,1 -> 0:3,1:4


def test_pad_2_true_idx():
    assert ms.pad_2_true_idx(idx=0, size=3, pad=1) == 2
    assert ms.pad_2_true_idx(idx=1, size=3, pad=1) == 0
    assert ms.pad_2_true_idx(idx=2, size=3, pad=1) == 1
    assert ms.pad_2_true_idx(idx=3, size=3, pad=1) == 2
    assert ms.pad_2_true_idx(idx=4, size=3, pad=1) == 0

