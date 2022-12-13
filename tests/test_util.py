import torch
import magicsoup.util as util


def test_pad_unpad_map():
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

    m = torch.tensor(true_map).to(torch.bool)
    mp = torch.tensor(padded_map).to(torch.bool)

    res = util.pad_map(m=m)
    assert torch.all(res == mp)

    res = util.unpad_map(m=res)
    assert torch.all(res == m)


def test_padding_indices():
    xs, ys = util.padded_indices(x=0, y=0, s=5)
    assert set(zip(xs, ys)) == {(0, 0), (3, 3), (0, 3), (3, 0)}

    xs, ys = util.padded_indices(x=1, y=1, s=5)
    assert set(zip(xs, ys)) == {(1, 1), (1, 4), (4, 1), (4, 4)}

    xs, ys = util.padded_indices(x=0, y=1, s=5)
    assert set(zip(xs, ys)) == {(0, 1), (3, 1), (3, 4), (0, 4)}

    xs, ys = util.padded_indices(x=0, y=2, s=5)
    assert set(zip(xs, ys)) == {(0, 2), (3, 2)}

    xs, ys = util.padded_indices(x=2, y=2, s=5)
    assert set(zip(xs, ys)) == {(2, 2)}


def test_pad_2_true_idx():
    assert util.pad_2_true_idx(idx=0, size=3, pad=1) == 2
    assert util.pad_2_true_idx(idx=1, size=3, pad=1) == 0
    assert util.pad_2_true_idx(idx=2, size=3, pad=1) == 1
    assert util.pad_2_true_idx(idx=3, size=3, pad=1) == 2
    assert util.pad_2_true_idx(idx=4, size=3, pad=1) == 0

