import numpy as np
import pytest
from magicsoup.core.strarray import StrArray


def test_construction_and_reading() -> None:
    strings = ["ATG", "", "GGGCC", "T"]
    arr = StrArray(strings)
    assert len(arr) == 4
    assert arr.shape == (4,)
    assert arr.tolist() == strings
    assert list(arr) == strings
    assert arr.nbytes == sum(len(s) for s in strings)


def test_empty() -> None:
    arr = StrArray.empty(3)
    assert len(arr) == 3
    assert arr.tolist() == ["", "", ""]
    assert arr.nbytes == 0

    assert StrArray.empty().tolist() == []
    assert StrArray().tolist() == []


def test_integer_indexing() -> None:
    arr = StrArray(["ATG", "GGGCC", "T"])
    assert arr[0] == "ATG"
    assert arr[1] == "GGGCC"
    assert arr[-1] == "T"
    assert isinstance(arr[0], str)


@pytest.mark.parametrize(
    "idx, exp",
    [
        ([0, 2], ["ATG", "T"]),
        ([2, 2, 0], ["T", "T", "ATG"]),
        ([], []),
        (np.array([1, 0]), ["GGGCC", "ATG"]),
        (np.array([True, False, True]), ["ATG", "T"]),
    ],
)
def test_array_indexing(idx, exp) -> None:
    arr = StrArray(["ATG", "GGGCC", "T"])
    sub = arr[idx]
    assert isinstance(sub, StrArray)
    assert sub.tolist() == exp


def test_getitem_shares_buffer_no_copy() -> None:
    arr = StrArray(["ATG", "GGGCC", "T"])
    sub = arr[np.array([0, 2])]
    assert isinstance(sub, StrArray)
    # gather must not copy the string bytes
    assert sub._buf is arr._buf


@pytest.mark.parametrize(
    "idx, val, exp",
    [
        (np.array([0, 2]), ["TTTT", "G"], ["TTTT", "GGGCC", "G"]),
        (np.array([0, 2]), "AAA", ["AAA", "GGGCC", "AAA"]),
        (1, "AAA", ["ATG", "AAA", "T"]),
        (1, "", ["ATG", "", "T"]),
        (np.array([0, 1]), StrArray(["XX", "YY"]), ["XX", "YY", "T"]),
    ],
)
def test_setitem(idx, val, exp) -> None:
    arr = StrArray(["ATG", "GGGCC", "T"])
    arr[idx] = val
    assert arr.tolist() == exp


def test_setitem_scalar_kill() -> None:
    arr = StrArray(["ATG", "GGGCC", "T", "CC"])
    buf_before = arr._buf
    arr[np.array([1, 3])] = ""
    assert arr.tolist() == ["ATG", "", "T", ""]
    # killing must not touch the buffer
    assert arr._buf is buf_before


def test_setitem_strarray_same_buffer_is_zero_copy() -> None:
    # emulates cell division: genomes[children] = genomes[parents]
    arr = StrArray(["aaa", "bb", "c", "dddd"])
    buf_before = arr._buf
    parents = arr[np.array([0, 3])]
    assert parents._buf is arr._buf  # gather shares buffer
    arr[np.array([1, 2])] = parents
    assert arr.tolist() == ["aaa", "aaa", "dddd", "dddd"]
    # pure handle copy: buffer must be untouched
    assert arr._buf is buf_before


def test_division_then_independent_mutation() -> None:
    arr = StrArray(["parent", "x"])
    arr[np.array([1])] = arr[np.array([0])]  # child copies parent
    assert arr.tolist() == ["parent", "parent"]
    arr[1] = "mutated"  # mutating child must not affect parent
    assert arr.tolist() == ["parent", "mutated"]


def test_setitem_length_mismatch_raises() -> None:
    arr = StrArray(["a", "b", "c"])
    with pytest.raises(ValueError):
        arr[np.array([0, 1])] = ["only-one"]
    with pytest.raises(ValueError):
        arr[np.array([0, 1])] = StrArray(["x", "y", "z"])


def test_concat() -> None:
    a = StrArray(["ATG", "GG"])
    b = StrArray.empty(2)
    c = StrArray(["T"])
    joined = StrArray.concat([a, b, c])
    assert joined.tolist() == ["ATG", "GG", "", "", "T"]
    assert StrArray.concat([]).tolist() == []


# TODO: rm?
def test_grow() -> None:
    genomes = StrArray(["ATG", "GGGCC"])
    grown = StrArray.concat([genomes, StrArray.empty(3)])
    assert grown.tolist() == ["ATG", "GGGCC", "", "", ""]
    assert len(grown) == 5


def test_compact_reclaims_dead_bytes_and_preserves_values() -> None:
    arr = StrArray(["ATG", "GGGCC", "T"])
    arr[np.array([0, 2])] = ["TTTTTTTT", "AAAA"]  # appends -> dead bytes
    assert arr.buffer_nbytes > arr.nbytes
    values_before = arr.tolist()

    arr.compact()
    assert arr.tolist() == values_before
    assert arr.buffer_nbytes == arr.nbytes


def test_copy_is_independent() -> None:
    arr = StrArray(["ATG", "GG"])
    dup = arr.copy()
    assert dup._buf is not arr._buf
    assert dup.tolist() == arr.tolist()
    arr[0] = "CCCC"
    assert dup.tolist() == ["ATG", "GG"]  # unaffected


# TODO: what is this testing?!
def test_as_buffers_roundtrip() -> None:
    strings = ["ATG", "", "GGGCC", "T"]
    arr = StrArray(strings)
    buf, starts, lens = arr.as_buffers()
    assert buf.dtype == np.uint8
    assert starts.dtype == np.int64
    assert lens.dtype == np.int64
    rebuilt = [
        buf[s : s + n].tobytes().decode()
        for s, n in zip(starts.tolist(), lens.tolist())
    ]
    assert rebuilt == strings


def test_unicode_roundtrip() -> None:
    strings = ["0.1.3", "αβγ", "", "lineage→child"]
    arr = StrArray(strings)
    assert arr.tolist() == strings
    assert arr[1] == "αβγ"


# TODO: werent these already tested?
def test_empty_array_operations() -> None:
    arr = StrArray.empty(0)
    assert arr[np.array([], dtype=np.int64)].tolist() == []
    arr.compact()
    assert arr.tolist() == []
    buf, starts, lens = arr.as_buffers()
    assert buf.size == 0 and starts.size == 0 and lens.size == 0
