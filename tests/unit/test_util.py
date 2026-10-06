from collections.abc import Iterable

import pytest
import torch
from magicsoup import rs, util
from magicsoup.constants import CODON_SIZE

# fmt: off
@pytest.mark.parametrize("tmp, exp", [
    ("ANC", ["ATC", "ACC", "AGC", "AAC"]),
    ("ANN", ["ATT", "ACT", "AGT", "AAT",
             "ATC", "ACC", "AGC", "AAC",
             "ATG", "ACG", "AGG", "AAG",
             "ATA", "ACA", "AGA", "AAA"]),
    ("ARC", ["AGC", "AAC"]),
    ("AYC", ["ATC", "ACC"]),
    ("AYN", ["ATC", "ATT", "ATG", "ATA",
             "ACC", "ACT", "ACG", "ACA"]),
])
def test_variants(tmp, exp):
    res = util.variants(seq=tmp)
    assert set(res) == set(exp)
# fmt: on


@pytest.mark.parametrize("n", [1, 2])
def test_codons(n: int):
    n_codons = 4**CODON_SIZE

    res = util.codons(n=n)
    assert len(set(res)) == len(res)
    assert all(len(d) == n * CODON_SIZE for d in res)
    assert len(res) == n_codons**n

    excl_codons = ["TTT"]
    res = util.codons(n=n, excl_codons=excl_codons)
    assert len(set(res)) == len(res)
    assert all(len(d) == n * CODON_SIZE for d in res)
    assert len(res) == (n_codons - len(excl_codons)) ** n
    for seq in res:
        codons = {seq[d : d + CODON_SIZE] for d in range(0, len(seq), CODON_SIZE)}
        assert len(set(codons) & set(excl_codons)) == 0

    excl_codons.append("AAA")
    res = util.codons(n=n, excl_codons=excl_codons)
    assert len(set(res)) == len(res)
    assert all(len(d) == n * CODON_SIZE for d in res)
    assert len(res) == (n_codons - len(excl_codons)) ** n
    for seq in res:
        codons = {seq[d : d + CODON_SIZE] for d in range(0, len(seq), CODON_SIZE)}
        assert len(set(codons) & set(excl_codons)) == 0


@pytest.mark.parametrize(
    "s, excl",
    [
        (0, []),
        (1, []),
        (10, []),
        (100, []),
        (0, ["TGA", "TAG", "TAA"]),
        (1, ["TGA", "TAG", "TAA"]),
        (10, ["TGA", "TAG", "TAA"]),
        (100, ["TGA", "TAG", "TAA"]),
    ],
)
def test_random_genome(s, excl):
    g = util.random_genome(s=s, excl=excl)
    assert len(g) == s

    for seq in excl:
        assert seq not in g


@pytest.mark.parametrize(
    "vals, key, exp",
    [
        ([1.0, 1.4, 1.8], 1.5, 1.4),
        ([1.0, 1.4, 1.6], 1.5, 1.4),
        ([1.0, 1.4, 1.6], -100.0, 1.0),
        ([1.0, 1.4, 1.6], 100.0, 1.6),
        ([-1.0, 1.4, 1.6], 0.0, -1.0),
        ({0.1: "a", 0.2: "b"}, 0.0, 0.1),
        ({3: "a", 4: "b"}, 2, 3),
    ],
)
def test_closest_value(vals: Iterable, key: float, exp: float):
    res = util.closest_value(values=vals, key=key)
    assert res == exp


@pytest.mark.parametrize(
    "a, b, exp",
    [
        (0, 1, 1),
        (0, 2, 2),
        (0, 3, 2),
        (0, 4, 1),
        (0, 0, 0),
    ],
)
def test_dist_1d(a: int, b: int, exp: int):
    res = util.dist_1d(a=a, b=b, m=5)
    assert res == exp


def test_array():
    arr = util.Array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
    M = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]).bool()

    assert arr[[0]] == ["a"]
    assert arr[[-1]] == ["j"]
    assert arr[3:5] == ["d", "e"]
    assert arr[torch.tensor([0, 1, 2])] == ["a", "b", "c"]
    assert arr[M] == ["a", "c", "e", "g", "i"]

    with pytest.raises(ValueError):
        _ = arr[0]

    arr[1:3] = ["B", "C"]
    assert arr[:4] == ["a", "B", "C", "d"]
    arr[M] = ["A", "C", "E", "G", "I"]
    assert arr[:] == ["A", "B", "C", "d", "E", "f", "G", "h", "I", "j"]
    arr[torch.tensor([-2, -1])] = ["X", "X"]
    assert arr[-3:] == ["h", "X", "X"]


def test_reverse_complement() -> None:
    seq = "ACTGG"
    res = rs.reverse_complement(seq=seq)
    assert res == "CCAGT"
