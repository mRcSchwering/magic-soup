from typing import Iterable
import pytest
import magicsoup.util as util
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
        codons = set(seq[d : d + CODON_SIZE] for d in range(0, len(seq), CODON_SIZE))
        assert len(set(codons) & set(excl_codons)) == 0

    excl_codons.append("AAA")
    res = util.codons(n=n, excl_codons=excl_codons)
    assert len(set(res)) == len(res)
    assert all(len(d) == n * CODON_SIZE for d in res)
    assert len(res) == (n_codons - len(excl_codons)) ** n
    for seq in res:
        codons = set(seq[d : d + CODON_SIZE] for d in range(0, len(seq), CODON_SIZE))
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


@pytest.mark.parametrize(
    "x, y, exp",
    [
        (2, 2, [(1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3), (2, 1), (2, 3)]),
        (0, 0, [(4, 4), (4, 0), (4, 1), (1, 4), (1, 0), (1, 1), (0, 4), (0, 1)]),
        (4, 4, [(3, 3), (3, 4), (3, 0), (0, 3), (0, 4), (0, 0), (4, 3), (4, 0)]),
        (0, 4, [(4, 3), (4, 4), (4, 0), (1, 3), (1, 4), (1, 0), (0, 3), (0, 0)]),
        (4, 0, [(3, 4), (4, 4), (0, 4), (3, 1), (4, 1), (0, 1), (3, 0), (0, 0)]),
    ],
)
def test_free_moores_nghbhd(x: int, y: int, exp: list[tuple[int, int]]):
    res = util.free_moores_nghbhd(x=x, y=y, positions=[], map_size=5)
    assert set(res) == set(exp)

    occ = res[0]
    res1 = util.free_moores_nghbhd(x=x, y=y, positions=[occ], map_size=5)
    assert set(res1) == set(exp) - {occ}

    res2 = util.free_moores_nghbhd(x=x, y=y, positions=res, map_size=5)
    assert len(res2) == 0
