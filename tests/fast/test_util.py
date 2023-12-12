from typing import Iterable
import pytest
import magicsoup.util as util


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


def test_moores_ngbrhd():
    m = [
        ["00", "01", "02", "03"],
        ["10", "11", "12", "13"],
        ["20", "21", "22", "23"],
        ["30", "31", "32", "33"],
    ]

    res = util.moore_nghbrhd(1, 1, size=len(m))
    assert set(res) == {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}

    res = util.moore_nghbrhd(0, 0, size=len(m))
    assert set(res) == {(0, 1), (1, 0), (1, 1), (0, 3), (1, 3), (3, 3), (3, 0), (3, 1)}


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
