from pathlib import Path

import pytest
import torch
from magicsoup import util
from magicsoup.config import CODON_SIZE

from tests.util import idxtensor


@pytest.mark.parametrize(
    "seq, exp",
    [
        ("ACTGATTA", "TAATCAGT"),
        ("GGAAT", "ATTCC"),
        ("", ""),
    ],
)
def test_reverse_complement(seq: str, exp: str) -> None:
    res = util.reverse_complement(seq)
    assert res == exp


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


def test_writing_reading_fasta(tmp_path: Path) -> None:
    orig = {"a": "gnome-a", "b": "", "": "genome-c", "d": "genome-d"}
    util.write_fasta(name_2_seq=orig, file=tmp_path / "test.fasta")
    new = util.read_fasta(file=tmp_path / "test.fasta")
    assert orig == new


@pytest.mark.parametrize(
    "idx, alp, src, exp",
    [
        ([0, 0, 0, 1, 1], [1, 2, 3], [0, -1, 2, 2, 0], [2, 4, 3]),
        ([0, 0, 1, 1, 2], [1, 2, 3], [0, 1, 2, 3, 4], [2, 7, 7]),
        ([0, 0, 0, 2, 2], [1, 2, 3], [0, 1, 2, 3, 4], [4, 2, 10]),
        ([], [], [], []),
        ([0], [0], [0], [0]),
        ([0, 0, 1], [[0, 0], [0, 0]], [[3, 2], [2, 1], [1, 2]], [[5, 3], [1, 2]]),
    ],
)
def test_index_add(idx: list, alp: list, src: list, exp: list) -> None:
    idx_ = idxtensor(idx)
    alp_ = idxtensor(alp)
    src_ = idxtensor(src)
    exp_ = idxtensor(exp)
    alp_.index_add_(dim=0, index=idx_, source=src_)
    assert torch.equal(alp_, exp_)


@pytest.mark.parametrize(
    "idx, src, red, exp",
    [
        ([0, 0, 1], [1, 2, 3], "sum", [3, 3]),
        ([0, 0, 1], [1, 2, 3], "amax", [2, 3]),
        ([0, 0, 1], [2, 2, 3], "mean", [2, 3]),
        ([0, 0, 0], [1, 2, 2], "prod", [4, 0]),
        ([0, 0, 0, 2], [1, 2, 3, 4], "sum", [6, 0, 4]),
        ([], [], "amax", []),
        ([0], [0], "prod", [0]),
    ],
)
def test_scatter_reduce(idx: list, src: list, red: str, exp: list) -> None:
    idx_ = idxtensor(idx)
    alp_ = idxtensor([0] * len(exp))  # since include_self=False
    src_ = idxtensor(src)
    exp_ = idxtensor(exp)
    alp_.scatter_reduce_(0, index=idx_, src=src_, reduce=red, include_self=False)
    assert torch.equal(alp_, exp_)
