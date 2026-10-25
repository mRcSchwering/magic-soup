from pathlib import Path

import pytest
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


def test_reverse_complement() -> None:
    seq = "ACTGG"
    res = rs.reverse_complement(seq=seq)
    assert res == "CCAGT"


def test_writing_reading_fasta(tmp_path: Path) -> None:
    orig = {"a": "gnome-a", "b": "", "": "genome-c", "d": "genome-d"}
    util.write_fasta(name_2_seq=orig, file=tmp_path / "test.fasta")
    new = util.read_fasta(file=tmp_path / "test.fasta")
    assert orig == new
