import time
import pytest
from util import rand_genome
from genetics import Genetics, DOMAINS


# (genome, cds)
DATA = [
    (
        """
        TACCGGATA GCAGCTTTT CTTGGAATA GCCAAGGGT
        CGCCTTTAT ACCTATCTA CAACTACTA CTCGGTTGG
        TAACAAAGG TTAAAACGC CAAACGAGT ATCGGCCAA
        TCCTGTCAC TGTGAGAAG TTTCAATTA TAGATTCCT
        GGGGCGATT GGCGATGGT
        """,
        ["TTGGAATAG", "TTGGTAACAAAGGTTAAAACGCCAAACGAGTATCGGCCAATCCTGTCACTGTGA"],
    ),
    (
        """
        AACATATCC ACCATCCCT TAAGGGGCG ATGAATTAC
        GAAAGCGGG CGTACTACT TCTGGGGAT ACGATTAGT
        GTACTCGGT TCTCTTAAC GACTACCCT GTGTTACGT
        TATTGAAAG AGCAAATTG CGAGCTCCC CGTGACACT
        TGTGCGGCG CTATACACC CCTGCAGTT ATTTAAGGG
        CTTAGGCGA GAAGTTCCG CCTGCTAAG GAGTCCCTG
        TTGGGTGAA GTAACGCAC AGCCAGGCC TTGGCAGGA
        CGTTTCCGT TCTCGT
        """,
        [
            "ATGAATTACGAAAGCGGGCGTACTACTTCTGGGGATACGATTAGTGTACTCGGTTCTCTTAACGACTACCCTGTGTTACGTTATTGA",
            "GTGTACTCGGTTCTCTTAACGACTACCCTGTGTTACGTTATTGAAAGAGCAAATTGCGAGCTCCCCGTGACACTTGTGCGGCGCTATACACCCCTGCAGTTATTTAAGGGCTTAGGCGAGAAGTTCCGCCTGCTAAGGAGTCCCTGTTGGGTGAAGTAA",
            "GTGTTACGTTATTGA",
            "TTGAAAGAGCAAATTGCGAGCTCCCCGTGA",
            "TTGCGAGCTCCCCGTGACACTTGTGCGGCGCTATACACCCCTGCAGTTATTTAA",
            "GTGACACTTGTGCGGCGCTATACACCCCTGCAGTTATTTAAGGGCTTAGGCGAGAAGTTCCGCCTGCTAAGGAGTCCCTGTTGGGTGAAGTAA",
            "TTGTGCGGCGCTATACACCCCTGCAGTTATTTAAGGGCTTAG",
            "GTGCGGCGCTATACACCCCTGCAGTTATTTAAGGGCTTAGGCGAGAAGTTCCGCCTGCTAAGGAGTCCCTGTTGGGTGAAGTAA",
            "GTGAAGTAA",
        ],
    ),
]


@pytest.mark.parametrize("seq, exp", DATA)
def test_get_coding_regions(seq, exp):
    genetics = Genetics(domain_map={})
    res = genetics.get_coding_regions(seq="".join(seq.replace("\n", "").split()))
    assert len(res) == len(exp)
    assert set(res) == set(exp)


def test_performance():
    genetics = Genetics(domain_map=DOMAINS)
    gs = [rand_genome((1000, 5000)) for _ in range(100)]

    t0 = time.time()
    _ = [genetics.get_proteome(seq=d) for d in gs]
    td = time.time() - t0
    assert td < 0.4, "Used to take 0.19"