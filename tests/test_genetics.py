import pytest
import magicsoup as ms
from magicsoup.genetics import _get_coding_regions


# (genome, cds)
# starts: "TTG", "GTG", "ATG"
# stops: "TGA", "TAG", "TAA"
# forward only
DATA = [
    (
        """
        TACCGGATA GCAGCTTTT CTTGGAATA GCCAAGGGT
        CGCCTTTAT ACCTATCTA CAACTACTA CTCGGTTGG
        TAACAAAGG TTAAAACGC CAAACGAGT ATCGGCCAA
        TCCTGTCAC TGTGAGAAG TTTCAATTA TAGATTCCT
        GGGGCGATT GGCGATGGT
        """,
        # "TTGGAATAG" is too short
        ["TTGGTAACAAAGGTTAAAACGCCAAACGAGTATCGGCCAATCCTGTCACTGTGA"],
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
            # "GTGTTACGTTATTGA" is too short
            # "GTGAAGTAA" is too short
            "ATGAATTACGAAAGCGGGCGTACTACTTCTGGGGATACGATTAGTGTACTCGGTTCTCTTAACGACTACCCTGTGTTACGTTATTGA",
            "GTGTACTCGGTTCTCTTAACGACTACCCTGTGTTACGTTATTGAAAGAGCAAATTGCGAGCTCCCCGTGACACTTGTGCGGCGCTATACACCCCTGCAGTTATTTAAGGGCTTAGGCGAGAAGTTCCGCCTGCTAAGGAGTCCCTGTTGGGTGAAGTAA",
            "TTGAAAGAGCAAATTGCGAGCTCCCCGTGA",
            "TTGCGAGCTCCCCGTGACACTTGTGCGGCGCTATACACCCCTGCAGTTATTTAA",
            "GTGACACTTGTGCGGCGCTATACACCCCTGCAGTTATTTAAGGGCTTAGGCGAGAAGTTCCGCCTGCTAAGGAGTCCCTGTTGGGTGAAGTAA",
            "TTGTGCGGCGCTATACACCCCTGCAGTTATTTAAGGGCTTAG",
            "GTGCGGCGCTATACACCCCTGCAGTTATTTAAGGGCTTAGGCGAGAAGTTCCGCCTGCTAAGGAGTCCCTGTTGGGTGAAGTAA",
        ],
    ),
]


@pytest.mark.parametrize("seq, exp", DATA)
def test_get_coding_regions(seq: str, exp: list[str]):
    # 1 codon is too small to express p=0.01 domain types
    with pytest.warns(UserWarning):
        genetics = ms.Genetics(n_dom_type_nts=3)

    kwargs = {
        "start_codons": genetics.start_codons,
        "stop_codons": genetics.stop_codons,
        "min_cds_size": 18,
    }

    seq = "".join(seq.replace("\n", "").split())
    res = _get_coding_regions(seq, **kwargs)

    assert len(res) == len(exp)
    assert set(res) == set(exp)


# TODO: test _translate_genome
