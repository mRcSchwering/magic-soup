import pytest
import magicsoup as ms
from magicsoup.examples.wood_ljungdahl import MOLECULES, REACTIONS


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
def test_get_coding_regions(seq, exp):
    chemistry = ms.Chemistry(
        reactions=[([MOLECULES[0]], [MOLECULES[1]])], molecules=MOLECULES[:2]
    )

    # min domain size is 2 + 4 codons = 18 nucleotides
    # min CDS is 24 nucleotides (with start and stop codons)
    with pytest.warns(UserWarning):
        genetics = ms.Genetics(
            chemistry=chemistry,
            n_dom_type_nts=3,
            n_reaction_nts=3,
            n_affinity_nts=3,
            n_velocity_nts=3,
            n_orientation_nts=3,
        )
    res = genetics.get_coding_regions(seq="".join(seq.replace("\n", "").split()))
    assert len(res) == len(exp)
    assert set(res) == set(exp)


@pytest.mark.parametrize(
    "doms1, doms2",
    [
        (
            [
                ms.CatalyticDomain(REACTIONS[0], 1.0, 1.0, False),
                ms.CatalyticDomain(REACTIONS[1], 2.0, 2.0, True),
            ],
            [
                ms.CatalyticDomain(REACTIONS[1], 2.0, 2.0, True),
                ms.CatalyticDomain(REACTIONS[0], 1.0, 1.0, False),
            ],
        ),
        (
            [
                ms.CatalyticDomain(REACTIONS[0], 1.0, 1.0, False),
                ms.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
            ],
            [
                ms.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
                ms.CatalyticDomain(REACTIONS[0], 1.0, 1.0, False),
            ],
        ),
        (
            [
                ms.RegulatoryDomain(MOLECULES[0], 1.0, False, False),
                ms.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
            ],
            [
                ms.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
                ms.RegulatoryDomain(MOLECULES[0], 1.0, False, False),
            ],
        ),
    ],
)
def test_comparing_same_proteins(doms1, doms2):
    p1 = ms.Protein(domains=doms1, label="P1")
    p2 = ms.Protein(domains=doms2, label="P2")
    assert p1 == p2


@pytest.mark.parametrize(
    "doms1, doms2",
    [
        (
            [
                ms.CatalyticDomain(REACTIONS[0], 1.0, 2.0, False),
                ms.CatalyticDomain(REACTIONS[1], 2.0, 2.0, True),
            ],
            [
                ms.CatalyticDomain(REACTIONS[1], 2.0, 2.0, True),
                ms.CatalyticDomain(REACTIONS[0], 1.0, 1.0, False),
            ],
        ),
        (
            [
                ms.CatalyticDomain(REACTIONS[1], 1.0, 1.0, False),
                ms.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
            ],
            [
                ms.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
                ms.CatalyticDomain(REACTIONS[0], 1.0, 1.0, False),
            ],
        ),
        (
            [
                ms.RegulatoryDomain(MOLECULES[0], 1.0, True, False),
                ms.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
            ],
            [
                ms.TransporterDomain(MOLECULES[0], 1.0, 1.0, False),
                ms.RegulatoryDomain(MOLECULES[0], 1.0, False, False),
            ],
        ),
    ],
)
def test_comparing_different_proteins(doms1, doms2):
    p1 = ms.Protein(domains=doms1, label="P1")
    p2 = ms.Protein(domains=doms2, label="P2")
    assert p1 != p2
