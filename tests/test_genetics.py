import time
import pytest
import torch
from util import rand_genome
from genetics import CellSignal, WorldSignal, Genetics

TOLERANCE = 1e-3


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
    genetics = Genetics()
    res = genetics.get_coding_regions(seq="".join(seq.replace("\n", "").split()))
    assert len(res) == len(exp)
    assert set(res) == set(exp)


def test_cell_time_step():
    # fmt: off
    cell = [
        {
            ("RcF", WorldSignal.F, True): 0.1,
            ("OutA", CellSignal.MA, False): 0.8,
            ("OutB", CellSignal.MB, False): 0.5,
            ("OutD", CellSignal.MD, False): 0.3,
        },
        {
            ("InC", CellSignal.MC, True): 0.4,
            ("OutB", CellSignal.MB, False): 0.9,
            ("OutC", CellSignal.MC, False): 0.4,
        },
        {
            ("InB", CellSignal.MB, True): 0.2,
            ("OutC", CellSignal.MC, False): 0.3
        },
        {
            ("InD", CellSignal.MD, True): 0.6,
            ("OutA", CellSignal.MA, False): 0.7,
            ("OutD", CellSignal.MD, False): 0.8,
        },
    ]
    # fmt: on

    # initial concentrations
    c0_a = 1.1
    c0_b = 1.2
    c0_c = 1.3
    c0_d = 1.4
    c0_f = 1.5
    C0 = torch.tensor([[0.0, c0_a, c0_b, c0_c, c0_d, c0_f, 0.0, 0.0]])

    # by hand with:
    # def f(x: float) -> float:
    #    return (1 - math.exp(-(x ** 3)))
    c1_a = 0.3157  # f(c0_f * 0.1) * 0.8 + f(c0_d * 0.6) * 0.7
    c1_b = 0.1197  # f(c0_f * 0.1) * 0.5 + f(c0_c * 0.4) * 0.9
    c1_c = 0.0565  # f(c0_c * 0.4) * 0.4 + f(c0_b * 0.2) * 0.3
    c1_d = 0.3587  # f(c0_f * 0.1) * 0.3 + f(c0_d * 0.6) * 0.8
    c1_f = 0.0  # no edge points to f

    genetics = Genetics()

    A, B = genetics.get_cell_params(cells=[cell])
    assert A.shape == (1, 8, 4)
    assert B.shape == (1, 8, 4)

    C1 = genetics.simulate_protein_work(C=C0, A=A, B=B)
    assert C1.shape == (1, 8)
    assert C1[0, 0] == 0.0
    assert C1[0, 1] == pytest.approx(c1_a, abs=TOLERANCE)
    assert C1[0, 2] == pytest.approx(c1_b, abs=TOLERANCE)
    assert C1[0, 3] == pytest.approx(c1_c, abs=TOLERANCE)
    assert C1[0, 4] == pytest.approx(c1_d, abs=TOLERANCE)
    assert C1[0, 5] == pytest.approx(c1_f, abs=TOLERANCE)
    assert C1[0, 6] == 0.0
    assert C1[0, 7] == 0.0


def test_performance():
    genetics = Genetics()
    gs = [rand_genome((1000, 5000)) for _ in range(100)]

    # TODO: improve transcription and translation performance
    t0 = time.time()
    cells = [genetics.get_proteome(g=d) for d in gs]
    td = time.time() - t0
    assert td < 0.4, "get_proteome performance degraded a lot"

    # TODO: improve get cell params performance
    t0 = time.time()
    A, B = genetics.get_cell_params(cells=cells)
    td = time.time() - t0
    assert td < 0.5, "get_cell_params performance degraded a lot"

    C = torch.randn(len(cells), len(CellSignal) + len(WorldSignal))

    t0 = time.time()
    _ = genetics.simulate_protein_work(C=C, A=A, B=B)
    td = time.time() - t0
    assert td < 0.01, "get_cell_params performance degraded a lot"

