import pytest
import magicsoup as ms
from magicsoup.genetics import _get_coding_regions, _extract_domains
from magicsoup.constants import CODON_SIZE


# (genome, (cds, start))
# starts: "TTG", "GTG", "ATG"
# stops: "TGA", "TAG", "TAA"
# forward only
_DATA: list[tuple[str, list[tuple[str, int]]]] = [
    (
        """
        TACCGGATA GCAGCTTTT CTTGGAATA GCCAAGGGT
        CGCCTTTAT ACCTATCTA CAACTACTA CTCGGTTGG
        TAACAAAGG TTAAAACGC CAAACGAGT ATCGGCCAA
        TCCTGTCAC TGTGAGAAG TTTCAATTA TAGATTCCT
        GGGGCGATT GGCGATGGT
        """,
        # "TTGGAATAG" is too short
        [("TTGGTAACAAAGGTTAAAACGCCAAACGAGTATCGGCCAATCCTGTCACTGTGA", 68)],
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
            (
                "ATGAATTACGAAAGCGGGCGTACTACTTCTGGGGATACGATTAGTGTACTCGGTTCTCTTAACGACTACCCTGTGTTACGTTATTGA",
                27,
            ),
            (
                "GTGTACTCGGTTCTCTTAACGACTACCCTGTGTTACGTTATTGAAAGAGCAAATTGCGAGCTCCCCGTGACACTTGTGCGGCGCTATACACCCCTGCAGTTATTTAAGGGCTTAGGCGAGAAGTTCCGCCTGCTAAGGAGTCCCTGTTGGGTGAAGTAA",
                70,
            ),
            ("TTGAAAGAGCAAATTGCGAGCTCCCCGTGA", 110),
            ("TTGCGAGCTCCCCGTGACACTTGTGCGGCGCTATACACCCCTGCAGTTATTTAA", 123),
            (
                "GTGACACTTGTGCGGCGCTATACACCCCTGCAGTTATTTAAGGGCTTAGGCGAGAAGTTCCGCCTGCTAAGGAGTCCCTGTTGGGTGAAGTAA",
                136,
            ),
            ("TTGTGCGGCGCTATACACCCCTGCAGTTATTTAAGGGCTTAG", 143),
            (
                "GTGCGGCGCTATACACCCCTGCAGTTATTTAAGGGCTTAGGCGAGAAGTTCCGCCTGCTAAGGAGTCCCTGTTGGGTGAAGTAA",
                145,
            ),
        ],
    ),
]


@pytest.mark.parametrize("seq, exp", _DATA)
def test_get_coding_regions(seq: str, exp: list[tuple[str, int]]):
    # 1 codon is too small to express p=0.01 domain types
    with pytest.warns(UserWarning):
        genetics = ms.Genetics(n_dom_type_codons=1)

    kwargs = {
        "start_codons": genetics.start_codons,
        "stop_codons": genetics.stop_codons,
        "min_cds_size": 18,
        "is_fwd": False,
    }

    seq = "".join(seq.replace("\n", "").split())
    res = _get_coding_regions(seq, **kwargs)  # type: ignore
    exp_cdss, exp_starts = map(list, zip(*exp))

    assert len(res) == len(exp)
    assert set(d[0] for d in res) == set(exp_cdss)

    for cds, start, stop, is_fwd in res:
        idx = exp_cdss.index(cds)  # type: ignore
        assert start == exp_starts[idx]
        assert stop == exp_starts[idx] + len(exp_cdss[idx])  # type: ignore
        assert not is_fwd


def test_extract_domains():
    dom_type_map = {"AAA": 1, "GGG": 2, "CCC": 3}
    two_codon_map = {"ACTGAT": 1, "CTGTAT": 2, "CCGCGA": 3, "GGAATC": 4, "TGTCGA": 5}
    one_codon_map = {"ACT": 1, "CTG": 2, "CCG": 3, "GGA": 4, "TGT": 5}
    dom_type_size = len(next(iter(dom_type_map)))
    dom_size = dom_type_size + 5 * CODON_SIZE

    # fmt: off
    cdss: list[tuple[str, int, int, bool]] = [
        ("AGACAAAAACTGTGTACTCCGCGATAGACTAGACG", 1, 36, True),  # (1, 2, 5, 1, 3)
        ("AGACTATAGCTAGAAGCCCCTGTACTCCGTGTCGATAGACG", 10, 51, False),  # (3, 5, 1, 3, 5)
        ("AGACTAGGGCCGGGACTGCCGCGACTAGAAGCTAGACTAACG", 4, 47, True),  # (2, 3, 4, 2, 3)
        ("AAACCGGGATGTCTGTAT", 17, 35, False),  # (1, 3, 4, 5, 2)
        ("CCCCCGGGACTGCCGCGAGGGACTCTGCCGGGAATC", 12, 48, True),  # (3, 3, 4, 2, 3) (2, 1, 2, 3, 4)
    ]
    # - cds 0: normal domain                                                    => 1 res[0]
    # - cds 1: single type 3 domain, so it is removed
    # - cds 2: has 2 domain 2 starts, but the second is part of the 1 domain    => 1 res[1]
    # - cds 3: defines exactly 1 domain from start to end                       => 1 res[2]
    # - cds 4: defines exactly 2 domains, a 3rd type 2 start is in the middle   => 2 res[3]
    # fmt: on

    res = _extract_domains(
        cdss=cdss,
        dom_type_size=dom_type_size,
        dom_size=dom_size,
        dom_type_map=dom_type_map,
        one_codon_map=one_codon_map,
        two_codon_map=two_codon_map,
    )

    # res[i]: (domain list, cds start, cds end, is fwd)
    # res[i][0][j]: (domain spec, dom start, dom end)
    assert len(res[0][0]) == 1
    assert len(res[1][0]) == 1
    assert len(res[2][0]) == 1
    assert len(res[3][0]) == 2
    assert res[0][1] == 1
    assert res[0][2] == 36
    assert res[0][3] is True
    assert res[1][1] == 4
    assert res[1][2] == 47
    assert res[1][3] is True
    assert res[2][1] == 17
    assert res[2][2] == 35
    assert res[2][3] is False
    assert res[3][1] == 12
    assert res[3][2] == 48
    assert res[3][3] is True
    assert res[0][0][0][0] == (1, 2, 5, 1, 3)
    assert res[0][0][0][1] == 6
    assert res[0][0][0][2] == 6 + dom_size
    assert res[1][0][0][0] == (2, 3, 4, 2, 3)
    assert res[1][0][0][1] == 6
    assert res[1][0][0][2] == 6 + dom_size
    assert res[2][0][0][0] == (1, 3, 4, 5, 2)
    assert res[2][0][0][1] == 0
    assert res[2][0][0][2] == 0 + dom_size
    assert res[3][0][0][0] == (3, 3, 4, 2, 3)
    assert res[3][0][0][1] == 0
    assert res[3][0][0][2] == 0 + dom_size
    assert res[3][0][1][0] == (2, 1, 2, 3, 4)
    assert res[3][0][1][1] == 18
    assert res[3][0][1][2] == 18 + dom_size


def test_genetics():
    # 1=catalytic, 2=transporter, 3=regulatory
    # regulatory-only proteins get sorted out, so there is a bias towards
    # fewer regulatory domains

    # all same likelihood (while considering reg bias)
    kwargs = {"p_catal_dom": 0.1, "p_transp_dom": 0.1, "p_reg_dom": 0.1}
    genetics = ms.Genetics(**kwargs)
    genomes = [ms.random_genome(s=500) for _ in range(1000)]
    proteomes_data = genetics.translate_genomes(genomes=genomes)

    n_catal = _sum_dom_type(proteomes_data, 1)
    n_trnsp = _sum_dom_type(proteomes_data, 2)
    n_reg = _sum_dom_type(proteomes_data, 3)
    n = n_catal + n_trnsp + n_reg
    assert abs(n_catal - n_trnsp) < 0.1 * n
    assert abs(n_trnsp - n_reg) < 0.2 * n

    # fewer catalytics (while considering reg bias)
    kwargs["p_catal_dom"] = 0.01
    genetics = ms.Genetics(**kwargs)
    genomes = [ms.random_genome(s=500) for _ in range(1000)]
    proteomes_data = genetics.translate_genomes(genomes=genomes)

    n_catal = _sum_dom_type(proteomes_data, 1)
    n_trnsp = _sum_dom_type(proteomes_data, 2)
    n_reg = _sum_dom_type(proteomes_data, 3)
    n = n_catal + n_trnsp + n_reg
    assert n_trnsp - n_catal > 0.9 * n / 3
    assert n_reg - n_catal > 0.6 * n / 3

    # also fewer transporters (while considering reg bias)
    kwargs["p_transp_dom"] = 0.01
    genetics = ms.Genetics(**kwargs)
    genomes = [ms.random_genome(s=500) for _ in range(1000)]
    proteomes_data = genetics.translate_genomes(genomes=genomes)

    n_catal = _sum_dom_type(proteomes_data, 1)
    n_trnsp = _sum_dom_type(proteomes_data, 2)
    n_reg = _sum_dom_type(proteomes_data, 3)
    n = n_catal + n_trnsp + n_reg
    assert n_reg - n_catal > 0.6 * n / 3
    assert n_reg - n_trnsp > 0.6 * n / 3


def _sum_dom_type(data: list[list[ms.ProteinSpecType]], type_: int) -> int:
    out = 0
    for cell in data:
        for protein, *_ in cell:
            for dom, *_ in protein:
                if dom[0] == type_:
                    out += 1
    return out
