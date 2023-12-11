import pytest
import magicsoup as ms
from magicsoup.constants import CODON_SIZE, ProteinSpecType
from magicsoup import _lib  # type: ignore


# (genome, (start, stop))
# starts: "TTG", "GTG", "ATG"
# stops: "TGA", "TAG", "TAA"
# forward only
_DATA: list[tuple[str, list[tuple[int, int]]]] = [
    (
        """
        TACCGGATA GCAGCTTTT CTTGGAATA GCCAAGGGT
        CGCCTTTAT ACCTATCTA CAACTACTA CTCGGTTGG
        TAACAAAGG TTAAAACGC CAAACGAGT ATCGGCCAA
        TCCTGTCAC TGTGAGAAG TTTCAATTA TAGATTCCT
        GGGGCGATT GGCGATGGT
        """,
        # "TTGGAATAG" at 19 is too short
        [(68, 122)],
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
            # "GTGTTACGTTATTGA" at 99 is too short
            # "GTGAAGTAA" at 220 is too short
            (27, 114),
            (70, 229),
            (110, 140),
            (123, 177),
            (136, 229),
            (143, 185),
            (145, 229),
        ],
    ),
    (
        # min CDS size from start to end
        "TTGAAAGA GCAAATTT GA",
        [(0, 18)],
    ),
    (
        # two overlapping start GTG, different stops
        "GTGTGCTCG AAAGAGAAC GCAAATTCG TAACCTAG",
        [
            (0, 30),
            (2, 35),
        ],
    ),
]


def _get_coding_regions_rs(
    seq: str,
    min_cds_size: int,
    start_codons: list[str],
    stop_codons: list[str],
    is_fwd: bool,
) -> list[tuple[int, int, bool]]:
    return _lib.get_coding_regions(seq, min_cds_size, start_codons, stop_codons, is_fwd)


def _extract_domains_rs(
    genome: str,
    cdss: list[tuple[int, int, bool]],
    dom_size: int,
    dom_type_size: int,
    dom_type_map: dict[str, int],
    one_codon_map: dict[str, int],
    two_codon_map: dict[str, int],
) -> list[ProteinSpecType]:
    return _lib.extract_domains(
        genome,
        cdss,
        dom_size,
        dom_type_size,
        dom_type_map,
        one_codon_map,
        two_codon_map,
    )


def _reverse_complement_rs(seq: str) -> str:
    return _lib.reverse_complement(seq)


def test_reverse_complement():
    seq = "ACTGG"
    res = _reverse_complement_rs(seq=seq)
    assert res == "CCAGT"


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
    res = _get_coding_regions_rs(seq, **kwargs)  # type: ignore
    exp_starts, exp_stops = map(list, zip(*exp))

    assert len(res) == len(exp)
    assert set(d[0] for d in res) == set(exp_starts)
    assert set(d[1] for d in res) == set(exp_stops)

    for start, stop, is_fwd in res:
        idx = exp_starts.index(start)  # type: ignore
        assert start == exp_starts[idx]
        assert stop == exp_stops[idx]
        assert not is_fwd


def test_extract_domains():
    dom_type_map = {"AAA": 1, "GGG": 2, "CCC": 3}
    two_codon_map = {"ACTGAT": 1, "CTGTAT": 2, "CCGCGA": 3, "GGAATC": 4, "TGTCGA": 5}
    one_codon_map = {"ACT": 1, "CTG": 2, "CCG": 3, "GGA": 4, "TGT": 5}
    dom_type_size = len(next(iter(dom_type_map)))
    dom_size = dom_type_size + 5 * CODON_SIZE

    # fmt: off
    genome = (
        "AGACAAAAACTGTGTACTCCGCGATAGACTAGACG"
        "AGACTATAGCTAGAAGCCCCTGTACTCCGTGTCGATAGACG"
        "AGACTAGGGCCGGGACTGCCGCGACTAGAAGCTAGACTAACG"
        "AAACCGGGATGTCTGTAT"
        "CCCCCGGGACTGCCGCGAGGGACTCTGCCGGGAATC"
    )
    cdss: list[tuple[int, int, bool]] = [
        (0, 35, True),  # (1, 2, 5, 1, 3)
        (35, 76, False),  # (3, 5, 1, 3, 5)
        (76, 118, True),  # (2, 3, 4, 2, 3)
        (118, 136, False),  # (1, 3, 4, 5, 2)
        (136, 172, True),  # (3, 3, 4, 2, 3) (2, 1, 2, 3, 4)
    ]
    # - cds 0: normal domain                                                    => 1 res[0]
    # - cds 1: single type 3 domain, so it is removed
    # - cds 2: has 2 domain 2 starts, but the second is part of the 1 domain    => 1 res[1]
    # - cds 3: defines exactly 1 domain from start to end                       => 1 res[2]
    # - cds 4: defines exactly 2 domains, a 3rd type 2 start is in the middle   => 2 res[3]
    # fmt: on

    res = _extract_domains_rs(
        genome=genome,
        cdss=cdss,
        dom_type_size=dom_type_size,
        dom_size=dom_size,
        dom_type_map=dom_type_map,
        one_codon_map=one_codon_map,
        two_codon_map=two_codon_map,
    )
    # TODO: DomainTypeSpec is a list now

    # res[i]: (domain list, cds start, cds end, is fwd)
    # res[i][0][j]: (domain spec, dom start, dom end)
    assert len(res[0][0]) == 1
    assert len(res[1][0]) == 1
    assert len(res[2][0]) == 1
    assert len(res[3][0]) == 2
    assert res[0][1] == 0
    assert res[0][2] == 35
    assert res[0][3] is True
    assert res[1][1] == 76
    assert res[1][2] == 118
    assert res[1][3] is True
    assert res[2][1] == 118
    assert res[2][2] == 136
    assert res[2][3] is False
    assert res[3][1] == 136
    assert res[3][2] == 172
    assert res[3][3] is True
    assert res[0][0][0][0] == [1, 2, 5, 1, 3]
    assert res[0][0][0][1] == 6
    assert res[0][0][0][2] == 6 + dom_size
    assert res[1][0][0][0] == [2, 3, 4, 2, 3]
    assert res[1][0][0][1] == 6
    assert res[1][0][0][2] == 6 + dom_size
    assert res[2][0][0][0] == [1, 3, 4, 5, 2]
    assert res[2][0][0][1] == 0
    assert res[2][0][0][2] == 0 + dom_size
    assert res[3][0][0][0] == [3, 3, 4, 2, 3]
    assert res[3][0][0][1] == 0
    assert res[3][0][0][2] == 0 + dom_size
    assert res[3][0][1][0] == [2, 1, 2, 3, 4]
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
