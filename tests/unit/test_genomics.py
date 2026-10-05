import pytest
from magicsoup import _lib  # type: ignore
from magicsoup.constants import CODON_SIZE, ProteinSpecType
from magicsoup.genomics import Genomics
from magicsoup.util import random_genome

from tests.util import load_test_json

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


def _reverse_complement_rs(seq: str) -> str:
    return _lib.reverse_complement(seq)


def test_reverse_complement() -> None:
    seq = "ACTGG"
    res = _reverse_complement_rs(seq=seq)
    assert res == "CCAGT"


@pytest.mark.parametrize("seq, exp", _DATA)
def test_get_coding_regions(seq: str, exp: list[tuple[str, int]]) -> None:
    # 1 codon is too small to express p=0.01 domain types
    with pytest.warns(UserWarning):
        genomics = Genomics(n_dom_type_codons=1)

    seq = "".join(seq.replace("\n", "").split())
    res = genomics.get_coding_regions(seq=seq, min_cds_size=18, is_fwd=False)
    exp_starts, exp_stops = map(list, zip(*exp))

    assert len(res) == len(exp)
    assert {d[0] for d in res} == set(exp_starts)
    assert {d[1] for d in res} == set(exp_stops)

    for start, stop, is_fwd in res:
        idx = exp_starts.index(start)  # type: ignore
        assert start == exp_starts[idx]
        assert stop == exp_stops[idx]
        assert not is_fwd


def test_extract_domains() -> None:
    dom_type_map = {"AAA": 1, "GGG": 2, "CCC": 3}
    two_codon_map = {"ACTGAT": 1, "CTGTAT": 2, "CCGCGA": 3, "GGAATC": 4, "TGTCGA": 5}
    one_codon_map = {"ACT": 1, "CTG": 2, "CCG": 3, "GGA": 4, "TGT": 5}

    genomics = Genomics()
    genomics.domain_map = dom_type_map
    genomics.one_codon_map = one_codon_map
    genomics.two_codon_map = two_codon_map
    genomics.dom_type_size = len(next(iter(genomics.domain_map)))
    genomics.dom_size = genomics.dom_type_size + 5 * CODON_SIZE
    genomics._setup_rs()

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

    res = genomics.extract_domains(genome=genome, cdss=cdss)

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
    assert res[0][0][0][0] == (1, 2, 5, 1, 3)
    assert res[0][0][0][1] == 6
    assert res[0][0][0][2] == 6 + genomics.dom_size
    assert res[1][0][0][0] == (2, 3, 4, 2, 3)
    assert res[1][0][0][1] == 6
    assert res[1][0][0][2] == 6 + genomics.dom_size
    assert res[2][0][0][0] == (1, 3, 4, 5, 2)
    assert res[2][0][0][1] == 0
    assert res[2][0][0][2] == 0 + genomics.dom_size
    assert res[3][0][0][0] == (3, 3, 4, 2, 3)
    assert res[3][0][0][1] == 0
    assert res[3][0][0][2] == 0 + genomics.dom_size
    assert res[3][0][1][0] == (2, 1, 2, 3, 4)
    assert res[3][0][1][1] == 18
    assert res[3][0][1][2] == 18 + genomics.dom_size


def test_genomics() -> None:
    # fmt: off
    seq = """
    GACCAACGA CCGTCTTGA CCGCTGCGT TTCAACCGG 
    ACCTACCTA TCCCTCCTA AAGAACACG TCTTTCCGC 
    GTAGCTTCC CTGAGACTT ACTAGGGAA GCGTTTGGA 
    GAGTACGTG ATGGTTTGA TACAAGCAA TGGAGGATA 
    CACCAACAA AGATACTAT GCCCGCGGG TTCTATGTC 
    TCCATGGGA CAACTATGC TCGGAGTTA CAACTATGT 
    GGACCCCTC CTCAGAATC AAGGTAACG TTGAGCA
    """
    # fmt: on

    genomics = Genomics(
        start_codons=("TTG", "GTG", "ATG"),
        stop_codons=("TGA", "TAG", "TAA"),
        n_dom_type_codons=2,
    )
    genomics.domain_map = load_test_json("Genomics.domain_map.json")
    genomics.one_codon_map = load_test_json("Genomics.one_codon_map.json")
    genomics.two_codon_map = load_test_json("Genomics.two_codon_map.json")
    genomics._setup_rs()

    exp_proteome = [
        ([((1, 18, 44, 50, 783), 60, 81)], 14, 95, True),
        ([((1, 33, 5, 3, 163), 27, 48)], 58, 199, False),
        ([((1, 17, 33, 54, 955), 30, 51)], 119, 212, False),
    ]

    genome = "".join(seq.replace("\n", "").split())
    proteome = genomics.translate_genomes(genomes=[genome])[0]
    assert proteome == exp_proteome


def _sum_dom_type(data: list[list[ProteinSpecType]], type_: int) -> int:
    out = 0
    for cell in data:
        for protein, *_ in cell:
            for dom, *_ in protein:
                if dom[0] == type_:
                    out += 1
    return out


def test_genomics_domain_likelihoods() -> None:
    # 1=catalytic, 2=transporter, 3=regulatory
    # regulatory-only proteins get sorted out, so there is a bias towards
    # fewer regulatory domains

    # all same likelihood (while considering reg bias)
    kwargs = {"p_catal_dom": 0.1, "p_transp_dom": 0.1, "p_reg_dom": 0.1}
    genomics = Genomics(**kwargs)  # type: ignore
    genomes = [random_genome(s=500) for _ in range(1000)]
    proteomes_data = genomics.translate_genomes(genomes=genomes)

    n_catal = _sum_dom_type(proteomes_data, 1)
    n_trnsp = _sum_dom_type(proteomes_data, 2)
    n_reg = _sum_dom_type(proteomes_data, 3)
    n = n_catal + n_trnsp + n_reg
    assert abs(n_catal - n_trnsp) < 0.1 * n
    assert abs(n_trnsp - n_reg) < 0.2 * n

    # fewer catalytics (while considering reg bias)
    kwargs["p_catal_dom"] = 0.01
    genomics = Genomics(**kwargs)  # type: ignore
    genomes = [random_genome(s=500) for _ in range(1000)]
    proteomes_data = genomics.translate_genomes(genomes=genomes)

    n_catal = _sum_dom_type(proteomes_data, 1)
    n_trnsp = _sum_dom_type(proteomes_data, 2)
    n_reg = _sum_dom_type(proteomes_data, 3)
    n = n_catal + n_trnsp + n_reg
    assert n_trnsp - n_catal > 0.9 * n / 3
    assert n_reg - n_catal > 0.6 * n / 3

    # also fewer transporters (while considering reg bias)
    kwargs["p_transp_dom"] = 0.01
    genomics = Genomics(**kwargs)  # type: ignore
    genomes = [random_genome(s=500) for _ in range(1000)]
    proteomes_data = genomics.translate_genomes(genomes=genomes)

    n_catal = _sum_dom_type(proteomes_data, 1)
    n_trnsp = _sum_dom_type(proteomes_data, 2)
    n_reg = _sum_dom_type(proteomes_data, 3)
    n = n_catal + n_trnsp + n_reg
    assert n_reg - n_catal > 0.6 * n / 3
    assert n_reg - n_trnsp > 0.6 * n / 3
