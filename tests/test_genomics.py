import math

import pytest
from magicsoup.config import CODON_SIZE
from magicsoup.genomics import DomainType, Genomics, ProteinSpec, _DomainTypeMap
from magicsoup.util import codons, random_genome

from tests.util import load_test_json


def test_domain_type_map() -> None:
    probs = {
        DomainType.CATALYTIC: 0.1,
        DomainType.TRANSPORTER: 0.1,
        DomainType.REGULATORY: 0.1,
    }
    stop_codons = ["TGA", "TAG", "TAA"]
    dom_type_map = _DomainTypeMap(
        dom_type_probs=probs,
        n_dom_type_codons=2,
        stop_codons=stop_codons,
    )

    possible_codons = codons(n=2, excl_codons=stop_codons)
    n_total = len(possible_codons)

    # check that probabilities are in binom 99% Wald interval
    for dom_type, prob in probs.items():
        exp = prob * n_total  # binom exp value
        exp_var = 2.56 * math.sqrt(exp * prob * (1 - prob))  # 99%
        act = len(dom_type_map.type_2_seqs[dom_type])
        assert exp - exp_var <= act <= exp + exp_var

    # check that mapping is bijective
    for dom_type, seqs in dom_type_map.type_2_seqs.items():
        for seq in seqs:
            token = dom_type_map.seq_2_token[seq]
            assert DomainType(token) is dom_type


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


@pytest.mark.parametrize("seq, exp", _DATA)
def test_get_coding_regions(seq: str, exp: list[tuple[str, int]]) -> None:
    # 1 codon is too small to express p=0.01 domain types
    with pytest.warns(UserWarning):
        genomics = Genomics(n_dom_type_codons=0)

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
    genomics.domain_type_map.seq_2_token = dom_type_map
    genomics.one_codon_map.seq_2_token = one_codon_map
    genomics.two_codon_map.seq_2_token = two_codon_map
    genomics.dom_type_size_nt = 3
    genomics.dom_size_nt = genomics.dom_type_size_nt + 5 * CODON_SIZE
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
    assert res[0][0][0][2] == 6 + genomics.dom_size_nt
    assert res[1][0][0][0] == (2, 3, 4, 2, 3)
    assert res[1][0][0][1] == 6
    assert res[1][0][0][2] == 6 + genomics.dom_size_nt
    assert res[2][0][0][0] == (1, 3, 4, 5, 2)
    assert res[2][0][0][1] == 0
    assert res[2][0][0][2] == 0 + genomics.dom_size_nt
    assert res[3][0][0][0] == (3, 3, 4, 2, 3)
    assert res[3][0][0][1] == 0
    assert res[3][0][0][2] == 0 + genomics.dom_size_nt
    assert res[3][0][1][0] == (2, 1, 2, 3, 4)
    assert res[3][0][1][1] == 18
    assert res[3][0][1][2] == 18 + genomics.dom_size_nt


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
    genomics.domain_type_map.seq_2_token = load_test_json("Genomics.domain_map.json")
    genomics.one_codon_map.seq_2_token = load_test_json("Genomics.one_codon_map.json")
    genomics.two_codon_map.seq_2_token = load_test_json("Genomics.two_codon_map.json")
    genomics._setup_rs()

    exp_proteome = [
        ([((1, 18, 44, 50, 783), 60, 81)], 14, 95, True),
        ([((1, 33, 5, 3, 163), 27, 48)], 58, 199, False),
        ([((1, 17, 33, 54, 955), 30, 51)], 119, 212, False),
    ]

    genome = "".join(seq.replace("\n", "").split())
    proteome = genomics.translate_genomes(genomes=[genome])[0]
    assert proteome == exp_proteome


def _sum_dom_type(data: list[list[ProteinSpec]], type_: int) -> int:
    out = 0
    for cell in data:
        for protein, *_ in cell:
            for dom, *_ in protein:
                if dom[0] == type_:
                    out += 1
    return out


def test_genomics_domain_likelihoods() -> None:
    n_genomes = 1000
    genome_size = 1000

    genomics = Genomics()  # type: ignore
    genomes = [random_genome(s=genome_size) for _ in range(n_genomes)]
    proteomes_data = genomics.translate_genomes(genomes=genomes)

    n_proteins = sum(len(d) for d in proteomes_data)
    n_domains = sum(len(dd[0]) for d in proteomes_data for dd in d)
    assert abs(n_domains / n_proteins - 2.0) < 0.5

    # 20% to 40% should have more than 2 domains
    n_multi_dom = 0
    for proteins in proteomes_data:
        for protein in proteins:
            if len(protein[0]) > 2:
                n_multi_dom += 1

    assert 0.2 <= n_multi_dom / n_proteins <= 0.4
