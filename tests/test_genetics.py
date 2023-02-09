import pytest
import magicsoup as ms
from magicsoup.genetics import _get_coding_regions, _extract_domains
from magicsoup.constants import CODON_SIZE


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


def test_extract_domains():
    dom_type_map = {"AAA": 1, "GGG": 2, "CCC": 3}
    two_codon_map = {"ACTGAT": 1, "CTGTAT": 2, "CCGCGA": 3, "GGAATC": 4, "TGTCGA": 5}
    one_codon_map = {"ACT": 1, "CTG": 2, "CCG": 3, "GGA": 4, "TGT": 5}
    dom_type_size = len(next(iter(dom_type_map)))

    # fmt: off
    cdss = [
        "AGACAAAAACCGCGACTGTGTACTTAGACTAGACG",  # (1, 3, 2, 5, 1)
        "AGACTATAGCTAGAAGCCCCTGTCGATGTACTCCGTAGACG",  # (3, 5, 5, 1, 3)
        "AGACTAGGGCCGCGACCGGGACTGCTAGAAGCTAGACTAGACG",  # (2, 3, 3, 4, 2)
        "AAACTGTATCCGGGATGT",  # (1, 2, 3, 4, 5)
        "CCCCCGCGACCGGGACTGGGGGGAATCACTCTGCCG",  # (3, 3, 3, 4, 2) (2, 4, 1, 2, 3)
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
        dom_size=dom_type_size + 5 * CODON_SIZE,
        dom_type_map=dom_type_map,
        one_codon_map=one_codon_map,
        two_codon_map=two_codon_map,
    )

    assert len(res[0]) == 1
    assert len(res[1]) == 1
    assert len(res[2]) == 1
    assert len(res[3]) == 2
    assert res[0][0] == (1, 3, 2, 5, 1)
    assert res[1][0] == (2, 3, 3, 4, 2)
    assert res[2][0] == (1, 2, 3, 4, 5)
    assert res[3][0] == (3, 3, 3, 4, 2)
    assert res[3][1] == (2, 4, 1, 2, 3)



def test_genetics():
    # 1=catalytic, 2=transporter, 3=regulatory
    # regulatory-only proteins get sorted out, so there is a bias towards
    # fewer regulatory domains

    # all same likelihood (while considering reg bias)
    kwargs = {"p_catal_dom": 0.1, "p_transp_dom": 0.1, "p_reg_dom": 0.1}
    genetics = ms.Genetics(**kwargs)
    genomes = [ms.random_genome(s=500) for _ in range(1000)]
    proteomes = genetics.translate_genomes(genomes=genomes)
    
    n_catal = sum(ddd[0] == 1 for d in proteomes for dd in d for ddd in dd)
    n_trnsp = sum(ddd[0] == 2 for d in proteomes for dd in d for ddd in dd)
    n_reg = sum(ddd[0] == 3 for d in proteomes for dd in d for ddd in dd)
    n = n_catal + n_trnsp + n_reg
    assert abs(n_catal - n_trnsp) < 0.1 * n
    assert abs(n_trnsp - n_reg) < 0.2 * n

    # fewer catalytics (while considering reg bias)
    kwargs["p_catal_dom"] = 0.01
    genetics = ms.Genetics(**kwargs)
    genomes = [ms.random_genome(s=500) for _ in range(1000)]
    proteomes = genetics.translate_genomes(genomes=genomes)
    
    n_catal = sum(ddd[0] == 1 for d in proteomes for dd in d for ddd in dd)
    n_trnsp = sum(ddd[0] == 2 for d in proteomes for dd in d for ddd in dd)
    n_reg = sum(ddd[0] == 3 for d in proteomes for dd in d for ddd in dd)
    n = n_catal + n_trnsp + n_reg
    assert n_trnsp - n_catal > 0.9 * n / 3
    assert n_reg - n_catal > 0.6 * n / 3

    # also fewer transporters (while considering reg bias)
    kwargs["p_transp_dom"] = 0.01
    genetics = ms.Genetics(**kwargs)
    genomes = [ms.random_genome(s=500) for _ in range(1000)]
    proteomes = genetics.translate_genomes(genomes=genomes)
    
    n_catal = sum(ddd[0] == 1 for d in proteomes for dd in d for ddd in dd)
    n_trnsp = sum(ddd[0] == 2 for d in proteomes for dd in d for ddd in dd)
    n_reg = sum(ddd[0] == 3 for d in proteomes for dd in d for ddd in dd)
    n = n_catal + n_trnsp + n_reg
    assert n_reg - n_catal > 0.6 * n / 3
    assert n_reg - n_trnsp > 0.6 * n / 3

                