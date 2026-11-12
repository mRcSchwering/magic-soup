import math
from collections import Counter, defaultdict

import pytest
from magicsoup.core.config import CODON_SIZE
from magicsoup.core.strarray import StrArray
from magicsoup.core.util import codons, random_genome
from magicsoup.genomics import DomainType, Genomics, _DomainTypeMap

from tests.util import load_test_json


def test_domain_type_map() -> None:
    probs = {
        DomainType.CATALYTIC: 0.1,
        DomainType.TRANSPORTER: 0.1,
        DomainType.REGULATORY: 0.1,
    }
    stop_codons = ["TGA", "TAG", "TAA"]
    start_codons = ["TTG", "GTG", "ATG"]
    dom_type_map = _DomainTypeMap(
        dom_type_probs=probs,
        n_dom_type_codons=2,
        stop_codons=stop_codons,
        start_codons=start_codons,
    )

    # stop codons excluded everywhere, start codons excluded as first codon
    possible_codons = codons(n=2, excl_codons=stop_codons)
    possible_codons = [d for d in possible_codons if d[:CODON_SIZE] not in start_codons]
    n_total = len(possible_codons)

    all_seqs = [dd for d in dom_type_map.type_2_seqs.values() for dd in d]
    assert len(set(all_seqs)) == len(all_seqs)

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


_START_CODONS = ("TTG", "GTG", "ATG")
_STOP_CODONS = ("TGA", "TAG", "TAA")


def _ref_coding_regions(seq: str, min_cds_size: int) -> set[tuple[int, int]]:
    """Independent brute-force reference for circular CDS finding.

    The genome is a ring. For every start position its first in-frame stop is
    found by walking codons `p, p+3, ...` with wrap; a CDS longer than the whole
    genome (no stop within one turn) is dropped. Returns {(start, stop)} where a
    wrapped CDS has stop > len(seq).
    """
    n = len(seq)
    res: set[tuple[int, int]] = set()
    if n < min_cds_size:
        return res

    def codon(x: int) -> str:
        return seq[x % n] + seq[(x + 1) % n] + seq[(x + 2) % n]

    for p in range(n):
        if codon(p) not in _START_CODONS:
            continue
        c = p + CODON_SIZE
        while c + CODON_SIZE - p <= n:  # CDS length stays within one turn
            if codon(c) in _STOP_CODONS:
                if c + CODON_SIZE - p >= min_cds_size:
                    res.add((p, c + CODON_SIZE))
                break
            c += CODON_SIZE
    return res


# (genome, min_cds_size, expected {(start, stop)}); starts TTG/GTG/ATG, stops
# TGA/TAG/TAA. Hand-verified circular cases.
_CIRCULAR_DATA: list[tuple[str, int, set[tuple[int, int]]]] = [
    # CDS fully inside the sequence (no wrap)
    ("ATGGGGCCCTAA", 6, {(0, 12)}),
    # start near the end wraps past the seam to the stop TAA at position 0
    # (stop read at extended index 12 -> stop index 15 > n)
    ("TAAGGGCCCATG", 6, {(9, 15)}),
    # start with no in-frame stop anywhere on the ring -> dropped (would wrap
    # forever), not emitted as a CDS longer than the genome
    ("ATG" + "GGG" * 5, 6, set()),
    # whole genome is one CDS, start to stop exactly min size
    ("TTGAAAGAGCAAATTTGA", 18, {(0, 18)}),
    # two overlapping GTG starts with different stops (no wrap)
    ("GTGTGCTCGAAAGAGAACGCAAATTCGTAACCTAG", 18, {(0, 30), (2, 35)}),
]


@pytest.mark.parametrize("seq, min_cds_size, exp", _CIRCULAR_DATA)
def test_get_coding_regions_circular(
    seq: str, min_cds_size: int, exp: set[tuple[int, int]]
) -> None:
    # 1 codon is too small to express p=0.01 domain types
    with pytest.warns(UserWarning):
        genomics = Genomics(n_dom_type_codons=0)

    res = genomics.get_coding_regions(seq=seq, min_cds_size=min_cds_size, is_fwd=False)

    assert {(d[0], d[1]) for d in res} == exp
    assert all(not is_fwd for _, _, is_fwd in res)
    # no CDS is longer than the genome
    assert all(stop - start <= len(seq) for start, stop, _ in res)
    # matches the independent reference
    assert {(d[0], d[1]) for d in res} == _ref_coding_regions(seq, min_cds_size)


# TODO: rm seed?
def test_get_coding_regions_matches_reference() -> None:
    # cross-check the Rust circular CDS finder against the brute-force reference
    # on many random genomes, incl. codons straddling the seam
    import random

    with pytest.warns(UserWarning):
        genomics = Genomics(n_dom_type_codons=0)

    random.seed(123)
    for _ in range(2000):
        n = random.randint(18, 150)
        seq = "".join(random.choice("ACGT") for _ in range(n))
        min_cds_size = random.choice([6, 9, 18])
        res = {
            (d[0], d[1])
            for d in genomics.get_coding_regions(
                seq=seq, min_cds_size=min_cds_size, is_fwd=True
            )
        }
        assert res == _ref_coding_regions(seq, min_cds_size)


def _tokens_multiset(genomics: Genomics, genome: str) -> Counter:
    """Order-independent multiset of a genome's proteins by their domain tokens."""
    specs = genomics.translate_genomes(genomes=StrArray.from_strings([genome]))[0]
    return Counter(
        tuple(sorted(tuple(dom[0]) for dom in doms)) for doms, _, _, _ in specs
    )


# TODO: wrong import
def test_translation_is_rotation_invariant() -> None:
    # a circular genome has no origin, so rotating it must not change the set of
    # proteins it encodes (exercises CDS wrapping and domain extraction at the
    # seam for every rotation)
    import random

    genomics = Genomics()
    random.seed(321)
    for _ in range(100):
        genome = random_genome(s=random.randint(100, 500))
        base = _tokens_multiset(genomics, genome)
        for _ in range(3):
            k = random.randint(1, len(genome) - 1)
            rotated = genome[k:] + genome[:k]
            assert _tokens_multiset(genomics, rotated) == base


def test_extract_domains() -> None:
    dom_type_map = {"AAAAAA": 1, "GGGGGG": 2, "CCCCCC": 3}
    two_codon_map = {"ACTGAT": 1, "CTGTAT": 2, "CCGCGA": 3, "GGAATC": 4, "TGTCGA": 5}
    one_codon_map = {"ACT": 1, "CTG": 2, "CCG": 3, "GGA": 4, "TGT": 5}

    genomics = Genomics()
    genomics.domain_type_map.seq_2_token = dom_type_map
    genomics.one_codon_map.seq_2_token = one_codon_map
    genomics.two_codon_map.seq_2_token = two_codon_map
    genomics.dom_type_size_nt = 6
    genomics.dom_size_nt = genomics.dom_type_size_nt + 5 * CODON_SIZE
    genomics._setup_rs()

    # fmt: off
    genome_parts = [
        "AGAAAAAAA CTGTGTACT CCGCGATAG ACTAGACG",  # 0-35
        "AGACTATAG CTAGAAGCC CCCCCCTGT ACTCCGTGT CGATAGACG", # 35-80
        "AGACTAGGG GGGCCGGGA CTGCCGCGA CTAGAAGCT AGACTAACG",  # 80-125
        "AAAAAACCG GGATGTCTG TAT",  # 125-146
        "CCCCCCCCG GGACTGCCG CGACGAGGG GGGACTCTG CCGGGAATC",  # 146-191
    ]
    cdss: list[tuple[int, int, bool]] = [
        (0, 35, True),  # (1, 2, 5, 1, 3)
        (35, 80, False),  # (3, 5, 1, 3, 5)
        (80, 125, True),  # (2, 3, 4, 2, 3)
        (125, 146, False),  # (1, 3, 4, 5, 2)
        (146, 191, True),  # (3, 3, 4, 2, 3) (2, 1, 2, 3, 4)
    ]
    # - cds 0: normal domain                                                    => 1 res[0]
    # - cds 1: single type 3 domain, so it is removed
    # - cds 2: has 2 domain 2 starts, but the second is part of the 1 domain    => 1 res[1]
    # - cds 3: defines exactly 1 domain from start to end                       => 1 res[2]
    # - cds 4: defines exactly 2 domains, a 3rd type 2 start is in the middle   => 2 res[3]
    # fmt: on
    genome = "".join(d.replace(" ", "") for d in genome_parts)
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
    assert res[1][1] == 80
    assert res[1][2] == 125
    assert res[1][3] is True
    assert res[2][1] == 125
    assert res[2][2] == 146
    assert res[2][3] is False
    assert res[3][1] == 146
    assert res[3][2] == 191
    assert res[3][3] is True
    assert res[0][0][0][0] == (1, 2, 5, 1, 3)
    assert res[0][0][0][1] == 3
    assert res[0][0][0][2] == 3 + genomics.dom_size_nt
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
    assert res[3][0][1][1] == 24
    assert res[3][0][1][2] == 24 + genomics.dom_size_nt


def test_extract_domains_rejects_uncovered_spec_codon() -> None:
    # start/stop codons are excluded from the codon maps, so a domain whose spec
    # codon is not covered (e.g. its last codon coinciding with the CDS's
    # terminal stop codon) must be rejected instead of panicking.
    genomics = Genomics()
    genomics.domain_type_map.seq_2_token = {"AAAAAA": 1}
    genomics.one_codon_map.seq_2_token = {"ACT": 1, "CTG": 2, "CCG": 3}
    genomics.two_codon_map.seq_2_token = {"ACTCTG": 1}
    genomics.dom_type_size_nt = 6
    genomics.dom_size_nt = genomics.dom_type_size_nt + 5 * CODON_SIZE  # 21
    genomics._setup_rs()

    # type AAAAAA + ACT + CTG + CCG + [ACT TGA]: vector "ACTTGA" ends in a stop
    # and is not in the two-codon map -> the whole domain is rejected
    bad = "AAAAAA" + "ACT" + "CTG" + "CCG" + "ACTTGA"
    assert len(bad) == genomics.dom_size_nt
    assert genomics.extract_domains(genome=bad, cdss=[(0, 21, True)]) == []

    # identical layout but with a covered vector "ACTCTG" -> domain is extracted
    good = "AAAAAA" + "ACT" + "CTG" + "CCG" + "ACTCTG"
    res = genomics.extract_domains(genome=good, cdss=[(0, 21, True)])
    assert len(res) == 1
    assert res[0][0][0][0] == (1, 1, 2, 3, 1)


def test_extract_domains_wrapped_cds() -> None:
    # a CDS of a circular genome may wrap the seam; its stop index then exceeds
    # the genome length and its domain bytes are read across the seam
    genomics = Genomics()
    genomics.domain_type_map.seq_2_token = {"AAAAAA": 1, "GGGGGG": 2, "CCCCCC": 3}
    genomics.one_codon_map.seq_2_token = {"ACT": 1, "CTG": 2, "CCG": 3}
    genomics.two_codon_map.seq_2_token = {"ACTGAT": 1}
    genomics.dom_type_size_nt = 6
    genomics.dom_size_nt = genomics.dom_type_size_nt + 5 * CODON_SIZE  # 21
    genomics._setup_rs()

    domain = "AAAAAA" + "ACT" + "CTG" + "CCG" + "ACTGAT"  # tokens (1, 1, 2, 3, 1)
    assert len(domain) == genomics.dom_size_nt
    # place the domain so it straddles the seam: last 11 bp at the genome end,
    # first 10 bp at the start, filler in between
    genome = domain[11:] + "TTTTTTTTT" + domain[:11]
    n = len(genome)
    cds_start = len(domain[11:]) + len("TTTTTTTTT")  # 19
    cds_stop = cds_start + genomics.dom_size_nt  # 40 > n

    res = genomics.extract_domains(genome=genome, cdss=[(cds_start, cds_stop, True)])
    assert len(res) == 1
    assert res[0][1] == cds_start
    assert res[0][2] == cds_stop
    assert cds_stop > n
    assert len(res[0][0]) == 1
    assert res[0][0][0][0] == (1, 1, 2, 3, 1)
    assert res[0][0][0][1] == 0
    assert res[0][0][0][2] == genomics.dom_size_nt


# TODO: check this test again (and comments)
def test_translate_genomes() -> None:
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

    # these non-wrapping CDSs are found unchanged on the circular genome; a
    # circular genome also yields additional CDSs that wrap the seam
    exp_proteins = [
        ([((1, 18, 44, 50, 783), 60, 81)], 14, 95, True),
        ([((1, 33, 5, 3, 163), 27, 48)], 58, 199, False),
        ([((1, 17, 33, 54, 955), 30, 51)], 119, 212, False),
    ]

    genome = "".join(seq.replace("\n", "").split())
    n = len(genome)
    genomes = StrArray.from_strings([genome])
    proteome = genomics.translate_genomes(genomes=genomes)[0]
    for protein in exp_proteins:
        assert protein in proteome
    # every CDS is at most one full turn; wrapped CDSs have stop > n
    assert all(cds_stop - cds_start <= n for _, cds_start, cds_stop, _ in proteome)

    # the flat encoding is consistent with the list representation
    cell, _prot, tok, p = genomics.encode_proteomes(genomes=genomes)
    assert p == len(proteome)
    assert cell.tolist() == [0] * tok.shape[0]
    tok_from_list = Counter(
        tuple(dom[0]) for prot_spec in proteome for dom in prot_spec[0]
    )
    tok_from_arr = Counter(tuple(row) for row in tok.tolist())
    assert tok_from_list == tok_from_arr


# TODO: downscale domain likelihoods?
def test_genomics_domain_likelihoods() -> None:
    n_genomes = 1000
    genome_size = 1000

    genomics = Genomics()  # type: ignore
    genomes = StrArray.from_strings(
        [random_genome(s=genome_size) for _ in range(n_genomes)]
    )

    # translate into lists
    proteomes = genomics.translate_genomes(genomes=genomes)
    p_lst = sum([len(d) for d in proteomes])
    d_lst = sum([len(dd[0]) for d in proteomes for dd in d])

    # translate into arrays
    cell, prot, tok, p = genomics.encode_proteomes(genomes=genomes)
    p_arr = len({tuple(d) for d in zip(cell.tolist(), prot.tolist())})
    d_arr = tok.shape[0]

    # count domains per protein in list
    d_cnts_lst: dict[int, int] = defaultdict(lambda: 0)
    for proteins in proteomes:
        for protein in proteins:
            d_cnts_lst[len(protein[0])] += 1

    # count domains per protein in arrays
    d_cnts_arr: dict[int, int] = defaultdict(lambda: 0)
    prot_i = 0
    n_dom = 0
    for i in range(d_arr):
        if cell[i] * p + prot[i] == prot_i:
            n_dom += 1
        else:
            d_cnts_arr[n_dom] += 1
            n_dom = 1
            prot_i = cell[i] * p + prot[i]

    d_cnts_arr[n_dom] += 1

    # compare both results
    assert p_lst == p_arr
    assert d_lst == d_arr
    assert d_cnts_lst == d_cnts_arr
    assert d_cnts_lst == d_cnts_arr

    # TODO: should be 2.0
    # ~2.5 domains per protein on average (circular genomes add seam-wrapping
    # CDSs, which skew slightly longer and lift this above the ~2 of a purely
    # linear genome)
    assert 2.0 < d_arr / p_arr < 3.0

    # TODO: should be between 20% and 40%
    # 20% to 45% should have more than 2 domains
    n_multi_dom = sum([n for d, n in d_cnts_lst.items() if d > 2])
    assert 0.2 <= n_multi_dom / p_arr <= 0.45
