import random
import warnings
from collections.abc import Iterable
from typing import TypedDict

from magicsoup import rs
from magicsoup.chemistry import Reaction
from magicsoup.constants import CODON_SIZE, ProteinSpecType
from magicsoup.util import codons, random_genome


def _get_n(p: float, s: int, name: str) -> int:
    n = int(p * s)
    if n == 0 and p > 0.0:
        warnings.warn(
            f"There will be no {name}."
            f" Increase dom_type_size to accomodate low probabilities of having {name}."
        )
    return n


class _CodonMap:

    def __init__(
        self, n_codons: int, stop_codons: Iterable[str], allow_last_stop: bool = False
    ) -> None:
        seqs = codons(n=n_codons)
        check_nts = CODON_SIZE * n_codons
        if allow_last_stop:
            check_nts -= CODON_SIZE

        seqs = [d for d in seqs if d[:check_nts] not in stop_codons]
        self.seq_2_token = {d: i + 1 for i, d in enumerate(seqs)}
        self.token_2_seq = {v: k for k, v in self.seq_2_token.items()}
        self.encoding_size = max(self.seq_2_token.values())


class GenomicsKwargs(TypedDict, total=False):
    start_codons: tuple[str, ...]
    stop_codons: tuple[str, ...]
    p_catal_dom: float
    p_transp_dom: float
    p_reg_dom: float
    n_dom_type_codons: int


# TODO: defaults should lead to ~20% 3 domains
#       so that proteins have decent promiscuity


class Genomics:

    def __init__(
        self,
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        p_catal_dom: float = 0.01,
        p_transp_dom: float = 0.01,
        p_reg_dom: float = 0.01,
        n_dom_type_codons: int = 2,
    ):
        if any(len(d) != CODON_SIZE for d in start_codons):
            raise ValueError(f"Not all start codons are of length {CODON_SIZE}")
        if any(len(d) != CODON_SIZE for d in stop_codons):
            raise ValueError(f"Not all stop codons are of length {CODON_SIZE}")
        if len(set(start_codons) & set(stop_codons)) > 0:
            raise ValueError(
                "Overlapping start and stop codons:"
                f" {','.join(str(d) for d in set(start_codons) & set(stop_codons))}"
            )
        if p_catal_dom + p_transp_dom + p_reg_dom > 1.0:
            raise ValueError(
                "p_catal_dom, p_transp_dom, p_reg_dom together must not be greater 1.0"
            )

        self.start_codons = list(start_codons)
        self.stop_codons = list(stop_codons)

        # Domain structure:
        # domain type definition (domain type codons)
        # 3 x 1-codon specifications (3 simple tokens)
        # 1 x 2-codon specification (1 complex token)
        # a domain can begin and end with start/stop codons
        # so min CDS size = dom_size
        self.dom_size = (n_dom_type_codons + 5) * CODON_SIZE
        self.dom_type_size = n_dom_type_codons * CODON_SIZE

        # setup domain type definitions
        # sequences including stop codons are useless, since they would terminate the CDS
        sets = codons(n=n_dom_type_codons, excl_codons=self.start_codons)
        random.shuffle(sets)
        n = len(sets)

        n_catal_doms = _get_n(p=p_catal_dom, s=n, name="catalytic domains")
        n_transp_doms = _get_n(p=p_transp_dom, s=n, name="transporter domains")
        n_reg_doms = _get_n(p=p_reg_dom, s=n, name="allosteric domains")

        # 1=catalytic, 2=transporter, 3=regulatory
        self.domain_types: dict[int, list[str]] = {}
        self.domain_types[1] = sets[:n_catal_doms]
        del sets[:n_catal_doms]
        self.domain_types[2] = sets[:n_transp_doms]
        del sets[:n_transp_doms]
        self.domain_types[3] = sets[:n_reg_doms]
        del sets[:n_reg_doms]

        self.domain_map = {d: k for k, v in self.domain_types.items() for d in v}

        # pre-mature stop codons are excluded
        self.one_codon_map = _CodonMap(n_codons=1, stop_codons=stop_codons)

        # because at domain end the second codon is allowed to be a stop codon
        self.two_codon_map = _CodonMap(
            n_codons=2, stop_codons=stop_codons, allow_last_stop=True
        )

        # setup rust class
        self._setup_rs()

    def _setup_rs(self) -> None:
        self.rs = rs.Genomics(
            start_codons=self.start_codons,
            stop_codons=self.stop_codons,
            domain_map=self.domain_map,
            one_codon_map=self.one_codon_map.seq_2_token,
            two_codon_map=self.two_codon_map.seq_2_token,
            dom_size=self.dom_size,
            dom_type_size=self.dom_type_size,
        )

    def translate_genomes(self, genomes: list[str]) -> list[list[ProteinSpecType]]:
        if len(genomes) == 0:
            return []
        return self.rs.translate_genomes(genomes=genomes)

    def get_coding_regions(
        self,
        seq: str,
        min_cds_size: int,
        is_fwd: bool,
    ) -> list[tuple[int, int, bool]]:
        return self.rs.get_coding_regions(
            seq=seq, min_cds_size=min_cds_size, is_fwd=is_fwd
        )

    def extract_domains(
        self,
        genome: str,
        cdss: list[tuple[int, int, bool]],
    ) -> list[ProteinSpecType]:
        return self.rs.extract_domains(genome=genome, cdss=cdss)

    def get_catalytic_domain_seq(
        self,
        reaction: Reaction,
        react_2_tokens: dict[Reaction, list[int]],
    ) -> str:
        """Generate a nucleotide sequence for a catalytic domain"""
        token_2_two_codon = self.one_codon_map.token_2_seq
        domain_types = self.domain_types

        # catalytic domain type: 1
        # idx0: Vmax (1 codon, no stop)
        # idx1: Km (1 codon, no stop)
        # idx2: direction (1 codon, no stop)
        # idx3: reaction (2 codon, 2nd can be stop)
        dom_seq = random.choice(domain_types[1])
        i0_seq = random_genome(s=CODON_SIZE, excl=self.stop_codons)
        i1_seq = random_genome(s=CODON_SIZE, excl=self.stop_codons)
        i2_seq = random_genome(s=CODON_SIZE, excl=self.stop_codons)
        i3 = random.choice(react_2_tokens[reaction])
        i3_seq = token_2_two_codon[i3]

        return dom_seq + i0_seq + i1_seq + i2_seq + i3_seq

    def _get_single_codons(self) -> list[str]:
        seqs = codons(n=1)
        seqs = [d for d in seqs if d not in self.stop_codons]
        return seqs

    def _get_double_codons(self) -> list[str]:
        seqs = codons(n=2)
        seqs = [d for d in seqs if d[:CODON_SIZE] not in self.stop_codons]
        return seqs

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
