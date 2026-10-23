import math
import random
import warnings
from collections.abc import Iterable
from enum import IntEnum
from typing import TypedDict

from magicsoup import rs
from magicsoup.constants import CODON_SIZE, ProteinSpecType
from magicsoup.util import codons, random_genome


class DomainType(IntEnum):
    CATALYTIC = 1
    TRANSPORTER = 2
    REGULATORY = 3

    def __str__(self) -> str:
        return self.name.capitalize()


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


class _DomainTypeMap:

    def __init__(
        self,
        dom_type_probs: dict[DomainType, float],
        n_dom_type_codons: int,
        stop_codons: list[str],
    ) -> None:
        # generate all possible valid sequences for domain types
        pool = codons(n=n_dom_type_codons, excl_codons=stop_codons)
        random.shuffle(pool)

        # domain types -> nt sequences
        self.type_2_seqs: dict[DomainType, list[str]] = {k: [] for k in dom_type_probs}

        while len(pool) > 0:
            for dom_type, p in dom_type_probs.items():
                if random.random() < p:
                    self.type_2_seqs[dom_type].append(pool.pop())

        for dom_type, seqs in self.type_2_seqs.items():
            if len(seqs) == 0:
                warnings.warn(
                    f"There will be no {dom_type} domains."
                    f" Increase the number of codons encoding domain type or its probability"
                )

        # nt sequence -> domain type token
        self.seq_2_token = {d: k.value for k, v in self.type_2_seqs.items() for d in v}


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
        # domain type definition (n domain type codons)
        # 3 x 1-codon specifications (3 simple tokens)
        # 1 x 2-codon specification (1 complex token)
        # a domain can begin and end with start/stop codons
        # so min CDS size = dom_size
        self.dom_size_nt = (n_dom_type_codons + 5) * CODON_SIZE
        self.dom_type_size_nt = n_dom_type_codons * CODON_SIZE

        # setup nt sequences which start a domain
        domain_type_probs = {
            DomainType.CATALYTIC: p_catal_dom,
            DomainType.TRANSPORTER: p_transp_dom,
            DomainType.REGULATORY: p_reg_dom,
        }
        self.domain_type_map = _DomainTypeMap(
            dom_type_probs=domain_type_probs,
            n_dom_type_codons=n_dom_type_codons,
            stop_codons=self.stop_codons,
        )

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
            domain_type_2_token=self.domain_type_map.seq_2_token,
            one_codon_2_token=self.one_codon_map.seq_2_token,
            two_codon_2_token=self.two_codon_map.seq_2_token,
            dom_size_nt=self.dom_size_nt,
            dom_type_size_nt=self.dom_type_size_nt,
        )

    def translate_genomes(self, genomes: list[str]) -> list[list[ProteinSpecType]]:
        """Get proteomes in form of domain specifications from genomes"""
        if len(genomes) == 0:
            return []
        return self.rs.translate_genomes(genomes=genomes)

    def get_coding_regions(
        self,
        seq: str,
        min_cds_size: int,
        is_fwd: bool,
    ) -> list[tuple[int, int, bool]]:
        """Get CDS marker from genome"""
        return self.rs.get_coding_regions(
            seq=seq, min_cds_size=min_cds_size, is_fwd=is_fwd
        )

    def extract_domains(
        self,
        genome: str,
        cdss: list[tuple[int, int, bool]],
    ) -> list[ProteinSpecType]:
        """Get proteome in form of domain specifications from genome and CDS markers"""
        return self.rs.extract_domains(genome=genome, cdss=cdss)

    def generate_domain_seq(self, domain_type: DomainType, tokens: list[int]) -> str:
        """
        Generate a nucleotide sequence for a domain:

        Arguments:
            domain_type: Which domain it should be
            tokens: Tokens which encode the main entity of the domain

        For catalytic domains the main entity is the reaction, for transporters the
        transported molecule, for regulatory domains the effector molecule.
        """
        token_2_two_codon = self.two_codon_map.token_2_seq
        domtype_2_seqs = self.domain_type_map.type_2_seqs

        # domain type: catal/trnsp/reg
        # idx0: Vmax/hill (1 codon, no stop)
        # idx1: Km (1 codon, no stop)
        # idx2: sign/orientation (1 codon, no stop)
        # idx3: reaction/molecule/effector (2 codon, 2nd can be stop)
        dom_seq = random.choice(domtype_2_seqs[domain_type])
        i0_seq = random_genome(s=CODON_SIZE, excl=self.stop_codons)
        i1_seq = random_genome(s=CODON_SIZE, excl=self.stop_codons)
        i2_seq = random_genome(s=CODON_SIZE, excl=self.stop_codons)
        i3 = random.choice(tokens)
        i3_seq = token_2_two_codon[i3]

        return dom_seq + i0_seq + i1_seq + i2_seq + i3_seq

    def generate_genome(self, cdss: list[list[str]], n_pad_nts: int = 10) -> str:
        """
        Generate a genome based on CDSs

        Arguments:
            cdss: list of proteins, each a list of CDSs
            n_pad_nts: number of padding nucleotides between all CDSs

        The generated genome has at least the provided CDSs but it may have more.
        """
        n_pads = len(cdss) + 1
        pad_size = math.floor(n_pad_nts / n_pads)
        remaining_nts = n_pad_nts - n_pads * pad_size

        excl_cdss = self.start_codons + self.stop_codons
        pads = [random_genome(s=pad_size, excl=excl_cdss) for _ in range(n_pads)]
        tail = random_genome(s=remaining_nts, excl=excl_cdss)

        parts: list[str] = []
        for cds in cdss:
            parts.append(pads.pop())
            parts.append(random.choice(self.start_codons))
            parts.extend([d for d in cds])
            parts.append(random.choice(self.stop_codons))
        parts.append(pads.pop())
        parts.append(tail)

        return "".join(parts)

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
