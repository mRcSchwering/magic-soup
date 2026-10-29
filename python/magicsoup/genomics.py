import random
import warnings
from collections.abc import Iterable
from enum import IntEnum
from itertools import product
from typing import TypedDict

import numpy as np

from magicsoup import rs
from magicsoup.config import CODON_SIZE
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
        # all possible sequences
        if allow_last_stop:
            seqs0 = codons(n=n_codons - 1, excl_codons=list(stop_codons))
            seqs1 = codons(n=1)
            seqs = [a + b for a, b in product(seqs0, seqs1)]
        else:
            seqs = codons(n=n_codons, excl_codons=list(stop_codons))

        # tokens 1...n+1 (none token 0 gets no sequence)
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
        seqs = codons(n=n_dom_type_codons, excl_codons=stop_codons)
        n_seqs = len(seqs)
        dom_type_2_n = {k: int(p * n_seqs) for k, p in dom_type_probs.items()}

        if sum(dom_type_2_n.values()) > n_seqs:
            raise ValueError("Domain type probabilities exceed 1")
        for dom_type, n in dom_type_2_n.items():
            if n == 0 and dom_type_probs[dom_type] > 0:
                warnings.warn(
                    f"{n_dom_type_codons} domain type codons"
                    f" are too small to encode {dom_type}"
                )

        # domain types -> nt sequences
        self.type_2_seqs: dict[DomainType, list[str]] = {k: [] for k in dom_type_probs}

        random.shuffle(seqs)
        for dom_type, n in dom_type_2_n.items():
            for _ in range(n):
                self.type_2_seqs[dom_type].append(seqs.pop())

        # nt sequence -> domain type token
        self.seq_2_token = {d: k.value for k, v in self.type_2_seqs.items() for d in v}


type CdsSpec = tuple[int, int, bool]
type DomainTokens = tuple[int, int, int, int, int]
type DomainSpec = tuple[DomainTokens, int, int]
type ProteinSpec = tuple[list[DomainSpec], int, int, bool]


class GenomicsKwargs(TypedDict, total=False):
    start_codons: tuple[str, ...]
    stop_codons: tuple[str, ...]
    p_catal_dom: float
    p_transp_dom: float
    p_reg_dom: float
    n_dom_type_codons: int


class Genomics:
    """
    Defines simulation genetics and translates genomes.

    Arguments:
        start_codons: Codons which start a coding sequence
        stop_codons: Codons which stop a coding sequence
        p_catal_dom: Chance of encountering a [CatalyticDomain][magicsoup.biology.CatalyticDomain]
            in a random nucleotide sequence.
        p_transp_dom: Chance of encountering a [TransporterDomain][magicsoup.biology.TransporterDomain]
            in a random nucleotide sequence.
        p_reg_dom: Chance of encountering a [RegulatoryDomain][magicsoup.biology.RegulatoryDomain]
            in a random nucleotide sequence.
        n_dom_type_codons: Number of codons that encode the domain type:
            [CatalyticDomain][magicsoup.biology.CatalyticDomain],
            [TransporterDomain][magicsoup.biology.TransporterDomain], or
            [RegulatoryDomain][magicsoup.biology.RegulatoryDomain].

    Translation happens only within coding sequences (CDSs).
    A CDS starts at every start codon and ends with the first in-frame encountered stop codon.
    Un-stopped CDSs are discarded.
    Both the forward and reverse complement of the nucleotide sequence are considered.
    Each CDS represents one [Protein][magicsoup.biology.Protein].
    All domains found within a CDS will be added as domains to that protein.
    """

    def __init__(
        self,
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        p_catal_dom: float = 0.05,
        p_transp_dom: float = 0.05,
        p_reg_dom: float = 0.05,
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

        # TODO: domain structure logic is currently spread out over
        #       Rust in proteomics (needs to know which token refers to which value),
        #       and genomics (needs to extract tokens in correct order)
        #       and here (needs to build sequences in correct order)
        #       this should somehow be centralized

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

    def translate_genomes(self, genomes: list[str]) -> list[list[ProteinSpec]]:
        """Get proteomes in form of domain specifications from genomes"""
        if len(genomes) == 0:
            return []
        return self.rs.translate_genomes(genomes=genomes)

    def translate_genomes2(
        self, genomes: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Get proteomes in form of domain specifications from genomes"""
        if len(genomes) == 0:
            return (np.zeros(0), np.zeros(0), np.zeros((0, 5)), 0)
        return self.rs.translate_genomes_new(genomes=genomes)

    def get_coding_regions(
        self,
        seq: str,
        min_cds_size: int,
        is_fwd: bool,
    ) -> list[CdsSpec]:
        """Get CDS markers for a genome"""
        return self.rs.get_coding_regions(
            seq=seq, min_cds_size=min_cds_size, is_fwd=is_fwd
        )

    def extract_domains(
        self,
        genome: str,
        cdss: list[CdsSpec],
    ) -> list[ProteinSpec]:
        """Get proteome in form of domain specifications for a genome and CDS markers"""
        return self.rs.extract_domains(genome=genome, cdss=cdss)

    def generate_domain_seq(self, domain_type: DomainType, tokens: list[int]) -> str:
        """
        Generate a nucleotide sequence encoding a domain:

        Arguments:
            domain_type: Which domain type to encode
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
        i0_seq = random.choice(codons(n=1, excl_codons=self.stop_codons))
        i1_seq = random.choice(codons(n=1, excl_codons=self.stop_codons))
        i2_seq = random.choice(codons(n=1, excl_codons=self.stop_codons))
        i3_seq = token_2_two_codon[random.choice(tokens)]

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
        pad_size = n_pad_nts // n_pads
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
