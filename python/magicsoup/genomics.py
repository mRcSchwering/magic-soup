import random
import warnings

from magicsoup import _lib  # type: ignore
from magicsoup.constants import CODON_SIZE, ProteinSpecType
from magicsoup.util import codons


def _get_n(p: float, s: int, name: str) -> int:
    n = int(p * s)
    if n == 0 and p > 0.0:
        warnings.warn(
            f"There will be no {name}."
            f" Increase dom_type_size to accomodate low probabilities of having {name}."
        )
    return n


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
        self.one_codon_map = {d: i + 1 for i, d in enumerate(self._get_single_codons())}

        # the second codon is allowed to be a stop codon
        self.two_codon_map = {d: i + 1 for i, d in enumerate(self._get_double_codons())}

        # inverse maps for genome generation
        self.idx_2_one_codon = {v: k for k, v in self.one_codon_map.items()}
        self.idx_2_two_codon = {v: k for k, v in self.two_codon_map.items()}

        self._setup_genome_translator()

    def _setup_genome_translator(self) -> None:
        self.genome_translator = _lib.GenomeTranslator(
            self.start_codons,
            self.stop_codons,
            self.domain_map,
            self.one_codon_map,
            self.two_codon_map,
            self.dom_size,
            self.dom_type_size,
        )

    def translate_genomes(self, genomes: list[str]) -> list[list[ProteinSpecType]]:
        if len(genomes) < 1:
            return []
        return self.genome_translator(genomes)

    def _get_single_codons(self) -> list[str]:
        seqs = codons(n=1)
        seqs = [d for d in seqs if d not in self.stop_codons]
        return seqs

    def _get_double_codons(self) -> list[str]:
        seqs = codons(n=2)
        seqs = [d for d in seqs if d[:CODON_SIZE] not in self.stop_codons]
        return seqs
