from typing import Optional
from util import (
    ALL_NTS,
    CODON_SIZE,
    variants,
    reverse_complement,
    weight_map_fact,
)


class Information:
    """
    General piece of information that can be handled by a domain.
    Could be a molecule, or some action done by a cell.
    Gets its concrete meaning together with domain.
    """

    def __init__(self, name: str, energy: float):
        self.name = name
        self.energy = energy


MA = Information("A", 5)  # molceule A
MB = Information("B", 4)  # molceule B
MC = Information("C", 3)  # molceule C
MD = Information("D", 3)  # molceule D
ME = Information("E", 2)  # molceule E
MF = Information("F", 1)  # molceule F

MOLECULES = [MA, MB, MC, MD, ME, MF]

CM = Information("CM", -2)  # cell migration
CR = Information("CR", -4)  # cell replication
DR = Information("DR", -1)  # DNA repair
TP = Information("TP", -1)  # transposon

ACTIONS = [CM, CR, DR, TP]


class Domain:
    """Baseclass for domains"""

    def __init__(
        self,
        info: Optional[Information] = None,
        sigma_a: Optional[float] = None,
        sigma_b: Optional[float] = None,
    ):
        self.info = info
        self.energy = info.energy if info else None
        self.a_weight_map: dict[str, float] = {}
        self.b_weight_map: dict[str, float] = {}
        if sigma_a is not None:
            self.a_weight_map = weight_map_fact(n_nts=6, mu=0, sd=sigma_a)
        if sigma_b is not None:
            self.b_weight_map = weight_map_fact(n_nts=6, mu=0, sd=sigma_b, do_abs=True)


# TODO: adjustable weight maps


class ActionDomain(Domain):
    """Domain that causes the cell to do some action if protein is active"""


class ReceptorDomain(Domain):
    """Domain that activates protein if a molecule is present. Cytoplasmic receptor by default."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy = 0


class SynthesisDomain(Domain):
    """Domain that synthesizes molecule from abundant monomers if protein is active"""


class DecompositionDomain(Domain):
    """Domain that decomposes molecule into monomers if protein is active"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy = self.energy * -1


class TransmembraneDomain(Domain):
    """Creates transmembrane protein. Receptors become transmembrane receptors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy = 0


class ImporterDomain(Domain):
    """
    Domain that imports molecules from outside the cell if protein is active.
    Receptor domains on this protein become transmembrane receptors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy = 0


class ExporterDomain(Domain):
    """
    Domain that exports molecules from inside the cell if protein is active.
    Receptor domains on this protein become transmembrane receptors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy = 0


DOMAINS: dict[Domain, list[str]] = {
    ActionDomain(CM, sigma_b=1.3): variants("GGNCNN"),
    ActionDomain(CR, sigma_b=1.3): variants("GTNCNN"),
    ActionDomain(DR, sigma_b=1.3): variants("CANANN"),
    ActionDomain(TP, sigma_b=1.3): variants("CGNANN"),
    ReceptorDomain(MA, sigma_a=1.3): variants("CTNTNN"),
    ReceptorDomain(MB, sigma_a=1.3): variants("CGNCNN"),
    ReceptorDomain(MC, sigma_a=1.3): variants("AANCNN"),
    ReceptorDomain(MD, sigma_a=1.3): variants("CANGNN"),
    ReceptorDomain(ME, sigma_a=1.3): variants("CANTNN"),
    ReceptorDomain(MF, sigma_a=1.3): variants("CGNGNN"),
    SynthesisDomain(MA, sigma_b=1.3): variants("ATNCNN"),
    SynthesisDomain(MB, sigma_b=1.3): variants("ACNANN"),
    SynthesisDomain(MC, sigma_b=1.3): variants("TCNCNN"),
    SynthesisDomain(MD, sigma_b=1.3): variants("TGNTNN"),
    SynthesisDomain(ME, sigma_b=1.3): variants("GGNANN"),
    SynthesisDomain(MF, sigma_b=1.3): variants("CTNANN"),
    DecompositionDomain(MA, sigma_b=1.3): variants("CANCNN"),
    DecompositionDomain(MB, sigma_b=1.3): variants("CCNTNN"),
    DecompositionDomain(MC, sigma_b=1.3): variants("CGNTNN"),
    DecompositionDomain(MD, sigma_b=1.3): variants("ACNTNN"),
    DecompositionDomain(ME, sigma_b=1.3): variants("TANCNN"),
    DecompositionDomain(MF, sigma_b=1.3): variants("TANTNN"),
    ImporterDomain(MA, sigma_b=1.3): variants("GGNTNN"),
    ImporterDomain(MB, sigma_b=1.3): variants("CTNGNN"),
    ImporterDomain(MC, sigma_b=1.3): variants("AGNTNN"),
    ImporterDomain(MD, sigma_b=1.3): variants("GCNTNN"),
    ImporterDomain(ME, sigma_b=1.3): variants("TTNCNN"),
    ImporterDomain(MF, sigma_b=1.3): variants("GANTNN"),
    ExporterDomain(MA, sigma_b=1.3): variants("GGNGNN"),
    ExporterDomain(MB, sigma_b=1.3): variants("CTNCNN"),
    ExporterDomain(MC, sigma_b=1.3): variants("AGNANN"),
    ExporterDomain(MD, sigma_b=1.3): variants("GCNGNN"),
    ExporterDomain(ME, sigma_b=1.3): variants("TTNANN"),
    ExporterDomain(MF, sigma_b=1.3): variants("GANCNN"),
    TransmembraneDomain(): variants("AANANN"),
}


class Genetics:
    """Defines possible protein domains and how they are encoded on the genome."""

    def __init__(
        self,
        domain_map: dict[Domain, list[str]],
        n_dom_def_nts=6,
        n_dom_act_nts=6,
        n_prot_act_nts=6,
        start_codons=("TTG", "GTG", "ATG"),
        stop_codons=("TGA", "TAG", "TAA"),
    ):
        self.domain_map = domain_map
        self.n_dom_def_nts = n_dom_def_nts
        self.n_dom_act_nts = n_dom_act_nts
        self.n_prot_act_nts = n_prot_act_nts
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.seq_2_dom = {d: k for k, v in self.domain_map.items() for d in v}
        self._validate_init()

    def get_coding_regions(self, seq: str) -> list[str]:
        """
        Get all possible coding regions in nucleotide sequence

        Assuming coding region can start at any start codon and
        is stopped with the first stop codon encountered in same
        frame.

        Ribosomes will stall without stop codon. So, a coding region
        without a stop codon is not considerd.
        (https://pubmed.ncbi.nlm.nih.gov/27934701/)
        """
        cdss = []
        hits: list[list[int]] = [[] for _ in range(CODON_SIZE)]
        i = 0
        j = CODON_SIZE
        k = 0
        n = len(seq) + 1
        while j <= n:
            codon = seq[i:j]
            if codon in self.start_codons:
                hits[k].append(i)
            elif codon in self.stop_codons:
                for hit in hits[k]:
                    cdss.append(seq[hit:j])
                hits[k] = []
            i += 1
            j += 1
            k = i % CODON_SIZE
        return cdss

    def get_proteome(
        self, seq: str
    ) -> list[dict[Domain, tuple[Optional[float], Optional[float]]]]:
        """
        Get all possible proteins encoded by a nucleotide sequence.
        Proteins are represented as dicts with domain labels and correspondig
        weights.
        """
        bwd = reverse_complement(seq)
        cds = list(set(self.get_coding_regions(seq) + self.get_coding_regions(bwd)))
        min_n = self.n_dom_def_nts + self.n_dom_act_nts + 2 * CODON_SIZE
        cds = [d for d in cds if len(d) < min_n]
        return [self.translate_seq(d) for d in cds]

    # TODO: add protein constant, needs protein class
    def translate_seq(
        self, seq: str
    ) -> dict[Domain, tuple[Optional[float], Optional[float]]]:
        """
        Translate nucleotide sequence into dict that represents a protein
        with domains and corresponding activation weights.
        """
        i = 0
        j = self.n_dom_def_nts
        res: dict[Domain, tuple[Optional[float], Optional[float]]] = {}
        while j + self.n_dom_act_nts <= len(seq):
            dom = self.seq_2_dom.get(seq[i:j])
            if dom is not None:
                a = dom.a_weight_map.get(seq[j : j + self.n_dom_act_nts])
                b = dom.b_weight_map.get(seq[j : j + self.n_dom_act_nts])
                res[dom] = (a, b)
                i += self.n_dom_act_nts + self.n_dom_def_nts
                j += self.n_dom_act_nts + self.n_dom_def_nts
            else:
                i += CODON_SIZE
                j += CODON_SIZE
        return res

    def _validate_init(self):
        if self.n_dom_act_nts % CODON_SIZE != 0:
            raise ValueError(
                f"The number of nucleotides encoding domain activation regions should be a multiple of CODON_SIZE={CODON_SIZE}. "
                f"Now it is n_dom_act_nts={self.n_dom_act_nts}"
            )

        if self.n_dom_def_nts % CODON_SIZE != 0:
            raise ValueError(
                f"The number of nucleotides encoding a domain definition region should be a multiple of CODON_SIZE={CODON_SIZE}. "
                f"Now it is n_dom_def_nts={self.n_dom_def_nts}"
            )

        if self.n_prot_act_nts % CODON_SIZE != 0:
            raise ValueError(
                f"The number of nucleotides encoding protein activation regions should be a multiple of CODON_SIZE={CODON_SIZE}. "
                f"Now it is n_prot_act_nts={self.n_prot_act_nts}"
            )

        if any(len(d) != CODON_SIZE for d in self.start_codons):
            raise ValueError(
                f"Not all start codons are of length CODON_SIZE={CODON_SIZE}"
            )

        if any(len(d) != CODON_SIZE for d in self.stop_codons):
            raise ValueError(
                f"Not all stop codons are of length CODON_SIZE={CODON_SIZE}"
            )

        dfnd_doms = set(dd for d in self.domain_map.values() for dd in d)
        act_doms = set(self.seq_2_dom)
        if act_doms != dfnd_doms:
            rdff = dfnd_doms - act_doms
            raise ValueError(
                "Following domain sequences are overlapping: " + ", ".join(rdff)
            )

        if any(len(d) != self.n_dom_def_nts for d in act_doms):
            raise ValueError(
                f"Not all domains are of length n_dom_def_nts={self.n_dom_def_nts}"
            )

        exp_nts = set(ALL_NTS)
        wrng_dom_nts = [d for d in self.seq_2_dom if set(d) - exp_nts]
        if len(wrng_dom_nts) > 0:
            raise ValueError(
                f"Some domains include unknown nucleotides: {', '.join(wrng_dom_nts)}. "
                f"Known nucleotides are: {', '.join(exp_nts)}."
            )

        for dom in self.domain_map:
            na = [len(d) for d in dom.a_weight_map]
            nb = [len(d) for d in dom.b_weight_map]
            n_nts = set(na + nb)
            if len(n_nts) > 0 and n_nts != {self.n_dom_act_nts}:
                raise ValueError(
                    f"Found weight map with wrong nucleotide lengths in domain: {dom}. "
                    f"All weight should be encoded with n_dom_act_nts={self.n_dom_act_nts} nucleatoides."
                )
