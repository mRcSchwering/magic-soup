from enum import IntEnum
from util import (
    ALL_NTS,
    CODON_SIZE,
    variants,
    reverse_complement,
    weight_map_fact,
)


class CellMigration(IntEnum):
    """Domain that promotes cell migration if protein is active"""

    CM = 0  # cell migration


class ReceptorDomain:
    """Domain that activates protein if a molecule is present"""


class MessengerReceptor(IntEnum):
    """Domain that activates protein if a messenger is present"""

    MA = 1  # messenger A
    MB = 2  # messenger B
    MC = 3  # messenger C
    MD = 4  # messenger D


class MessengerSynthesis(IntEnum):
    """Domain that synthesizes a messenger if protein is active"""

    MA = 1  # messenger A
    MB = 2  # messenger B
    MC = 3  # messenger C
    MD = 4  # messenger D


class WorldSignal(IntEnum):
    """Signals/molecules that live on world map"""

    F = 5  # food
    CK = 6  # cytokine K
    CL = 7  # cytokine L


class Genetics:
    # TODO: add protein-based params
    # could add a constant to x and y (shift x or y axis)
    # based on protein -> need another genetic expression for that
    # imitates persistent expression, or reluctance of protein to express
    # until certain threshold is overcome

    # TODO: domains ideas
    # - explicit oscillator (although transduction pathways can already become oscillators)
    # - explicit memory/switch (transduction pathways could already become switches)

    # domains: (name, signal, is_incomming)
    # fmt: off
    domains: dict[tuple[str, IntEnum, bool], list[str]] = {
        ("RcF", WorldSignal.F, True): variants("CTNTNN") + variants("CANANN"),
        ("RcK", WorldSignal.CK, True): variants("CGNCNN") + variants("CGNANN"),
        ("RcL", WorldSignal.CL, True): variants("AANCNN") + variants("ATNCNN"),
        ("ExK", WorldSignal.CK, False): variants("ACNANN") + variants("ACNTNN"),
        ("ExL", WorldSignal.CL, False): variants("TCNCNN") + variants("TANCNN"),
        ("Mig", CellMigration.CM, False): variants("GGNCNN") + variants("GTNCNN"),
        ("InA", MessengerReceptor.MA, True): variants("CANGNN") + variants("CANCNN"),
        ("InB", MessengerReceptor.MB, True): variants("CANTNN") + variants("CCNTNN"),
        ("InC", MessengerReceptor.MC, True): variants("CGNGNN") + variants("CGNTNN"),
        ("InD", MessengerReceptor.MD, True): variants("TTNGNN") + variants("TGNGNN"),
        ("OutA", MessengerSynthesis.MA, False): variants("TGNTNN") + variants("TANTNN"),
        ("OutB", MessengerSynthesis.MB, False): variants("GGNANN") + variants("GGNTNN"),
        ("OutC", MessengerSynthesis.MC, False): variants("CTNANN") + variants("CTNGNN"),
        ("OutD", MessengerSynthesis.MD, False): variants("TTNTNN") + variants("TTNANN"),
    }
    # fmt: on
    domain_size = 6  # with each 3 Ns in 2 codons ^= 3% chance of randomly appearing
    weight_size = 6  # 50% chance domain itself is mutated vs weight is mutated
    start_codons = ("TTG", "GTG", "ATG")
    stop_codons = ("TGA", "TAG", "TAA")

    def __init__(self):
        self.seq_2_dom = {d: k for k, v in self.domains.items() for d in v}
        self.codon_2_range0 = weight_map_fact(
            n_nts=self.weight_size, mu=0, sd=1.3, is_positive=True
        )
        self.codon_2_range1 = weight_map_fact(
            n_nts=self.weight_size, mu=0, sd=1.3, is_positive=False
        )
        self.validate()

    def validate(self):
        """Check that configuration makes sense"""
        dfnd_doms = set(dd for d in self.domains.values() for dd in d)
        act_doms = set(self.seq_2_dom)
        if act_doms != dfnd_doms:
            rdff = dfnd_doms - act_doms
            raise ValueError(
                "Following domain sequences are overlapping: " + ", ".join(rdff)
            )
        if any(len(d) != self.domain_size for d in act_doms):
            raise ValueError(
                f"Not all domains are of length domain_size={self.domain_size}"
            )
        exp_nts = set(ALL_NTS)
        wrng_dom_nts = [d for d in self.seq_2_dom if set(d) - exp_nts]
        if len(wrng_dom_nts) > 0:
            raise ValueError(
                f"Some domains include unknown nucleotides: {', '.join(wrng_dom_nts)}. "
                f"Known nucleotides are: {', '.join(exp_nts)}."
            )

    def get_coding_regions(self, seq: str) -> list[str]:
        """
        Get all possible coding regions in nucleotide sequence

        Assuming coding region can start at any start codon and
        is stopped with the first stop codon encountered in same
        frame.

        Ribosomes will stall without stop codon. So, a coding region
        without a stop codon is not considerd.
        (https://pubmed.ncbi.nlm.nih.gov/27934701/)
        
        27.11.22 takes 0.7s on 1000 coding regions from genomes
        of size (1000, 5000).
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

    def get_proteome(self, g: str) -> list[dict[tuple[str, IntEnum, bool], float]]:
        """
        Get all possible proteins encoded by a nucleotide sequence.
        Proteins are represented as dicts with domain labels and correspondig
        weights.

        - `g` genome sequence (forward)
        """
        gbwd = reverse_complement(g)
        cds = list(set(self.get_coding_regions(g) + self.get_coding_regions(gbwd)))
        return [self.translate_seq(d) for d in cds]

    def translate_seq(self, seq: str) -> dict[tuple[str, IntEnum, bool], float]:
        """
        Translate nucleotide sequence into dict that represents a protein
        with domains and corresponding weights.

        > 27.11.22 on CDSs from 1000 genomes of (1000, 5000) size took 0.5s
        """
        i = 0
        j = self.domain_size
        res = {}
        while j + self.weight_size <= len(seq):
            dom = self.seq_2_dom.get(seq[i:j])
            if dom is not None:
                range_map = self.codon_2_range1 if dom[-1] else self.codon_2_range0
                res[dom] = range_map[seq[j : j + self.weight_size]]
                i += self.weight_size + self.domain_size
                j += self.weight_size + self.domain_size
            else:
                i += CODON_SIZE
                j += CODON_SIZE
        return res

