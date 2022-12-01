from util import (
    ALL_NTS,
    CODON_SIZE,
    variants,
    reverse_complement,
    weight_map_fact,
    bool_map_fact,
)


class Signal:
    """
    General piece of information that can be handled by a domain.
    Could be a molecule, or some action done by a cell.
    Gets its concrete meaning together with domain.
    """

    is_molecule = False
    is_action = False

    def __init__(self, name: str, energy: float):
        self.name = name
        self.energy = energy

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(name=%r,energy=%r)" % (clsname, self.name, self.energy)


# TODO: signals as static classes?


class Molecule(Signal):
    """Representing a molecule. A domain would be specific to this molecule."""

    is_molecule = True


MA = Molecule("A", 5)  # molceule A
MB = Molecule("B", 4)  # molceule B
MC = Molecule("C", 3)  # molceule C
MD = Molecule("D", 3)  # molceule D
ME = Molecule("E", 2)  # molceule E
MF = Molecule("F", 1)  # molceule F

MOLECULES: list[Signal] = [MA, MB, MC, MD, ME, MF]


class Action(Signal):
    """Representing an action. A domain would trigger this action."""

    is_action = True


CM = Action("CM", 2)  # cell migration
CR = Action("CR", 4)  # cell replication
DR = Action("DR", 1)  # DNA repair
TP = Action("TP", 1)  # transposon


ACTIONS: list[Signal] = [CM, CR, DR, TP]


class Domain:
    """Baseclass for domains"""

    def __init__(
        self,
        signal: Signal,
        weight: float,
        is_transmembrane: bool,
        is_energy_neutral=False,
        is_action=False,
        is_receptor=False,
        is_synthesis=False,
    ):
        self.signal = signal
        self.weight = weight
        self.is_transmembrane = is_transmembrane
        self.is_energy_neutral = is_energy_neutral
        self.is_action = is_action
        self.is_receptor = is_receptor
        self.is_synthesis = is_synthesis

        self.energy = signal.energy

        if is_synthesis and weight < 0:
            self.energy = self.energy * -1.0

        if is_energy_neutral:
            self.energy = 0.0

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(signal=%r,is_receptor=%r)" % (clsname, self.signal, self.is_receptor)


class DomainFact:
    """Generate domains from `weight`, `is_transmembrane`"""

    is_energy_neutral = False
    is_action = False
    is_receptor = False
    is_synthesis = False

    def __init__(self, signal: Signal):
        self.signal = signal

    def __call__(self, weight: float, is_transmembrane: bool = False) -> Domain:
        return Domain(
            signal=self.signal,
            weight=weight,
            is_transmembrane=is_transmembrane,
            is_energy_neutral=self.is_energy_neutral,
            is_action=self.is_action,
            is_receptor=self.is_receptor,
            is_synthesis=self.is_synthesis,
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(signal=%r,is_receptor=%r)" % (clsname, self.signal, self.is_receptor)


# TODO: rather have action domains instead of "fake" molecules?
class ActionDomainFact(DomainFact):
    """Domain that causes the cell to do some action if protein is active"""

    is_action = True


class ReceptorDomainFact(DomainFact):
    """
    Domain that activates protein if a molecule is present.
    
    If the domain gets a negative weight, it will instead surpress
    the protein's activity.

    Intracellular receptor by default. If the domain has a transmembrane
    component it will become a transmembrane receptor instead.
    """

    is_energy_neutral = True
    is_receptor = True


class SynthesisDomainFact(DomainFact):
    """
    Domain that synthesizes molecule from abundant monomers if protein is active.
    Energy for the creation of that molecule must be provided.
    
    If this domain gets a negative weight, it instead catalyzes the reverse reaction.
    Then, it deconstructs the molecule and gains energy from it.

    If the domain has a transmembrane component it will synthesize or deconstruct
    molecules outside the cell instead.
    """

    is_synthesis = True


# TODO: transporter domain: energy from concentration gradients
# TODO: calculate equilibrium constant from energy

DOMAINS: dict[DomainFact, list[str]] = {
    ActionDomainFact(CM): variants("GGNCNN") + variants("AGNTNN"),
    ActionDomainFact(CR): variants("GTNCNN") + variants("CTNGNN"),
    ActionDomainFact(DR): variants("CANANN") + variants("GGNTNN"),
    ActionDomainFact(TP): variants("CGNANN") + variants("GGNGNN"),
    ReceptorDomainFact(MA): variants("CTNTNN") + variants("CTNCNN"),
    ReceptorDomainFact(MB): variants("CGNCNN") + variants("AGNANN"),
    ReceptorDomainFact(MC): variants("AANCNN") + variants("GCNGNN"),
    ReceptorDomainFact(MD): variants("CANGNN") + variants("TTNANN"),
    ReceptorDomainFact(ME): variants("CANTNN") + variants("AANANN"),
    ReceptorDomainFact(MF): variants("CGNGNN") + variants("GANCNN"),
    SynthesisDomainFact(MA): variants("ATNCNN") + variants("CANCNN"),
    SynthesisDomainFact(MB): variants("ACNANN") + variants("CCNTNN"),
    SynthesisDomainFact(MC): variants("TCNCNN") + variants("CGNTNN"),
    SynthesisDomainFact(MD): variants("TGNTNN") + variants("ACNTNN"),
    SynthesisDomainFact(ME): variants("GGNANN") + variants("TANCNN"),
    SynthesisDomainFact(MF): variants("CTNANN") + variants("TANTNN"),
}


class Protein:
    """Container class for a protein"""

    def __init__(self, domains: list[Domain]):
        self.domains = domains

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(n_domains=%s)" % (clsname, len(self.domains))


class Genetics:
    """Defines possible protein domains and how they are encoded on the genome."""

    def __init__(
        self,
        domain_map: dict[DomainFact, list[str]],
        n_dom_def_nts=6,
        n_dom_wgt_nts=6,
        n_dom_bl_nts=3,
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        weight_range: tuple[int, int] = (-1, 1),
    ):
        self.domain_map = domain_map
        self.n_dom_def_nts = n_dom_def_nts
        self.n_dom_wgt_nts = n_dom_wgt_nts
        self.n_dom_bl_nts = n_dom_bl_nts
        self.min_n_dom_nts = (
            n_dom_def_nts + n_dom_wgt_nts + n_dom_bl_nts + 2 * CODON_SIZE
        )
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.seq_2_dom = {d: k for k, v in self.domain_map.items() for d in v}

        self.nts_2_weight = weight_map_fact(
            n_nts=self.n_dom_wgt_nts, min_w=weight_range[0], max_w=weight_range[1]
        )
        self.nts_2_bool = bool_map_fact(n_nts=self.n_dom_bl_nts)

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

    def get_proteome(self, seq: str) -> list[Protein]:
        """
        Get all possible proteins encoded by a nucleotide sequence.
        Proteins are represented as dicts with domain labels and correspondig
        weights.
        """
        bwd = reverse_complement(seq)
        cds = list(set(self.get_coding_regions(seq) + self.get_coding_regions(bwd)))
        cds = [d for d in cds if len(d) < self.min_n_dom_nts]
        return [self.translate_seq(d) for d in cds]

    def translate_seq(self, seq: str) -> Protein:
        """
        Translate nucleotide sequence into dict that represents a protein
        with domains and corresponding activation weights.
        """
        i = 0
        j = self.n_dom_def_nts
        doms = []
        while j + self.n_dom_wgt_nts <= len(seq):
            domfact = self.seq_2_dom.get(seq[i:j])
            if domfact is not None:
                dom = domfact(self.nts_2_weight[seq[j : j + self.n_dom_wgt_nts]])
                doms.append(dom)
                i += self.n_dom_wgt_nts + self.n_dom_def_nts
                j += self.n_dom_wgt_nts + self.n_dom_def_nts
            else:
                i += CODON_SIZE
                j += CODON_SIZE
        return Protein(domains=doms)

    def _validate_init(self):
        if self.n_dom_wgt_nts % CODON_SIZE != 0:
            raise ValueError(
                f"The number of nucleotides encoding domain activation regions should be a multiple of CODON_SIZE={CODON_SIZE}. "
                f"Now it is n_dom_wgt_nts={self.n_dom_wgt_nts}"
            )

        if self.n_dom_bl_nts % CODON_SIZE != 0:
            raise ValueError(
                f"The number of nucleotides encoding domain transmembrane regions should be a multiple of CODON_SIZE={CODON_SIZE}. "
                f"Now it is n_dom_bl_nts={self.n_dom_bl_nts}"
            )

        if self.n_dom_def_nts % CODON_SIZE != 0:
            raise ValueError(
                f"The number of nucleotides encoding a domain definition region should be a multiple of CODON_SIZE={CODON_SIZE}. "
                f"Now it is n_dom_def_nts={self.n_dom_def_nts}"
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

        if set(len(d) for d in self.nts_2_weight) != {self.n_dom_wgt_nts}:
            raise ValueError(
                "Found wrong nucleotide lengths in nts_2_weight. "
                f"All weights should be encoded with n_dom_wgt_nts={self.n_dom_wgt_nts} nucleatoides."
            )

        if set(len(d) for d in self.nts_2_bool) != {self.n_dom_bl_nts}:
            raise ValueError(
                "Found wrong nucleotide lengths in nts_2_bool. "
                f"All weights should be encoded with n_dom_bl_nts={self.n_dom_bl_nts} nucleatoides."
            )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(n_dom_def_nts=%r,n_dom_act_nts=%r,n_start_codons=%s,n_stop_codons=%r)"
            % (
                clsname,
                self.n_dom_def_nts,
                self.n_dom_wgt_nts,
                len(self.start_codons),
                len(self.stop_codons),
            )
        )

