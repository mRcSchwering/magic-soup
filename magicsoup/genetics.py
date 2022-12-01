from util import (
    ALL_NTS,
    CODON_SIZE,
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


class Molecule(Signal):
    """
    Representing a molecule.
    A domain would be specific to this molecule.

    It's energy describes free energy required to synthesize
    this molecule, or free energy released when deconstructing
    this molecule.
    """

    is_molecule = True


class Action(Signal):
    """
    Representing a cell action.
    A domain would trigger this action.
    
    It's energy describes free energy required to perform
    this action.
    """

    is_action = True


class Domain:
    """
    Baseclass for domains
    
    Should be instantiated by a factory while translating a genome to
    avoid carrying a reference to the same domain on multiple
    genomes. The domain might get updated during calculations.

    It's combination of bools describes what exactly this domain
    does and how it will be treaded during signal integration.
    See documentation of `DomainFact`.
    """

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
        return "%s(signal=%r,weight=%r,is_transmembrane=%r)" % (
            clsname,
            self.signal,
            self.weight,
            self.is_transmembrane,
        )


class DomainFact:
    """
    Factory for creating domains specific to `signal`

    A domain is part of a protein and gives the protein a certain function.
    A protein can have one or more domains. Here, the combination of all domains
    on a protein define how a protein becomes active (and by how much) and what signal
    this protein will produce (or surpress, and by how much).

    Bools defined on this class will be used to instantiate the domain.
    Override these to change the type of domains created.

    E.g. with `is_receptor=True` the domain would be a receptor domain which will activate
    or surpress the protein. The molecule by which it is activated would be `signal`.
    By how much that happens is eventually defined by the weight the domain will get during
    instantiation. Whether it is a intracellular receptor (reacting to molecule
    concentrations inside the cell) or a transmembrane receptor (reacting to molecule
    concentrations outside the cell) will decided by `is_transmembrane` during instantiation.
    """

    is_energy_neutral = False
    is_action = False
    is_receptor = False
    is_synthesis = False

    def __init__(self, signal: Signal):
        self.signal = signal

        if self.is_action and not isinstance(signal, Action):
            raise ValueError(
                "DomainFact with is_action=True must be instantiated with a Action signal. "
                f"But was instantiated with {signal}"
            )

        if not self.is_action and not isinstance(signal, Molecule):
            raise ValueError(
                "DomainFact with is_action=False must be instantiated with a Molecule signal. "
                f"But was instantiated with {signal}"
            )

    def __call__(self, weight: float, is_transmembrane: bool = False) -> Domain:
        """
        Generate a domain

        - `weight` domain weight (defines things such as sensitivity, reaction direction)
        - `is_transmembrane` whether this domain has a transmembrane region

        These arguments are abstract and get their actual meaning in combination with
        the other bools that are used to instantiate the domain. E.g. with `is_receptor=True`
        the domain would be a receptor domain which will activate or surpress the protein.
        By how much that happens is defined by the weight (and it's sign). Whether it is
        a intracellular receptor (reacting to molecule concentrations inside the cell)
        or a transmembrane receptor (reacting to molecule concentrations outside the cell)
        is decided by `is_transmembrane`. What molecule that would be is defined by `signal`.
        """
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
    """
    Domain that causes the cell to do some action if protein is active.
    Needs a `Action` signal.
    """

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


class Protein:
    """
    Container class for a protein

    This is basically just a container class for a list of domains.
    Should be instantiated just during translation.
    """

    def __init__(self, domains: list[Domain]):
        self.domains = domains

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(domains=%r)" % (clsname, self.domains)


class Genetics:
    """
    Defines possible protein domains and how they are encoded on the genome.
    
    - `domain_map` dict of all possible domains and their encoding sequences.
      The se sequences only define the domain itself, not yet the weight and transmembrane region.
    - `n_dom_def_bps` how many base pairs define the domain itself
    - `n_dom_wgt_bps` how many base pairs define the domain weight (all possible combinations will be assigned a weight)
    - `n_dom_bl_bps` how many base pair define the transmembrane region (all possible combination will be assigned a bool)
    - `weight_range` define the range within which possible weight will be uniformly sampled
    - `transm_freq` define the likelihood of transmembrane regions appearing
    - `start_codons` set start codons which start a coding sequence (translation only happens within coding sequences)
    - `stop_codons` set stop codons which stop a coding sequence (translation only happens within coding sequences)

    Sampling for assigning codons to weights and transmembrane regions happens once during instantiation of this
    class. Then, all cells use the same rules for transscribing and translating their genomes.
    """

    def __init__(
        self,
        domain_map: dict[DomainFact, list[str]],
        n_dom_def_bps=6,
        n_dom_wgt_bps=6,
        n_dom_bl_bps=3,
        weight_range: tuple[int, int] = (-1, 1),
        transm_freq: float = 0.5,
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
    ):
        self.domain_map = domain_map
        self.n_dom_def_bps = n_dom_def_bps
        self.n_dom_wgt_bps = n_dom_wgt_bps
        self.n_dom_bl_bps = n_dom_bl_bps
        self.min_n_dom_bps = (
            n_dom_def_bps + n_dom_wgt_bps + n_dom_bl_bps + 2 * CODON_SIZE
        )
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.seq_2_dom = {d: k for k, v in self.domain_map.items() for d in v}

        self.bps_2_weight = weight_map_fact(
            n_nts=self.n_dom_wgt_bps, min_w=weight_range[0], max_w=weight_range[1]
        )
        self.bps_2_bool = bool_map_fact(n_nts=self.n_dom_bl_bps, p=transm_freq)

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

    # TODO: with transporters a 1 domain protein is viable
    def get_proteome(self, seq: str) -> list[Protein]:
        """
        Get all possible proteins encoded by a nucleotide sequence.
        Proteins are represented as dicts with domain labels and correspondig
        weights.
        
        Proteins which could theoretically be translated, but from which we can
        already tell by now that they would not be functional, will be sorted
        out at this point.
        """
        bwd = reverse_complement(seq)
        cds = list(set(self.get_coding_regions(seq) + self.get_coding_regions(bwd)))
        cds = [d for d in cds if len(d) < self.min_n_dom_bps]
        return [self.translate_seq(d) for d in cds]

    def translate_seq(self, seq: str) -> Protein:
        """
        Translate nucleotide sequence into a protein with domains, corresponding
        weights, transmembrane regions, and signals.
        """
        i = 0
        j = self.n_dom_def_bps
        doms = []
        while j + self.n_dom_wgt_bps <= len(seq):
            domfact = self.seq_2_dom.get(seq[i:j])
            if domfact is not None:
                dom = domfact(self.bps_2_weight[seq[j : j + self.n_dom_wgt_bps]])
                doms.append(dom)
                i += self.n_dom_wgt_bps + self.n_dom_def_bps
                j += self.n_dom_wgt_bps + self.n_dom_def_bps
            else:
                i += CODON_SIZE
                j += CODON_SIZE
        return Protein(domains=doms)

    def _validate_init(self):
        if self.n_dom_wgt_bps % CODON_SIZE != 0:
            raise ValueError(
                f"The number of nucleotides encoding domain activation regions should be a multiple of CODON_SIZE={CODON_SIZE}. "
                f"Now it is n_dom_wgt_bps={self.n_dom_wgt_bps}"
            )

        if self.n_dom_bl_bps % CODON_SIZE != 0:
            raise ValueError(
                f"The number of nucleotides encoding domain transmembrane regions should be a multiple of CODON_SIZE={CODON_SIZE}. "
                f"Now it is n_dom_bl_bps={self.n_dom_bl_bps}"
            )

        if self.n_dom_def_bps % CODON_SIZE != 0:
            raise ValueError(
                f"The number of nucleotides encoding a domain definition region should be a multiple of CODON_SIZE={CODON_SIZE}. "
                f"Now it is n_dom_def_bps={self.n_dom_def_bps}"
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

        if any(len(d) != self.n_dom_def_bps for d in act_doms):
            raise ValueError(
                f"Not all domains are of length n_dom_def_bps={self.n_dom_def_bps}"
            )

        exp_bps = set(ALL_NTS)
        wrng_dom_bps = [d for d in self.seq_2_dom if set(d) - exp_bps]
        if len(wrng_dom_bps) > 0:
            raise ValueError(
                f"Some domains include unknown nucleotides: {', '.join(wrng_dom_bps)}. "
                f"Known nucleotides are: {', '.join(exp_bps)}."
            )

        if set(len(d) for d in self.bps_2_weight) != {self.n_dom_wgt_bps}:
            raise ValueError(
                "Found wrong nucleotide lengths in bps_2_weight. "
                f"All weights should be encoded with n_dom_wgt_bps={self.n_dom_wgt_bps} nucleatoides."
            )

        if set(len(d) for d in self.bps_2_bool) != {self.n_dom_bl_bps}:
            raise ValueError(
                "Found wrong nucleotide lengths in bps_2_bool. "
                f"All weights should be encoded with n_dom_bl_bps={self.n_dom_bl_bps} nucleatoides."
            )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(n_dom_def_bps=%r,n_dom_act_bps=%r,n_start_codons=%s,n_stop_codons=%r)"
            % (
                clsname,
                self.n_dom_def_bps,
                self.n_dom_wgt_bps,
                len(self.start_codons),
                len(self.stop_codons),
            )
        )

