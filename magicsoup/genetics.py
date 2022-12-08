from .proteins import Domain, DomainFact, Protein
from .util import (
    ALL_NTS,
    CODON_SIZE,
    reverse_complement,
    weight_map_fact,
    bool_map_fact,
)


class Genetics:
    """
    Defines possible protein domains and how they are encoded on the genome.
    
    - `domain_map` dict of all possible domain types and their encoding sequences.
    - `vmax_range` define the range within which possible maximum protein velocities can occur
    - `km_range` define the range within which possible protein substrate affinities can occur
    - `start_codons` set start codons which start a coding sequence (translation only happens within coding sequences)
    - `stop_codons` set stop codons which stop a coding sequence (translation only happens within coding sequences)

    Sampling for assigning codons to weights and transmembrane regions happens once during instantiation of this
    class. Then, all cells use the same rules for transscribing and translating their genomes.
    """

    def __init__(
        self,
        domain_map: dict[DomainFact, list[str]],
        vmax_range: tuple[float, float] = (0.2, 5.0),
        km_range: tuple[float, float] = (0.1, 10.0),
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
    ):
        self.domain_map = domain_map
        self.vmax_range = vmax_range
        self.km_range = km_range
        self.start_codons = start_codons
        self.stop_codons = stop_codons

        self.seq_2_dom = {d: k for k, v in self.domain_map.items() for d in v}
        self.n_dom_type_def_nts = len(next(iter(self.seq_2_dom)))
        self.n_dom_detail_def_nts = 3 * CODON_SIZE
        self.n_dom_def_nts = self.n_dom_type_def_nts + self.n_dom_detail_def_nts
        self.codon_2_vmax = weight_map_fact(CODON_SIZE, *vmax_range)
        self.codon_2_km = weight_map_fact(CODON_SIZE, *km_range)
        self.codon_2_inhibit = bool_map_fact(CODON_SIZE)
        self.min_n_seq_nts = self.n_dom_def_nts + 2 * CODON_SIZE

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
        
        Proteins which could theoretically be translated, but from which we can
        already tell by now that they would not be functional, will be sorted
        out at this point.
        """
        bwd = reverse_complement(seq)
        cds = list(set(self.get_coding_regions(seq) + self.get_coding_regions(bwd)))
        cds = [d for d in cds if len(d) > self.min_n_seq_nts]
        proteins = [self.translate_seq(d) for d in cds]
        return [Protein(domains=d, name=f"P{i}") for i, d in enumerate(proteins)]

    def translate_seq(self, seq: str) -> list[Domain]:
        """
        Translate nucleotide sequence into a protein with domains, corresponding
        weights, transmembrane regions, and signals.
        """
        i = 0
        j = self.n_dom_type_def_nts
        doms: list[Domain] = []
        while j + self.n_dom_detail_def_nts <= len(seq):
            domfact = self.seq_2_dom.get(seq[i:j])
            if domfact is not None:
                dom = domfact(seq[j : j + self.n_dom_detail_def_nts])
                doms.append(dom)
                i += self.n_dom_def_nts
                j += self.n_dom_def_nts
            else:
                i += CODON_SIZE
                j += CODON_SIZE

        return doms

    def _validate_init(self):
        lens = set(len(dd) for d in self.domain_map.values() for dd in d)
        if {self.n_dom_type_def_nts} != lens:
            raise ValueError(
                "Not all domain types are defined by the same amount of nucleotides. "
                "All sequences in domain_map must be of equal lengths. "
                f"Now there are multiple lengths: {', '.join(lens)}"
            )

        if any(d % CODON_SIZE != 0 for d in lens):
            raise ValueError(
                "Domain types must be defined as codons. "
                f"Lengths of all sequences in domain_map must be a multiple of CODON_SIZE={CODON_SIZE}. "
                f"Now they have length {', '.join(lens)}"
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
                "Some sequences for domain definitions were defined multiple times. "
                f"In domain_map {len(dfnd_doms)} sequences were defined. "
                f"But only {len(act_doms)} of them are unqiue. "
                f"Following sequences are overlapping: {', '.join(rdff)}"
            )

        exp_nts = set(ALL_NTS)
        wrng_dom_nts = set(d for d in self.seq_2_dom if set(d) - exp_nts)
        if len(wrng_dom_nts) > 0:
            raise ValueError(
                "Some domain type definitions include unknown nucleotides. "
                f"These nucleotides were found in domain_map: {', '.join(wrng_dom_nts)}. "
                f"Known nucleotides are: {', '.join(exp_nts)}."
            )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(domain_map=%r,vmax_range=%s,km_range=%r,start_codons=%r,stop_codons=%r)"
            % (
                clsname,
                self.domain_map,
                self.vmax_range,
                self.km_range,
                self.start_codons,
                self.stop_codons,
            )
        )
