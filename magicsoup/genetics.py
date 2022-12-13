from typing import Optional
from itertools import product
import random
import torch
from .util import generic_map_fact, weight_map_fact, bool_map_fact
from .containers import Domain, DomainFact, Protein, Molecule
from .constants import ALL_NTS, CODON_SIZE

# TODO: summary() to see likelyhoods of different domains appearing


# TODO: Transformation mechanism
# TODO: conjugation mechanism
# TODO: Slipped_strand_mispairing / replication slippage
# TODO: ectopic recombination


def variants(seq: str) -> list[str]:
    """
    Generate all variants of sequence where
    'N' can be any nucleotide.
    """
    s = seq
    n = s.count("N")
    for i in range(n):
        idx = s.find("N")
        s = s[:idx] + "{" + str(i) + "}" + s[idx + 1 :]
    nts = [ALL_NTS] * n
    return [s.format(*d) for d in product(*nts)]


def random_genome(s=100) -> str:
    """
    Generate a random nucleotide sequence with length `s`
    """
    return "".join(random.choices(ALL_NTS, k=s))


def random_genomes(n: int, s=100) -> list[str]:
    """
    Generate `n` random nucleotide sequences each with length `s`
    """
    return [random_genome(s=s) for _ in range(n)]


def reverse_complement(seq: str) -> str:
    """Reverse-complement of a nucleotide sequence"""
    rvsd = seq[::-1]
    return (
        rvsd.replace("A", "-")
        .replace("T", "A")
        .replace("-", "T")
        .replace("G", "-")
        .replace("C", "G")
        .replace("-", "C")
    )


def substitution(seq: str, idx: int) -> str:
    """Create a 1 nucleotide substitution at index"""
    nt = random.choice(ALL_NTS)
    return seq[:idx] + nt + seq[idx + 1 :]


def indel(seq: str, idx: int) -> str:
    """Create a 1 nucleotide insertion or deletion at index"""
    if random.choice([True, False]):
        return seq[:idx] + seq[idx + 1 :]
    nt = random.choice(ALL_NTS)
    return seq[:idx] + nt + seq[idx:]


def point_mutatations(seq: str, p=1e-3, p_indel=0.1) -> Optional[str]:
    """
    Mutate sequence with point mutations.

    - `seq` nucleotide sequence
    - `p` probability of a point per nucleotide
    - `p_indel` probability of any point mutation being a deletion or insertion
      (inverse probability of it being a substitution)
    
    Returns mutated sequence, or None if nothing was mutated.
    """
    n = len(seq)
    muts = torch.bernoulli(torch.tensor([p] * n))
    mut_idxs = torch.argwhere(muts).flatten().tolist()

    n_muts = len(mut_idxs)
    if n_muts == 0:
        return None

    indels = torch.bernoulli(torch.tensor([p_indel] * n_muts))
    is_indel = indels.to(torch.bool).tolist()

    tmp = seq
    for idx, is_indel in zip(mut_idxs, is_indel):
        if is_indel:
            tmp = indel(seq=tmp, idx=idx)
        else:
            tmp = substitution(seq=tmp, idx=idx)
    return tmp


class Genetics:
    """
    Defines possible protein domains and how they are encoded on the genome.
    
    - `domain_facts` dict mapping available domain factories to all possible nucleotide sequences
      by which they are encoded. During translation if any of these nucleotide sequences appears
      (in-frame) in the coding sequence it will create the mapped domain. Further following nucleotides
      will be used to configure that domain.
    - `vmax_range` Define the range within which possible maximum protein velocities can occur.
    - `max_km` Define the maximum Km (i.e. lowest affinity) a domain can have to its substrate(s).
      `1 / max_km` will be the minimum Km value (i.e. highest affinity).
    - `start_codons` set start codons which start a coding sequence (translation only happens within coding sequences)
    - `stop_codons` set stop codons which stop a coding sequence (translation only happens within coding sequences)

    Sampling for assigning codons to weights and transmembrane regions happens once during instantiation of this
    class. Then, all cells use the same rules for transscribing and translating their genomes.
    """

    def __init__(
        self,
        domain_facts: dict[DomainFact, list[str]],
        molecules: list[Molecule],
        reactions: list[tuple[list[Molecule], list[Molecule]]],
        vmax_range: tuple[float, float] = (1, 1000),
        max_km: float = 10.0,
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        n_region_codons=2,
    ):
        self.domain_facts = domain_facts
        self.molecules = molecules
        self.reactions = reactions
        self.vmax_range = vmax_range
        self.max_km = max_km
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.n_region_codons = n_region_codons

        codons = variants("N" * n_region_codons * CODON_SIZE)
        self.reaction_map = generic_map_fact(codons, reactions)
        self.molecule_map = generic_map_fact(codons, molecules)
        self.affinity_map = weight_map_fact(codons, 1 / max_km, max_km)
        self.velocity_map = weight_map_fact(codons, *vmax_range)
        self.bool_map = bool_map_fact(codons)

        for domain_fact in self.domain_facts:
            domain_fact.region_size = n_region_codons * CODON_SIZE
            domain_fact.reaction_map = self.reaction_map
            domain_fact.molecule_map = self.molecule_map
            domain_fact.affinity_map = self.affinity_map
            domain_fact.velocity_map = self.velocity_map
            domain_fact.orientation_map = self.bool_map

        self.domain_map = {d: k for k, v in self.domain_facts.items() for d in v}
        max_n_regions = max(d.n_regions for d in domain_facts)

        self.n_dom_type_def_nts = n_region_codons * CODON_SIZE
        self.n_dom_detail_def_nts = max_n_regions * CODON_SIZE * n_region_codons
        self.n_dom_def_nts = self.n_dom_type_def_nts + self.n_dom_detail_def_nts
        self.min_n_seq_nts = self.n_dom_def_nts + 2 * CODON_SIZE

        self._validate_init()

    def get_proteomes(self, sequences: list[str]) -> list[list[Protein]]:
        """For each nucleotide sequence get all possible proteins"""
        return [self.get_proteome(seq=d) for d in sequences]

    def get_proteome(self, seq: str) -> list[Protein]:
        """Get all possible proteins encoded by a nucleotide sequence"""
        bwd = reverse_complement(seq)
        cds = list(set(self.get_coding_regions(seq) + self.get_coding_regions(bwd)))
        cds = [d for d in cds if len(d) > self.min_n_seq_nts]
        proteins = [self.translate_seq(d) for d in cds]
        proteins = [d for d in proteins if len(d) > 0]
        return [Protein(domains=d, label=f"P{i}") for i, d in enumerate(proteins)]

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

    def translate_seq(self, seq: str) -> list[Domain]:
        """
        Translate nucleotide sequence into a protein with domains, corresponding
        weights, transmembrane regions, and signals.
        """
        i = 0
        j = self.n_dom_type_def_nts
        doms: list[Domain] = []
        while j + self.n_dom_detail_def_nts <= len(seq):
            domfact = self.domain_map.get(seq[i:j])
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
        lens = set(len(d) for d in self.domain_map)
        if len(lens) > 1:
            raise ValueError(
                "Not all domain types are defined by the same amount of nucleotides. "
                "All sequences in domain_facts must be of equal lengths. "
                f"Now there are multiple lengths: {', '.join(lens)}"
            )

        region_nts = self.n_region_codons * CODON_SIZE
        if {region_nts} != lens:
            raise ValueError(
                f"Domain regions were defined to have n_region_codons={self.n_region_codons} codons ({region_nts} nucleotides). "
                f"Currently sequences in domain_facts have {', '.join(lens)} nucleotides."
            )

        if any(len(d) != CODON_SIZE for d in self.start_codons):
            raise ValueError(
                f"Not all start codons are of length CODON_SIZE={CODON_SIZE}"
            )

        if any(len(d) != CODON_SIZE for d in self.stop_codons):
            raise ValueError(
                f"Not all stop codons are of length CODON_SIZE={CODON_SIZE}"
            )

        dfnd_doms = set(dd for d in self.domain_facts.values() for dd in d)
        act_doms = set(self.domain_map)
        if act_doms != dfnd_doms:
            rdff = dfnd_doms - act_doms
            raise ValueError(
                "Some sequences for domain definitions were defined multiple times. "
                f"In domain_facts {len(dfnd_doms)} sequences were defined. "
                f"But only {len(act_doms)} of them are unqiue. "
                f"Following sequences are overlapping: {', '.join(rdff)}"
            )

        exp_nts = set(ALL_NTS)
        wrng_dom_nts = set(d for d in self.domain_map if set(d) - exp_nts)
        if len(wrng_dom_nts) > 0:
            raise ValueError(
                "Some domain type definitions include unknown nucleotides. "
                f"These nucleotides were found in domain_facts: {', '.join(wrng_dom_nts)}. "
                f"Known nucleotides are: {', '.join(exp_nts)}."
            )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(domain_map=%r,vmax_range=%s,max_km=%r,start_codons=%r,stop_codons=%r)"
            % (
                clsname,
                self.domain_map,
                self.vmax_range,
                self.max_km,
                self.start_codons,
                self.stop_codons,
            )
        )
