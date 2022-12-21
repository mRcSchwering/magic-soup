from typing import Optional
from itertools import product
import random
from multiprocessing import Pool
import torch
from magicsoup.util import generic_map_fact, weight_map_fact, bool_map_fact
from magicsoup.containers import _Domain, _DomainFact, Protein, Molecule
from magicsoup.constants import ALL_NTS, CODON_SIZE

# TODO: move translate sequences to world
# TODO: translate into tensors instead of protein/domain classes
# TODO: collect tensors in matrix to do aggretations and assigning
#       to kinetics tensors purely in pytorch
# TODO: then chek whether I can have a CNN or RNN instead of these
#       weight/bool/domain maps, so that the whole mapping process
#       can be expressed purely in a pytorch NN

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


def point_mutatations(
    seqs: list[str], p=1e-3, p_indel=0.1
) -> tuple[list[str], list[int]]:
    """
    Mutate sequences with point mutations.

    - `seqs` nucleotide sequences
    - `p` probability of a point per nucleotide
    - `p_indel` probability of any point mutation being a deletion or insertion
      (inverse probability of it being a substitution)
    
    Returns all sequences (muated or not).
    """
    n = len(seqs)
    lens = [len(d) for d in seqs]
    s_max = max(lens)

    mask = torch.zeros(n, s_max)
    for i, s in enumerate(lens):
        mask[i, :s] = True

    probs = torch.full((n, s_max), p)
    muts = torch.bernoulli(probs)
    mut_idxs = torch.argwhere(muts * mask).tolist()

    probs = torch.full((len(mut_idxs),), p_indel)
    indels = torch.bernoulli(probs).to(torch.bool).tolist()

    tmps = [d for d in seqs]
    for (seq_i, pos_i), is_indel in zip(mut_idxs, indels):
        if is_indel:
            tmps[seq_i] = indel(seq=tmps[seq_i], idx=pos_i)
        else:
            tmps[seq_i] = substitution(seq=tmps[seq_i], idx=pos_i)

    idxs = list(set(d[0] for d in mut_idxs))
    return [tmps[i] for i in idxs], idxs


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
        domain_facts: dict[_DomainFact, list[str]],
        molecules: list[Molecule],
        reactions: list[tuple[list[Molecule], list[Molecule]]],
        vmax_range: tuple[float, float] = (1, 10),
        max_km: float = 10.0,
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        n_region_codons=2,
        n_workers=4,
    ):
        self.domain_facts = domain_facts
        self.molecules = molecules
        self.reactions = reactions
        self.vmax_range = vmax_range
        self.max_km = max_km
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.n_region_codons = n_region_codons
        self.n_workers = n_workers

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

    def random_genomes(self, n: int, s=100) -> list[str]:
        """
        Generate `n` random nucleotide sequences each with length `s`
        """
        with Pool(self.n_workers) as pool:
            return pool.map(random_genome, [s] * n)

    def get_proteomes(self, sequences: list[str]) -> list[list[Protein]]:
        """For each nucleotide sequence get all possible proteins"""
        with Pool(self.n_workers) as pool:
            return pool.map(self.get_proteome, sequences)

    def get_proteome(self, seq: str) -> list[Protein]:
        """Get all possible proteins encoded by a nucleotide sequence"""
        bwd = reverse_complement(seq)
        cds = list(set(self.get_coding_regions(seq) + self.get_coding_regions(bwd)))
        cds = [d for d in cds if len(d) > self.min_n_seq_nts]
        proteins = [self.translate_seq(d) for d in cds]
        proteins = [d for d in proteins if len(d) > 0]
        proteins = [d for d in proteins if not all(dd.is_allosteric for dd in d)]
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
        n = len(seq)

        start_idxs = []
        for start_codon in self.start_codons:
            i = 0
            # could sort out for too small CDS already (too far at end)
            while i < n - 2 * CODON_SIZE:
                try:
                    hit = seq[i:].index(start_codon)
                    start_idxs.append(i + hit)
                    i = i + hit + CODON_SIZE
                except ValueError:
                    break

        stop_idxs = []
        for stop_codon in self.stop_codons:
            i = 0
            while i < n - CODON_SIZE:
                try:
                    hit = seq[i:].index(stop_codon)
                    stop_idxs.append(i + hit)
                    i = i + hit + CODON_SIZE
                except ValueError:
                    break

        start_idxs.sort()
        stop_idxs.sort()

        by_frame: list[tuple[list[int], ...]] = [([], []), ([], []), ([], [])]
        for start_idx in start_idxs:
            if start_idx % 3 == 0:
                by_frame[0][0].append(start_idx)
            elif (start_idx + 1) % 3 == 0:
                by_frame[1][0].append(start_idx)
            else:
                by_frame[2][0].append(start_idx)
        for stop_idx in stop_idxs:
            if stop_idx % 3 == 0:
                by_frame[0][1].append(stop_idx)
            elif (stop_idx + 1) % 3 == 0:
                by_frame[1][1].append(stop_idx)
            else:
                by_frame[2][1].append(stop_idx)

        cdss = []
        for start_idxs, stop_idxs in by_frame:
            for start_idx in start_idxs:
                # could sort out for too small CDS already (too close to start)
                stop_idxs = [d for d in stop_idxs if d > start_idx + CODON_SIZE]
                if len(stop_idxs) > 0:
                    cdss.append(seq[start_idx : min(stop_idxs) + CODON_SIZE])
                else:
                    break

        return cdss

    def translate_seq(self, seq: str) -> list[_Domain]:
        """
        Translate nucleotide sequence into a protein with domains, corresponding
        weights, transmembrane regions, and signals.
        """
        i = 0
        j = self.n_dom_type_def_nts
        doms: list[_Domain] = []
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

    def summary(self, as_dict=False) -> Optional[dict]:
        """Summary of the current genetics setup"""
        n_genomes = 1000
        sizes = (100, 1000)
        out: dict[str, dict[str, float]] = {}

        for size in sizes:
            gs = self.random_genomes(n=n_genomes, s=size)
            ps = self.get_proteomes(sequences=gs)
            n_viable_proteomes = 0
            n_proteins = 0
            n_domains = 0
            n_transp = 0
            n_reg_transp = 0
            n_catal = 0
            n_reg_catal = 0
            n_catal_transp = 0
            n_reg_catal_transp = 0
            for proteome in ps:
                if len(proteome) > 0:
                    n_viable_proteomes += 1
                for prot in proteome:
                    n_proteins += 1
                    n_domains += len(prot.domains)
                    has_transp = any(d.is_transporter for d in prot.domains)
                    has_catal = any(d.is_catalytic for d in prot.domains)
                    has_allos = any(d.is_allosteric for d in prot.domains)
                    if has_transp and has_catal and has_allos:
                        n_reg_catal_transp += 1
                        continue
                    if has_transp and has_catal:
                        n_catal_transp += 1
                        continue
                    if has_transp and has_allos:
                        n_reg_transp += 1
                        continue
                    if has_transp:
                        n_transp += 1
                        continue
                    if has_catal and has_allos:
                        n_reg_catal += 1
                        continue
                    if has_catal:
                        n_catal += 1
                        continue

            # fmt: off
            out[f"genomeSize{size}"] = {
                "pctViableProteomes": n_viable_proteomes / n_genomes * 100,
                "avgProteinsPerGenome": n_proteins / n_viable_proteomes,
                "avgDomainsPerProtein": n_domains / n_proteins,
                "pctTransporterProteins": n_transp / n_proteins * 100,
                "pctRegulatedTransporterProteins": n_reg_transp / n_proteins * 100,
                "pctCatalyticProteins": n_catal / n_proteins * 100,
                "pctRegulatedCatalyticProteins": n_reg_catal / n_proteins * 100,
                "pctCatalyticTransporterProteins": n_catal_transp / n_proteins * 100,
                "pctRegulatedCatalyticTransporterProteins": n_reg_catal_transp / n_proteins * 100,
            }
            # fmt: on

        if as_dict:
            return out

        def f(x: float, x_reg: float) -> tuple[float, float]:
            both = x + x_reg
            if both == 0.0:
                return (0.0, 0.0)
            return both, x_reg / both * 100

        print("Expected proteomes")
        for size in sizes:
            # fmt: off
            print(f"\nWith genome size {size}")
            res = out[f"genomeSize{size}"]
            print(f"{res['pctViableProteomes']:.1f}% yield viable proteomes, for these viable proteomes:")
            print(f"- {res['avgProteinsPerGenome']:.1f} average proteins per genome")
            print(f"- {res['avgDomainsPerProtein']:.1f} average domains per protein")
            both, reg = f(res['pctTransporterProteins'], res['pctRegulatedTransporterProteins'])
            print(f"- {both:.0f}% pure transporter proteins, {reg:.0f}% of them are regulated")
            both, reg = f(res['pctCatalyticProteins'], res['pctRegulatedCatalyticProteins'])
            print(f"- {both:.0f}% pure catalytic proteins, {reg:.0f}% of them are regulated")
            both, reg = f(res['pctCatalyticTransporterProteins'], res['pctRegulatedCatalyticTransporterProteins'])
            print(f"- {both:.0f}% catalytic transporter proteins, {reg:.0f}% of them are regulated")
            # fmt: on

        print("")
        return None

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

