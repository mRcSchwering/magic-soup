import warnings
import random
import abc
from magicsoup.util import (
    reverse_complement,
    variants,
    generic_map_fact,
    weight_map_fact,
    bool_map_fact,
)
from magicsoup.containers import Domain, Protein, Chemistry, Molecule
from magicsoup.constants import CODON_SIZE


def _check_n(n: int, label: str):
    if n % CODON_SIZE != 0:
        raise ValueError(f"{label} must be a multiple of {CODON_SIZE}. Now it is {n}")


class _DomainFact(abc.ABC):
    """Base class to create domain factory. Must implement __call__."""

    min_len = 0

    @abc.abstractmethod
    def __call__(self, seq: str) -> Domain:
        """Instantiate domain object from encoding nucleotide sequence"""
        raise NotImplementedError("Implement __call__")

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(min_len=%r)" % (clsname, self.min_len)


class CatalyticDomain(Domain):
    def __init__(
        self,
        reaction: tuple[list[Molecule], list[Molecule]],
        affinity: float,
        velocity: float,
        is_bkwd: bool,
    ):
        subs, prods = reaction
        super().__init__(
            substrates=subs,
            products=prods,
            affinity=affinity,
            velocity=velocity,
            is_bkwd=is_bkwd,
            is_catalytic=True,
        )

    def __str__(self) -> str:
        if self.is_bkwd:
            outs = ",".join(str(d) for d in self.substrates)
            ins = ",".join(str(d) for d in self.products)
        else:
            ins = ",".join(str(d) for d in self.substrates)
            outs = ",".join(str(d) for d in self.products)
        return f"CatalyticDomain({ins}->{outs})"


class CatalyticFact(_DomainFact):
    """
    Factory for generating catalytic domains from nucleotide sequences.

    - `reaction_map` Map nucleotide sequences to reactions. Reactions are defined
      by a tuple of `(substrates, products)`.
    - `affinity_map` Map nucleotide sequences to affinity values (Km in MM. kinetic)
    - `velocity_map` Map nucleotide sequences to maximum velocity values (Vmax in MM. kinetic)

    Depending on molecule concentrations and energies each reaction can take place
    in both directions (substrates -> products or products -> substrates). Thus, it
    is not necessary to additionally define the reverse reaction.
    """

    def __init__(
        self,
        reactions: list[tuple[list[Molecule], list[Molecule]]],
        km_range=(0.1, 10.0),
        vmax_range=(1.0, 10.0),
        n_reaction_nts=6,
        n_affinity_nts=6,
        n_velocity_nts=6,
        n_orientation_nts=3,
    ):
        self.reaction_map = generic_map_fact(variants("N" * n_reaction_nts), reactions)
        self.affinity_map = weight_map_fact(variants("N" * n_affinity_nts), *km_range)
        self.velocity_map = weight_map_fact(variants("N" * n_velocity_nts), *vmax_range)
        self.orientation_map = bool_map_fact(variants("N" * n_orientation_nts))

        _check_n(n_reaction_nts, "n_reaction_nts")
        _check_n(n_affinity_nts, "n_affinity_nts")
        _check_n(n_velocity_nts, "n_velocity_nts")
        _check_n(n_orientation_nts, "n_orientation_nts")

        if len(reactions) > 4 ** n_reaction_nts:
            raise ValueError(
                f"There are {len(reactions)} reactions."
                f" But with n_reaction_nts={n_reaction_nts} only {4 ** n_reaction_nts} reactions can be encoded."
            )

        self.react_slice = slice(0, n_reaction_nts)
        self.aff_slice = slice(n_reaction_nts, n_reaction_nts + n_affinity_nts)
        self.velo_slice = slice(
            n_reaction_nts + n_affinity_nts,
            n_reaction_nts + n_affinity_nts + n_velocity_nts,
        )
        self.orient_slice = slice(
            n_reaction_nts + n_affinity_nts + n_velocity_nts,
            n_reaction_nts + n_affinity_nts + n_velocity_nts + n_orientation_nts,
        )

        self.min_len = (
            n_reaction_nts + n_affinity_nts + n_velocity_nts + n_orientation_nts
        )

    def __call__(self, seq: str) -> Domain:
        react = self.reaction_map[seq[self.react_slice]]
        aff = self.affinity_map[seq[self.aff_slice]]
        velo = self.velocity_map[seq[self.velo_slice]]
        is_bkwd = self.orientation_map[seq[self.orient_slice]]
        return CatalyticDomain(
            reaction=react, affinity=aff, velocity=velo, is_bkwd=is_bkwd
        )


class TransporterDomain(Domain):
    def __init__(
        self, molecule: Molecule, affinity: float, velocity: float, is_bkwd: bool
    ):
        super().__init__(
            substrates=[molecule],
            products=[],
            affinity=affinity,
            velocity=velocity,
            is_bkwd=is_bkwd,
            is_transporter=True,
        )

    def __str__(self) -> str:
        d = "outwards" if self.is_bkwd else "inwards"
        return f"TransporterDomain({self.substrates[0]},{d})"


class TransporterFact(_DomainFact):
    """
    Factory for generating transporter domains from nucleotide sequences. Transporters
    essentially convert a type of molecule from their intracellular version to their
    extracellular version and/or vice versa.

    - `molecule_map` Map nucleotide sequences to transported molecules.
    - `affinity_map` Map nucleotide sequences to affinity values (Km in MM. kinetic)
    - `velocity_map` Map nucleotide sequences to maximum velocity values (Vmax in MM. kinetic)

    Depending on molecule concentrations the transporter can work in both directions.
    Thus it is not necessary to define both the intracellular and extracellular version
    for each type of molecule.
    """

    def __init__(
        self,
        molecules: list[Molecule],
        km_range=(0.1, 10.0),
        vmax_range=(1.0, 10.0),
        n_molecule_nts=6,
        n_affinity_nts=6,
        n_velocity_nts=6,
        n_orientation_nts=3,
    ):
        self.molecule_map = generic_map_fact(variants("N" * n_molecule_nts), molecules)
        self.affinity_map = weight_map_fact(variants("N" * n_affinity_nts), *km_range)
        self.velocity_map = weight_map_fact(variants("N" * n_velocity_nts), *vmax_range)
        self.orientation_map = bool_map_fact(variants("N" * n_orientation_nts))

        _check_n(n_molecule_nts, "n_molecule_nts")
        _check_n(n_affinity_nts, "n_affinity_nts")
        _check_n(n_velocity_nts, "n_velocity_nts")
        _check_n(n_orientation_nts, "n_orientation_nts")

        if len(molecules) > 4 ** n_molecule_nts:
            raise ValueError(
                f"There are {len(molecules)} molecules."
                f" But with n_molecule_nts={n_molecule_nts} only {4 ** n_molecule_nts} molecules can be encoded."
            )

        self.mol_slice = slice(0, n_molecule_nts)
        self.aff_slice = slice(n_molecule_nts, n_molecule_nts + n_affinity_nts)
        self.velo_slice = slice(
            n_molecule_nts + n_affinity_nts,
            n_molecule_nts + n_affinity_nts + n_velocity_nts,
        )
        self.orient_slice = slice(
            n_molecule_nts + n_affinity_nts + n_velocity_nts,
            n_molecule_nts + n_affinity_nts + n_velocity_nts + n_orientation_nts,
        )

        self.min_len = (
            n_molecule_nts + n_affinity_nts + n_velocity_nts + n_orientation_nts
        )

    def __call__(self, seq: str) -> Domain:
        mol = self.molecule_map[seq[self.mol_slice]]
        aff = self.affinity_map[seq[self.aff_slice]]
        velo = self.velocity_map[seq[self.velo_slice]]
        is_bkwd = self.orientation_map[seq[self.orient_slice]]
        return TransporterDomain(
            molecule=mol, affinity=aff, velocity=velo, is_bkwd=is_bkwd
        )


class RegulatoryDomain(Domain):
    def __init__(
        self,
        effector: Molecule,
        affinity: float,
        is_inhibiting: bool,
        is_transmembrane: bool,
    ):
        super().__init__(
            substrates=[effector],
            products=[],
            affinity=affinity,
            velocity=0.0,
            is_bkwd=False,
            is_regulatory=True,
            is_inhibiting=is_inhibiting,
            is_transmembrane=is_transmembrane,
        )

    def __str__(self) -> str:
        loc = "transmembrane" if self.is_transmembrane else "cytosolic"
        eff = "inhibiting" if self.is_inhibiting else "activating"
        return f"ReceptorDomain({self.substrates[0]},{loc},{eff})"


class RegulatoryFact(_DomainFact):
    """
    Factory for generating regulatory domains from nucleotide sequences. These domains
    can activate or inhibit the protein non-competitively.

    - `molecule_map` Map nucleotide sequences to effector molecules.
    - `affinity_map` Map nucleotide sequences to affinity values (Km in MM. kinetic)
    - `is_transmembrane` whether this factory creates transmembrane receptors or not
      (=intracellular receptors)
    - `is_inhibitor` whether the domain will be inhibiting or not (=activating)

    In case of a transmembrane receptor (`is_transmembrane=True`) the regulatory region
    reacts to the extracellular version of the effector molecule. In case of an intracellular
    receptor (`is_transmembrane=False`) the region reacts to the intracellular version of
    the effector molecule.
    """

    def __init__(
        self,
        molecules: list[Molecule],
        km_range=(0.1, 10.0),
        n_molecule_nts=6,
        n_affinity_nts=6,
        n_transmembrane_nts=3,
        n_inhibit_nts=3,
    ):
        self.molecule_map = generic_map_fact(variants("N" * n_molecule_nts), molecules)
        self.affinity_map = weight_map_fact(variants("N" * n_affinity_nts), *km_range)
        self.transmembrane_map = bool_map_fact(variants("N" * n_transmembrane_nts))
        self.inhibit_map = bool_map_fact(variants("N" * n_inhibit_nts))

        _check_n(n_molecule_nts, "n_molecule_nts")
        _check_n(n_affinity_nts, "n_affinity_nts")
        _check_n(n_transmembrane_nts, "n_transmembrane_nts")
        _check_n(n_inhibit_nts, "n_inhibit_nts")

        if len(molecules) > 4 ** n_molecule_nts:
            raise ValueError(
                f"There are {len(molecules)} molecules."
                f" But with n_molecule_nts={n_molecule_nts} only {4 ** n_molecule_nts} molecules can be encoded."
            )

        self.mol_slice = slice(0, n_molecule_nts)
        self.aff_slice = slice(n_molecule_nts, n_molecule_nts + n_affinity_nts)
        self.trans_slice = slice(
            n_molecule_nts + n_affinity_nts,
            n_molecule_nts + n_affinity_nts + n_transmembrane_nts,
        )
        self.inh_slice = slice(
            n_molecule_nts + n_affinity_nts + n_transmembrane_nts,
            n_molecule_nts + n_affinity_nts + n_transmembrane_nts + n_inhibit_nts,
        )

        self.min_len = (
            n_molecule_nts + n_affinity_nts + n_transmembrane_nts + n_inhibit_nts
        )

    def __call__(self, seq: str) -> Domain:
        mol = self.molecule_map[seq[self.mol_slice]]
        aff = self.affinity_map[seq[self.aff_slice]]
        trans = self.transmembrane_map[seq[self.inh_slice]]
        inh = self.inhibit_map[seq[self.inh_slice]]
        return RegulatoryDomain(
            effector=mol, affinity=aff, is_inhibiting=inh, is_transmembrane=trans,
        )


def _get_n(p: float, s: int, name: str) -> int:
    n = int(p * s)
    if n == 0 and p > 0.0:
        warnings.warn(
            f"There will be no {name}."
            f" Increase dom_type_size to accomodate low probabilities of having {name}."
        )
    return n


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

    # TODO: function to generate desired genome
    #       e.g. specify proteome, get genome

    def __init__(
        self,
        chemistry: Chemistry,
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        dom_type_size=6,
        p_catal_dom=0.1,
        p_transp_dom=0.1,
        p_allo_dom=0.1,
    ):
        if any(len(d) != CODON_SIZE for d in start_codons):
            raise ValueError(f"Not all start codons are of length {CODON_SIZE}")
        if any(len(d) != CODON_SIZE for d in stop_codons):
            raise ValueError(f"Not all stop codons are of length {CODON_SIZE}")

        self.chemistry = chemistry
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.dom_type_size = dom_type_size

        if p_catal_dom + p_transp_dom + p_allo_dom > 1.0:
            raise ValueError(
                "p_catal_dom, p_transp_dom, p_allo_dom together must not be greater 1.0"
            )

        sets = variants("N" * dom_type_size)
        random.shuffle(sets)
        n = len(sets)

        n_catal_doms = _get_n(p=p_catal_dom, s=n, name="catalytic domains")
        n_transp_doms = _get_n(p=p_transp_dom, s=n, name="transporter domains")
        n_allo_doms = _get_n(p=p_allo_dom, s=n, name="allosteric domains")

        catal_dom_fact = CatalyticFact(reactions=chemistry.reactions)
        transp_dom_fact = TransporterFact(molecules=chemistry.molecules)
        allo_dom_fact = RegulatoryFact(molecules=chemistry.molecules)

        domain_facts: dict[_DomainFact, list[str]] = {}
        domain_facts[catal_dom_fact] = sets[:n_catal_doms]
        del sets[:n_catal_doms]
        domain_facts[transp_dom_fact] = sets[:n_transp_doms]
        del sets[:n_transp_doms]
        domain_facts[allo_dom_fact] = sets[:n_allo_doms]
        del sets[:n_allo_doms]

        self.domain_map = {d: k for k, v in domain_facts.items() for d in v}

        if self.dom_type_size % CODON_SIZE != 0:
            raise ValueError(
                f"Sequences that define domains should be a multiple of {CODON_SIZE}."
                f" Now sequences in domain_facts have a length of {self.dom_type_size}"
            )

        self.dom_details_size = max(d.min_len for d in domain_facts)
        self.dom_size = self.dom_type_size + self.dom_details_size
        self.min_cds_size = self.dom_size + 2 * CODON_SIZE

    def get_proteome(self, seq: str) -> list[Protein]:
        """Get all possible proteins encoded by a nucleotide sequence"""
        bwd = reverse_complement(seq)
        cds = list(set(self.get_coding_regions(seq) + self.get_coding_regions(bwd)))
        proteins = [self.translate_seq(d) for d in cds]
        proteins = [d for d in proteins if len(d) > 0]
        proteins = [d for d in proteins if not all(dd.is_regulatory for dd in d)]
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
        max_start_idx = n - self.min_cds_size

        start_idxs = []
        for start_codon in self.start_codons:
            i = 0
            while i < max_start_idx:
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
                stop_idxs = [d for d in stop_idxs if d > start_idx + CODON_SIZE]
                if len(stop_idxs) > 0:
                    cds_end_idx = min(stop_idxs) + CODON_SIZE
                    if cds_end_idx - start_idx > self.min_cds_size:
                        cdss.append(seq[start_idx:cds_end_idx])
                else:
                    break

        return cdss

    def translate_seq(self, seq: str) -> list[Domain]:
        """
        Translate nucleotide sequence into a protein with domains, corresponding
        affinities, velocities, transmembrane regions and so on as defined by
        `domain_facts`.
        """
        i = 0
        j = self.dom_type_size
        doms: list[Domain] = []
        while i + self.dom_size <= len(seq):
            domfact = self.domain_map.get(seq[i:j])
            if domfact is not None:
                dom = domfact(seq[j : j + self.dom_details_size])
                doms.append(dom)
                i += self.dom_size
                j += self.dom_size
            else:
                i += CODON_SIZE
                j += CODON_SIZE

        return doms

