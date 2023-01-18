import warnings
import random
import abc
from magicsoup.util import (
    reverse_complement,
    nt_seqs,
    generic_map_fact,
    log_weight_map_fact,
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
    """
    Container holding the specification for a catalytic domain.
    Usually, you don't need to manually instantiate domains.
    During the simulation they are automatically instantiated through factories.

    - `reaction` Tuple of substrate and product molecule species that describe the
      reaction catalyzed by this domain. For stoichiometric coefficients > 1, list
      the molecule species multiple times.
    - `affinity` Michaelis Menten constant of the reaction (in mol).
    - `velocity` Maximum velocity of the reaction (in mol per time step).
    - `is_bkwd` Flag indicating whether in which orientation this reaction will be
      coupled with other domains.
    """

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

    - `reactions` All reactions that can be catalyzed by domains. Each reaction is a
      tuple of substrate and product molecule species. For stoichiometric coefficients > 1, list
      the molecule species multiple times.
    - `km_range` The range from which to sample Michaelis Menten constants for this reaction (in mol).
      The sampling will happen in the log transformed intervall, so all values must be > 0.
    - `vmax_range` The range from which to sample maximum velocities for this reaction (in mol per time step).
      The sampling will happen in the log transformed intervall, so all values must be > 0.
    - `n_reaction_nts` The number of nucleotides that should encode the reaction.
    - `n_affinity_nts` The number of nucleotides that should encode the Michaelis Menten constant.
    - `n_velocity_nts` The number of nucleotides that should encode the maximum Velocity.
    - `n_orientation_nts` The number of nucleotides that should encode the orientation of the reaction.

    This factory randomly assigns nucleotide sequences to reactions, Michaelis Menten constants, maximum
    velocities, and domain orientations on initialization. When calling the instantiated factory with a
    nucleotide sequence, it will return the encoded domain. How many nucleotides encode which part of the
    domain is defined by the `n_*_nts` arguments. The overall length of this domain type in nucleotides
    is the sum of all these. These domain factories are instantiated when initializing `Genetics`, which
    also happens when initializing `World`. They are used during translation to map nucleotide sequences
    to domains.

    ```
        DomainFact = CatalyticFact(reactions=[([A], [B])])
        DomainFact("ATCGATATATTTGCAAATTGA")
    ```

    Any reaction can happen in both directions, so it is not necessary to define the reverse reaction again.
    The orientation of a reaction only matters in combination with other domains in the way how a protein
    energetically couples multiple actions. This orientation is defined by an attribute `is_bkwd` on the domain.
    That attribute will be sampled during initialization of this factory and there is a 1:1 chance for any orientation.
    """

    def __init__(
        self,
        reactions: list[tuple[list[Molecule], list[Molecule]]],
        km_range=(1e-5, 1.0),
        vmax_range=(0.01, 10.0),
        n_reaction_nts=6,
        n_affinity_nts=6,
        n_velocity_nts=6,
        n_orientation_nts=3,
    ):
        self.reaction_map = generic_map_fact(nt_seqs(n_reaction_nts), reactions)
        self.affinity_map = log_weight_map_fact(nt_seqs(n_affinity_nts), *km_range)
        self.velocity_map = log_weight_map_fact(nt_seqs(n_velocity_nts), *vmax_range)
        self.orientation_map = bool_map_fact(nt_seqs(n_orientation_nts))

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
    """
    Container holding the specification for a transporter domain.
    Usually, you don't need to manually instantiate domains.
    During the simulation they are automatically instantiated through factories.

    - `molecule` The molecule species which can be transported into or out of the cell
      by this domain.
    - `affinity` Michaelis Menten constant of the transport (in mol).
    - `velocity` Maximum velocity of the transport (in mol per time step).
    - `is_bkwd` Flag indicating whether in which orientation this transporter will be
      coupled with other domains.
    """

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
        return f"TransporterDomain({self.substrates[0]})"


class TransporterFact(_DomainFact):
    """
    Factory for generating transporter domains from nucleotide sequences.

    - `molecules` All molecule species that can be transported into or out of the cell.
    - `km_range` The range from which to sample Michaelis Menten constants for this transport (in mol).
      The sampling will happen in the log transformed intervall, so all values must be > 0.
    - `vmax_range` The range from which to sample maximum velocities for this transport (in mol per time step).
      The sampling will happen in the log transformed intervall, so all values must be > 0.
    - `n_molecule_nts` The number of nucleotides that should encode the molecule species.
    - `n_affinity_nts` The number of nucleotides that should encode the Michaelis Menten constant.
    - `n_velocity_nts` The number of nucleotides that should encode the maximum Velocity.
    - `n_orientation_nts` The number of nucleotides that should encode the orientation of the transport.

    This factory randomly assigns nucleotide sequences to molecule species, Michaelis Menten constants, maximum
    velocities, and domain orientations on initialization. When calling the instantiated factory with a
    nucleotide sequence, it will return the encoded domain. How many nucleotides encode which part of the
    domain is defined by the `n_*_nts` arguments. The overall length of this domain type in nucleotides
    is the sum of all these. These domain factories are instantiated when initializing `Genetics`, which
    also happens when initializing `World`. They are used during translation to map nucleotide sequences
    to domains.

    ```
        DomainFact = TransporterFact(molecules=[A])
        DomainFact("ATCGATATATTTGCAAATTGA")
    ```

    Any transporter works in both directions.
    The orientation only matters in combination with other domains in the way how a protein energetically
    couples multiple actions. This orientation is defined by an attribute `is_bkwd` on the domain.
    That attribute will be sampled during initialization of this factory and there is a 1:1 chance for any orientation.
    """

    def __init__(
        self,
        molecules: list[Molecule],
        km_range=(1e-5, 1.0),
        vmax_range=(0.01, 10.0),
        n_molecule_nts=6,
        n_affinity_nts=6,
        n_velocity_nts=6,
        n_orientation_nts=3,
    ):
        self.molecule_map = generic_map_fact(nt_seqs(n_molecule_nts), molecules)
        self.affinity_map = log_weight_map_fact(nt_seqs(n_affinity_nts), *km_range)
        self.velocity_map = log_weight_map_fact(nt_seqs(n_velocity_nts), *vmax_range)
        self.orientation_map = bool_map_fact(nt_seqs(n_orientation_nts))

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
    """
    Container holding the specification for a regulatory domain.
    Usually, you don't need to manually instantiate domains.
    During the simulation they are automatically instantiated through factories.

    - `effector` The molecule species which will be the effector molecule.
    - `affinity` Michaelis Menten constant of the transport (in mol).
    - `is_inhibiting` Whether this is an inhibiting regulatory domain (otherwise activating).
    - `is_transmembrane` Whether this is also a transmembrane domain. If true, the
      domain will react to extracellular molecules instead of intracellular ones.

    I think the term Michaelis Menten constant in a regulatory domain is a bit off
    since there is no product being created. However, the kinetics of the amount of
    activation or inhibition are the same.
    """

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
    Factory for generating regulatory domains from nucleotide sequences.

    - `molecules` All molecule species that can be inhibiting or activating effectors.
    - `km_range` The range from which to sample Michaelis Menten constants for the strength 
      of inhibition or activation (in mol). The sampling will happen in the log transformed intervall,
      so all values must be > 0.
    - `n_molecule_nts` The number of nucleotides that should encode the molecule species.
    - `n_affinity_nts` The number of nucleotides that should encode the Michaelis Menten constant.
    - `n_transmembrane_nts` The number of nucleotides that should encode whether the domain also is
      a transmembrane domain (which will make it react to extracellular molecules instead).
    - `n_inhibit_nts` The number of nucleotides that should encode whether it is an activating or
      inhibiting regulatory domain.

    This factory randomly assigns nucleotide sequences to molecule species, Michaelis Menten constants,
    whether the domain is transmembrane, and whether the domain is inhibiting on initialization.
    When calling the instantiated factory with a nucleotide sequence, it will return the encoded domain.
    How many nucleotides encode which part of the domain is defined by the `n_*_nts` arguments.
    The overall length of this domain type in nucleotides is the sum of all these.
    These domain factories are instantiated when initializing `Genetics`, which
    also happens when initializing `World`. They are used during translation to map nucleotide sequences
    to domains.

    ```
        DomainFact = RegulatoryFact(molecules=[A])
        DomainFact("ATCGATATATTTGCAAAT")
    ```

    I think the term Michaelis Menten constant in a regulatory domain is a bit off
    since there is no product being created. However, the kinetics of the amount of
    activation or inhibition are the same.
    """

    def __init__(
        self,
        molecules: list[Molecule],
        km_range=(1e-5, 1.0),
        n_molecule_nts=6,
        n_affinity_nts=6,
        n_transmembrane_nts=3,
        n_inhibit_nts=3,
    ):
        self.molecule_map = generic_map_fact(nt_seqs(n_molecule_nts), molecules)
        self.affinity_map = log_weight_map_fact(nt_seqs(n_affinity_nts), *km_range)
        self.transmembrane_map = bool_map_fact(nt_seqs(n_transmembrane_nts))
        self.inhibit_map = bool_map_fact(nt_seqs(n_inhibit_nts))

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

    def __init__(
        self,
        chemistry: Chemistry,
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        p_catal_dom=0.01,
        p_transp_dom=0.01,
        p_allo_dom=0.01,
        n_dom_type_nts=6,
        n_reaction_nts=6,
        n_molecule_nts=6,
        n_affinity_nts=6,
        n_velocity_nts=6,
        n_orientation_nts=3,
        n_transmembrane_nts=3,
        n_inhibit_nts=3,
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
        if n_dom_type_nts % CODON_SIZE != 0:
            raise ValueError(f"n_dom_type_nts should be a multiple of {CODON_SIZE}.")

        self.chemistry = chemistry
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.dom_type_size = n_dom_type_nts

        if p_catal_dom + p_transp_dom + p_allo_dom > 1.0:
            raise ValueError(
                "p_catal_dom, p_transp_dom, p_allo_dom together must not be greater 1.0"
            )

        sets = nt_seqs(n_dom_type_nts)
        random.shuffle(sets)
        n = len(sets)

        n_catal_doms = _get_n(p=p_catal_dom, s=n, name="catalytic domains")
        n_transp_doms = _get_n(p=p_transp_dom, s=n, name="transporter domains")
        n_allo_doms = _get_n(p=p_allo_dom, s=n, name="allosteric domains")

        catal_dom_fact = CatalyticFact(
            reactions=chemistry.reactions,
            n_reaction_nts=n_reaction_nts,
            n_affinity_nts=n_affinity_nts,
            n_velocity_nts=n_velocity_nts,
            n_orientation_nts=n_orientation_nts,
        )
        transp_dom_fact = TransporterFact(
            molecules=chemistry.molecules,
            n_molecule_nts=n_molecule_nts,
            n_affinity_nts=n_affinity_nts,
            n_velocity_nts=n_velocity_nts,
            n_orientation_nts=n_orientation_nts,
        )
        allo_dom_fact = RegulatoryFact(
            molecules=chemistry.molecules,
            n_molecule_nts=n_molecule_nts,
            n_affinity_nts=n_affinity_nts,
            n_transmembrane_nts=n_transmembrane_nts,
            n_inhibit_nts=n_inhibit_nts,
        )

        domain_facts: dict[_DomainFact, list[str]] = {}
        if len(self.chemistry.reactions) > 0:
            domain_facts[catal_dom_fact] = sets[:n_catal_doms]
            del sets[:n_catal_doms]
        domain_facts[transp_dom_fact] = sets[:n_transp_doms]
        del sets[:n_transp_doms]
        domain_facts[allo_dom_fact] = sets[:n_allo_doms]
        del sets[:n_allo_doms]

        self.domain_map = {d: k for k, v in domain_facts.items() for d in v}

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

