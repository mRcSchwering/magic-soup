import random
from magicsoup.constants import CODON_SIZE
from magicsoup.containers import Molecule
from magicsoup.util import closest_value, random_genome, round_down
from magicsoup.world import World


class _DomainFact:
    """
    Domain factory base.
    All domain factories should inherit from this class.
    """

    def validate(
        self,
        world: World,  # pylint: disable=unused-argument
    ):
        raise NotImplementedError

    def gen_coding_sequence(
        self,
        world: World,  # pylint: disable=unused-argument
    ) -> str:
        raise NotImplementedError


class CatalyticDomainFact(_DomainFact):
    """
    Factory for generating catalytic domains.

    Arguments:
        reaction: Tuple of substrate and product molecules which are involved in the reaction.
        km: Desired Michaelis Menten constant of the transport (in mM).
        vmax: Desired Maximum velocity of the transport (in mmol/s).

    For stoichiometric coefficients > 1, list the molecule species multiple times.
    E.g. for $2A + B \rightleftharpoons C$ use `reaction=([A, A, B], [C])`
    when `A,B,C` are molecule A, B, C instances.

    `km` and `vmax` are target values.
    Due to the way how codons are sampled for specific floats, the actual values for `km` and `vmax` might differ.
    The closest available value to the given value will be used.

    If any optional argument is left out (`None`) it will be sampled randomly.
    """

    def __init__(
        self,
        reaction: tuple[list[Molecule], list[Molecule]],
        km: float | None = None,
        vmax: float | None = None,
    ):
        substrates, products = reaction
        self.substrates = sorted(substrates)
        self.products = sorted(products)
        self.km = km
        self.vmax = vmax

    def validate(self, world: World):
        """Validate this domain factory's attributes"""
        all_reacts = [
            (tuple(sorted(s)), tuple(sorted(p))) for s, p in world.chemistry.reactions
        ]
        all_reacts.extend([(p, s) for s, p in all_reacts])
        if (tuple(self.substrates), tuple(self.products)) not in all_reacts:
            lft = " + ".join(d.name for d in self.substrates)
            rgt = " + ".join(d.name for d in self.products)
            raise ValueError(
                f"CatalyticDomainFact has this reaction defined: {lft} <-> {rgt}."
                " This world's chemistry doesn't define this reaction."
            )

    def gen_coding_sequence(self, world: World) -> str:
        """Generate a nucleotide sequence for this domain"""
        # catalytic domain type: 1
        # idx0: Vmax (1 codon, no stop)
        # idx1: Km (1 codon, no stop)
        # idx2: direction (1 codon, no stop)
        # idx3: reaction (2 codon, 2nd can be stop)
        kinetics = world.kinetics
        genetics = world.genetics
        dom_seq = random.choice(genetics.domain_types[1])

        if self.vmax is not None:
            val = closest_value(values=kinetics.vmax_2_idxs, key=self.vmax)
            i0 = random.choice(kinetics.vmax_2_idxs[val])
            i0_seq = genetics.idx_2_one_codon[i0]
        else:
            i0_seq = random_genome(s=CODON_SIZE, excl=genetics.stop_codons)

        if self.km is not None:
            val = closest_value(values=kinetics.km_2_idxs, key=self.km)
            i1 = random.choice(kinetics.km_2_idxs[val])
            i1_seq = genetics.idx_2_one_codon[i1]
        else:
            i1_seq = random_genome(s=CODON_SIZE, excl=genetics.stop_codons)

        react = (tuple(self.substrates), tuple(self.products))
        is_fwd = True
        if react not in kinetics.catal_2_idxs:
            react = (tuple(self.products), tuple(self.substrates))
            is_fwd = False
        i2 = random.choice(kinetics.sign_2_idxs[is_fwd])
        i2_seq = genetics.idx_2_one_codon[i2]
        i3 = random.choice(kinetics.catal_2_idxs[react])
        i3_seq = genetics.idx_2_two_codon[i3]

        return dom_seq + i0_seq + i1_seq + i2_seq + i3_seq


class TransporterDomainFact(_DomainFact):
    """
    Factory for generating transporter domains.

    Arguments:
        molecule: Molecule species which can be transported into or out of the cell by this domain.
        km: Desired Michaelis Menten constant of the transport (in mM).
        vmax: Desired Maximum velocity of the transport (in mmol/s).
        is_exporter: Whether the transporter is exporting this molecule species out of the cell.

    `is_exporter` is only relevant in combination with other domains on the same protein.
    It defines in which transport direction this domain is energetically coupled with others.

    `km` and `vmax` are target values.
    Due to the way how codons are sampled for specific floats, the actual values for `km` and `vmax` might differ.
    The closest available value to the given value will be used.

    If any optional argument is left out (`None`) it will be sampled randomly.
    """

    def __init__(
        self,
        molecule: Molecule,
        km: float | None = None,
        vmax: float | None = None,
        is_exporter: bool | None = None,
    ):
        self.molecule = molecule
        self.km = km
        self.vmax = vmax
        self.is_exporter = is_exporter

    def validate(self, world: World):
        """Validate this domain factory's attributes"""
        if self.molecule not in world.chemistry.molecules:
            raise ValueError(
                f"TransporterDomainFact has this molecule defined: {self.molecule}."
                " This world's chemistry doesn't define this molecule species."
            )

    def gen_coding_sequence(self, world: World) -> str:
        """Generate a nucleotide sequence for this domain"""
        # transporter domain type: 2
        # idx0: Vmax (1 codon, no stop)
        # idx1: Km (1 codon, no stop)
        # idx2: is_exporter (1 codon, no stop)
        # idx3: molecule (2 codon, 2nd can be stop)
        kinetics = world.kinetics
        genetics = world.genetics
        dom_seq = random.choice(world.genetics.domain_types[2])

        if self.vmax is not None:
            val = closest_value(values=kinetics.vmax_2_idxs, key=self.vmax)
            i0 = random.choice(kinetics.vmax_2_idxs[val])
            i0_seq = genetics.idx_2_one_codon[i0]
        else:
            i0_seq = random_genome(s=CODON_SIZE, excl=genetics.stop_codons)

        if self.km is not None:
            val = closest_value(values=kinetics.km_2_idxs, key=self.km)
            i1 = random.choice(kinetics.km_2_idxs[val])
            i1_seq = genetics.idx_2_one_codon[i1]
        else:
            i1_seq = random_genome(s=CODON_SIZE, excl=genetics.stop_codons)

        if self.is_exporter is not None:
            i2 = random.choice(kinetics.sign_2_idxs[self.is_exporter])
            i2_seq = genetics.idx_2_one_codon[i2]
        else:
            i2_seq = random_genome(s=CODON_SIZE, excl=genetics.stop_codons)

        i3 = random.choice(kinetics.trnsp_2_idxs[self.molecule])
        i3_seq = genetics.idx_2_two_codon[i3]

        return dom_seq + i0_seq + i1_seq + i2_seq + i3_seq


class RegulatoryDomainFact(_DomainFact):
    """
    Factory for generating regulatory domains.

    Arguments:
        effector: Molecule species which will be the effector molecule.
        is_transmembrane: Whether this is also a transmembrane domain.
            If true, the domain will react to extracellular molecules instead of intracellular ones.
        km: Desired ligand concentration producing half occupation (in mM).
        hill: Hill coefficient describing degree of cooperativity (currently 1, 3, 5 available)
        is_inhibiting: Whether the effector will have an activating or inhibiting effect.

    `km` is a target value.
    Due to the way how codons are sampled for specific values,
    the final value for `km` might differ.
    The closest available value to the given value will be used.

    If any optional argument is left out (`None`) it will be sampled randomly.
    """

    def __init__(
        self,
        effector: Molecule,
        is_transmembrane: bool,
        is_inhibiting: bool | None = None,
        km: float | None = None,
        hill: int | None = None,
    ):
        self.effector = effector
        self.is_transmembrane = is_transmembrane
        self.is_inhibiting = is_inhibiting
        self.km = km
        self.hill = hill

    def validate(self, world: World):
        """Validate this domain factory's attributes"""
        if self.effector not in world.chemistry.molecules:
            raise ValueError(
                f"RegulatoryDomainFact has this effector defined: {self.effector}."
                " This world's chemistry doesn't define this molecule species."
            )

    def gen_coding_sequence(self, world: World) -> str:
        """Generate a nucleotide sequence for this domain"""
        # regulatory domain type: 3
        # idx0: hill coefficient (1 codon, no stop)
        # idx1: Km (1 codon, no stop)
        # idx2: sign (1 codon, no stop)
        # idx3: effector (2 codon, 2nd can be stop)
        # is_transmembrane defined by effector (int/ext molecules)
        kinetics = world.kinetics
        genetics = world.genetics
        dom_seq = random.choice(genetics.domain_types[3])

        if self.hill is not None:
            val = closest_value(values=kinetics.hill_2_idxs, key=self.hill)
            i0 = random.choice(kinetics.hill_2_idxs[int(val)])
            i0_seq = genetics.idx_2_one_codon[i0]
        else:
            i0_seq = random_genome(s=CODON_SIZE, excl=genetics.stop_codons)

        if self.km is not None:
            val = closest_value(values=kinetics.km_2_idxs, key=self.km)
            i1 = random.choice(kinetics.km_2_idxs[val])
            i1_seq = genetics.idx_2_one_codon[i1]
        else:
            i1_seq = random_genome(s=CODON_SIZE, excl=genetics.stop_codons)

        if self.is_inhibiting is not None:
            i2 = random.choice(kinetics.sign_2_idxs[not self.is_inhibiting])
            i2_seq = genetics.idx_2_one_codon[i2]
        else:
            i2_seq = random_genome(s=CODON_SIZE, excl=genetics.stop_codons)

        i3 = random.choice(
            kinetics.regul_2_idxs[(self.effector, self.is_transmembrane)]
        )
        i3_seq = genetics.idx_2_two_codon[i3]

        return dom_seq + i0_seq + i1_seq + i2_seq + i3_seq


class GenomeFact:
    """
    Factory for generating genomes that translate into a desired proteome.

    Arguments:
        world: World object in which the genome will be used
        proteome: Desired proteome as a list of lists of domain factories.
            Each list of domain factories represents a protein.
        target_size: Optional target size of generated genome.
            Smallest possible genome size will be generated if `None`.
            If supplied, domains and proteins are padded with random sequences.

    The desired proteome is given as a list of proteins where each protein
    is represented by a list of domain factories.
    As domain factories [CatalyticDomainFact][magicsoup.factories.CatalyticDomainFact],
    [TransporterDomainFact][magicsoup.factories.TransporterDomainFact],
    [RegulatoryDomainFact][magicsoup.factories.RegulatoryDomainFact] can be used.
    Use `generate()` to sample and generate a genome.
    There are always multiple nucleotide sequences which can encode the same proteome.
    Thus each generated genome might be different.

    While the generated genome will always have the desired domains and proteins
    it might also include proteins which were not defined.
    This factory assembles a genome in one reading frame and direction.
    There is always the possibility that in another reading frame or on the
    reverse-complement of the genome other proteins are encoded.
    The larger the final genome, the higher the chances of this happening.
    """

    def __init__(
        self,
        world: World,
        proteome: list[list[_DomainFact]],
        target_size: int | None = None,
    ):
        self.world = world
        self.proteome = proteome

        try:
            _ = iter(proteome)
        except TypeError as err:
            raise ValueError(
                "Proteome must be a list of lists representing domains in proteins."
            ) from err

        for pi, prot in enumerate(proteome):
            try:
                _ = iter(prot)
            except TypeError as err:
                raise ValueError(
                    "Proteome must be a list of lists representing domains in proteins."
                    f" Element {pi} of proteome is not iterable."
                ) from err

        for prot in proteome:
            for dom in prot:
                dom.validate(world=world)

        self.req_nts = sum(
            self.world.genetics.dom_size * len(d) + 2 * CODON_SIZE
            for d in self.proteome
        )
        self.target_size = self.req_nts if target_size is None else target_size
        if self.req_nts > self.target_size:
            raise ValueError(
                "Genome size too small."
                f" The given proteome would require at least {self.req_nts} nucleotides."
                f" But the given genome target size is target_size={self.target_size}."
            )

    def generate(self) -> str:
        """Generate a genome with the desired proteome"""
        cdss = [
            [d.gen_coding_sequence(world=self.world) for d in p] for p in self.proteome
        ]
        n_pads = len(cdss) + 1
        n_pad_nts = self.target_size - self.req_nts
        pad_size = round_down(n_pad_nts / n_pads, to=1)
        remaining_nts = n_pad_nts - n_pads * pad_size

        start_codons = self.world.genetics.start_codons
        stop_codons = self.world.genetics.stop_codons
        excl_cdss = start_codons + stop_codons
        pads = [random_genome(s=pad_size, excl=excl_cdss) for _ in range(n_pads)]
        tail = random_genome(s=remaining_nts, excl=excl_cdss)

        parts: list[str] = []
        for cds in cdss:
            parts.append(pads.pop())
            parts.append(random.choice(start_codons))
            parts.extend([d for d in cds])
            parts.append(random.choice(stop_codons))
        parts.append(pads.pop())
        parts.append(tail)

        return "".join(parts)
