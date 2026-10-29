import random

from magicsoup.chemistry import Molecule, Reaction
from magicsoup.config import CODON_SIZE
from magicsoup.genomics import DomainType
from magicsoup.world import World


class CatalyticDomainSynthesizer:
    """
    Factory for generating nucleotide sequences with [CatalyticDomains][magicsoup.biology.CatalyticDomain].

    Arguments:
        reaction: Tuple of `(substrates, products)` from [Chemistry][magicsoup.biology.Chemistry]
            where both `substrates` and `products` are lists of [Molecules][magicsoup.biology.Molecule].

    For stoichiometric coefficients > 1, list the molecule species multiple times.
    E.g. for `2A + B <-> C` use `reaction=([A, A, B], [C])`
    when `A,B,C` are molecule A, B, C instances.
    """

    def __init__(self, reaction: Reaction):
        self.reaction = reaction

    def validate(self, world: World):
        """Validate this domain factory's attributes"""
        if self.reaction not in world.chemistry.reactions:
            raise ValueError(
                f"CatalyticDomainFact has this reaction defined: {self.reaction}."
                " But this world's chemistry doesn't define this reaction."
            )

    def gen_coding_sequence(self, world: World) -> str:
        """Generate a nucleotide sequence for this domain"""
        react_2_tokens = world.cells.proteomics.reaction_map.react_2_tokens
        return world.cells.genomics.generate_domain_seq(
            domain_type=DomainType.CATALYTIC, tokens=react_2_tokens[self.reaction]
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.reaction})"

    def __str__(self) -> str:
        return str(self.reaction)


class TransporterDomainSynthesizer:
    """
    Factory for generating nucleotide sequences with [TransporterDomains][magicsoup.biology.TransporterDomain].

    Arguments:
        molecule: [Molecules][magicsoup.biology.Molecule] which can be transported into or out of the cell by this domain.
    """

    def __init__(self, molecule: Molecule):
        self.molecule = molecule

    def validate(self, world: World):
        """Validate this domain factory's attributes"""
        if self.molecule not in world.chemistry.molecules:
            raise ValueError(
                f"TransporterDomainFact has this molecule defined: {self.molecule}."
                " This world's chemistry doesn't define this molecule species."
            )

    def gen_coding_sequence(self, world: World) -> str:
        """Generate a nucleotide sequence for this domain"""
        mol_2_tokens = world.cells.proteomics.transport_map.mol_2_tokens
        return world.cells.genomics.generate_domain_seq(
            domain_type=DomainType.TRANSPORTER, tokens=mol_2_tokens[self.molecule]
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.molecule})"

    def __str__(self) -> str:
        return f"{self.molecule} transporter"


class RegulatoryDomainSynthesizer:
    """
    Factory for generating nucleotide sequences with [RegulatoryDomains][magicsoup.biology.RegulatoryDomain].

    Arguments:
        effector: Effector [Molecules][magicsoup.biology.Molecule]
    """

    def __init__(self, effector: Molecule):
        self.effector = effector

    def validate(self, world: World):
        """Validate this domain factory's attributes"""
        if self.effector not in world.chemistry.molecules:
            raise ValueError(
                f"RegulatoryDomainFact has this effector defined: {self.effector}."
                " This world's chemistry doesn't define this molecule species."
            )

    def gen_coding_sequence(self, world: World) -> str:
        """Generate a nucleotide sequence for this domain"""
        eff_2_tokens = world.cells.proteomics.effector_map.eff_2_tokens
        return world.cells.genomics.generate_domain_seq(
            domain_type=DomainType.REGULATORY,
            tokens=eff_2_tokens[(self.effector, random.choice([True, False]))],
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.effector})"

    def __str__(self) -> str:
        return f"{self.effector} receptor"


type DomainSynthesizer = CatalyticDomainSynthesizer | TransporterDomainSynthesizer | RegulatoryDomainSynthesizer


class ProteinSynthesizer:

    def __init__(
        self,
        catalyzed_reactions: list[Reaction] | None = None,
        transported_molecules: list[Molecule] | None = None,
        allosteric_effectors: list[Molecule] | None = None,
    ) -> None:
        self.domains: list[DomainSynthesizer] = []

        for react in catalyzed_reactions or []:
            self.domains.append(CatalyticDomainSynthesizer(reaction=react))

        for mol in transported_molecules or []:
            self.domains.append(TransporterDomainSynthesizer(molecule=mol))

        for eff in allosteric_effectors or []:
            self.domains.append(RegulatoryDomainSynthesizer(effector=eff))

    def __repr__(self) -> str:
        doms = [repr(d) for d in self.domains]
        return f"{type(self).__name__}({','.join(doms)})"

    def __str__(self) -> str:
        doms = [str(d) for d in self.domains]
        return f"{type(self).__name__}: {' | '.join(doms)}"


class GenomeSynthesizer:
    """
    Factory for generating genomes that translate into a desired proteome.

    Arguments:
        world: [World][magicsoup.world.World] object in which the genome will be used
        proteins: Desired proteins as a list of [ProteinSynthesizers][..ProteinSynthesizer].
            Each list of domain factories represents a protein.
        target_size: Optional target size of generated genome.
            Smallest possible genome size will be generated if `None`.
            If supplied, domains and proteins are padded with random sequences.

    The desired proteome is given as a list of proteins where each protein
    is represented by a list of domain factories.
    As domain factories [CatalyticDomainFact][..CatalyticDomainFact],
    [TransporterDomainFact][..TransporterDomainFact],
    [RegulatoryDomainFact][..RegulatoryDomainFact] can be used.

    Use [generate()][.generate] to sample and generate a genome.
    There are always multiple base pair sequences which can encode the same proteome.
    Thus each generated genome might be different.
    While the generated genome will always have at least the desired domains and proteins
    it might also include proteins which were not defined.
    """

    def __init__(
        self,
        world: World,
        proteins: list[ProteinSynthesizer],
        target_size: int | None = None,
    ) -> None:
        self.world = world
        self.proteins = proteins

        for prot in proteins:
            for dom in prot.domains:
                dom.validate(world=world)

        self.req_nts = sum(
            self.world.cells.genomics.dom_size_nt * len(d.domains) + 2 * CODON_SIZE
            for d in self.proteins
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
        return self.world.cells.genomics.generate_genome(
            cdss=[
                [d.gen_coding_sequence(world=self.world) for d in p.domains]
                for p in self.proteins
            ],
            n_pad_nts=self.target_size - self.req_nts,
        )
