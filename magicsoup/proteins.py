from typing import Iterable
import abc
from .util import CODON_SIZE

# TODO: currently multiple domains can reference the same
#       molecule. And some domains change the molecule
#       e.g. set is_intracellular
#       I should either use a factory or a clone method


def _validate_seq_lens(seqs: Iterable[str], name: str):
    lens = set(len(d) for d in seqs)
    if lens != {CODON_SIZE}:
        raise ValueError(
            f"All sequences used to map a nucleotide sequence to a value must have length CODON_SIZE={CODON_SIZE}. "
            f"The following sequence lengths were found in {name}: {', '.join([str(d) for d in lens])}"
        )


class Molecule:
    """
    Represents a type of molecule which is part of the world, can diffuse,
    degrade, and be converted into other molecules. In reactions it has
    number attached which represents its concentration.

    - `name` Name for recognition and uniqueness.
    - `energy` In concept a standard free Gibb's energy.
      This amount of energy is released is the molecule would be deconstructed.
      Molecule concentrations and energies involved in a reaction decide
      whether the reaction can occur. Catalytic domains in a protein can
      couple multiple reactions.
    - `is_intracellular` A flag of whether this molecule exists outside
      or inside the cell. This can be left on its default for the mere
      definition of the molecule. It has a more technical purpose.
      The simulation treats them as 2 different molecules
      to make calculations simpler.
    
    Each type of molecule which is supposed to be unique should have a unique `name`.
    Molecule types are compared based on `name` and `energy`. And since `energy` levels
    of different molecules can very well be equal, the only real comparison attribute
    left is the name. Thus, every type of molecule should have a unique name.

    Intra- and extracellular molecules should not have a different name. Their location
    (inside or outside a cell) is handled by `is_intracellular`. If you print a molecule
    it will be prefixed with `i-` for intracellular and `e-` for extracellular.
    """

    def __init__(self, name: str, energy: float, is_intracellular=True):
        self.name = name
        self.energy = energy
        self.is_intracellular = is_intracellular

    def __hash__(self) -> int:
        clsname = type(self).__name__
        return hash((clsname, self.name, self.energy, self.is_intracellular))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(name=%r,energy=%r,is_intracellular=%r)" % (
            clsname,
            self.name,
            self.energy,
            self.is_intracellular,
        )

    def __str__(self) -> str:
        prefix = "i" if self.is_intracellular else "e"
        return prefix + "-" + self.name


class Domain:
    """
    Container that defines a domain. A protein can have multiple domains
    which together define the function of that protein.

    - `substrates` All molecules used by this domain. The concrete interpretation
      depends on the type of domain. In a catalytic domain this would be the left
      side of the chemical equation.
    - `products` All molecules produced by this domain. The concrete interpretation
      depends on the type of domain. In a catalytic domain this would be the right
      side of the chemical equation.
    - `affinity` The substrate affinity of this domain. Represents Km in Michaelis
      Menten kinetics.
    - `velocity` The reaction velocity of this domain. Represents Vmax in Michaelis
      Menten kinetics. This is only relevant for certain types of domains.
    - `energy` In concept a standard free Gibb's energy for the action performed by
      this domain (e.g. the reaction catalyzed). Molecule concentrations and this
      energy decide whether the reaction can take place and/or in which direction
      it will go.
    - `orientation` Bool which decides in which direction the domain will be coulpled.
      This is relevant for proteins with multiple domains. Depending on molecule
      concentrations and energies the overall reaction catalyzed and/or performed
      by the protein can occur from left to right, or right to left. Whether a domain's
      substrates will be part of the left or the right side of the equation, is decided
      by this bool.
    - `is_catalytic` Flag to indicate that this is a catalytic domain.
    - `is_transporter` Flag to indicate that this is a transporter domain.
    - `is_allosteric` Flag to indicate that this is a allosteric domain.
    - `is_inhibiting` Flag to indicate that this is a inhibiting domain. This is only
      relevant for allosteric domains.
    """

    def __init__(
        self,
        substrates: list[Molecule],
        products: list[Molecule],
        affinity: float,
        velocity: float,
        energy: float,
        orientation: bool,
        is_catalytic=False,
        is_transporter=False,
        is_allosteric=False,
        is_inhibiting=False,
    ):
        self.substrates = substrates
        self.products = products
        self.affinity = affinity
        self.velocity = velocity
        self.energy = energy
        self.orientation = orientation

        self.is_catalytic = is_catalytic
        self.is_transporter = is_transporter
        self.is_allosteric = is_allosteric
        self.is_inhibiting = is_inhibiting

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(substrates=%r,products=%r,affinity=%r,velocity=%r,energy=%r,orientation=%r,is_catalytic=%r,is_transporter=%r,is_allosteric=%r,is_inhibiting=%r)"
            % (
                clsname,
                self.substrates,
                self.products,
                self.affinity,
                self.velocity,
                self.energy,
                self.orientation,
                self.is_catalytic,
                self.is_transporter,
                self.is_allosteric,
                self.is_inhibiting,
            )
        )

    def __str__(self) -> str:
        ins = ",".join(str(d) for d in self.substrates)
        outs = ",".join(str(d) for d in self.products)
        if self.is_transporter:
            return f"TransporterDomain({ins}->{outs})"
        if self.is_allosteric:
            return f"ReceptorDomain({ins})"
        if self.is_catalytic:
            return f"CatalyticDomain({ins}->{outs})"
        return f"Domain({ins}->{outs})"


class DomainFact(abc.ABC):
    """Base class to create domain factory. Must implement __call__."""

    @abc.abstractmethod
    def __call__(self, seq: str) -> Domain:
        """Instantiate domain object from encoding nucleotide sequence"""
        raise NotImplementedError("Implement __call__")


class CatalyticFact(DomainFact):
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
        reaction_map: dict[str, tuple[list[Molecule], list[Molecule]]],
        affinity_map: dict[str, float],
        velocity_map: dict[str, float],
        orientation_map: dict[str, bool],
    ):
        energies: dict[str, float] = {}
        for seq, (substrates, products) in reaction_map.items():
            energy = 0.0
            for sig in substrates:
                energy -= sig.energy
            for sig in products:
                energy += sig.energy
            energies[seq] = energy

        self.energy_map = energies
        self.reaction_map = reaction_map
        self.affinity_map = affinity_map
        self.velocity_map = velocity_map
        self.orientation_map = orientation_map

        _validate_seq_lens(reaction_map, "reaction_map")
        _validate_seq_lens(affinity_map, "affinity_map")
        _validate_seq_lens(velocity_map, "velocity_map")
        _validate_seq_lens(orientation_map, "orientation_map")

    def __call__(self, seq: str) -> Domain:
        subs, prods = self.reaction_map[seq[0:CODON_SIZE]]
        energy = self.energy_map[seq[0:CODON_SIZE]]
        aff = self.affinity_map[seq[CODON_SIZE : CODON_SIZE * 2]]
        velo = self.velocity_map[seq[CODON_SIZE * 2 : CODON_SIZE * 3]]
        orient = self.orientation_map[seq[CODON_SIZE * 3 :]]
        return Domain(
            substrates=subs,
            products=prods,
            affinity=aff,
            velocity=velo,
            energy=energy,
            orientation=orient,
            is_catalytic=True,
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(reactions=%r)" % (clsname, set(self.reaction_map.values()))


class TransporterFact(DomainFact):
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
        molecule_map: dict[str, Molecule],
        affinity_map: dict[str, float],
        velocity_map: dict[str, float],
        orientation_map: dict[str, bool],
    ):

        self.molecule_map = molecule_map
        self.affinity_map = affinity_map
        self.velocity_map = velocity_map
        self.orientation_map = orientation_map

        _validate_seq_lens(molecule_map, "molecule_map")
        _validate_seq_lens(affinity_map, "affinity_map")
        _validate_seq_lens(velocity_map, "velocity_map")
        _validate_seq_lens(orientation_map, "orientation_map")

    def __call__(self, seq: str) -> Domain:
        mol1 = self.molecule_map[seq[0:CODON_SIZE]]
        aff = self.affinity_map[seq[CODON_SIZE : CODON_SIZE * 2]]
        velo = self.velocity_map[seq[CODON_SIZE * 2 : CODON_SIZE * 3]]
        orient = self.orientation_map[seq[CODON_SIZE * 3 :]]
        mol2 = Molecule(
            name=mol1.name,
            energy=mol1.energy,
            is_intracellular=not mol1.is_intracellular,
        )
        return Domain(
            substrates=[mol1],
            products=[mol2],
            affinity=aff,
            velocity=velo,
            energy=0.0,
            orientation=orient,
            is_transporter=True,
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(molecules=%r)" % (clsname, set(self.molecule_map.values()))


class AllostericFact(DomainFact):
    """
    Factory for generating allosteric domains from nucleotide sequences. These domains
    can activate or inhibit the protein non-competitively.

    - `molecule_map` Map nucleotide sequences to effector molecules.
    - `affinity_map` Map nucleotide sequences to affinity values (Km in MM. kinetic)
    - `inhibitor_map` Map nucleotide sequences to bools deciding whether the domain
      will be inhibiting or not (=activating).
    - `is_transmembrane` whether this factory creates transmembrane receptors or not
      (=intracellular receptors)

    In case of a transmembrane receptor (`is_transmembrane=True`) the allosteric region
    reacts to the extracellular version of the effector molecule. In case of an intracellular
    receptor (`is_transmembrane=False`) the region reacts to the intracellular version of
    the effector molecule.
    """

    def __init__(
        self,
        molecule_map: dict[str, Molecule],
        affinity_map: dict[str, float],
        inhibitor_map: dict[str, bool],
        orientation_map: dict[str, bool],
        is_transmembrane=False,
    ):

        self.molecule_map = molecule_map
        self.affinity_map = affinity_map
        self.inhibitor_map = inhibitor_map
        self.orientation_map = orientation_map
        self.is_transmembrane = is_transmembrane

        _validate_seq_lens(molecule_map, "molecule_map")
        _validate_seq_lens(affinity_map, "affinity_map")
        _validate_seq_lens(inhibitor_map, "inhibitor_map")
        _validate_seq_lens(orientation_map, "orientation_map")

    def __call__(self, seq: str) -> Domain:
        mol = self.molecule_map[seq[0:CODON_SIZE]]
        aff = self.affinity_map[seq[CODON_SIZE : CODON_SIZE * 2]]
        inhib = self.inhibitor_map[seq[CODON_SIZE * 2 : CODON_SIZE * 3]]
        orient = self.orientation_map[seq[CODON_SIZE * 3 :]]
        mol.is_intracellular = not self.is_transmembrane
        return Domain(
            substrates=[mol],
            products=[],
            affinity=aff,
            velocity=0.0,
            energy=0.0,
            orientation=orient,
            is_allosteric=True,
            is_inhibiting=inhib,
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(molecules=%r)" % (clsname, set(self.molecule_map.values()))


class Protein:
    """Container class to carry domains of a protein"""

    def __init__(self, domains: list[Domain], name=""):
        self.name = name
        self.domains = domains

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(domains=%r,name=%r)" % (clsname, self.domains, self.name)

    def __str__(self) -> str:
        return self.name
