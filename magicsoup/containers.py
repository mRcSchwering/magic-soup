from typing import Optional
import abc
import torch


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
    
    Each type of molecule which is supposed to be unique should have a unique `name`.
    Molecule types are compared based on `name` and `energy`. And since `energy` levels
    of different molecules can very well be equal, the only real comparison attribute
    left is the name. Thus, every type of molecule should have a unique name.
    """

    def __init__(self, name: str, energy: float):
        self.name = name
        self.energy = energy

    def copy(self) -> "Molecule":
        """Instatiate this type of molecule again"""
        return Molecule(name=self.name, energy=self.energy)

    def __hash__(self) -> int:
        clsname = type(self).__name__
        return hash((clsname, self.name, self.energy))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(name=%r,energy=%r)" % (clsname, self.name, self.energy)

    def __str__(self) -> str:
        return self.name


class _Domain:
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
    - `is_bkwd` Bool which decides in which direction the domain will be coulpled.
      This is relevant for proteins with multiple domains. Depending on molecule
      concentrations and energies the overall reaction catalyzed and/or performed
      by the protein can occur from left to right, or right to left. Whether a domain's
      substrates will be part of the left or the right side of the equation, is decided
      by this bool.
    - `label` a label to recognize it, has no effect
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
        is_bkwd: bool,
        label="D",
        is_catalytic=False,
        is_transporter=False,
        is_allosteric=False,
        is_inhibiting=False,
        is_transmembrane=False,
    ):
        self.substrates = substrates
        self.products = products
        self.affinity = affinity
        self.velocity = velocity
        self.is_bkwd = is_bkwd
        self.label = label

        self.is_catalytic = is_catalytic
        self.is_transporter = is_transporter
        self.is_allosteric = is_allosteric
        self.is_inhibiting = is_inhibiting
        self.is_transmembrane = is_transmembrane

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(substrates=%r,products=%r,affinity=%r,velocity=%r,is_bkwd=%r,label=%r,is_catalytic=%r,is_transporter=%r,is_allosteric=%r,is_inhibiting=%r,is_transmembrane=%r)"
            % (
                clsname,
                self.substrates,
                self.products,
                self.affinity,
                self.velocity,
                self.is_bkwd,
                self.label,
                self.is_catalytic,
                self.is_transporter,
                self.is_allosteric,
                self.is_inhibiting,
                self.is_transmembrane,
            )
        )

    def __str__(self) -> str:
        # TODO: Transporter molecule A in->out not very useful
        #       maybe put __str__ into each class
        ins = ",".join(str(d) for d in self.substrates)
        outs = ",".join(str(d) for d in self.products)
        if self.is_transporter:
            return f"TransporterDomain({ins}->{outs})"
        if self.is_allosteric:
            return f"ReceptorDomain({ins})"
        if self.is_catalytic:
            return f"CatalyticDomain({ins}->{outs})"
        return f"Domain({ins}->{outs})"


class _DomainFact(abc.ABC):
    """Base class to create domain factory. Must implement __call__."""

    n_regions: int

    def __init__(self):
        self.region_size = 0
        self.molecule_map: dict[str, Molecule] = {}
        self.reaction_map: dict[str, tuple[list[Molecule], list[Molecule]]] = {}
        self.affinity_map: dict[str, float] = {}
        self.velocity_map: dict[str, float] = {}
        self.orientation_map: dict[str, bool] = {}

    @abc.abstractmethod
    def __call__(self, seq: str) -> _Domain:
        """Instantiate domain object from encoding nucleotide sequence"""
        raise NotImplementedError("Implement __call__")


class CatalyticDomain(_Domain):
    def __init__(
        self,
        reaction: tuple[list[Molecule], list[Molecule]],
        affinity: float,
        velocity: float,
        is_bkwd: bool,
    ):
        subs, prods = reaction
        super().__init__(
            substrates=[d.copy() for d in subs],
            products=[d.copy() for d in prods],
            affinity=affinity,
            velocity=velocity,
            is_bkwd=is_bkwd,
            is_catalytic=True,
        )


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

    n_regions = 4

    def __call__(self, seq: str) -> _Domain:
        react = self.reaction_map[seq[0 : self.region_size]]
        aff = self.affinity_map[seq[self.region_size : self.region_size * 2]]
        velo = self.velocity_map[seq[self.region_size * 2 : self.region_size * 3]]
        is_bkwd = self.orientation_map[seq[self.region_size * 3 : self.region_size * 4]]
        return CatalyticDomain(
            reaction=react, affinity=aff, velocity=velo, is_bkwd=is_bkwd
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(n_regions=%r)" % (clsname, self.n_regions)


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

    n_regions = 4

    def __call__(self, seq: str) -> _Domain:
        mol = self.molecule_map[seq[0 : self.region_size]].copy()
        aff = self.affinity_map[seq[self.region_size : self.region_size * 2]]
        velo = self.velocity_map[seq[self.region_size * 2 : self.region_size * 3]]
        is_bkwd = self.orientation_map[seq[self.region_size * 3 : self.region_size * 4]]
        return _Domain(
            substrates=[mol],
            products=[],
            affinity=aff,
            velocity=velo,
            is_bkwd=is_bkwd,
            is_transporter=True,
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(n_regions=%r)" % (clsname, self.n_regions)


class AllostericDomain(_Domain):
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
            is_allosteric=True,
            is_inhibiting=is_inhibiting,
            is_transmembrane=is_transmembrane,
        )


class AllostericFact(_DomainFact):
    """
    Factory for generating allosteric domains from nucleotide sequences. These domains
    can activate or inhibit the protein non-competitively.

    - `molecule_map` Map nucleotide sequences to effector molecules.
    - `affinity_map` Map nucleotide sequences to affinity values (Km in MM. kinetic)
    - `is_transmembrane` whether this factory creates transmembrane receptors or not
      (=intracellular receptors)
    - `is_inhibitor` whether the domain will be inhibiting or not (=activating)

    In case of a transmembrane receptor (`is_transmembrane=True`) the allosteric region
    reacts to the extracellular version of the effector molecule. In case of an intracellular
    receptor (`is_transmembrane=False`) the region reacts to the intracellular version of
    the effector molecule.
    """

    n_regions = 2

    def __init__(
        self, is_transmembrane=False, is_inhibiting=False,
    ):
        super().__init__()
        self.is_transmembrane = is_transmembrane
        self.is_inhibiting = is_inhibiting

    def __call__(self, seq: str) -> _Domain:
        mol = self.molecule_map[seq[0 : self.region_size]].copy()
        aff = self.affinity_map[seq[self.region_size : self.region_size * 2]]
        return AllostericDomain(
            effector=mol,
            affinity=aff,
            is_inhibiting=self.is_inhibiting,
            is_transmembrane=self.is_transmembrane,
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(n_regions?%r,is_transmembrane=%r,is_inhibiting=%r)" % (
            clsname,
            self.n_regions,
            self.is_transmembrane,
            self.is_inhibiting,
        )


class Protein:
    """
    Container class to carry domains of a protein
    
    - `domains` all domains of the protein
    - `label` a label, only to recognize it, has no effect
    """

    def __init__(self, domains: list[_Domain], label="P"):
        self.domains = domains
        self.label = label
        self.n_domains = len(domains)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(domains=%r,label=%r)" % (clsname, self.domains, self.label)

    def __str__(self) -> str:
        return self.label


class Cell:
    """
    Object representing a cell with its environment

    - `genome` this cells full genome
    - `proteome` the translated genome
    - `position` Cell's position on the cell map. Will be set when added to the `World`
    - `idx` Cell's index. Managed by `World`
    - `label` Optional label you can use to track cells.
    - `n_survived_steps` Number of time steps this cell has survived.
    """

    def __init__(
        self,
        genome: str,
        proteome: list[Protein],
        position: tuple[int, int] = (-1, -1),
        idx=-1,
        label="C",
        n_survived_steps=0,
    ):
        self.genome = genome
        self.proteome = proteome
        self.label = label
        self.position = position
        self.idx = idx
        self.n_survived_steps = n_survived_steps
        self.int_molecules: Optional[torch.Tensor] = None
        self.ext_molecules: Optional[torch.Tensor] = None

    def copy(self, **kwargs) -> "Cell":
        old_kwargs = {
            "genome": self.genome,
            "proteome": self.proteome,
            "position": self.position,
            "idx": self.idx,
            "label": self.label,
            "n_survived_steps": self.n_survived_steps,
        }
        return Cell(**{**old_kwargs, **kwargs})

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(genome=%r,proteome=%r,position=%r,idx=%r,label=%r,n_survived_steps=%r)"
            % (
                clsname,
                self.genome,
                self.proteome,
                self.position,
                self.idx,
                self.label,
                self.n_survived_steps,
            )
        )
