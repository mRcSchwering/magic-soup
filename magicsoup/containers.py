from typing import Optional, Any
import warnings
import abc
import torch
from magicsoup.constants import CODON_SIZE, ALL_NTS


# TODO: tests for containers, e.g. comparisons


def _get_region_size(seqs: dict[str, Any], label: str) -> int:
    lens = set(len(d) for d in seqs)
    if len(lens) != 1:
        raise ValueError(
            f"Keys of {label} must all have the same length."
            f" Now there are lengths: {', '.join(str(d) for d in lens)}"
        )
    size = lens.pop()
    if size % CODON_SIZE != 0:
        raise ValueError(
            f"Key lengths of {label} must be a multiple of {CODON_SIZE}."
            f" Now they have lengths {size}"
        )
    all_chars = set(dd for d in seqs for dd in d)
    if all_chars > set(ALL_NTS):
        raise ValueError(
            f"Some unknown nucleotides were defined in {label}:"
            f" {', '.join(all_chars - set(ALL_NTS))}."
            f" Known nucleotides are: {', '.join(ALL_NTS)}"
        )
    return size


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

    _instances: dict[str, "Molecule"] = {}

    def __new__(cls, name: str, energy: float):
        if name in cls._instances:
            if cls._instances[name].energy != energy:
                raise ValueError(
                    f"Trying to instantiate Molecule {name} with energy {energy}."
                    f" But {name} already exists with energy {cls._instances[name].energy}"
                )
        else:
            name_ = name.lower()
            matches = [k for k in cls._instances if k.lower() == name_]
            if len(matches) > 0:
                warnings.warn(
                    f"Creating new molecule {name}."
                    f" There are molecues with similar names: {', '.join(matches)}."
                    " Give them identical names if these are the same molecules."
                )
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]

    def __init__(self, name: str, energy: float):
        self.name = name
        self.energy = energy
        self.idx = -1
        self._idx2 = -1

        self._hash = hash((self.name, self.energy))

    def __hash__(self) -> int:
        return self._hash

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

        self._hash = hash(
            (
                tuple(self.substrates),
                tuple(self.products),
                self.affinity,
                self.velocity,
                self.is_bkwd,
                self.is_catalytic,
                self.is_transporter,
                self.is_allosteric,
                self.is_transmembrane,
            )
        )

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

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
        ins = ",".join(str(d) for d in self.substrates)
        outs = ",".join(str(d) for d in self.products)
        return f"Domain({ins}<->{outs})"


class _DomainFact(abc.ABC):
    """Base class to create domain factory. Must implement __call__."""

    min_len = 0

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


# TODO: calculate maps in init with option to give them directly
#       would only need reactions, number of codons, for each map
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
        reaction_map: dict[str, tuple[list[Molecule], list[Molecule]]],
        affinity_map: dict[str, float],
        velocity_map: dict[str, float],
        orientation_map: dict[str, bool],
    ):
        self.reaction_map = reaction_map
        self.affinity_map = affinity_map
        self.velocity_map = velocity_map
        self.orientation_map = orientation_map

        n_react = _get_region_size(reaction_map, "reaction_map")
        n_aff = _get_region_size(affinity_map, "affinity_map")
        n_velo = _get_region_size(velocity_map, "velocity_map")
        n_orient = _get_region_size(orientation_map, "orientation_map")

        self.react_slice = slice(0, n_react)
        self.aff_slice = slice(n_react, n_react + n_aff)
        self.velo_slice = slice(n_react + n_aff, n_react + n_aff + n_velo)
        self.orient_slice = slice(
            n_react + n_aff + n_velo, n_react + n_aff + n_velo + n_orient,
        )

        self.min_len = n_react + n_aff + n_velo + n_orient

    def __call__(self, seq: str) -> _Domain:
        react = self.reaction_map[seq[self.react_slice]]
        aff = self.affinity_map[seq[self.aff_slice]]
        velo = self.velocity_map[seq[self.velo_slice]]
        is_bkwd = self.orientation_map[seq[self.orient_slice]]
        return CatalyticDomain(
            reaction=react, affinity=aff, velocity=velo, is_bkwd=is_bkwd
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(min_len=%r)" % (clsname, self.min_len)


class TransporterDomain(_Domain):
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


# TODO: do maps in init and only provide molecules
#       and number of codons for each region
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
    region_size = 4

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

        n_mol = _get_region_size(molecule_map, "molecule_map")
        n_aff = _get_region_size(affinity_map, "affinity_map")
        n_velo = _get_region_size(velocity_map, "velocity_map")
        n_orient = _get_region_size(orientation_map, "orientation_map")

        self.mol_slice = slice(0, n_mol)
        self.aff_slice = slice(n_mol, n_mol + n_aff)
        self.velo_slice = slice(n_mol + n_aff, n_mol + n_aff + n_velo)
        self.orient_slice = slice(
            n_mol + n_aff + n_velo, n_mol + n_aff + n_velo + n_orient,
        )

        self.min_len = n_mol + n_aff + n_velo + n_orient

    def __call__(self, seq: str) -> _Domain:
        mol = self.molecule_map[seq[self.mol_slice]]
        aff = self.affinity_map[seq[self.aff_slice]]
        velo = self.velocity_map[seq[self.velo_slice]]
        is_bkwd = self.orientation_map[seq[self.orient_slice]]
        return TransporterDomain(
            molecule=mol, affinity=aff, velocity=velo, is_bkwd=is_bkwd
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

    def __str__(self) -> str:
        loc = "transmembrane" if self.is_transmembrane else "cytosolic"
        eff = "inhibiting" if self.is_inhibiting else "activating"
        return f"ReceptorDomain({self.substrates[0]},{loc},{eff})"


# TODO: do maps in init and only provide molecules
#       and number of codons for each region
# TODO: have is_transmembrane and is_inhibiting as map as well
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
    region_size = 4

    def __init__(
        self,
        molecule_map: dict[str, Molecule],
        affinity_map: dict[str, float],
        is_transmembrane=False,
        is_inhibiting=False,
    ):
        self.is_transmembrane = is_transmembrane
        self.is_inhibiting = is_inhibiting
        self.molecule_map = molecule_map
        self.affinity_map = affinity_map

        n_mol = _get_region_size(molecule_map, "molecule_map")
        n_aff = _get_region_size(affinity_map, "affinity_map")

        self.mol_slice = slice(0, n_mol)
        self.aff_slice = slice(n_mol, n_mol + n_aff)

        self.min_len = n_mol + n_aff

    def __call__(self, seq: str) -> _Domain:
        mol = self.molecule_map[seq[self.mol_slice]]
        aff = self.affinity_map[seq[self.aff_slice]]
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

        self._hash = hash(tuple(self.domains))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

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

    def __hash__(self) -> int:
        return hash(self.genome)

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

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
