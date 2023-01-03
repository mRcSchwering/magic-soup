from typing import Optional
import warnings
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
        self.idx_ext = -1

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


class Chemistry:
    """
    Container class that holds definition for basic chemistry of simulation.
    
    - `molecules` list of all molecules species that are part of this simulation
    - `reactions` list of all possible reactions in this simulation as a list of tuples: `(substrates, products)`.
                  All reactions can happen in both directions (left to right or vice versa).
    """

    def __init__(
        self,
        molecules: list[Molecule],
        reactions: list[tuple[list[Molecule], list[Molecule]]],
    ):
        self.molecules = molecules
        self.reactions = reactions

        dfnd_mols = set(molecules)
        react_mols = set()
        for substs, prods in reactions:
            for mol in substs:
                react_mols.add(mol)
            for mol in prods:
                react_mols.add(mol)
        if react_mols > dfnd_mols:
            raise ValueError(
                "These molecules were not defined but are part of some reactions:"
                f" {', '.join(str(d) for d in react_mols - dfnd_mols)}."
                "Please define all molecules."
            )


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
    - `is_bkwd` Bool which decides in which direction the domain will be coulpled.
      This is relevant for proteins with multiple domains. Depending on molecule
      concentrations and energies the overall reaction catalyzed and/or performed
      by the protein can occur from left to right, or right to left. Whether a domain's
      substrates will be part of the left or the right side of the equation, is decided
      by this bool.
    - `label` a label to recognize it, has no effect
    - `is_catalytic` Flag to indicate that this is a catalytic domain.
    - `is_transporter` Flag to indicate that this is a transporter domain.
    - `is_regulatory` Flag to indicate that this is a regulatory domain.
    - `is_inhibiting` Flag to indicate that this is a inhibiting domain. This is only
      relevant for regulatory domains.

    This class shouldn't be instantiated directly. There are factories that build the appropriate
    domain from a nucleotide sequence in `genetics`.
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
        is_regulatory=False,
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
        self.is_regulatory = is_regulatory
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
                self.is_regulatory,
                self.is_inhibiting,
                self.is_transmembrane,
            )
        )

    def __lt__(self, other: "Domain") -> bool:
        return hash(self) < hash(other)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(substrates=%r,products=%r,affinity=%r,velocity=%r,is_bkwd=%r,label=%r,is_catalytic=%r,is_transporter=%r,is_regulatory=%r,is_inhibiting=%r,is_transmembrane=%r)"
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
                self.is_regulatory,
                self.is_inhibiting,
                self.is_transmembrane,
            )
        )

    def __str__(self) -> str:
        ins = ",".join(str(d) for d in self.substrates)
        outs = ",".join(str(d) for d in self.products)
        return f"Domain({ins}<->{outs})"


class Protein:
    """
    Container class to carry domains of a protein
    
    - `domains` all domains of the protein
    - `label` a label, only to recognize it, has no effect
    """

    def __init__(self, domains: list[Domain], label="P"):
        self.domains = domains
        self.label = label
        self.n_domains = len(domains)

        self._hash = hash(tuple(sorted(self.domains)))

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
