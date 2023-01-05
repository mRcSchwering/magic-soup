from typing import Optional
import warnings
import torch


class Molecule:
    """
    Represents a type of molecule which is part of the world, can diffuse,
    degrade, and be converted into other molecules.

    - `name` used to uniquely identify a molecule species.
    - `energy` Fictional standard free Gibbs energy this molecule is supposed to have.
      This amount of energy is released is the molecule would be deconstructed.
      It can be energetically coupled to other activities in a protein.
    - `half_life` Half life of this molecule. Molecules degrade by one step if you
      call `world.degrade_molecules()`. Must be > 0.0.
    - `diff_coef` Diffusion coefficient for this molecule. Molecules diffuse by one step if you
      call `world.diffuse_molecules()`. 0.0 means no diffusion at all. With the current implementation
      1e-6 is the maximum, where all molecules are spread out equally around the pixel's Moor's neighborhood.
    
    Each type of molecule which is supposed to be unique should have a unique `name`.
    This is later on used as a mechanism to compare molecules efficiently.
    In fact, if you initialize a molecule with the same name multiple times, only one
    instance of this molecule will be created. E.g.

    ```
        atp = Molecule("ATP", 10)
        atp2 = Molecule("ATP", 10)
        assert atp is atp2
    ```

    This is used later on in the simulation to make efficient comparisons.
    It also allows you to define overlapping chemistries without creating multiple
    molecules of the same molecule species.
    
    However, this also means that if 2 molecules have the same name, other attributes
    like e.g. energy must also match:

    ```
        atp = Molecule("ATP", 10)
        Molecule("ATP", 20)  # raises error
    ```

    Molecule half life should represent the half life if the molecule is not actively deconstructed
    by a protein. Molecules degrade by one step whenever you call `world.degrade_molecules()`.
    You can setup the simulation to always call `world.degrade_molecules()` whenever a time step
    is finished. Assuming 1 time step represents 1s, values around 1e5 are reasonable for small molecules.
    You could increase this number to represent less stable molecules.

    Molecular diffusion happens only on the molecule map (`world.molecule_map`). All molecule species are
    diffused by one step if you call `world.diffuse_molecules()`. If you do this call once at the end of every
    time step, and you assume that one time step represents 1s, and each pixel in the map represents 10um x 10um,
    then values around 1e-8 are a reasonable guess for small molecules. In the current implementation molecules
    diffuse only around Moore's neighborhood and `diff_coef` is essentially multiplied with 1e6 and then used
    as a factor to determine the kernel. This means values greater than 1e-6 don't increase diffusion. At that
    point all molecules are already equally spread out in Moore's neighborhood at each call.

    There is another attribute `idx` on molecules worth mentioning. Initially it is `-1`. Once `World` is
    instantiated with a chemistry object, all unique molecules get an unique index `idx`. This index can
    be used to access molecule concentrations in `world.molecule_map` or `world.cell_molecules`.

    ```
        atp = Molecule("ATP", 10)
        ...  # define other molecules and reactions
        
        chemistry = Chemistry(molecules=molecules, reactions=reactions)
        world = World(chemistry=chemistry)  # now molecule indexes are set

        world.molecule_map[atp.idx]  # current world map for ATP activites
        world.cell_molecules[0, atp.idx]  # current ATP activity in cell 0
    ```

    You can use this to measure or change molecules in cells or generally anywhere on the map.
    """

    _instances: dict[str, "Molecule"] = {}

    def __new__(cls, name: str, energy: float, half_life=100_000, diff_coef=1e-8):
        if name in cls._instances:
            if cls._instances[name].energy != energy:
                raise ValueError(
                    f"Trying to instantiate Molecule {name} with energy {energy}."
                    f" But {name} already exists with energy {cls._instances[name].energy}"
                )
            if cls._instances[name].half_life != half_life:
                raise ValueError(
                    f"Trying to instantiate Molecule {name} with half_life {half_life}."
                    f" But {name} already exists with half_life {cls._instances[name].half_life}"
                )
            if cls._instances[name].diff_coef != diff_coef:
                raise ValueError(
                    f"Trying to instantiate Molecule {name} with diff_coef {diff_coef}."
                    f" But {name} already exists with diff_coef {cls._instances[name].diff_coef}"
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

    def __init__(self, name: str, energy: float, half_life=100_000, diff_coef=1e-8):
        self.name = name
        self.energy = energy
        self.half_life = half_life
        self.diff_coef = diff_coef
        self.idx = -1
        self.idx_ext = -1

        self._hash = hash(self.name)

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, other) -> bool:
        return hash(self) < hash(other)

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(name=%r,energy=%rhalf_life=%r,diff_coef=%r)" % (
            clsname,
            self.name,
            self.energy,
            self.half_life,
            self.diff_coef,
        )

    def __str__(self) -> str:
        return self.name


class Chemistry:
    """
    Container class that holds definition for basic chemistry of simulation.
    
    - `molecules` list of all molecules species that are part of this simulation
    - `reactions` list of all possible reactions in this simulation as a list of tuples: `(substrates, products)`.
                  All reactions can happen in both directions (left to right or vice versa).

    `molecules` should include at least all molecule species that are mentioned in `reactions`.
    As any reaction can take place in both directions, it is not necessary to define both directions.

    Duplicate reactions and molecules will be removed on initialization. To combine multiple chemistries
    you can do `both = chemistry1 & chemistry2` which will combine all molecules and reactions.
    """

    def __init__(
        self,
        molecules: list[Molecule],
        reactions: list[tuple[list[Molecule], list[Molecule]]],
    ):
        # remove duplicates while keeping order
        self.molecules = list(dict.fromkeys(molecules))
        hash_reacts = [(tuple(sorted(s)), tuple(sorted(p))) for s, p in reactions]
        unq_reacts = list(dict.fromkeys(hash_reacts))
        self.reactions = [(list(s), list(p)) for s, p in unq_reacts]

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

    def __and__(self, other: "Chemistry") -> "Chemistry":
        return Chemistry(
            molecules=self.molecules + other.molecules,
            reactions=self.reactions + other.reactions,
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(molecules=%r,eactions=%r)" % (
            clsname,
            self.molecules,
            self.reactions,
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
