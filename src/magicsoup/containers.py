from typing import Optional
import warnings
import torch


class Molecule:
    """
    Represents a molecule species which is part of the world, can diffuse, degrade,
    and be converted into other molecules.

    Arguments:
        name: Used to uniquely identify this molecule species.
        energy: In principle a standard free Gibbs energy of formation for 1 mol of this molecule species.
            This amount of energy is released if the molecule would be deconstructed.
            Energetically coupled in a protein it could power other activities.
        half_life: Half life of this molecule species in time steps.
            Molecules degrade by one step if you call `world.degrade_molecules()`.
            Must be > 0.0.
        diffusivity: A measure for how quick this molecule species diffuses over the molecule map during each time step.
            Molecules diffuse when calling `world.diffuse_molecules()`.
            0.0 would mean it doesn't diffuse at all.
            1.0 would mean it is spread out equally around its Moore's neighborhood within one time step.
        permeability: A measure for how quick this molecule species permeates cell membranes during each time step.
            Molecules permeate cell membranes when calling `world.diffuse_molecules()`.
            0.0 would mean it can't permeate cell membranes.
            1.0 would mean it spreads equally between cell and molecule map pixel within one time step.

    Each molecule species which is supposed to be unique should have a unique `name`.
    In fact, if you initialize a molecule with the same name multiple times,
    only one instance of this molecule will be created.

    ```
        atp = Molecule("ATP", 10)
        atp2 = Molecule("ATP", 10)
        assert atp is atp2
    ```

    This is used later on in the simulation to make efficient comparisons.
    It also allows you to define overlapping chemistries without creating multiple molecule instances of the same molecule species.

    However, this also means that if 2 molecules have the same name, other attributes like e.g. energy must also match:

    ```
        atp = Molecule("ATP", 10)
        Molecule("ATP", 20)  # raises error
    ```

    Molecule half life should represent the half life if the molecule is not actively deconstructed by a protein.
    Molecules degrade by one step whenever you call [degrade_molecules()][magicsoup.world.World.degrade_molecules].
    You can setup the simulation to always call [degrade_molecules()][magicsoup.world.World.degrade_molecules] whenever a time step is finished.
    You could define one time step to equal one second and then use the real half life value for your molecule species.

    Molecular diffusion in the 2D molecule map happens whenever you call [diffuse_molecules()][magicsoup.world.World.diffuse_molecules].
    The molecule map is `world.molecule_map`.
    It is a 3D tensor where dimension 0 represents all molecule species of the simulation.
    They are ordered in the same way the attribute `molecules` is ordered in the [Chemistry][magicsoup.containers.Chemistry] you defined.
    Dimension 1 represents x positions and dimension 2 y positions.
    Diffusion is implemented as a 2D convolution over the x-y tensor for each molecule species.
    This convolution has a 9x9 kernel.
    So, it alters the Moore's neighborhood of each pixel.
    How much of the center pixel's molecules are allowed to diffuse to the surrounding 8 pixels is defined by `diffusivity`.
    `diffusivity` is the ratio `a/b` when `a` is the amount of molecules diffusing to each of the 8 surrounding pixels,
    and `b` is the amount of molecules on the center pixel.
    Thus, `diffusivity=1.0` means all molecules of the center pixel are spread equally across the 9 pixels.

    Molecules permeating cell membranes also happens with [diffuse_molecules()][magicsoup.world.World.diffuse_molecules].
    Cell molecules are defined in `world.cell_molecules`.
    It is a 2D tensor where dimension 0 represents all cells and dimension 1 represents all molecule species.
    Again, molecule species are ordered in the same way the attribute `molecules` is ordered in the [Chemistry][magicsoup.containers.Chemistry] you defined.
    Dimension 0 always changes its length depedning on how cell replicate or die.
    The cell index (`cell.idx`) for any cell equals the index in `world.cell_molecules`.
    So, the amount of molecule species currently in cell with index 100 are defined in `world.cell_molecules[100]`.
    `permeability` defines how much molecules can permeate from `world.molecule_map` into `world.cell_molecules`.
    Each cell lives on a certain pixel with x and y position.
    And although there are already molecules on this pixel, the cell has its own molecules.
    You could imagine the cell as a bag of molecule hovering over the pixel.
    `permeability` allows molecules from that pixel in the molecule map to permeate into the cell that lives on that pixel (and vice versa).
    So e.g. if cell 100 lives on pixel 12, 450
    Molecules would be allowed to move from `world.molecule_map[:, 12, 450]` to `world.cell_molecules[100, :]`.
    Again, this happens separately for every molecule species depending on its `permeability` value.
    Specifically, `permeability` is the ratio of molecules that can permeate into the cell and the molecules that stay outside.
    Thus, a value of 1.0 means within one call half the molecules permeate into the cell.
    This permeation also happens the other way round, from inside the cell to the outside.
    """

    _instances: dict[str, "Molecule"] = {}

    def __new__(
        cls,
        name: str,
        energy: float,
        half_life: int = 100_000,
        diffusivity: float = 0.1,
        permeability: float = 0.0,
    ):
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
            if cls._instances[name].diffusivity != diffusivity:
                raise ValueError(
                    f"Trying to instantiate Molecule {name} with diffusivity {diffusivity}."
                    f" But {name} already exists with diffusivity {cls._instances[name].diffusivity}"
                )
            if cls._instances[name].permeability != permeability:
                raise ValueError(
                    f"Trying to instantiate Molecule {name} with permeability {permeability}."
                    f" But {name} already exists with permeability {cls._instances[name].permeability}"
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

    def __getnewargs__(self):
        # so that pickle can load instances
        return (
            self.name,
            self.energy,
            self.half_life,
            self.diffusivity,
            self.permeability,
        )

    def __init__(
        self,
        name: str,
        energy: float,
        half_life: int = 100_000,
        diffusivity: float = 0.1,
        permeability: float = 0.0,
    ):
        self.name = name
        self.energy = energy
        self.half_life = half_life
        self.diffusivity = diffusivity
        self.permeability = permeability
        self._hash = hash(self.name)

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, other) -> bool:
        return hash(self) < hash(other)

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        kwargs = {
            "name": self.name,
            "energy": self.energy,
            "half_life": self.half_life,
            "diffusivity": self.diffusivity,
            "permeability": self.permeability,
        }
        args = [f"{k}:{repr(d)}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"

    def __str__(self) -> str:
        return self.name

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "energy": self.energy,
            "half_life": self.half_life,
            "diffusivity": self.diffusivity,
        }


class Chemistry:
    """
    Container class that holds definition for the chemistry of the simulation.

    Arguments:
        molecules: List of all molecules species that are part of this simulation
        reactions: List of all possible reactions in this simulation as a list of tuples: `(substrates, products)`.
            All reactions can happen in both directions (left to right or vice versa).

    `molecules` should include at least all molecule species that are mentioned in `reactions`.
    But it is possible to define more molecule species. Cells could use any molecule species in transporer or regulatory domains.

    Duplicate reactions and molecules will be removed on initialization.
    As any reaction can take place in both directions, it is not necessary to define both directions.
    To combine multiple chemistries you can do `both = chemistry1 & chemistry2` which will combine all molecules and reactions.

    The chemistry object is used by [World][magicsoup.world.World] to know what molecule species exist.
    On initialization it is also passed to [Genetics][magicsoup.genetics.Genetics],
    which uses `chemistry` to know about each molecule species and reaction and then build domain factories for those.
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

    def to_dict(self) -> dict:
        return {
            "molecules": [d.to_dict() for d in self.molecules],
            "reactions": [
                ([d.name for d in a], [d.name for d in b]) for a, b in self.reactions
            ],
        }

    @classmethod
    def from_dict(cls, dct: dict) -> "Chemistry":
        mols = [Molecule(**d) for d in dct["molecules"]]
        name_2_mol = {d.name: d for d in mols}
        reacts = []
        for subs, prods in dct["reactions"]:
            reacts.append(
                ([name_2_mol[d] for d in subs], [name_2_mol[d] for d in prods])
            )
        return cls(molecules=mols, reactions=reacts)

    def __repr__(self) -> str:
        kwargs = {
            "molecules": self.molecules,
            "reactions": self.reactions,
        }
        args = [f"{k}:{repr(d)}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"


class Domain:
    """
    Container class that defines a domain.

    Domains should not be instantiated directly as `Domain`.
    [Genetics][magicsoup.genetics.Genetics] will create factories for all possible domains and use them when translating genomes.
    So, there should be no reason to instantiate them by hand.
    However, if you want to directly create domain objects, better use [CatalyticDomain][magicsoup.genetics.CatalyticDomain],
    [TransporterDomain][magicsoup.genetics.TransporterDomain], or [RegulatoryDomain][magicsoup.genetics.RegulatoryDomain].

    Arguments:
        is_catalytic: Flag to indicate that this is a catalytic domain.
        is_transporter: Flag to indicate that this is a transporter domain.
        is_regulatory: Flag to indicate that this is a regulatory domain.

    When cells are updated (or new cells are created) their genomes are translated into proteomes.
    These proteomes are lists of proteins, which in turn carry lists of domains.
    [World][magicsoup.world.World] then reads through these proteins and domains and sets the kinetic parameters for each cell accordingly.
    """

    def __init__(
        self,
        is_catalytic: bool = False,
        is_transporter: bool = False,
        is_regulatory: bool = False,
    ):
        self.is_catalytic = is_catalytic
        self.is_transporter = is_transporter
        self.is_regulatory = is_regulatory

    def __repr__(self) -> str:
        kwargs = {
            "is_catalytic": self.is_catalytic,
            "is_transporter": self.is_transporter,
            "is_regulatory": self.is_regulatory,
        }
        args = [f"{k}:{repr(d)}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"


class CatalyticDomain(Domain):
    """
    Container holding the specification for a catalytic domain.
    Usually, you don't need to manually instantiate domains.
    During the simulation they are automatically instantiated through factories.

    Arguments:
        reaction: Tuple of substrate and product molecule species that describe the reaction catalyzed by this domain.
            For stoichiometric coefficients > 1, list the molecule species multiple times.
        affinity: Michaelis Menten constant of the reaction (in mol).
        velocity: Maximum velocity of the reaction (in mol per time step).
    """

    def __init__(
        self,
        reaction: tuple[list[Molecule], list[Molecule]],
        affinity: float,
        velocity: float,
    ):
        super().__init__(is_catalytic=True)
        subs, prods = reaction
        self.substrates = subs
        self.products = prods
        self.affinity = affinity
        self.velocity = velocity

    def __str__(self) -> str:
        ins = ",".join(str(d) for d in self.substrates)
        outs = ",".join(str(d) for d in self.products)
        return f"CatalyticDomain({ins}->{outs})"


class TransporterDomain(Domain):
    """
    Container holding the specification for a transporter domain.
    Usually, you don't need to manually instantiate domains.
    During the simulation they are automatically instantiated through factories.

    Arguments:
        molecule: The molecule species which can be transported into or out of the cell by this domain.
        affinity: Michaelis Menten constant of the transport (in mol).
        velocity: Maximum velocity of the transport (in mol per time step).
        is_bkwd: Flag indicating whether in which orientation this transporter will be coupled with other domains.
    """

    def __init__(
        self, molecule: Molecule, affinity: float, velocity: float, is_bkwd: bool
    ):
        super().__init__(is_transporter=True)
        self.molecule = molecule
        self.affinity = affinity
        self.velocity = velocity
        self.is_bkwd = is_bkwd

    def __str__(self) -> str:
        return f"TransporterDomain({self.molecule})"


class RegulatoryDomain(Domain):
    """
    Container holding the specification for a regulatory domain.
    Usually, you don't need to manually instantiate domains.
    During the simulation they are automatically instantiated through factories.

    Arguments:
        effector: The molecule species which will be the effector molecule.
        affinity: Michaelis Menten constant of the transport (in mol).
        is_inhibiting: Whether this is an inhibiting regulatory domain (otherwise activating).
        is_transmembrane: Whether this is also a transmembrane domain.
            If true, the domain will react to extracellular molecules instead of intracellular ones.

    I think the term Michaelis Menten constant in a regulatory domain is a bit weird
    since there is no product being created.
    However, the kinetics of the amount of activation or inhibition are the same.
    """

    def __init__(
        self,
        effector: Molecule,
        affinity: float,
        is_inhibiting: bool,
        is_transmembrane: bool,
    ):
        super().__init__(is_regulatory=True)
        self.effector = effector
        self.affinity = affinity
        self.is_transmembrane = is_transmembrane
        self.is_inhibiting = is_inhibiting

    def __str__(self) -> str:
        loc = "transmembrane" if self.is_transmembrane else "cytosolic"
        eff = "inhibiting" if self.is_inhibiting else "activating"
        return f"ReceptorDomain({self.effector},{loc},{eff})"


# TODO: reprs for domains


class Protein:
    """
    Container class to carry domains of a protein.

        domains: All domains of the protein
        label: Can be used to label this protein. Has no effect.

    If you print a protein, it will show all its domains in detail.
    But if you convert it to a string (e.g. `str(protein)` or `f"{protein}"`) it will only show the label.
    """

    def __init__(self, domains: list[Domain], label: str = "P"):
        self.domains = domains
        self.label = label
        self.n_domains = len(domains)

    def __repr__(self) -> str:
        kwargs = {
            "domains": self.domains,
            "label": self.label,
        }
        args = [f"{k}:{repr(d)}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"

    def __str__(self) -> str:
        return self.label


class Cell:
    """
    Object representing a cell with its environment.

    Arguments:
        genome: Full genome sequence of this cell.
        proteome: List of proteins this cell has.
        position: Position on the cell map.
        idx: The current index of this cell.
        label: Label which can be used to track cells. Has no effect.
        n_survived_steps: Number of time steps this cell has survived.
        n_replications: Number of times this cell has divided itself.

    When new cells are added to [World][magicsoup.world.World] (directly or by replication),
    they are automatically placed, their genomes are translated, and they are given an index.
    Then they are added as an entry to a list `world.cells`.
    So, usually you would not directly instantiate a cell.

    [World][magicsoup.world.World] maintains this list of cells `world.cells` during the simulation,
    removing killed cells and adding new or replicated cells.
    However, for performance reasons not all attributes on this list of cell objects is always updated.
    If you want to get a cell object with all its current attributes, use [get_cell()][magicsoup.world.World].

    Apart from the initialization arguments, there are some other useful attributes:
    - `int_molecules` Intracellular molecules.
      A tensor with one dimension that represents each molecule species in the same order as defined in [Chemistry][magicsoup.containers.Chemistry].
    - `ext_molecules` Extracellular molecules.
      A tensor with one dimension that represents each molecule species in the same order as defined in [Chemistry][magicsoup.containers.Chemistry].
      These are the molecules of the pixel the cell is currently living on.

    A map of all pixels occupied by a cell exists as a tensor `world.cell_map`.
    A map of all extracellular molecules exists as tensor `world.molecule_map`.
    All intracellular molecules are defined as a tensor `world.cell_molecules`.
    `world.cell_survival` and `world.cell_divisions` are tensors that define how many round each cell has survived
    and how many times each cell was replicated.
    For performance reasons the simulation is working with these tensors.
    Their values are not copied into the list of cell objects `world.cells` during the simulation.
    By default the cell objects in `world.cells` only always have a genome, proteome, and a position.
    If you also want to know the other attributes of a specific cell use [get_cell()][magicsoup.world.World].

    During cell division this simulation has a concept of parent and child.
    The parent is the cell that stays on the same pixel, while the child is the new cell that will occupy another pixel.
    The child will have `n_survived_steps=0` and `n_replications=0` when it is born.
    The parent will keep these values. Both cells will have the same genome and proteome.
    """

    def __init__(
        self,
        genome: str,
        proteome: list[Protein],
        position: tuple[int, int] = (-1, -1),
        idx: int = -1,
        label: str = "C",
        n_survived_steps: int = -1,
        n_replications: int = -1,
    ):
        self.genome = genome
        self.proteome = proteome
        self.label = label
        self.position = position
        self.idx = idx
        self.n_survived_steps = n_survived_steps
        self.n_replications = n_replications
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
            "n_replications": self.n_replications,
        }
        return Cell(**{**old_kwargs, **kwargs})  # type: ignore

    def __repr__(self) -> str:
        kwargs = {
            "genome": self.genome,
            "proteome": self.proteome,
            "position": self.position,
            "idx": self.idx,
            "label": self.label,
            "n_survived_steps": self.n_survived_steps,
            "n_replications": self.n_replications,
        }
        args = [f"{k}:{repr(d)}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"
