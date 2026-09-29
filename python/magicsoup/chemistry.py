import warnings


class Molecule:
    """
    Represents a molecule species which is part of the world, can diffuse, degrade,
    and be converted into other molecules.

    Parameters:
        name: Used to uniquely identify this molecule species.
        energy: Energy for 1 mol of this molecule species (in J).
            The hypothetical amount of energy that is released if the molecule would be fully deconstructed.
        half_life: Half life of this molecule species in time steps (s by default).
            Molecules degrade by one step if you call [degrade_molecules()][magicsoup.world.World.degrade_molecules].
            Must be > 0.0.
        diffusivity: A measure for how quick this molecule species diffuses over the molecule map during each time step (s by default).
            Molecules diffuse when calling [diffuse_molecules()][magicsoup.world.World.diffuse_molecules].
            0.0 means it doesn't diffuse at all.
            1.0 means it is spread out equally around its Moore's neighborhood within one time step.
        permeability: A measure for how quick this molecule species permeates cell membranes during each time step (s by default).
            Molecules permeate cell membranes when calling [diffuse_molecules()][magicsoup.world.World.diffuse_molecules].
            0.0 means it can't permeate cell membranes.
            1.0 means it spreads equally between cell and molecule map pixel within one time step.

    Each molecule species which is supposed to be unique should have a unique `name`.
    In fact, if you initialize a molecule with the same name multiple times,
    only one instance of this molecule will be created.
    It allows you to define overlapping chemistries without creating multiple molecule instances of the same molecule species.
    You can also get a molecule instance from its name alone ([Molecule.from_name()][magicsoup.containers.Molecule.from_name]).
    However, this also means that if 2 molecules have the same name all attributes must match.

    ```
    atp = Molecule("ATP", 10)
    atp2 = Molecule("ATP", 10)
    assert atp is atp2

    atp3 = Molecule.from_name("ATP")
    assert atp3 is atp

    Molecule("ATP", 20)  # error because energy is different
    ```

    By default in this simulation molecule numbers can be thought of as being in mM, time steps in s, energies in J/mol.
    Eventually, they are just numbers and can be interpreted as anything.
    However, together with the default parameters in [Kinetics][magicsoup.kinetics.Kinetics]
    it makes sense to interpret them in mM, s, and J.

    Molecule half life should represent the half life if the molecule is not actively deconstructed by a protein.
    Molecules degrade by one step whenever you call [degrade_molecules()][magicsoup.world.World.degrade_molecules].
    You can setup the simulation to always call [degrade_molecules()][magicsoup.world.World.degrade_molecules] whenever a time step is finished.

    Molecular diffusion in the 2D molecule map happens whenever you call [diffuse_molecules()][magicsoup.world.World.diffuse_molecules].
    The molecule map is `world.molecule_map` (see [World][magicsoup.world.World]).
    It is a 3D tensor where dimension 0 represents all molecule species of the simulation.
    They are ordered in the same way the attribute `molecules` is ordered in the [Chemistry][magicsoup.containers.Chemistry] you defined.
    Dimension 1 represents x-positions and dimension 2 y-positions.
    Diffusion is implemented as a 2D convolution over the x-y tensor for each molecule species.
    This convolution has a 9x9 kernel.
    So, it alters the Moore's neighborhood of each pixel.
    How much of the center pixel's molecules are allowed to diffuse to the surrounding 8 pixels is defined by `diffusivity`.
    `diffusivity` is the ratio `a/b` when `a` is the amount of molecules diffusing to each of the 8 surrounding pixels,
    and `b` is the amount of molecules that stays on the center pixel.
    Thus, `diffusivity=1.0` means all molecules of the center pixel are spread equally across the 9 pixels.

    Molecules permeating cell membranes also happens with [diffuse_molecules()][magicsoup.world.World.diffuse_molecules].
    Cell molecules are defined in `world.cell_molecules` (see [World][magicsoup.world.World]).
    It is a 2D tensor where dimension 0 represents all cells and dimension 1 represents all molecule species.
    Again, molecule species are ordered in the same way the attribute `molecules` is ordered in the [Chemistry][magicsoup.containers.Chemistry] you defined.
    Dimension 0 always changes its length depedning on how cells replicate or die.
    The cell index (`cell.idx` of [Cell][magicsoup.containers.Cell]) for any cell equals the index in `world.cell_molecules`.
    _E.g._ the amount of molecule species currently in cell with index 100 is defined as `world.cell_molecules[100]`.
    `permeability` defines how much molecules can permeate from `world.molecule_map` into `world.cell_molecules`.
    Each cell lives on a certain pixel with x- and y-position.
    And although there are already molecules on this pixel, the cell has its own molecules.
    You could imagine the cell as a bag of molecule hovering over that pixel.
    `permeability` allows molecules from that pixel in the molecule map to permeate into the cell that lives on that pixel (and vice versa).
    So e.g. if cell 100 lives on pixel 12, 450
    Molecules would be allowed to move from `world.molecule_map[:, 12, 450]` to `world.cell_molecules[100, :]`.
    Again, this happens separately for every molecule species depending on the `permeability` value.
    Specifically, `permeability` is the ratio of molecules that can permeate into the cell and the molecules that stay outside.
    Thus, a value of 1.0 means within one time step this molecule species spreads evenly between molecule map and cell.
    """

    _instances: dict[str, "Molecule"] = {}  # noqa: RUF012

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

    @classmethod
    def from_name(cls, name: str) -> "Molecule":
        """Get Molecule instance from its name (if it has already been defined)"""
        if name not in Molecule._instances:
            raise ValueError(f"Molecule {name} was not defined yet")
        return Molecule._instances[name]

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
        self.energy = float(energy)  # int would error out in kinetics
        self.half_life = half_life
        self.diffusivity = diffusivity
        self.permeability = permeability
        self._hash = hash(self.name)

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, other: "Molecule") -> bool:
        return self.name < other.name

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
        args = [f"{k}:{d!r}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"

    def __str__(self) -> str:
        return self.name


class Chemistry:
    """
    Represents the chemistry with molecules and reactions available in a simulation.

    Parameters:
        molecules: List of all [Molecules][magicsoup.containers.Molecule] that are part of this simulation
        reactions: List of all possible reactions in this simulation as a list of tuples: `(substrates, products)`
            where both `substrates` and `products` are lists of [Molecules][magicsoup.containers.Molecule].
            All reactions can happen in both directions (left to right or vice versa).

    `molecules` should include at least all molecule species that are mentioned in `reactions`.
    But it is possible to define more molecule species.
    Cells can use molecule species in transporer and regulatory domains, even if they are not included in any reaction.
    For stoichiometric coefficients > 1, list the molecule species multiple times.
    E.g. for `2A + B <-> C` use `reaction=([A, A, B], [C])`
    when `A,B,C` are molecule A, B, C instances.

    Duplicate reactions and molecules will be removed on initialization.
    As any reaction can take place in both directions, it is not necessary to define both directions.
    You can use `__and__` to combine multiple chemistries:

    ```
    both = chemistry1 & chemistry2  # union of molecules and reactions
    ```

    The chemistry object is used by [World][magicsoup.world.World] to know what molecule species exist.
    Reactions and molecule species are used to set up the world and create mappings for domains.
    On the [world][magicsoup.world.World] object there are some tensors that refer to molecule species (e.g. `world.molecule_map`).
    The molecule ordering in such tensors is always the same as the ordering in `chemistry.molecules`.
    E.g. if `chemistry.molecules[2]` is pyruvate, `world.molecule_map[2]` refers to pyruvate concentrations on the world molecule map.
    For convenience mappings are provided on this object:

    - `chemistry.mol_2_idx` to map a [Molecule][magicsoup.containers.Molecule] object to its index
    - `chemistry.molname_2_idx` to map a [Molecule][magicsoup.containers.Molecule] name string to its index
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

        self.mol_2_idx = {d: i for i, d in enumerate(self.molecules)}
        self.molname_2_idx = {d.name: i for i, d in enumerate(self.molecules)}

    def __and__(self, other: "Chemistry") -> "Chemistry":
        return Chemistry(
            molecules=self.molecules + other.molecules,
            reactions=self.reactions + other.reactions,
        )

    def __repr__(self) -> str:
        kwargs = {
            "molecules": self.molecules,
            "reactions": self.reactions,
        }
        args = [f"{k}:{d!r}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"
