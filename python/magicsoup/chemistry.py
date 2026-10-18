from collections import Counter


class Molecule:
    """
    Represents a molecule species

    Parameters:
        name: Used to uniquely identify this molecule species.
        energy: Energy of this molecule species in kJ/mol (kilo Joule per mol)
            The hypothetical amount of energy that is released if the molecule would be fully deconstructed.
        half_life: Half-life of this molecule species in days (-1 for no degradation).
        diffusivity: Diffusion coefficient of this molecule species
            in water at used temperature in cm²/s (square centi meter per second)
        permeability: Permeation coefficient of this molecule species
            through cell membranes at used temperature in cm/s (centi meter per second)

    Each molecule species defined in a [Chemistry][magicsoup.chemistry.Chemistry] must have a unique name.
    Molecule half-life, diffusivity, permeability are all parameters for how this molecule species behaves in the simulation.
    See the respective [Degradation][magicsoup.physics.Degradation], [Diffusion][magicsoup.physics.Diffusion], [Permeation][magicsoup.physics.Permeation]
    classes for details.

    Molecule energy defines in which direction a chemical reaction progresses.
    A reaction's energy is the difference in sums of energies of all product and all substrates.
    Proteins can only catalyze reaction which were defined in [Chemistry][magicsoup.chemistry.Chemistry].
    But since they can couple multiple reactions, one reaction can progress against its energy gradient
    if it is powered by another reaction that progresses along its energy gradient.
    """

    def __init__(
        self,
        name: str,
        energy: float,  # kJ/mol
        half_life: int = -1,  # days
        diffusivity: float = 6.7 * 1e-6,  # cm²/s
        permeability: float = 2.4 * 1e-10,  # cm/s
    ) -> None:
        self.name = name
        self.energy = energy
        self.half_life = half_life
        self.diffusivity = diffusivity
        self.permeability = permeability
        self._hash = hash(self.name)

    def __hash__(self) -> int:
        return self._hash

    def __lt__(self, other: "Molecule") -> bool:
        return self.name < other.name

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        same_name = hash(self) == hash(other)
        if not same_name:
            return False

        attrs = {
            "energy": (self.energy, other.energy),
            "half_life": (self.half_life, other.half_life),
            "diffusivity": (self.diffusivity, other.diffusivity),
            "permeability": (self.permeability, other.permeability),
        }
        for attr, (a, b) in attrs.items():
            if a != b:
                raise ValueError(
                    f"2 molecule species defined as {self.name}"
                    f" have different {attr}"
                )

        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name})"

    def __str__(self) -> str:
        return self.name


class Reaction:
    """
    Represents a reaction

    Parameters:
        substrates: List of [Molecules][magicsoup.containers.Molecule] that are consumed in this reaction
        products: List of [Molecules][magicsoup.containers.Molecule] that are produced in this reaction

    All reactions are reversible, meaning that they can happen in both directions (left to right or vice versa).
    Also, the order in which substrates and products are listed does not matter.
    Thus, `Reaction([A, B], [C])` is equal to `Reaction([B, A], [C])` and also equal to `Reaction([C], [A, B])`.

    For stoichiometric coefficients > 1, list the molecule species multiple times.
    E.g. for `2A + B <-> C` use `Reaction([A, A, B], [C])` or equally `Reaction([A]*2 + [B], [C])`.
    """

    def __init__(self, substrates: list[Molecule], products: list[Molecule]) -> None:
        self.substrates = substrates
        self.products = products
        self._hash = hash(
            tuple(sorted([tuple(sorted(substrates)), tuple(sorted(products))]))
        )

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return hash(self) == hash(other)

    def __repr__(self) -> str:
        subs_cntr = Counter(self.substrates)
        prods_cntr = Counter(self.products)
        subs = [f"{n} {m}" if n > 1 else str(m) for m, n in subs_cntr.items()]
        prods = [f"{n} {m}" if n > 1 else str(m) for m, n in prods_cntr.items()]
        return f"{type(self).__name__}(({subs}), ({prods}))"

    def __str__(self) -> str:
        subs_cntr = Counter(self.substrates)
        prods_cntr = Counter(self.products)
        subs = [f"{n} {m}" if n > 1 else str(m) for m, n in subs_cntr.items()]
        prods = [f"{n} {m}" if n > 1 else str(m) for m, n in prods_cntr.items()]
        return f"{' + '.join(subs)} <-> {' + '.join(prods)}"


class Chemistry:
    """
    Represents the chemistry with molecules and reactions available in a simulation.

    Parameters:
        molecules: List of all [Molecules][magicsoup.containers.Molecule] that are part of this simulation
        reactions: List of all possible reactions in this simulation given as list of [Reactions][magicsoup.containers.Reaction]

    `molecules` must include at least all molecule species that are used in `reactions`.
    Duplicate reactions and molecules will be removed on initialization.
    Use `__and__` to combine multiple chemistries:

    ```
    both = chemistry1 & chemistry2  # union of molecules and reactions
    ```

    The chemistry object is used throughout the simulation to refer to molecules.
    All tensors that refer to molecule species (_e.g._ molecule concentrations) always order
    each molecule concentration according to the order of `chemistry.molecules`.
    E.g. if `chemistry.molecules[2]` is pyruvate, `culture.molecule_map[2]` ([Culture][magicsoup.culture.Culture])
    are pyruvate concentrations on in the cell culture map and `cells.molecules[2]` ([Cells][magicsoup.cells.Cells])
    are intracellular pyruvate concentrations.
    For convenience mappings are provided:

    - `chemistry.mol_2_idx` to map a [Molecule][magicsoup.containers.Molecule] instance to its index
    - `chemistry.molname_2_idx` to map a [Molecule][magicsoup.containers.Molecule] name to its index
    """

    def __init__(
        self,
        molecules: list[Molecule],
        reactions: list[Reaction],
    ) -> None:
        # remove duplicates while keeping order
        self.molecules = list(dict.fromkeys(molecules))
        self.reactions = list(dict.fromkeys(reactions))

        # check for undefined molecules in reactions
        dfnd_mols = set(molecules)
        react_mols = set(
            [dd for d in self.reactions for dd in d.substrates]
            + [dd for d in self.reactions for dd in d.products]
        )
        if react_mols > dfnd_mols:
            raise ValueError(
                "These molecules were not defined but are part of some reactions:"
                f" {', '.join(str(d) for d in react_mols - dfnd_mols)}."
            )

        self.mol_2_idx = {d: i for i, d in enumerate(self.molecules)}
        self.molname_2_idx = {d.name: i for i, d in enumerate(self.molecules)}
        self.molname_2_mol = {d.name: d for d in self.molecules}

    def __and__(self, other: "Chemistry") -> "Chemistry":
        return Chemistry(
            molecules=self.molecules + other.molecules,
            reactions=self.reactions + other.reactions,
        )

    def __repr__(self) -> str:
        args = [
            f"{len(self.molecules):d} molecules",
            f"{len(self.reactions):d} reactions",
        ]
        return f"{type(self).__name__}({','.join(args)})"
