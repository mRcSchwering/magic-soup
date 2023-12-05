import warnings
from collections import Counter


class Molecule:
    """
    Represents a molecule species which is part of the world, can diffuse, degrade,
    and be converted into other molecules.

    Arguments:
        name: Used to uniquely identify this molecule species.
        energy: Energy for 1 mol of this molecule species (in J).
            This amount of energy is released if the molecule would be deconstructed to nothing.
        half_life: Half life of this molecule species in time steps (in s by default).
            Molecules degrade by one step if you call `world.degrade_molecules()`.
            Must be > 0.0.
        diffusivity: A measure for how quick this molecule species diffuses over the molecule map during each time step (in s by default).
            Molecules diffuse when calling `world.diffuse_molecules()`.
            0.0 would mean it doesn't diffuse at all.
            1.0 would mean it is spread out equally around its Moore's neighborhood within one time step.
        permeability: A measure for how quick this molecule species permeates cell membranes during each time step (in s by default).
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
    Additionally, it allows you to define overlapping chemistries without creating multiple molecule instances of the same molecule species.

    However, this also means that if 2 molecules have the same name, other attributes like e.g. energy must also match:

    ```
        atp = Molecule("ATP", 10)
        Molecule("ATP", 20)  # raises error
    ```

    By default in this simulation molecule numbers can be thought of as being in mM, time steps in s, energies in J/mol.
    Eventually, they are just numbers and can be interpreted as anything.
    However, together with the default parameters in [Kinetics][magicsoup.kinetics.Kinetics]
    it makes sense to interpret them in mM, s, and J.

    Molecule half life should represent the half life if the molecule is not actively deconstructed by a protein.
    Molecules degrade by one step whenever you call [degrade_molecules()][magicsoup.world.World.degrade_molecules].
    You can setup the simulation to always call [degrade_molecules()][magicsoup.world.World.degrade_molecules] whenever a time step is finished.

    Molecular diffusion in the 2D molecule map happens whenever you call [diffuse_molecules()][magicsoup.world.World.diffuse_molecules].
    The molecule map is `world.molecule_map`.
    It is a 3D tensor where dimension 0 represents all molecule species of the simulation.
    They are ordered in the same way the attribute `molecules` is ordered in the [Chemistry][magicsoup.containers.Chemistry] you defined.
    Dimension 1 represents x-positions and dimension 2 y-positions.
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
    Each cell lives on a certain pixel with x- and y-position.
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
    But it is possible to define more molecule species. Cells can use any molecule species in transporer or regulatory domains.

    Duplicate reactions and molecules will be removed on initialization.
    As any reaction can take place in both directions, it is not necessary to define both directions.
    To combine multiple chemistries you can do `both = chemistry1 & chemistry2` which will combine all molecules and reactions.

    The chemistry object is used by [World][magicsoup.world.World] to know what molecule species exist.
    Reactions and molecule species are used to set up the world and create mappings for domains.
    On the [world][magicsoup.world.World] object there are some tensors that refer to molecule species (e.g. `world.molecule_map`).
    The molecule ordering in such tensors is always the same as the ordering in `chemistry.molecules`.
    So, e.g. if `chemistry.molecules[2]` is pyruvate, `world.molecule_map[2]` refers to the pyruvate concentration
    of the world molecule map.
    For convenience mappings are provided:

    - `chemistry.mol_2_idx` to map a [Molecule][magicsoup.containers.Molecule] object
    - `chemistry.molname_2_idx` to map a [Molecule][magicsoup.containers.Molecule] name string to its index.

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
    Base Domain. All Domains should inherit from this class.

    Arguments:
        start: domain start on the CDS
        end: domain end on the CDS

    In the simulation domains for all proteins and cells exist as a set of tensors.
    This object is just a representation of a domain extracted from these tensors.
    You shouldn't need to instantiate it.
    These domain objects are created when calling _e.g._ [get_cell()][magicsoup.world.World.get_cell].

    Domain start and end are python indexes of the domain sequence.
    They subset the domain sequence from the CDS string.
    The index starts with 0, start is included, end is excluded.
    So, _e.g._ a domain with `start=3` and `end=18` is a domain
    that starts with the 4th nucleotide and ends with the 18th nucleotide
    on the CDS.
    """

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end


class DomainFact:
    """
    Domain factory base.
    All domain factories should inherit from this class.

    This is currently only used for [World.generate_genome()][magicsoup.world.World.generate_genome].
    Use this factory to describe a domain that should eventually be encoded.
    Domain factories can be stringed together into a protein in [ProteinFact][magicsoup.containers.ProteinFact]
    """


class CatalyticDomain(Domain):
    """
    Container for describing a catalytic domain.

    Arguments:
        reaction: Tuple of substrate and product molecule species that describe the reaction catalyzed by this domain.
            For stoichiometric coefficients > 1, list the molecule species multiple times.
        km: Michaelis Menten constant of the reaction (in mol).
        vmax: Maximum velocity of the reaction (in mol per time step).

    The catalytic domain is described for the direction from substrates to products.
    For the reverse reaction, the reciprocal of Km applies. Vmax stays the same.
    """

    def __init__(
        self,
        reaction: tuple[list[Molecule], list[Molecule]],
        km: float,
        vmax: float,
        **kwargs: int,
    ):
        super().__init__(**kwargs)
        subs, prods = reaction
        self.substrates = subs
        self.products = prods
        self.km = km
        self.vmax = vmax

    def __repr__(self) -> str:
        ins = ",".join(str(d) for d in self.substrates)
        outs = ",".join(str(d) for d in self.products)
        return f"CatalyticDomain({ins}<->{outs},Km={self.km:.2e},Vmax={self.vmax:.2e})"

    def __str__(self) -> str:
        subs_cnts = Counter(str(d) for d in self.substrates)
        prods_cnts = Counter([str(d) for d in self.products])
        subs_str = " + ".join([f"{d} {k}" for k, d in subs_cnts.items()])
        prods_str = " + ".join([f"{d} {k}" for k, d in prods_cnts.items()])
        return f"{subs_str} <-> {prods_str} | Km {self.km:.2e} Vmax {self.vmax:.2e}"


class CatalyticDomainFact(DomainFact):
    """
    Container for generating a catalytic domain.

    Arguments:
        reaction: Tuple of substrate and product molecule species that describe the reaction catalyzed by this domain.
            For stoichiometric coefficients > 1, list the molecule species multiple times.
        km: Desired Michaelis Menten constant of the transport (in mol).
        vmax: Desired Maximum velocity of the transport (in mol per time step).

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


class TransporterDomain(Domain):
    """
    Container for describing a transporter domain.

    Arguments:
        molecule: The molecule species which can be transported into or out of the cell by this domain.
        km: Michaelis Menten constant of the transport (in mol).
        vmax: Maximum velocity of the transport (in mol per time step).
        is_exporter: Whether the transporter is exporting this molecule species out of the cell.

    `is_exporter` is only relevant in combination with other domains on the same protein.
    It defines in which transport direction this domain is energetically coupled with others.
    """

    def __init__(
        self,
        molecule: Molecule,
        km: float,
        vmax: float,
        is_exporter: bool,
        **kwargs: int,
    ):
        super().__init__(**kwargs)
        self.molecule = molecule
        self.km = km
        self.vmax = vmax
        self.is_exporter = is_exporter

    def __repr__(self) -> str:
        sign = "exporter" if self.is_exporter else "importer"
        return f"TransporterDomain({self.molecule},Km={self.km:.2e},Vmax={self.vmax:.2e},{sign})"

    def __str__(self) -> str:
        sign = "exporter" if self.is_exporter else "importer"
        return f"{self.molecule} {sign} | Km {self.km:.2e} Vmax {self.vmax:.2e}"


class TransporterDomainFact(DomainFact):
    """
    Container for generating a transporter domain.

    Arguments:
        molecule: The molecule species which can be transported into or out of the cell by this domain.
        km: Desired Michaelis Menten constant of the transport (in mol).
        vmax: Desired Maximum velocity of the transport (in mol per time step).
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


class RegulatoryDomain(Domain):
    """
    Container for describing a regulatory domain.

    Arguments:
        effector: The molecule species which will be the effector molecule.
        hill: Hill coefficient describing degree of cooperativity
        km: ligand concentration producing half occupation
        is_inhibiting: Whether this is an inhibiting regulatory domain (otherwise activating).
        is_transmembrane: Whether this is also a transmembrane domain.
            If true, the domain will react to extracellular molecules instead of intracellular ones.
    """

    def __init__(
        self,
        effector: Molecule,
        hill: int,
        km: float,
        is_inhibiting: bool,
        is_transmembrane: bool,
        **kwargs: int,
    ):
        super().__init__(**kwargs)
        self.effector = effector
        self.km = km
        self.hill = int(hill)
        self.is_transmembrane = is_transmembrane
        self.is_inhibiting = is_inhibiting

    def __repr__(self) -> str:
        loc = "transmembrane" if self.is_transmembrane else "cytosolic"
        eff = "inhibiting" if self.is_inhibiting else "activating"
        return f"ReceptorDomain({self.effector},Km={self.km:.2e},hill={self.hill},{loc},{eff})"

    def __str__(self) -> str:
        loc = "[e]" if self.is_transmembrane else "[i]"
        post = "inhibitor" if self.is_inhibiting else "activator"
        return f"{self.effector}{loc} {post} | Km {self.km:.2e} Hill {self.hill}"


class RegulatoryDomainFact(DomainFact):
    """
    Container for generating a regulatory domain.

    Arguments:
        effector: The molecule species which will be the effector molecule.
        is_transmembrane: Whether this is also a transmembrane domain.
            If true, the domain will react to extracellular molecules instead of intracellular ones.
        km: Desired Michaelis Menten constant of the transport (in mol).
        hill: Hill coefficient describing degree of cooperativity (1, 3, or 5)
        is_inhibiting: Whether the effector will have an activating or inhibiting effect.

    `km` and `hill` are target values.
    Due to the way how codons are sampled for specific values,
    the final value for `km` and `hill` might differ.
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


class Protein:
    """
    Container for describing a protein.

    Arguments:
        domains: All domains of the protein
        cds_start: Start coordinate of its coding region
        cds_end: End coordinate of its coding region
        is_fwd: Whether its CDS is in the forward or reverse-complement of the genome.

    In the simulation proteins for all cells exist as a set of tensors.
    This object is just a representation of a single protein.
    You shouldn't need to instantiate it.
    Protein objects are created when calling _e.g._ [get_cell()][magicsoup.world.World.get_cell].

    CDS start and end are the indices that subset the CDS in the genome string.
    The index starts with 0, `cds_start` is included in the CDS, `cds_end` is not included.
    So, `cds_start=2` and `cds_end=31` describe a CDS whose first nucleotide is the 3rd basepair
    on the genome, and whose last nucleotide is the 31st basepair on the genome.

    `is_fwd` describes whether the CDS is found on the forward (hypothetical 5'-3')
    or the reverse-complement (hypothetical 3'-5') side of the genome.
    `cds_start` and `cds_end` always describe the parsing direction / the direction
    of the hypothetical transcriptase.
    So, if you want to visualize a `is_fwd=False` CDS on the genome in 5'-3' direction
    you have to do `n - cds_start` and `n - cds_stop` if `n` is the genome length.
    """

    def __init__(
        self, domains: list[Domain], cds_start: int, cds_end: int, is_fwd: bool
    ):
        self.domains = domains
        self.n_domains = len(domains)
        self.cds_start = cds_start
        self.cds_end = cds_end
        self.is_fwd = is_fwd

    def __repr__(self) -> str:
        kwargs = {
            "cds_start": self.cds_start,
            "cds_end": self.cds_end,
            "domains": self.domains,
        }
        args = [f"{k}:{repr(d)}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"

    def __str__(self) -> str:
        domstrs = [str(d).split(" | ")[0] for d in self.domains]
        return " | ".join(domstrs)


class ProteinFact:
    """
    Container for generating a protein

    Arguments:
        domain_facts: List of all domains, each described by a domain factory.

    This is currently only used for [World.generate_genome()][magicsoup.world.World.generate_genome].
    Use this factory to describe a protein that should eventually be encoded.
    Domain factories can be [CatalyticDomainFact][magicsoup.containers.CatalyticDomainFact],
    [TransporterDomainFact][magicsoup.containers.TransporterDomainFact],
    [RegulatoryDomainFact][magicsoup.containers.RegulatoryDomainFact].
    """

    def __init__(self, domain_facts: list[DomainFact] | DomainFact):
        if not isinstance(domain_facts, list):
            domain_facts = [domain_facts]
        self.domain_facts = domain_facts
        self.n_domains = len(domain_facts)
