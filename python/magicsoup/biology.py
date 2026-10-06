from collections import Counter
from typing import TYPE_CHECKING, Protocol

import torch

from magicsoup.chemistry import Molecule

if TYPE_CHECKING:
    from magicsoup.world import World


class DomainType(Protocol):
    """Protocol for domains"""

    start: int
    end: int

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, dct: dict) -> "DomainType": ...


class CatalyticDomain:
    """
    Object describing a catalytic domain.

    Parameters:
        reaction: Tuple `(substrates, products)` where both `substrates` and `products`
            are lists of [Molecules][magicsoup.containers.Molecule] which are involved in the reaction.
            Stoichiometric coefficients > 1 are defined by listing molecules multiple times.
        km: Michaelis Menten constant of the reaction (in mM).
        vmax: Maximum velocity of the reaction (in mmol/s).
        start: Domain start on the CDS
        end: Domain end on the CDS

    The simulation works with tensor representations of cell proteomes.
    This class is a helper which makes interpreting a cell's proteome easier.
    You should not instantiate this class but instead get it from `cell.proteome`
    (see [Cell][magicsoup.containers.Cell]).

    Domain start and end describe the slice of the CDS python string.
    _I.e._ the index starts with 0, start is included, end is excluded.
    _E.g._ `start=3`, `end=18` starts with the 4th and ends with the 18th base pair on the CDS.
    """

    def __init__(
        self,
        reaction: tuple[list[Molecule], list[Molecule]],
        km: float,
        vmax: float,
        start: int,
        end: int,
    ):
        self.start = start
        self.end = end
        subs, prods = reaction
        self.substrates = subs
        self.products = prods
        self.km = km
        self.vmax = vmax

    def to_dict(self) -> dict:
        """Get dict representation of domain"""
        reaction = (
            [d.name for d in self.substrates],
            [d.name for d in self.products],
        )
        kwargs = {
            "reaction": reaction,
            "km": self.km,
            "vmax": self.vmax,
            "start": self.start,
            "end": self.end,
        }
        return {"type": "C", "spec": kwargs}

    @classmethod
    def from_dict(cls, dct: dict) -> "CatalyticDomain":
        """
        Convencience method for creating an instance from a dict.
        All parameters must be present as keys.
        Molecules are provided by their name.
        """
        lft, rgt = dct["reaction"]
        reaction = (
            [Molecule.from_name(name=d) for d in lft],
            [Molecule.from_name(name=d) for d in rgt],
        )
        return cls(
            reaction=reaction,
            km=dct["km"],
            vmax=dct["vmax"],
            start=dct["start"],
            end=dct["end"],
        )

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


class TransporterDomain:
    """
    Object describing a transporter domain.

    Parameters:
        molecule: [Molecule][magicsoup.containers.Molecule] which can be transported by this domain.
        km: Michaelis Menten constant of the transport (in mM).
        vmax: Maximum velocity of the transport (in mmol/s).
        is_exporter: Whether the transporter is exporting this molecule species out of the cell.
        start: Domain start on the CDS
        end: Domain end on the CDS

    The simulation works with tensor representations of cell proteomes.
    This class is a helper which makes interpreting a cell's proteome easier.
    You should not instantiate this class but instead get it from `cell.proteome`
    (see [Cell][magicsoup.containers.Cell]).

    `is_exporter` is only relevant in combination with other domains on the same protein.
    It defines in which transport direction this domain is energetically coupled with others.

    Domain start and end describe the slice of the CDS python string.
    _I.e._ the index starts with 0, start is included, end is excluded.
    _E.g._ `start=3`, `end=18` starts with the 4th and ends with the 18th base pair on the CDS.
    """

    def __init__(
        self,
        molecule: Molecule,
        km: float,
        vmax: float,
        is_exporter: bool,
        start: int,
        end: int,
    ):
        self.start = start
        self.end = end
        self.molecule = molecule
        self.km = km
        self.vmax = vmax
        self.is_exporter = is_exporter

    def to_dict(self) -> dict:
        """Get dict representation of domain"""
        kwargs = {
            "molecule": self.molecule.name,
            "km": self.km,
            "vmax": self.vmax,
            "is_exporter": self.is_exporter,
            "start": self.start,
            "end": self.end,
        }
        return {"type": "T", "spec": kwargs}

    @classmethod
    def from_dict(cls, dct: dict) -> "TransporterDomain":
        """
        Convencience method for creating an instance from a dict.
        All parameters must be present as keys.
        Molecules are provided by their name.
        """
        return cls(
            molecule=Molecule.from_name(name=dct["molecule"]),
            km=dct["km"],
            vmax=dct["vmax"],
            is_exporter=dct["is_exporter"],
            start=dct["start"],
            end=dct["end"],
        )

    def __repr__(self) -> str:
        sign = "exporter" if self.is_exporter else "importer"
        return f"TransporterDomain({self.molecule},Km={self.km:.2e},Vmax={self.vmax:.2e},{sign})"

    def __str__(self) -> str:
        sign = "exporter" if self.is_exporter else "importer"
        return f"{self.molecule} {sign} | Km {self.km:.2e} Vmax {self.vmax:.2e}"


class RegulatoryDomain:
    """
    Object describing a regulatory domain.

    Parameters:
        effector: Effector [Molecules][magicsoup.containers.Molecule]
        hill: Hill coefficient describing degree of cooperativity
        km: Ligand concentration producing half occupation (in mM)
        is_inhibiting: Whether this is an inhibiting regulatory domain (otherwise activating).
        is_transmembrane: Whether this is also a transmembrane domain.
            If true, the domain will react to extracellular molecules instead of intracellular ones.
        start: Domain start on the CDS
        end: Domain end on the CDS

    The simulation works with tensor representations of cell proteomes.
    This class is a helper which makes interpreting a cell's proteome easier.
    You should not instantiate this class but instead get it from `cell.proteome`
    (see [Cell][magicsoup.containers.Cell]).

    Domain start and end describe the slice of the CDS python string.
    _I.e._ the index starts with 0, start is included, end is excluded.
    _E.g._ `start=3`, `end=18` starts with the 4th and ends with the 18th base pair on the CDS.
    """

    def __init__(
        self,
        effector: Molecule,
        hill: int,
        km: float,
        is_inhibiting: bool,
        is_transmembrane: bool,
        start: int,
        end: int,
    ):
        self.start = start
        self.end = end
        self.effector = effector
        self.km = km
        self.hill = int(hill)
        self.is_transmembrane = is_transmembrane
        self.is_inhibiting = is_inhibiting

    def to_dict(self) -> dict:
        """Get dict representation of domain"""
        kwargs = {
            "effector": self.effector.name,
            "km": self.km,
            "hill": self.hill,
            "is_inhibiting": self.is_inhibiting,
            "is_transmembrane": self.is_transmembrane,
            "start": self.start,
            "end": self.end,
        }
        return {"type": "R", "spec": kwargs}

    @classmethod
    def from_dict(cls, dct: dict) -> "RegulatoryDomain":
        """
        Convencience method for creating an instance from a dict.
        All parameters must be present as keys.
        Molecules are provided by their name.
        """
        return cls(
            effector=Molecule.from_name(name=dct["effector"]),
            km=dct["km"],
            hill=dct["hill"],
            is_inhibiting=dct["is_inhibiting"],
            is_transmembrane=dct["is_transmembrane"],
            start=dct["start"],
            end=dct["end"],
        )

    def __repr__(self) -> str:
        loc = "transmembrane" if self.is_transmembrane else "cytosolic"
        eff = "inhibiting" if self.is_inhibiting else "activating"
        return f"ReceptorDomain({self.effector},Km={self.km:.2e},hill={self.hill},{loc},{eff})"

    def __str__(self) -> str:
        loc = "[ext]" if self.is_transmembrane else "[cyt]"
        post = "inhibitor" if self.is_inhibiting else "activator"
        return f"{self.effector}{loc} {post} | Km {self.km:.2e} Hill {self.hill}"


class Protein:
    """
    Object describing a protein.

    Parameters:
        domains: All domains of the protein as a list of
            [CatalyticDomains][magicsoup.containers.CatalyticDomain],
            [TransporterDomains][magicsoup.containers.TransporterDomain],
            and [RegulatoryDomains][magicsoup.containers.RegulatoryDomain].
        cds_start: Start coordinate of its coding region
        cds_end: End coordinate of its coding region
        is_fwd: Whether its CDS is in the forward or reverse-complement of the genome.

    The simulation works with tensor representations of cell proteomes.
    This class is a helper which makes interpreting a cell's proteome easier.
    You should not instantiate this class but instead get it from `cell.proteome`
    (see [Cell][magicsoup.containers.Cell]).

    CDS start and end describe the slice of the genome python string.
    _I.e._ the index starts with 0, start is included, end is excluded.
    _E.g._ `cds_start=2`, `cds_end=31` starts with the 3rd and ends with the 31st base pair on the genome.

    `is_fwd` describes whether the CDS is found on the forward (hypothetical 5'-3')
    or the reverse-complement (hypothetical 3'-5') side of the genome.
    `cds_start` and `cds_end` always describe the parsing direction / the direction of the hypothetical transcriptase.
    So, if you want to visualize a `is_fwd=False` CDS on the genome in 5'-3' direction
    you have to do `n - cds_start` and `n - cds_stop` if `n` is the genome length.
    """

    def __init__(
        self, domains: list[DomainType], cds_start: int, cds_end: int, is_fwd: bool
    ):
        self.domains = domains
        self.n_domains = len(domains)
        self.cds_start = cds_start
        self.cds_end = cds_end
        self.is_fwd = is_fwd

    def to_dict(self) -> dict:
        """Get dict representation of protein"""
        return {
            "domains": [d.to_dict() for d in self.domains],
            "cds_start": self.cds_start,
            "cds_end": self.cds_end,
            "is_fwd": self.is_fwd,
        }

    @classmethod
    def from_dict(cls, dct: dict) -> "Protein":
        """
        Create Protein instance from dict. Key must match arguments.
        Domains are set as a list of dicts `{"type": ".", "spec": {})` where
        `type` is domain type `"C"` (catalytic), `"T"` (transporter), or
        `"R"` (regulatory) and `spec` is a dict with kwargs for each domain's `from_dict()`.
        """
        doms: list[DomainType] = []
        for dom in dct["domains"]:
            dom_type = dom["type"]
            dom_spec = dom["spec"]
            if dom_type == "C":
                doms.append(CatalyticDomain.from_dict(dom_spec))
            elif dom_type == "T":
                doms.append(TransporterDomain.from_dict(dom_spec))
            elif dom_type == "R":
                doms.append(RegulatoryDomain.from_dict(dom_spec))

        return Protein(
            cds_start=dct["cds_start"],
            cds_end=dct["cds_end"],
            is_fwd=dct["is_fwd"],
            domains=doms,
        )

    def __repr__(self) -> str:
        kwargs = {
            "cds_start": self.cds_start,
            "cds_end": self.cds_end,
            "domains": self.domains,
        }
        args = [f"{k}:{d!r}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"

    def __str__(self) -> str:
        domstrs = [str(d).split(" | ")[0] for d in self.domains]
        return " | ".join(domstrs)


class Cell:
    """
    Object describing a cell and its environment.

    Parameters:
        world: Reference to the origin [World][magicsoup.world.World] object.
        genome: Genome sequence string of this cell.
        position: Position on the cell map as tuple `(x, y)`.
        idx: Current cell index in [World][magicsoup.world.World].
        label: Label of origin which can be used to track cells.
        n_steps_alive: Number of time steps this cell has lived since last division.
        n_divisions: Number of times this cell's ancestors already divided.
        proteome: List of [Protein][magicsoup.containers.Protein] objects.
        int_molecules: Intracellular molecule concentrations. A 1D tensor that describes
            each [Molecule][magicsoup.containers.Molecule]
            in the same order as defined in [Chemistry][magicsoup.containers.Chemistry].
        ext_molecules: Extracellular molecule concentrations. A 1D tensor that describes
            each [Molecule][magicsoup.containers.Molecule]
            in the same order as defined in [Chemistry][magicsoup.containers.Chemistry].
            These are the molecules in `world.molecule_map` of the pixel the cell is currently living on.

    The simulation works with tensor representations of cell proteomes and molecule concentrations.
    This class is a helper which makes interpreting a cell easier.
    You should not instantiate this class but instead get it from [get_cell()][magicsoup.world.World.get_cell].

    All parameters are exposed as attributes on the cell object.
    Some are calculated lazily just when they are accessed.

    ```
    cell = world.get_cell(by_idx=3)
    assert cell.idx == 5  # the cell's current index
    cell.proteome  # proteome is calculated now
    ```

    When a cell divides its genome and proteome are copied.
    Both descendants will recieve half of all molecules each.
    Both their `n_divisions` attributes are incremented.
    The cell's `label` will be copied as well.
    This way you can track how cells spread.
    """

    def __init__(
        self,
        world: "World",
        genome: str,
        position: tuple[int, int] = (-1, -1),
        idx: int = -1,
        label: str = "C",
        age: float = 0.0,
        generation: int = 0,
        proteome: list[Protein] | None = None,
        int_molecules: torch.Tensor | None = None,
        ext_molecules: torch.Tensor | None = None,
    ):
        self.world = world
        self.genome = genome
        self.label = label
        self.position = position
        self.idx = idx
        self.age = age
        self.generation = generation

        self._proteome = proteome
        self._int_molecules = int_molecules
        self._ext_molecules = ext_molecules

    @property
    def int_molecules(self) -> torch.Tensor:
        if self._int_molecules is None:
            self._int_molecules = self.world.cells.molecules[self.idx, :]
        return self._int_molecules

    @property
    def ext_molecules(self) -> torch.Tensor:
        if self._ext_molecules is None:
            pos = self.position
            self._ext_molecules = self.world.culture.molecules[:, pos[0], pos[1]]
        return self._ext_molecules

    @property
    def proteome(self) -> list[Protein]:
        if self._proteome is None:
            (cdss,) = self.world.cells.genomics.translate_genomes(genomes=[self.genome])
            if len(cdss) > 0:
                self._proteome = self.world.cells.proteomics.get_proteome(proteome=cdss)
            else:
                self._proteome = []
        return self._proteome

    def __repr__(self) -> str:
        kwargs = {
            "genome": self.genome,
            "position": self.position,
            "idx": self.idx,
            "label": self.label,
            "age": self.age,
            "generation": self.generation,
        }
        args = [f"{k}:{d!r}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"
