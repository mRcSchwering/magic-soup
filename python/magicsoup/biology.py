from collections import Counter

import torch

from magicsoup.chemistry import Molecule, Reaction


class CatalyticDomain:
    """Object describing a catalytic domain.

    Arguments:
        reaction: reaction catalyzed by the domain.
        km: Michaelis Menten constant of the reaction (in mM).
        vmax: Maximum velocity of the reaction (in mmol/s with 1 µM enzyme).
        start: Domain start on the CDS
        end: Domain end on the CDS
        is_fwd: How reaction is coupled with other domains in the same CDS

    The simulation works with tensor representations of cell proteomes.
    This class is a helper which makes interpreting a cell's proteome easier.
    You should not instantiate this class but instead get it from `cell.proteome`
    (see [Cell][magicsoup.biology.Cell]).

    Domain start and end describe the slice of the CDS python string.
    _I.e._ the index starts with 0, start is included, end is excluded.
    _E.g._ `start=3`, `end=18` starts with the 4th and ends with the 18th base pair on the CDS.
    """

    def __init__(
        self,
        reaction: Reaction,
        km: float,
        vmax: float,
        start: int,
        end: int,
        is_fwd: bool = True,
    ):
        self.start = start
        """Domain start on the CDS"""

        self.end = end
        """Domain end on the CDS"""

        self.reaction = reaction
        """Reaction catalyzed by the domain"""

        self.km = km
        """Michaelis Menten constant of the catalyzed reaction (in mM)"""

        self.vmax = vmax
        """Maximum velocity of the reaction (in mmol/s with 1 µM enzyme)"""

        self.is_fwd = is_fwd
        """How reaction is coupled with other domains in the same CDS"""

    def __repr__(self) -> str:
        return f"CatalyticDomain({self.reaction},Km={self.km:.2e},Vmax={self.vmax:.2e},fwd={self.is_fwd})"

    def __str__(self) -> str:
        subs_cntr = Counter(self.reaction.substrates)
        prods_cntr = Counter(self.reaction.products)
        lft = [f"{n} {m}" if n > 1 else str(m) for m, n in subs_cntr.items()]
        rgt = [f"{n} {m}" if n > 1 else str(m) for m, n in prods_cntr.items()]
        subs, prods = (lft, rgt) if self.is_fwd else (rgt, lft)
        return f"{' + '.join(subs)} -> {' + '.join(prods)} | Km {self.km:.2e} Vmax {self.vmax:.2e}"


class TransporterDomain:
    """Object describing a transporter domain.

    Arguments:
        molecule: molecule which can be transported by this domain.
        km: Michaelis Menten constant of the transport (in mM).
        vmax: Maximum velocity of the transport (in mmol/s with 1 µM enzyme).
        is_exporter: The direction (import or export) in which the transporter is coupled with other domains.
        start: Domain start on the CDS
        end: Domain end on the CDS

    The simulation works with tensor representations of cell proteomes.
    This class is a helper which makes interpreting a cell's proteome easier.
    You should not instantiate this class but instead get it from `cell.proteome`
    (see [Cell][magicsoup.biology.Cell]).

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
        """Domain start on the CDS"""

        self.end = end
        """Domain end on the CDS"""

        self.molecule = molecule
        """Molecule transported by this domain"""

        self.km = km
        """Michaelis Menten constant of the catalyzed reaction (in mM)"""

        self.vmax = vmax
        """Maximum velocity of the reaction (in mmol/s with 1 µM enzyme)"""

        self.is_exporter = is_exporter
        """The direction (import or export) in which the transporter is coupled with other domains"""

    def __repr__(self) -> str:
        sign = "exporter" if self.is_exporter else "importer"
        return f"TransporterDomain({self.molecule},Km={self.km:.2e},Vmax={self.vmax:.2e},{sign})"

    def __str__(self) -> str:
        sign = "exporter" if self.is_exporter else "importer"
        return f"{self.molecule} {sign} | Km {self.km:.2e} Vmax {self.vmax:.2e}"


class RegulatoryDomain:
    """Object describing a regulatory domain.

    Arguments:
        effector: Effector molecule
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
    (see [Cell][magicsoup.biology.Cell]).

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
        """Domain start on the CDS"""

        self.end = end
        """Domain end on the CDS"""

        self.effector = effector
        """Effector molecule"""

        self.km = km
        """Ligand concentration producing half occupation (in mM)"""

        self.hill = int(hill)
        """Hill coefficient describing degree of cooperativity"""

        self.is_transmembrane = is_transmembrane
        """Whether this is also a transmembrane domain"""

        self.is_inhibiting = is_inhibiting
        """Whether this is an inhibiting regulatory domain (otherwise activating)"""

    def __repr__(self) -> str:
        loc = "transmembrane" if self.is_transmembrane else "cytosolic"
        eff = "inhibiting" if self.is_inhibiting else "activating"
        return f"ReceptorDomain({self.effector},Km={self.km:.2e},hill={self.hill},{loc},{eff})"

    def __str__(self) -> str:
        loc = "[ext]" if self.is_transmembrane else "[cyt]"
        post = "inhibitor" if self.is_inhibiting else "activator"
        return f"{self.effector}{loc} {post} | Km {self.km:.2e} Hill {self.hill}"


type Domain = RegulatoryDomain | CatalyticDomain | TransporterDomain


class Protein:
    """Object describing a protein.

    Arguments:
        domains: All domains of the protein as a list of
            [CatalyticDomains][magicsoup.biology.CatalyticDomain],
            [TransporterDomains][magicsoup.biology.TransporterDomain],
            and [RegulatoryDomains][magicsoup.biology.RegulatoryDomain].
        cds_start: Start coordinate of its coding region
        cds_end: End coordinate of its coding region
        is_53: Whether its CDS is in the 5'-3' or 3'-5' direction of the genome.

    The simulation works with tensor representations of cell proteomes.
    This class is a helper which makes interpreting a cell's proteome easier.
    You should not instantiate this class but instead get it from `cell.proteome`
    (see [Cell][magicsoup.biology.Cell]).

    CDS start and end describe the slice of the genome python string.
    _I.e._ the index starts with 0, start is included, end is excluded.
    _E.g._ `cds_start=2`, `cds_end=31` starts with the 3rd and ends with the 31st base pair on the genome.

    The genome is circular, so a CDS can wrap around the origin. Such a CDS has
    `cds_end > n` (and thus `cds_end % n < cds_start`) where `n` is the genome
    length. To slice it you have to wrap, _e.g._ `genome[cds_start:] + genome[:cds_end % n]`.

    `is_53` describes whether the CDS is found on the forward (hypothetical 5'-3')
    or the reverse-complement (hypothetical 3'-5') side of the genome.
    `cds_start` and `cds_end` always describe the parsing direction / the direction of the hypothetical transcriptase.
    So, if you want to visualize a `is_53=False` CDS on the genome in 5'-3' direction
    you have to do `n - cds_start` and `n - cds_stop` if `n` is the genome length
    (taking `cds_end % n` first for a CDS that wraps the origin).
    """

    def __init__(self, domains: list[Domain], cds_start: int, cds_end: int, is_53: bool):
        self.domains = domains
        """All protein domains"""

        self.cds_start = cds_start
        """Start coordinate of the protein's coding region"""

        self.cds_end = cds_end
        """End coordinate of the protein's coding region"""

        self.is_53 = is_53
        """Whether the protein's CDS is in the 5'-3' or 3'-5' direction of the genome"""

    def __repr__(self) -> str:
        kwargs = {"cds_start": self.cds_start, "cds_end": self.cds_end, "domains": self.domains}
        args = [f"{k}:{d!r}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"

    def __str__(self) -> str:
        domstrs = [str(d).split(" | ")[0] for d in self.domains]
        return " | ".join(domstrs)


class Cell:
    """Object describing a cell.

    Arguments:
        genome: Genome sequence string of this cell.
        lineage: Cell marker for lineage tracing.
        age: Time passed since last cell division (in s).
        generation: Number of times this cell's ancestors divided.
        proteome: Cell proteome as list of [Protein][magicsoup.biology.Protein].
        molecules: float (m,) tensor for intracellular molecule concentrations.

    The simulation works with tensor representations of cell proteomes and molecule concentrations.
    This class is a helper which makes interpreting a cell easier.
    Pick particular cells using [world.get_cell()][magicsoup.world.World.get_cell].
    """

    def __init__(
        self,
        genome: str,
        lineage: str,
        age: float,
        generation: int,
        proteome: list[Protein],
        molecules: torch.Tensor,
    ):
        self.genome = genome
        """Cell's genome as string"""

        self.lineage = lineage
        """Cell marker for lineage tracing"""

        self.age = age
        """Time passed since last cell division (in s)"""

        self.generation = generation
        """Number of times this cell's ancestors divided"""

        self.proteome = proteome
        """Cell proteome as list of proteins"""

        self.molecules = molecules
        """float (m,) tensor for intracellular molecule concentrations"""

    def __repr__(self) -> str:
        kwargs = {
            "genome": self.genome,
            "lineage": self.lineage,
            "age": self.age,
            "generation": self.generation,
        }
        args = [f"{k}:{d!r}" for k, d in kwargs.items()]
        return f"{type(self).__name__}({','.join(args)})"

    def __str__(self) -> str:
        return f"{type(self).__name__} {self.lineage}"
