import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import torch

from magicsoup.chemistry import Chemistry, Molecule
from magicsoup.util import Array

if TYPE_CHECKING:
    from magicsoup.world2 import World


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
            self._int_molecules = self.world.cells.x_i[self.idx, :]
        return self._int_molecules

    @property
    def ext_molecules(self) -> torch.Tensor:
        if self._ext_molecules is None:
            pos = self.position
            self._ext_molecules = self.world.map.molecules[:, pos[0], pos[1]]
        return self._ext_molecules

    @property
    def proteome(self) -> list[Protein]:
        if self._proteome is None:
            (cdss,) = self.world.genomics.translate_genomes(genomes=[self.genome])
            if len(cdss) > 0:
                self._proteome = self.world.proteomics.get_proteome(proteome=cdss)
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


class Cells:

    def __init__(
        self,
        chemistry: Chemistry,
        device: str = "cpu",
        itype: torch.dtype = torch.int8,
        ftype: torch.dtype = torch.float32,
    ):
        molecules = chemistry.molecules
        n_molecules = len(molecules)

        self.device = device
        self.itype = itype
        self.ftype = ftype

        self._int_mol_idxs = list(range(n_molecules))
        self._ext_mol_idxs = list(range(n_molecules, n_molecules * 2))
        self._mol_degrad = [self._get_degrade(m.half_life) for m in molecules]
        self._permeation = [
            self._get_permeate(mol_perm_rate=m.permeability) for m in molecules
        ]

        self.c = 0
        self.p = 0
        self.m = 2 * n_molecules

        self.genomes: Array[str] = Array([])
        self.labels: Array[str] = Array([])
        self.alive = self._izeros(0).bool()
        self.positions = self._idxzeros(0, 2)
        self.ages = self._idxzeros(0)
        self.generations = self._idxzeros(0)
        self.x_i = self._fzeros(0, n_molecules)  # only intracellular
        self.k_e = self._fzeros(0, 0)
        self.k_f = self._fzeros(0, 0)
        self.k_b = self._fzeros(0, 0)
        self.K_r = self._fzeros(0, 0, self.m)
        self.v_max = self._fzeros(0, 0)
        self.N = self._izeros(0, 0, self.m)
        self.N_f = self._izeros(0, 0, self.m)
        self.N_b = self._izeros(0, 0, self.m)
        self.N_h = self._izeros(0, 0, self.m)

    def get_alive_cells(self) -> int:
        return int(self.alive.sum().item())

    def get_available_idxs(self) -> torch.Tensor:
        return (~self.alive).nonzero(as_tuple=True)[0]

    def set_c(self, c: int) -> None:
        # TODO: bad name, I just report the number of cells
        #       c is the size of the dimension (which can be more)
        if c < self.c:
            self._decrease_c(by_n=self.c - c)
        if c > self.c:
            self._increase_c(by_n=c - self.c)

        self.c = c

    def set_p(self, p: int) -> None:
        # TODO: bad name, I just report the number of proteins
        #       p is the size of the dimension (which can be more)
        if p < self.p:
            self._decrease_p(by_n=self.p - p)
        if p > self.p:
            self._increase_p(by_n=p - self.p)

        self.p = p

    def save_state(self, statedir: Path) -> None:
        name = type(self).__name__
        torch.save(self.alive, statedir / f"{name}.alive.pt")
        torch.save(self.ages, statedir / f"{name}.ages.pt")
        torch.save(self.positions, statedir / f"{name}.positions.pt")
        torch.save(self.generations, statedir / f"{name}.generations.pt")
        torch.save(self.x_i, statedir / f"{name}.x_i.pt")
        torch.save(self.k_e, statedir / f"{name}.k_e.pt")
        torch.save(self.k_f, statedir / f"{name}.k_f.pt")
        torch.save(self.k_b, statedir / f"{name}.k_b.pt")
        torch.save(self.K_r, statedir / f"{name}.K_r.pt")
        torch.save(self.v_max, statedir / f"{name}.v_max.pt")
        torch.save(self.N, statedir / f"{name}.N.pt")
        torch.save(self.N_b, statedir / f"{name}.N_b.pt")
        torch.save(self.N_f, statedir / f"{name}.N_f.pt")
        torch.save(self.N_h, statedir / f"{name}.N_h.pt")

        lines: list[str] = [
            f">{i} {l}\n{g}"
            for i, (g, l) in enumerate(zip(self.genomes.items, self.labels.items))
        ]

        with open(statedir / f"{name}.genomes.fasta", "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def load_state(self, statedir: Path) -> None:
        name = type(self).__name__
        self.alive = torch.load(
            statedir / f"{name}.alive.pt",
            map_location=self.device,
            dtype=torch.bool,
        )
        self.ages = torch.load(
            statedir / f"{name}.ages.pt",
            map_location=self.device,
            dtype=torch.int32,
        )
        self.positions = torch.load(
            statedir / f"{name}.positions.pt",
            map_location=self.device,
            dtype=torch.int32,
        )
        self.generations = torch.load(
            statedir / f"{name}generations.pt",
            map_location=self.device,
            dtype=torch.int32,
        )
        self.x_i = torch.load(
            statedir / f"{name}.x_i.pt",
            map_location=self.device,
            dtype=self.ftype,
        )
        self.k_e = torch.load(
            statedir / f"{name}.k_e.pt",
            map_location=self.device,
            dtype=self.ftype,
        )
        self.k_f = torch.load(
            statedir / f"{name}.k_f.pt",
            map_location=self.device,
            dtype=self.ftype,
        )
        self.k_b = torch.load(
            statedir / f"{name}.k_b.pt",
            map_location=self.device,
            dtype=self.ftype,
        )
        self.K_r = torch.load(
            statedir / f"{name}.K_r.pt",
            map_location=self.device,
            dtype=self.ftype,
        )
        self.v_max = torch.load(
            statedir / f"{name}.v_max.pt",
            map_location=self.device,
            dtype=self.ftype,
        )
        self.N = torch.load(
            statedir / f"{name}.N.pt",
            map_location=self.device,
            dtype=self.itype,
        )
        self.N_f = torch.load(
            statedir / f"{name}.N_f.pt",
            map_location=self.device,
            dtype=self.itype,
        )
        self.N_b = torch.load(
            statedir / f"{name}.N_b.pt",
            map_location=self.device,
            dtype=self.itype,
        )
        self.N_h = torch.load(
            statedir / f"{name}.N_h.pt",
            map_location=self.device,
            dtype=self.itype,
        )

        with open(statedir / f"{name}.genomes.fasta", encoding="utf-8") as fh:
            text: str = fh.read()
            entries = [d.strip() for d in text.split(">") if len(d.strip()) > 0]

        self.labels = Array([])
        self.genomes = Array([])
        for entry in entries:
            parts = entry.split("\n")
            descr = parts[0]
            seq = "" if len(parts) < 2 else parts[1]
            names = descr.split()
            label = names[1].strip() if len(names) > 1 else ""
            self.genomes.items.append(seq)
            self.labels.items.append(label)

    def _increase_c(self, by_n: int) -> None:
        self.genomes.items.extend([""] * by_n)
        self.labels.items.extend([""] * by_n)
        self.alive = self._expand_c(t=self.alive, by_n=by_n)
        self.ages = self._expand_c(t=self.ages, by_n=by_n)
        self.positions = self._expand_c(t=self.positions, by_n=by_n)
        self.generations = self._expand_c(t=self.generations, by_n=by_n)
        self.x_i = self._expand_c(t=self.x_i, by_n=by_n)
        self.k_e = self._expand_c(t=self.k_e, by_n=by_n)
        self.k_f = self._expand_c(t=self.k_f, by_n=by_n)
        self.k_b = self._expand_c(t=self.k_b, by_n=by_n)
        self.K_r = self._expand_c(t=self.K_r, by_n=by_n)
        self.v_max = self._expand_c(t=self.v_max, by_n=by_n)
        self.N = self._expand_c(t=self.N, by_n=by_n)
        self.N_f = self._expand_c(t=self.N_f, by_n=by_n)
        self.N_b = self._expand_c(t=self.N_b, by_n=by_n)
        self.N_h = self._expand_c(t=self.N_h, by_n=by_n)

    def _decrease_c(self, by_n: int) -> None:
        dead = (~self.alive).nonzero(as_tuple=True)[0]
        n_avail = dead.shape[0]
        if n_avail < by_n:
            raise ValueError(f"Cannot decrease c by {by_n}, only {n_avail} available")

        dead = dead[:by_n]
        keep = torch.ones_like(self.alive, dtype=torch.bool)
        keep[dead] = False

        self.genomes = Array(self.genomes[keep])
        self.labels = Array(self.labels[keep])
        self.alive = self.alive[keep]
        self.ages = self.ages[keep]
        self.positions = self.positions[keep]
        self.generations = self.generations[keep]
        self.x_i = self.x_i[keep]
        self.k_e = self.k_e[keep]
        self.k_f = self.k_f[keep]
        self.k_b = self.k_b[keep]
        self.K_r = self.K_r[keep]
        self.v_max = self.v_max[keep]
        self.N = self.N[keep]
        self.N_f = self.N_f[keep]
        self.N_b = self.N_b[keep]
        self.N_h = self.N_h[keep]

    def _increase_p(self, by_n: int) -> None:
        self.k_e = self._expand_p(t=self.k_e, by_n=by_n)
        self.k_f = self._expand_p(t=self.k_f, by_n=by_n)
        self.k_b = self._expand_p(t=self.k_b, by_n=by_n)
        self.K_r = self._expand_p(t=self.K_r, by_n=by_n)
        self.v_max = self._expand_p(t=self.v_max, by_n=by_n)
        self.N = self._expand_p(t=self.N, by_n=by_n)
        self.N_f = self._expand_p(t=self.N_f, by_n=by_n)
        self.N_b = self._expand_p(t=self.N_b, by_n=by_n)
        self.N_h = self._expand_p(t=self.N_h, by_n=by_n)

    def _decrease_p(self, by_n: int) -> None:
        self.k_e = self.k_e[:, :-by_n]
        self.k_f = self.k_f[:, :-by_n]
        self.k_b = self.k_b[:, :-by_n]
        self.K_r = self.K_r[:, :-by_n]
        self.v_max = self.v_max[:, :-by_n]
        self.N = self.N[:, :-by_n]
        self.N_f = self.N_f[:, :-by_n]
        self.N_b = self.N_b[:, :-by_n]
        self.N_h = self.N_h[:, :-by_n]

    def _get_permeate(self, mol_perm_rate: float) -> float:
        if mol_perm_rate < 0.0:
            mol_perm_rate = -mol_perm_rate

        mol_perm_rate = min(mol_perm_rate, 1.0)

        if mol_perm_rate == 0.0:
            return 0.0

        d = 1 / mol_perm_rate
        return 1 / (d + 1)

    def _get_degrade(self, half_life: float) -> float:
        return math.exp(-math.log(2) / half_life)

    def _expand_c(self, t: torch.Tensor, by_n: int) -> torch.Tensor:
        size = t.size()
        zeros = torch.zeros(by_n, *size[1:], device=t.device, dtype=t.dtype)
        return torch.cat([t, zeros], dim=0)

    def _expand_p(self, t: torch.Tensor, by_n: int) -> torch.Tensor:
        size = t.size()
        zeros = torch.zeros(size[0], by_n, *size[2:], device=t.device, dtype=t.dtype)
        return torch.cat([t, zeros], dim=1)

    def _idxzeros(self, *args) -> torch.Tensor:
        return torch.zeros(*args, device=self.device, dtype=torch.float32)

    def _izeros(self, *args) -> torch.Tensor:
        return torch.zeros(*args, device=self.device, dtype=self.itype)

    def _fzeros(self, *args) -> torch.Tensor:
        return torch.zeros(*args, device=self.device, dtype=self.ftype)
