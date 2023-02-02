from typing import Optional
import warnings
import random
import abc
import torch
import torch.multiprocessing as mp
from magicsoup.util import (
    reverse_complement,
    nt_seqs,
    generic_map_fact,
    log_weight_map_fact,
    bool_map_fact,
)
from magicsoup.containers import Domain, Protein, Chemistry, Molecule, DomType
from magicsoup.constants import CODON_SIZE


# TODO: tryout
def get_domain(
    cell_i: int,
    prot_i: int,
    dom_i: int,
    seq: str,
    domain_map: dict[str, tuple[bool, bool, bool]],
    molecule_map: dict[str, int],
    reaction_map: dict[str, tuple[list[int], list[int]]],
    affinity_map: dict[str, float],
    velocity_map: dict[str, float],
    bool_map: dict[str, bool],
    n_nts: dict[str, int],
    n_molecules: int,
    N: torch.Tensor,
    A: torch.Tensor,
    Km: torch.Tensor,
    Vmax: torch.Tensor,
) -> bool:
    n_dom_type_nts = n_nts["n_dom_type_nts"]
    n_reaction_nts = n_nts["n_reaction_nts"]
    n_molecule_nts = n_nts["n_molecule_nts"]
    n_affinity_nts = n_nts["n_affinity_nts"]
    n_velocity_nts = n_nts["n_velocity_nts"]
    n_bool_nts = n_nts["n_bool_nts"]

    dom_type_seq = seq[:n_dom_type_nts]
    if dom_type_seq not in domain_map:
        return False

    i = n_dom_type_nts
    is_catal, is_trnsp, is_reg = domain_map[dom_type_seq]

    if is_catal:
        lft, rgt = reaction_map[seq[i : i + n_reaction_nts]]
        i += n_reaction_nts
        aff = affinity_map[seq[i : i + n_affinity_nts]]
        i += n_affinity_nts
        velo = velocity_map[seq[i : i + n_velocity_nts]]
        i += n_velocity_nts
        orient = bool_map[seq[i : i + n_bool_nts]]

        Vmax[cell_i, prot_i, dom_i] = velo

        if orient:
            prods = lft
            subs = rgt
        else:
            subs = lft
            prods = rgt

        for mol_i in subs:
            N[cell_i, prot_i, dom_i, mol_i] = -1.0
            Km[cell_i, prot_i, dom_i, mol_i] = aff

        for mol_i in prods:
            N[cell_i, prot_i, dom_i, mol_i] = 1.0
            Km[cell_i, prot_i, dom_i, mol_i] = 1 / aff

        return True

    if is_trnsp:
        mol = molecule_map[seq[i : i + n_molecule_nts]]
        i += n_molecule_nts
        aff = affinity_map[seq[i : i + n_affinity_nts]]
        i += n_affinity_nts
        velo = velocity_map[seq[i : i + n_velocity_nts]]
        i += n_velocity_nts
        orient = bool_map[seq[i : i + n_bool_nts]]

        Vmax[cell_i, prot_i, dom_i] = velo

        if orient:
            sub_i = mol + n_molecules
            prod_i = mol
        else:
            sub_i = mol
            prod_i = mol + n_molecules

        N[cell_i, prot_i, dom_i, sub_i] = 1
        N[cell_i, prot_i, dom_i, prod_i] = 1
        Km[cell_i, prot_i, dom_i, sub_i] = aff
        Km[cell_i, prot_i, dom_i, prod_i] = 1 / aff

        return True

    if is_reg:
        mol = molecule_map[seq[i : i + n_molecule_nts]]
        i += n_molecule_nts
        aff = affinity_map[seq[i : i + n_affinity_nts]]
        i += n_affinity_nts
        transm = bool_map[seq[i : i + n_bool_nts]]
        i += n_bool_nts
        inh = bool_map[seq[i : i + n_bool_nts]]

        mol_i = mol + n_molecules if transm else mol
        A[cell_i, prot_i, dom_i, mol_i] = -1.0 if inh else 1.0
        Km[cell_i, prot_i, dom_i, mol_i] = aff

        return True

    return False


# TODO: tryout
def translate_seq(
    cell_i: int,
    prot_i: int,
    seq: str,
    maps: dict,
    n_nts: dict[str, int],
    dom_size: int,
    n_molecules: int,
    N: torch.Tensor,
    A: torch.Tensor,
    Km: torch.Tensor,
    Vmax: torch.Tensor,
):
    domain_map = maps["domain_map"]
    molecule_map = maps["molecule_map"]
    reaction_map = maps["reaction_map"]
    affinity_map = maps["affinity_map"]
    velocity_map = maps["velocity_map"]
    bool_map = maps["bool_map"]

    kwargs = {
        "cell_i": cell_i,
        "prot_i": prot_i,
        "domain_map": domain_map,
        "molecule_map": molecule_map,
        "reaction_map": reaction_map,
        "affinity_map": affinity_map,
        "velocity_map": velocity_map,
        "bool_map": bool_map,
        "n_nts": n_nts,
        "n_molecules": n_molecules,
        "N": N,
        "A": A,
        "Km": Km,
        "Vmax": Vmax,
    }

    dom_i = 0
    i = 0
    j = dom_size
    while i + dom_size <= len(seq):
        is_dom = get_domain(dom_i=dom_i, seq=seq[i : i + dom_size], **kwargs)  # type: ignore
        if is_dom:
            dom_i += 1
            i += dom_size
            j += dom_size
        else:
            i += CODON_SIZE
            j += CODON_SIZE


# TODO: tryout
def translate_all_seqs(
    cell_i: int,
    seqs: list[str],
    maps: dict,
    n_nts: dict[str, int],
    dom_size: int,
    n_molecules: int,
    N: torch.Tensor,
    A: torch.Tensor,
    Km: torch.Tensor,
    Vmax: torch.Tensor,
):
    kwargs = {
        "cell_i": cell_i,
        "maps": maps,
        "n_nts": n_nts,
        "dom_size": dom_size,
        "n_molecules": n_molecules,
        "N": N,
        "A": A,
        "Km": Km,
        "Vmax": Vmax,
    }
    for i, d in enumerate(seqs):
        translate_seq(prot_i=i, seq=d, **kwargs)  # type: ignore


def _check_n(n: int, label: str):
    if n % CODON_SIZE != 0:
        raise ValueError(f"{label} must be a multiple of {CODON_SIZE}. Now it is {n}")


class _DomainFact(abc.ABC):
    """Base class to create domain factory. Must implement __call__."""

    min_len = 0

    @abc.abstractmethod
    def __call__(self, seq: str) -> Domain:
        """Instantiate domain object from encoding nucleotide sequence"""
        raise NotImplementedError("Implement __call__")

    def __repr__(self) -> str:
        kwargs = {
            "min_len": self.min_len,
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
        is_bkwd: Flag indicating whether in which orientation this reaction will be coupled with other domains.
    """

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


class CatalyticFact(_DomainFact):
    """
    Factory for generating catalytic domains from nucleotide sequences.

    Arguments:
        reactions: All reactions that can be catalyzed by domains.
            Each reaction is a tuple of substrate and product molecule species.
            For stoichiometric coefficients > 1, list the molecule species multiple times.
        km_range: The range from which to sample Michaelis Menten constants for this reaction (in mol).
            The sampling will happen in the log transformed intervall, so all values must be > 0.
        vmax_range: The range from which to sample maximum velocities for this reaction (in mol per time step).
            The sampling will happen in the log transformed intervall, so all values must be > 0.
        n_reaction_nts: The number of nucleotides that should encode the reaction.
        n_affinity_nts: The number of nucleotides that should encode the Michaelis Menten constant.
        n_velocity_nts: The number of nucleotides that should encode the maximum Velocity.
        n_orientation_nts: The number of nucleotides that should encode the orientation of the reaction.

    This factory randomly assigns nucleotide sequences to reactions, Michaelis Menten constants,
    maximum velocities, and domain orientations on initialization.
    When calling the instantiated factory with a nucleotide sequence, it will return the encoded domain.
    How many nucleotides encode which part of the domain is defined by the `n_*_nts` arguments.
    The overall length of this domain type in nucleotides is the sum of all these.
    These domain factories are instantiated when initializing [Genetics][magicsoup.genetics.Genetics],
    which also happens when initializing [World][magicsoup.world.World].
    They are used during translation to map nucleotide sequences to domains.

    ```
        DomainFact = CatalyticFact(reactions=[([A], [B])])
        DomainFact("ATCGATATATTTGCAAATTGA")
    ```

    After this factory has been instantiated you can look into the mappings that map nucleotide sequences to domain specifications.
    They are on these attributes:

    - `reaction_map` maps nucleotide sequences to reactions
    - `affinity_map` maps nucleotide sequences to Michaelis Menten constants
    - `velocity_map` maps nucleotide sequences to maximum velocities
    - `orientation_map` maps nucleotide sequences to orientations

    Any reaction can happen in both directions, so it is not necessary to define the reverse reaction again.
    The orientation of a reaction only matters in combination with other domains
    in the way how a protein energetically couples multiple actions.
    This orientation is defined by an attribute `is_bkwd` on the domain.
    That attribute will be sampled during initialization of this factory and there is a 1:1 chance for any orientation.
    """

    def __init__(
        self,
        reactions: list[tuple[list[Molecule], list[Molecule]]],
        km_range: tuple[float, float] = (1e-5, 1.0),
        vmax_range: tuple[float, float] = (0.01, 10.0),
        n_reaction_nts: int = 6,
        n_affinity_nts: int = 6,
        n_velocity_nts: int = 6,
        n_orientation_nts: int = 3,
    ):
        self.reaction_map = generic_map_fact(nt_seqs(n_reaction_nts), reactions)
        self.affinity_map = log_weight_map_fact(nt_seqs(n_affinity_nts), *km_range)
        self.velocity_map = log_weight_map_fact(nt_seqs(n_velocity_nts), *vmax_range)
        self.orientation_map = bool_map_fact(nt_seqs(n_orientation_nts))

        _check_n(n_reaction_nts, "n_reaction_nts")
        _check_n(n_affinity_nts, "n_affinity_nts")
        _check_n(n_velocity_nts, "n_velocity_nts")
        _check_n(n_orientation_nts, "n_orientation_nts")

        if len(reactions) > 4**n_reaction_nts:
            raise ValueError(
                f"There are {len(reactions)} reactions."
                f" But with n_reaction_nts={n_reaction_nts} only {4 ** n_reaction_nts} reactions can be encoded."
            )

        self.react_slice = slice(0, n_reaction_nts)
        self.aff_slice = slice(n_reaction_nts, n_reaction_nts + n_affinity_nts)
        self.velo_slice = slice(
            n_reaction_nts + n_affinity_nts,
            n_reaction_nts + n_affinity_nts + n_velocity_nts,
        )
        self.orient_slice = slice(
            n_reaction_nts + n_affinity_nts + n_velocity_nts,
            n_reaction_nts + n_affinity_nts + n_velocity_nts + n_orientation_nts,
        )

        self.min_len = (
            n_reaction_nts + n_affinity_nts + n_velocity_nts + n_orientation_nts
        )

    def __call__(self, seq: str) -> Domain:
        react = self.reaction_map[seq[self.react_slice]]
        aff = self.affinity_map[seq[self.aff_slice]]
        velo = self.velocity_map[seq[self.velo_slice]]
        is_bkwd = self.orientation_map[seq[self.orient_slice]]
        return CatalyticDomain(
            reaction=react, affinity=aff, velocity=velo, is_bkwd=is_bkwd
        )


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
        super().__init__(
            substrates=[molecule],
            products=[],
            affinity=affinity,
            velocity=velocity,
            is_bkwd=is_bkwd,
            is_transporter=True,
        )

    def __str__(self) -> str:
        return f"TransporterDomain({self.substrates[0]})"


class TransporterFact(_DomainFact):
    """
    Factory for generating transporter domains from nucleotide sequences.

    Arguments:
        molecules: All molecule species that can be transported into or out of the cell.
        km_range: The range from which to sample Michaelis Menten constants for this transport (in mol).
            The sampling will happen in the log transformed intervall, so all values must be > 0.
        vmax_range: The range from which to sample maximum velocities for this transport (in mol per time step).
            The sampling will happen in the log transformed intervall, so all values must be > 0.
        n_molecule_nts: The number of nucleotides that should encode the molecule species.
        n_affinity_nts: The number of nucleotides that should encode the Michaelis Menten constant.
        n_velocity_nts: The number of nucleotides that should encode the maximum Velocity.
        n_orientation_nts: The number of nucleotides that should encode the orientation of the transport.

    This factory randomly assigns nucleotide sequences to molecule species, Michaelis Menten constants,
    maximum velocities, and domain orientations on initialization.
    When calling the instantiated factory with a nucleotide sequence, it will return the encoded domain.
    How many nucleotides encode which part of the domain is defined by the `n_*_nts` arguments.
    The overall length of this domain type in nucleotides is the sum of all these.
    These domain factories are instantiated when initializing [Genetics][magicsoup.genetics.Genetics],
    which also happens when initializing [World][magicsoup.world.World].
    They are used during translation to map nucleotide sequences to domains.

    ```
        DomainFact = TransporterFact(molecules=[A])
        DomainFact("ATCGATATATTTGCAAATTGA")
    ```

    After this factory has been instantiated you can look into the mappings that map nucleotide sequences to domain specifications.
    They are on these attributes:

    - `molecule_map` maps nucleotide sequences to molecules
    - `affinity_map` maps nucleotide sequences to Michaelis Menten constants
    - `velocity_map` maps nucleotide sequences to maximum velocities
    - `orientation_map` maps nucleotide sequences to orientations

    Any transporter works in both directions.
    The orientation only matters in combination with other domains in the way how a protein energetically couples multiple actions.
    This orientation is defined by an attribute `is_bkwd` on the domain.
    That attribute will be sampled during initialization of this factory and there is a 1:1 chance for any orientation.
    """

    def __init__(
        self,
        molecules: list[Molecule],
        km_range: tuple[float, float] = (1e-5, 1.0),
        vmax_range: tuple[float, float] = (0.01, 10.0),
        n_molecule_nts: int = 6,
        n_affinity_nts: int = 6,
        n_velocity_nts: int = 6,
        n_orientation_nts: int = 3,
    ):
        self.molecule_map = generic_map_fact(nt_seqs(n_molecule_nts), molecules)
        self.affinity_map = log_weight_map_fact(nt_seqs(n_affinity_nts), *km_range)
        self.velocity_map = log_weight_map_fact(nt_seqs(n_velocity_nts), *vmax_range)
        self.orientation_map = bool_map_fact(nt_seqs(n_orientation_nts))

        _check_n(n_molecule_nts, "n_molecule_nts")
        _check_n(n_affinity_nts, "n_affinity_nts")
        _check_n(n_velocity_nts, "n_velocity_nts")
        _check_n(n_orientation_nts, "n_orientation_nts")

        if len(molecules) > 4**n_molecule_nts:
            raise ValueError(
                f"There are {len(molecules)} molecules."
                f" But with n_molecule_nts={n_molecule_nts} only {4 ** n_molecule_nts} molecules can be encoded."
            )

        self.mol_slice = slice(0, n_molecule_nts)
        self.aff_slice = slice(n_molecule_nts, n_molecule_nts + n_affinity_nts)
        self.velo_slice = slice(
            n_molecule_nts + n_affinity_nts,
            n_molecule_nts + n_affinity_nts + n_velocity_nts,
        )
        self.orient_slice = slice(
            n_molecule_nts + n_affinity_nts + n_velocity_nts,
            n_molecule_nts + n_affinity_nts + n_velocity_nts + n_orientation_nts,
        )

        self.min_len = (
            n_molecule_nts + n_affinity_nts + n_velocity_nts + n_orientation_nts
        )

    def __call__(self, seq: str) -> Domain:
        mol = self.molecule_map[seq[self.mol_slice]]
        aff = self.affinity_map[seq[self.aff_slice]]
        velo = self.velocity_map[seq[self.velo_slice]]
        is_bkwd = self.orientation_map[seq[self.orient_slice]]
        return TransporterDomain(
            molecule=mol, affinity=aff, velocity=velo, is_bkwd=is_bkwd
        )


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
        super().__init__(
            substrates=[effector],
            products=[],
            affinity=affinity,
            velocity=0.0,
            is_bkwd=False,
            is_regulatory=True,
            is_inhibiting=is_inhibiting,
            is_transmembrane=is_transmembrane,
        )

    def __str__(self) -> str:
        loc = "transmembrane" if self.is_transmembrane else "cytosolic"
        eff = "inhibiting" if self.is_inhibiting else "activating"
        return f"ReceptorDomain({self.substrates[0]},{loc},{eff})"


class RegulatoryFact(_DomainFact):
    """
    Factory for generating regulatory domains from nucleotide sequences.

    Arguments:
        molecules: All molecule species that can be inhibiting or activating effectors.
        km_range: The range from which to sample Michaelis Menten constants for regulatory activity (in mol).
            The sampling will happen in the log transformed intervall, so all values must be > 0.
        n_molecule_nts: The number of nucleotides that should encode the molecule species.
        n_affinity_nts: The number of nucleotides that should encode the Michaelis Menten constant.
        n_transmembrane_nts: The number of nucleotides that should encode whether the domain also is a transmembrane domain
            (which will make it react to extracellular molecules instead).
        n_inhibit_nts: The number of nucleotides that should encode whether it is an activating or inhibiting regulatory domain.

    This factory randomly assigns nucleotide sequences to molecule species, Michaelis Menten constants,
    whether the domain is transmembrane, and whether the domain is inhibiting on initialization.
    When calling the instantiated factory with a nucleotide sequence, it will return the encoded domain.
    How many nucleotides encode which part of the domain is defined by the `n_*_nts` arguments.
    The overall length of this domain type in nucleotides is the sum of all these.
    These domain factories are instantiated when initializing [Genetics][magicsoup.genetics.Genetics],
    which also happens when initializing [World][magicsoup.world.World].
    They are used during translation to map nucleotide sequences to domains.

    ```
        DomainFact = RegulatoryFact(molecules=[A])
        DomainFact("ATCGATATATTTGCAAAT")
    ```

    After this factory has been instantiated you can look into the mappings that map nucleotide sequences to domain specifications.
    They are on these attributes:

    - `molecule_map` maps nucleotide sequences to molecules
    - `affinity_map` maps nucleotide sequences to Michaelis Menten constants
    - `transmembrane_map` maps nucleotide sequences to the transmembrane flag
    - `orientation_map` maps nucleotide sequences to orientations

    I think the term Michaelis Menten constant in a regulatory domain is a bit weird
    since there is no product being created.
    However, the kinetics of the amount of activation or inhibition are the same.
    """

    def __init__(
        self,
        molecules: list[Molecule],
        km_range: tuple[float, float] = (1e-5, 1.0),
        n_molecule_nts: int = 6,
        n_affinity_nts: int = 6,
        n_transmembrane_nts: int = 3,
        n_inhibit_nts: int = 3,
    ):
        self.molecule_map = generic_map_fact(nt_seqs(n_molecule_nts), molecules)
        self.affinity_map = log_weight_map_fact(nt_seqs(n_affinity_nts), *km_range)
        self.transmembrane_map = bool_map_fact(nt_seqs(n_transmembrane_nts))
        self.inhibit_map = bool_map_fact(nt_seqs(n_inhibit_nts))

        _check_n(n_molecule_nts, "n_molecule_nts")
        _check_n(n_affinity_nts, "n_affinity_nts")
        _check_n(n_transmembrane_nts, "n_transmembrane_nts")
        _check_n(n_inhibit_nts, "n_inhibit_nts")

        if len(molecules) > 4**n_molecule_nts:
            raise ValueError(
                f"There are {len(molecules)} molecules."
                f" But with n_molecule_nts={n_molecule_nts} only {4 ** n_molecule_nts} molecules can be encoded."
            )

        self.mol_slice = slice(0, n_molecule_nts)
        self.aff_slice = slice(n_molecule_nts, n_molecule_nts + n_affinity_nts)
        self.trans_slice = slice(
            n_molecule_nts + n_affinity_nts,
            n_molecule_nts + n_affinity_nts + n_transmembrane_nts,
        )
        self.inh_slice = slice(
            n_molecule_nts + n_affinity_nts + n_transmembrane_nts,
            n_molecule_nts + n_affinity_nts + n_transmembrane_nts + n_inhibit_nts,
        )

        self.min_len = (
            n_molecule_nts + n_affinity_nts + n_transmembrane_nts + n_inhibit_nts
        )

    def __call__(self, seq: str) -> Domain:
        mol = self.molecule_map[seq[self.mol_slice]]
        aff = self.affinity_map[seq[self.aff_slice]]
        trans = self.transmembrane_map[seq[self.inh_slice]]
        inh = self.inhibit_map[seq[self.inh_slice]]
        return RegulatoryDomain(
            effector=mol,
            affinity=aff,
            is_inhibiting=inh,
            is_transmembrane=trans,
        )


def _get_n(p: float, s: int, name: str) -> int:
    n = int(p * s)
    if n == 0 and p > 0.0:
        warnings.warn(
            f"There will be no {name}."
            f" Increase dom_type_size to accomodate low probabilities of having {name}."
        )
    return n


# TODO: tryout
def get_coding_regions(
    seq: str,
    min_cds_size: int,
    start_codons: tuple[str, ...],
    stop_codons: tuple[str, ...],
) -> list[str]:
    s = CODON_SIZE
    n = len(seq)
    max_start_idx = n - min_cds_size

    start_idxs = []
    for start_codon in start_codons:
        i = 0
        while i < max_start_idx:
            try:
                hit = seq[i:].index(start_codon)
                start_idxs.append(i + hit)
                i = i + hit + s
            except ValueError:
                break

    stop_idxs = []
    for stop_codon in stop_codons:
        i = 0
        while i < n - s:
            try:
                hit = seq[i:].index(stop_codon)
                stop_idxs.append(i + hit)
                i = i + hit + s
            except ValueError:
                break

    start_idxs.sort()
    stop_idxs.sort()

    by_frame: list[tuple[list[int], ...]] = [([], []), ([], []), ([], [])]
    for start_idx in start_idxs:
        if start_idx % 3 == 0:
            by_frame[0][0].append(start_idx)
        elif (start_idx + 1) % 3 == 0:
            by_frame[1][0].append(start_idx)
        else:
            by_frame[2][0].append(start_idx)
    for stop_idx in stop_idxs:
        if stop_idx % 3 == 0:
            by_frame[0][1].append(stop_idx)
        elif (stop_idx + 1) % 3 == 0:
            by_frame[1][1].append(stop_idx)
        else:
            by_frame[2][1].append(stop_idx)

    cdss = []
    for start_idxs, stop_idxs in by_frame:
        for start_idx in start_idxs:
            stop_idxs = [d for d in stop_idxs if d > start_idx + s]
            if len(stop_idxs) > 0:
                cds_end_idx = min(stop_idxs) + s
                if cds_end_idx - start_idx > min_cds_size:
                    cdss.append(seq[start_idx:cds_end_idx])
            else:
                break

    return cdss


# TODO: tryout
def get_all_cdss(
    genome: str,
    min_cds_size: int,
    start_codons: tuple[str, ...],
    stop_codons: tuple[str, ...],
) -> list[str]:
    cdsf = get_coding_regions(
        seq=genome,
        min_cds_size=min_cds_size,
        start_codons=start_codons,
        stop_codons=stop_codons,
    )
    bwd = reverse_complement(genome)
    cdsb = get_coding_regions(
        seq=bwd,
        min_cds_size=min_cds_size,
        start_codons=start_codons,
        stop_codons=stop_codons,
    )
    return cdsf + cdsb


class Genetics:
    """
    Class holding logic about translating nucleotide sequences into proteomes.

    Arguments:
        chemistry: The chemistry object used for this simulation.
            If no reactions were defined, there will be no catalytic domain factory, i.e. no catalytic domains defined.
        start_codons: Start codons which start a coding sequence
        stop_codons: Stop codons which stop a coding sequence
        p_catal_dom: Chance of encountering a catalytic domain in a random nucleotide sequence.
        p_transp_dom: Chance of encountering a transporter domain in a random nucleotide sequence.
        p_allo_dom: Chance of encountering a regulatory domain in a random nucleotide sequence.
        n_dom_type_nts: Number of nucleotides that encodes the domain type (catalytic, transporter, regulatory).
        n_reaction_nts: Number of nucleotides that encodes the reaction in catalytic domains
            (will be passed to catalytic domain factory).
        n_molecule_nts: Number of nucleotides that encodes the molecule species in transporter and regulatory domain
            (will be passed to their domain factories).
        n_affinity_nts: Number of nucleotides that encodes the Michaelis Menten constants of domains
            (will be passed to domain factories).
        n_velocity_nts: Number of nucleotides that encodes maximum velocitires in catalytic and regulatory domains
            (will be passed to their domain factories).
        n_orientation_nts: Number of nucleotides that encodes domain orientation
            (will be passed to domain factories).
        n_transmembrane_nts: Number of nucleotides that encodes whether a regulatory domain is also a transmembrane domain,
            reacting to extracellular molecules instead (will be passed to regulatory domain factory).
        n_inhibit_nts: Number of nucleotides that encodes whether a regulatory domain is a inhibiting or activating
            (will be passed to regulatory domain factory).
        workers: number of workers

    When this class is initialized it generates the mappings from nucleotide sequences to domains by random sampling.
    These mappings are then used throughout the simulation.
    If you initialize this class again, these mappings will be different.
    Initializing [World][magicsoup.world.World] will also create one `Genetics` instance. It is on `world.genetics`.
    If you want to access nucleotide to domain mappings of your simulation, you should use `world.genetics`.

    During the simulation [World][magicsoup.world.World] uses `genetics.get_proteome()` on all genomes to get the proteome for each cell.
    If you are interested in CDSs only you can use [get_coding_regions()][magicsoup.genetics.Genetics.get_coding_regions] to get all CDs for a particular genome.
    To translate a single CDS you can use [translate_seq()][magicsoup.genetics.Genetics.translate_seq].

    The attribute `genetics.domain_map` holds the actual domain mappings.
    This maps nucleotide sequences to a domain factory.
    Any of these domain factories is either a catalytic, transporter, or regulatory domain factory.
    For how nucleotides map to further domain specifications (e.g. affinity) is saved on the domain factory object.

    Translation happens only within coding sequences (CDSs).
    A CDS starts wherever a start codon is found and ends with the first in-frame encountered stop codon.
    Un-stopped CDSs are discarded.
    Both the forward and reverse complement of the nucleotide sequence are considered.
    Each CDS represents one protein.
    All domains found within a CDS will be added as domains to that protein.
    Unviable proteins, like proteins without domains or proteins with only regulatory domains, are discarded.

    If you want to use your own `genetics` object for the simulation you can just assign it after creating [World][magicsoup.world.World]:

    ```
        world = World(chemistry=chemistry)
        my_genetics = Genetics(chemistry=chemistry, p_transp_dom=0.1, stop_codons=("TGA", ))
        world.genetics = my_genetics
    ```

    Changing these attributes has a large effect on the information content of genomes.
    E.g. how many CDSs per nucleotide, i.e. how many proteins per nucleotide;
    how long are the average CDSs, i.e. how many domains per protein; how likely is it to encounter a domain;
    how many domains does each nucleotide encode, i.e. how likely is it for a single substitution to change the proteome,
    how likely is it that this mutation will only slightly change a domain or completely change it.
    """

    def __init__(
        self,
        chemistry: Chemistry,
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        p_catal_dom: float = 0.01,
        p_transp_dom: float = 0.01,
        p_allo_dom: float = 0.01,
        n_dom_type_nts: int = 6,
        n_reaction_nts: int = 6,
        n_molecule_nts: int = 6,
        n_affinity_nts: int = 6,
        n_velocity_nts: int = 6,
        n_orientation_nts: int = 3,
        n_transmembrane_nts: int = 3,
        n_inhibit_nts: int = 3,
        workers: int = 2,
    ):
        if any(len(d) != CODON_SIZE for d in start_codons):
            raise ValueError(f"Not all start codons are of length {CODON_SIZE}")
        if any(len(d) != CODON_SIZE for d in stop_codons):
            raise ValueError(f"Not all stop codons are of length {CODON_SIZE}")
        if len(set(start_codons) & set(stop_codons)) > 0:
            raise ValueError(
                "Overlapping start and stop codons:"
                f" {','.join(str(d) for d in set(start_codons) & set(stop_codons))}"
            )
        if n_dom_type_nts % CODON_SIZE != 0:
            raise ValueError(f"n_dom_type_nts should be a multiple of {CODON_SIZE}.")

        self.chemistry = chemistry
        self.start_codons = start_codons
        self.stop_codons = stop_codons
        self.dom_type_size = n_dom_type_nts
        self.workers = workers

        if p_catal_dom + p_transp_dom + p_allo_dom > 1.0:
            raise ValueError(
                "p_catal_dom, p_transp_dom, p_allo_dom together must not be greater 1.0"
            )

        sets = nt_seqs(n_dom_type_nts)
        random.shuffle(sets)
        n = len(sets)

        n_catal_doms = _get_n(p=p_catal_dom, s=n, name="catalytic domains")
        n_transp_doms = _get_n(p=p_transp_dom, s=n, name="transporter domains")
        n_allo_doms = _get_n(p=p_allo_dom, s=n, name="allosteric domains")

        catal_dom_fact = CatalyticFact(
            reactions=chemistry.reactions,
            n_reaction_nts=n_reaction_nts,
            n_affinity_nts=n_affinity_nts,
            n_velocity_nts=n_velocity_nts,
            n_orientation_nts=n_orientation_nts,
        )
        transp_dom_fact = TransporterFact(
            molecules=chemistry.molecules,
            n_molecule_nts=n_molecule_nts,
            n_affinity_nts=n_affinity_nts,
            n_velocity_nts=n_velocity_nts,
            n_orientation_nts=n_orientation_nts,
        )
        allo_dom_fact = RegulatoryFact(
            molecules=chemistry.molecules,
            n_molecule_nts=n_molecule_nts,
            n_affinity_nts=n_affinity_nts,
            n_transmembrane_nts=n_transmembrane_nts,
            n_inhibit_nts=n_inhibit_nts,
        )

        domain_facts: dict[_DomainFact, list[str]] = {}
        if len(self.chemistry.reactions) > 0:
            domain_facts[catal_dom_fact] = sets[:n_catal_doms]
            del sets[:n_catal_doms]
        domain_facts[transp_dom_fact] = sets[:n_transp_doms]
        del sets[:n_transp_doms]
        domain_facts[allo_dom_fact] = sets[:n_allo_doms]
        del sets[:n_allo_doms]

        self.domain_map = {d: k for k, v in domain_facts.items() for d in v}

        self.dom_details_size = max(d.min_len for d in domain_facts)
        self.dom_size = self.dom_type_size + self.dom_details_size
        self.min_cds_size = self.dom_size + 2 * CODON_SIZE

        # TODO: tryout
        self.n_nts = {
            "n_dom_type_nts": n_dom_type_nts,
            "n_reaction_nts": n_reaction_nts,
            "n_molecule_nts": n_molecule_nts,
            "n_affinity_nts": n_affinity_nts,
            "n_velocity_nts": n_velocity_nts,
            "n_bool_nts": n_orientation_nts,
            "n_molecules": len(chemistry.molecules),
        }

        self._dom_map: dict[str, tuple[bool, bool, bool]] = {}
        for seq, dom_fact in self.domain_map.items():
            if isinstance(dom_fact, CatalyticFact):
                self._dom_map[seq] = (True, False, False)
            if isinstance(dom_fact, TransporterFact):
                self._dom_map[seq] = (False, True, False)
            if isinstance(dom_fact, RegulatoryFact):
                self._dom_map[seq] = (False, False, True)

        _react_map = generic_map_fact(nt_seqs(n_reaction_nts), chemistry.reactions)
        self._react_map: dict[str, tuple[list[int], list[int]]] = {}
        for seq, (subs, prods) in _react_map.items():
            subsi = [chemistry.molecules.index(d) for d in subs]
            prodsi = [chemistry.molecules.index(d) for d in prods]
            self._react_map[seq] = (subsi, prodsi)

        _mol_map = generic_map_fact(nt_seqs(n_reaction_nts), chemistry.molecules)
        self._mol_map = {k: chemistry.molecules.index(v) for k, v in _mol_map.items()}

        self._aff_map = log_weight_map_fact(nt_seqs(n_affinity_nts), 1e-5, 1.0)
        self._velo_map = log_weight_map_fact(nt_seqs(n_velocity_nts), 0.01, 10)
        self._bool_map = bool_map_fact(nt_seqs(n_orientation_nts))

    # TODO: tryout
    def get_all_cdss(self, genomes: list[str]) -> list[list[str]]:
        inputs = [
            (d, self.min_cds_size, self.start_codons, self.stop_codons) for d in genomes
        ]

        with mp.Pool(self.workers) as pool:
            cdss = pool.starmap(get_all_cdss, inputs)

        return cdss

    # TODO: tryout
    def get_all_proteomes(
        self, cdss_lst: list[list[str]], n_prots: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_cells = len(cdss_lst)
        n_doms = 10
        n_mols = len(self.chemistry.molecules)
        dims = [n_cells, n_prots, n_doms]

        # TODO: n_prots might be smaller than current p
        #       but I should already have correct p here
        #       because later I want to broadcast them

        N = torch.zeros(*dims, n_mols * 2).share_memory_()
        A = torch.zeros(*dims, n_mols * 2).share_memory_()
        Km = torch.full((*dims, n_mols * 2), torch.nan).share_memory_()
        Vmax = torch.full((*dims,), torch.nan).share_memory_()

        # TODO: diese Dinger nach Kinetics umzuiehen, dann habe ich auch zugriff
        #       auf die aktuelle protein dimension
        #       außerdem kann ich die vielleicht einmal instantiierenund dann
        #       immer wieder neu verwenden
        #       muss ich testen, resizen geht nicht, aber während dem multiproc muss
        #       ich das ja nicht.
        #       man könnte das resizen später auch etwas robuster machen indem man
        #       immer +10 proteine/domänen resized und es dann so lässt bis wieder weniger
        #       proteine kommen
        #       bei den Zellen auch, dann hat man da immer eine Ablage mit der man arbeitet

        maps = {
            "domain_map": self._dom_map,
            "molecule_map": self._mol_map,
            "reaction_map": self._react_map,
            "affinity_map": self._aff_map,
            "velocity_map": self._velo_map,
            "bool_map": self._bool_map,
        }

        args = [
            (
                i,
                d,
                maps,
                self.n_nts,
                self.dom_size,
                n_mols,
                N,
                A,
                Km,
                Vmax,
            )
            for i, d in enumerate(cdss_lst)
        ]

        print("start pool")
        with mp.Pool(self.workers) as pool:
            pool.starmap(translate_all_seqs, args)
        print("finish pool")

        return N, A, Km, Vmax

    def get_proteome(self, seq: str) -> list[Protein]:
        """
        Get all proteins encoded by a nucleotide sequence

        Arguments:
            seq: nucleotide sequence

        Both forward and reverse-complement are considered.
        CDSs are extracted (see [get_coding_regions()][magicsoup.genetics.Genetics.get_coding_regions])
        and a protein is translated for every CDS (see [translate_seq()][magicsoup.genetics.Genetics.translate_seq]).
        Unviable proteins (no domains or only regulatory domains) are discarded.
        """
        bwd = reverse_complement(seq)
        cds = list(set(self.get_coding_regions(seq) + self.get_coding_regions(bwd)))
        proteins = [self.translate_seq(d) for d in cds]
        proteins = [d for d in proteins if len(d) > 0]
        proteins = [d for d in proteins if not all(dd.is_regulatory for dd in d)]
        return [Protein(domains=d, label=f"P{i}") for i, d in enumerate(proteins)]

    def get_coding_regions(self, seq: str) -> list[str]:
        """
        Get all coding regions in nucleotide sequence

        Arguments:
            seq: nucleotide sequence

        Assuming coding region can start at any start codon
        and is stopped with the first in-frame stop codon encountered.
        Coding regions without a stop codon are not considerd.
        """
        n = len(seq)
        max_start_idx = n - self.min_cds_size

        start_idxs = []
        for start_codon in self.start_codons:
            i = 0
            while i < max_start_idx:
                try:
                    hit = seq[i:].index(start_codon)
                    start_idxs.append(i + hit)
                    i = i + hit + CODON_SIZE
                except ValueError:
                    break

        stop_idxs = []
        for stop_codon in self.stop_codons:
            i = 0
            while i < n - CODON_SIZE:
                try:
                    hit = seq[i:].index(stop_codon)
                    stop_idxs.append(i + hit)
                    i = i + hit + CODON_SIZE
                except ValueError:
                    break

        start_idxs.sort()
        stop_idxs.sort()

        by_frame: list[tuple[list[int], ...]] = [([], []), ([], []), ([], [])]
        for start_idx in start_idxs:
            if start_idx % 3 == 0:
                by_frame[0][0].append(start_idx)
            elif (start_idx + 1) % 3 == 0:
                by_frame[1][0].append(start_idx)
            else:
                by_frame[2][0].append(start_idx)
        for stop_idx in stop_idxs:
            if stop_idx % 3 == 0:
                by_frame[0][1].append(stop_idx)
            elif (stop_idx + 1) % 3 == 0:
                by_frame[1][1].append(stop_idx)
            else:
                by_frame[2][1].append(stop_idx)

        cdss = []
        for start_idxs, stop_idxs in by_frame:
            for start_idx in start_idxs:
                stop_idxs = [d for d in stop_idxs if d > start_idx + CODON_SIZE]
                if len(stop_idxs) > 0:
                    cds_end_idx = min(stop_idxs) + CODON_SIZE
                    if cds_end_idx - start_idx > self.min_cds_size:
                        cdss.append(seq[start_idx:cds_end_idx])
                else:
                    break

        return cdss

    def translate_seq(self, seq: str) -> list[Domain]:
        """
        Translate a coding region into a protein.
        The CDS should be a desoxy-ribonucleotide sequence (i.e. TGCA).

        Arguments:
            seq: nucleotide sequence
        """
        i = 0
        j = self.dom_type_size
        doms: list[Domain] = []
        while i + self.dom_size <= len(seq):
            domfact = self.domain_map.get(seq[i:j])
            if domfact is not None:
                dom = domfact(seq[j : j + self.dom_details_size])
                doms.append(dom)
                i += self.dom_size
                j += self.dom_size
            else:
                i += CODON_SIZE
                j += CODON_SIZE

        return doms
