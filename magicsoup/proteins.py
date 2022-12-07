import abc
from magicsoup.util import (
    weight_map_fact,
    double_bool_map_fact,
    reverse_complement,
    CODON_SIZE,
)
import torch


class Molecule:
    def __init__(self, name: str, energy: float, is_intracellular=True):
        self.name = name
        self.energy = energy
        self.is_intracellular = is_intracellular

    def __hash__(self) -> int:
        clsname = type(self).__name__
        return hash((clsname, self.name, self.energy, self.is_intracellular))

    def __eq__(self, other) -> bool:
        return hash(self) == hash(other)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(name=%r,energy=%r,is_intracellular=%r)" % (
            clsname,
            self.name,
            self.energy,
            self.is_intracellular,
        )

    def __str__(self) -> str:
        prefix = "i" if self.is_intracellular else "e"
        return prefix + "-" + self.name


class Domain:
    def __init__(
        self,
        substrates: tuple[Molecule, ...],
        products: tuple[Molecule, ...],
        affinity: float,
        velocity: float,
        energy: float,
        is_transporter=False,
        is_receptor=False,
        is_inhibiting=False,
    ):
        self.substrates = substrates
        self.products = products
        self.affinity = affinity
        self.velocity = velocity
        self.energy = energy

        self.is_transporter = is_transporter
        self.is_receptor = is_receptor
        self.is_inhibiting = is_inhibiting

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return (
            "%s(substrates=%r,products=%r,affinity=%r,velocity=%r,is_transporter=%r,is_receptor=%r,is_inhibiting=%r)"
            % (
                clsname,
                self.substrates,
                self.products,
                self.affinity,
                self.velocity,
                self.is_transporter,
                self.is_receptor,
                self.is_inhibiting,
            )
        )

    def __str__(self) -> str:
        ins = ",".join(str(d) for d in self.substrates)
        outs = ",".join(str(d) for d in self.products)
        if self.is_transporter:
            return f"TransporterDomain({ins}->{outs})"
        if self.is_receptor:
            return f"ReceptorDomain({ins})"
        return f"EnzymeDomain({ins}->{outs})"


class DomainFact(abc.ABC):
    @abc.abstractmethod
    def __call__(self, seq: str) -> Domain:
        raise NotImplementedError("Implement __call__")


class EnzymeFact(DomainFact):
    def __init__(
        self,
        reaction_map: dict[str, tuple[tuple[Molecule, ...], tuple[Molecule, ...]]],
        affinity_map: dict[str, float],
        velocity_map: dict[str, float],
    ):
        energies: dict[str, float] = {}
        for seq, (substrates, products) in reaction_map.items():
            energy = 0.0
            for sig in substrates:
                energy -= sig.energy
            for sig in products:
                energy += sig.energy
            energies[seq] = energy

        self.energy_map = energies
        self.reaction_map = reaction_map
        self.affinity_map = affinity_map
        self.velocity_map = velocity_map

        self.n_nts = len(next(iter(reaction_map)))
        # TODO: validate lengths

    def __call__(self, seq: str) -> Domain:
        subs, prods = self.reaction_map[seq[0 : self.n_nts]]
        energy = self.energy_map[seq[0 : self.n_nts]]
        aff = self.affinity_map[seq[self.n_nts : self.n_nts * 2]]
        velo = self.velocity_map[seq[self.n_nts * 2 :]]
        return Domain(
            substrates=subs, products=prods, affinity=aff, velocity=velo, energy=energy,
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(reactions=%r)" % (clsname, set(self.reaction_map.values()))


class TransporterFact(DomainFact):
    def __init__(
        self,
        molecule_map: dict[str, Molecule],
        affinity_map: dict[str, float],
        velocity_map: dict[str, float],
    ):

        self.molecule_map = molecule_map
        self.affinity_map = affinity_map
        self.velocity_map = velocity_map

        self.n_nts = len(next(iter(molecule_map)))
        # TODO: validate lengths

    def __call__(self, seq: str) -> Domain:
        mol1 = self.molecule_map[seq[0 : self.n_nts]]
        aff = self.affinity_map[seq[self.n_nts : self.n_nts * 2]]
        velo = self.velocity_map[seq[self.n_nts * 2 :]]
        mol1.is_intracellular = True
        mol2 = Molecule(name=mol1.name, energy=mol1.energy, is_intracellular=False)
        return Domain(
            substrates=(mol1,),
            products=(mol2,),
            affinity=aff,
            velocity=velo,
            energy=0.0,
            is_transporter=True,
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(molecules=%r)" % (clsname, set(self.molecule_map.values()))


class AllostericFact(DomainFact):
    def __init__(
        self,
        molecule_map: dict[str, Molecule],
        affinity_map: dict[str, float],
        allosteric_map: dict[str, tuple[bool, bool]],
    ):

        self.molecule_map = molecule_map
        self.affinity_map = affinity_map
        self.allosteric_map = allosteric_map

        self.n_nts = len(next(iter(molecule_map)))
        # TODO: validate lengths

    def __call__(self, seq: str) -> Domain:
        mol = self.molecule_map[seq[0 : self.n_nts]]
        aff = self.affinity_map[seq[self.n_nts : self.n_nts * 2]]
        inh, intr = self.allosteric_map[seq[self.n_nts * 2 :]]
        mol.is_intracellular = intr
        return Domain(
            substrates=(mol,),
            products=tuple(),
            affinity=aff,
            velocity=0.0,
            energy=0.0,
            is_receptor=True,
            is_inhibiting=inh,
        )

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return "%s(molecules=%r)" % (clsname, set(self.molecule_map.values()))


class Proteins:
    def __init__(
        self,
        molecules: list[Molecule],
        domain_map: dict[DomainFact, list[str]],
        vmax_range: tuple[float, float] = (0.2, 5.0),
        km_range: tuple[float, float] = (0.1, 10.0),
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
    ):
        self.molecules = molecules
        self.domain_map = domain_map
        self.vmax_range = vmax_range
        self.km_range = km_range
        self.start_codons = start_codons
        self.stop_codons = stop_codons

        self.seq_2_dom = {d: k for k, v in self.domain_map.items() for d in v}
        self.n_dom_type_def_nts = len(next(iter(self.seq_2_dom)))
        self.n_dom_detail_def_nts = 3 * CODON_SIZE
        self.n_dom_def_nts = self.n_dom_type_def_nts + self.n_dom_detail_def_nts
        self.codon_2_vmax = weight_map_fact(CODON_SIZE, *vmax_range)
        self.codon_2_km = weight_map_fact(CODON_SIZE, *km_range)
        self.codon_2_allo = double_bool_map_fact(CODON_SIZE)
        self.min_n_seq_nts = self.n_dom_def_nts + 2 * CODON_SIZE

        # 1: internal molecules, 2: external molecules
        self.int_mol_pad = 0
        self.ext_mol_pad = len(molecules)
        self.n_molecules = len(molecules) * 2

    def get_coding_regions(self, seq: str) -> list[str]:
        """
        Get all possible coding regions in nucleotide sequence

        Assuming coding region can start at any start codon and
        is stopped with the first stop codon encountered in same
        frame.

        Ribosomes will stall without stop codon. So, a coding region
        without a stop codon is not considerd.
        (https://pubmed.ncbi.nlm.nih.gov/27934701/)
        """
        cdss = []
        hits: list[list[int]] = [[] for _ in range(CODON_SIZE)]
        i = 0
        j = CODON_SIZE
        k = 0
        n = len(seq) + 1
        while j <= n:
            codon = seq[i:j]
            if codon in self.start_codons:
                hits[k].append(i)
            elif codon in self.stop_codons:
                for hit in hits[k]:
                    cdss.append(seq[hit:j])
                hits[k] = []
            i += 1
            j += 1
            k = i % CODON_SIZE
        return cdss

    def get_proteome(
        self, seq: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all possible proteins encoded by a nucleotide sequence.
        Proteins are represented as dicts with domain labels and correspondig
        weights.
        
        Proteins which could theoretically be translated, but from which we can
        already tell by now that they would not be functional, will be sorted
        out at this point.
        """
        bwd = reverse_complement(seq)
        cds = list(set(self.get_coding_regions(seq) + self.get_coding_regions(bwd)))
        cds = [d for d in cds if len(d) > self.min_n_seq_nts]
        tensors = [self.translate_seq(d) for d in cds]
        Km = torch.stack([d[0] for d in tensors])
        Vmax = torch.stack([d[1] for d in tensors])
        Ke = torch.stack([d[2] for d in tensors])
        N = torch.stack([d[3] for d in tensors])
        A = torch.stack([d[4] for d in tensors])
        return (Km, Vmax, Ke, N, A)

    def translate_seq(
        self, seq: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Translate nucleotide sequence into a protein with domains, corresponding
        weights, transmembrane regions, and signals.
        """
        i = 0
        j = self.n_dom_type_def_nts
        doms: list[Domain] = []
        while j + self.n_dom_detail_def_nts <= len(seq):
            domfact = self.seq_2_dom.get(seq[i:j])
            if domfact is not None:
                dom = domfact(seq[j : j + self.n_dom_detail_def_nts])
                doms.append(dom)
                i += self.n_dom_def_nts
                j += self.n_dom_def_nts
            else:
                i += CODON_SIZE
                j += CODON_SIZE

        n_doms = len(doms)
        Km = torch.full((self.n_molecules, n_doms), torch.nan)
        Vmax = torch.full((n_doms,), torch.nan)
        N = torch.full((self.n_molecules, n_doms), torch.nan)
        A = torch.zeros(self.n_molecules, n_doms)

        for dom_i, dom in enumerate(doms):
            if dom.is_receptor:
                mol = dom.substrates[0]
                offset = self.int_mol_pad if mol.is_intracellular else self.ext_mol_pad
                mol_i = self.molecules.index(mol)
                val = -1.0 if dom.is_inhibiting else 1.0
                A[mol_i + offset, dom_i] = val
                Km[mol_i + offset, dom_i] = dom.affinity
            elif dom.is_transporter:
                # TODO: rather generic
                mol1 = dom.substrates[0]
                mol_i = self.molecules.index(mol1)
                offset1 = self.int_mol_pad
                offset2 = self.ext_mol_pad
                N[mol_i + offset1]

        A = A.sum(dim=1).clamp(-1, 1)
        Km = Km.nanmean(dim=1).nan_to_num(0.0)
        Vmax = Vmax.nanmean(dim=0).nan_to_num(0.0)
        return


# TODO: is_receptor -> is_allosteric ?
