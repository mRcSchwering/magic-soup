import warnings
import random
from magicsoup.util import codons
from magicsoup.constants import CODON_SIZE, ProteinSpecType
from magicsoup import _lib  # type: ignore


def _get_n(p: float, s: int, name: str) -> int:
    n = int(p * s)
    if n == 0 and p > 0.0:
        warnings.warn(
            f"There will be no {name}."
            f" Increase dom_type_size to accomodate low probabilities of having {name}."
        )
    return n


class Genetics:
    """
    Class holding logic about transcribing and translating nucleotide sequence pairs.

    Arguments:
        start_codons: Codons which start a coding sequence
        stop_codons: Codons which stop a coding sequence
        p_catal_dom: Chance of encountering a [CatalyticDomain][magicsoup.containers.CatalyticDomain]
            in a random nucleotide sequence.
        p_transp_dom: Chance of encountering a [TransporterDomain][magicsoup.containers.TransporterDomain]
            in a random nucleotide sequence.
        p_reg_dom: Chance of encountering a [RegulatoryDomain][magicsoup.containers.RegulatoryDomain]
            in a random nucleotide sequence.
        n_dom_type_codons: Number of codons that encode the domain type:
            [CatalyticDomain][magicsoup.containers.CatalyticDomain],
            [TransporterDomain][magicsoup.containers.TransporterDomain], or
            [RegulatoryDomain][magicsoup.containers.RegulatoryDomain].

    During the simulation [translate_genomes()][magicsoup.genetics.Genetics.translate_genomes] is used.
    Its return value can be used to update cell parameters using [Kinetics.set_cell_params()][magicsoup.kinetics.Kinetics.set_cell_params]
    or to translate its abstract return value into a human readable [Proteins][magicsoup.containers.Protein]
    description using [Kinetics.get_proteome()][magicsoup.kinetics.Kinetics.get_proteome].

    Translation happens only within coding sequences (CDSs).
    A CDS starts at every start codon and ends with the first in-frame encountered stop codon.
    Un-stopped CDSs are discarded.
    Both the forward and reverse complement of the nucleotide sequence are considered.
    Each CDS represents one [Protein][magicsoup.containers.Protein].
    All domains found within a CDS will be added as domains to that protein.

    If you want to use your own `genetics` object for the simulation you can just assign it after creating [World][magicsoup.world.World].
    _E.g._:

    ```
    world = World(chemistry=chemistry)
    my_genetics = Genetics(p_transp_dom=0.1, stop_codons=("TGA",))
    world.genetics = my_genetics
    ```
    """

    def __init__(
        self,
        start_codons: tuple[str, ...] = ("TTG", "GTG", "ATG"),
        stop_codons: tuple[str, ...] = ("TGA", "TAG", "TAA"),
        p_catal_dom: float = 0.01,
        p_transp_dom: float = 0.01,
        p_reg_dom: float = 0.01,
        n_dom_type_codons: int = 2,
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
        if p_catal_dom + p_transp_dom + p_reg_dom > 1.0:
            raise ValueError(
                "p_catal_dom, p_transp_dom, p_reg_dom together must not be greater 1.0"
            )

        self.start_codons = list(start_codons)
        self.stop_codons = list(stop_codons)

        # Domain structure:
        # domain type definition (domain type codons)
        # 3 x 1-codon specifications (3 simple tokens)
        # 1 x 2-codon specification (1 complex token)
        # a domain can begin and end with start/stop codons
        # so min CDS size = dom_size
        self.dom_size = (n_dom_type_codons + 5) * CODON_SIZE
        self.dom_type_size = n_dom_type_codons * CODON_SIZE

        # setup domain type definitions
        # sequences including stop codons are useless, since they would terminate the CDS
        sets = codons(n=n_dom_type_codons, excl_codons=self.start_codons)
        random.shuffle(sets)
        n = len(sets)

        n_catal_doms = _get_n(p=p_catal_dom, s=n, name="catalytic domains")
        n_transp_doms = _get_n(p=p_transp_dom, s=n, name="transporter domains")
        n_reg_doms = _get_n(p=p_reg_dom, s=n, name="allosteric domains")

        # 1=catalytic, 2=transporter, 3=regulatory
        self.domain_types: dict[int, list[str]] = {}
        self.domain_types[1] = sets[:n_catal_doms]
        del sets[:n_catal_doms]
        self.domain_types[2] = sets[:n_transp_doms]
        del sets[:n_transp_doms]
        self.domain_types[3] = sets[:n_reg_doms]
        del sets[:n_reg_doms]

        self.domain_map = {d: k for k, v in self.domain_types.items() for d in v}

        # pre-mature stop codons are excluded
        self.one_codon_map = {d: i + 1 for i, d in enumerate(self._get_single_codons())}

        # the second codon is allowed to be a stop codon
        self.two_codon_map = {d: i + 1 for i, d in enumerate(self._get_double_codons())}

        # inverse maps for genome generation
        self.idx_2_one_codon = {v: k for k, v in self.one_codon_map.items()}
        self.idx_2_two_codon = {v: k for k, v in self.two_codon_map.items()}

    def translate_genomes(self, genomes: list[str]) -> list[list[ProteinSpecType]]:
        """
        Translate all genomes into proteomes

        Arguments:
            genomes: list of base pair sequences

        Returns:
            List of proteomes. This is a list (proteomes) of list (proteins)
            where each protein is a tuple `(domains, cds_start, cds_end, is_fwd)`.
            `domains` is a list of tuples `(indices, start, end)` with indices which
            will be mapped to specific domain specifications by
            [Kinetics][magicsoup.kinetics.Kinetics].

        The result of this function can be used to update cell parameters
        using [Kinetics.set_cell_params()][magicsoup.kinetics.Kinetics.set_cell_params]
        or to translate this abstract return type into a human readable [Proteins][magicsoup.containers.Protein]
        description using [Kinetics.get_proteome()][magicsoup.kinetics.Kinetics.get_proteome].

        Both forward and reverse-complement are considered.
        CDSs are extracted and a protein is translated for every CDS.
        Unviable proteins (no domains or only regulatory domains) are discarded.

        CDS start and end describe the slice of the genome python string.
        _I.e._ the index starts with 0, start is included, end is excluded.
        _E.g._ `cds_start=2`, `cds_end=31` starts with the 3rd and ends with the 31st base pair on the genome.

        `is_fwd` describes whether the CDS is found on the forward (hypothetical 5'-3')
        or the reverse-complement (hypothetical 3'-5') side of the genome.
        `cds_start` and `cds_end` always describe the parsing direction / the direction of the hypothetical transcriptase.
        So, if you want to visualize a `is_fwd=False` CDS on the genome in 5'-3' direction
        you have to do `n - cds_start` and `n - cds_stop` if `n` is the genome length.
        """
        if len(genomes) < 1:
            return []
        return _lib.translate_genomes(
            genomes,
            self.start_codons,
            self.stop_codons,
            self.domain_map,
            self.one_codon_map,
            self.two_codon_map,
            self.dom_size,
            self.dom_type_size,
        )

    def _get_single_codons(self) -> list[str]:
        seqs = codons(n=1)
        seqs = [d for d in seqs if d not in self.stop_codons]
        return seqs

    def _get_double_codons(self) -> list[str]:
        seqs = codons(n=2)
        seqs = [d for d in seqs if d[:CODON_SIZE] not in self.stop_codons]
        return seqs
