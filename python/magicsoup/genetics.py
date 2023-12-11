import warnings
import random
from magicsoup.util import nt_seqs
from magicsoup.constants import CODON_SIZE, ProteinSpecType
from magicsoup import _lib  # type: ignore

# TODO: constant dom_type_size and dom_size
#       in py and rs


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
    Class holding logic about translating nucleotide sequences into proteomes.

    Arguments:
        start_codons: Start codons which start a coding sequence
        stop_codons: Stop codons which stop a coding sequence
        p_catal_dom: Chance of encountering a catalytic domain in a random nucleotide sequence.
        p_transp_dom: Chance of encountering a transporter domain in a random nucleotide sequence.
        p_reg_dom: Chance of encountering a regulatory domain in a random nucleotide sequence.
        n_dom_type_codons: Number of codons that encode the domain type (catalytic, transporter, regulatory).

    During the simulation [World][magicsoup.world.World] uses [translate_genomes][magicsoup.genetics.Genetics.translate_genomes].
    The return value of this method is a nested list of tokens.
    These tokens are then mapped into concrete domain specifications (_e.g._ Km, Vmax, reactions, ...) by [Kinetics][magicsoup.kinetics.Kinetics].

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
        my_genetics = Genetics(p_transp_dom=0.1, stop_codons=("TGA", ))
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
        sets = self._get_non_stop_seqs(n_codons=n_dom_type_codons)
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
        self.one_codon_map: dict[str, int] = {}
        for i, seq in enumerate(self._get_single_codons()):
            self.one_codon_map[seq] = i + 1

        # the second codon is allowed to be a stop codon
        self.two_codon_map: dict[str, int] = {}
        for i, seq in enumerate(self._get_double_codons()):
            self.two_codon_map[seq] = i + 1

        # inverse maps for genome generation
        self.idx_2_one_codon = {v: k for k, v in self.one_codon_map.items()}
        self.idx_2_two_codon = {v: k for k, v in self.two_codon_map.items()}

    def translate_genomes(self, genomes: list[str]) -> list[list[ProteinSpecType]]:
        """
        Translate all genomes into proteomes

        Arguments:
            genomes: list of nucleotide sequences

        Returns:
            List of proteomes. This is a list (proteomes) of tuples (proteins)
            where each tuple (protein) has a list of domains, the start, and stop
            coordinate, and the direction of its CDS.
            Domains are a list of tuples with indices which will be mapped to
            specific domain specifications by [Kinetics][magicsoup.kinetics.Kinetics]

        Both forward and reverse-complement are considered.
        CDSs are extracted and a protein is translated for every CDS.
        Unviable proteins (no domains or only regulatory domains) are discarded.

        Start and stop indices for each protein always describe the start and stop coordinates
        of its CDS on the cell's genome (1st nucleotide has index 0) in the reported direction
        (`True` for _forward_, `False` for _reverse-complement_).
        So `(2, 20, False)` would be a CDS in the reverse-complement whose first nucleotide
        has index 2 (the third nucleotide) and whose last nucleotide has index 19 (the 20th nucleotide).
        If this would be projected onto the genome in forward direction,
        the CDS would start extend from `n - 2` to `n - 20` (where `n` is the length of the genome).
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

    def _get_non_stop_seqs(self, n_codons: int) -> list[str]:
        all_seqs = nt_seqs(n=n_codons * CODON_SIZE)
        seqs = []
        for seq in all_seqs:
            has_stop = False
            for i in range(n_codons):
                a = i * CODON_SIZE
                b = (i + 1) * CODON_SIZE
                if seq[a:b] in self.stop_codons:
                    has_stop = True
            if not has_stop:
                seqs.append(seq)
        return seqs

    def _get_single_codons(self) -> list[str]:
        seqs = nt_seqs(n=CODON_SIZE)
        seqs = [d for d in seqs if d not in self.stop_codons]
        return seqs

    def _get_double_codons(self) -> list[str]:
        seqs = nt_seqs(n=2 * CODON_SIZE)
        seqs = [d for d in seqs if d[:CODON_SIZE] not in self.stop_codons]
        return seqs
